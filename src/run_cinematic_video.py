"""
run_cinematic_video.py

Friend-like cinematic LiDAR video:
- SIDE view camera preset (matches friend's code)
- Correct speed (fps=10) + optional frame_repeat
- IU watermark panel (logo + university name) on the right side
- Own rendered LiDAR scene (NOT overlaying the dataset mp4)

Run:
    python -m src.run_cinematic_video

IMPORTANT:
Place IU logo here (PNG recommended, transparent background preferred):
    assets/iu_logo.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import time


import cv2
import numpy as np
import open3d as o3d

from src.logger import setup_logger
from src.loader import list_frame_files, load_frame
from src.preprocess import PreprocessConfig, preprocess_frame
from src.detect import DetectConfig, detect_objects
from src.classify import classify_detections
from src.track import MultiObjectTracker, TrackConfig

logger = setup_logger()


# ---------------------------
# Friend-style distance coloring (same stops)
# ---------------------------

def distance_to_color(distance: float, max_distance: float = 80.0) -> np.ndarray:
    t = min(distance / max_distance, 1.0)
    if t < 0.15:  # 0-12m cyan
        return np.array([0.0, 1.0, 1.0])
    if t < 0.30:  # cyan -> green
        ratio = (t - 0.15) / 0.15
        return np.array([0.0, 1.0, 1.0 - ratio])
    if t < 0.50:  # green -> yellow
        ratio = (t - 0.30) / 0.20
        return np.array([ratio, 1.0, 0.0])
    if t < 0.75:  # yellow -> orange
        ratio = (t - 0.50) / 0.25
        return np.array([1.0, 1.0 - ratio * 0.5, 0.0])
    # orange -> red
    ratio = (t - 0.75) / 0.25
    return np.array([1.0, 0.5 - ratio * 0.5, 0.0])


def colorize_by_distance(xyz: np.ndarray, max_distance: float = 80.0) -> np.ndarray:
    d = np.linalg.norm(xyz[:, :3], axis=1)
    colors = np.zeros((xyz.shape[0], 3), dtype=np.float64)
    for i, dist in enumerate(d):
        colors[i] = distance_to_color(float(dist), max_distance=max_distance)
    return colors


# ---------------------------
# Grid + bbox helpers
# ---------------------------

def create_grid_floor(
    size: float = 150.0,
    spacing: float = 10.0,
    height: float = -0.5,
    color: Tuple[float, float, float] = (0.10, 0.10, 0.15),
) -> o3d.geometry.LineSet:
    lines = []
    points = []
    half = size / 2
    num = int(size / spacing) + 1
    idx = 0

    # parallel to X
    for i in range(num):
        y = -half + i * spacing
        points.append([-half, y, height])
        points.append([half, y, height])
        lines.append([idx, idx + 1])
        idx += 2

    # parallel to Y
    for i in range(num):
        x = -half + i * spacing
        points.append([x, -half, height])
        points.append([x, half, height])
        lines.append([idx, idx + 1])
        idx += 2

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
    grid.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    grid.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float64), (len(lines), 1)))
    return grid


def aabb_to_lineset(aabb: o3d.geometry.AxisAlignedBoundingBox, rgb: Tuple[float, float, float]) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    ls.paint_uniform_color(list(rgb))
    return ls


# ---------------------------
# Projection for labels (3D -> 2D)
# ---------------------------

def project_points(pts_world: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    pts = np.hstack([pts_world, np.ones((pts_world.shape[0], 1))])  # Nx4
    cam = (extrinsic @ pts.T).T
    x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
    z = np.where(z <= 1e-6, 1e-6, z)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return np.vstack([u, v]).T


def draw_label_callout(img_bgr: np.ndarray, anchor_xy: Tuple[int, int], text: str, box_color_bgr: Tuple[int, int, int]) -> None:
    x, y = int(anchor_xy[0]), int(anchor_xy[1])
    lx, ly = x + 70, y - 40

    # leader line
    cv2.line(img_bgr, (x, y), (lx, ly), (255, 255, 255), 2)

    # label box (semi-transparent)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
    pad = 10

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (lx, ly - th - pad), (lx + tw + pad, ly + pad), box_color_bgr, -1)
    cv2.addWeighted(overlay, 0.72, img_bgr, 0.28, 0, img_bgr)

    cv2.putText(img_bgr, text, (lx + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)


def motion_prefix(history: List[Tuple[float, float]]) -> str:
    # Smooth dy using 3 history points
    if len(history) < 3:
        return ""
    dy = history[-1][1] - history[-3][1]
    return "approaching " if dy < -0.20 else ""


# ---------------------------
# IU watermark panel (right side)
# ---------------------------

def overlay_iu_branding(
    img_bgr: np.ndarray,
    logo_path: Path,
    uni_text: str = "IU International University of Applied Sciences",
) -> np.ndarray:
    """
    Adds a semi-transparent vertical branding panel on the right side with:
    - IU logo (if file exists)
    - University name
    """
    h, w = img_bgr.shape[:2]
    panel_w = int(w * 0.18)
    x0 = w - panel_w

    # panel background
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), (10, 10, 15), -1)
    img_bgr = cv2.addWeighted(overlay, 0.35, img_bgr, 0.65, 0)

    # logo (optional)
    y_cursor = 30
    if logo_path.exists():
        logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
        if logo is not None:
            # resize logo to fit panel
            target_w = int(panel_w * 0.70)
            scale = target_w / max(1, logo.shape[1])
            target_h = max(1, int(logo.shape[0] * scale))
            logo = cv2.resize(logo, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # place
            lx = x0 + (panel_w - target_w) // 2
            ly = y_cursor

            # alpha blend if RGBA
            if logo.shape[2] == 4:
                alpha = logo[:, :, 3] / 255.0
                for c in range(3):
                    img_bgr[ly:ly+target_h, lx:lx+target_w, c] = (
                        alpha * logo[:, :, c] + (1 - alpha) * img_bgr[ly:ly+target_h, lx:lx+target_w, c]
                    ).astype(np.uint8)
            else:
                img_bgr[ly:ly+target_h, lx:lx+target_w] = logo[:, :, :3]

            y_cursor = ly + target_h + 25

    # university text (wrapped)
    words = uni_text.split()
    lines = []
    line = []
    max_chars = 18
    for w_ in words:
        if sum(len(x) for x in line) + len(line) + len(w_) <= max_chars:
            line.append(w_)
        else:
            lines.append(" ".join(line))
            line = [w_]
    if line:
        lines.append(" ".join(line))

    font = cv2.FONT_HERSHEY_SIMPLEX
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, font, 0.6, 2)
        tx = x0 + (panel_w - tw) // 2
        ty = y_cursor + th
        cv2.putText(img_bgr, ln, (tx, ty), font, 0.6, (255, 255, 255), 2)
        y_cursor = ty + 18

    return img_bgr


# ---------------------------
# Camera preset: SIDE view (friend)
# ---------------------------

@dataclass(frozen=True)
class CameraPreset:
    front: List[float]
    lookat: List[float]
    up: List[float]
    zoom: float


def friend_side_camera() -> CameraPreset:
    # Exactly from friend's real_data_visualization.py side view
    return CameraPreset(
        front=[-0.9, -0.3, 0.3],
        lookat=[0.0, 30.0, 0.0],
        up=[0.0, 0.0, 1.0],
        zoom=0.10,
    )


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw" / "dataset"
    out_dir = project_root / "results" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Put IU logo here:
    logo_path = project_root / "assets" / "iu_logo.png"
    (project_root / "assets").mkdir(parents=True, exist_ok=True)

    out_mp4 = out_dir / "taskpdf_friend_sideview_IU.mp4"

    # Speed controls (fix high speed)
    fps = 10          # friend uses fps=10 for real-data video generation
    frame_repeat = 2  # write each frame twice => slower + smoother

    frames = list_frame_files(raw_dir)
    if not frames:
        raise RuntimeError(f"No CSV frames found in: {raw_dir}")
    frames = frames[:min(51, len(frames))]

    pre_cfg = PreprocessConfig(
        voxel_size=0.10,
        nb_neighbors=20,
        std_ratio=2.0,
        ransac_distance_threshold=0.15,
        ransac_n=3,
        ransac_num_iterations=200,
    )
    det_cfg = DetectConfig(eps=0.9, min_points=20, min_cluster_size=40)

    tracker = MultiObjectTracker(
        TrackConfig(max_match_distance=3.5, max_age=8, min_hits=2, ignore_background=True)
    )

    # Open3D Visualizer (GUI)
    vis = o3d.visualization.Visualizer()
    vis.create_window("Cinematic LiDAR (Side View + IU)", width=1280, height=720, visible=True)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.02, 0.02, 0.05])  # dark blue-black like friend
    opt.point_size = 2.0
    opt.show_coordinate_frame = False

    grid = create_grid_floor(size=150.0, spacing=10.0, height=-0.5, color=(0.10, 0.10, 0.15))
    pcd = o3d.geometry.PointCloud()

    vis.add_geometry(grid)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    cam = friend_side_camera()
    camera_initialized = False

    vw = None

    # bbox colors (RGB Open3D)
    bbox_rgb = {
        "car": (1.0, 0.3, 0.3),
        "pedestrian": (0.3, 1.0, 0.3),
    }
    # label boxes (BGR OpenCV)
    label_box_bgr = {
        "car": (40, 120, 255),
        "pedestrian": (200, 150, 60),
    }

    for i, ff in enumerate(frames, start=1):
        df = load_frame(ff.path)
        out = preprocess_frame(df, pre_cfg)
        nonground = out["nonground_pcd"]

        dets = detect_objects(nonground, det_cfg)
        classified = classify_detections(dets)

        # tracking for stable IDs + "approaching"
        det_for_track = [{"x": c.center[0], "y": c.center[1], "label": c.label} for c in classified]
        tracks = tracker.step(det_for_track)
        confirmed = [t for t in tracks if t.confirmed and t.label in ("car", "pedestrian")]

        xyz = np.asarray(nonground.points)
        if xyz.size == 0:
            continue

        # distance colors
        colors = colorize_by_distance(xyz, max_distance=80.0)

        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # redraw scene
        vis.clear_geometries()
        vis.add_geometry(grid)
        vis.add_geometry(pcd)

        # bboxes
        for d, c in zip(dets, classified):
            if c.label not in ("car", "pedestrian"):
                continue
            vis.add_geometry(aabb_to_lineset(d.aabb, rgb=bbox_rgb[c.label]))

        vis.poll_events()
        vis.update_renderer()

        # set camera ONCE (side view + closer look)
        if not camera_initialized:
            ctr.set_front(cam.front)
            ctr.set_lookat(cam.lookat)
            ctr.set_up(cam.up)
            ctr.set_zoom(cam.zoom)
            camera_initialized = True
            vis.poll_events()
            vis.update_renderer()

        # capture frame
        img = vis.capture_screen_float_buffer(do_render=True)
        img = (255 * np.asarray(img)).astype(np.uint8)  # RGB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # project and draw callouts
        params = ctr.convert_to_pinhole_camera_parameters()
        intrinsic = np.asarray(params.intrinsic.intrinsic_matrix)
        extrinsic = np.asarray(params.extrinsic)

        h, w = img_bgr.shape[:2]
        for t in confirmed:
            prefix = motion_prefix(t.history)
            text = f"{prefix}{t.label}".strip()

            z_anchor = 2.0 if t.label == "pedestrian" else 1.5
            pix = project_points(
                np.array([[t.x, t.y, z_anchor]], dtype=np.float64),
                intrinsic, extrinsic
            )[0]

            if 0 <= pix[0] < w and 0 <= pix[1] < h:
                draw_label_callout(img_bgr, (int(pix[0]), int(pix[1])), text, label_box_bgr[t.label])

        # IU branding panel (right side)
        img_bgr = overlay_iu_branding(img_bgr, logo_path)

        # small HUD
        cv2.putText(
            img_bgr,
            f"Frame {i}/{len(frames)} | ID:{ff.frame_id}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
        )

        # init writer
        if vw is None:
            hh, ww = img_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (ww, hh))

        # slow down smoothly
        for _ in range(frame_repeat):
            vw.write(img_bgr)

        if i % 10 == 0:
            logger.info("Rendered %d/%d frames", i, len(frames))

        time.sleep(0.01)

    if vw is not None:
        vw.release()
    vis.destroy_window()

    logger.info("Saved: %s", out_mp4)
    print("Saved:", out_mp4)


if __name__ == "__main__":
    main()
