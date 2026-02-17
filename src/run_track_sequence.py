"""
run_track_sequence.py

Run multi-frame detection + classification + tracking and export an MP4.
Run:
    python -m src.run_track_sequence
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.logger import setup_logger
from src.loader import list_frame_files, load_frame
from src.preprocess import PreprocessConfig, preprocess_frame
from src.detect import DetectConfig, detect_objects
from src.classify import classify_detections
from src.track import MultiObjectTracker, TrackConfig
import time
from src.performance import make_analyzer_for_assignment


import cv2  # from opencv-python

logger = setup_logger()


def fig_to_rgb_array(fig) -> np.ndarray:
    """
    Convert a Matplotlib figure to an RGB numpy array (backend-safe).

    Works on TkAgg/QtAgg/etc by using the renderer buffer.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # TkAgg supports tostring_argb; convert ARGB -> RGB
    argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
    rgba = argb[:, :, [1, 2, 3, 0]]   # ARGB -> RGBA
    rgb = rgba[:, :, :3]             # drop alpha
    return rgb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw" / "dataset"

    out_vid_dir = project_root / "results" / "videos"
    out_vid_dir.mkdir(parents=True, exist_ok=True)

    frames = list_frame_files(raw_dir)
    if not frames:
        raise RuntimeError(f"No frames found in: {raw_dir}")

    # Limit frames for first run (you can increase later)
    max_frames = min(51, len(frames))
    frames = frames[:max_frames]
    logger.info("Tracking on %d frames", len(frames))

    pre_cfg = PreprocessConfig()
    det_cfg = DetectConfig(eps=0.9, min_points=20, min_cluster_size=40)

    tracker = MultiObjectTracker(
        TrackConfig(
            max_match_distance=2.5,
            max_age=5,
            min_hits=3,
            ignore_background=True,
        )
    )

    video_frames = []
    xlim = (-20, 20)
    ylim = (0, 80)

    for k, ff in enumerate(frames, start=1):
        df = load_frame(ff.path)
        out = preprocess_frame(df, pre_cfg)
        nonground = out["nonground_pcd"]

        dets = detect_objects(nonground, det_cfg)
        classified = classify_detections(dets)

        # Convert to detections for tracker (centroid + label)
        det_for_track = [
            {"x": c.center[0], "y": c.center[1], "label": c.label}
            for c in classified
        ]

        tracks = tracker.step(det_for_track)
        confirmed = [t for t in tracks if t.confirmed]

        # Plot a top-view frame
        xyz = np.asarray(nonground.points)

        fig = plt.figure(figsize=(6, 6), dpi=140)
        ax = fig.add_subplot(111)
        ax.scatter(xyz[:, 0], xyz[:, 1], s=0.2)

        # draw confirmed tracks
        for t in confirmed:
            ax.text(t.x, t.y, f"ID{t.track_id}-{t.label}", fontsize=7)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"Tracking (frame {ff.frame_id})")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.text(xlim[0] + 1, ylim[1] - 3, f"Frame {k}/{len(frames)}", fontsize=8)

        img = fig_to_rgb_array(fig)
        plt.close(fig)
        video_frames.append(img)

    # Write MP4
    out_path = out_vid_dir / "lidar_tracking_topview.mp4"
    h, w = video_frames[0].shape[:2]
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for fr in video_frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    vw.release()
    logger.info("Saved tracking video: %s", out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
