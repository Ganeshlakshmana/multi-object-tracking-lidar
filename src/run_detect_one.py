"""
run_detect_one.py

Run detection on one example frame AFTER preprocessing and save a report-ready plot.

What it does:
1) Loads the first available LiDAR CSV frame
2) Runs preprocessing (outliers + voxel + ground removal)
3) Runs DBSCAN clustering on non-ground points
4) Classifies clusters (car/pedestrian/background) using rule-based geometry
5) Prints size stats for threshold tuning
6) Saves a top-view plot with bounding boxes and ID-label text

Run (from project root):
    python -m src.run_detect_one
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

logger = setup_logger()


def main() -> None:
    """Entry point for single-frame detection + classification visualization."""
    project_root = Path(__file__).resolve().parents[1]

    # NOTE: You currently store CSVs under data/raw/dataset
    raw_dir = project_root / "data" / "raw" / "dataset"

    out_fig = project_root / "results" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    frames = list_frame_files(raw_dir)
    if not frames:
        raise RuntimeError(f"No frames found in: {raw_dir}")

    ff = frames[0]
    logger.info("Using example frame: %s (ID=%d)", ff.path.name, ff.frame_id)

    df = load_frame(ff.path)

    # --- Preprocess ---
    pre_cfg = PreprocessConfig(
        voxel_size=0.10,
        nb_neighbors=20,
        std_ratio=2.0,
        ransac_distance_threshold=0.15,
        ransac_n=3,
        ransac_num_iterations=200,
    )
    out = preprocess_frame(df, pre_cfg)
    nonground = out["nonground_pcd"]

    # --- Detect ---
    det_cfg = DetectConfig(
        eps=0.9,
        min_points=20,
        min_cluster_size=40,
    )
    dets = detect_objects(nonground, det_cfg)
    logger.info("Detections kept: %d", len(dets))

    # --- Classify ---
    classified = classify_detections(dets)

    # Build quick lookup: cluster_id -> label
    id_to_label = {c.cluster_id: c.label for c in classified}

    # Print classification summary
    print("\nClassified detections:")
    for c in classified:
        print(
            f"id={c.cluster_id:>3}  label={c.label:<12} "
            f"extent={tuple(round(e, 2) for e in c.extent)}  points={c.n_points}"
        )

    # Print a few detections to tune thresholds
    print("\nSample detections (for threshold tuning):")
    for d in dets[:10]:
        sx, sy, sz = d.extent
        print(
            f"cluster={d.cluster_id:>3}  points={d.n_points:>5}  "
            f"extent(sx,sy,sz)=({sx:.2f},{sy:.2f},{sz:.2f})  center={d.center}"
        )

    # --- Plot top view with bounding boxes + labels ---
    xyz = np.asarray(nonground.points)

    plt.figure(figsize=(6, 6))
    plt.scatter(xyz[:, 0], xyz[:, 1], s=0.2)

    for d in dets:
        cx, cy, _ = d.center
        sx, sy, _ = d.extent

        rect = plt.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fill=False)
        plt.gca().add_patch(rect)

        label = id_to_label.get(d.cluster_id, "unknown")
        plt.text(cx, cy, f"{d.cluster_id}-{label}", fontsize=7)

    plt.title(f"Detections + Rule-based Labels (frame {ff.frame_id})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()

    out_path = out_fig / f"detections_labeled_{ff.frame_id}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("\nSaved:", out_path)
    print("Detections:", len(dets))


if __name__ == "__main__":
    main()
