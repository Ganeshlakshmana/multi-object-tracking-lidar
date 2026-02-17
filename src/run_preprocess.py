"""
run_preprocess.py

Run preprocessing on one example frame and save evidence plots/outputs.
Run:
    python -m src.run_preprocess
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.loader import list_frame_files, load_frame
from src.preprocess import PreprocessConfig, preprocess_frame


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw" / "dataset"
    out_fig = project_root / "results" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    frames = list_frame_files(raw_dir)
    ff = frames[0]  # first frame as example

    df = load_frame(ff.path)

    cfg = PreprocessConfig(
        voxel_size=0.10,
        nb_neighbors=20,
        std_ratio=2.0,
        ransac_distance_threshold=0.15,
        ransac_n=3,
        ransac_num_iterations=200,
    )

    out = preprocess_frame(df, cfg)

    # --- Save BEFORE vs AFTER top-view scatter as evidence ---
    raw_xyz = np.asarray(out["raw_pcd"].points)
    ng_xyz = np.asarray(out["nonground_pcd"].points)

    plt.figure(figsize=(6, 6))
    plt.scatter(raw_xyz[:, 0], raw_xyz[:, 1], s=0.2)
    plt.title(f"Top view BEFORE preprocessing (frame {ff.frame_id})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(out_fig / f"topview_before_{ff.frame_id}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(ng_xyz[:, 0], ng_xyz[:, 1], s=0.2)
    plt.title(f"Top view AFTER ground removal (frame {ff.frame_id})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(out_fig / f"topview_after_nonground_{ff.frame_id}.png", dpi=200)
    plt.close()

    print("Saved preprocessing evidence figures to:", out_fig)


if __name__ == "__main__":
    main()
