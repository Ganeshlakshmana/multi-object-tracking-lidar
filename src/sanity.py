"""
sanity.py

IU Task 1 requirement: first assessment and sanity check of LiDAR data.

This module:
- computes per-frame statistics
- saves a frame index + stats to data/metadata/
- generates report-ready plots in results/figures/

Docstring style: Google-style.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd

from .logger import setup_logger
from .loader import list_frame_files, load_frame

logger = setup_logger()


@dataclass(frozen=True)
class FrameStats:
    """Summary statistics for a single LiDAR frame."""
    frame_id: int
    filename: str
    n_points: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    dist_p50: float
    dist_p95: float
    intensity_p50: float


def compute_frame_stats(df: pd.DataFrame, frame_id: int, filename: str) -> FrameStats:
    """Compute frame-level stats used for sanity checking."""
    return FrameStats(
        frame_id=frame_id,
        filename=filename,
        n_points=int(len(df)),
        x_min=float(df["X"].min()),
        x_max=float(df["X"].max()),
        y_min=float(df["Y"].min()),
        y_max=float(df["Y"].max()),
        z_min=float(df["Z"].min()),
        z_max=float(df["Z"].max()),
        dist_p50=float(df["DISTANCE"].median()),
        dist_p95=float(df["DISTANCE"].quantile(0.95)),
        intensity_p50=float(df["INTENSITY"].median()),
    )


def save_plots_example_frame(df: pd.DataFrame, out_dir: Path, frame_tag: str) -> None:
    """Save core distributions for one example frame."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Z distribution
    plt.figure()
    plt.hist(df["Z"].values, bins=120)
    plt.title(f"Z distribution (example: {frame_tag})")
    plt.xlabel("Z (m)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / f"z_hist_{frame_tag}.png", dpi=200)
    plt.close()

    # Distance distribution
    plt.figure()
    plt.hist(df["DISTANCE"].values, bins=120)
    plt.title(f"Distance distribution (example: {frame_tag})")
    plt.xlabel("Distance (m)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / f"distance_hist_{frame_tag}.png", dpi=200)
    plt.close()

    # Intensity distribution
    plt.figure()
    plt.hist(df["INTENSITY"].values, bins=120)
    plt.title(f"Intensity distribution (example: {frame_tag})")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / f"intensity_hist_{frame_tag}.png", dpi=200)
    plt.close()

    # Top-view XY scatter (thin dots)
    plt.figure(figsize=(6, 6))
    plt.scatter(df["X"].values, df["Y"].values, s=0.2)
    plt.title(f"Top view X-Y (example: {frame_tag})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(out_dir / f"xy_topview_{frame_tag}.png", dpi=200)
    plt.close()


def run_sanity(raw_dir: Union[str, Path]) -> Path:
    """
    Run dataset sanity checks:
    - compute stats for all frames
    - write stats CSV to data/metadata/
    - generate plots to results/figures/

    Args:
        raw_dir: Directory containing raw CSV frames.

    Returns:
        Path to the saved stats CSV.
    """
    root = Path.cwd()
    raw_dir = Path(raw_dir)

    metadata_dir = root / "data" / "metadata"
    figures_dir = root / "results" / "figures"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frames = list_frame_files(raw_dir)
    if not frames:
        raise RuntimeError(f"No frames found in {raw_dir}")

    logger.info("Running sanity check on %d frames...", len(frames))

    stats_rows: List[Dict] = []
    for ff in frames:
        df = load_frame(ff.path)
        st = compute_frame_stats(df, ff.frame_id, ff.path.name)
        stats_rows.append(asdict(st))

    stats_df = pd.DataFrame(stats_rows).sort_values("frame_id")
    stats_csv_path = metadata_dir / "frame_stats.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    logger.info("Saved frame stats to: %s", stats_csv_path)

    # Plot: points per frame
    plt.figure()
    plt.plot(stats_df["frame_id"].values, stats_df["n_points"].values)
    plt.title("Points per frame")
    plt.xlabel("Frame ID")
    plt.ylabel("Number of points")
    plt.tight_layout()
    plt.savefig(figures_dir / "points_per_frame.png", dpi=200)
    plt.close()

    # Choose example frame = first one
    example = frames[0]
    df0 = load_frame(example.path)
    save_plots_example_frame(df0, figures_dir, frame_tag=f"{example.frame_id}")

    logger.info("Saved sanity-check figures to: %s", figures_dir)
    return stats_csv_path
