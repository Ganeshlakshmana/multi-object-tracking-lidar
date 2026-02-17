"""
preprocess.py

LiDAR preprocessing for IU Task 1:
- Convert pandas frame -> Open3D PointCloud
- Statistical outlier removal
- Voxel downsampling
- Ground-plane removal via RANSAC

Outputs are used for detection + tracking downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import open3d as o3d
import pandas as pd

from .logger import setup_logger
from .exceptions import DataIntegrityError

logger = setup_logger()


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration parameters for preprocessing."""
    voxel_size: float = 0.10
    nb_neighbors: int = 20
    std_ratio: float = 2.0

    # Ground segmentation (RANSAC)
    ransac_distance_threshold: float = 0.15
    ransac_n: int = 3
    ransac_num_iterations: int = 200


def df_to_o3d(df: pd.DataFrame) -> o3d.geometry.PointCloud:
    """
    Convert a LiDAR frame DataFrame to an Open3D PointCloud.

    Args:
        df: DataFrame containing X, Y, Z columns.

    Returns:
        Open3D PointCloud.

    Raises:
        DataIntegrityError: If required columns are missing or empty.
    """
    for col in ("X", "Y", "Z"):
        if col not in df.columns:
            raise DataIntegrityError(f"Missing column '{col}' required for point cloud.")

    xyz = df[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
    if xyz.size == 0:
        raise DataIntegrityError("Empty point cloud (0 points).")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def remove_outliers(pcd: o3d.geometry.PointCloud, cfg: PreprocessConfig) -> o3d.geometry.PointCloud:
    """
    Remove statistical outliers from a point cloud.

    Returns:
        Filtered point cloud.
    """
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=cfg.nb_neighbors,
        std_ratio=cfg.std_ratio,
    )
    filtered = pcd.select_by_index(ind)
    logger.info("Outlier removal: %d -> %d points", len(pcd.points), len(filtered.points))
    return filtered


def voxel_downsample(pcd: o3d.geometry.PointCloud, cfg: PreprocessConfig) -> o3d.geometry.PointCloud:
    """
    Downsample point cloud using voxel grid filter.

    Returns:
        Downsampled point cloud.
    """
    if cfg.voxel_size <= 0:
        return pcd
    down = pcd.voxel_down_sample(voxel_size=cfg.voxel_size)
    logger.info("Voxel downsample (%.2f m): %d -> %d points", cfg.voxel_size, len(pcd.points), len(down.points))
    return down


def remove_ground_ransac(
    pcd: o3d.geometry.PointCloud,
    cfg: PreprocessConfig
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, Tuple[float, float, float, float]]:
    """
    Segment ground plane using RANSAC and return:
    - non-ground points (objects)
    - ground points
    - plane model (a, b, c, d) for ax+by+cz+d=0

    Returns:
        (pcd_nonground, pcd_ground, plane_model)
    """
    if len(pcd.points) < 50:
        raise DataIntegrityError("Too few points for RANSAC ground segmentation.")

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=cfg.ransac_distance_threshold,
        ransac_n=cfg.ransac_n,
        num_iterations=cfg.ransac_num_iterations,
    )

    ground = pcd.select_by_index(inliers)
    nonground = pcd.select_by_index(inliers, invert=True)

    a, b, c, d = plane_model
    logger.info(
        "Ground RANSAC: ground=%d, nonground=%d, plane=[%.3f, %.3f, %.3f, %.3f]",
        len(ground.points), len(nonground.points), a, b, c, d
    )

    return nonground, ground, (float(a), float(b), float(c), float(d))


def preprocess_frame(df: pd.DataFrame, cfg: PreprocessConfig):
    """
    Full preprocessing pipeline for one frame.

    Returns:
        dict containing:
        - raw_pcd
        - filtered_pcd
        - down_pcd
        - nonground_pcd
        - ground_pcd
        - plane_model
    """
    raw = df_to_o3d(df)
    filtered = remove_outliers(raw, cfg)
    down = voxel_downsample(filtered, cfg)
    nonground, ground, plane = remove_ground_ransac(down, cfg)

    return {
        "raw_pcd": raw,
        "filtered_pcd": filtered,
        "down_pcd": down,
        "nonground_pcd": nonground,
        "ground_pcd": ground,
        "plane_model": plane,
    }
