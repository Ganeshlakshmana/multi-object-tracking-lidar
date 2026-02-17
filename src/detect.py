"""
detect.py

Clustering-based object detection on non-ground LiDAR point clouds.

Pipeline:
- DBSCAN clustering on non-ground points
- remove tiny clusters (noise)
- compute axis-aligned bounding boxes (AABB)
- output detections for downstream tracking

This is a verification/prototyping detector suitable for IU Task 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import open3d as o3d

from .logger import setup_logger

logger = setup_logger()


@dataclass(frozen=True)
class DetectConfig:
    """Parameters for clustering-based detection."""
    eps: float = 0.8              # neighborhood radius (meters)
    min_points: int = 20          # DBSCAN min points
    min_cluster_size: int = 30    # post-filter cluster size


@dataclass(frozen=True)
class Detection:
    """A single detected object candidate."""
    cluster_id: int
    n_points: int
    center: Tuple[float, float, float]
    extent: Tuple[float, float, float]  # size in x,y,z
    aabb: o3d.geometry.AxisAlignedBoundingBox


def cluster_dbscan(pcd: o3d.geometry.PointCloud, cfg: DetectConfig) -> np.ndarray:
    """
    Run DBSCAN clustering using Open3D.

    Returns:
        labels array of shape (N,), -1 for noise.
    """
    if len(pcd.points) == 0:
        return np.array([], dtype=int)

    labels = np.array(
        pcd.cluster_dbscan(eps=cfg.eps, min_points=cfg.min_points, print_progress=False),
        dtype=int,
    )
    n_clusters = int(labels.max() + 1) if labels.size and labels.max() >= 0 else 0
    noise = int((labels == -1).sum()) if labels.size else 0

    logger.info("DBSCAN: clusters=%d, noise_points=%d, total=%d", n_clusters, noise, len(labels))
    return labels


def detections_from_labels(
    pcd: o3d.geometry.PointCloud, labels: np.ndarray, cfg: DetectConfig
) -> List[Detection]:
    """
    Convert clustering labels into bounding-box detections.

    Returns:
        List of Detection objects.
    """
    detections: List[Detection] = []
    if labels.size == 0:
        return detections

    xyz = np.asarray(pcd.points)

    for cid in np.unique(labels):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        n = len(idx)
        if n < cfg.min_cluster_size:
            continue

        pts = xyz[idx]
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        extent = (maxs - mins)

        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=mins, max_bound=maxs)

        detections.append(
            Detection(
                cluster_id=int(cid),
                n_points=int(n),
                center=(float(center[0]), float(center[1]), float(center[2])),
                extent=(float(extent[0]), float(extent[1]), float(extent[2])),
                aabb=aabb,
            )
        )

    logger.info("Detections kept after filtering: %d", len(detections))
    return detections


def detect_objects(nonground_pcd: o3d.geometry.PointCloud, cfg: DetectConfig) -> List[Detection]:
    """
    Full detection routine: cluster + bounding boxes.

    Returns:
        List[Detection]
    """
    labels = cluster_dbscan(nonground_pcd, cfg)
    return detections_from_labels(nonground_pcd, labels, cfg)
