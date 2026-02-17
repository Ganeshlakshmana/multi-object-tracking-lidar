"""
classify.py

Geometric rule-based classification of detected LiDAR clusters.

Classes:
- car
- pedestrian
- background
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ClassifiedDetection:
    cluster_id: int
    label: str
    center: Tuple[float, float, float]
    extent: Tuple[float, float, float]
    n_points: int


def classify_extent(extent: Tuple[float, float, float]) -> str:
    """
    Classify object based on bounding box dimensions.
    Loosened thresholds for real-world partial visibility.
    """
    sx, sy, sz = extent

    length = max(sx, sy)
    width = min(sx, sy)
    height = sz

    # --- Pedestrian ---
    if (
        length < 1.2
        and width < 1.2
        and 1.2 <= height <= 2.5
    ):
        return "pedestrian"

    # --- Car (looser constraints) ---
    if (
        2.0 <= length <= 7.0
        and 1.0 <= width <= 3.0
        and 0.8 <= height <= 3.0
    ):
        return "car"

    # --- Likely wall/building ---
    if length > 7.0:
        return "background"

    return "background"


def classify_detections(detections):
    """
    Classify list of Detection objects.
    """
    classified = []

    for d in detections:
        label = classify_extent(d.extent)
        classified.append(
            ClassifiedDetection(
                cluster_id=d.cluster_id,
                label=label,
                center=d.center,
                extent=d.extent,
                n_points=d.n_points,
            )
        )

    return classified
