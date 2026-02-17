"""
track.py

Multi-object tracking (MOT) for LiDAR detections.

Tracking approach:
- Track state: (x, y) centroid + metadata
- Data association: nearest-neighbor with distance gating
- Track lifecycle: create, update, age, delete, confirm

This is a strong baseline for IU Task 1 verification pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .logger import setup_logger

logger = setup_logger()


@dataclass
class Track:
    """Represents one tracked object across frames."""
    track_id: int
    label: str  # car/pedestrian/background
    x: float
    y: float
    age: int = 0              # frames since last matched
    hits: int = 1             # number of matches
    confirmed: bool = False   # becomes True after min_hits
    history: List[Tuple[float, float]] = field(default_factory=list)

    def update(self, x: float, y: float, label: str | None = None) -> None:
        """Update track with new detection."""
        self.x = float(x)
        self.y = float(y)
        self.age = 0
        self.hits += 1
        if label is not None:
            self.label = label
        self.history.append((self.x, self.y))


@dataclass(frozen=True)
class TrackConfig:
    """Tracking parameters."""
    max_match_distance: float = 2.5  # meters
    max_age: int = 5                # frames to keep unmatched tracks
    min_hits: int = 3               # frames to confirm a track
    ignore_background: bool = True  # optional: track only cars/pedestrians


def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between two sets of 2D points."""
    # a: (N,2), b: (M,2) -> (N,M)
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def associate_detections_to_tracks(
    det_xy: np.ndarray,
    track_xy: np.ndarray,
    max_dist: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedy nearest-neighbor association with gating.

    Returns:
        matches: list of (det_index, track_index)
        unmatched_dets: list of det indices
        unmatched_tracks: list of track indices
    """
    n_d = det_xy.shape[0]
    n_t = track_xy.shape[0]

    if n_d == 0:
        return [], [], list(range(n_t))
    if n_t == 0:
        return [], list(range(n_d)), []

    dist = _pairwise_dist(det_xy, track_xy)

    unmatched_dets = set(range(n_d))
    unmatched_tracks = set(range(n_t))
    matches: List[Tuple[int, int]] = []

    while True:
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if not np.isfinite(dist[i, j]) or dist[i, j] > max_dist:
            break

        if i in unmatched_dets and j in unmatched_tracks:
            matches.append((i, j))
            unmatched_dets.remove(i)
            unmatched_tracks.remove(j)

        dist[i, :] = np.inf
        dist[:, j] = np.inf

        if not np.isfinite(dist).any():
            break

    return matches, sorted(list(unmatched_dets)), sorted(list(unmatched_tracks))


class MultiObjectTracker:
    """Simple multi-object tracker with centroid association and lifecycle management."""

    def __init__(self, cfg: TrackConfig):
        self.cfg = cfg
        self.tracks: List[Track] = []
        self.next_id: int = 1

    def step(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracker with current frame detections.

        Each detection dict must contain:
            - 'x', 'y' (centroid)
            - 'label' (car/pedestrian/background)

        Returns:
            Current list of tracks (including unconfirmed).
        """
        # Optionally ignore background
        if self.cfg.ignore_background:
            detections = [d for d in detections if d["label"] in ("car", "pedestrian")]

        det_xy = np.array([[d["x"], d["y"]] for d in detections], dtype=float) if detections else np.zeros((0, 2))
        track_xy = np.array([[t.x, t.y] for t in self.tracks], dtype=float) if self.tracks else np.zeros((0, 2))

        matches, um_dets, um_tracks = associate_detections_to_tracks(
            det_xy, track_xy, self.cfg.max_match_distance
        )

        # Update matched tracks
        for det_i, trk_i in matches:
            d = detections[det_i]
            self.tracks[trk_i].update(d["x"], d["y"], label=d["label"])

        # Age unmatched tracks
        for trk_i in um_tracks:
            self.tracks[trk_i].age += 1

        # Create new tracks for unmatched detections
        for det_i in um_dets:
            d = detections[det_i]
            t = Track(
                track_id=self.next_id,
                label=d["label"],
                x=float(d["x"]),
                y=float(d["y"]),
            )
            t.history.append((t.x, t.y))
            self.tracks.append(t)
            self.next_id += 1

        # Confirm tracks
        for t in self.tracks:
            if not t.confirmed and t.hits >= self.cfg.min_hits:
                t.confirmed = True

        # Remove dead tracks
        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.age <= self.cfg.max_age]
        removed = before - len(self.tracks)
        if removed > 0:
            logger.info("Removed %d stale tracks", removed)

        return self.tracks
