"""
loader.py

LiDAR frame loading utilities for IU Task 1.

Responsibilities:
- Discover CSV frame files in data/raw
- Parse frame IDs from filenames
- Load a single frame with schema validation
- Convert columns to numeric safely
- Drop invalid XYZ rows

Docstring style: Google-style.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import pandas as pd

from .logger import setup_logger
from .exceptions import FrameParsingError, SchemaValidationError

logger = setup_logger()

# Expected columns in each LiDAR CSV frame (based on your dataset samples)
REQUIRED_COLUMNS = [
    "X", "Y", "Z",
    "DISTANCE",
    "INTENSITY",
    "POINT_ID",
    "RETURN_ID",
    "AMBIENT",
    "TIMESTAMP",
]

# Example filename:
# 192.168.26.26_2020-11-25_20-01-45_frame-1849.csv
FRAME_ID_PATTERN = re.compile(r"_frame-(\d+)$")


@dataclass(frozen=True)
class FrameFile:
    """Metadata for a LiDAR frame file."""
    path: Path
    frame_id: int


def parse_frame_id(csv_path: Path) -> int:
    """
    Extract frame number from a LiDAR filename.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Parsed frame ID as integer.

    Raises:
        FrameParsingError: If the frame ID cannot be extracted from filename.
    """
    stem = csv_path.stem
    match = FRAME_ID_PATTERN.search(stem)

    if not match:
        logger.error("Failed to parse frame ID from filename: %s", csv_path.name)
        raise FrameParsingError(f"Could not parse frame ID from: {csv_path.name}")

    return int(match.group(1))


def list_frame_files(raw_dir: Union[str, Path]) -> List[FrameFile]:
    """
    List all CSV frame files in raw_dir, sorted by frame_id.

    Args:
        raw_dir: Directory containing raw LiDAR frames.

    Returns:
        List of FrameFile sorted by frame_id.

    Raises:
        FileNotFoundError: If raw_dir does not exist.
    """
    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        logger.error("Raw data directory not found: %s", raw_dir)
        raise FileNotFoundError(f"Directory not found: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV frames found in: %s", raw_dir)

    frames: List[FrameFile] = []
    skipped = 0

    for f in csv_files:
        try:
            fid = parse_frame_id(f)
            frames.append(FrameFile(path=f, frame_id=fid))
        except FrameParsingError:
            skipped += 1
            logger.warning("Skipping file (frame id parse failed): %s", f.name)

    frames.sort(key=lambda x: x.frame_id)

    logger.info("Discovered %d frames (skipped %d files).", len(frames), skipped)
    return frames


def load_frame(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single LiDAR frame from semicolon-separated CSV.

    Performs:
    - schema validation (required columns)
    - safe numeric conversion
    - dropping rows missing X/Y/Z

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with validated schema and numeric columns.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        SchemaValidationError: If required columns are missing.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        raise FileNotFoundError(f"File not found: {csv_path}")

    logger.info("Loading frame: %s", csv_path.name)
    df = pd.read_csv(csv_path, sep=";")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Missing columns in %s: %s", csv_path.name, missing)
        raise SchemaValidationError(f"{csv_path.name} missing columns: {missing}")

    # Convert required columns to numeric; invalid parsing -> NaN
    for c in REQUIRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows where XYZ is missing
    before = len(df)
    df = df.dropna(subset=["X", "Y", "Z"]).reset_index(drop=True)
    dropped = before - len(df)

    if dropped > 0:
        logger.warning("Dropped %d rows with invalid XYZ in %s", dropped, csv_path.name)

    logger.info("Loaded %s with %d valid points", csv_path.name, len(df))
    return df
