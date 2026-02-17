"""
test_loader.py

Quick smoke test for loader functions.
Run from project root:
    python -m src.test_loader
"""

from pathlib import Path

from src.loader import list_frame_files, load_frame
from src.exceptions import FrameParsingError, SchemaValidationError


def main() -> None:
    raw_dir = Path("data/raw/dataset")

    try:
        frames = list_frame_files(raw_dir)
        print("Found frames:", len(frames))

        if not frames:
            print("No frames found. Put CSVs into data/raw and retry.")
            return

        print("First frame:", frames[0].path.name, "ID:", frames[0].frame_id)
        print("Last frame:", frames[-1].path.name, "ID:", frames[-1].frame_id)

        df0 = load_frame(frames[0].path)
        print("First frame shape:", df0.shape)
        print("Columns:", df0.columns.tolist())
        print(df0.head(3))

    except FileNotFoundError as e:
        print("ERROR:", e)
    except (FrameParsingError, SchemaValidationError) as e:
        print("DATA ERROR:", e)


if __name__ == "__main__":
    main()
