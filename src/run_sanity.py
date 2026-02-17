"""
run_sanity.py

Entry point to run sanity checks.
Run from project root:
    python -m src.run_sanity
"""

from pathlib import Path
from src.sanity import run_sanity


def main() -> None:
    raw_dir = Path("data/raw/dataset")
    out = run_sanity(raw_dir)
    print("Sanity check complete.")
    print("Frame stats saved to:", out)


if __name__ == "__main__":
    main()
