"""
exceptions.py

Custom exception definitions for LiDAR Task 1 project.
"""


class FrameParsingError(Exception):
    """Raised when frame ID cannot be parsed from filename."""


class SchemaValidationError(Exception):
    """Raised when required CSV columns are missing."""


class DataIntegrityError(Exception):
    """Raised when data fails validation checks."""
