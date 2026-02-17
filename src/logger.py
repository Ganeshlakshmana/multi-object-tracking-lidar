"""
logger.py

Central logging configuration for LiDAR Task 1 project.
"""

import logging
from pathlib import Path


def setup_logger(log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a project-wide logger.

    Args:
        log_level (int): Logging level (default: logging.INFO)

    Returns:
        logging.Logger: Configured root logger
    """
    logger = logging.getLogger("lidar_task1")
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
