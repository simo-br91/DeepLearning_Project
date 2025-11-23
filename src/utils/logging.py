# src/utils/logging.py

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_file: str | Path = "experiments/logs/train.log"):
    """
    Initialize Python logging with a single file handler + console.

    IMPORTANT:
    - If the user passes "experiments/logs/train.log", we MUST NOT
      re-prepend "experiments/logs" again.
    - So we treat log_file as a *full* path.
    """

    log_file = Path(log_file)

    # Ensure directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.getLogger().info(f"Logging initialized. Writing to: {log_file}")
