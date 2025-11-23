# src/utils/logging.py

import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "fer_project",
    log_dir: str | Path = "experiments/logs",
    log_level: int = logging.INFO,
    reset_handlers: bool = False,
) -> logging.Logger:
    """
    Create and configure a logger that logs to both console and a file.

    Parameters
    ----------
    name : str
        Name of the logger (e.g. experiment or module name).
    log_dir : str | Path
        Directory where the log file will be written.
    log_level : int
        Logging level (e.g. logging.INFO, logging.DEBUG).
    reset_handlers : bool
        If True, existing handlers on this logger are removed first.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if reset_handlers:
        logger.handlers.clear()

    # avoid adding duplicate handlers if called multiple times
    if not logger.handlers:
        log_file = log_dir / f"{name}.log"

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)

        # Format
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.debug(f"Logger '{name}' initialized. Log file: {log_file}")

    return logger
