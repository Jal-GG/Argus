"""Logging helpers with consistent formatting across the system."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(
    level: str | int | None = None,
    log_file: str | os.PathLike[str] | None = None,
) -> None:
    """Configure root logging once; subsequent calls are no-ops."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATEFMT, handlers=handlers)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring defaults on first use."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
