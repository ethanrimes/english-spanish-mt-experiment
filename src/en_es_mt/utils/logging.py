"""Structured logging using rich. One configure() call at process start."""
from __future__ import annotations

import logging
import os

from rich.logging import RichHandler


def configure(level: str | int | None = None) -> None:
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
