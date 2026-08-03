"""Application-wide logging for diagnostics and recoverable failures."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core.settings import get_config_dir


LOG_FILENAME = "slunderstudio.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5
_HANDLER_MARKER = "_slunderstudio_file_handler"


def application_log_path(config_dir: str | Path | None = None) -> Path:
    """Return the rotating application log path beside ``crash.log``."""
    root = Path(config_dir) if config_dir is not None else get_config_dir()
    return root / LOG_FILENAME


def configure_logging(config_dir: str | Path | None = None) -> Path:
    """Install the bounded application log handler and return its path.

    Reconfiguration is deliberately idempotent. Tests and diagnostic launch
    paths can point the handler at a temporary directory without accumulating
    stale handlers or writing to a previous profile.
    """
    path = application_log_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root_logger.removeHandler(handler)
            handler.close()

    handler = RotatingFileHandler(
        path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)
    return path
