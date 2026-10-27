"""Turning NanoClaw's runtime logs on and off.

Mirrors nanobot/cli/log_control.py (``_set_nanobot_logs``) and the console sink
of nanobot/utils/log_config.py (``configure_console_logging``). The package
logger is disabled when ``nanoclaw`` is imported; a command such as
``nanoclaw chat --logs`` enables it.
"""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger

__all__ = ["LOG_FORMAT", "configure_console_logging", "set_nanoclaw_logs"]

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DDTHH:mm:ss.SSSZZ}</green> | "
    "<level>{level: <7}</level> | "
    "<cyan>{name}</cyan> | "
    "<level>{message}</level>"
)


def set_nanoclaw_logs(enabled: bool) -> None:
    """Show (``True``) or hide (``False``) every log record from the ``nanoclaw`` package."""
    if enabled:
        logger.enable("nanoclaw")
    else:
        logger.disable("nanoclaw")


def configure_console_logging(stream: Any = None, *, level: str = "INFO") -> int:
    """Replace loguru's handlers with one console sink and return its handler id.

    loguru's built-in handler prints DEBUG and up; nanobot's CLI swaps it for an
    INFO sink on stderr like this one.
    """
    logger.remove()
    return logger.add(stream if stream is not None else sys.stderr, format=LOG_FORMAT, level=level, colorize=None)
