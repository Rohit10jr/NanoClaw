"""Detect the host's IANA timezone for ``agents.defaults.timezone``.

nanobot ``config/timezone.py`` asks the ``tzlocal`` package. NanoClaw avoids
the extra dependency and checks the usual places itself: ``TZ``, the
``/etc/localtime`` symlink and ``/etc/timezone``. Anything else (including
Windows) falls back to ``UTC``, as nanobot does when detection fails.
"""

from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# nanobot config/timezone.py:7
_UTC_ALIASES = frozenset({"Etc/GMT", "Etc/UTC", "GMT", "GMT0", "Greenwich", "UCT", "Universal", "Zulu"})


def _local_zone_name() -> str:
    tz = os.environ.get("TZ", "").strip().lstrip(":")
    if tz:
        return tz
    try:
        target = os.path.realpath("/etc/localtime")
        if "zoneinfo/" in target:
            return target.split("zoneinfo/", 1)[1]
    except OSError:
        pass
    try:
        return Path("/etc/timezone").read_text(encoding="utf-8").strip() or "UTC"
    except OSError:
        return "UTC"


def detect_system_timezone() -> str:
    """Return the host's IANA timezone, falling back safely to UTC (nanobot config/timezone.py:13)."""
    try:
        name = _local_zone_name()
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return "UTC"
    return "UTC" if name in _UTC_ALIASES else name
