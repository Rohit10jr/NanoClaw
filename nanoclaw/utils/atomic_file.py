"""Durable atomic replacement of a text file.

From nanobot/utils/atomic_file.py (``atomic_write_text``). Left out: the
Windows extended-path and POSIX-rename handling (``io_path``,
``replace_atomic``); NanoClaw uses ``os.replace``, which is atomic on POSIX
and replaces an unopened file on Windows.
"""

from __future__ import annotations

import errno
import os
import uuid
from contextlib import suppress
from pathlib import Path


def _fsync_directory_after_replace(directory: Path) -> None:
    """Fsync a directory, ignoring only unsupported platforms/filesystems."""
    with suppress(PermissionError):
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            try:
                os.fsync(fd)
            except OSError as exc:
                if exc.errno != errno.EINVAL:
                    raise
        finally:
            os.close(fd)


def atomic_write_text(path: Path, content: str) -> None:
    """Create parent directories and durably replace exact UTF-8 text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "x", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        if os.name != "nt":
            _fsync_directory_after_replace(path.parent)
    finally:
        tmp.unlink(missing_ok=True)
