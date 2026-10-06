"""The conversation summary left behind by compaction.

Mirrors nanobot/session/summary.py and history_visibility.py. When old messages
are compacted, the session gets a hidden marker message at the boundary (so the
saved transcript shows where the summary starts) and the summary text is kept in
``session.metadata["_last_summary"]``, from where it goes into the system prompt.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, TypedDict

HIDDEN_HISTORY_META = "_hidden_history"
SUMMARY_CONTINUATION_TEXT = "Continue the active task from the working-memory checkpoint above."


class SessionSummary(TypedDict):
    text: str
    last_active: str


def is_hidden_history_message(message: Mapping[str, Any]) -> bool:
    marker = message.get(HIDDEN_HISTORY_META)
    return marker is True or isinstance(marker, Mapping)


def is_summary_checkpoint(message: Mapping[str, Any]) -> bool:
    """The hidden marker that starts the part of the session after a summary."""
    return is_hidden_history_message(message) and message.get("content") == SUMMARY_CONTINUATION_TEXT


def session_summary_from_metadata(
    metadata: Mapping[str, Any] | None, *, fallback_last_active: datetime
) -> SessionSummary | None:
    """The stored summary, if there is a valid one."""
    raw = metadata.get("_last_summary") if metadata is not None else None
    if not isinstance(raw, Mapping):
        return None
    text = raw.get("text")
    if not isinstance(text, str) or not text:
        return None
    last_active = raw.get("last_active")
    if isinstance(last_active, str):
        try:
            datetime.fromisoformat(last_active)
        except ValueError:
            last_active = fallback_last_active.isoformat()
    else:
        last_active = fallback_last_active.isoformat()
    return {"text": text, "last_active": last_active}
