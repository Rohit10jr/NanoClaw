"""Shared session key constants and helpers. (nanobot session/keys.py, c4dac64)

NanoClaw has no unified session, so nanobot's ``UNIFIED_SESSION_KEY`` and
``last_channel`` helpers are left out.
"""

from __future__ import annotations

HEARTBEAT_SESSION_KEY = "heartbeat"


def is_dream_session(key: str) -> bool:
    """Identify the per-run session namespace owned by Dream."""
    return key.startswith("dream:")


def is_internal_session(key: str) -> bool:
    """Identify maintenance sessions whose context is not a user conversation."""
    return key == HEARTBEAT_SESSION_KEY or is_dream_session(key)


def session_key_for_channel(channel: str, chat_id: str) -> str:
    """Return the session key for a channel/chat pair."""
    return f"{channel}:{chat_id}"
