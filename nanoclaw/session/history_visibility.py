"""Visibility helpers for persisted session history messages.

Mirrors nanobot/session/history_visibility.py and the parts of
session/automation_turns.py the WebUI history projection needs. A message is
hidden from chat views when it carries the ``_hidden_history`` marker (the
compaction checkpoint, recovery notes) or is the trigger record of an
automation turn (cron).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nanoclaw.session.automation_spec import AUTOMATION_HISTORY_META
from nanoclaw.session.summary import HIDDEN_HISTORY_META

# nanobot registers the cron and local-trigger automation specs. NanoClaw has
# only cron; "local_trigger" stays a recognised kind so stored sources from
# either project project the same way.
_CRON_LEGACY_HISTORY_META = "_cron_turn"
_AUTOMATION_KINDS = frozenset({"trigger", "cron", "local_trigger"})

__all__ = [
    "HIDDEN_HISTORY_META",
    "is_automation_history_message",
    "is_automation_kind",
    "is_hidden_history_message",
]


def _has_hidden_history_marker(message: Mapping[str, Any] | None) -> bool:
    if not message:
        return False
    marker = message.get(HIDDEN_HISTORY_META)
    return marker is True or isinstance(marker, Mapping)


def is_automation_history_message(message: Mapping[str, Any] | None) -> bool:
    """True for hidden automation trigger records in session history."""
    if not message:
        return False
    marker = message.get(AUTOMATION_HISTORY_META)
    if marker is True or isinstance(marker, Mapping):
        return True
    return message.get(_CRON_LEGACY_HISTORY_META) is True


def is_automation_kind(value: Any) -> bool:
    return isinstance(value, str) and value in _AUTOMATION_KINDS


def is_hidden_history_message(message: Mapping[str, Any] | None) -> bool:
    """True for persisted messages that should not be shown as chat turns."""
    return _has_hidden_history_marker(message) or is_automation_history_message(message)
