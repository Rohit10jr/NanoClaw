"""Closing a turn that was interrupted before it could be saved.

Mirrors the parts of nanobot/session/recovery.py that a cancelled or crashed
turn needs:

- ``restore_pending_interruption``: the loop flags the session when it saves
  the user's message and clears the flag once the turn is saved. If the flag is
  still set when the next turn starts, the process died mid-turn: the
  unanswered message gets an honest error reply, so the model doesn't treat
  the old request as part of the new one.
- ``restore_runtime_checkpoint`` (since layer 14): while a turn runs, the loop
  keeps the runner's latest checkpoint in ``session.metadata``. A cancelled
  turn (``/stop``) materializes it: the assistant message and its finished
  tool results are saved, and each unfinished tool call gets an
  "interrupted" result (never run again).

Left for later: nanobot's WebUI follow-up journal, the "Continue" recovery
flow, checkpoint validation for it, and provider conversation state.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from nanoclaw.session.manager import Session

RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
PENDING_USER_TURN_KEY = "pending_user_turn"
INTERRUPTED_TURN_MESSAGE = "Error: Task interrupted before a response was generated."
INTERRUPTED_TOOL_MESSAGE = "Error: Task interrupted before this tool finished."


def _checkpoint_message_key(message: Mapping[str, Any]) -> tuple[Any, ...]:
    """What makes two saved rows the same message. (nanobot recovery.py:170)"""
    return (
        message.get("role"),
        message.get("content"),
        message.get("tool_call_id"),
        message.get("name"),
        message.get("tool_calls"),
        message.get("reasoning_content"),
        message.get("thinking_blocks"),
    )


def restore_runtime_checkpoint(session: Session) -> bool:
    """Save the turn's latest checkpoint into the history, once, and clear it.

    Pending tool calls become interrupted tool results; they are never run
    here. Rows already at the end of the session (the same checkpoint
    restored twice) are not added again. Returns False when there is no
    checkpoint. (nanobot recovery.py:278, without provider state)
    """
    checkpoint = session.metadata.get(RUNTIME_CHECKPOINT_KEY)
    if not isinstance(checkpoint, dict):
        return False
    assistant = checkpoint.get("assistant_message")
    completed = checkpoint.get("completed_tool_results")
    pending = checkpoint.get("pending_tool_calls")
    now = datetime.now().isoformat()

    restored: list[dict[str, Any]] = []
    if isinstance(assistant, dict) and assistant.get("role") == "assistant":
        row = dict(assistant)
        row.setdefault("timestamp", now)
        restored.append(row)
    for value in completed if isinstance(completed, list) else []:
        if isinstance(value, dict) and value.get("role") == "tool":
            row = dict(value)
            row.setdefault("timestamp", now)
            restored.append(row)
    for value in pending if isinstance(pending, list) else []:
        if not isinstance(value, dict):
            continue
        call_id = value.get("id")
        if not isinstance(call_id, str) or not call_id:
            continue
        function = value.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        restored.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name if isinstance(name, str) and name else "tool",
                "content": INTERRUPTED_TOOL_MESSAGE,
                "timestamp": now,
                "_recovery_interrupted": True,
            }
        )

    overlap = 0
    for size in range(min(len(session.messages), len(restored)), 0, -1):
        if all(
            _checkpoint_message_key(left) == _checkpoint_message_key(right)
            for left, right in zip(session.messages[-size:], restored[:size], strict=True)
        ):
            overlap = size
            break
    session.messages.extend(restored[overlap:])

    session.metadata.pop(PENDING_USER_TURN_KEY, None)
    session.metadata.pop(RUNTIME_CHECKPOINT_KEY, None)
    session.updated_at = datetime.now()
    return True


def restore_pending_interruption(session: Session) -> bool:
    """Close a saved user-only turn without pretending it was answered.

    Returns True if the session changed (and should be saved).
    (nanobot recovery.py:385)
    """
    if not session.metadata.get(PENDING_USER_TURN_KEY):
        return False
    if session.messages and session.messages[-1].get("role") == "user":
        session.messages.append(
            {
                "role": "assistant",
                "content": INTERRUPTED_TURN_MESSAGE,
                "timestamp": datetime.now().isoformat(),
                "_recovery_interrupted": True,
            }
        )
        session.updated_at = datetime.now()
    session.metadata.pop(PENDING_USER_TURN_KEY, None)
    return True


def finish_turn(session: Session) -> None:
    """Clear the turn's recovery markers before it is saved. (nanobot session/state.py:660)"""
    session.metadata.pop(PENDING_USER_TURN_KEY, None)
    session.metadata.pop(RUNTIME_CHECKPOINT_KEY, None)
