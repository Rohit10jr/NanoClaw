"""Closing a turn that was interrupted before it could be saved.

Mirrors the parts of nanobot/session/recovery.py that a cancelled or crashed
turn needs:

- ``restore_pending_interruption``: the loop flags the session when it saves
  the user's message and clears the flag once the turn is saved. If the flag is
  still set when the next turn starts, the process died mid-turn: the
  unanswered message gets an honest error reply, so the model doesn't treat
  the old request as part of the new one.
- ``restore_runtime_checkpoint`` (since layer 14): while a turn runs, the loop
  keeps the runner's latest checkpoint in ``session.metadata`` (saved to disk
  at each step). A cancelled turn (``/stop``) materializes it at once; a turn
  whose process died (crash, kill, Ctrl-C mid-tool) leaves it on disk, and the
  next turn in that session materializes it before anything else (nanobot
  loop.py:2134). The assistant message and its finished tool results are
  saved, each unfinished tool call gets an "interrupted" result (never run
  again), and rows already at the end of the session are not added twice.
  Restoring is tolerant: rows of the wrong shape are left out, and a
  checkpoint that is not a dict is ignored.
- ``runtime_checkpoint_is_well_formed`` and ``discard_runtime_checkpoint``:
  the strict check nanobot uses before offering "Continue" on a checkpoint,
  and the way to drop one. No NanoClaw flow continues a checkpoint yet; they
  are here for the recovery flows that will.

Left out (nanobot's WebUI): the follow-up journal (``record_pending_followup``,
``acknowledge_pending_followups``), the recovery state shown to the WebUI,
``append_recovery_interruption`` and the "superseded" wording, request
origins of internal continuations, and provider conversation state (NanoClaw
keeps none, so nothing is cleared).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

from nanoclaw.session.manager import Session

RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
PENDING_USER_TURN_KEY = "pending_user_turn"
INTERRUPTED_TURN_MESSAGE = "Error: Task interrupted before a response was generated."
INTERRUPTED_TOOL_MESSAGE = "Error: Task interrupted before this tool finished."


def _checkpoint_message_key(message: Mapping[str, Any]) -> tuple[Any, ...]:
    """What makes two saved rows the same message. (nanobot recovery.py:172)"""
    return (
        message.get("role"),
        message.get("content"),
        message.get("tool_call_id"),
        message.get("name"),
        message.get("tool_calls"),
        message.get("reasoning_content"),
        message.get("thinking_blocks"),
    )


def _checkpoint_tool_call_ids(value: object, *, result_rows: bool = False) -> list[str] | None:
    """Validate checkpoint tool rows and return their IDs, or None.
    (nanobot recovery.py:184)"""
    if not isinstance(value, list):
        return None
    ids: list[str] = []
    for raw in cast(list[object], value):
        if not isinstance(raw, dict):
            return None
        row = cast(dict[str, Any], raw)
        call_id = row.get("tool_call_id" if result_rows else "id")
        if not isinstance(call_id, str) or not call_id:
            return None
        if result_rows:
            if row.get("role") != "tool":
                return None
        else:
            function = row.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(name, str) or not name:
                return None
        ids.append(call_id)
    return ids if len(ids) == len(set(ids)) else None


def runtime_checkpoint_is_well_formed(checkpoint: Mapping[str, Any]) -> bool:
    """Whether a checkpoint is exact enough to continue from.

    Restoring stays tolerant, so a broken checkpoint can always be cleared;
    continuing is stricter, because silently dropping a malformed tool result
    could make the model repeat a side effect. The phase must match the rows:
    ``final_response`` is text only, ``awaiting_tools`` has every call of the
    assistant message pending, ``tools_completed`` has every call answered.
    Error and unknown phases are never well formed. (nanobot recovery.py:216)
    """
    assistant = checkpoint.get("assistant_message")
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        return False
    completed_ids = _checkpoint_tool_call_ids(checkpoint.get("completed_tool_results"), result_rows=True)
    pending_ids = _checkpoint_tool_call_ids(checkpoint.get("pending_tool_calls"))
    if completed_ids is None or pending_ids is None:
        return False
    calls = assistant.get("tool_calls")
    call_ids = [] if calls is None else _checkpoint_tool_call_ids(calls)
    if call_ids is None:
        return False

    phase = checkpoint.get("phase")
    if phase == "final_response":
        content = assistant.get("content")
        return (
            isinstance(content, str) and bool(content.strip()) and not call_ids and not completed_ids and not pending_ids
        )
    if phase == "awaiting_tools":
        return (
            bool(call_ids)
            and not completed_ids
            and len(call_ids) == len(pending_ids)
            and set(call_ids) == set(pending_ids)
        )
    if phase == "tools_completed":
        return (
            bool(call_ids)
            and not pending_ids
            and len(call_ids) == len(completed_ids)
            and set(call_ids) == set(completed_ids)
        )
    return False


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


def discard_runtime_checkpoint(session: Session) -> bool:
    """Drop a checkpoint without saving any of it. (nanobot recovery.py:376)"""
    if RUNTIME_CHECKPOINT_KEY not in session.metadata:
        return False
    session.metadata.pop(RUNTIME_CHECKPOINT_KEY, None)
    session.updated_at = datetime.now()
    return True


def restore_pending_interruption(session: Session) -> bool:
    """Close a saved user-only turn without pretending it was answered.

    Returns True if the session changed (and should be saved).
    (nanobot recovery.py:386)
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
