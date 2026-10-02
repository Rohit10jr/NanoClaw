"""Preparing the exact messages sent to the model.

Mirrors one step of nanobot's ``ContextGovernor.prepare_messages_for_model``
(agent/context_governance.py): adjacent user messages are merged in the copy
sent to the model, with runtime-context blocks (the time) kept at the end of
the merged message. The rest of nanobot's context governance (token budgets,
compaction) is a phase 2 feature.
"""

from __future__ import annotations

import copy
from typing import Any

from nanoclaw.runtime_context import (
    RUNTIME_CONTEXT_MESSAGE_META,
    detach_runtime_context,
    reattach_runtime_context,
)


def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
    if isinstance(left, str) and isinstance(right, str):
        return f"{left}\n\n{right}" if left else right

    def to_blocks(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
        if value is None:
            return []
        return [{"type": "text", "text": str(value)}]

    return to_blocks(left) + to_blocks(right)


def _runtime_marker(message: dict[str, Any]) -> dict[str, Any] | None:
    meta = message.get("_meta")
    marker = meta.get(RUNTIME_CONTEXT_MESSAGE_META) if isinstance(meta, dict) else None
    return marker if isinstance(marker, dict) else None


def merge_adjacent_user_messages_for_model(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent user messages; the input list is not changed.

    E.g. the user's message (ending in a time block) followed by a "please
    answer" prompt becomes one message: text, prompt, then the time block.
    """
    prepared: list[dict[str, Any]] = []
    for source in messages:
        message = copy.deepcopy(source)
        if not (prepared and message.get("role") == "user" and prepared[-1].get("role") == "user"):
            prepared.append(message)
            continue
        merged = dict(prepared[-1])
        left_marker, right_marker = _runtime_marker(merged), _runtime_marker(message)
        left = (
            detach_runtime_context(merged.get("content"), left_marker)
            if left_marker is not None
            else (merged.get("content"), [], [])
        )
        right = (
            detach_runtime_context(message.get("content"), right_marker)
            if right_marker is not None
            else (message.get("content"), [], [])
        )
        if left is not None and right is not None:
            content = _merge_message_content(left[0], right[0])
            blocks = [*left[2], *right[2]]
            if blocks:
                content, marker = reattach_runtime_context(content, [*left[1], *right[1]], blocks)
                meta = dict(merged.get("_meta") or {})
                for key, value in (message.get("_meta") or {}).items():
                    meta.setdefault(key, value)
                meta[RUNTIME_CONTEXT_MESSAGE_META] = marker
                merged["_meta"] = meta
            merged["content"] = content
        else:
            merged["content"] = _merge_message_content(merged.get("content"), message.get("content"))
        prepared[-1] = merged
    return prepared
