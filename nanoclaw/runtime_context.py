"""Per-turn context attached to the user's message.

Mirrors nanobot/runtime_context.py. Information that changes every turn is
appended to the *current user message* inside clearly marked tags, rather than
put in the system prompt. The system prompt then stays byte-for-byte the same
between turns, so providers can cache it.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Where the marker lives: on a saved session message, and inside ``_meta`` on
# the message being built for this turn (``_meta`` is never sent to the model).
RUNTIME_CONTEXT_HISTORY_META = "_runtime_context"
RUNTIME_CONTEXT_MESSAGE_META = "runtime_context"
RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
RUNTIME_CONTEXT_END = "[/Runtime Context]"


@dataclass(frozen=True)
class RuntimeContextBlock:
    """Context appended verbatim to the current user content.

    ``source`` names who produced it (e.g. ``"time"``); ``content`` is the text.
    """

    source: str
    content: str


def wrap_runtime_context_lines(lines: Iterable[str]) -> str:
    """Wrap non-empty lines in the runtime-context markers ("" if there are none)."""
    content = "\n".join(line for line in lines if line)
    if not content:
        return ""
    return f"{RUNTIME_CONTEXT_TAG}\n{content}\n{RUNTIME_CONTEXT_END}"


def append_runtime_context(
    content: Any, blocks: Sequence[RuntimeContextBlock]
) -> tuple[Any, dict[str, Any] | None]:
    """Append the blocks to ``content`` (text or a list of content parts).

    Also returns a marker recording exactly what was appended (``None`` when
    nothing was). Sessions store it with the message, so the appended part can
    later be told apart from what the user typed.
    """
    if not blocks:
        return content, None
    rendered = [block.content for block in blocks]
    sources = [block.source for block in blocks]
    if isinstance(content, list):
        context_blocks = [{"type": "text", "text": text} for text in rendered]
        return [*content, *context_blocks], {"version": 1, "sources": sources, "blocks": context_blocks}
    text = "" if content is None else str(content)
    suffix = "\n\n".join(rendered)
    merged = f"{text}\n\n{suffix}" if text else suffix
    return merged, {"version": 1, "sources": sources, "suffix": suffix}


def detach_runtime_context(
    content: Any, marker: dict[str, Any]
) -> tuple[Any, list[str], list[dict[str, Any]]] | None:
    """Split ``content`` into (what the user wrote, sources, appended blocks).

    Uses the marker from ``append_runtime_context``; returns ``None`` if the
    content no longer ends with what the marker describes.
    """
    if marker.get("version") != 1:
        return None
    raw_sources = marker.get("sources")
    sources = [x for x in raw_sources if isinstance(x, str) and x] if isinstance(raw_sources, list) else []
    suffix = marker.get("suffix")
    if isinstance(content, str) and isinstance(suffix, str) and suffix:
        if content == suffix:
            clean = ""
        elif content.endswith("\n\n" + suffix):
            clean = content[: -(len(suffix) + 2)]
        else:
            return None
        return clean, sources, [{"type": "text", "text": suffix}]
    expected = marker.get("blocks")
    if isinstance(content, list) and isinstance(expected, list) and expected:
        count = len(expected)
        if content[-count:] != expected:
            return None
        return content[:-count], sources, copy.deepcopy(expected)
    return None


def reattach_runtime_context(
    content: Any, sources: Sequence[str], blocks: Sequence[dict[str, Any]]
) -> tuple[Any, dict[str, Any]]:
    """Append detached blocks again (after merging messages); returns a new marker."""
    context_blocks = [copy.deepcopy(dict(block)) for block in blocks]
    if isinstance(content, str) and all(
        block.get("type") == "text" and isinstance(block.get("text"), str) for block in context_blocks
    ):
        suffix = "\n\n".join(block["text"] for block in context_blocks)
        merged = f"{content}\n\n{suffix}" if content else suffix
        return merged, {"version": 1, "sources": list(sources), "suffix": suffix}
    visible = list(content) if isinstance(content, list) else ([] if content is None else [{"type": "text", "text": str(content)}])
    return [*visible, *context_blocks], {"version": 1, "sources": list(sources), "blocks": context_blocks}


def public_history_message(message: dict[str, Any]) -> dict[str, Any]:
    """A copy of a saved message with its runtime context removed exactly.

    Used where history is shown or summarized as what the user wrote, e.g.
    the raw fallback of memory archiving. (nanobot ``public_history_message``)
    """
    cleaned = copy.deepcopy(dict(message))
    marker = cleaned.pop(RUNTIME_CONTEXT_HISTORY_META, None)
    if not isinstance(marker, dict) or marker.get("version") != 1:
        return cleaned
    content = cleaned.get("content")
    suffix = marker.get("suffix")
    if isinstance(content, str) and isinstance(suffix, str) and suffix:
        if content == suffix:
            cleaned["content"] = ""
        elif content.endswith("\n\n" + suffix):
            cleaned["content"] = content[: -(len(suffix) + 2)]
        return cleaned
    expected = marker.get("blocks")
    if isinstance(content, list) and isinstance(expected, list) and expected:
        if content[-len(expected):] == expected:
            cleaned["content"] = content[: -len(expected)]
    return cleaned


def current_time_block(now: datetime | None = None) -> RuntimeContextBlock:
    """The current local date, time and timezone.

    NanoClaw addition: nanobot (at the reference commit) gives the model no
    date or time at all. See docs/notes/04-context.md, "Differences from nanobot".
    """
    now = now or datetime.now()
    if now.tzinfo is None:  # a naive time is local time; attach the local zone
        now = now.astimezone()
    offset = now.strftime("%z")  # "+0530"
    utc = f"UTC{offset[:3]}:{offset[3:]}" if offset not in ("", "+0000") else "UTC"
    zone = now.tzname() or utc
    label = utc if zone in (utc, "UTC") else f"{zone}, {utc}"
    line = f"Current Time: {now:%Y-%m-%d %H:%M} ({now:%A}) ({label})"
    return RuntimeContextBlock(source="time", content=wrap_runtime_context_lines([line]))


# Set when a user turn ended at the tool-iteration limit (nanobot
# session/turn_continuation.py:32 ``_ITERATION_LIMIT_META``).
ITERATION_LIMIT_META = "_stopped_at_iteration_limit"


def record_iteration_limit(metadata: dict[str, Any], stop_reason: str) -> None:
    """Remember that a user turn stopped at the iteration limit; forget it otherwise.

    (nanobot session/turn_continuation.py:44 ``record_iteration_limit``; NanoClaw
    has no internal continuations, so ``continuation_pending`` is always false.)
    """
    if stop_reason == "max_iterations":
        metadata[ITERATION_LIMIT_META] = True
    else:
        metadata.pop(ITERATION_LIMIT_META, None)


def iteration_limit_block(metadata: dict[str, Any]) -> RuntimeContextBlock | None:
    """Tell the next user turn that the previous one was cut off, so the model
    follows the new request instead of resuming the old work.

    (nanobot session/turn_continuation.py:54 ``iteration_limit_context``, a
    runtime-context provider that nanobot skips for temporary sessions and
    internal continuations; the caller here only asks on saved user turns.)
    """
    if not metadata.get(ITERATION_LIMIT_META):
        return None
    return RuntimeContextBlock(
        source="iteration_limit",
        content=wrap_runtime_context_lines([
            "The previous turn stopped at the tool-iteration limit without finishing its work.",
            "Follow the current request. Resume unfinished work only if the user asks for it.",
            "Completed tool calls and their results remain in the history; do not repeat them blindly.",
        ]),
    )
