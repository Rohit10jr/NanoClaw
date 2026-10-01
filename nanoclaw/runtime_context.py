"""Per-turn context attached to the user's message.

Mirrors nanobot/runtime_context.py. Information that changes every turn is
appended to the *current user message* inside clearly marked tags, rather than
put in the system prompt. The system prompt then stays byte-for-byte the same
between turns, so providers can cache it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

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


def append_runtime_context(content: Any, blocks: Sequence[RuntimeContextBlock]) -> Any:
    """Return ``content`` with the blocks appended (text or a list of content parts)."""
    if not blocks:
        return content
    rendered = [block.content for block in blocks]
    if isinstance(content, list):
        return [*content, *({"type": "text", "text": text} for text in rendered)]
    text = "" if content is None else str(content)
    suffix = "\n\n".join(rendered)
    return f"{text}\n\n{suffix}" if text else suffix


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
