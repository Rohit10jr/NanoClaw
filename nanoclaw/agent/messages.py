"""Message builders and the fixed texts the runner sends or returns.

Texts are copied from nanobot (utils/runtime.py, agent/runner.py,
templates/agent/max_iterations_message.md) so behaviour matches.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

TRUNCATED_SUFFIX = "\n... (truncated)"

FINALIZATION_RETRY_PROMPT = (
    "Please provide your response to the user based on the conversation above."
)

BUDGET_EXHAUSTED_FINALIZATION_PROMPT = (
    "The tool-call budget for this turn is exhausted. Based only on the "
    "conversation and tool results above, provide a concise final response to "
    "the user. Do not call or request tools. Do not claim the task is complete "
    "unless the evidence above clearly shows it is complete. State what was "
    "done, what remains, and the best next step if anything is incomplete."
)

EMPTY_FINAL_RESPONSE_MESSAGE = (
    "I completed the tool steps but couldn't produce a final answer. "
    "Please try again or narrow the task."
)

MODEL_ERROR_PLACEHOLDER = "[Assistant reply unavailable due to model error.]"

DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."

MAX_ITERATIONS_MESSAGE = (
    "I reached the maximum number of tool call iterations ({max_iterations}) "
    "without completing the task. You can try breaking the task into smaller steps."
)


MALFORMED_TOOL_CALL_NOTE = (
    "The previous model response attempted to call tools, but every tool call "
    "was malformed: the tool_use blocks had missing or non-string tool names. "
    "Do not answer with a promise to use tools. Either call the required tools again "
    "using valid tool names from the provided tool list and JSON object inputs, or give "
    "a final answer only if no tool is required."
)


def build_malformed_tool_call_note(assistant_text: str | None) -> dict[str, str]:
    note = MALFORMED_TOOL_CALL_NOTE
    if assistant_text:
        note += f"\n\nPrevious assistant text before the malformed calls:\n{assistant_text}"
    return {"role": "user", "content": note}


def build_assistant_message(
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    return msg


def build_tool_message(tool_call_id: str, name: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": content}


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATED_SUFFIX


def is_blank(text: str | None) -> bool:
    return text is None or not text.strip()


# Inline reasoning tags. Copied from nanobot utils/helpers.py:156-170.
def _tag_regex(tags: tuple[str, ...]) -> str:
    return rf"(?:{'|'.join(re.escape(tag) for tag in tags)})"


_THINKING_TAGS = ("think", "thinking", "thought")
_THINKING_TAG = _tag_regex(_THINKING_TAGS)
_INLINE_SELF_CLOSING_THINKING_TAG = r"(?:thinking)"
_THINKING_TAG_PREFIX = "|".join(
    sorted(
        {re.escape(tag[:i]) for tag in _THINKING_TAGS for i in range(1, len(tag) + 1)},
        key=len,
        reverse=True,
    )
)
_PARTIAL_THINKING_TAG = rf"</?(?:{_THINKING_TAG_PREFIX})>?"


def strip_think(text: str) -> str:
    """Remove inline reasoning blocks some models put in their answer text.

    Local reasoning models (DeepSeek-R1, Qwen3 via Ollama, ...) often reply
    with ``<think>...</think>answer``. Copied from nanobot utils/helpers.py:173
    and handles, in order:

    1. closed ``<think>`` / ``<thinking>`` / ``<thought>`` blocks;
    2. an unclosed block at the start (a stream still inside the block);
    3. ``<thinking/>`` markers at the edges;
    4. malformed opening tags with no ``>`` (``<think广场...``);
    5. orphan closing tags at the very start or end only;
    6. harmony-style ``<channel|>`` markers at the start only;
    7. a partial control tag cut off at the end of a stream chunk (``<thi``).

    Edge-only rules (5, 6) keep prose that discusses these tokens intact.
    """
    # Every supported control tag contains '<'; ordinary text only needs trimming.
    if "<" not in text:
        return text.strip()
    text = re.sub(rf"<(?P<tag>{_THINKING_TAG})>[\s\S]*?</(?P=tag)>", "", text)
    text = re.sub(rf"^\s*<{_THINKING_TAG}>[\s\S]*$", "", text)
    text = re.sub(rf"^\s*<{_INLINE_SELF_CLOSING_THINKING_TAG}/>\s*", "", text)
    text = re.sub(rf"\s*<{_INLINE_SELF_CLOSING_THINKING_TAG}/>\s*$", "", text)
    # Not \w after the tag name: in Unicode mode it matches CJK text too.
    text = re.sub(rf"<{_THINKING_TAG}(?![A-Za-z0-9_\-:>/])", "", text)
    text = re.sub(rf"^\s*</{_THINKING_TAG}>\s*", "", text)
    text = re.sub(rf"\s*</{_THINKING_TAG}>\s*$", "", text)
    text = re.sub(r"^\s*<\|?channel\|?>\s*", "", text)
    partial_control_tag = (
        rf"{_PARTIAL_THINKING_TAG}|"
        r"<\|?(?:c|ch|cha|chan|chann|channe|channel)(?:\|?>?)?"
    )
    text = re.sub(rf"(?:{partial_control_tag})$", "", text)
    text = re.sub(r"^\s*<\|?$", "", text)
    return text.strip()


def strip_reasoning_tags(text: object) -> str:
    """Remove wrapper tags from text already known to be reasoning.

    (nanobot utils/helpers.py:232)
    """
    if not isinstance(text, str):
        return ""
    if "<" not in text:
        return text.strip()
    text = re.sub(rf"^\s*<{_THINKING_TAG}/>\s*", "", text)
    text = re.sub(rf"\s*<{_THINKING_TAG}/>\s*$", "", text)
    text = re.sub(rf"^\s*<{_THINKING_TAG}>\s*", "", text)
    text = re.sub(rf"\s*</{_THINKING_TAG}>\s*$", "", text)
    text = re.sub(rf"\s*(?:{_PARTIAL_THINKING_TAG})$", "", text)
    return text.strip()


def extract_think(text: str) -> tuple[str | None, str]:
    """Split inline reasoning from the answer: ``(thinking, cleaned_text)``.

    Only closed blocks count as thinking; joined with blank lines.
    (nanobot utils/helpers.py:245)
    """
    if "<" not in text:
        return None, text.strip()
    parts = [
        m.group(2).strip()
        for m in re.finditer(rf"<(?P<tag>{_THINKING_TAG})>([\s\S]*?)</(?P=tag)>", text)
    ]
    thinking = "\n\n".join(parts) if parts else None
    return thinking, strip_think(text)


class IncrementalThinkExtractor:
    """Surface inline ``<think>`` reasoning from a growing stream buffer.

    Feed it the whole buffer after each delta; it emits only the reasoning
    text it hasn't emitted yet. (nanobot utils/helpers.py:261)
    """

    __slots__ = ("_emitted",)

    def __init__(self) -> None:
        self._emitted = ""

    def reset(self) -> None:
        self._emitted = ""

    async def feed(self, buf: str, emit: Callable[[str], Awaitable[Any]]) -> bool:
        """Emit new thinking found in ``buf``; True if anything was emitted."""
        thinking, _ = extract_think(buf)
        if not thinking or thinking == self._emitted:
            return False
        new = thinking[len(self._emitted):]
        if not new.strip():
            return False
        self._emitted = thinking
        await emit(new)
        return True
