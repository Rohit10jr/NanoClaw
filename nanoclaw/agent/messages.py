"""Message builders and the fixed texts the runner sends or returns.

Texts are copied from nanobot (utils/runtime.py, agent/runner.py,
templates/agent/max_iterations_message.md) so behaviour matches.
"""

from __future__ import annotations

import re
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


_THINKING_TAG = r"(?:think|thinking|thought)"


def strip_think(text: str) -> str:
    """Remove inline reasoning blocks some models put in their answer text.

    Local reasoning models (DeepSeek-R1, Qwen3 via Ollama, ...) often reply
    with ``<think>...</think>answer``. Removes closed blocks and an unclosed
    block at the start. nanobot's version (utils/helpers.py strip_think) also
    cleans up malformed tags and streaming fragments.
    """
    if "<" not in text:
        return text.strip()
    text = re.sub(rf"<(?P<tag>{_THINKING_TAG})>[\s\S]*?</(?P=tag)>", "", text)
    text = re.sub(rf"^\s*<{_THINKING_TAG}>[\s\S]*$", "", text)
    return text.strip()
