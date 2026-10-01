"""Message builders and the fixed texts the runner sends or returns.

Texts are copied from nanobot (utils/runtime.py, agent/runner.py,
templates/agent/max_iterations_message.md) so behaviour matches.
"""

from __future__ import annotations

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
