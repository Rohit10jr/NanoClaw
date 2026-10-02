"""Provider-neutral types and the LLMProvider interface.

Mirrors nanobot/providers/base.py. Messages everywhere in NanoClaw use the
OpenAI chat format ({"role": ..., "content": ..., "tool_calls": [...]});
each provider translates to and from its own wire format.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# finish_reason values after which tool calls may run. Anything else
# ("length", "content_filter", "error", ...) means the turn was cut short.
_TOOL_CAPABLE_FINISH_REASONS = frozenset({"tool_calls", "function_call", "stop"})


def parse_tool_arguments(arguments: Any) -> Any:
    """Parse a tool call's JSON arguments without guessing.

    Valid JSON becomes a Python value (normally a dict); an empty or missing
    value becomes ``{}``. Malformed JSON is returned unchanged as a string so
    the tool registry can reject it with a clear error instead of running the
    tool with invented parameters.
    """
    if arguments is None:
        return {}
    if not isinstance(arguments, str):
        return arguments
    stripped = arguments.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return arguments
    return arguments if parsed is None else parsed


@dataclass
class ToolCallRequest:
    """One tool call the model asked for."""

    id: str
    name: str
    arguments: Any

    def has_valid_name(self) -> bool:
        """Some models emit calls with an empty name; those cannot run."""
        return isinstance(self.name, str) and bool(self.name)

    def to_openai_tool_call(self) -> dict[str, Any]:
        """Serialize for the assistant message stored in history."""
        arguments = (
            self.arguments
            if isinstance(self.arguments, str)
            else json.dumps(self.arguments, ensure_ascii=False)
        )
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": arguments},
        }


@dataclass(frozen=True)
class LLMUsage:
    """Token counts reported by the provider for one call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    """A provider's answer, normalized.

    Errors are responses too (``finish_reason == "error"`` with the message in
    ``content``), so callers never need try/except around a model call.
    """

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: LLMUsage | None = None
    # Thinking text some models return separately (DeepSeek-R1, Kimi, ...).
    reasoning_content: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def should_execute_tools(self) -> bool:
        """Run tools only if the model asked for them and stopped normally."""
        return self.has_tool_calls and self.finish_reason in _TOOL_CAPABLE_FINISH_REASONS

    @property
    def is_error(self) -> bool:
        return self.finish_reason == "error"


def tool_arguments_json_for_replay(arguments: Any) -> str:
    """JSON object string for a tool call being *replayed* from history.

    Unlike :func:`parse_tool_arguments` (used for calls about to run), this may
    replace bad arguments: malformed JSON or a non-object becomes ``"{}"`` so
    the request isn't rejected. nanobot first tries ``json_repair``.
    """
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return "{}"
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
    return "{}"


def enforce_role_alternation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive user (or assistant) messages; drop trailing assistant messages.

    Many OpenAI-compatible servers (vLLM, Ollama, Azure, ...) reject two user
    messages in a row or a request ending with an assistant message. This
    happens, for example, when the runner adds a "please answer" prompt right
    after the user's message. Input messages are not mutated.
    """
    merged: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if merged and role in ("user", "assistant") and merged[-1].get("role") == role:
            prev = merged[-1]
            if role == "assistant":
                if msg.get("tool_calls"):
                    merged[-1] = dict(msg)
                    continue
                if prev.get("tool_calls"):
                    continue
            prev_content = prev.get("content") or ""
            curr_content = msg.get("content") or ""
            if isinstance(prev_content, str) and isinstance(curr_content, str):
                prev["content"] = (prev_content + "\n\n" + curr_content).strip()
            else:
                merged[-1] = dict(msg)
        else:
            merged.append(dict(msg))
    while merged and merged[-1].get("role") == "assistant":
        merged.pop()
    return merged


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop internal ``_meta`` and replace empty string content, which most chat APIs reject.

    An assistant message that only carries tool calls gets ``None``; any other
    empty message gets a visible placeholder. Input messages are not mutated.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if "_meta" in msg:  # internal bookkeeping, never sent
            msg = {k: v for k, v in msg.items() if k != "_meta"}
        if msg.get("content") == "":
            msg = dict(msg)
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["content"] = None
            else:
                msg["content"] = "(empty)"
        result.append(msg)
    return result


class LLMProvider(ABC):
    """Base class for model backends."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "",
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base
        self.default_model = default_model

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send one chat request and return the normalized response.

        Args:
            messages: OpenAI-format messages.
            tools: OpenAI-format function tool definitions.
            model: Model name; falls back to ``default_model``.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            tool_choice: "auto", "required", "none" or a specific tool dict.
        """

    async def safe_chat(self, **kwargs: Any) -> LLMResponse:
        """Call :meth:`chat`, turning any unexpected exception into an error response.

        Cancellation (``asyncio.CancelledError``) is a ``BaseException`` and
        still propagates, so a user can interrupt a turn.
        """
        try:
            return await self.chat(**kwargs)
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            return LLMResponse(content=f"Error calling LLM: {detail}", finish_reason="error")
