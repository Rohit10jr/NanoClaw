"""Provider for any OpenAI-compatible Chat Completions API.

Mirrors nanobot/providers/openai_compat_provider.py. One class covers OpenAI,
OpenRouter, DeepSeek, Groq, Ollama, vLLM and others: only the base URL, key
and model name differ.
"""

from __future__ import annotations

import secrets
import string
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from nanoclaw.providers.base import (
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ToolCallRequest,
    enforce_role_alternation,
    parse_tool_arguments,
    sanitize_messages,
    tool_arguments_json_for_replay,
)

_ALNUM = string.ascii_letters + string.digits

# Message keys the Chat Completions API understands; anything else is dropped.
_ALLOWED_MSG_KEYS = frozenset(
    {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"}
)


def _short_tool_id() -> str:
    """9-char alphanumeric ID, accepted by every provider (Mistral is strictest)."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def _extract_text(value: Any) -> str | None:
    """Content is usually a string, but some gateways return a list of parts."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            item["text"]
            for item in value
            if isinstance(item, dict)
            and item.get("type") != "thinking"
            and isinstance(item.get("text"), str)
        ]
        return "".join(parts) or None
    return str(value)


class OpenAICompatProvider(LLMProvider):
    """Chat Completions over the official ``openai`` SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "gpt-4o-mini",
        *,
        client: AsyncOpenAI | None = None,
        max_retries: int = 2,
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__(api_key, api_base, default_model)
        # nanobot sets max_retries=0 and runs its own retry policy; until we
        # build one, the SDK's retries (with backoff on 429/5xx) cover us.
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        # Created lazily so constructing a provider never touches the network.
        self._client = client

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                # Local servers (Ollama, vLLM) need no key, but the SDK requires one.
                api_key=self.api_key or "no-key",
                base_url=self.api_base or None,
                max_retries=self._max_retries,
                timeout=self._timeout_s,
            )
        return self._client

    def build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the keyword arguments for ``chat.completions.create``."""
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._sanitize_messages(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        return kwargs

    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Make stored history safe to send (nanobot: _sanitize_messages).

        Drops unknown keys, rewrites replayed tool-call arguments as valid JSON
        objects, fixes empty content, and enforces role alternation.
        """
        cleaned: list[dict[str, Any]] = []
        for msg in sanitize_messages(messages):
            clean = {k: v for k, v in msg.items() if k in _ALLOWED_MSG_KEYS}
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = None
            if clean.get("tool_calls"):
                clean["tool_calls"] = [
                    {
                        **tc,
                        "function": {
                            **tc.get("function", {}),
                            "arguments": tool_arguments_json_for_replay(
                                tc.get("function", {}).get("arguments")
                            ),
                        },
                    }
                    for tc in clean["tool_calls"]
                ]
            cleaned.append(clean)
        return enforce_role_alternation(cleaned)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs = self.build_kwargs(messages, tools, model, max_tokens, temperature, tool_choice)
        try:
            raw = await self._get_client().chat.completions.create(**kwargs)
        except Exception as exc:
            return self._error_response(exc)
        return self.parse(raw.model_dump())

    @staticmethod
    def parse(data: dict[str, Any]) -> LLMResponse:
        """Convert a Chat Completions response (as a dict) into an LLMResponse."""
        choices = data.get("choices") or []
        if not choices:
            return LLMResponse(content="Error: API returned empty choices.", finish_reason="error")

        choice = choices[0]
        message = choice.get("message") or {}

        tool_calls: list[ToolCallRequest] = []
        seen_ids: set[str] = set()
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            # Some providers omit IDs or reuse one ID for parallel calls; each
            # tool result must answer a unique ID, so make them unique here.
            call_id = str(tc.get("id") or "")
            if not call_id or call_id in seen_ids:
                call_id = _short_tool_id()
            seen_ids.add(call_id)
            tool_calls.append(
                ToolCallRequest(
                    id=call_id,
                    name=str(fn.get("name") or ""),
                    arguments=parse_tool_arguments(fn.get("arguments")),
                )
            )

        usage = None
        raw_usage = data.get("usage")
        if raw_usage:
            usage = LLMUsage(
                input_tokens=int(raw_usage.get("prompt_tokens") or 0),
                output_tokens=int(raw_usage.get("completion_tokens") or 0),
            )

        reasoning = message.get("reasoning_content") or message.get("reasoning")
        return LLMResponse(
            content=_extract_text(message.get("content")),
            tool_calls=tool_calls,
            finish_reason=str(choice.get("finish_reason") or "stop"),
            usage=usage,
            reasoning_content=reasoning if isinstance(reasoning, str) else None,
        )

    @staticmethod
    def _error_response(exc: Exception) -> LLMResponse:
        if isinstance(exc, APIStatusError):
            # The SDK's exc.message embeds the whole JSON body; prefer the
            # API's own human-readable message when it sent one.
            body = exc.body
            message = body.get("message") if isinstance(body, dict) else None
            detail = f"HTTP {exc.status_code}: {message or exc.message}"
        else:
            detail = str(exc).strip() or type(exc).__name__
        return LLMResponse(content=f"Error calling LLM: {detail}", finish_reason="error")
