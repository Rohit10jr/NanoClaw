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


# Moonshot's Kimi K3 models have a fixed temperature and take
# max_completion_tokens (nanobot/providers/registry.py:213).
_KIMI_K3_MODELS = frozenset({"kimi-k3", "kimi-k3.1-pro", "kimi-k3.1-pro-highspeed"})


def _model_slug(model_name: str) -> str:
    """``openai/GPT-5`` -> ``gpt-5`` (nanobot openai_compat_provider.py:147)."""
    return model_name.lower().rsplit("/", 1)[-1]


def _requires_max_completion_tokens(model_name: str) -> bool:
    """Reasoning models reject ``max_tokens`` (nanobot openai_compat_provider.py:151).

    Matches the bare model name, so gateway names like ``openai/o3-mini``
    count too; ``foo3-mini`` does not.
    """
    slug = _model_slug(model_name)
    return (
        slug in _KIMI_K3_MODELS
        or "gpt-5" in slug
        or any(slug == p or slug.startswith((p + "-", p + ".")) for p in ("o1", "o3", "o4"))
    )


def _supports_temperature(model_name: str) -> bool:
    """False for fixed-temperature models (nanobot openai_compat_provider.py:859).

    nanobot also takes a reasoning_effort: GPT-6 accepts temperature only with
    effort "none". NanoClaw has no effort setting, so GPT-6 never gets one.
    Unlike the slug match above, this is a substring test on the full name,
    exactly as nanobot does it.
    """
    if _model_slug(model_name) in _KIMI_K3_MODELS:
        return False
    name = model_name.lower()
    if "gpt-6" in name:
        return False
    return not any(token in name for token in ("gpt-5", "o1", "o3", "o4"))


def _is_direct_openai(api_base: str | None) -> bool:
    """True when requests go to api.openai.com (no base means the SDK default).

    nanobot's "openai" spec sets supports_max_completion_tokens=True
    (nanobot/providers/registry.py:504), so every model sent there uses
    ``max_completion_tokens``.
    """
    return not api_base or "api.openai.com" in api_base.lower()


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
        model_name = model or self.default_model
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": self._sanitize_messages(messages),
        }
        # nanobot openai_compat_provider.py:919-927.
        if _supports_temperature(model_name):
            kwargs["temperature"] = temperature
        if _is_direct_openai(self.api_base) or _requires_max_completion_tokens(model_name):
            kwargs["max_completion_tokens"] = max(1, max_tokens)
        else:
            kwargs["max_tokens"] = max(1, max_tokens)
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
        usage = OpenAICompatProvider._parse_usage(data.get("usage"))
        if not choices:
            # Some gateways reply with top-level text instead of choices
            # (nanobot openai_compat_provider.py:1470).
            content = _extract_text(data.get("content") or data.get("output_text"))
            if content is not None:
                return LLMResponse(
                    content=content,
                    finish_reason=str(data.get("finish_reason") or "stop"),
                    usage=usage,
                    reasoning_content=_extract_text(data.get("reasoning_content")),
                )
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

        reasoning = message.get("reasoning_content") or message.get("reasoning")
        return LLMResponse(
            content=_extract_text(message.get("content")),
            tool_calls=tool_calls,
            finish_reason=str(choice.get("finish_reason") or "stop"),
            usage=usage,
            reasoning_content=reasoning if isinstance(reasoning, str) else None,
        )

    @staticmethod
    def _parse_usage(raw_usage: Any) -> LLMUsage | None:
        if not isinstance(raw_usage, dict) or not raw_usage:
            return None
        return LLMUsage(
            input_tokens=int(raw_usage.get("prompt_tokens") or 0),
            output_tokens=int(raw_usage.get("completion_tokens") or 0),
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
