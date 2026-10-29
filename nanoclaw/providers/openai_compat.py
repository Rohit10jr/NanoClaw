"""Provider for any OpenAI-compatible Chat Completions API.

Mirrors nanobot/providers/openai_compat_provider.py. One class covers OpenAI,
OpenRouter, DeepSeek, Groq, Ollama, vLLM and others: the base URL, key and
model name differ, and the provider's registry spec (``spec``) adds its
request rules: model-prefix stripping, ``max_completion_tokens``, per-model
overrides and how a ``reasoning_effort`` is sent.

Not ported: nanobot's Responses API path (nanobot sends GPT-5 / o-series
models on api.openai.com there; NanoClaw always uses Chat Completions),
OpenRouter's "nanobot" attribution headers, the random session-affinity
header and Gemini thought signatures.
"""

from __future__ import annotations

import asyncio
import inspect
import secrets
import string
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

from openai import APIStatusError, AsyncOpenAI

from nanoclaw.providers.base import (
    ContentDeltaCallback,
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ToolCallDeltaCallback,
    ToolCallRequest,
    enforce_role_alternation,
    parse_tool_arguments,
    resolve_stream_idle_timeout_s,
    sanitize_messages,
    tool_arguments_json_for_replay,
)

if TYPE_CHECKING:
    from nanoclaw.providers.registry import ProviderSpec

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

# nanobot openai_compat_provider.py:85-131.
_KIMI_THINKING_MODELS = frozenset(
    {"kimi-k2.5", "kimi-k2.6", "kimi-k2.7", "kimi-k2.7-code", "kimi-k2.7-code-highspeed", "k2.6-code-preview"}
)
_KIMI_ALWAYS_THINKING_MODELS = frozenset({"kimi-k2.7-code", "kimi-k2.7-code-highspeed"})
_KIMI_SERVER_MANAGED_TEMPERATURE_MODELS = frozenset({"kimi-k2.5", "kimi-k2.6"})
_MIMO_THINKING_MODELS = frozenset({"mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro", "mimo-v2-omni"})
_QWEN_THINKING_MODELS = frozenset(
    {
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-max-preview",
        "qwen3.6-plus",
        "qwen3.6-flash",
        "qwen3.5-plus",
        "qwen3.5-flash",
    }
)
_MODEL_THINKING_STYLES: dict[str, str] = {
    **dict.fromkeys(_KIMI_THINKING_MODELS, "thinking_type"),
    **dict.fromkeys(_MIMO_THINKING_MODELS, "thinking_type"),
    **dict.fromkeys(_QWEN_THINKING_MODELS, "enable_thinking"),
}

# ProviderSpec.thinking_style -> the extra_body for thinking on/off
# (nanobot openai_compat_provider.py:116).
_THINKING_STYLE_MAP: dict[str, Callable[[bool], dict[str, Any]]] = {
    "thinking_type": lambda on: {"thinking": {"type": "enabled" if on else "disabled"}},
    "enable_thinking": lambda on: {"enable_thinking": on},
    "reasoning_split": lambda on: {"reasoning_split": on},
}
_GATEWAY_REASONING_STYLE_MAP: dict[str, Callable[[str], dict[str, Any]]] = {
    "reasoning_effort": lambda effort: {"reasoning": {"effort": effort}},
}


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


def _supports_temperature(model_name: str, reasoning_effort: str | None = None) -> bool:
    """False for fixed-temperature models (nanobot openai_compat_provider.py:859).

    GPT-6 accepts a temperature only with reasoning effort "none". Unlike the
    slug match above, this is a substring test on the full name, exactly as
    nanobot does it.
    """
    if _model_slug(model_name) in _KIMI_K3_MODELS:
        return False
    name = model_name.lower()
    if "gpt-6" in name:
        return bool(reasoning_effort and reasoning_effort.lower() == "none")
    return not any(token in name for token in ("gpt-5", "o1", "o3", "o4"))


def _model_thinking_style(model_name: str) -> str:
    return _MODEL_THINKING_STYLES.get(_model_slug(model_name), "")


def _thinking_styles_for(spec: ProviderSpec | None, model_name: str) -> list[str]:
    """nanobot openai_compat_provider.py:163."""
    styles: list[str] = []
    if spec and spec.thinking_style:
        styles.append(spec.thinking_style)
    elif (
        spec
        and spec.name == "zhipu"
        and _model_slug(model_name) in {"glm-4.6", "glm-4.7", "glm-5", "glm-5.1", "glm-5.2"}
    ):
        # Model-specific: GLM-5.3 requires thinking and cannot disable it.
        styles.append("thinking_type")
    model_style = _model_thinking_style(model_name)
    if model_style and model_style not in styles:
        styles.append(model_style)
    return styles


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into a copy of ``base``; nested dicts key by key (nanobot :393)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(cast(dict[str, Any], merged[key]), cast(dict[str, Any], value))
        else:
            merged[key] = value
    return merged


def _merge_chat_extra_body(kwargs: dict[str, Any], extra_body: dict[str, Any]) -> dict[str, Any]:
    """Merge configured request fields without clobbering tools (nanobot :436)."""
    regular_extra = {key: value for key, value in extra_body.items() if key != "tools"}
    merged = dict(kwargs)
    if regular_extra:
        merged["extra_body"] = _deep_merge(kwargs.get("extra_body", {}), regular_extra)
    if "tools" in extra_body:
        current_tools = kwargs.get("tools")
        configured_tools = extra_body["tools"]
        if isinstance(current_tools, list) and isinstance(configured_tools, list):
            merged["tools"] = [*current_tools, *configured_tools]
        else:
            merged["tools"] = configured_tools
    return merged


def _is_direct_openai(api_base: str | None) -> bool:
    """True when requests go to api.openai.com (no base means the SDK default).

    nanobot's "openai" spec sets supports_max_completion_tokens=True
    (nanobot/providers/registry.py:504), so every model sent there uses
    ``max_completion_tokens``.
    """
    return not api_base or "api.openai.com" in api_base.lower()


def _extract_text(value: Any) -> str | None:
    """Content is usually a string, but some gateways return a list of parts.

    Mistral-style ``{"type": "thinking"}`` parts are skipped: they belong in
    the reasoning (nanobot openai_compat_provider.py:1328).
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") != "thinking" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts) or None
    return str(value)


def _extract_thinking(value: Any) -> str | None:
    """Text of Mistral's ``{"type": "thinking", "thinking": [...]}`` content parts.

    (nanobot openai_compat_provider.py:1356)
    """
    if not isinstance(value, list):
        return None
    parts = [
        text
        for item in value
        if isinstance(item, dict) and item.get("type") == "thinking"
        if (text := _extract_text(item.get("thinking")))
    ]
    return "".join(parts) or None


def _delta_reasoning(delta: dict[str, Any]) -> str | None:
    """Reasoning text in one stream delta (nanobot openai_compat_provider.py:1627-1633)."""
    return (
        _extract_text(delta.get("reasoning_content"))
        or _extract_text(delta.get("reasoning"))
        or _extract_thinking(delta.get("content"))
    )


def _as_mapping(value: Any) -> dict[str, Any] | None:
    """A dict, or an SDK model dumped to one (nanobot openai_compat_provider.py:1317)."""
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            # Gateways put lists where the SDK types expect strings (Mistral's
            # content parts); dump them as sent, without pydantic warnings.
            dumped = model_dump(warnings=False)
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return None


class OpenAICompatProvider(LLMProvider):
    """Chat Completions over the official ``openai`` SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "gpt-4o-mini",
        *,
        extra_headers: dict[str, str] | None = None,
        spec: ProviderSpec | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_query: dict[str, str] | None = None,
        reasoning_effort: str | None = None,
        provider_name: str = "openai",
        client: AsyncOpenAI | None = None,
        max_retries: int = 0,
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__(api_key, api_base, default_model)
        self.provider_name = provider_name
        self.extra_headers = extra_headers or {}
        self._spec = spec
        self._extra_body = dict(extra_body or {})
        self._extra_query = dict(extra_query or {})
        # The preset's reasoning effort, sent with every request (nanobot
        # passes it per call from ``GenerationSettings``; NanoClaw's chat
        # calls take no effort argument, so the provider holds it).
        self.reasoning_effort = reasoning_effort
        # The SDK's own retries are off: chat_with_retry is the one retry
        # policy, so retries don't multiply (nanobot openai_compat_provider.py:581).
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
                default_headers=self.extra_headers or None,
                default_query=self._extra_query or None,
                max_retries=self._max_retries,
                timeout=self._timeout_s,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the HTTP client (nanobot openai_compat_provider.py:611)."""
        client, self._client = self._client, None
        if client is not None:
            await client.close()

    def _request_model_name(self, model_name: str) -> str:
        return self._spec.request_model_name(model_name) if self._spec else model_name

    def _uses_max_completion_tokens(self) -> bool:
        """The spec decides (nanobot: ``supports_max_completion_tokens``).

        Without a spec, or for a ``custom`` entry with no apiBase (NanoClaw's
        api.openai.com default), requests to api.openai.com use it.
        """
        spec = self._spec
        if spec is not None and not (spec.name == "custom" and not self.api_base):
            return spec.supports_max_completion_tokens
        return _is_direct_openai(self.api_base)

    def build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the keyword arguments for ``chat.completions.create``
        (nanobot ``_build_kwargs``, openai_compat_provider.py:890)."""
        spec = self._spec
        reasoning_effort = self.reasoning_effort
        configured_model = model or self.default_model
        model_name = self._request_model_name(configured_model)
        messages = self._sanitize_messages(messages)
        if spec and spec.strip_history_reasoning_content:
            messages = [{k: v for k, v in msg.items() if k != "reasoning_content"} for msg in messages]
        # Anthropic models through a caching gateway (OpenRouter) get
        # cache_control markers (nanobot openai_compat_provider.py:904).
        if spec and spec.supports_prompt_caching and configured_model.lower().startswith(("anthropic/", "claude")):
            messages, tools = self._apply_cache_control(messages, tools)
        kwargs: dict[str, Any] = {"model": model_name, "messages": messages}

        if _supports_temperature(model_name, reasoning_effort):
            kwargs["temperature"] = temperature
        if self._uses_max_completion_tokens() or _requires_max_completion_tokens(model_name):
            kwargs["max_completion_tokens"] = max(1, max_tokens)
        else:
            kwargs["max_tokens"] = max(1, max_tokens)

        if spec:
            model_lower = model_name.lower()
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    break
        # Moonshot picks the only valid temperature from the K2.5/K2.6 thinking mode.
        if spec and spec.name == "moonshot" and _model_slug(model_name) in _KIMI_SERVER_MANAGED_TEMPERATURE_MODELS:
            kwargs.pop("temperature", None)

        self._apply_reasoning_effort(kwargs, spec, model_name, reasoning_effort)

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        # User-configured extra_body last, so it can override provider defaults.
        if self._extra_body:
            kwargs = _merge_chat_extra_body(kwargs, self._extra_body)
        return kwargs

    @staticmethod
    def _apply_cache_control(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Mark the system prompt, the second-to-last message and the tool
        list's built-in/MCP boundary and tail as cacheable (nanobot :620)."""
        cache_marker = {"type": "ephemeral"}
        new_messages = list(messages)

        def _mark(msg: dict[str, Any]) -> dict[str, Any]:
            content = msg.get("content")
            if isinstance(content, str):
                return {**msg, "content": [{"type": "text", "text": content, "cache_control": cache_marker}]}
            if isinstance(content, list) and content:
                parts = list(cast(list[dict[str, Any]], content))
                parts[-1] = {**parts[-1], "cache_control": cache_marker}
                return {**msg, "content": parts}
            return msg

        if new_messages and new_messages[0].get("role") == "system":
            new_messages[0] = _mark(new_messages[0])
        if len(new_messages) >= 3:
            new_messages[-2] = _mark(new_messages[-2])

        new_tools = tools
        if tools:
            new_tools = list(tools)

            def _tool_name(tool: dict[str, Any]) -> str:
                name = tool.get("name") or (tool.get("function") or {}).get("name")
                return name if isinstance(name, str) else ""

            tail_idx = len(new_tools) - 1
            last_builtin_idx = next(
                (i for i in range(tail_idx, -1, -1) if not _tool_name(new_tools[i]).startswith("mcp_")), None
            )
            for idx in dict.fromkeys(i for i in (last_builtin_idx, tail_idx) if i is not None):
                new_tools[idx] = {**new_tools[idx], "cache_control": cache_marker}
        return new_messages, new_tools

    @staticmethod
    def _apply_reasoning_effort(
        kwargs: dict[str, Any], spec: ProviderSpec | None, model_name: str, reasoning_effort: str | None
    ) -> None:
        """Send ``reasoning_effort`` the way the provider accepts it (nanobot openai_compat_provider.py:946-1060).

        The OpenAI vocabulary (none/minimal/low/medium/high/...) is remapped
        per provider, dropped for models that always reason, and turned into
        a provider's thinking switch in ``extra_body``. No effort set means
        nothing is sent, so each provider keeps its default.
        """
        semantic_effort: str | None = None
        if isinstance(reasoning_effort, str):
            semantic_effort = reasoning_effort.lower()
            if semantic_effort == "minimum":
                semantic_effort = "minimal"
        wire_effort = reasoning_effort
        slug = _model_slug(model_name)
        if slug in _KIMI_K3_MODELS and semantic_effort is not None:
            # K3 always reasons and accepts low/high/max.
            if semantic_effort in ("none", "minimal"):
                wire_effort = None
            else:
                if semantic_effort not in ("low", "high", "max"):
                    semantic_effort = "max"
                wire_effort = semantic_effort
        if spec and spec.name == "dashscope" and semantic_effort == "minimal":
            wire_effort = "minimum"  # DashScope's spelling; "minimal" is rejected
        elif spec and spec.name == "deepseek" and semantic_effort == "minimal":
            semantic_effort = wire_effort = "low"

        strip_effort = bool(spec and any(pat in model_name.lower() for pat in spec.implicit_reasoning_models))
        if not strip_effort and spec and spec.reasoning_effort_remap and isinstance(semantic_effort, str):
            mapped = dict(spec.reasoning_effort_remap).get(semantic_effort)
            if mapped is not None:
                wire_effort = mapped or None
                semantic_effort = mapped or "none"
        if strip_effort:
            wire_effort = None
        elif wire_effort and semantic_effort != "none":
            kwargs["reasoning_effort"] = wire_effort

        if reasoning_effort is not None:
            thinking_enabled = semantic_effort not in ("none", "minimal")
            for style in _thinking_styles_for(spec, model_name):
                if not thinking_enabled and slug in _KIMI_ALWAYS_THINKING_MODELS:
                    continue
                if builder := _THINKING_STYLE_MAP.get(style):
                    kwargs.setdefault("extra_body", {}).update(builder(thinking_enabled))
            gateway_style = spec.gateway_reasoning_style if spec else ""
            if (
                gateway_style
                and semantic_effort
                and _model_thinking_style(model_name)
                and (thinking_enabled or slug not in _KIMI_ALWAYS_THINKING_MODELS)
            ):
                if builder_g := _GATEWAY_REASONING_STYLE_MAP.get(gateway_style):
                    kwargs.setdefault("extra_body", {}).update(builder_g(semantic_effort))
            # Moonshot rejects reasoning_effort next to its native thinking switch.
            if slug in _KIMI_THINKING_MODELS:
                kwargs.pop("reasoning_effort", None)

        # DeepSeek-style thinking rejects assistant history without
        # reasoning_content; "" reads as "no thinking that turn".
        explicit_thinking = (
            reasoning_effort is not None
            and semantic_effort not in ("none", "minimal")
            and ((spec and spec.thinking_style) or _model_thinking_style(model_name))
        )
        implicit_deepseek_thinking = (
            spec is not None
            and spec.name == "deepseek"
            and semantic_effort not in ("none", "minimal", "minimum")
            and (slug == "deepseek-flash" or any(t in model_name.lower() for t in ("deepseek-v4", "deepseek-reasoner")))
        )
        if explicit_thinking or implicit_deepseek_thinking:
            for msg in kwargs["messages"]:
                if msg.get("role") == "assistant" and "reasoning_content" not in msg:
                    msg["reasoning_content"] = ""

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

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: ContentDeltaCallback | None = None,
        on_thinking_delta: ContentDeltaCallback | None = None,
        on_tool_call_delta: ToolCallDeltaCallback | None = None,
    ) -> LLMResponse:
        """Native streaming (nanobot openai_compat_provider.py:1839).

        Each chunk must arrive within the idle timeout
        (``NANOCLAW_STREAM_IDLE_TIMEOUT_S``); the whole call has no wall-clock
        limit. Errors come back as error responses, like :meth:`chat`.
        """
        idle_timeout_s = resolve_stream_idle_timeout_s()
        stream: Any = None
        try:
            kwargs = self.build_kwargs(messages, tools, model, max_tokens, temperature, tool_choice)
            kwargs["stream"] = True
            kwargs["timeout"] = idle_timeout_s
            kwargs["stream_options"] = {"include_usage": True}
            stream = await self._get_client().chat.completions.create(**kwargs)
            chunks: list[dict[str, Any]] = []
            completed = False
            stream_iter = stream.__aiter__()
            while True:
                try:
                    raw_chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=idle_timeout_s)
                except StopAsyncIteration:
                    break
                chunk = _as_mapping(raw_chunk)
                if chunk is None:
                    raise TypeError(f"Unsupported chat chunk type: {type(raw_chunk).__name__}")
                chunks.append(chunk)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = _as_mapping(choices[0]) or {}
                completed |= bool(choice.get("finish_reason"))
                delta = _as_mapping(choice.get("delta")) or {}
                await self._emit_stream_deltas(delta, on_content_delta, on_thinking_delta, on_tool_call_delta)
            if not completed:
                # Retryable: _extract_error_metadata classifies it as "connection".
                raise ConnectionError("Model stream ended before a finish reason was received")
            return self._parse_chunks(chunks)
        except asyncio.TimeoutError:
            return LLMResponse(
                content=f"Error calling LLM: stream stalled for more than {idle_timeout_s:g} seconds",
                finish_reason="error",
                error_kind="timeout",
            )
        except Exception as exc:
            return self._error_response(exc)
        finally:
            # Release the connection even when the stream was abandoned midway
            # (nanobot leaves this to garbage collection).
            close = getattr(stream, "close", None)
            if callable(close):
                with suppress(Exception):
                    result = close()
                    if inspect.isawaitable(result):
                        await result

    @staticmethod
    async def _emit_stream_deltas(
        delta: dict[str, Any],
        on_content_delta: ContentDeltaCallback | None,
        on_thinking_delta: ContentDeltaCallback | None,
        on_tool_call_delta: ToolCallDeltaCallback | None,
    ) -> None:
        """Pass one chunk's delta to the callbacks (nanobot openai_compat_provider.py:1906-1950)."""
        if on_content_delta and (text := _extract_text(delta.get("content"))):
            await on_content_delta(text)
        if on_thinking_delta and (reasoning := _delta_reasoning(delta)):
            await on_thinking_delta(reasoning)
        if not on_tool_call_delta:
            return
        for idx, tool_delta in enumerate(delta.get("tool_calls") or []):
            tool_delta = _as_mapping(tool_delta) or {}
            fn = _as_mapping(tool_delta.get("function"))
            index = tool_delta.get("index")
            await on_tool_call_delta(
                {
                    "index": index if index is not None else idx,
                    "call_id": str(tool_delta.get("id") or ""),
                    "name": str(fn.get("name") or "") if fn is not None else "",
                    "arguments_delta": str(fn.get("arguments") or "") if fn is not None else "",
                }
            )
        function_call = _as_mapping(delta.get("function_call"))
        if function_call:
            await on_tool_call_delta(
                {
                    "index": 0,
                    "call_id": "",
                    "name": str(function_call.get("name") or ""),
                    "arguments_delta": str(function_call.get("arguments") or ""),
                }
            )

    @classmethod
    def _parse_chunks(cls, chunks: list[Any]) -> LLMResponse:
        """Assemble streamed chunks into the response :meth:`parse` would give.

        Chunks may be dicts, SDK chunk objects or bare strings (text). Tool-call
        argument fragments are joined per ``index``; the legacy
        ``delta.function_call`` form is call 0. (nanobot openai_compat_provider.py:1556)
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_buffers: dict[int, dict[str, str]] = {}
        finish_reason = "stop"
        usage: LLMUsage | None = None

        def buffer(index: int) -> dict[str, str]:
            return tool_buffers.setdefault(index, {"id": "", "name": "", "arguments": ""})

        for raw_chunk in chunks:
            if isinstance(raw_chunk, str):
                content_parts.append(raw_chunk)
                continue
            chunk = _as_mapping(raw_chunk)
            if chunk is None:
                raise TypeError(f"Unsupported chat chunk type: {type(raw_chunk).__name__}")
            choices = chunk.get("choices") or []
            if not choices:
                # The include_usage chunk, or a gateway's top-level text.
                usage = cls._parse_usage(chunk.get("usage")) or usage
                if text := _extract_text(chunk.get("content") or chunk.get("output_text")):
                    content_parts.append(text)
                continue
            choice = _as_mapping(choices[0]) or {}
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
            delta = _as_mapping(choice.get("delta")) or {}
            if text := _extract_text(delta.get("content")):
                content_parts.append(text)
            if reasoning := _delta_reasoning(delta):
                reasoning_parts.append(reasoning)
            for idx, raw_tc in enumerate(delta.get("tool_calls") or []):
                tc = _as_mapping(raw_tc) or {}
                buf = buffer(tc["index"] if tc.get("index") is not None else idx)
                if tc.get("id"):
                    buf["id"] = str(tc["id"])
                fn = _as_mapping(tc.get("function"))
                if fn is not None:
                    if fn.get("name"):
                        buf["name"] = str(fn["name"])
                    if fn.get("arguments"):
                        buf["arguments"] += str(fn["arguments"])
            function_call = _as_mapping(delta.get("function_call"))
            if function_call:
                buf = buffer(0)
                if function_call.get("name"):
                    buf["name"] = str(function_call["name"])
                if function_call.get("arguments"):
                    buf["arguments"] += str(function_call["arguments"])
            usage = cls._parse_usage(chunk.get("usage")) or usage

        # Some providers (Zhipu/GLM) stream one ID for parallel calls; each
        # tool result must answer a unique ID (nanobot :1660).
        seen_ids: set[str] = set()
        tool_calls: list[ToolCallRequest] = []
        for buf in tool_buffers.values():
            call_id = buf["id"]
            if not call_id or call_id in seen_ids:
                call_id = _short_tool_id()
            seen_ids.add(call_id)
            tool_calls.append(
                ToolCallRequest(id=call_id, name=buf["name"], arguments=parse_tool_arguments(buf["arguments"]))
            )

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content="".join(reasoning_parts) or None,
        )

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
        """Usage with cache counts, normalized across providers.

        (nanobot openai_compat_provider.py:1381 ``_extract_usage``)
        """
        if not isinstance(raw_usage, dict) or not raw_usage:
            return None
        cache_read: int | None = None
        for path in (
            ("prompt_tokens_details", "cached_tokens"),  # OpenAI, Zhipu, MiniMax, Qwen, Mistral, xAI
            ("cached_tokens",),  # StepFun, Moonshot
            ("prompt_cache_hit_tokens",),  # DeepSeek, SiliconFlow
        ):
            cache_read = _get_nested_int(raw_usage, path)
            if cache_read is not None:
                break
        return LLMUsage.reported(
            input_tokens=int(raw_usage.get("prompt_tokens") or 0),
            output_tokens=int(raw_usage.get("completion_tokens") or 0),
            total_tokens=_get_nested_int(raw_usage, ("total_tokens",)),
            cache_read_tokens=cache_read,
            cache_write_tokens=_get_nested_int(raw_usage, ("prompt_tokens_details", "cache_write_tokens")),
        )

    @classmethod
    def _error_response(cls, exc: Exception) -> LLMResponse:
        """An error response carrying what the retry policy needs.

        (nanobot openai_compat_provider.py:1732 ``_handle_error``; NanoClaw keeps
        its own shorter message text.)
        """
        if isinstance(exc, APIStatusError):
            # The SDK's exc.message embeds the whole JSON body; prefer the
            # API's own human-readable message when it sent one.
            body = exc.body
            message = body.get("message") if isinstance(body, dict) else None
            detail = f"HTTP {exc.status_code}: {message or exc.message}"
        else:
            detail = str(exc).strip() or type(exc).__name__
        content = f"Error calling LLM: {detail}"
        retry_after = cls._extract_retry_after_from_headers(getattr(getattr(exc, "response", None), "headers", None))
        if retry_after is None:
            retry_after = cls._extract_retry_after(content)
        return LLMResponse(
            content=content,
            finish_reason="error",
            retry_after=retry_after,
            **cls._extract_error_metadata(exc),
        )

    @classmethod
    def _extract_error_metadata(cls, exc: Exception) -> dict[str, Any]:
        """Status, type/code, Retry-After and x-should-retry from an SDK error.

        (nanobot openai_compat_provider.py:1684)
        """
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        payload = getattr(exc, "body", None) or getattr(exc, "doc", None) or getattr(response, "text", None)
        error_type, error_code = cls._extract_error_type_code(payload)

        status_code = getattr(exc, "status_code", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)

        should_retry: bool | None = None
        if headers is not None:
            raw = headers.get("x-should-retry")
            if isinstance(raw, str):
                should_retry = {"true": True, "false": False}.get(raw.strip().lower())

        error_name = type(exc).__name__.lower()
        error_kind = "timeout" if "timeout" in error_name else "connection" if "connection" in error_name else None
        return {
            "error_status_code": int(status_code) if status_code is not None else None,
            "error_kind": error_kind,
            "error_type": error_type,
            "error_code": error_code,
            "error_retry_after_s": cls._extract_retry_after_from_headers(headers),
            "error_should_retry": should_retry,
        }


def _get_nested_int(obj: Any, path: tuple[str, ...]) -> int | None:
    """A count at ``path`` in nested dicts, keeping an explicit 0; ``None`` if absent.

    (nanobot openai_compat_provider.py:1439 ``_get_nested_int``)
    """
    current = obj
    for segment in path:
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    if current is None or isinstance(current, bool):
        return None
    try:
        return int(current)
    except (TypeError, ValueError):
        return None
