"""Anthropic provider: Claude over the official ``anthropic`` SDK.

Mirrors nanobot/providers/anthropic_provider.py. NanoClaw keeps every
conversation in the OpenAI chat format; this adapter translates it to the
Messages API on the way out (system split, ``tool_use``/``tool_result``
blocks, thinking blocks replayed first, consecutive same-role turns merged,
prompt-cache markers, extended thinking) and the answer back into an
:class:`LLMResponse` (text, tool calls, ``thinking_blocks``, usage with cache
reads/writes, finish reason). ``chat_stream`` streams natively.

Not ported: nanobot's ``prepare_message_images`` (reads local image files into
base64 before the request) - NanoClaw sends no images; ``image_url`` blocks
that do reach the adapter are still converted to Anthropic image blocks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import string
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any, cast

from loguru import logger

from nanoclaw.providers.base import (
    ContentDeltaCallback,
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ToolCallDeltaCallback,
    ToolCallRequest,
    resolve_stream_idle_timeout_s,
    tool_arguments_object_for_replay,
)

_ALNUM = string.ascii_letters + string.digits


def _gen_tool_id() -> str:
    return "toolu_" + "".join(secrets.choice(_ALNUM) for _ in range(22))


_VALID_TOOL_ID = re.compile(r"^[a-zA-Z0-9_-]+$")

_CLAUDE_MODEL_VERSION = re.compile(r"claude-(?P<family>[a-z]+)-(?P<major>\d+)(?:-(?P<minor>\d{1,2})(?=-|$))?")
# Models that only accept adaptive thinking (no budget_tokens, no temperature).
_ADAPTIVE_ONLY_MIN_VERSIONS = {
    "opus": (4, 7),
    "sonnet": (5, 0),
    "fable": (5, 0),
    "mythos": (5, 0),
}
# Models that think by default, so reasoning_effort="none" must say so.
_THINKING_DISABLE_MIN_VERSIONS = {
    "opus": (5, 0),
    "sonnet": (5, 0),
}
_SAMPLING_DEPRECATED_MODELS = {"claude-mythos-preview"}

# Anthropic stop_reason -> NanoClaw finish_reason; others pass through.
_STOP_REASONS = {"tool_use": "tool_calls", "end_turn": "stop", "max_tokens": "length"}

_UNSET: Any = object()


def _model_version_at_least(model_name: str, minimum_versions: dict[str, tuple[int, int]]) -> bool:
    """Whether a ``claude-<family>-<major>[-<minor>]`` name is at least the family's minimum.

    A dated suffix (``claude-opus-4-20250514``) is not a minor version.
    """
    match = _CLAUDE_MODEL_VERSION.search(model_name.lower())
    if match is None:
        return False
    minimum = minimum_versions.get(match.group("family"))
    if minimum is None:
        return False
    version = (int(match.group("major")), int(match.group("minor") or 0))
    return version >= minimum


def _sanitize_tool_id(tid: str) -> str:
    """Make a tool id match Anthropic's ``^[a-zA-Z0-9_-]+$``.

    Ids from other providers or restored sessions can contain pipes, dots or
    other characters the API rejects with a 400. A short hash of the original
    keeps two different ids from collapsing into one.
    """
    if not tid or _VALID_TOOL_ID.match(tid):
        return tid
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", tid)[:48].strip("_") or "toolu"
    digest = hashlib.sha1(tid.encode()).hexdigest()[:8]
    return f"{safe_prefix}_{digest}"


def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop ``_meta``, replace empty content and empty text blocks (nanobot base.py:602).

    nanobot also scrubs lone UTF-16 surrogates here; NanoClaw's terminal
    already removes them from input.
    """
    result: list[dict[str, Any]] = []
    for raw_msg in messages:
        msg = {key: value for key, value in raw_msg.items() if key != "_meta"}
        content = msg.get("content")
        if isinstance(content, str) and not content:
            msg["content"] = None if (msg.get("role") == "assistant" and msg.get("tool_calls")) else "(empty)"
        elif isinstance(content, list):
            new_items: list[Any] = []
            changed = False
            for item in content:
                if isinstance(item, dict) and item.get("type") in ("text", "input_text", "output_text") and not item.get("text"):
                    changed = True
                    continue
                if isinstance(item, dict) and "_meta" in item:
                    new_items.append({k: v for k, v in item.items() if k != "_meta"})
                    changed = True
                else:
                    new_items.append(item)
            if changed:
                if new_items:
                    msg["content"] = new_items
                elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                    msg["content"] = None
                else:
                    msg["content"] = "(empty)"
        elif isinstance(content, dict):
            msg["content"] = [content]
        result.append(msg)
    return result


class AnthropicProvider(LLMProvider):
    """Claude models over the native Anthropic SDK.

    Message conversion (OpenAI -> Messages API), prompt caching, extended
    thinking, tool calls and streaming. The SDK's own retries are off
    (``max_retries=0``): ``chat_with_retry`` is the one retry policy.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "claude-sonnet-4-6",
        extra_headers: dict[str, str] | None = None,
        *,
        provider_name: str = "anthropic",
        extra_body: dict[str, Any] | None = None,
        extra_query: dict[str, str] | None = None,
        proxy: str | None = None,
        spec: Any = None,
        reasoning_effort: str | None = None,
        client: Any = None,
    ) -> None:
        """
        Args:
            api_key: Anthropic key (the SDK falls back to ``ANTHROPIC_API_KEY``).
            api_base: Another Anthropic-compatible endpoint; a trailing ``/v1``
                is dropped because the SDK adds it.
            default_model: Used when a call names no model; an ``anthropic/``
                prefix is stripped.
            extra_headers: Sent with every request.
            provider_name: For logs and the registry.
            extra_body / extra_query: Merged into every request / its query string.
            proxy: HTTP(S) proxy URL for the SDK's httpx client.
            spec: Optional registry spec; if it has ``request_model_name(model)``
                that maps the model name (nanobot passes it for custom
                Anthropic-compatible providers).
            reasoning_effort: The default thinking setting when a call passes
                none: ``None``/"" (off), ``"none"``, ``"adaptive"``, ``"low"``,
                ``"medium"``, ``"high"`` (or newer efforts such as ``"max"``).
                nanobot reads it from ``provider.generation``.
            client: A ready ``AsyncAnthropic`` (tests); skips building one.
        """
        super().__init__(api_key, api_base, default_model)
        self.provider_name = provider_name
        self.extra_headers = extra_headers or {}
        self._extra_body = dict(extra_body or {})
        self._spec = spec
        self.reasoning_effort = reasoning_effort

        if client is not None:
            self._client = client
            return
        from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

        client_kw: dict[str, Any] = {}
        if api_key:
            client_kw["api_key"] = api_key
        if api_base:
            client_kw["base_url"] = self._normalize_base_url(api_base)
        if extra_headers:
            client_kw["default_headers"] = extra_headers
        if extra_query:
            client_kw["default_query"] = extra_query
        if proxy:
            client_kw["http_client"] = DefaultAsyncHttpxClient(proxy=proxy, trust_env=False)
        # Retries live in LLMProvider._run_with_retry, so they don't multiply.
        client_kw["max_retries"] = 0
        self._client = AsyncAnthropic(**client_kw)

    @staticmethod
    def _normalize_base_url(api_base: str) -> str:
        """The SDK appends ``/v1`` to request paths itself."""
        normalized = api_base.rstrip("/")
        if normalized.endswith("/v1"):
            return normalized[: -len("/v1")]
        return normalized

    # -- errors ----------------------------------------------------------------

    @classmethod
    def _handle_error(cls, e: Exception) -> LLMResponse:
        """An error response with what the retry policy reads (nanobot :137).

        Status code, the API's error type/code, Retry-After / retry-after-ms
        headers, ``x-should-retry``, and a timeout/connection kind from the
        exception's class name.
        """
        response = getattr(e, "response", None)
        headers = getattr(response, "headers", None)
        payload = getattr(e, "body", None) or getattr(e, "doc", None) or getattr(response, "text", None)
        if payload is None and response is not None:
            response_json = getattr(response, "json", None)
            if callable(response_json):
                try:
                    payload = response_json()
                except Exception:
                    payload = None
        payload_text = payload if isinstance(payload, str) else str(payload) if payload is not None else ""
        msg = f"Error: {payload_text.strip()[:500]}" if payload_text.strip() else f"Error calling LLM: {e}"
        retry_after = cls._extract_retry_after_from_headers(headers)
        if retry_after is None:
            retry_after = cls._extract_retry_after(msg)

        status_code = getattr(e, "status_code", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)

        should_retry: bool | None = None
        if headers is not None:
            raw = headers.get("x-should-retry")
            if isinstance(raw, str):
                should_retry = {"true": True, "false": False}.get(raw.strip().lower())

        error_name = e.__class__.__name__.lower()
        error_kind = "timeout" if "timeout" in error_name else "connection" if "connection" in error_name else None
        error_type, error_code = cls._extract_error_type_code(payload)

        return LLMResponse(
            content=msg,
            finish_reason="error",
            retry_after=retry_after,
            error_status_code=int(status_code) if status_code is not None else None,
            error_kind=error_kind,
            error_type=error_type,
            error_code=error_code,
            error_retry_after_s=retry_after,
            error_should_retry=should_retry,
        )

    @staticmethod
    def _strip_prefix(model: str) -> str:
        return model[len("anthropic/") :] if model.startswith("anthropic/") else model

    def _request_model_name(self, model: str) -> str:
        mapper = getattr(self._spec, "request_model_name", None)
        if callable(mapper):
            return str(mapper(model))
        return self._strip_prefix(model)

    # -- messages: OpenAI chat format -> Messages API ----------------------------

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | list[dict[str, Any]], list[dict[str, Any]]]:
        """``(system, anthropic_messages)``.

        Tool results become ``tool_result`` blocks in a user turn; assistant
        tool calls become ``tool_use`` blocks. Tool ids are sanitized and made
        unique, and each result follows its call's (possibly renamed) id.
        """
        system: str | list[dict[str, Any]] = ""
        raw: list[dict[str, Any]] = []
        seen_tool_ids: set[str] = set()
        pending_tool_ids: dict[str, deque[str]] = {}

        def unique_tool_id(value: Any) -> str:
            raw_key = str(value) if value else ""
            mapped_id = _sanitize_tool_id(raw_key) if raw_key else _gen_tool_id()
            if mapped_id and mapped_id not in seen_tool_ids:
                seen_tool_ids.add(mapped_id)
                if raw_key:
                    pending_tool_ids.setdefault(raw_key, deque()).append(mapped_id)
                return mapped_id
            seed = mapped_id or _gen_tool_id()
            suffix = 2
            while True:
                candidate = f"{seed}__dedupe_{suffix}"
                if candidate not in seen_tool_ids:
                    seen_tool_ids.add(candidate)
                    if raw_key:
                        pending_tool_ids.setdefault(raw_key, deque()).append(candidate)
                    return candidate
                suffix += 1

        def map_tool_result_id(value: Any) -> str:
            if not value:
                return _sanitize_tool_id(value or "")
            raw_id = str(value)
            queue = pending_tool_ids.get(raw_id)
            if queue:
                mapped_id = queue.popleft()
                if not queue:
                    pending_tool_ids.pop(raw_id, None)
                return mapped_id
            return _sanitize_tool_id(raw_id)

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")
            if role == "system":
                system = (
                    cast(list[dict[str, Any]], content)
                    if isinstance(content, list)
                    else content
                    if isinstance(content, str)
                    else str(content or "")
                )
            elif role == "tool":
                block = self._tool_result_block(msg, map_tool_result_id=map_tool_result_id)
                if raw and raw[-1]["role"] == "user":
                    prev_c = raw[-1]["content"]
                    if isinstance(prev_c, list):
                        prev_c.append(block)
                    else:
                        raw[-1]["content"] = [{"type": "text", "text": prev_c or ""}, block]
                else:
                    raw.append({"role": "user", "content": [block]})
            elif role == "assistant":
                raw.append({"role": "assistant", "content": self._assistant_blocks(msg, map_tool_id=unique_tool_id)})
            elif role == "user":
                raw.append({"role": "user", "content": self._convert_user_content(content)})

        return system, self._merge_consecutive(raw)

    @staticmethod
    def _tool_result_block(
        msg: dict[str, Any],
        *,
        map_tool_result_id: Callable[[Any], str] | None = None,
    ) -> dict[str, Any]:
        content = msg.get("content")
        tool_call_id = msg.get("tool_call_id", "")
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": (
                map_tool_result_id(tool_call_id) if map_tool_result_id is not None else _sanitize_tool_id(tool_call_id)
            ),
        }
        if isinstance(content, list):
            block["content"] = AnthropicProvider._convert_user_content(content)
        elif isinstance(content, str):
            block["content"] = content
        else:
            block["content"] = str(content) if content else ""
        return block

    @staticmethod
    def _assistant_blocks(
        msg: dict[str, Any],
        *,
        map_tool_id: Callable[[Any], str] | None = None,
    ) -> list[dict[str, Any]]:
        """Thinking blocks first (the API requires it), then text, then ``tool_use``."""
        blocks: list[dict[str, Any]] = []
        content = msg.get("content")

        for tb in cast(Iterable[object], msg.get("thinking_blocks") or []):
            if isinstance(tb, dict) and tb.get("type") == "thinking":
                blocks.append(
                    {"type": "thinking", "thinking": tb.get("thinking", ""), "signature": tb.get("signature", "")}
                )

        if isinstance(content, str) and content:
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if not item.get("type"):
                        # Every block needs a "type"; a bare dict becomes text.
                        blocks.append({"type": "text", "text": AnthropicProvider._stringify_typeless_block(item)})
                    else:
                        blocks.append(item)
                else:
                    blocks.append({"type": "text", "text": str(item)})

        for tc in cast(Iterable[object], msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            func = cast(dict[str, Any], tc.get("function", {}))
            raw_id = tc.get("id") or _gen_tool_id()
            blocks.append(
                {
                    "type": "tool_use",
                    "id": map_tool_id(raw_id) if map_tool_id is not None else _sanitize_tool_id(cast(str, raw_id)),
                    "name": func.get("name", ""),
                    "input": tool_arguments_object_for_replay(func.get("arguments", "{}")),
                }
            )

        return blocks or [{"type": "text", "text": ""}]

    @staticmethod
    def _convert_user_content(content: Any) -> Any:
        """User (or tool-result) content: ``image_url`` -> image blocks, bare dicts -> text."""
        if isinstance(content, str) or content is None:
            return content or "(empty)"
        if not isinstance(content, list):
            return str(content)
        result: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                result.append({"type": "text", "text": str(item)})
                continue
            if item.get("type") == "image_url":
                converted = AnthropicProvider._convert_image_block(item)
                if converted:
                    result.append(converted)
                continue
            if not item.get("type"):
                # Avoids the API's "content.0.type: Field required" rejection.
                result.append({"type": "text", "text": AnthropicProvider._stringify_typeless_block(item)})
                continue
            result.append(item)
        return result or "(empty)"

    @staticmethod
    def _stringify_typeless_block(block: dict[str, Any]) -> str:
        return json.dumps(block, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _convert_image_block(block: dict[str, Any]) -> dict[str, Any] | None:
        """An OpenAI ``image_url`` block as an Anthropic image block (data URL or URL)."""
        image_url = cast(dict[str, Any], block.get("image_url") or {})
        url = cast(str, image_url.get("url", ""))
        if not url:
            return None
        m = re.match(r"data:(image/\w+);base64,(.+)", url, re.DOTALL)
        if m:
            return {"type": "image", "source": {"type": "base64", "media_type": m.group(1), "data": m.group(2)}}
        return {"type": "image", "source": {"type": "url", "url": url}}

    @staticmethod
    def _has_tool_use(msg: dict[str, Any]) -> bool:
        """True if the turn carries a ``tool_use`` block (invalid in a user turn)."""
        content = msg.get("content")
        if not isinstance(content, list):
            return False
        return any(isinstance(block, dict) and block.get("type") == "tool_use" for block in content)

    @staticmethod
    def _merge_consecutive(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Make the sequence valid for ``/messages`` (nanobot :442).

        1. Consecutive same-role turns are merged into one.
        2. Trailing assistant turns are dropped (no prefill); if that empties
           the list, the last one is sent as a user turn instead - unless it
           carries ``tool_use``.
        3. A leading assistant turn gets a synthetic user opener - unless it
           carries ``tool_use`` (an opener would orphan its results).
        """
        merged: list[dict[str, Any]] = []
        for msg in msgs:
            if merged and merged[-1]["role"] == msg["role"]:
                prev_c = merged[-1]["content"]
                cur_c = msg["content"]
                if isinstance(prev_c, str):
                    prev_c = [{"type": "text", "text": prev_c}]
                if isinstance(cur_c, str):
                    cur_c = [{"type": "text", "text": cur_c}]
                if isinstance(cur_c, list):
                    prev_c.extend(cur_c)
                merged[-1]["content"] = prev_c
            else:
                merged.append(msg)

        last_popped: dict[str, Any] | None = None
        while merged and merged[-1].get("role") == "assistant":
            last_popped = merged.pop()
        if not merged and last_popped is not None and not AnthropicProvider._has_tool_use(last_popped):
            merged.append({"role": "user", "content": last_popped.get("content")})

        if merged and merged[0].get("role") == "assistant" and not AnthropicProvider._has_tool_use(merged[0]):
            merged.insert(0, {"role": "user", "content": "(conversation continued)"})
        return merged

    # -- tools -------------------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """OpenAI function tools as Anthropic tools (``input_schema``)."""
        if not tools:
            return None
        result: list[dict[str, Any]] = []
        for tool in tools:
            func = tool.get("function", tool)
            entry: dict[str, Any] = {
                "name": func.get("name", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            }
            if desc := func.get("description"):
                entry["description"] = desc
            if "cache_control" in tool:
                entry["cache_control"] = tool["cache_control"]
            result.append(entry)
        return result

    @staticmethod
    def _convert_tool_choice(
        tool_choice: str | dict[str, Any] | None,
        thinking_enabled: bool = False,
    ) -> dict[str, Any] | None:
        """``auto``/``required``/``none``/a named tool -> Anthropic; thinking forces ``auto``."""
        if thinking_enabled:
            # The API rejects a forced tool choice while thinking.
            return {"type": "auto"}
        if tool_choice is None or tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return None
        if isinstance(tool_choice, dict):
            name = tool_choice.get("function", {}).get("name")
            if name:
                return {"type": "tool", "name": name}
        return {"type": "auto"}

    # -- prompt caching ------------------------------------------------------------

    @staticmethod
    def _tool_name(tool: dict[str, Any]) -> str:
        """A tool's name in either OpenAI or Anthropic shape (nanobot base.py:664)."""
        name = tool.get("name")
        if isinstance(name, str):
            return name
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return fn["name"]
        return ""

    @classmethod
    def _tool_cache_marker_indices(cls, tools: list[dict[str, Any]]) -> list[int]:
        """The last built-in tool (before the ``mcp_*`` ones) and the last tool (nanobot base.py:678)."""
        if not tools:
            return []
        tail_idx = len(tools) - 1
        last_builtin_idx: int | None = None
        for i in range(tail_idx, -1, -1):
            if not cls._tool_name(tools[i]).startswith("mcp_"):
                last_builtin_idx = i
                break
        ordered_unique: list[int] = []
        for idx in (last_builtin_idx, tail_idx):
            if idx is not None and idx not in ordered_unique:
                ordered_unique.append(idx)
        return ordered_unique

    @classmethod
    def _apply_cache_control(
        cls,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str | list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Ephemeral cache markers: end of system, second-to-last message, tool boundaries."""
        marker = {"type": "ephemeral"}

        if isinstance(system, str) and system:
            system = [{"type": "text", "text": system, "cache_control": marker}]
        elif isinstance(system, list) and system:
            system = list(system)
            system[-1] = {**system[-1], "cache_control": marker}

        new_msgs = list(messages)
        if len(new_msgs) >= 3:
            m = new_msgs[-2]
            c = m.get("content")
            if isinstance(c, str):
                new_msgs[-2] = {**m, "content": [{"type": "text", "text": c, "cache_control": marker}]}
            elif isinstance(c, list) and c:
                nc = list(c)
                nc[-1] = {**nc[-1], "cache_control": marker}
                new_msgs[-2] = {**m, "content": nc}

        new_tools = tools
        if tools:
            new_tools = list(tools)
            for idx in cls._tool_cache_marker_indices(new_tools):
                new_tools[idx] = {**new_tools[idx], "cache_control": marker}

        return system, new_msgs, new_tools

    # -- request ---------------------------------------------------------------------

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
        supports_caching: bool = True,
    ) -> dict[str, Any]:
        """The keyword arguments for ``messages.create`` / ``messages.stream`` (nanobot :591).

        Thinking, from ``reasoning_effort``:
        - unset/"": off (temperature as given);
        - "none": off, or ``{"type": "disabled"}`` for models that think by default;
        - "adaptive": ``{"type": "adaptive"}``, temperature 1.0;
        - any other effort on adaptive-only models: adaptive + ``output_config.effort``;
        - otherwise a token budget (low 1024, medium 4096, high max(8192, max_tokens)),
          ``max_tokens`` raised to budget + 4096, temperature 1.0.
        Adaptive-only models (and Mythos Preview) never get a temperature.
        """
        model_name = self._request_model_name(model or self.default_model)
        system, anthropic_msgs = self._convert_messages(_sanitize_empty_content(messages))
        anthropic_tools = self._convert_tools(tools)

        if supports_caching:
            system, anthropic_msgs, anthropic_tools = self._apply_cache_control(system, anthropic_msgs, anthropic_tools)

        max_tokens = max(1, max_tokens)
        effort = reasoning_effort.lower() if reasoning_effort else None
        thinking_enabled = effort not in (None, "", "none")
        adaptive_only = _model_version_at_least(model_name, _ADAPTIVE_ONLY_MIN_VERSIONS)
        # Mythos Preview rejects sampling parameters but still accepts budgets.
        omit_temperature = adaptive_only or model_name.lower() in _SAMPLING_DEPRECATED_MODELS

        kwargs: dict[str, Any] = {"model": model_name, "messages": anthropic_msgs, "max_tokens": max_tokens}
        if system:
            kwargs["system"] = system

        if effort == "none" and _model_version_at_least(model_name, _THINKING_DISABLE_MIN_VERSIONS):
            kwargs["thinking"] = {"type": "disabled"}
        elif effort == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
            if not omit_temperature:
                kwargs["temperature"] = 1.0
        elif thinking_enabled and adaptive_only:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": effort}
        elif thinking_enabled:
            budget_map = {"low": 1024, "medium": 4096, "high": max(8192, max_tokens)}
            budget = budget_map.get(cast(str, effort), 4096)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["max_tokens"] = max(max_tokens, budget + 4096)
            if not omit_temperature:
                kwargs["temperature"] = 1.0
        elif not omit_temperature:
            kwargs["temperature"] = temperature

        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
            tc = self._convert_tool_choice(tool_choice, thinking_enabled)
            if tc:
                kwargs["tool_choice"] = tc

        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        return kwargs

    # -- response ----------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        """A Messages API response (or the stream's final message) as an LLMResponse."""
        content_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        thinking_blocks: list[dict[str, Any]] = []
        seen_tool_ids: set[str] = set()

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_id = str(block.id or _gen_tool_id())
                if tool_id in seen_tool_ids:
                    original_id = tool_id
                    while tool_id in seen_tool_ids:
                        tool_id = _gen_tool_id()
                    logger.warning("remapping duplicate tool_use id from response: {} -> {}", original_id, tool_id)
                seen_tool_ids.add(tool_id)
                tool_calls.append(ToolCallRequest(id=tool_id, name=block.name, arguments=block.input))
            elif block.type == "thinking":
                thinking_blocks.append(
                    {"type": "thinking", "thinking": block.thinking, "signature": getattr(block, "signature", "")}
                )

        finish_reason = _STOP_REASONS.get(response.stop_reason or "", response.stop_reason or "stop")

        usage: LLMUsage | None = None
        if response.usage:
            cache_write_raw = getattr(response.usage, "cache_creation_input_tokens", None)
            cache_read_raw = getattr(response.usage, "cache_read_input_tokens", None)
            cache_write = int(cache_write_raw) if cache_write_raw is not None else None
            cache_read = int(cache_read_raw) if cache_read_raw is not None else None
            # Anthropic's input_tokens excludes cache reads and writes.
            logical_input = int(response.usage.input_tokens) + (cache_write or 0) + (cache_read or 0)
            usage = LLMUsage.reported(
                input_tokens=logical_input,
                output_tokens=int(response.usage.output_tokens),
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            )

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            thinking_blocks=thinking_blocks or None,
        )

    # -- public API ----------------------------------------------------------------------

    @staticmethod
    def _is_streaming_required_error(e: Exception) -> bool:
        """The SDK refuses long non-stream requests with ``ValueError("Streaming is required ...")``."""
        return isinstance(e, ValueError) and "streaming is required" in str(e).lower()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
        reasoning_effort: str | None = _UNSET,
    ) -> LLMResponse:
        """One request. ``reasoning_effort`` left out means ``self.reasoning_effort``."""
        if reasoning_effort is _UNSET:
            reasoning_effort = self.reasoning_effort
        kwargs = self._build_kwargs(messages, tools, model, max_tokens, temperature, reasoning_effort, tool_choice)
        try:
            response = await self._client.messages.create(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            if self._is_streaming_required_error(e):
                # max_tokens (plus a thinking budget) could exceed the API's
                # 10-minute non-stream limit (nanobot #2709): stream instead,
                # without delta callbacks.
                return await self.chat_stream(
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    tool_choice=tool_choice,
                )
            return self._handle_error(e)

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
        reasoning_effort: str | None = _UNSET,
    ) -> LLMResponse:
        """Native streaming (nanobot :788).

        Text deltas go to ``on_content_delta``, thinking deltas to
        ``on_thinking_delta``, and each ``tool_use`` start and JSON fragment to
        ``on_tool_call_delta`` as ``{"index", "call_id", "name", "arguments_delta"}``.
        Every SSE event (not only text) must arrive within the idle timeout
        (``NANOCLAW_STREAM_IDLE_TIMEOUT_S``); there is no total limit. A stream
        that ends without a stop reason is a (retryable) connection error.
        """
        if reasoning_effort is _UNSET:
            reasoning_effort = self.reasoning_effort
        kwargs = self._build_kwargs(messages, tools, model, max_tokens, temperature, reasoning_effort, tool_choice)
        idle_timeout_s = resolve_stream_idle_timeout_s()
        kwargs["timeout"] = idle_timeout_s
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                # The SDK accumulates the final message while we iterate, so
                # get_final_message() below returns at once.
                tool_blocks: dict[int, dict[str, str]] = {}
                while True:
                    try:
                        chunk = await asyncio.wait_for(stream.__anext__(), timeout=idle_timeout_s)
                    except StopAsyncIteration:
                        break
                    await self._emit_stream_event(
                        chunk, tool_blocks, on_content_delta, on_thinking_delta, on_tool_call_delta
                    )
                response = await stream.get_final_message()
                if not response.stop_reason:
                    raise ConnectionError("Model stream ended before a stop reason was received")
            return self._parse_response(response)
        except asyncio.TimeoutError:
            return LLMResponse(
                content=f"Error calling LLM: stream stalled for more than {idle_timeout_s:g} seconds",
                finish_reason="error",
                error_kind="timeout",
            )
        except Exception as e:
            return self._handle_error(e)

    @staticmethod
    async def _emit_stream_event(
        chunk: Any,
        tool_blocks: dict[int, dict[str, str]],
        on_content_delta: ContentDeltaCallback | None,
        on_thinking_delta: ContentDeltaCallback | None,
        on_tool_call_delta: ToolCallDeltaCallback | None,
    ) -> None:
        """Turn one SSE event into NanoClaw stream deltas."""
        if chunk.type == "content_block_start":
            block = getattr(chunk, "content_block", None)
            if getattr(block, "type", None) == "tool_use":
                index = int(getattr(chunk, "index", 0) or 0)
                state = {
                    "call_id": str(getattr(block, "id", "") or ""),
                    "name": str(getattr(block, "name", "") or ""),
                }
                tool_blocks[index] = state
                if on_tool_call_delta:
                    await on_tool_call_delta({"index": index, **state, "arguments_delta": ""})
            return
        if chunk.type != "content_block_delta":
            return
        delta_type = getattr(chunk.delta, "type", None)
        if delta_type == "thinking_delta":
            piece = getattr(chunk.delta, "thinking", None) or ""
            if piece and on_thinking_delta:
                await on_thinking_delta(piece)
        elif delta_type == "text_delta":
            text = getattr(chunk.delta, "text", None) or ""
            if text and on_content_delta:
                await on_content_delta(text)
        elif delta_type == "input_json_delta":
            partial = getattr(chunk.delta, "partial_json", None) or ""
            if partial and on_tool_call_delta:
                index = int(getattr(chunk, "index", 0) or 0)
                state = tool_blocks.get(index, {})
                await on_tool_call_delta(
                    {
                        "index": index,
                        "call_id": state.get("call_id", ""),
                        "name": state.get("name", ""),
                        "arguments_delta": partial,
                    }
                )

    def get_default_model(self) -> str:
        return self.default_model

    async def aclose(self) -> None:
        """Close the SDK client's connection pool."""
        await self._client.close()
