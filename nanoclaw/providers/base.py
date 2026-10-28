"""Provider-neutral types and the LLMProvider interface.

Mirrors nanobot/providers/base.py. Messages everywhere in NanoClaw use the
OpenAI chat format ({"role": ..., "content": ..., "tool_calls": [...]});
each provider translates to and from its own wire format.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from loguru import logger

# Longest silence allowed between two streamed chunks (nanobot base.py:38-40).
STREAM_IDLE_TIMEOUT_ENV = "NANOCLAW_STREAM_IDLE_TIMEOUT_S"
DEFAULT_STREAM_IDLE_TIMEOUT_S = 90.0
MAX_STREAM_IDLE_TIMEOUT_S = 3600.0

# Added to a server's Retry-After before retrying (nanobot base.py:41).
RETRY_AFTER_BUFFER = 1

# Called with a short status text while a retry waits or when retrying stops
# (nanobot ``RetryEventCallback``).
RetryWaitCallback = Callable[[str], Awaitable[None]]

# Streaming callbacks (nanobot base.py:1085 ``chat_stream`` parameters).
# A tool-call delta is ``{"index", "call_id", "name", "arguments_delta"}``.
ContentDeltaCallback = Callable[[str], Awaitable[None]]
ToolCallDeltaCallback = Callable[[dict[str, Any]], Awaitable[None]]
StreamRecoverCallback = Callable[[], Awaitable[None]]

# finish_reason values after which tool calls may run. Anything else
# ("length", "content_filter", "error", ...) means the turn was cut short.
_TOOL_CAPABLE_FINISH_REASONS = frozenset({"tool_calls", "function_call", "stop"})


def resolve_stream_idle_timeout_s(
    *,
    env_value: str | None = None,
    default: float = DEFAULT_STREAM_IDLE_TIMEOUT_S,
    maximum: float = MAX_STREAM_IDLE_TIMEOUT_S,
) -> float:
    """The streaming idle timeout in seconds, from ``NANOCLAW_STREAM_IDLE_TIMEOUT_S``.

    Invalid or non-positive values fall back to the default (90s); values
    above an hour are clamped. ``env_value`` replaces the environment, for
    tests. (nanobot base.py:50)
    """
    raw = os.environ.get(STREAM_IDLE_TIMEOUT_ENV) if env_value is None else env_value
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid {}={!r}; using {}", STREAM_IDLE_TIMEOUT_ENV, raw, default)
        return default
    if value <= 0:
        logger.warning("Ignoring non-positive {}={!r}; using {}", STREAM_IDLE_TIMEOUT_ENV, raw, default)
        return default
    if value > maximum:
        logger.warning("Clamping {}={!r} to {}", STREAM_IDLE_TIMEOUT_ENV, raw, maximum)
        return maximum
    return value


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
    """Token counts reported by, or estimated for, one or more model calls.

    Mirrors nanobot llm_usage/types.py (``LLMUsage``) without the streaming
    timing fields (``ttft_ms``/``generation_ms``: NanoClaw has no call
    observer to report them to).

    ``input_tokens`` includes cache reads and writes. ``None`` cache counts
    mean the provider didn't report them; ``0`` means it reported none.
    ``total_tokens`` keeps a provider-reported total larger than input plus
    output (hidden reasoning tokens); left out, it is input + output.
    ``reported_tokens + estimated_tokens == total_tokens`` always; left out,
    ``reported_tokens`` is whatever isn't estimated. ``context_tokens`` is the
    prompt size of the latest call; ``request_count`` the number of calls summed.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = None  # type: ignore[assignment]  # filled in __post_init__
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reported_tokens: int = None  # type: ignore[assignment]  # filled in __post_init__
    estimated_tokens: int = 0
    context_tokens: int | None = None
    request_count: int = 0

    def __post_init__(self) -> None:
        # Fill the derived defaults, then validate (nanobot types.py:38).
        visible_total = self.input_tokens + self.output_tokens
        if self.total_tokens is None:
            object.__setattr__(self, "total_tokens", visible_total)
        if self.reported_tokens is None:
            object.__setattr__(self, "reported_tokens", max(0, self.total_tokens - self.estimated_tokens))
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reported_tokens",
            "estimated_tokens",
            "request_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("cache_read_tokens", "cache_write_tokens", "context_tokens"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be None or a non-negative integer")
        if self.total_tokens < visible_total:
            raise ValueError("total_tokens must be at least input_tokens + output_tokens")
        if self.reported_tokens + self.estimated_tokens != self.total_tokens:
            raise ValueError("reported_tokens + estimated_tokens must equal total_tokens")
        if (self.cache_read_tokens or 0) + (self.cache_write_tokens or 0) > self.input_tokens:
            raise ValueError("cache token counts cannot exceed input_tokens")

    @classmethod
    def reported(
        cls,
        *,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
    ) -> LLMUsage:
        """Usage taken from a provider response (nanobot types.py:85)."""
        visible_total = input_tokens + output_tokens
        total = visible_total if total_tokens is None else max(visible_total, total_tokens)
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            reported_tokens=total,
            context_tokens=input_tokens,
            request_count=1,
        )

    @classmethod
    def estimated(cls, *, input_tokens: int, output_tokens: int) -> LLMUsage:
        """Usage counted locally because the provider sent none (nanobot types.py:111)."""
        total = input_tokens + output_tokens
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            reported_tokens=0,
            estimated_tokens=total,
            context_tokens=input_tokens,
            request_count=1,
        )

    @classmethod
    def empty_request(cls) -> LLMUsage:
        """A finished request with nothing to count, e.g. an error (nanobot types.py:123)."""
        return cls(request_count=1)

    @property
    def source(self) -> str:
        """``reported``, ``estimated`` or ``mixed``."""
        if self.estimated_tokens == 0:
            return "reported"
        if self.reported_tokens == 0:
            return "estimated"
        return "mixed"

    def __add__(self, other: LLMUsage) -> LLMUsage:
        """Sum two usages; a cache count survives only if both sides reported it.

        The latest call's ``context_tokens`` wins. (nanobot types.py:157)
        """

        def _sum_cache(left: int | None, right: int | None) -> int | None:
            return left + right if left is not None and right is not None else None

        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_read_tokens=_sum_cache(self.cache_read_tokens, other.cache_read_tokens),
            cache_write_tokens=_sum_cache(self.cache_write_tokens, other.cache_write_tokens),
            reported_tokens=self.reported_tokens + other.reported_tokens,
            estimated_tokens=self.estimated_tokens + other.estimated_tokens,
            context_tokens=other.context_tokens if other.context_tokens is not None else self.context_tokens,
            request_count=self.request_count + other.request_count,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON form, e.g. for session metadata (nanobot types.py:188)."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reported_tokens": self.reported_tokens,
            "estimated_tokens": self.estimated_tokens,
            "source": self.source,
            "context_tokens": self.context_tokens,
            "request_count": self.request_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> LLMUsage | None:
        """Read :meth:`to_dict` output back; ``None`` unless it is exactly that shape.

        (nanobot types.py:233)
        """
        if not isinstance(value, dict) or set(value) != set(_USAGE_DICT_KEYS):
            return None
        fields = {key: value[key] for key in _USAGE_DICT_KEYS if key != "source"}
        if any(isinstance(item, bool) or not isinstance(item, (int, type(None))) for item in fields.values()):
            return None
        try:
            usage = cls(**fields)
        except (TypeError, ValueError):
            return None
        return usage if value["source"] == usage.source else None


_USAGE_DICT_KEYS = tuple(LLMUsage().to_dict())


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
    # Anthropic extended-thinking blocks ({"type": "thinking", "thinking",
    # "signature"}), replayed on the next request (nanobot base.py:299).
    thinking_blocks: list[dict[str, Any]] | None = None
    # Seconds the provider asked us to wait before retrying.
    retry_after: float | None = None
    # Error details the retry policy reads when finish_reason == "error"
    # (nanobot base.py:318-324).
    error_status_code: int | None = None
    error_kind: str | None = None  # "timeout", "connection", "authentication", ...
    error_type: str | None = None  # the API's error type, e.g. "insufficient_quota"
    error_code: str | None = None  # the API's error code, e.g. "rate_limit_exceeded"
    error_retry_after_s: float | None = None  # from the Retry-After headers
    error_should_retry: bool | None = None  # from the x-should-retry header

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


def tool_arguments_object_for_replay(arguments: Any) -> dict[str, Any]:
    """Object-shaped arguments for a tool call being *replayed* from history.

    Unlike :func:`parse_tool_arguments` (used for calls about to run), this may
    replace bad arguments: malformed JSON or a non-object becomes ``{}`` so
    the request isn't rejected. nanobot first tries ``json_repair``.
    (nanobot base.py:144)
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def tool_arguments_json_for_replay(arguments: Any) -> str:
    """:func:`tool_arguments_object_for_replay` as a JSON string (nanobot base.py:172)."""
    return json.dumps(tool_arguments_object_for_replay(arguments), ensure_ascii=False)


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

    # True when chat_stream itself accepts ``on_stream_recover`` (it restarts
    # a stalled stream internally). nanobot base.py:1278; no NanoClaw provider
    # does yet, so the retry loop calls the recover callback instead.
    supports_stream_recover_callback = False

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

    async def aclose(self) -> None:
        """Release network resources (nanobot base.py ``aclose``). Nothing by default."""
        return None

    async def safe_chat(self, **kwargs: Any) -> LLMResponse:
        """Call :meth:`chat`, turning any unexpected exception into an error response.

        Cancellation (``asyncio.CancelledError``) is a ``BaseException`` and
        still propagates, so a user can interrupt a turn.
        """
        try:
            return await self.chat(**kwargs)
        except Exception as exc:
            return self._error_response_from_exception(exc)

    # --- streaming (nanobot base.py:1085-1297) -----------------------------------

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
        """Like :meth:`chat`, calling ``on_content_delta`` with each piece of text.

        Returns the same ``LLMResponse`` as :meth:`chat`. This default makes a
        normal call (bounded by the stream idle timeout) and delivers the whole
        content as a single delta; the thinking and tool-call callbacks are not
        called. Providers that can stream natively override it.
        (nanobot base.py:1085)
        """
        _ = on_thinking_delta, on_tool_call_delta
        response = await asyncio.wait_for(
            self.chat(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                tool_choice=tool_choice,
            ),
            timeout=resolve_stream_idle_timeout_s(),
        )
        if on_content_delta and response.content:
            await on_content_delta(response.content)
        return response

    async def safe_chat_stream(self, **kwargs: Any) -> LLMResponse:
        """:meth:`chat_stream`, turning unexpected exceptions into error responses.

        (nanobot base.py:1147 ``_safe_chat_stream``, without the TTFT timing that
        only feeds nanobot's call observer.)
        """
        try:
            return await self.chat_stream(**kwargs)
        except Exception as exc:
            return self._error_response_from_exception(exc)

    async def chat_stream_with_retry(
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
        on_stream_recover: StreamRecoverCallback | None = None,
        retry_mode: str = "standard",
        on_retry_wait: RetryWaitCallback | None = None,
    ) -> LLMResponse:
        """:meth:`chat_stream` under the :meth:`chat_with_retry` policy (nanobot base.py:1230).

        Once text has reached ``on_content_delta`` a failed stream is not
        retried, since the user has already seen part of an answer. The
        exception is an idle timeout: ``on_stream_recover`` is awaited (the
        caller starts a new output segment) and the request is retried; without
        that callback the retry runs with all delta callbacks switched off.
        Thinking and tool-call deltas don't block retries.
        """
        has_streamed_content = False

        async def _tracking_delta(text: str) -> None:
            nonlocal has_streamed_content
            if text:
                has_streamed_content = True
            if on_content_delta:
                await on_content_delta(text)

        async def _recover_stream() -> None:
            nonlocal has_streamed_content
            if on_stream_recover:
                await on_stream_recover()
            has_streamed_content = False

        kwargs: dict[str, Any] = dict(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tool_choice=tool_choice,
            on_content_delta=_tracking_delta if on_content_delta is not None else None,
            on_thinking_delta=on_thinking_delta,
            on_tool_call_delta=on_tool_call_delta,
        )
        if on_stream_recover and self.supports_stream_recover_callback:
            kwargs["on_stream_recover"] = _recover_stream
        return await self._run_with_retry(
            kwargs,
            retry_mode=retry_mode,
            on_retry_wait=on_retry_wait,
            call=self.safe_chat_stream,
            should_retry_guard=lambda: not has_streamed_content,
            on_stream_recover=_recover_stream if on_stream_recover else None,
        )

    @staticmethod
    def _error_response_from_exception(exc: Exception) -> LLMResponse:
        """An error response for an unexpected exception, classified by class name.

        (nanobot base.py:738)
        """
        error_names = tuple(cls.__name__.lower() for cls in type(exc).__mro__)

        def named(*tokens: str) -> bool:
            return any(token in name for name in error_names for token in tokens)

        error_kind: str | None = None
        error_should_retry: bool | None = None
        if named("timeout"):
            error_kind, error_should_retry = "timeout", True
        elif named("connect", "connection", "network", "protocol", "transport"):
            error_kind, error_should_retry = "connection", True
        elif named("ratelimit", "throttl"):
            error_kind, error_should_retry = "rate_limit", True
        elif named("server", "internal"):
            error_kind, error_should_retry = "server_error", True
        elif named("auth", "credential", "permissiondenied", "unauthor"):
            error_kind = "authentication"

        raw_status = getattr(exc, "status_code", None)
        if raw_status is None:
            raw_status = getattr(getattr(exc, "response", None), "status_code", None)
        try:
            status = int(raw_status) if raw_status is not None else None
        except (TypeError, ValueError):
            status = None
        raw_type = getattr(exc, "error_type", None)
        raw_code = getattr(exc, "error_code", None)
        detail = str(exc).strip() or type(exc).__name__
        return LLMResponse(
            content=f"Error calling LLM: {detail}",
            finish_reason="error",
            error_status_code=status,
            error_kind=error_kind,
            error_type=str(raw_type) if raw_type is not None else None,
            error_code=str(raw_code) if raw_code is not None else None,
            error_should_retry=error_should_retry,
        )

    # --- retries (nanobot base.py:354-428, 800-900, 1299-1771) ----------------

    _CHAT_RETRY_DELAYS = (1, 2, 4)  # standard mode: 4 attempts in all
    _PERSISTENT_MAX_DELAY = 60
    _PERSISTENT_IDENTICAL_ERROR_LIMIT = 10
    _RETRY_HEARTBEAT_CHUNK = 30  # long waits report progress every 30s
    _TRANSIENT_ERROR_MARKERS = (
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "overloaded",
        "timeout",
        "timed out",
        "connection",
        "server error",
        "server_error",
        "temporarily unavailable",
        "速率限制",
        "访问量过大",
    )
    _RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
    _TRANSIENT_ERROR_KINDS = frozenset({"timeout", "connection"})
    # 429s that mean "out of money", not "slow down": retrying can't help.
    _NON_RETRYABLE_429_ERROR_TOKENS = frozenset(
        {
            "insufficient_quota",
            "quota_exceeded",
            "quota_exhausted",
            "billing_hard_limit_reached",
            "insufficient_balance",
            "insufficient_credits",
            "credit_balance_too_low",
            "billing_not_active",
            "payment_required",
        }
    )
    _NON_RETRYABLE_429_TEXT_MARKERS = (
        "insufficient_quota",
        "insufficient quota",
        "quota exceeded",
        "quota exhausted",
        "billing hard limit",
        "billing_hard_limit_reached",
        "billing not active",
        "insufficient balance",
        "insufficient_balance",
        "insufficient credits",
        "insufficient_credits",
        "credit balance too low",
        "payment required",
        "out of credits",
        "out of quota",
        "exceeded your current quota",
    )
    # nanobot also lists rate-limit tokens and texts (_RETRYABLE_429_*), but an
    # unknown 429 is retried anyway, so they never change the outcome.

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
        retry_mode: str = "standard",
        on_retry_wait: RetryWaitCallback | None = None,
    ) -> LLMResponse:
        """:meth:`chat`, retried on transient failures (nanobot base.py:1299).

        ``retry_mode="standard"`` makes up to 4 attempts, waiting 1, 2 and 4
        seconds (or the server's Retry-After plus 1s). ``"persistent"`` keeps
        going, with waits capped at 60s, until the same error has come back 10
        times in a row. Auth (401/403) and billing/quota errors stop at once.
        ``on_retry_wait(text)``, if given, hears about each wait and about
        giving up (nanobot's ``on_retry_wait`` / ``on_retry_exhausted``).
        """
        kwargs: dict[str, Any] = dict(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tool_choice=tool_choice,
        )
        return await self._run_with_retry(kwargs, retry_mode=retry_mode, on_retry_wait=on_retry_wait)

    async def _run_with_retry(
        self,
        kwargs: dict[str, Any],
        *,
        retry_mode: str,
        on_retry_wait: RetryWaitCallback | None,
        call: Callable[..., Awaitable[LLMResponse]] | None = None,
        should_retry_guard: Callable[[], bool] | None = None,
        on_stream_recover: StreamRecoverCallback | None = None,
    ) -> LLMResponse:
        """The retry loop (nanobot base.py:1580), without image stripping or status events.

        ``call`` is :meth:`safe_chat` (the default) or :meth:`safe_chat_stream`
        (nanobot base.py:1388). For streams, ``should_retry_guard()`` turns
        False once text was shown, and ``on_stream_recover`` lets a retry after
        an idle timeout start a new output segment.
        """
        call = call or self.safe_chat
        delays = self._CHAT_RETRY_DELAYS
        persistent = retry_mode == "persistent"
        attempt = 0
        last_error_key: str | None = None
        identical_error_count = 0
        while True:
            attempt += 1
            response = await call(**kwargs)
            if response.finish_reason != "error":
                return response

            # Text already shown can't be taken back, so only a stalled stream
            # is retried (nanobot base.py:1621-1645).
            if should_retry_guard is not None and not should_retry_guard():
                if (response.error_kind or "").lower() != "timeout":
                    logger.warning("LLM stream failed after content was emitted; skipping retry")
                    return response
                if on_stream_recover:
                    logger.warning(
                        "LLM stream stalled after content was emitted; starting a new stream segment and retrying"
                    )
                    await on_stream_recover()
                else:
                    logger.warning(
                        "LLM stream stalled after content was emitted; suppressing delta callbacks and retrying"
                    )
                    kwargs["on_content_delta"] = None
                    kwargs["on_thinking_delta"] = None
                    kwargs["on_tool_call_delta"] = None
                    should_retry_guard = None

            error_key = (response.content or "").strip().lower() or None
            if error_key and error_key == last_error_key:
                identical_error_count += 1
            else:
                last_error_key = error_key
                identical_error_count = 1 if error_key else 0

            if not self.is_transient_response(response):
                # nanobot retries once without images here unless the error is
                # auth or billing; NanoClaw sends no images, so it just stops.
                return response

            if persistent and identical_error_count >= self._PERSISTENT_IDENTICAL_ERROR_LIMIT:
                logger.warning(
                    "Stopping persistent retry after {} identical transient errors: {}",
                    identical_error_count,
                    (response.content or "")[:120].lower(),
                )
                if on_retry_wait:
                    await on_retry_wait(f"Persistent retry stopped after {identical_error_count} identical errors.")
                return response
            if not persistent and attempt > len(delays):
                logger.warning(
                    "LLM request failed after {} attempts, giving up: {}",
                    attempt,
                    (response.content or "")[:120].lower(),
                )
                if on_retry_wait:
                    await on_retry_wait(f"Model request failed after {attempt} attempts, giving up.")
                return response

            retry_after = self._extract_retry_after_from_response(response)
            base_delay = delays[min(attempt - 1, len(delays) - 1)]
            delay = retry_after + RETRY_AFTER_BUFFER if retry_after else base_delay
            if persistent:
                delay = min(delay, self._PERSISTENT_MAX_DELAY)
            # nanobot base.py:1754
            logger.warning(
                "LLM transient error (attempt {}{}), retrying in {}s: {}",
                attempt,
                "+" if persistent and attempt > len(delays) else f"/{len(delays)}",
                int(round(delay)),
                (response.content or "")[:120].lower(),
            )
            await self._sleep_with_heartbeat(
                delay, attempt=attempt, persistent=persistent, on_retry_wait=on_retry_wait
            )

    async def _sleep_with_heartbeat(
        self,
        delay: float,
        *,
        attempt: int,
        persistent: bool,
        on_retry_wait: RetryWaitCallback | None,
    ) -> None:
        """Sleep in chunks of up to 30s, reporting before each one (nanobot base.py:1527).

        Uses ``asyncio.sleep`` so tests can patch it out.
        """
        remaining = max(0.0, delay)
        while remaining > 0:
            if on_retry_wait:
                kind = "persistent retry" if persistent else "retry"
                await on_retry_wait(
                    f"Model request failed, {kind} in {max(1, int(round(remaining)))}s (attempt {attempt})."
                )
            chunk = min(remaining, self._RETRY_HEARTBEAT_CHUNK)
            await asyncio.sleep(chunk)
            remaining -= chunk

    # --- error classification ------------------------------------------------

    @classmethod
    def _is_transient_error(cls, content: str | None) -> bool:
        err = (content or "").lower()
        return any(marker in err for marker in cls._TRANSIENT_ERROR_MARKERS)

    @classmethod
    def is_transient_response(cls, response: LLMResponse) -> bool:
        """Whether an error response is worth retrying (nanobot base.py:800).

        Structured details win (x-should-retry, then status code, then kind);
        the error text is the fallback.
        """
        if response.error_should_retry is not None:
            return bool(response.error_should_retry)
        if response.error_status_code is not None:
            status = int(response.error_status_code)
            if status == 429:
                return cls._is_retryable_429_response(response)
            if status in cls._RETRYABLE_STATUS_CODES or status >= 500:
                return True
        kind = (response.error_kind or "").strip().lower()
        if kind in cls._TRANSIENT_ERROR_KINDS:
            return True
        return cls._is_transient_error(response.content)

    @classmethod
    def is_arrearage_response(cls, response: LLMResponse) -> bool:
        """A quota/billing error (402, ``insufficient_quota``, ...) that retrying won't fix.

        (nanobot base.py:819)
        """
        if response.error_status_code is not None and int(response.error_status_code) == 402:
            return True
        tokens = (cls._normalize_error_token(response.error_type), cls._normalize_error_token(response.error_code))
        if any(token in cls._NON_RETRYABLE_429_ERROR_TOKENS for token in tokens if token is not None):
            return True
        content = (response.content or "").lower()
        return any(marker in content for marker in cls._NON_RETRYABLE_429_TEXT_MARKERS)

    @staticmethod
    def _normalize_error_token(value: Any) -> str | None:
        if value is None:
            return None
        token = str(value).strip().lower()
        return token or None

    @classmethod
    def _extract_error_type_code(cls, payload: Any) -> tuple[str | None, str | None]:
        """``type`` and ``code`` from an error body (dict or JSON text), lower-cased.

        (nanobot base.py:845)
        """
        data: dict[str, Any] | None = None
        if isinstance(payload, dict):
            data = payload
        elif isinstance(payload, str) and payload.strip():
            try:
                parsed = json.loads(payload)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                data = parsed
        if data is None:
            return None, None
        type_value = data.get("type")
        code_value = data.get("code")
        error_obj = data.get("error")
        if isinstance(error_obj, dict):
            type_value = error_obj.get("type") or type_value
            code_value = error_obj.get("code") or code_value
        return cls._normalize_error_token(type_value), cls._normalize_error_token(code_value)

    @classmethod
    def _is_retryable_429_response(cls, response: LLMResponse) -> bool:
        """Rate limit (retry) or out of quota (stop)? Unknown 429s are retried.

        (nanobot base.py:876)
        """
        semantic_tokens = {
            token
            for token in (cls._normalize_error_token(response.error_type), cls._normalize_error_token(response.error_code))
            if token is not None
        }
        if semantic_tokens & cls._NON_RETRYABLE_429_ERROR_TOKENS:
            return False
        content = (response.content or "").lower()
        if any(marker in content for marker in cls._NON_RETRYABLE_429_TEXT_MARKERS):
            return False
        return True

    # --- Retry-After (nanobot base.py:1441-1525) -------------------------------

    @classmethod
    def _extract_retry_after(cls, content: str | None) -> float | None:
        """A wait in seconds named in error text ("retry after 5s", "try again in 1m30s")."""
        text = (content or "").lower()
        patterns = (
            r"retry after\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)?",
            r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)",
            r"wait\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)\s*before retry",
            r"retry[_-]?after[\"'\s:=]+(\d+(?:\.\d+)?)",
        )
        for idx, pattern in enumerate(patterns):
            if idx == 1 and (compound := cls._extract_compound_try_again_in(text)) is not None:
                return compound
            match = re.search(pattern, text)
            if not match:
                continue
            unit = match.group(2) if idx < 3 else "s"
            return cls._to_retry_seconds(float(match.group(1)), unit)
        return None

    @classmethod
    def _extract_compound_try_again_in(cls, text: str) -> float | None:
        """Sum Go-style durations such as OpenAI's ``try again in 1m30s``."""
        match = re.search(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)(?![a-z])", text)
        if not match:
            return None
        unit_seconds = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", match.group(1))
        return max(0.1, sum(float(value) * unit_seconds[unit] for value, unit in parts))

    @classmethod
    def _to_retry_seconds(cls, value: float, unit: str | None = None) -> float:
        normalized = (unit or "s").lower()
        if normalized in {"ms", "milliseconds"}:
            return max(0.1, value / 1000.0)
        if normalized in {"m", "min", "minutes"}:
            return max(0.1, value * 60.0)
        return max(0.1, value)

    @classmethod
    def _extract_retry_after_from_headers(cls, headers: Any) -> float | None:
        """Seconds from ``retry-after-ms`` or ``retry-after`` (seconds or an HTTP date)."""
        if not headers:
            return None

        def header(name: str) -> Any:
            if hasattr(headers, "get"):
                value = headers.get(name) or headers.get(name.title())
                if value is not None:
                    return value
            if isinstance(headers, dict):
                for key, value in headers.items():
                    if isinstance(key, str) and key.lower() == name:
                        return value
            return None

        with suppress(TypeError, ValueError):
            retry_ms = header("retry-after-ms")
            if retry_ms is not None:
                seconds = float(retry_ms) / 1000.0
                if seconds > 0:
                    return seconds
        retry_after = header("retry-after")
        text = str(retry_after).strip() if retry_after is not None else ""
        if not text:
            return None
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            return cls._to_retry_seconds(float(text), "s")
        try:
            retry_at = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.1, (retry_at - datetime.now(retry_at.tzinfo)).total_seconds())

    @classmethod
    def _extract_retry_after_from_response(cls, response: LLMResponse) -> float | None:
        if response.error_retry_after_s is not None and response.error_retry_after_s > 0:
            return response.error_retry_after_s
        if response.retry_after is not None and response.retry_after > 0:
            return response.retry_after
        return cls._extract_retry_after(response.content)

    # --- usage -----------------------------------------------------------------

    def usage_for_call(self, response: LLMResponse, kwargs: dict[str, Any]) -> LLMUsage | None:
        """The response's usage, or a tiktoken estimate when the provider sent none.

        Errors get no estimate. (nanobot base.py:449 ``_usage_for_call``; nanobot runs it in a
        thread, NanoClaw synchronously.)
        """
        usage = response.usage
        if usage is not None and usage.total_tokens > 0:
            return usage
        if response.finish_reason in {"error", "cancelled"}:
            return None
        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            return usage
        tools = kwargs.get("tools") if isinstance(kwargs.get("tools"), list) else None
        # Imported here: tokens may load tiktoken, which most provider users never need.
        from nanoclaw.utils.tokens import estimate_message_tokens, estimate_prompt_tokens

        input_tokens, _ = estimate_prompt_tokens(messages, tools)
        assistant: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            assistant["tool_calls"] = [call.to_openai_tool_call() for call in response.tool_calls]
        if response.reasoning_content is not None or response.thinking_blocks:
            assistant["reasoning_content"] = response.reasoning_content or ""
        if response.thinking_blocks:
            assistant["thinking_blocks"] = response.thinking_blocks
        return LLMUsage.estimated(
            input_tokens=max(0, input_tokens),
            output_tokens=max(0, estimate_message_tokens(assistant)),
        )
