"""Provider wrapper that transparently fails over to fallback models on error.

Mirrors nanobot/providers/fallback_provider.py (``FallbackProvider``,
``FallbackModelSelection``). The failover rules (``_should_fallback``), the
order of attempts, the circuit breaker, the stream rules and the returned
error are nanobot's. What NanoClaw leaves out, because its providers don't
have it:

- ``ProviderCallContext`` and its entry points (``chat_with_context``,
  ``chat_stream_with_context``): conversation state, native compaction
  windows and per-call response-source attribution;
- ``RetryStatusEvent`` callbacks (``on_retry_status``): NanoClaw's retry loop
  only reports text (``on_retry_wait`` / ``on_retry_exhausted``);
- per-call ``reasoning_effort``: a NanoClaw provider takes its effort when it
  is built, so each fallback's effort comes with the provider the factory
  builds for its preset;
- ``GenerationSettings`` forwarding and the LLM call observer.
"""

# pyright: reportIncompatibleMethodOverride=false, reportIncompatibleVariableOverride=false

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from loguru import logger

from nanoclaw.providers.base import (
    LLMProvider,
    LLMResponse,
    RetryWaitCallback,
    StreamRecoverCallback,
)
from nanoclaw.providers.registry import find_by_name

__all__ = ["FallbackModelObserver", "FallbackModelSelection", "FallbackProvider"]

# Circuit breaker tuned to match OpenAICompatProvider's Responses API breaker.
_PRIMARY_FAILURE_THRESHOLD = 3
_PRIMARY_COOLDOWN_S = 60
_FALLBACK_ERROR_KINDS = frozenset({
    "timeout",
    "connection",
    "server_error",
    "rate_limit",
    "overloaded",
})
_AUTHENTICATION_ERROR_KINDS = frozenset({
    "oauth_auth_required",
    "authentication",
    "auth",
    "permission",
})
_AUTHENTICATION_ERROR_TOKENS = (
    "authentication_error",
    "authentication error",
    "invalid_api_key",
    "invalid api key",
    "incorrect_api_key",
    "incorrect api key",
    "expired_api_key",
    "expired api key",
    "invalid credential",
    "expired credential",
    "credential has expired",
    "credentials have expired",
    "invalid_token",
    "invalid token",
    "expired_token",
    "expired token",
    "unauthorized",
    "permission_denied",
    "permission denied",
    "access_denied",
    "account_deactivated",
    "organization_deactivated",
    "not logged in",
)
_NON_FALLBACK_ERROR_KINDS = frozenset({
    "content_filter",
    "refusal",
    "context_length",
    "invalid_request",
})
_FALLBACK_ERROR_TOKENS = (
    "rate_limit",
    "rate limit",
    "too_many_requests",
    "too many requests",
    "overloaded",
    "server_error",
    "server error",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "connection",
    "empty",  # API returned empty choices (e.g. DeepSeek peak hours), transient
    "insufficient_quota",
    "insufficient quota",
    "quota_exceeded",
    "quota exceeded",
    "quota_exhausted",
    "quota exhausted",
    "billing_hard_limit",
    "insufficient_balance",
    "balance",
    "out of credits",
)
# OAuth credential failures (nanobot oauth_catalog_cache.py:24).
_OAUTH_REJECTED_CODES = frozenset({
    "invalid_grant",
    "invalid_token",
    "token_revoked",
    "refresh_token_expired",
    "refresh_token_reused",
    "refresh_token_invalidated",
})


def _oauth_auth_rejected(status: int, payload: object = None) -> bool:
    """Classify only explicit credential failures (nanobot ``oauth_catalog_auth_rejected``)."""
    if status == 401:
        return True
    if status not in {400, 403} or not isinstance(payload, dict):
        return False
    error = payload.get("error")
    code = error.get("code") if isinstance(error, dict) else error
    return isinstance(code, str) and code in _OAUTH_REJECTED_CODES


@dataclass(frozen=True)
class FallbackModelSelection:
    """Display-safe fallback result; no upstream error text or credentials."""

    model: str
    reauth_provider: str | None = None


FallbackModelObserver = Callable[[FallbackModelSelection], Awaitable[None]]


class FallbackProvider(LLMProvider):
    """Wrap a primary provider and transparently failover to fallback models.

    When the primary model returns a fallbackable error before content has been
    streamed, the wrapper tries each fallback model in order. Streamed timeout
    errors are the recovery exception: the caller may close the current stream
    segment, then the wrapper continues failover with later deltas in a new
    segment. Each fallback model may reside on a different provider — a factory
    callable creates the underlying provider on-the-fly.

    Key design:
    - Failover is request-scoped (the wrapper itself is stateless between turns).
    - Retrying entry points exhaust one provider's retry policy before failover.
    - Skipped when content was already streamed to avoid duplicate output,
      except timeout recovery can resume in a new stream segment.
    - Recursive failover is prevented by the factory returning plain providers.
    - Primary provider is circuit-broken after repeated failures to avoid
      wasting requests on a known-bad endpoint.
    - A fallback provider is built for one attempt and closed right after it
      (nanobot's ``finally: await fallback_provider.aclose()``); ``aclose``
      closes the primary.
    """

    supports_stream_recover_callback = True

    def __init__(
        self,
        primary: LLMProvider,
        fallback_presets: list[Any],
        provider_factory: Callable[[Any], LLMProvider],
        fallback_model_observer: FallbackModelObserver | None = None,
        primary_context_window_tokens: int | None = None,
        fallback_preset_names: list[str | None] | None = None,
    ):
        self._primary = primary
        super().__init__(api_key=primary.api_key, api_base=primary.api_base, default_model=primary.default_model)
        self._fallback_presets = list(fallback_presets)
        self._fallback_preset_names = tuple(fallback_preset_names or [None] * len(fallback_presets))
        if len(self._fallback_preset_names) != len(self._fallback_presets):
            raise ValueError("fallback preset names must match fallback candidates")
        self._provider_factory = provider_factory
        self._fallback_model_observer = fallback_model_observer
        # nanobot gives it to the primary's native compaction; NanoClaw has
        # none and keeps it for inspection only.
        self._primary_context_window_tokens = primary_context_window_tokens
        self._has_fallbacks = bool(fallback_presets)
        self._primary_failures = 0
        self._primary_tripped_at: float | None = None

    # The primary's identity stands for the chain (nanobot reads it from the primary).
    @property
    def default_model(self) -> str:  # type: ignore[override]
        return self._primary.default_model

    @default_model.setter
    def default_model(self, value: str) -> None:
        self._primary.default_model = value

    @property
    def provider_name(self) -> str:
        return str(getattr(self._primary, "provider_name", "") or "")

    @property
    def primary(self) -> LLMProvider:
        return self._primary

    @property
    def fallback_presets(self) -> tuple[Any, ...]:
        return tuple(self._fallback_presets)

    def get_default_model(self) -> str:
        return self._primary.default_model

    async def aclose(self) -> None:
        await self._primary.aclose()

    def set_fallback_model_observer(self, observer: FallbackModelObserver | None) -> None:
        """Attach a process-level observer without changing request call signatures."""
        self._fallback_model_observer = observer

    def _primary_available(self) -> bool:
        """Return True if the primary provider is not currently tripped."""
        if self._primary_tripped_at is None:
            return True
        if time.monotonic() - self._primary_tripped_at >= _PRIMARY_COOLDOWN_S:
            # Half-open: allow one probe attempt.
            return True
        return False

    async def chat(self, **kwargs: Any) -> LLMResponse:
        if not self._has_fallbacks:
            return await self._primary.chat(**kwargs)
        return await self._try_with_fallback(lambda p, kw: p.chat(**kw), kwargs, has_streamed=None)

    async def _run_chat_with_retry(
        self,
        kwargs: dict[str, Any],
        *,
        stream: bool,
        retry_mode: str,
        on_retry_wait: RetryWaitCallback | None,
        on_retry_exhausted: RetryWaitCallback | None,
        should_retry_guard: Callable[[], bool] | None = None,
        on_stream_recover: StreamRecoverCallback | None = None,
    ) -> LLMResponse:
        """Retry each provider before advancing through the fallback chain."""
        call_kwargs = dict(kwargs)
        if not self._has_fallbacks:
            call_kwargs.update({
                "retry_mode": retry_mode,
                "on_retry_wait": on_retry_wait,
                "on_retry_exhausted": on_retry_exhausted,
            })
            if stream:
                return await self._primary.chat_stream_with_retry(**call_kwargs)
            return await self._primary.chat_with_retry(**call_kwargs)

        has_streamed: list[bool] | None = None
        recover_stream = on_stream_recover
        if stream:
            streamed = [False]
            has_streamed = streamed
            original_delta = call_kwargs.get("on_content_delta")

            async def _tracking_delta(text: str) -> None:
                if text:
                    streamed[0] = True
                if original_delta:
                    await original_delta(text)

            async def _recover_stream() -> None:
                streamed[0] = False
                if on_stream_recover:
                    await on_stream_recover()

            if original_delta is not None:
                call_kwargs["on_content_delta"] = _tracking_delta
            if on_stream_recover is not None:
                call_kwargs["on_stream_recover"] = _recover_stream
                recover_stream = _recover_stream

        async def _call_provider(provider: LLMProvider, provider_kwargs: dict[str, Any]) -> LLMResponse:
            if stream:
                return await provider.chat_stream_with_retry(**provider_kwargs)
            return await provider.chat_with_retry(**provider_kwargs)

        return await self._retry_with_fallback(
            _call_provider,
            call_kwargs,
            retry_mode=retry_mode,
            on_retry_wait=on_retry_wait,
            on_retry_exhausted=on_retry_exhausted,
            has_streamed=has_streamed,
            on_stream_recover=recover_stream,
            persistent_retry_guard=should_retry_guard,
        )

    async def chat_stream(self, **kwargs: Any) -> LLMResponse:
        on_stream_recover = kwargs.pop("on_stream_recover", None)
        if not self._has_fallbacks:
            return await self._primary.chat_stream(**kwargs)

        has_streamed: list[bool] = [False]
        original_delta = kwargs.get("on_content_delta")

        async def _tracking_delta(text: str) -> None:
            if text:
                has_streamed[0] = True
            if original_delta:
                await original_delta(text)

        kwargs["on_content_delta"] = _tracking_delta
        return await self._try_with_fallback(
            lambda p, kw: p.chat_stream(**kw),
            kwargs,
            has_streamed=has_streamed,
            on_stream_recover=on_stream_recover,
        )

    async def _retry_with_fallback(
        self,
        call: Callable[[LLMProvider, dict[str, Any]], Awaitable[LLMResponse]],
        kwargs: dict[str, Any],
        *,
        retry_mode: str,
        on_retry_wait: RetryWaitCallback | None,
        on_retry_exhausted: RetryWaitCallback | None,
        has_streamed: list[bool] | None,
        on_stream_recover: StreamRecoverCallback | None,
        persistent_retry_guard: Callable[[], bool] | None,
    ) -> LLMResponse:
        """Retry each candidate, deferring terminal events until the chain fails."""

        async def _call_chain(**chain_kwargs: Any) -> LLMResponse:
            last_exhausted_message: str | None = None
            primary_attempted = False
            primary_exhausted_message: str | None = None

            async def _capture_exhaustion(message: str) -> None:
                nonlocal last_exhausted_message
                last_exhausted_message = message

            async def _call_candidate(provider: LLMProvider, candidate_kwargs: dict[str, Any]) -> LLMResponse:
                nonlocal last_exhausted_message, primary_attempted, primary_exhausted_message
                last_exhausted_message = None
                if provider is self._primary:
                    primary_attempted = True
                response = await call(provider, {
                    **candidate_kwargs,
                    "retry_mode": "standard",
                    "on_retry_wait": on_retry_wait,
                    "on_retry_exhausted": _capture_exhaustion,
                })
                if provider is self._primary:
                    primary_exhausted_message = last_exhausted_message
                return response

            response = await self._try_with_fallback(
                _call_candidate,
                chain_kwargs,
                has_streamed=has_streamed,
                on_stream_recover=on_stream_recover,
            )
            if retry_mode != "persistent" and response.finish_reason == "error":
                terminal_message = primary_exhausted_message if primary_attempted else last_exhausted_message
                if terminal_message and on_retry_exhausted:
                    await on_retry_exhausted(terminal_message)
            return response

        if retry_mode != "persistent":
            return await _call_chain(**kwargs)
        return await self._run_with_retry(
            dict(kwargs),
            retry_mode="persistent",
            on_retry_wait=on_retry_wait,
            on_retry_exhausted=on_retry_exhausted,
            call=_call_chain,
            should_retry_guard=persistent_retry_guard,
            on_stream_recover=on_stream_recover,
        )

    async def _try_with_fallback(
        self,
        call: Callable[[LLMProvider, dict[str, Any]], Awaitable[LLMResponse]],
        kwargs: dict[str, Any],
        has_streamed: list[bool] | None,
        on_stream_recover: StreamRecoverCallback | None = None,
        on_fallback_attempt: Callable[[], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        primary_model = kwargs.get("model") or self.get_default_model()
        primary_was_attempted = False
        primary_response: LLMResponse | None = None
        primary_error = "unknown error"

        if self._primary_available():
            primary_was_attempted = True
            response, primary_exception = await self._call_provider(call, self._primary, kwargs)
            if primary_exception is not None:
                logger.warning(
                    "Primary model '{}' raised {} before responding",
                    primary_model, type(primary_exception).__name__,
                )
            if response.finish_reason != "error":
                self._primary_failures = 0
                self._primary_tripped_at = None
                return response
            primary_response = response
            primary_error = (response.content or primary_error)[:120]

            if has_streamed is not None and has_streamed[0]:
                is_timeout = (response.error_kind or "").lower() == "timeout"
                if is_timeout:
                    logger.warning(
                        "Primary model '{}' stream stalled after content was emitted; "
                        "attempting failover anyway",
                        primary_model,
                    )
                    has_streamed[0] = False
                    if on_stream_recover:
                        await on_stream_recover()
                    else:
                        kwargs["on_content_delta"] = None
                else:
                    logger.warning("Primary model error but content already streamed; skipping failover")
                    return response

            if not self._should_fallback(response):
                logger.warning(
                    "Primary model '{}' failed with non-fallbackable error: {}",
                    primary_model,
                    (response.content or "")[:120],
                )
                return response

            self._primary_failures += 1
            if self._primary_failures >= _PRIMARY_FAILURE_THRESHOLD:
                self._primary_tripped_at = time.monotonic()
                logger.warning(
                    "Primary model '{}' circuit open after {} consecutive failures",
                    primary_model, self._primary_failures,
                )
        else:
            logger.debug("Primary model '{}' circuit open; skipping", primary_model)

        last_response = primary_response
        primary_skipped = not primary_was_attempted
        for idx, fallback in enumerate(self._fallback_presets):
            fallback_model = fallback.model
            if has_streamed is not None and has_streamed[0]:
                is_timeout = last_response is not None and (last_response.error_kind or "").lower() == "timeout"
                if is_timeout and on_stream_recover:
                    logger.warning(
                        "Fallback model '{}' stream stalled after content was emitted; "
                        "starting a new stream segment and trying next fallback",
                        self._fallback_presets[idx - 1].model if idx > 0 else primary_model,
                    )
                    has_streamed[0] = False
                    await on_stream_recover()
                else:
                    break
            if idx == 0 and primary_skipped:
                logger.info(
                    "Primary model '{}' circuit open, trying fallback '{}'",
                    primary_model, fallback_model,
                )
            elif idx == 0:
                logger.info(
                    "Primary model '{}' failed: {}; trying fallback '{}'",
                    primary_model, primary_error, fallback_model,
                )
            else:
                logger.info(
                    "Fallback '{}' also failed, trying next fallback '{}'",
                    self._fallback_presets[idx - 1].model, fallback_model,
                )
            if on_fallback_attempt is not None:
                await on_fallback_attempt()
            try:
                fallback_provider = self._provider_factory(fallback)
            except Exception as exc:
                logger.warning("Failed to create provider for fallback '{}': {}", fallback_model, exc)
                continue

            fallback_kwargs = {
                **kwargs,
                "model": fallback_model,
                "max_tokens": fallback.max_tokens,
                "temperature": fallback.temperature,
            }
            try:
                fallback_response, fallback_exception = await self._call_provider(
                    call, fallback_provider, fallback_kwargs
                )
            finally:
                await fallback_provider.aclose()
            fallback_response.is_fallback = True
            if fallback_exception is not None:
                logger.warning(
                    "Fallback '{}' raised {}",
                    fallback_model, type(fallback_exception).__name__,
                )

            if fallback_response.finish_reason != "error":
                # Publish a model switch only once this response is known to be usable.
                await self._notify_fallback_model(fallback_model, primary_response)
                logger.info(
                    "Fallback '{}' succeeded after primary '{}' failed",
                    fallback_model, primary_model,
                )
                return fallback_response

            last_response = fallback_response
            logger.warning(
                "Fallback '{}' also failed: {}",
                fallback_model,
                (fallback_response.content or "")[:120],
            )

        logger.warning("All {} fallback model(s) failed", len(self._fallback_presets))
        # Return the last error response we saw (primary or last fallback).
        if last_response is not None:
            return replace(
                last_response,
                primary_error_kind=(
                    self.public_error_kind(primary_response) if primary_response is not None else None
                ),
            )
        # Primary was skipped and no fallback returned a response. Keep the result
        # transient until the primary circuit is eligible for another probe.
        retry_after_s = (
            max(0.1, _PRIMARY_COOLDOWN_S - (time.monotonic() - self._primary_tripped_at))
            if self._primary_tripped_at is not None
            else None
        )
        return LLMResponse(
            content=f"Primary model '{primary_model}' circuit open and no fallbacks available",
            finish_reason="error",
            error_retry_after_s=retry_after_s,
            error_should_retry=True,
        )

    @staticmethod
    async def _call_provider(
        call: Callable[[LLMProvider, dict[str, Any]], Awaitable[LLMResponse]],
        provider: LLMProvider,
        kwargs: dict[str, Any],
    ) -> tuple[LLMResponse, Exception | None]:
        """Turn provider exceptions into error responses without swallowing cancellation."""
        try:
            return await call(provider, kwargs), None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            response = LLMProvider._error_response_from_exception(exc)
            if response.error_kind is None and any(
                token in (str(exc).strip() or type(exc).__name__).lower()
                for token in _AUTHENTICATION_ERROR_TOKENS
            ):
                response.error_kind = "authentication"
            return response, exc

    async def _notify_fallback_model(self, model: str, primary_response: LLMResponse | None) -> None:
        if self._fallback_model_observer is None:
            return
        reauth_provider = None
        spec = find_by_name(self.provider_name) if self.provider_name else None
        if spec is not None and getattr(spec, "is_oauth", False) and primary_response is not None:
            # Plain 403, rate limits, transport errors, and message substrings are
            # not evidence of revoked credentials. A skipped circuit has no new
            # auth result either; never retain credential state on this wrapper.
            if primary_response.error_kind == "oauth_auth_required" or _oauth_auth_rejected(
                primary_response.error_status_code or 0,
                {"error": {"code": primary_response.error_code}},
            ):
                reauth_provider = spec.name
        try:
            await self._fallback_model_observer(FallbackModelSelection(model, reauth_provider))
        except Exception:
            logger.exception("fallback model observer failed for '{}'", model)

    @staticmethod
    def _should_fallback(response: LLMResponse) -> bool:
        if LLMProvider.is_arrearage_response(response):
            return True
        status = response.error_status_code
        kind = (response.error_kind or "").lower()
        error_type = (response.error_type or "").lower()
        code = (response.error_code or "").lower()
        text = (response.content or "").lower()
        structured_values = (kind, error_type, code)

        if _oauth_auth_rejected(status or 0, {"error": {"code": code}}):
            return True
        if kind in _AUTHENTICATION_ERROR_KINDS:
            return True
        if any(token in value for value in structured_values for token in _AUTHENTICATION_ERROR_TOKENS):
            return True
        if kind in _NON_FALLBACK_ERROR_KINDS:
            return False
        if any(token in value for value in structured_values for token in _NON_FALLBACK_ERROR_KINDS):
            return False
        if status in {401, 403}:
            return True
        if any(token in text for token in _AUTHENTICATION_ERROR_TOKENS):
            return True
        if response.error_should_retry is False:
            return False
        if status in {400, 404, 422}:
            return False
        if response.error_should_retry is True:
            return True
        if status is not None and (status in {408, 409, 429} or 500 <= status <= 599):
            return True
        if kind in _FALLBACK_ERROR_KINDS:
            return True
        return any(token in value for value in (kind, error_type, code, text) for token in _FALLBACK_ERROR_TOKENS)
