"""Fallback models: config, FallbackProvider, factory wiring (layer 20, task 20.3).

Adapted from nanobot tests/agent/test_runner_fallback.py. Left out: the tests
of ``ProviderCallContext`` (conversation state, native compaction windows,
response presets), ``RetryStatusEvent`` sequences, provider signatures and
``GenerationSettings``, which NanoClaw's providers don't have.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
from loguru import logger

from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.model_runtime import ModelRuntimeResolver
from nanoclaw.agent.runner import publish_fallback_model
from nanoclaw.bus.outbound_events import TurnModelUpdatedEvent
from nanoclaw.cli.terminal import progress_event_text
from nanoclaw.config.schema import AgentDefaults, Config, InlineFallbackConfig, ModelPresetConfig
from nanoclaw.events import EventSink
from nanoclaw.providers import FakeProvider
from nanoclaw.providers import fallback_provider as fallback_module
from nanoclaw.providers.base import LLMProvider, LLMResponse
from nanoclaw.providers.factory import (
    _inline_fallback_preset,
    build_preset_runtime,
    chain_context_window_tokens,
    make_provider,
)
from nanoclaw.providers.fallback_provider import FallbackModelSelection, FallbackProvider
from nanoclaw.session import SessionManager
from nanoclaw.utils.llm_runtime import LLMRuntime

SLEEP = "nanoclaw.providers.base.asyncio.sleep"


def _make_response(
    content: str | None = "ok",
    finish_reason: str = "stop",
    *,
    error_kind: str | None = None,
    error_status_code: int | None = None,
    error_type: str | None = None,
    error_code: str | None = None,
    error_should_retry: bool | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        finish_reason=finish_reason,
        error_kind=error_kind,
        error_status_code=error_status_code,
        error_type=error_type,
        error_code=error_code,
        error_should_retry=error_should_retry,
    )


def _error_response(content: str = "api error") -> LLMResponse:
    return _make_response(content, finish_reason="error", error_kind="server_error")


def _retryable_error(content: str = "") -> LLMResponse:
    return _make_response(content, finish_reason="error", error_status_code=503)


def _fallback(
    model: str,
    provider: str = "custom",
    *,
    max_tokens: int = 8192,
    context_window_tokens: int = 65_536,
    temperature: float = 0.1,
    reasoning_effort: str | None = None,
) -> ModelPresetConfig:
    return ModelPresetConfig(
        model=model,
        provider=provider,
        max_tokens=max_tokens,
        context_window_tokens=context_window_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )


class _FakeProvider(LLMProvider):
    """Fake provider for testing (nanobot test_runner_fallback.py:74)."""

    def __init__(
        self,
        name: str = "fake",
        response: LLMResponse | None = None,
        *,
        responses: list[LLMResponse] | None = None,
    ):
        super().__init__(default_model=f"{name}/model")
        self.provider_name = name
        self.name = name
        self._response = response or _make_response()
        self._responses = iter(responses) if responses is not None else None
        self.chat_calls: list[dict[str, Any]] = []
        self.chat_stream_calls: list[dict[str, Any]] = []

    def _next_response(self) -> LLMResponse:
        return next(self._responses) if self._responses is not None else self._response

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.chat_calls.append(dict(kwargs))
        return self._next_response()

    async def chat_stream(self, **kwargs: Any) -> LLMResponse:
        self.chat_stream_calls.append(dict(kwargs))
        response = self._next_response()
        on_delta = kwargs.get("on_content_delta")
        if on_delta and response.content:
            await on_delta(response.content)
        return response


class _RaisingProvider(LLMProvider):
    """Provider whose chat/chat_stream raise, like an auth/setup failure."""

    def __init__(self, name: str = "raiser", exc: BaseException | None = None):
        super().__init__(default_model=f"{name}/model")
        self.provider_name = name
        self._exc = exc if exc is not None else RuntimeError("GitHub Copilot is not logged in.")

    async def chat(self, **kwargs: Any) -> LLMResponse:
        raise self._exc

    async def chat_stream(self, **kwargs: Any) -> LLMResponse:
        raise self._exc


class _StreamingThenRaisingProvider(_RaisingProvider):
    async def chat_stream(self, **kwargs: Any) -> LLMResponse:
        on_content_delta = kwargs.get("on_content_delta")
        if on_content_delta:
            await on_content_delta("partial")
        raise self._exc


HI = [{"role": "user", "content": "hi"}]


# -- config-level tests --


def test_fallback_models_default_empty() -> None:
    assert AgentDefaults().fallback_models == []


def test_fallback_models_accept_preset_refs_and_inline_configs() -> None:
    config = Config.model_validate({
        "agents": {
            "defaults": {
                "fallbackModels": [
                    "deep",
                    {"provider": "openai", "model": "gpt-4.1", "maxTokens": 4096},
                ]
            }
        },
        "modelPresets": {"deep": {"provider": "anthropic", "model": "claude-opus-4-7"}},
    })

    assert config.agents.defaults.fallback_models[0] == "deep"
    assert config.agents.defaults.fallback_models[1] == InlineFallbackConfig(
        provider="openai", model="gpt-4.1", max_tokens=4096
    )


def test_fallback_models_serialize_with_camel_case_names() -> None:
    config = Config.model_validate({
        "agents": {"defaults": {"fallbackModels": [{"provider": "openai", "model": "gpt-4.1", "maxTokens": 4096}]}},
    })
    dumped = config.model_dump(mode="json", by_alias=True)["agents"]["defaults"]["fallbackModels"]
    assert dumped[0]["maxTokens"] == 4096
    assert dumped[0]["contextWindowTokens"] is None


def test_fallback_model_preset_ref_must_exist() -> None:
    with pytest.raises(ValueError, match="fallback_models.*not found"):
        Config.model_validate({"agents": {"defaults": {"fallbackModels": ["missing"]}}, "modelPresets": {}})


def _two_provider_config(**defaults: Any) -> Config:
    return Config.model_validate({
        "agents": {"defaults": {"modelPreset": "fast", "fallbackModels": ["deep"], **defaults}},
        "modelPresets": {
            "fast": {"model": "openai/gpt-4.1", "provider": "openai", "contextWindowTokens": 128000},
            "deep": {"model": "deepseek-chat", "provider": "deepseek", "contextWindowTokens": 64000},
        },
        "providers": {"openai": {"apiKey": "primary-key"}, "deepseek": {"apiKey": "fallback-key"}},
    })


def test_preset_runtime_uses_smallest_fallback_context_window() -> None:
    """nanobot test_provider_snapshot_uses_smallest_fallback_context_window."""
    config = _two_provider_config()
    runtime = build_preset_runtime(config)

    assert runtime.context_window_tokens == 64000
    assert isinstance(runtime.provider, FallbackProvider)
    assert runtime.provider._primary_context_window_tokens == 128000
    assert chain_context_window_tokens(config, config.resolve_preset()) == 64000


def test_make_loop_uses_the_chain_context_window(tmp_path) -> None:
    from nanoclaw.__main__ import _make_loop

    config = _two_provider_config()
    loop = _make_loop(FakeProvider(), tmp_path, config=config)
    assert loop.context_window_tokens == 64000


def test_factory_injects_configured_identity_into_primary_and_fallback_leaves() -> None:
    config = Config.model_validate({
        "agents": {"defaults": {"modelPreset": "primary", "fallbackModels": ["backup"]}},
        "modelPresets": {
            "primary": {"model": "primary-model", "provider": "primary_edge"},
            "backup": {"model": "backup-model", "provider": "backup_edge"},
        },
        "providers": {
            "primary_edge": {"apiKey": "primary-key", "apiBase": "https://primary.example/v1"},
            "backup_edge": {"apiKey": "backup-key", "apiBase": "https://backup.example/v1"},
        },
    })

    provider = make_provider(config)

    assert isinstance(provider, FallbackProvider)
    assert provider._primary.provider_name == "primary_edge"
    assert provider.provider_name == "primary_edge"
    fallback = provider._provider_factory(provider._fallback_presets[0])
    assert fallback.provider_name == "backup_edge"
    assert fallback.api_base == "https://backup.example/v1"
    assert provider._fallback_preset_names == ("backup",)


def test_fallback_may_use_a_different_backend() -> None:
    pytest.importorskip("anthropic")
    config = Config.model_validate({
        "agents": {"defaults": {"model": "gpt-4.1", "provider": "openai",
                                "fallbackModels": [{"provider": "anthropic", "model": "claude-sonnet-4-6"}]}},
        "providers": {"openai": {"apiKey": "k1"}, "anthropic": {"apiKey": "k2"}},
    })
    provider = make_provider(config)
    assert isinstance(provider, FallbackProvider)
    fallback = provider._provider_factory(provider._fallback_presets[0])
    assert type(fallback).__name__ == "AnthropicProvider"
    assert type(provider._primary).__name__ == "OpenAICompatProvider"


def test_no_fallback_models_returns_the_plain_provider() -> None:
    config = Config.model_validate({"providers": {"openai": {"apiKey": "k"}}})
    assert not isinstance(make_provider(config), FallbackProvider)


def test_fallback_setup_errors_surface_only_when_the_fallback_is_needed() -> None:
    config = Config.model_validate({
        "agents": {"defaults": {"provider": "openai", "fallbackModels": [{"provider": "deepseek", "model": "x"}]}},
        "providers": {"openai": {"apiKey": "k"}},
    })
    provider = make_provider(config)  # no deepseek key: built lazily
    assert isinstance(provider, FallbackProvider)
    with pytest.raises(ValueError, match="No API key"):
        provider._provider_factory(provider._fallback_presets[0])


def test_inline_fallback_inherits_limits_but_not_reasoning_effort() -> None:
    primary = ModelPresetConfig(model="p", provider="anthropic", max_tokens=1000,
                                context_window_tokens=50_000, temperature=0.3, reasoning_effort="high")
    inline = _inline_fallback_preset(primary, InlineFallbackConfig(provider="openai", model="gpt-4.1"))
    assert (inline.max_tokens, inline.context_window_tokens, inline.temperature) == (1000, 50_000, 0.3)
    assert inline.reasoning_effort is None
    own = _inline_fallback_preset(
        primary,
        InlineFallbackConfig(provider="openai", model="gpt-4.1", max_tokens=5, context_window_tokens=6,
                             temperature=0.9, reasoning_effort="low", extra_headers={"X": "1"}),
    )
    assert (own.max_tokens, own.context_window_tokens, own.temperature, own.reasoning_effort) == (5, 6, 0.9, "low")
    assert own.extra_headers == {"X": "1"}


# -- FallbackProvider tests --


async def test_no_fallback_when_primary_succeeds() -> None:
    primary = _FakeProvider("primary", _make_response("primary ok"))
    factory = MagicMock()
    fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

    result = await fb.chat(messages=HI)
    assert result.content == "primary ok"
    assert result.finish_reason == "stop"
    assert result.is_fallback is False
    factory.assert_not_called()


class TestFallbackWhenPrimaryRaises:
    @pytest.mark.parametrize(
        "exc",
        [TimeoutError(), httpx.ReadTimeout(""), httpx.ConnectError(""), httpx.ReadError("")],
        ids=["asyncio-timeout", "httpx-timeout", "connection", "httpx-network-error"],
    )
    async def test_transient_exception_triggers_fallback(self, exc: Exception) -> None:
        primary = _RaisingProvider("primary", exc)
        fallback = _FakeProvider("fallback", _make_response("fallback ok"))
        factory = MagicMock(return_value=fallback)
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

        result = await fb.chat(messages=HI, model="primary-model")

        assert result.content == "fallback ok"
        assert result.finish_reason == "stop"
        factory.assert_called_once_with(_fallback("fallback-a"))

    async def test_primary_exception_triggers_fallback(self) -> None:
        primary = _RaisingProvider("primary")
        fallback = _FakeProvider("fallback", _make_response("fallback ok"))
        factory = MagicMock(return_value=fallback)
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

        result = await fb.chat(messages=HI, model="primary-model")
        assert result.content == "fallback ok"
        factory.assert_called_once_with(_fallback("fallback-a"))

    @pytest.mark.parametrize("stream", [False, True], ids=["chat", "stream"])
    async def test_retry_entry_point_preserves_transient_exception_metadata(self, stream: bool) -> None:
        primary = _RaisingProvider("primary", TimeoutError())
        fallback = _FakeProvider("fallback", _make_response("fallback ok"))
        factory = MagicMock(return_value=fallback)
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

        retry = fb.chat_stream_with_retry if stream else fb.chat_with_retry
        with patch(SLEEP, new_callable=AsyncMock) as sleep:
            result = await retry(messages=HI, model="primary-model")

        assert result.content == "fallback ok"
        factory.assert_called_once_with(_fallback("fallback-a"))
        assert sleep.await_args_list == [call(1), call(2), call(4)]

    async def test_authentication_exception_message_is_classified(self) -> None:
        primary = _RaisingProvider("primary")
        response, exception = await FallbackProvider._call_provider(lambda p, kw: p.chat(**kw), primary, {})
        assert exception is primary._exc
        assert response.error_kind == "authentication"

    async def test_non_string_exception_metadata_is_normalized_before_fallback(self) -> None:
        class NumericMetadataError(Exception):
            error_type = 429
            error_code = 429
            status_code = 429

        primary = _RaisingProvider("primary", NumericMetadataError("rate limited"))
        response, exception = await FallbackProvider._call_provider(lambda p, kw: p.chat(**kw), primary, {})

        assert exception is primary._exc
        assert response.error_type == "429"
        assert response.error_code == "429"
        assert FallbackProvider._should_fallback(response) is True

    @pytest.mark.parametrize(
        "exc",
        [ValueError("unexpected response shape"), PermissionError("local provider file access failed")],
        ids=["value-error", "local-permission-error"],
    )
    async def test_non_fallbackable_primary_exception_does_not_trigger_fallback(self, exc: Exception) -> None:
        primary = _RaisingProvider("primary", exc)
        factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback")))
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

        result = await fb.chat(messages=HI, model="primary-model")

        assert result.finish_reason == "error"
        assert str(exc) in (result.content or "")
        factory.assert_not_called()

    async def test_fallback_exception_advances_to_next_fallback(self) -> None:
        primary = _FakeProvider("primary", _error_response())
        factory = MagicMock(side_effect=[_RaisingProvider("fb1"), _FakeProvider("fb2", _make_response("second ok"))])
        fb = FallbackProvider(
            primary=primary,
            fallback_presets=[_fallback("fallback-a"), _fallback("fallback-b")],
            provider_factory=factory,
        )

        result = await fb.chat(messages=HI, model="primary-model")
        assert result.content == "second ok"
        assert result.finish_reason == "stop"

    async def test_primary_exception_stream_triggers_fallback(self) -> None:
        primary = _RaisingProvider("primary")
        factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok")))
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

        result = await fb.chat_stream(messages=HI, model="primary-model")
        assert result.content == "fallback ok"

    async def test_primary_exception_after_streaming_does_not_duplicate_output(self) -> None:
        primary = _StreamingThenRaisingProvider("primary")
        factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("duplicate")))
        deltas: list[str] = []
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

        async def collect_delta(text: str) -> None:
            deltas.append(text)

        result = await fb.chat_stream(messages=HI, model="primary-model", on_content_delta=collect_delta)

        assert deltas == ["partial"]
        assert result.finish_reason == "error"
        factory.assert_not_called()

    async def test_primary_cancellation_is_not_converted_to_failover(self) -> None:
        primary = _RaisingProvider("primary", asyncio.CancelledError())
        factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback")))
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

        with pytest.raises(asyncio.CancelledError):
            await fb.chat(messages=HI)
        factory.assert_not_called()


class TestFallbackOnPrimaryError:
    async def test_first_fallback_succeeds(self) -> None:
        primary = _FakeProvider("primary", _error_response())
        fallback = _FakeProvider("fallback", _make_response("fallback ok"))
        factory = MagicMock(return_value=fallback)
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

        result = await fb.chat(messages=HI, model="primary-model")
        assert result.content == "fallback ok"
        assert result.is_fallback is True
        factory.assert_called_once_with(_fallback("fallback-a"))
        assert primary.chat_calls[0]["model"] == "primary-model"
        assert fallback.chat_calls[0]["model"] == "fallback-a"

    async def test_reports_only_the_successful_fallback_model(self) -> None:
        primary = _FakeProvider("primary", _error_response())
        failed_fallback = _FakeProvider("failed", _error_response("backup overloaded"))
        successful_fallback = _FakeProvider("fallback", _make_response("fallback ok"))
        fallback_models: list[str] = []

        async def _observe(selection: FallbackModelSelection) -> None:
            fallback_models.append(selection.model)
            assert selection.reauth_provider is None

        fb = FallbackProvider(
            primary=primary,
            fallback_presets=[_fallback("fallback-a", provider="backup"), _fallback("fallback-b", provider="backup")],
            provider_factory=MagicMock(side_effect=[failed_fallback, successful_fallback]),
            fallback_model_observer=_observe,
        )

        with patch(SLEEP, new_callable=AsyncMock) as sleep:
            result = await fb.chat_with_retry(messages=HI, model="primary-model")

        assert result.content == "fallback ok"
        assert fallback_models == ["fallback-b"]
        assert sleep.await_args_list == [call(1), call(2), call(4)]

    @pytest.mark.parametrize(("provider_name", "status", "code", "kind", "expected"), [
        ("openai_codex", 401, "token_revoked", "http", "openai_codex"),
        ("xai_grok", 401, None, "http", "xai_grok"),
        ("github_copilot", 401, None, "http", "github_copilot"),
        ("openai_codex", 400, "invalid_grant", "http", "openai_codex"),
        ("openai_codex", 403, "token_revoked", "http", "openai_codex"),
        ("openai_codex", None, None, "oauth_auth_required", "openai_codex"),
        ("openai_codex", 403, "insufficient_scope", "permission", None),
        ("openai_codex", 429, None, "rate_limit", None),
        ("openai_codex", 503, None, "server_error", None),
        ("openai_codex", None, None, "timeout", None),
        ("openai_codex", None, None, "authentication", None),
        ("openai", 401, "invalid_api_key", "authentication", None),
    ])
    async def test_reports_only_confirmed_oauth_rejections(
        self, monkeypatch, provider_name, status, code, kind, expected,
    ) -> None:
        # NanoClaw's registry has no OAuth providers; stand them in.
        oauth = {"openai_codex", "xai_grok", "github_copilot"}
        monkeypatch.setattr(
            fallback_module, "find_by_name",
            lambda name: SimpleNamespace(name=name, is_oauth=name in oauth),
        )
        observer = AsyncMock()
        primary = _FakeProvider(provider_name, _make_response(
            "synthetic-secret invalid_token", "error", error_status_code=status, error_code=code, error_kind=kind,
        ))
        provider = FallbackProvider(
            primary=primary, fallback_presets=[_fallback("backup")],
            provider_factory=MagicMock(return_value=_FakeProvider("backup")),
            fallback_model_observer=observer,
        )
        await provider.chat(messages=HI)
        observer.assert_awaited_once_with(FallbackModelSelection("backup", expected))
        assert "synthetic-secret" not in repr(observer.await_args)

    async def test_no_reauth_provider_for_nanoclaw_registry_providers(self) -> None:
        observer = AsyncMock()
        provider = FallbackProvider(
            primary=_FakeProvider("openai", _make_response("bad", "error", error_status_code=401)),
            fallback_presets=[_fallback("backup")],
            provider_factory=MagicMock(return_value=_FakeProvider("backup")),
            fallback_model_observer=observer,
        )
        await provider.chat(messages=HI)
        observer.assert_awaited_once_with(FallbackModelSelection("backup", None))

    async def test_does_not_reuse_auth_diagnosis_when_circuit_skips_primary(self, monkeypatch) -> None:
        monkeypatch.setattr(fallback_module, "find_by_name", lambda name: SimpleNamespace(name=name, is_oauth=True))
        observer = AsyncMock()
        provider = FallbackProvider(
            primary=_FakeProvider("openai_codex", _make_response("revoked", "error", error_status_code=401)),
            fallback_presets=[_fallback("backup")],
            provider_factory=MagicMock(return_value=_FakeProvider("backup")),
            fallback_model_observer=observer,
        )
        for _ in range(4):
            await provider.chat(messages=HI)
        assert [c.args[0].reauth_provider for c in observer.await_args_list] == [
            "openai_codex", "openai_codex", "openai_codex", None,
        ]

    async def test_no_reauth_notice_when_all_fallbacks_fail(self) -> None:
        observer = AsyncMock()
        provider = FallbackProvider(
            primary=_FakeProvider("openai_codex", _make_response("revoked", "error", error_status_code=401)),
            fallback_presets=[_fallback("backup")],
            provider_factory=MagicMock(return_value=_FakeProvider("backup", _error_response())),
            fallback_model_observer=observer,
        )
        result = await provider.chat(messages=HI)
        assert result.finish_reason == "error"
        observer.assert_not_awaited()

    async def test_observer_failure_does_not_fail_the_request(self) -> None:
        provider = FallbackProvider(
            primary=_FakeProvider("primary", _error_response()),
            fallback_presets=[_fallback("backup")],
            provider_factory=MagicMock(return_value=_FakeProvider("backup", _make_response("ok"))),
            fallback_model_observer=AsyncMock(side_effect=RuntimeError("boom")),
        )
        assert (await provider.chat(messages=HI)).content == "ok"

    async def test_logs_primary_error_before_fallback(self) -> None:
        primary = _FakeProvider("primary", _error_response("primary overloaded"))
        factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok")))
        logs: list[str] = []
        logger.enable("nanoclaw")  # NanoClaw's logger is off outside the CLI
        sink_id = logger.add(lambda message: logs.append(str(message)), format="{message}")
        try:
            fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)
            await fb.chat(messages=HI, model="primary-model")
        finally:
            logger.remove(sink_id)
            logger.disable("nanoclaw")

        assert any(
            "Primary model 'primary-model' failed: primary overloaded; trying fallback 'fallback-a'" in line
            for line in logs
        )


async def test_non_timeout_error_after_streamed_content_skips_failover() -> None:
    primary = _FakeProvider("primary", _error_response())
    factory = MagicMock()
    fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

    async def _delta(text: str) -> None:
        pass

    result = await fb.chat_stream(messages=HI, on_content_delta=_delta)
    assert result.finish_reason == "error"
    factory.assert_not_called()


async def test_timeout_with_streamed_content_falls_back() -> None:
    primary = _FakeProvider("primary", _make_response("stream stalled", finish_reason="error", error_kind="timeout"))
    factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok")))
    fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)
    streamed: list[str] = []
    recoveries: list[str] = []

    async def _delta(text: str) -> None:
        streamed.append(text)

    async def _recover() -> None:
        recoveries.append("recover")

    result = await fb.chat_stream(messages=HI, on_content_delta=_delta, on_stream_recover=_recover)
    assert result.content == "fallback ok"
    factory.assert_called_once_with(_fallback("fallback-a"))
    assert streamed == ["stream stalled", "fallback ok"]
    assert recoveries == ["recover"]


async def test_timeout_with_streamed_content_without_recover_silences_the_fallback() -> None:
    primary = _FakeProvider("primary", _make_response("stream stalled", finish_reason="error", error_kind="timeout"))
    factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok")))
    fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)
    streamed: list[str] = []

    async def _delta(text: str) -> None:
        streamed.append(text)

    result = await fb.chat_stream(messages=HI, on_content_delta=_delta)
    assert result.content == "fallback ok"
    assert streamed == ["stream stalled"]


class TestFailoverOnEmptyChoices:
    def test_empty_choices_text_fallback(self) -> None:
        response = _make_response("Error: API returned empty choices.", finish_reason="error", error_kind="empty")
        assert FallbackProvider._should_fallback(response)

    def test_empty_choices_no_error_kind_text_fallback(self) -> None:
        response = _make_response("Error: API returned empty choices.", finish_reason="error")
        assert FallbackProvider._should_fallback(response)

    async def test_empty_choices_triggers_failover(self) -> None:
        primary = _FakeProvider("primary", _make_response("Error: API returned empty choices.", finish_reason="error"))
        factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok")))
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

        result = await fb.chat(messages=HI)
        assert result.content == "fallback ok"
        factory.assert_called_once()


async def test_rate_limit_fails_over() -> None:
    primary = _FakeProvider("primary", _error_response("rate limit exceeded"))
    factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok")))
    fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

    result = await fb.chat(messages=HI)
    assert result.content == "fallback ok"
    factory.assert_called_once_with(_fallback("fallback-a"))


class TestRetryBeforeFailover:
    @pytest.mark.parametrize("stream", [False, True])
    async def test_retry_waits_are_reported_but_not_candidate_exhaustion(self, stream: bool) -> None:
        """nanobot test_scoped_retry_events_preserve_fallback_chain_ownership (waits only)."""
        primary = _FakeProvider("primary", _retryable_error("primary unavailable"))
        provider = FallbackProvider(primary, [_fallback("fallback-a")],
                                    MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok"))))
        notices: list[str] = []

        async def on_wait(text: str) -> None:
            notices.append(text)

        retry = provider.chat_stream_with_retry if stream else provider.chat_with_retry
        with patch(SLEEP, new_callable=AsyncMock):
            result = await retry(HI, on_retry_wait=on_wait)
        assert result.content == "fallback ok"
        assert len(notices) == 3
        assert not any("giving up" in text for text in notices)

    async def test_persistent_retry_includes_chain_wait_and_one_terminal(self) -> None:
        primary = _FakeProvider("primary", _retryable_error("primary unavailable"))
        fallback = _FakeProvider("fallback", _retryable_error("fallback unavailable"))
        provider = FallbackProvider(primary, [_fallback("fallback-a")], MagicMock(return_value=fallback))
        provider._PERSISTENT_IDENTICAL_ERROR_LIMIT = 2
        notices: list[str] = []

        async def on_wait(text: str) -> None:
            notices.append(text)

        with patch(SLEEP, new_callable=AsyncMock):
            result = await provider.chat_with_retry(HI, retry_mode="persistent", on_retry_wait=on_wait)
        assert result.finish_reason == "error"
        # Two candidates, three waits each, for two chains, plus one chain wait.
        assert len(notices) == 14
        assert sum("Persistent retry stopped" in text for text in notices) == 1
        assert not any("giving up" in text for text in notices)

    @pytest.mark.parametrize("retry_mode", ["standard", "persistent"])
    async def test_primary_recovers_before_fallback(self, retry_mode: str) -> None:
        primary = _FakeProvider("primary", responses=[_error_response("rate limited"), _make_response("primary ok")])
        factory = MagicMock()
        provider = FallbackProvider(primary, [_fallback("fallback-a")], factory)

        with patch(SLEEP, new_callable=AsyncMock):
            result = await provider.chat_with_retry(HI, retry_mode=retry_mode)

        assert result.content == "primary ok"
        assert len(primary.chat_calls) == 2
        factory.assert_not_called()

    async def test_primary_exhausts_before_fallback_without_terminal_event(self) -> None:
        primary = _FakeProvider("primary", responses=[_retryable_error(f"attempt {n}") for n in range(4)])
        factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok")))
        retry_events = AsyncMock()
        terminal = AsyncMock()
        provider = FallbackProvider(primary, [_fallback("fallback-a")], factory)

        with patch(SLEEP, new_callable=AsyncMock):
            result = await provider.chat_with_retry(HI, on_retry_wait=retry_events, on_retry_exhausted=terminal)

        assert result.content == "fallback ok"
        assert len(primary.chat_calls) == 4
        assert not any("giving up" in c.args[0] for c in retry_events.await_args_list)
        terminal.assert_not_awaited()
        factory.assert_called_once_with(_fallback("fallback-a"))

    async def test_all_candidates_exhaust_emit_one_terminal_event(self) -> None:
        primary = _FakeProvider("primary", _retryable_error("primary unavailable"))
        fallback = _FakeProvider("fallback", _retryable_error("fallback unavailable"))
        retry_events = AsyncMock()
        terminal_event = AsyncMock()
        provider = FallbackProvider(primary, [_fallback("fallback-a")], MagicMock(return_value=fallback))

        with patch(SLEEP, new_callable=AsyncMock):
            result = await provider.chat_with_retry(HI, on_retry_wait=retry_events, on_retry_exhausted=terminal_event)

        assert result.finish_reason == "error"
        assert len(primary.chat_calls) == len(fallback.chat_calls) == 4
        assert not any("giving up" in c.args[0] for c in retry_events.await_args_list)
        terminal_event.assert_awaited_once_with("Model request failed after 4 attempts, giving up.")

    async def test_terminal_event_defaults_to_the_wait_callback(self) -> None:
        primary = _FakeProvider("primary", _retryable_error("primary unavailable"))
        fallback = _FakeProvider("fallback", _retryable_error("fallback unavailable"))
        notices: list[str] = []

        async def on_wait(text: str) -> None:
            notices.append(text)

        provider = FallbackProvider(primary, [_fallback("fallback-a")], MagicMock(return_value=fallback))
        with patch(SLEEP, new_callable=AsyncMock):
            await provider.chat_with_retry(HI, on_retry_wait=on_wait)
        assert [text for text in notices if "giving up" in text] == ["Model request failed after 4 attempts, giving up."]
        assert notices[-1].endswith("giving up.")

    async def test_fallback_auth_error_preserves_primary_diagnosis_and_exhaustion(self) -> None:
        primary = _FakeProvider("primary", _make_response(
            "private network detail", finish_reason="error", error_kind="connection",
        ))
        fallback = _FakeProvider("fallback", _make_response(
            "private auth detail", finish_reason="error", error_status_code=401,
        ))
        terminal = AsyncMock()
        provider = FallbackProvider(primary, [_fallback("backup")], MagicMock(return_value=fallback))

        with patch(SLEEP, new_callable=AsyncMock):
            result = await provider.chat_with_retry(HI, on_retry_exhausted=terminal)

        assert len(primary.chat_calls) == 4
        assert len(fallback.chat_calls) == 1
        assert result.primary_error_kind == "connection"
        assert LLMProvider.public_error_kind(result) == "connection"
        assert not LLMProvider.is_transient_response(result)  # Retry policy is still terminal.
        terminal.assert_awaited_once_with("Model request failed after 4 attempts, giving up.")

    @pytest.mark.parametrize("factory_fails", [False, True])
    async def test_persistent_mode_repeats_the_whole_chain(self, factory_fails: bool) -> None:
        primary = _FakeProvider("primary", _retryable_error("primary unavailable"))
        fallback = _FakeProvider("fallback", _retryable_error("fallback unavailable"))
        factory = (
            MagicMock(side_effect=ValueError("missing fallback credentials"))
            if factory_fails
            else MagicMock(return_value=fallback)
        )
        terminal_event = AsyncMock()
        provider = FallbackProvider(primary, [_fallback("fallback-a")], factory)
        provider._PERSISTENT_IDENTICAL_ERROR_LIMIT = 2

        with patch(SLEEP, new_callable=AsyncMock):
            result = await provider.chat_with_retry(HI, retry_mode="persistent", on_retry_exhausted=terminal_event)

        assert result.finish_reason == "error"
        assert len(primary.chat_calls) == 8
        assert len(fallback.chat_calls) == (0 if factory_fails else 8)
        assert factory.call_count == 2
        terminal_event.assert_awaited_once_with("Persistent retry stopped after 2 identical errors.")

    async def test_open_primary_circuit_remains_retryable_when_factory_fails(self) -> None:
        primary = _FakeProvider("primary")
        factory = MagicMock(side_effect=ValueError("missing fallback credentials"))
        terminal_event = AsyncMock()
        provider = FallbackProvider(primary, [_fallback("fallback-a")], factory)
        provider._primary_tripped_at = 100.0
        provider._PERSISTENT_IDENTICAL_ERROR_LIMIT = 2

        with (
            patch("nanoclaw.providers.fallback_provider.time.monotonic", return_value=100.0),
            patch(SLEEP, new_callable=AsyncMock),
        ):
            result = await provider.chat_with_retry(HI, retry_mode="persistent", on_retry_exhausted=terminal_event)

        assert result.error_should_retry is True
        assert result.error_retry_after_s == 60
        assert result.content == "Primary model 'primary/model' circuit open and no fallbacks available"
        assert primary.chat_calls == []
        assert factory.call_count == 2
        terminal_event.assert_awaited_once_with("Persistent retry stopped after 2 identical errors.")

    async def test_fallback_retries_before_trying_next_model(self) -> None:
        primary = _FakeProvider("primary", _make_response(
            "unauthorized", finish_reason="error", error_kind="authentication", error_should_retry=False,
        ))
        fallback_a = _FakeProvider("fallback-a", responses=[_error_response("rate limited"), _make_response("a ok")])
        factory = MagicMock(side_effect=[fallback_a, _FakeProvider("fallback-b")])
        fallback_a_preset = _fallback("fallback-a")
        provider = FallbackProvider(primary, [fallback_a_preset, _fallback("fallback-b")], factory)

        with patch(SLEEP, new_callable=AsyncMock):
            result = await provider.chat_with_retry(HI)

        assert result.content == "a ok"
        assert len(fallback_a.chat_calls) == 2
        factory.assert_called_once_with(fallback_a_preset)

    async def test_stream_recovery_keeps_fallback_eligible(self) -> None:
        primary = _FakeProvider("primary", responses=[
            _make_response("partial", finish_reason="error", error_kind="timeout"),
            *[_retryable_error() for _ in range(3)],
        ])
        factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok")))
        streamed = AsyncMock()
        recovered = AsyncMock()
        provider = FallbackProvider(primary, [_fallback("fallback-a")], factory)

        with patch(SLEEP, new_callable=AsyncMock):
            result = await provider.chat_stream_with_retry(HI, on_content_delta=streamed, on_stream_recover=recovered)

        assert result.content == "fallback ok"
        assert len(primary.chat_stream_calls) == 4
        assert [c.args[0] for c in streamed.await_args_list] == ["partial", "fallback ok"]
        recovered.assert_awaited_once_with()
        factory.assert_called_once_with(_fallback("fallback-a"))

    async def test_stream_without_delta_callback_retries_before_fallback(self) -> None:
        primary = _FakeProvider("primary", responses=[_retryable_error(f"attempt {n}") for n in range(4)])
        factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok")))
        provider = FallbackProvider(primary, [_fallback("fallback-a")], factory)

        with patch(SLEEP, new_callable=AsyncMock):
            result = await provider.chat_stream_with_retry(HI)

        assert result.content == "fallback ok"
        assert len(primary.chat_stream_calls) == 4
        factory.assert_called_once_with(_fallback("fallback-a"))

    async def test_unrecovered_stream_keeps_non_timeout_fallback_blocked(self) -> None:
        primary = _FakeProvider("primary", responses=[
            _make_response("partial", finish_reason="error", error_kind="timeout"),
            _retryable_error(),
            _retryable_error(),
            _retryable_error("last error"),
        ])
        factory = MagicMock()
        streamed = AsyncMock()
        provider = FallbackProvider(primary, [_fallback("fallback-a")], factory)

        with patch(SLEEP, new_callable=AsyncMock):
            result = await provider.chat_stream_with_retry(HI, on_content_delta=streamed)

        assert result.content == "last error"
        streamed.assert_awaited_once_with("partial")
        factory.assert_not_called()


class TestFailoverOnArrearageError:
    async def test_non_retryable_quota_tries_configured_fallback(self) -> None:
        primary = _FakeProvider("primary", _make_response(
            "insufficient quota", finish_reason="error", error_status_code=429,
            error_type="insufficient_quota", error_should_retry=False,
        ))
        factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok")))
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

        assert (await fb.chat(messages=HI)).content == "fallback ok"
        factory.assert_called_once_with(_fallback("fallback-a"))

    async def test_insufficient_credits_on_400_tries_configured_fallback(self) -> None:
        primary = _FakeProvider("primary", _make_response(
            "[400]: You have insufficient credits to make this request.", finish_reason="error",
            error_status_code=400, error_code="BAD_REQUEST", error_type="invalid_request_error",
            error_should_retry=False,
        ))
        factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok")))
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

        assert (await fb.chat(messages=HI)).content == "fallback ok"

    async def test_plain_bad_request_still_does_not_fall_back(self) -> None:
        bad_request = _make_response(
            "invalid parameter: temperature must be <= 2", finish_reason="error", error_status_code=400,
            error_code="BAD_REQUEST", error_type="invalid_request_error", error_should_retry=False,
        )
        factory = MagicMock()
        fb = FallbackProvider(
            primary=_FakeProvider("primary", bad_request),
            fallback_presets=[_fallback("fallback-a")],
            provider_factory=factory,
        )

        assert (await fb.chat(messages=HI)) is bad_request
        factory.assert_not_called()

    async def test_without_fallback_presets_returns_original_error(self) -> None:
        arrearage = _make_response("payment required", finish_reason="error", error_status_code=402,
                                   error_should_retry=False)
        factory = MagicMock()
        fb = FallbackProvider(primary=_FakeProvider("primary", arrearage), fallback_presets=[], provider_factory=factory)

        assert (await fb.chat(messages=HI)) is arrearage
        factory.assert_not_called()


@pytest.mark.parametrize(
    "authentication_error",
    [
        pytest.param(_make_response(
            "Error: {'error': {'type': 'authentication_error', "
            "'message': 'The API Key appears to be invalid or may have expired.'}}",
            finish_reason="error", error_type="authentication_error", error_should_retry=False,
        ), id="authentication-error-type"),
        pytest.param(_make_response(
            "unauthorized", finish_reason="error", error_status_code=401, error_kind="http", error_should_retry=False,
        ), id="http-401"),
        pytest.param(_make_response(
            "bad key", finish_reason="error", error_type="invalid_request_error", error_code="invalid_api_key",
            error_should_retry=False,
        ), id="invalid-api-key-code"),
        pytest.param(_make_response(
            "credentials have expired", finish_reason="error", error_should_retry=False,
        ), id="expired-credentials-text"),
        pytest.param(_make_response(
            "permission denied", finish_reason="error", error_status_code=403, error_kind="permission",
            error_should_retry=False,
        ), id="permission-error"),
    ],
)
async def test_authentication_error_tries_configured_fallback(authentication_error: LLMResponse) -> None:
    factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok")))
    fb = FallbackProvider(
        primary=_FakeProvider("primary", authentication_error),
        fallback_presets=[_fallback("fallback-a")],
        provider_factory=factory,
    )

    assert (await fb.chat(messages=HI)).content == "fallback ok"
    factory.assert_called_once_with(_fallback("fallback-a"))


@pytest.mark.parametrize(
    "response",
    [
        _make_response("invalid request", finish_reason="error", error_status_code=400, error_kind="invalid_request"),
        _make_response("request blocked by content filter", finish_reason="error", error_status_code=403,
                       error_kind="content_filter", error_should_retry=False),
        _make_response("prompt is too long", finish_reason="error", error_kind="context_length"),
        _make_response("not found", finish_reason="error", error_status_code=404),
        _make_response("unprocessable", finish_reason="error", error_status_code=422),
    ],
    ids=["bad-request", "content-filter-over-403", "context-length", "404", "422"],
)
async def test_non_fallbackable_errors_are_returned(response: LLMResponse) -> None:
    factory = MagicMock()
    fb = FallbackProvider(
        primary=_FakeProvider("primary", response), fallback_presets=[_fallback("fallback-a")], provider_factory=factory
    )
    result = await fb.chat(messages=HI)
    assert result is response
    factory.assert_not_called()


async def test_timeout_triggers_fallback() -> None:
    primary = _FakeProvider("primary", _make_response("timed out", finish_reason="error", error_kind="timeout"))
    factory = MagicMock(return_value=_FakeProvider("fallback", _make_response("fallback ok")))
    fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=factory)

    assert (await fb.chat(messages=HI)).content == "fallback ok"


async def test_fallback_tries_models_in_order() -> None:
    primary = _FakeProvider("primary", _error_response("primary fail"))
    factory = MagicMock(side_effect=[_FakeProvider("a", _error_response("a fail")), _FakeProvider("b", _make_response("b ok"))])
    fb = FallbackProvider(
        primary=primary, fallback_presets=[_fallback("fallback-a"), _fallback("fallback-b")], provider_factory=factory
    )

    result = await fb.chat(messages=HI)
    assert result.content == "b ok"
    assert factory.call_args_list == [call(_fallback("fallback-a")), call(_fallback("fallback-b"))]


async def test_all_fallbacks_fail_returns_the_last_error_with_the_primary_kind() -> None:
    primary = _FakeProvider("primary", _retryable_error("primary fail"))
    fb = FallbackProvider(
        primary=primary,
        fallback_presets=[_fallback("fallback-a")],
        provider_factory=MagicMock(return_value=_FakeProvider("fallback", _error_response("all fail"))),
    )

    result = await fb.chat(messages=HI)
    assert result.finish_reason == "error"
    assert result.content == "all fail"
    assert result.is_fallback is True
    assert result.primary_error_kind == "server"


async def test_factory_exception_skips_model() -> None:
    factory = MagicMock(side_effect=[ValueError("no key"), _FakeProvider("b", _make_response("b ok"))])
    fb = FallbackProvider(
        primary=_FakeProvider("primary", _error_response()),
        fallback_presets=[_fallback("fallback-a"), _fallback("fallback-b")],
        provider_factory=factory,
    )

    assert (await fb.chat(messages=HI)).content == "b ok"
    assert factory.call_count == 2


async def test_fallback_uses_its_model_and_generation_fields() -> None:
    fallback = _FakeProvider("fallback", _make_response("ok"))
    fb = FallbackProvider(
        primary=_FakeProvider("primary", _error_response()),
        fallback_presets=[_fallback("fallback-model", max_tokens=1234, temperature=0.4)],
        provider_factory=MagicMock(return_value=fallback),
    )

    await fb.chat(messages=HI, model="primary-model", max_tokens=8192, temperature=0.1)

    assert fallback.chat_calls[0]["model"] == "fallback-model"
    assert fallback.chat_calls[0]["max_tokens"] == 1234
    assert fallback.chat_calls[0]["temperature"] == 0.4
    # NanoClaw providers take reasoning effort when built, not per call.
    assert "reasoning_effort" not in fallback.chat_calls[0]


class TestNoFallbackWhenEmptyList:
    async def test(self) -> None:
        factory = MagicMock()
        fb = FallbackProvider(primary=_FakeProvider("primary", _error_response()), fallback_presets=[],
                              provider_factory=factory)
        assert (await fb.chat(messages=HI)).finish_reason == "error"
        factory.assert_not_called()

    async def test_retry_entrypoints_delegate_to_primary(self) -> None:
        primary = _FakeProvider("primary")
        provider = FallbackProvider(primary, [], MagicMock())
        response = _make_response("primary ok")

        with (
            patch.object(primary, "chat_with_retry", new_callable=AsyncMock, return_value=response) as chat_retry,
            patch.object(primary, "chat_stream_with_retry", new_callable=AsyncMock, return_value=response) as stream_retry,
        ):
            assert (await provider.chat_with_retry([])) is response
            assert (await provider.chat_stream_with_retry([])) is response

        chat_retry.assert_awaited_once()
        stream_retry.assert_awaited_once()


async def test_chat_stream_failover() -> None:
    fb = FallbackProvider(
        primary=_FakeProvider("primary", _error_response("")),
        fallback_presets=[_fallback("fallback-a")],
        provider_factory=MagicMock(return_value=_FakeProvider("fallback", _make_response("stream ok"))),
    )
    result = await fb.chat_stream(messages=HI)
    assert result.content == "stream ok"


def test_default_model_and_identity_come_from_the_primary() -> None:
    primary = _FakeProvider("primary")
    fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("a")], provider_factory=MagicMock())
    assert fb.get_default_model() == "primary/model"
    assert fb.default_model == "primary/model"
    assert fb.provider_name == "primary"


def test_fallback_preset_names_must_match() -> None:
    with pytest.raises(ValueError, match="must match"):
        FallbackProvider(_FakeProvider(), [_fallback("a")], MagicMock(), fallback_preset_names=["a", "b"])


async def test_fallback_closes_temporary_provider_and_owns_primary_shutdown() -> None:
    primary = _FakeProvider("primary", _error_response())
    fallback = _FakeProvider("fallback", _make_response("fallback ok"))
    primary.aclose = AsyncMock()
    fallback.aclose = AsyncMock()
    wrapper = FallbackProvider(primary, [_fallback("backup")], MagicMock(return_value=fallback))

    result = await wrapper.chat(messages=HI)

    assert result.content == "fallback ok"
    fallback.aclose.assert_awaited_once()
    primary.aclose.assert_not_awaited()
    await wrapper.aclose()
    primary.aclose.assert_awaited_once()


async def test_failed_fallback_is_closed_too() -> None:
    fallback = _FakeProvider("fallback", _error_response())
    fallback.aclose = AsyncMock()
    wrapper = FallbackProvider(_FakeProvider("primary", _error_response()), [_fallback("b")],
                               MagicMock(return_value=fallback))
    await wrapper.chat(messages=HI)
    fallback.aclose.assert_awaited_once()


class TestCircuitBreaker:
    async def test_skips_primary_after_three_failures(self) -> None:
        primary = _FakeProvider("primary", _error_response())
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")],
                              provider_factory=MagicMock(return_value=_FakeProvider("fallback", _make_response("fb ok"))))

        for _ in range(3):
            assert (await fb.chat(messages=HI)).content == "fb ok"
        assert len(primary.chat_calls) == 3

        primary.chat_calls.clear()
        assert (await fb.chat(messages=HI)).content == "fb ok"
        assert len(primary.chat_calls) == 0

    async def test_resets_on_success(self) -> None:
        primary = _FakeProvider("primary", _error_response())
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")],
                              provider_factory=MagicMock(return_value=_FakeProvider("fallback", _make_response("fb ok"))))

        for _ in range(2):
            await fb.chat(messages=HI)
        primary._response = _make_response("primary ok")
        assert (await fb.chat(messages=HI)).content == "primary ok"

        primary._response = _error_response()
        primary.chat_calls.clear()
        assert (await fb.chat(messages=HI)).content == "fb ok"
        assert len(primary.chat_calls) == 1

    async def test_cooldown_reopens_primary_after_sixty_seconds(self, monkeypatch) -> None:
        clock = [1000.0]
        monkeypatch.setattr(fallback_module.time, "monotonic", lambda: clock[0])
        primary = _FakeProvider("primary", _error_response())
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")],
                              provider_factory=MagicMock(return_value=_FakeProvider("fallback", _make_response("fb ok"))))

        for _ in range(3):
            await fb.chat(messages=HI)
        assert fb._primary_tripped_at == 1000.0

        clock[0] = 1059.9  # still open
        await fb.chat(messages=HI)
        assert len(primary.chat_calls) == 3

        clock[0] = 1060.0  # half-open: one probe, which fails and re-trips
        await fb.chat(messages=HI)
        assert len(primary.chat_calls) == 4
        assert fb._primary_tripped_at == 1060.0

        clock[0] = 1061.0
        await fb.chat(messages=HI)
        assert len(primary.chat_calls) == 4

        clock[0] = 1120.0  # probe succeeds: circuit closes
        primary._response = _make_response("primary back")
        assert (await fb.chat(messages=HI)).content == "primary back"
        assert fb._primary_tripped_at is None and fb._primary_failures == 0

    async def test_non_fallbackable_primary_errors_do_not_trip(self) -> None:
        primary = _FakeProvider("primary", _make_response("bad", finish_reason="error", error_status_code=400))
        fb = FallbackProvider(primary=primary, fallback_presets=[_fallback("fallback-a")], provider_factory=MagicMock())
        for _ in range(5):
            await fb.chat(messages=HI)
        assert len(primary.chat_calls) == 5
        assert fb._primary_tripped_at is None


# -- reporting the model that answered --


async def test_resolver_attaches_the_fallback_observer() -> None:
    fb = FallbackProvider(_FakeProvider("primary"), [_fallback("a")], MagicMock())
    ModelRuntimeResolver(LLMRuntime(provider=fb, model=None, max_tokens=1, temperature=0.1, context_window_tokens=1))
    assert fb._fallback_model_observer is publish_fallback_model


def test_fallback_model_event_is_a_progress_line() -> None:
    assert progress_event_text(TurnModelUpdatedEvent(model="backup", fallback=True)) == "Using fallback model backup"
    assert progress_event_text(TurnModelUpdatedEvent(model="primary")) is None


async def test_end_to_end_primary_503_then_fallback_answers_and_session_is_saved(tmp_path) -> None:
    """The primary keeps returning 503; after its retries the fallback answers."""
    ws = tmp_path / "ws"
    ws.mkdir()
    unavailable = LLMResponse(content="Error: 503 service unavailable", finish_reason="error", error_status_code=503)
    primary = FakeProvider([unavailable] * 4, retries=True)
    backup = FakeProvider(["Answer from the fallback model."])
    built: list[ModelPresetConfig] = []

    def factory(preset: ModelPresetConfig) -> LLMProvider:
        built.append(preset)
        return backup

    provider = FallbackProvider(
        primary, [_fallback("backup-model", max_tokens=777, temperature=0.2)], factory
    )
    sessions = SessionManager(ws, sessions_root=tmp_path / "sessions")
    loop = AgentLoop(provider, ws, sessions=sessions, model="primary-model")
    events: list[Any] = []

    async def publish(event: Any) -> None:
        events.append(event)

    with patch(SLEEP, new_callable=AsyncMock) as sleep:
        result = await loop.process_direct("hello", events=EventSink(publish))
    await loop.aclose()

    assert result.final_content == "Answer from the fallback model."
    assert sleep.await_args_list == [call(1), call(2), call(4)]
    assert len(primary.calls) == 4 and all(c.model == "primary-model" for c in primary.calls)
    assert [(c.model, c.max_tokens, c.temperature) for c in backup.calls] == [("backup-model", 777, 0.2)]
    assert [p.model for p in built] == ["backup-model"]
    notices = [e for e in events if isinstance(e, TurnModelUpdatedEvent)]
    assert notices == [TurnModelUpdatedEvent(model="backup-model", fallback=True)]
    saved = SessionManager(ws, sessions_root=tmp_path / "sessions").get_or_create("cli:direct").messages
    assert [m["role"] for m in saved] == ["user", "assistant"]
    assert saved[-1]["content"] == "Answer from the fallback model."


async def test_end_to_end_all_models_fail_saves_the_error_turn(tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    unavailable = LLMResponse(content="Error: 503 service unavailable", finish_reason="error", error_status_code=503)
    provider = FallbackProvider(
        FakeProvider([unavailable]), [_fallback("backup-model")],
        lambda preset: FakeProvider([LLMResponse(content="backup down", finish_reason="error", error_status_code=502)]),
    )
    loop = AgentLoop(provider, ws, sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"))
    result = await loop.process_direct("hello")
    await loop.aclose()
    assert result.stop_reason == "error"


async def test_base_retry_reports_giving_up_to_on_retry_exhausted() -> None:
    provider = _FakeProvider("solo", _retryable_error("down"))
    waits, exhausted = AsyncMock(), AsyncMock()
    with patch(SLEEP, new_callable=AsyncMock):
        await provider.chat_with_retry(HI, on_retry_wait=waits, on_retry_exhausted=exhausted)
    exhausted.assert_awaited_once_with("Model request failed after 4 attempts, giving up.")
    assert not any("giving up" in c.args[0] for c in waits.await_args_list)


def test_fallback_model_event_reaches_user_turns_only() -> None:
    from nanoclaw.agent.turn_delivery import notification_is_deliverable

    assert notification_is_deliverable(TurnModelUpdatedEvent, channel="cli", publish_lifecycle=True)
    assert not notification_is_deliverable(TurnModelUpdatedEvent, channel="cli", publish_lifecycle=False)
