"""AnthropicProvider: ported from nanobot tests/providers/test_anthropic_*.py,
test_prompt_cache_markers.py, test_cached_tokens.py, test_provider_error_metadata.py,
test_provider_retry_after_hints.py and test_provider_sdk_retry_defaults.py.

No network: the SDK client is a MagicMock or patched out.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanoclaw.providers.anthropic_provider import AnthropicProvider
from nanoclaw.providers.base import LLMProvider, LLMResponse

IDLE_ENV = "NANOCLAW_STREAM_IDLE_TIMEOUT_S"


def _make_provider(model: str = "claude-sonnet-4-6") -> AnthropicProvider:
    with patch("anthropic.AsyncAnthropic"):
        return AnthropicProvider(api_key="sk-test", default_model=model)


def _mock_client_provider() -> AnthropicProvider:
    provider = AnthropicProvider(api_key="sk-test")
    provider._client = MagicMock()
    return provider


# --- thinking / reasoning_effort (test_anthropic_thinking.py) -------------------


def _build(provider: AnthropicProvider, reasoning_effort: str | None, **overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = dict(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=reasoning_effort,
        tool_choice=None,
        supports_caching=False,
    )
    defaults.update(overrides)
    return provider._build_kwargs(**defaults)


def test_adaptive_sets_type_adaptive() -> None:
    assert _build(_make_provider(), "adaptive")["thinking"] == {"type": "adaptive"}


def test_adaptive_forces_temperature_one() -> None:
    assert _build(_make_provider(), "adaptive")["temperature"] == 1.0


def test_adaptive_does_not_inflate_max_tokens() -> None:
    assert _build(_make_provider(), "adaptive", max_tokens=2048)["max_tokens"] == 2048


def test_adaptive_no_budget_tokens() -> None:
    assert "budget_tokens" not in _build(_make_provider(), "adaptive")["thinking"]


def test_high_uses_enabled_with_budget() -> None:
    kw = _build(_make_provider(), "high", max_tokens=4096)
    assert kw["thinking"]["type"] == "enabled"
    assert kw["thinking"]["budget_tokens"] == 8192
    assert kw["max_tokens"] >= kw["thinking"]["budget_tokens"] + 4096


def test_low_uses_small_budget() -> None:
    assert _build(_make_provider(), "low")["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_none_does_not_enable_thinking() -> None:
    kw = _build(_make_provider(), None)
    assert "thinking" not in kw
    assert kw["temperature"] == 0.7


def test_empty_effort_does_not_enable_thinking() -> None:
    kw = _build(_make_provider(), "")
    assert "thinking" not in kw
    assert kw["temperature"] == 0.7


def test_opus_4_7_omits_temperature_adaptive() -> None:
    kw = _build(_make_provider("claude-opus-4-7"), "adaptive")
    assert "temperature" not in kw
    assert kw["thinking"] == {"type": "adaptive"}


def test_opus_4_7_high_uses_adaptive_effort() -> None:
    kw = _build(_make_provider("claude-opus-4-7"), "high", max_tokens=4096)
    assert "temperature" not in kw
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "high"}
    assert kw["max_tokens"] == 4096


def test_opus_4_7_omits_temperature_none() -> None:
    kw = _build(_make_provider("claude-opus-4-7"), None)
    assert "temperature" not in kw
    assert "thinking" not in kw


@pytest.mark.parametrize("model", ["claude-opus-4-8", "claude-fable-5", "claude-sonnet-5"])
def test_adaptive_models_use_high_effort(model: str) -> None:
    kw = _build(_make_provider(model), "high", max_tokens=4096)
    assert "temperature" not in kw
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "high"}


@pytest.mark.parametrize("model", ["claude-opus-4-8", "claude-fable-5", "claude-sonnet-5"])
@pytest.mark.parametrize("effort", ["adaptive", None])
def test_adaptive_only_models_omit_temperature(model: str, effort: str | None) -> None:
    assert "temperature" not in _build(_make_provider(model), effort)


def test_sonnet_5_none_disables_thinking() -> None:
    kw = _build(_make_provider("anthropic/claude-sonnet-5"), "none")
    assert kw["model"] == "claude-sonnet-5"
    assert "temperature" not in kw
    assert kw["thinking"] == {"type": "disabled"}
    assert "output_config" not in kw


def test_mythos_preview_omits_temperature_but_keeps_manual_budget() -> None:
    kw = _build(_make_provider("claude-mythos-preview"), "high", max_tokens=4096)
    assert "temperature" not in kw
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert "output_config" not in kw


@pytest.mark.parametrize("effort", [None, "none", "adaptive", "low", "medium", "high", "xhigh", "max"])
def test_opus_5_omits_temperature(effort: str | None) -> None:
    assert "temperature" not in _build(_make_provider("claude-opus-5"), effort)


def test_opus_5_none_disables_default_thinking() -> None:
    kw = _build(_make_provider("claude-opus-5"), "none")
    assert kw["thinking"] == {"type": "disabled"}
    assert "output_config" not in kw


def test_opus_5_unset_preserves_provider_default() -> None:
    kw = _build(_make_provider("claude-opus-5"), None)
    assert "thinking" not in kw
    assert "output_config" not in kw


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_opus_5_uses_adaptive_thinking_with_effort(effort: str) -> None:
    kw = _build(_make_provider("claude-opus-5"), effort, max_tokens=4096)
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": effort}
    assert kw["max_tokens"] == 4096


def test_dated_opus_5_model_uses_family_capabilities() -> None:
    kw = _build(_make_provider("claude-opus-5-20260724"), "medium")
    assert "temperature" not in kw
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "medium"}


def test_dated_opus_4_model_does_not_treat_date_as_minor_version() -> None:
    kw = _build(_make_provider("claude-opus-4-20250514"), "high")
    assert kw["temperature"] == 1.0
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert "output_config" not in kw


def test_reasoning_effort_string_none_does_not_enable_thinking() -> None:
    kw = _build(_make_provider(), "none")
    assert "thinking" not in kw
    assert kw["temperature"] == 0.7


def test_thinking_forces_auto_tool_choice() -> None:
    tools = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}]
    kw = _build(_make_provider(), "low", tools=tools, tool_choice="required")
    assert kw["tool_choice"] == {"type": "auto"}
    kw = _build(_make_provider(), None, tools=tools, tool_choice="required")
    assert kw["tool_choice"] == {"type": "any"}


@pytest.mark.parametrize(
    "choice, expected",
    [
        (None, {"type": "auto"}),
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ("none", None),
        ({"type": "function", "function": {"name": "x"}}, {"type": "tool", "name": "x"}),
        ({"type": "function"}, {"type": "auto"}),
    ],
)
def test_convert_tool_choice(choice: Any, expected: Any) -> None:
    assert AnthropicProvider._convert_tool_choice(choice) == expected


async def test_instance_reasoning_effort_is_the_default() -> None:
    provider = _mock_client_provider()
    provider.reasoning_effort = "low"
    captured: dict[str, Any] = {}

    async def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn", usage=None)

    provider._client.messages.create = create
    await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    captured.clear()
    await provider.chat(messages=[{"role": "user", "content": "hi"}], reasoning_effort=None)
    assert "thinking" not in captured
    assert captured["temperature"] == 0.7


# --- long-request fallback (test_anthropic_long_request_fallback.py) -------------

_LONG_REQUEST_MESSAGE = (
    "Streaming is required for operations that may take longer than 10 minutes. "
    "See https://github.com/anthropics/anthropic-sdk-python#long-requests for more details"
)


def test_is_streaming_required_error_matches_value_error() -> None:
    assert AnthropicProvider._is_streaming_required_error(ValueError(_LONG_REQUEST_MESSAGE)) is True


def test_is_streaming_required_error_ignores_other_value_errors() -> None:
    assert AnthropicProvider._is_streaming_required_error(ValueError("something else")) is False


def test_is_streaming_required_error_ignores_other_exception_types() -> None:
    assert AnthropicProvider._is_streaming_required_error(RuntimeError(_LONG_REQUEST_MESSAGE)) is False


async def test_chat_falls_back_to_stream_on_long_request_error() -> None:
    provider = _mock_client_provider()
    provider._client.messages.create = AsyncMock(side_effect=ValueError(_LONG_REQUEST_MESSAGE))
    expected = LLMResponse(content="streamed result", finish_reason="stop")
    captured: dict[str, Any] = {}

    async def _fake_chat_stream(**kwargs: Any) -> LLMResponse:
        captured.update(kwargs)
        return expected

    provider.chat_stream = _fake_chat_stream  # type: ignore[method-assign]
    result = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64_000,
        temperature=0.5,
        reasoning_effort="high",
        tool_choice="auto",
    )
    assert result is expected
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["max_tokens"] == 64_000
    assert captured["temperature"] == 0.5
    assert captured["reasoning_effort"] == "high"
    assert captured["tool_choice"] == "auto"
    assert "on_content_delta" not in captured


async def test_chat_does_not_fall_back_on_unrelated_value_error() -> None:
    provider = _mock_client_provider()
    provider._client.messages.create = AsyncMock(side_effect=ValueError("some other validation failure"))
    called = False

    async def _should_not_be_called(**_kwargs: Any) -> LLMResponse:
        nonlocal called
        called = True
        return LLMResponse(content="x")

    provider.chat_stream = _should_not_be_called  # type: ignore[method-assign]
    result = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert called is False
    assert result.finish_reason == "error"


# --- _merge_consecutive (test_anthropic_merge_consecutive.py) --------------------


class TestMergeConsecutive:
    def test_basic_alternation(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
        assert [m["role"] for m in AnthropicProvider._merge_consecutive(msgs)] == ["user", "assistant", "user"]

    def test_consecutive_same_role_merged(self) -> None:
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "reply"},
        ]
        result = AnthropicProvider._merge_consecutive(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_trailing_assistant_stripped(self) -> None:
        msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        assert AnthropicProvider._merge_consecutive(msgs) == [{"role": "user", "content": "hello"}]

    def test_multiple_trailing_assistant_stripped(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "b"},
            {"role": "assistant", "content": "c"},
        ]
        result = AnthropicProvider._merge_consecutive(msgs)
        assert len(result) == 3
        assert result[-1] == {"role": "user", "content": "ok"}

    def test_empty_messages(self) -> None:
        assert AnthropicProvider._merge_consecutive([]) == []

    def test_single_assistant_rerouted_to_user(self) -> None:
        assert AnthropicProvider._merge_consecutive([{"role": "assistant", "content": "hi"}]) == [
            {"role": "user", "content": "hi"}
        ]

    def test_all_assistants_collapse_then_rerouted(self) -> None:
        msgs = [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]
        assert AnthropicProvider._merge_consecutive(msgs) == [
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        ]

    def test_assistant_with_tool_use_not_rerouted(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me search"},
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
                ],
            }
        ]
        assert AnthropicProvider._merge_consecutive(msgs) == []

    def test_leading_assistant_gets_synthetic_user(self) -> None:
        msgs = [
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "reply"},
        ]
        result = AnthropicProvider._merge_consecutive(msgs)
        assert [m["content"] for m in result] == ["(conversation continued)", "hi", "ok"]

    def test_leading_assistant_with_tool_use_left_alone(self) -> None:
        msgs = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "search", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        ]
        assert [m["role"] for m in AnthropicProvider._merge_consecutive(msgs)] == ["assistant", "user"]


# --- streaming (test_anthropic_stream_idle.py) --------------------------------------


def _final_message_stub(text: str = "Hi") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=3, output_tokens=2, cache_creation_input_tokens=None, cache_read_input_tokens=None
        ),
    )


class _FakeAsyncStream:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks
        self._idx = 0
        self.get_final_message = AsyncMock(return_value=_final_message_stub())

    async def __anext__(self) -> SimpleNamespace:
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        c = self._chunks[self._idx]
        self._idx += 1
        return c

    def __aiter__(self) -> _FakeAsyncStream:
        return self


class _ConsumingFakeAsyncStream:
    def __init__(self, chunks: list[SimpleNamespace], per_chunk_delay: float) -> None:
        self._chunks = chunks
        self._idx = 0
        self._delay = per_chunk_delay

    async def __anext__(self) -> SimpleNamespace:
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        c = self._chunks[self._idx]
        self._idx += 1
        await asyncio.sleep(self._delay)
        return c

    def __aiter__(self) -> _ConsumingFakeAsyncStream:
        return self

    async def get_final_message(self) -> SimpleNamespace:
        async for _ in self:
            pass
        return _final_message_stub("ok")


def _install_stream(provider: AnthropicProvider, fake: Any) -> MagicMock:
    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=fake)
    stream_cm.__aexit__ = AsyncMock(return_value=None)
    provider._client.messages.stream = MagicMock(return_value=stream_cm)
    return stream_cm


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=text))


def _thinking(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking=text))


async def test_chat_stream_survives_long_active_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """The idle timeout is per chunk, not a total timeout."""
    monkeypatch.setenv(IDLE_ENV, "0.15")
    provider = _mock_client_provider()
    _install_stream(provider, _ConsumingFakeAsyncStream([_text("a") for _ in range(5)], per_chunk_delay=0.06))
    res = await provider.chat_stream(messages=[{"role": "user", "content": "hello"}])
    assert res.finish_reason != "error", res.content
    assert res.content == "ok"


async def test_chat_stream_still_enforces_idle_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(IDLE_ENV, "0.05")
    provider = _mock_client_provider()

    class _StalledStream(_FakeAsyncStream):
        async def __anext__(self) -> SimpleNamespace:
            await asyncio.sleep(3600)
            raise StopAsyncIteration

    _install_stream(provider, _StalledStream([]))
    res = await provider.chat_stream(messages=[{"role": "user", "content": "hello"}])
    assert res.finish_reason == "error"
    assert res.error_kind == "timeout"
    assert "stalled" in (res.content or "")


async def test_chat_stream_calls_on_content_delta_only_for_text_delta() -> None:
    provider = _mock_client_provider()
    fake = _FakeAsyncStream([_thinking("think"), _text("Hi")])
    _install_stream(provider, fake)
    out: list[str] = []

    async def on_delta(s: str) -> None:
        out.append(s)

    await provider.chat_stream(
        messages=[{"role": "user", "content": "hello"}], on_content_delta=on_delta, on_thinking_delta=None
    )
    assert out == ["Hi"]
    fake.get_final_message.assert_awaited_once()


async def test_chat_stream_invokes_on_thinking_delta() -> None:
    provider = _mock_client_provider()
    _install_stream(provider, _FakeAsyncStream([_thinking("a"), _thinking("b"), _text("X")]))
    thinking_parts: list[str] = []
    text_parts: list[str] = []

    async def on_thinking(s: str) -> None:
        thinking_parts.append(s)

    async def on_text(s: str) -> None:
        text_parts.append(s)

    await provider.chat_stream(
        messages=[{"role": "user", "content": "hello"}], on_content_delta=on_text, on_thinking_delta=on_thinking
    )
    assert thinking_parts == ["a", "b"]
    assert text_parts == ["X"]


async def test_chat_stream_invokes_tool_call_delta_for_input_json_delta() -> None:
    provider = _mock_client_provider()
    chunks = [
        SimpleNamespace(
            type="content_block_start",
            index=1,
            content_block=SimpleNamespace(type="tool_use", id="toolu_1", name="write_file"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"path":"notes.md","content":"'),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json="line\\n"),
        ),
    ]
    fake = _FakeAsyncStream(chunks)
    _install_stream(provider, fake)
    deltas: list[dict[str, Any]] = []

    async def on_tool_delta(delta: dict[str, Any]) -> None:
        deltas.append(delta)

    await provider.chat_stream(messages=[{"role": "user", "content": "write"}], on_tool_call_delta=on_tool_delta)
    base = {"index": 1, "call_id": "toolu_1", "name": "write_file"}
    assert deltas == [
        {**base, "arguments_delta": ""},
        {**base, "arguments_delta": '{"path":"notes.md","content":"'},
        {**base, "arguments_delta": "line\\n"},
    ]
    fake.get_final_message.assert_awaited_once()


async def test_chat_stream_rejects_eof_without_stop_reason() -> None:
    provider = _mock_client_provider()
    partial = _final_message_stub("partial summary")
    partial.stop_reason = None
    fake = _FakeAsyncStream([])
    fake.get_final_message = AsyncMock(return_value=partial)
    stream_cm = _install_stream(provider, fake)
    result = await provider.chat_stream(messages=[{"role": "user", "content": "summarize"}])
    assert result.finish_reason == "error"
    assert result.error_kind == "connection"
    assert LLMProvider.is_transient_response(result)
    stream_cm.__aexit__.assert_awaited_once()


async def test_chat_stream_without_callback_still_finalizes() -> None:
    provider = _mock_client_provider()
    fake = _FakeAsyncStream([])
    fake.get_final_message = AsyncMock(return_value=_final_message_stub("ok"))
    _install_stream(provider, fake)
    res = await provider.chat_stream(messages=[{"role": "user", "content": "hello"}], on_content_delta=None)
    assert res.content == "ok"
    fake.get_final_message.assert_awaited_once()


async def test_chat_stream_passes_idle_timeout_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(IDLE_ENV, "42")
    provider = _mock_client_provider()
    _install_stream(provider, _FakeAsyncStream([]))
    await provider.chat_stream(messages=[{"role": "user", "content": "hello"}])
    assert provider._client.messages.stream.call_args.kwargs["timeout"] == 42.0


# --- tool results / ids (test_anthropic_tool_result.py) -----------------------------


def test_tool_result_block_converts_image_url_in_list_content() -> None:
    msg = {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}, "_meta": {"path": "/x.png"}},
            {"type": "text", "text": "(Image file: /x.png)"},
        ],
    }
    block = AnthropicProvider._tool_result_block(msg)
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_1"
    assert block["content"] == [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        {"type": "text", "text": "(Image file: /x.png)"},
    ]


def test_tool_result_block_preserves_string_content() -> None:
    block = AnthropicProvider._tool_result_block({"role": "tool", "tool_call_id": "call_2", "content": "out"})
    assert block == {"type": "tool_result", "tool_use_id": "call_2", "content": "out"}


def test_convert_user_content_coerces_typeless_dict() -> None:
    result = AnthropicProvider._convert_user_content([{"foo": "bar"}, {"type": "text", "text": "ok"}])
    assert result == [{"type": "text", "text": '{"foo": "bar"}'}, {"type": "text", "text": "ok"}]


def test_convert_user_content_coerces_mixed_typeless() -> None:
    result = AnthropicProvider._convert_user_content([42, {"key": "val"}])
    assert result == [{"type": "text", "text": "42"}, {"type": "text", "text": '{"key": "val"}'}]


def test_url_image_becomes_url_source() -> None:
    block = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
    assert AnthropicProvider._convert_image_block(block) == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/a.png"},
    }


def test_assistant_blocks_coerce_typeless_dict_to_json_text() -> None:
    blocks = AnthropicProvider._assistant_blocks({"role": "assistant", "content": [{"answer": "ok", "count": 2}]})
    assert blocks == [{"type": "text", "text": '{"answer": "ok", "count": 2}'}]


def test_assistant_blocks_malformed_history_arguments_become_empty_object() -> None:
    """nanobot repairs ``{path:"foo.txt"}`` with json_repair; NanoClaw (no json_repair) sends ``{}``."""
    blocks = AnthropicProvider._assistant_blocks(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "toolu_1", "function": {"name": "read_file", "arguments": '{path:"foo.txt"}'}}],
        }
    )
    assert blocks == [{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}}]


def test_assistant_blocks_put_thinking_first() -> None:
    blocks = AnthropicProvider._assistant_blocks(
        {
            "role": "assistant",
            "content": "let me look",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "redacted_thinking", "data": "x"},
            ],
            "tool_calls": [{"id": "t1", "function": {"name": "f", "arguments": '{"a": 1}'}}],
        }
    )
    assert blocks == [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "text", "text": "let me look"},
        {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}},
    ]


def test_anthropic_sanitizes_invalid_tool_ids_consistently() -> None:
    blocks = AnthropicProvider._assistant_blocks(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_abc|rs.same", "function": {"name": "read_file", "arguments": "{}"}}],
        }
    )
    result = AnthropicProvider._tool_result_block({"role": "tool", "tool_call_id": "call_abc|rs.same", "content": "ok"})
    tool_id = blocks[0]["id"]
    assert tool_id == result["tool_use_id"]
    assert tool_id != "call_abc|rs.same"
    assert all(ch.isalnum() or ch in "_-" for ch in tool_id)


def test_anthropic_sanitized_tool_ids_avoid_simple_collisions() -> None:
    blocks = AnthropicProvider._assistant_blocks(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call.a", "function": {"name": "a", "arguments": "{}"}},
                {"id": "call|a", "function": {"name": "b", "arguments": "{}"}},
            ],
        }
    )
    ids = [block["id"] for block in blocks if block["type"] == "tool_use"]
    assert len(ids) == len(set(ids)) == 2


def test_convert_messages_remaps_duplicate_history_tool_ids() -> None:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    _system, messages = provider._convert_messages(
        [
            {"role": "user", "content": "check both files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "toolu_same", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'}},
                    {"id": "toolu_same", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"b.txt"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_same", "name": "read_file", "content": "a"},
            {"role": "tool", "tool_call_id": "toolu_same", "name": "read_file", "content": "b"},
        ]
    )
    tool_uses = [b for b in messages[1]["content"] if b.get("type") == "tool_use"]
    tool_results = [b for b in messages[2]["content"] if b.get("type") == "tool_result"]
    assert [b["id"] for b in tool_uses] == ["toolu_same", "toolu_same__dedupe_2"]
    assert [b["tool_use_id"] for b in tool_results] == ["toolu_same", "toolu_same__dedupe_2"]
    assert tool_uses[1]["input"] == {"path": "b.txt"}


def test_convert_messages_splits_system_and_groups_tool_results() -> None:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    system, messages = provider._convert_messages(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "function": {"name": "f", "arguments": "{}"}},
                {"id": "b", "function": {"name": "g", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "1"},
            {"role": "tool", "tool_call_id": "b", "content": "2"},
            {"role": "user", "content": "and?"},
        ]
    )
    assert system == "be brief"
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "a", "content": "1"},
        {"type": "tool_result", "tool_use_id": "b", "content": "2"},
        {"type": "text", "text": "and?"},
    ]


def test_parse_response_remaps_duplicate_tool_use_ids() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", id="toolu_same", name="read_file", input={"path": "a.txt"}),
            SimpleNamespace(type="tool_use", id="toolu_same", name="read_file", input={"path": "b.txt"}),
        ],
        stop_reason="tool_use",
        usage=None,
    )
    result = AnthropicProvider._parse_response(response)
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].id == "toolu_same"
    assert result.tool_calls[1].id != "toolu_same"
    assert result.tool_calls[1].id.startswith("toolu_")
    assert result.tool_calls[1].arguments == {"path": "b.txt"}


@pytest.mark.parametrize(
    "stop, finish", [("end_turn", "stop"), ("tool_use", "tool_calls"), ("max_tokens", "length"),
                     ("refusal", "refusal"), ("pause_turn", "pause_turn"), (None, "stop")]
)
def test_stop_reason_mapping(stop: str | None, finish: str) -> None:
    response = SimpleNamespace(content=[], stop_reason=stop, usage=None)
    assert AnthropicProvider._parse_response(response).finish_reason == finish


def test_parse_response_keeps_thinking_blocks() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="step 1", signature="sig1"),
            SimpleNamespace(type="text", text="answer"),
        ],
        stop_reason="end_turn",
        usage=None,
    )
    result = AnthropicProvider._parse_response(response)
    assert result.thinking_blocks == [{"type": "thinking", "thinking": "step 1", "signature": "sig1"}]
    assert result.content == "answer"
    assert result.reasoning_content is None


# --- prompt caching (test_prompt_cache_markers.py) -----------------------------------


def _anthropic_tools(*names: str) -> list[dict[str, Any]]:
    return [{"name": n, "description": f"{n} tool", "input_schema": {"type": "object", "properties": {}}} for n in names]


def test_marks_builtin_boundary_and_tail_tool() -> None:
    messages = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]
    system, marked_msgs, marked_tools = AnthropicProvider._apply_cache_control(
        "system", messages, _anthropic_tools("read_file", "write_file", "mcp_fs_ls", "mcp_git_status")
    )
    assert [t["name"] for t in marked_tools or [] if "cache_control" in t] == ["write_file", "mcp_git_status"]
    assert system == [{"type": "text", "text": "system", "cache_control": {"type": "ephemeral"}}]
    assert marked_msgs[1]["content"] == [{"type": "text", "text": "a1", "cache_control": {"type": "ephemeral"}}]
    assert messages[1]["content"] == "a1"  # input untouched


def test_marks_only_tail_without_mcp_and_skips_short_histories() -> None:
    messages = [{"role": "user", "content": "u1"}]
    _, marked_msgs, marked_tools = AnthropicProvider._apply_cache_control("", messages, _anthropic_tools("a", "b"))
    assert [t["name"] for t in marked_tools or [] if "cache_control" in t] == ["b"]
    assert marked_msgs == messages


# --- usage (test_cached_tokens.py) ---------------------------------------------------


def test_adds_native_cache_fields_to_logical_input() -> None:
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="hello")],
        usage=SimpleNamespace(
            input_tokens=800, output_tokens=200, cache_creation_input_tokens=300, cache_read_input_tokens=1200
        ),
    )
    usage = AnthropicProvider._parse_response(response).usage
    assert usage is not None
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (1200, 300)
    assert usage.input_tokens == 2300
    assert usage.total_tokens == 2500
    assert usage.source == "reported"


def test_no_cache_fields() -> None:
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="hello")],
        usage=SimpleNamespace(input_tokens=800, output_tokens=200),
    )
    usage = AnthropicProvider._parse_response(response).usage
    assert usage is not None
    assert usage.input_tokens == 800
    assert usage.cache_read_tokens is None and usage.cache_write_tokens is None


# --- errors (test_provider_error_metadata.py, test_provider_retry_after_hints.py) ----


def test_handle_error_extracts_structured_metadata() -> None:
    class FakeStatusError(Exception):
        pass

    err = FakeStatusError("boom")
    err.status_code = 408  # type: ignore[attr-defined]
    err.response = SimpleNamespace(  # type: ignore[attr-defined]
        status_code=408, headers={"retry-after": "1.5", "x-should-retry": "true"}, text=""
    )
    err.body = {"type": "error", "error": {"type": "rate_limit_error"}}  # type: ignore[attr-defined]
    response = AnthropicProvider._handle_error(err)
    assert response.finish_reason == "error"
    assert response.error_status_code == 408
    assert response.error_type == "rate_limit_error"
    assert response.error_retry_after_s == 1.5
    assert response.error_should_retry is True


def test_handle_error_marks_connection_kind() -> None:
    class FakeConnectionError(Exception):
        pass

    response = AnthropicProvider._handle_error(FakeConnectionError("connection"))
    assert response.error_kind == "connection"
    assert LLMProvider.is_transient_response(response)


def test_error_captures_retry_after_from_headers() -> None:
    err = Exception("boom")
    err.response = SimpleNamespace(headers={"Retry-After": "20"})  # type: ignore[attr-defined]
    assert AnthropicProvider._handle_error(err).retry_after == 20.0


# --- construction (test_provider_sdk_retry_defaults.py) -----------------------------


def test_disables_sdk_retries_by_default() -> None:
    with patch("anthropic.AsyncAnthropic") as mock_client:
        AnthropicProvider(api_key="sk-test", default_model="claude-sonnet-4-5")
    assert mock_client.call_args.kwargs["max_retries"] == 0


def test_normalizes_versioned_base_url() -> None:
    with patch("anthropic.AsyncAnthropic") as mock_client:
        AnthropicProvider(api_key="sk-test", api_base="https://api.minimax.io/anthropic/v1", default_model="m")
    assert mock_client.call_args.kwargs["base_url"] == "https://api.minimax.io/anthropic"


def test_constructor_passes_headers_query_and_proxy() -> None:
    with patch("anthropic.AsyncAnthropic") as mock_client, patch("anthropic.DefaultAsyncHttpxClient") as http:
        provider = AnthropicProvider(
            api_key="k",
            extra_headers={"X-A": "1"},
            extra_query={"q": "v"},
            proxy="http://proxy:8080",
            extra_body={"metadata": {"user_id": "u"}},
        )
    kwargs = mock_client.call_args.kwargs
    assert kwargs["default_headers"] == {"X-A": "1"}
    assert kwargs["default_query"] == {"q": "v"}
    assert kwargs["http_client"] is http.return_value
    http.assert_called_once_with(proxy="http://proxy:8080", trust_env=False)
    built = _build(provider, None)
    assert built["extra_headers"] == {"X-A": "1"}
    assert built["extra_body"] == {"metadata": {"user_id": "u"}}


def test_spec_request_model_name_maps_model() -> None:
    spec = SimpleNamespace(request_model_name=lambda m: m.split("/", 1)[-1].upper())
    with patch("anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="k", default_model="custom/claude-x", spec=spec)
    assert _build(provider, None)["model"] == "CLAUDE-X"


async def test_aclose_closes_client() -> None:
    provider = _mock_client_provider()
    provider._client.close = AsyncMock()
    await provider.aclose()
    provider._client.close.assert_awaited_once()
