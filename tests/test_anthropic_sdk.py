"""AnthropicProvider against the real ``anthropic`` SDK over an httpx MockTransport,
and Anthropic ``thinking_blocks`` through the runner, session history and replay.

No network: every request is answered by a local handler.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from anthropic import AsyncAnthropic

from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.messages import build_assistant_message, extract_reasoning
from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.tools import ToolRegistry
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.providers.anthropic_provider import AnthropicProvider
from nanoclaw.session.manager import Session
from toy_tools import AddTool

THINKING = {"type": "thinking", "thinking": "add them", "signature": "sig-1"}


def _provider(handler: Any, **kwargs: Any) -> tuple[AnthropicProvider, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "headers": dict(request.headers), "body": json.loads(request.content)})
        return handler(request)

    client = AsyncAnthropic(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(recording)),
    )
    return AnthropicProvider(api_key="sk-test", client=client, **kwargs), seen


def _message(content: list[dict[str, Any]], stop_reason: str = "end_turn", **usage: int) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5, **usage},
    }


def _sse(events: list[dict[str, Any]]) -> httpx.Response:
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def _stream_events() -> list[dict[str, Any]]:
    start = _message([], stop_reason=None)  # type: ignore[arg-type]
    start["stop_reason"] = None
    return [
        {"type": "message_start", "message": start},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "add "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "them"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-1"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Let me "}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "add."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "add", "input": {}}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"a": 2,'}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": ' "b": 3}'}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
         "usage": {"output_tokens": 30, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 7}},
        {"type": "message_stop"},
    ]


async def test_chat_over_sdk_sends_messages_api_request_and_parses_answer() -> None:
    reply = _message(
        [THINKING, {"type": "text", "text": "5"}, {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 1}}],
        stop_reason="tool_use",
        cache_read_input_tokens=40,
        cache_creation_input_tokens=2,
    )
    provider, seen = _provider(lambda request: httpx.Response(200, json=reply), extra_headers={"X-Trace": "t"})
    result = await provider.chat(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "2+3?"}],
        tools=[{"type": "function", "function": {"name": "add", "description": "Add", "parameters": {"type": "object"}}}],
    )
    assert seen[0]["url"] == "https://api.anthropic.com/v1/messages"
    assert seen[0]["headers"]["x-trace"] == "t"
    body = seen[0]["body"]
    assert body["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    assert body["messages"] == [{"role": "user", "content": "2+3?"}]
    assert body["tools"] == [
        {"name": "add", "input_schema": {"type": "object"}, "description": "Add", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["tool_choice"] == {"type": "auto"}
    assert body["temperature"] == 0.7
    assert result.content == "5"
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls == [ToolCallRequest(id="toolu_1", name="add", arguments={"a": 1})]
    assert result.thinking_blocks == [THINKING]
    assert result.usage is not None
    assert (result.usage.input_tokens, result.usage.cache_read_tokens, result.usage.cache_write_tokens) == (52, 40, 2)


async def test_chat_stream_over_sdk_emits_deltas_and_final_response() -> None:
    provider, seen = _provider(lambda request: _sse(_stream_events()))
    content: list[str] = []
    thinking: list[str] = []
    tools: list[dict[str, Any]] = []

    async def on_content(text: str) -> None:
        content.append(text)

    async def on_thinking(text: str) -> None:
        thinking.append(text)

    async def on_tool(delta: dict[str, Any]) -> None:
        tools.append(delta)

    result = await provider.chat_stream(
        messages=[{"role": "user", "content": "2+3?"}],
        on_content_delta=on_content,
        on_thinking_delta=on_thinking,
        on_tool_call_delta=on_tool,
    )
    assert seen[0]["body"]["stream"] is True
    assert content == ["Let me ", "add."]
    assert thinking == ["add ", "them"]
    assert [d["arguments_delta"] for d in tools] == ["", '{"a": 2,', ' "b": 3}']
    assert all(d["call_id"] == "toolu_9" and d["name"] == "add" and d["index"] == 2 for d in tools)
    assert result.content == "Let me add."
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls == [ToolCallRequest(id="toolu_9", name="add", arguments={"a": 2, "b": 3})]
    assert result.thinking_blocks == [THINKING]
    assert result.usage is not None
    assert result.usage.output_tokens == 30
    assert result.usage.cache_read_tokens == 100


async def test_long_non_stream_request_falls_back_to_streaming_with_real_sdk() -> None:
    """The SDK refuses a non-stream request with a huge max_tokens; chat() streams instead."""
    events = _stream_events()
    provider, seen = _provider(lambda request: _sse(events))
    result = await provider.chat(messages=[{"role": "user", "content": "hi"}], max_tokens=64_000)
    assert len(seen) == 1 and seen[0]["body"]["stream"] is True
    assert result.finish_reason == "tool_calls"


async def test_status_error_metadata_from_sdk_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "7", "x-should-retry": "true", "request-id": "r1"},
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
        )

    provider, _ = _provider(handler)
    result = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert result.finish_reason == "error"
    assert result.error_status_code == 429
    assert result.error_type == "rate_limit_error"
    assert result.error_retry_after_s == 7.0
    assert result.error_should_retry is True
    assert "slow down" in (result.content or "")
    assert provider.is_transient_response(result)


async def test_overloaded_and_auth_errors_classify_for_retry() -> None:
    provider, _ = _provider(
        lambda r: httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})
    )
    overloaded = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert overloaded.error_status_code == 529 and provider.is_transient_response(overloaded)

    provider, _ = _provider(
        lambda r: httpx.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "bad key"}})
    )
    auth = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert auth.error_status_code == 401 and not provider.is_transient_response(auth)


async def test_connection_error_is_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider, _ = _provider(handler)
    result = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert result.error_kind == "connection"
    assert provider.is_transient_response(result)


async def test_thinking_blocks_replayed_before_tool_use_on_next_request() -> None:
    provider, seen = _provider(lambda request: httpx.Response(200, json=_message([{"type": "text", "text": "ok"}])))
    history = [
        {"role": "user", "content": "2+3?"},
        build_assistant_message(
            "",
            tool_calls=[ToolCallRequest(id="toolu_1", name="add", arguments={"a": 2}).to_openai_tool_call()],
            thinking_blocks=[THINKING],
        ),
        {"role": "tool", "tool_call_id": "toolu_1", "name": "add", "content": "5"},
    ]
    await provider.chat(messages=history, reasoning_effort="low")
    body = seen[0]["body"]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert body["messages"][1]["content"] == [
        THINKING,
        # The second-to-last turn carries the prompt-cache marker.
        {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 2}, "cache_control": {"type": "ephemeral"}},
    ]


# --- thinking_blocks through NanoClaw ------------------------------------------------


def test_build_assistant_message_keeps_thinking_blocks() -> None:
    assert build_assistant_message("hi", thinking_blocks=[THINKING]) == {
        "role": "assistant",
        "content": "hi",
        "reasoning_content": "",
        "thinking_blocks": [THINKING],
    }
    assert build_assistant_message("hi") == {"role": "assistant", "content": "hi"}


def test_extract_reasoning_falls_back_to_thinking_blocks() -> None:
    blocks = [THINKING, {"type": "thinking", "thinking": "more"}, {"type": "redacted_thinking", "data": "x"}]
    assert extract_reasoning(None, blocks, "answer") == ("add them\n\nmore", "answer")
    assert extract_reasoning("rc", blocks, "answer") == ("rc", "answer")
    assert extract_reasoning(None, None, "<think>t</think>answer") == ("t", "answer")


async def test_runner_stores_thinking_blocks_and_replays_them() -> None:
    fake = FakeProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="c1", name="add", arguments={"a": 2, "b": 3})],
                finish_reason="tool_calls",
                thinking_blocks=[THINKING],
            ),
            LLMResponse(content="5", thinking_blocks=[{"type": "thinking", "thinking": "done", "signature": "s2"}]),
        ]
    )
    registry = ToolRegistry()
    registry.register(AddTool())
    result = await AgentRunner().run(
        AgentRunSpec(initial_messages=[{"role": "user", "content": "2+3?"}], tools=registry, provider=fake)
    )
    tool_turn = result.messages[1]
    assert tool_turn["thinking_blocks"] == [THINKING]
    assert tool_turn["reasoning_content"] == ""
    assert result.messages[-1]["thinking_blocks"] == [{"type": "thinking", "thinking": "done", "signature": "s2"}]
    # The second request carried the first answer's thinking blocks.
    assert fake.calls[1].messages[1]["thinking_blocks"] == [THINKING]

    session = Session(key="cli:test")
    session.add_message("user", "2+3?")
    AgentLoop._save_turn(session, result.messages, 1)
    replay = session.get_history()
    assert replay[1]["thinking_blocks"] == [THINKING]
    assert replay[-1]["thinking_blocks"][0]["signature"] == "s2"


def test_session_keeps_empty_assistant_with_thinking_blocks() -> None:
    session = Session(key="cli:test")
    session.messages.extend(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "thinking_blocks": [THINKING]},
            {"role": "user", "content": "again"},
        ]
    )
    assert [m["role"] for m in session.get_history()] == ["user", "assistant", "user"]


@pytest.mark.parametrize("effort", ["low", "adaptive"])
async def test_openai_compat_drops_thinking_blocks(effort: str) -> None:
    from nanoclaw.providers.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider(api_key="k")
    kwargs = provider.build_kwargs(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "thinking_blocks": [THINKING]},
         {"role": "user", "content": effort}]
    )
    assert all("thinking_blocks" not in m for m in kwargs["messages"])
