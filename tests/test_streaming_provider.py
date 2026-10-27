"""The provider half of streaming: chat_stream, chat_stream_with_retry, _parse_chunks.

Adapted from nanobot tests/providers/test_internal_streaming.py,
test_stream_idle_timeout_config.py, test_provider_retry.py (chat_stream_with_retry),
test_cached_tokens.py, test_reasoning_content.py, test_custom_provider.py and
test_mistral_provider.py (_parse_chunks). No test really waits: retry sleeps
are patched out and stalls are simulated.
"""

import asyncio
import json

import httpx2  # the openai SDK's HTTP library
import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from nanoclaw.providers import FakeProvider, LLMProvider, LLMResponse, OpenAICompatProvider, ToolCallRequest
from nanoclaw.providers.base import (
    DEFAULT_STREAM_IDLE_TIMEOUT_S,
    MAX_STREAM_IDLE_TIMEOUT_S,
    STREAM_IDLE_TIMEOUT_ENV,
    resolve_stream_idle_timeout_s,
)
from nanoclaw.providers.fake import ScriptedStream

USER = [{"role": "user", "content": "hello"}]
TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]


@pytest.fixture
def delays(monkeypatch):
    recorded: list[float] = []

    async def fake_sleep(delay):
        recorded.append(delay)

    monkeypatch.setattr("nanoclaw.providers.base.asyncio.sleep", fake_sleep)
    return recorded


class Recorder:
    """Collects what the stream callbacks receive."""

    def __init__(self):
        self.content: list[str] = []
        self.thinking: list[str] = []
        self.tools: list[dict] = []
        self.recoveries = 0

    async def on_content(self, text):
        self.content.append(text)

    async def on_thinking(self, text):
        self.thinking.append(text)

    async def on_tool(self, delta):
        self.tools.append(delta)

    async def on_recover(self):
        self.recoveries += 1

    def callbacks(self, *, recover=False):
        kwargs = dict(
            on_content_delta=self.on_content, on_thinking_delta=self.on_thinking, on_tool_call_delta=self.on_tool
        )
        if recover:
            kwargs["on_stream_recover"] = self.on_recover
        return kwargs


# --- resolve_stream_idle_timeout_s (nanobot test_stream_idle_timeout_config.py) ---


def test_idle_timeout_rejects_invalid_values():
    assert resolve_stream_idle_timeout_s(env_value="abc") == DEFAULT_STREAM_IDLE_TIMEOUT_S
    assert resolve_stream_idle_timeout_s(env_value="-1") == DEFAULT_STREAM_IDLE_TIMEOUT_S
    assert resolve_stream_idle_timeout_s(env_value="0") == DEFAULT_STREAM_IDLE_TIMEOUT_S
    assert resolve_stream_idle_timeout_s(env_value="  ") == DEFAULT_STREAM_IDLE_TIMEOUT_S


def test_idle_timeout_accepts_and_clamps_numbers():
    assert resolve_stream_idle_timeout_s(env_value="1.5") == 1.5
    assert resolve_stream_idle_timeout_s(env_value="7200") == MAX_STREAM_IDLE_TIMEOUT_S


def test_idle_timeout_reads_nanoclaw_env_var(monkeypatch):
    assert STREAM_IDLE_TIMEOUT_ENV == "NANOCLAW_STREAM_IDLE_TIMEOUT_S"
    monkeypatch.delenv(STREAM_IDLE_TIMEOUT_ENV, raising=False)
    assert resolve_stream_idle_timeout_s() == 90.0
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "12")
    assert resolve_stream_idle_timeout_s() == 12.0


# --- LLMProvider.chat_stream default (nanobot base.py:1085) ---


class ChatOnlyProvider(LLMProvider):
    """Implements only chat(); streaming comes from the base class."""

    def __init__(self, responses):
        super().__init__(default_model="test-model")
        self._responses = list(responses)
        self.calls = 0
        self.last_kwargs: dict = {}

    async def chat(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


async def test_default_chat_stream_sends_whole_content_as_one_delta():
    provider = ChatOnlyProvider([LLMResponse(content="all at once", reasoning_content="hmm")])
    rec = Recorder()
    response = await provider.chat_stream(messages=USER, tools=TOOLS, max_tokens=7, **rec.callbacks())
    assert response.content == "all at once"
    assert rec.content == ["all at once"]
    assert rec.thinking == [] and rec.tools == []
    assert provider.last_kwargs["max_tokens"] == 7 and provider.last_kwargs["tools"] == TOOLS


async def test_default_chat_stream_sends_no_delta_without_content():
    tool = ToolCallRequest("c1", "read_file", {})
    provider = ChatOnlyProvider([LLMResponse(content=None, tool_calls=[tool], finish_reason="tool_calls")])
    rec = Recorder()
    response = await provider.chat_stream(messages=USER, **rec.callbacks())
    assert response.tool_calls == [tool]
    assert rec.content == []


async def test_default_chat_stream_is_bounded_by_idle_timeout(monkeypatch):
    """nanobot test_chat_only_provider_still_has_a_timeout, with the wait simulated."""
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "5")
    seen: list[float] = []

    async def fake_wait_for(awaitable, timeout):
        seen.append(timeout)
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr("nanoclaw.providers.base.asyncio.wait_for", fake_wait_for)
    response = await ChatOnlyProvider([]).safe_chat_stream(messages=USER)
    assert seen == [5.0]
    assert response.finish_reason == "error"
    assert response.error_kind == "timeout"


async def test_safe_chat_stream_turns_exceptions_into_errors():
    provider = ChatOnlyProvider([ConnectionError("reset by peer")])
    response = await provider.safe_chat_stream(messages=USER)
    assert response.is_error and response.error_kind == "connection"
    assert "reset by peer" in response.content


async def test_safe_chat_stream_lets_cancellation_through():
    provider = ChatOnlyProvider([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await provider.safe_chat_stream(messages=USER)


# --- chat_stream_with_retry (nanobot test_provider_retry.py:279-395) ---


class StreamScriptProvider(LLMProvider):
    """Each script entry is (deltas, response); chat_stream emits the deltas first."""

    def __init__(self, script):
        super().__init__(default_model="test-model")
        self._script = list(script)
        self.calls = 0
        self.last_kwargs: dict = {}

    async def chat(self, **kwargs):  # pragma: no cover - streaming only
        raise AssertionError("chat() should not be called")

    async def chat_stream(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        deltas, response = self._script.pop(0)
        for kind, text in deltas:
            callback = kwargs.get(f"on_{kind}_delta")
            if callback:
                await callback(text)
        return response


def stall():
    return LLMResponse(
        content="Error calling LLM: stream stalled for more than 30 seconds",
        finish_reason="error",
        error_kind="timeout",
    )


async def test_retries_error_before_any_content(delays):
    provider = StreamScriptProvider(
        [
            ([], LLMResponse(content="503 overloaded", finish_reason="error")),
            ([("content", "ok")], LLMResponse(content="ok")),
        ]
    )
    rec = Recorder()
    response = await provider.chat_stream_with_retry(messages=USER, **rec.callbacks())
    assert response.content == "ok"
    assert provider.calls == 2
    assert rec.content == ["ok"]
    assert delays == [1]


async def test_does_not_retry_after_emitting_content(delays):
    first = LLMResponse(content="503 stream broke", finish_reason="error", error_kind="connection")
    provider = StreamScriptProvider([([("content", "partial")], first), ([], LLMResponse(content="ok"))])
    rec = Recorder()
    response = await provider.chat_stream_with_retry(messages=USER, **rec.callbacks())
    assert response is first
    assert provider.calls == 1
    assert rec.content == ["partial"]
    assert delays == []


async def test_retries_timeout_after_content_with_callbacks_suppressed(delays):
    provider = StreamScriptProvider(
        [([("content", "partial")], stall()), ([("content", "again")], LLMResponse(content="full retry response"))]
    )
    rec = Recorder()
    response = await provider.chat_stream_with_retry(messages=USER, **rec.callbacks())
    assert response.content == "full retry response"
    assert provider.calls == 2
    assert rec.content == ["partial"]
    assert delays == [1]
    assert provider.last_kwargs["on_content_delta"] is None
    assert provider.last_kwargs["on_thinking_delta"] is None
    assert provider.last_kwargs["on_tool_call_delta"] is None


async def test_retries_timeout_after_content_in_new_stream_segment(delays):
    second = LLMResponse(content="full retry response")
    provider = StreamScriptProvider(
        [([("content", "partial")], stall()), ([("content", "full retry response")], second)]
    )
    rec = Recorder()
    response = await provider.chat_stream_with_retry(messages=USER, **rec.callbacks(recover=True))
    assert response is second
    assert provider.calls == 2
    assert rec.content == ["partial", "full retry response"]
    assert rec.recoveries == 1
    assert delays == [1]
    assert provider.last_kwargs["on_content_delta"] is not None
    # NanoClaw providers don't restart streams themselves, so the retry loop
    # (not chat_stream) calls the recover callback.
    assert "on_stream_recover" not in provider.last_kwargs


async def test_new_segment_is_guarded_again_after_recovery(delays):
    """After a recovered stall, text in the new segment blocks retries again."""
    broken = LLMResponse(content="503 broke", finish_reason="error")
    provider = StreamScriptProvider(
        [([("content", "a")], stall()), ([("content", "b")], broken), ([], LLMResponse(content="never"))]
    )
    rec = Recorder()
    response = await provider.chat_stream_with_retry(messages=USER, **rec.callbacks(recover=True))
    assert response is broken
    assert provider.calls == 2
    assert rec.content == ["a", "b"]


async def test_recovered_segment_without_text_can_still_be_retried(delays):
    """Recovery resets the guard: a new segment that has shown nothing yet is
    retried on any transient error, not only on another stall."""
    provider = StreamScriptProvider(
        [
            ([("content", "a")], stall()),
            ([], LLMResponse(content="503 overloaded", finish_reason="error")),
            ([("content", "b")], LLMResponse(content="b")),
        ]
    )
    rec = Recorder()
    response = await provider.chat_stream_with_retry(messages=USER, **rec.callbacks(recover=True))
    assert response.content == "b"
    assert provider.calls == 3
    assert rec.recoveries == 1
    assert rec.content == ["a", "b"]


async def test_stream_recover_callback_goes_to_providers_that_support_it(delays):
    class RecoveringProvider(StreamScriptProvider):
        supports_stream_recover_callback = True

    provider = RecoveringProvider([([], LLMResponse(content="ok"))])
    rec = Recorder()
    await provider.chat_stream_with_retry(messages=USER, **rec.callbacks(recover=True))
    await provider.last_kwargs["on_stream_recover"]()
    assert rec.recoveries == 1


async def test_thinking_and_tool_deltas_do_not_block_retries(delays):
    provider = StreamScriptProvider(
        [
            ([("thinking", "hmm"), ("tool_call", {"index": 0})], LLMResponse(content="503", finish_reason="error")),
            ([], LLMResponse(content="ok")),
        ]
    )
    rec = Recorder()
    response = await provider.chat_stream_with_retry(messages=USER, **rec.callbacks())
    assert response.content == "ok"
    assert provider.calls == 2


async def test_without_content_callback_errors_are_always_retryable(delays):
    """nanobot only tracks text it delivered itself (base.py:1257)."""
    provider = StreamScriptProvider(
        [([("content", "unseen")], LLMResponse(content="503", finish_reason="error")), ([], LLMResponse(content="ok"))]
    )
    response = await provider.chat_stream_with_retry(messages=USER)
    assert response.content == "ok"
    assert provider.calls == 2
    assert provider.last_kwargs["on_content_delta"] is None


async def test_non_transient_stream_error_is_not_retried(delays):
    denied = LLMResponse(content="HTTP 401: bad key", finish_reason="error", error_status_code=401)
    provider = StreamScriptProvider([([], denied)])
    response = await provider.chat_stream_with_retry(messages=USER, **Recorder().callbacks())
    assert response is denied
    assert delays == []


async def test_stream_retry_gives_up_after_four_attempts(delays):
    provider = StreamScriptProvider([([], LLMResponse(content="503", finish_reason="error")) for _ in range(4)])
    notes: list[str] = []

    async def on_wait(text):
        notes.append(text)

    response = await provider.chat_stream_with_retry(messages=USER, on_retry_wait=on_wait)
    assert response.is_error
    assert provider.calls == 4
    assert delays == [1, 2, 4]
    assert notes[-1] == "Model request failed after 4 attempts, giving up."


async def test_stream_retry_persistent_mode(delays):
    provider = StreamScriptProvider([([], LLMResponse(content="503", finish_reason="error")) for _ in range(10)])
    response = await provider.chat_stream_with_retry(messages=USER, retry_mode="persistent")
    assert response.is_error
    assert provider.calls == 10


async def test_stream_retry_passes_request_arguments():
    provider = StreamScriptProvider([([], LLMResponse(content="ok"))])
    await provider.chat_stream_with_retry(
        messages=USER, tools=TOOLS, model="m", max_tokens=9, temperature=0.1, tool_choice="required"
    )
    kwargs = provider.last_kwargs
    assert (kwargs["messages"], kwargs["tools"], kwargs["model"]) == (USER, TOOLS, "m")
    assert (kwargs["max_tokens"], kwargs["temperature"], kwargs["tool_choice"]) == (9, 0.1, "required")


async def test_stream_retry_turns_exceptions_into_retryable_errors(delays):
    class Flaky(StreamScriptProvider):
        async def chat_stream(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError()
            return LLMResponse(content="ok")

    provider = Flaky([])
    response = await provider.chat_stream_with_retry(messages=USER)
    assert response.content == "ok" and provider.calls == 2


# --- OpenAICompatProvider.chat_stream over the real SDK (nanobot test_internal_streaming.py) ---


def chunk(delta=None, finish_reason=None, *, usage=None, choices=True):
    body = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": "test-model"}
    body["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}] if choices else []
    if usage is not None:
        body["usage"] = usage
    return body


class StreamAPI:
    """Serves one SSE stream (or an error) and remembers the request."""

    def __init__(self, events=None, *, then=None, status=200, body=None, headers=None):
        self.events, self.status, self.body, self.headers = events or [], status, body, headers or {}
        # Event lists for later requests; the last one repeats.
        self.later: list[list[dict]] = list(then or [])
        self.requests: list[dict] = []
        self.timeouts: list[dict] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        self.timeouts.append(request.extensions.get("timeout"))
        if self.status != 200:
            return httpx2.Response(self.status, json=self.body, headers=self.headers)
        events = self.events
        if len(self.requests) > 1 and self.later:
            events = self.later[min(len(self.requests) - 2, len(self.later) - 1)]
        payload = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=payload.encode())

    def provider(self, **kwargs) -> OpenAICompatProvider:
        client = AsyncOpenAI(
            api_key="test-key",
            base_url="https://llm.test/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler)),
        )
        kwargs.setdefault("api_base", "https://llm.test/v1")
        kwargs.setdefault("default_model", "test-model")
        return OpenAICompatProvider(client=client, **kwargs)


USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 125,
    "prompt_tokens_details": {"cached_tokens": 40},
}

TOOL_STREAM = [
    chunk({"role": "assistant", "reasoning_content": "Let me "}),
    chunk({"reasoning_content": "look."}),
    chunk({"content": "Reading "}),
    chunk({"content": "files."}),
    chunk(
        {
            "tool_calls": [
                {"index": 0, "id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": ""}}
            ]
        }
    ),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"path": '}}]}),
    chunk(
        {
            "tool_calls": [
                {"index": 1, "id": "call_b", "type": "function", "function": {"name": "exec", "arguments": '{"cmd"'}}
            ]
        }
    ),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"a.txt"}'}}]}),
    chunk({"tool_calls": [{"index": 1, "function": {"arguments": ': "ls"}'}}]}),
    chunk({}, "tool_calls"),
    chunk(choices=False, usage=USAGE),
]


async def test_stream_request_shape(monkeypatch):
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "7")
    api = StreamAPI([chunk({"content": "ok"}, "stop")])
    response = await api.provider().chat_stream(messages=USER, tools=TOOLS, max_tokens=50, temperature=0.2)
    assert response.content == "ok"
    sent = api.requests[0]
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["max_tokens"] == 50 and sent["temperature"] == 0.2
    assert sent["tools"] == TOOLS and sent["tool_choice"] == "auto"
    assert api.timeouts[0]["read"] == 7.0


async def test_stream_keeps_layer_11_parameter_rules():
    api = StreamAPI([chunk({"content": "ok"}, "stop")])
    await api.provider(default_model="gpt-5").chat_stream(messages=USER, max_tokens=50)
    sent = api.requests[0]
    assert sent["max_completion_tokens"] == 50
    assert "max_tokens" not in sent and "temperature" not in sent


async def test_stream_emits_deltas_and_assembles_response():
    api = StreamAPI(TOOL_STREAM)
    rec = Recorder()
    response = await api.provider().chat_stream(messages=USER, tools=TOOLS, **rec.callbacks())

    assert rec.content == ["Reading ", "files."]
    assert rec.thinking == ["Let me ", "look."]
    assert rec.tools == [
        {"index": 0, "call_id": "call_a", "name": "read_file", "arguments_delta": ""},
        {"index": 0, "call_id": "", "name": "", "arguments_delta": '{"path": '},
        {"index": 1, "call_id": "call_b", "name": "exec", "arguments_delta": '{"cmd"'},
        {"index": 0, "call_id": "", "name": "", "arguments_delta": '"a.txt"}'},
        {"index": 1, "call_id": "", "name": "", "arguments_delta": ': "ls"}'},
    ]
    assert response.content == "Reading files."
    assert response.reasoning_content == "Let me look."
    assert response.finish_reason == "tool_calls"
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [
        ("call_a", "read_file", {"path": "a.txt"}),
        ("call_b", "exec", {"cmd": "ls"}),
    ]
    assert response.usage.input_tokens == 100 and response.usage.output_tokens == 20
    assert response.usage.total_tokens == 125 and response.usage.cache_read_tokens == 40


async def test_stream_matches_non_streaming_parse():
    """The assembled stream equals what chat() makes of the same reply."""
    streamed = await StreamAPI(TOOL_STREAM).provider().chat_stream(messages=USER, tools=TOOLS)
    whole = OpenAICompatProvider.parse(
        {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "Reading files.",
                        "reasoning_content": "Let me look.",
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                            },
                            {
                                "id": "call_b",
                                "type": "function",
                                "function": {"name": "exec", "arguments": '{"cmd": "ls"}'},
                            },
                        ],
                    },
                }
            ],
            "usage": USAGE,
        }
    )
    assert streamed == whole


async def test_stream_without_callbacks_still_assembles():
    response = await StreamAPI(TOOL_STREAM).provider().chat_stream(messages=USER)
    assert response.content == "Reading files." and len(response.tool_calls) == 2


async def test_stream_legacy_function_call_deltas():
    api = StreamAPI(
        [
            chunk({"function_call": {"name": "get_weather", "arguments": '{"ci'}}),
            chunk({"function_call": {"arguments": 'ty": "Paris"}'}}),
            chunk({}, "function_call"),
        ]
    )
    rec = Recorder()
    response = await api.provider().chat_stream(messages=USER, **rec.callbacks())
    assert rec.tools == [
        {"index": 0, "call_id": "", "name": "get_weather", "arguments_delta": '{"ci'},
        {"index": 0, "call_id": "", "name": "", "arguments_delta": 'ty": "Paris"}'},
    ]
    (call,) = response.tool_calls
    assert call.name == "get_weather" and call.arguments == {"city": "Paris"}
    assert len(call.id) == 9  # generated: the legacy form has no ID
    assert response.finish_reason == "function_call"


async def test_stream_mistral_thinking_parts_go_to_thinking():
    thinking = {"type": "thinking", "thinking": [{"type": "text", "text": "deep"}]}
    api = StreamAPI([chunk({"content": [thinking, {"type": "text", "text": "visible"}]}), chunk({}, "stop")])
    rec = Recorder()
    response = await api.provider().chat_stream(messages=USER, **rec.callbacks())
    assert rec.content == ["visible"] and rec.thinking == ["deep"]
    assert response.content == "visible" and response.reasoning_content == "deep"


async def test_stream_without_finish_reason_is_a_retryable_connection_error():
    """nanobot test_silent_stream_rejects_eof_without_completion."""
    tool = chunk({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "read_file", "arguments": "{}"}}]})
    api = StreamAPI([chunk({"content": "partial"}), tool])
    response = await api.provider().chat_stream(messages=USER)
    assert response.finish_reason == "error"
    assert response.error_kind == "connection"
    assert "finish reason" in response.content
    assert not response.should_execute_tools
    assert LLMProvider.is_transient_response(response)


async def test_truncated_stream_is_retried_until_complete(delays):
    api = StreamAPI([chunk({"content": "partial"})], then=[[chunk({"content": "partial"}), chunk({}, "stop")]])
    # Without a content callback nothing was shown, so the retry is allowed.
    response = await api.provider().chat_stream_with_retry(messages=USER)
    assert response.content == "partial" and response.finish_reason == "stop"
    assert len(api.requests) == 2 and delays == [1]


async def test_truncated_stream_after_shown_text_is_not_retried(delays):
    api = StreamAPI([chunk({"content": "partial"})])
    rec = Recorder()
    response = await api.provider().chat_stream_with_retry(messages=USER, **rec.callbacks())
    assert response.error_kind == "connection"
    assert rec.content == ["partial"]
    assert len(api.requests) == 1 and delays == []


async def test_stream_http_error_keeps_error_metadata():
    api = StreamAPI(
        status=429,
        body={"error": {"message": "Rate limit reached", "type": "requests", "code": "rate_limit_exceeded"}},
        headers={"retry-after": "3", "x-should-retry": "true"},
    )
    response = await api.provider().chat_stream(messages=USER)
    assert response.is_error
    assert response.content == "Error calling LLM: HTTP 429: Rate limit reached"
    assert response.error_status_code == 429
    assert response.error_code == "rate_limit_exceeded"
    assert response.error_retry_after_s == 3.0 and response.retry_after == 3.0
    assert response.error_should_retry is True


async def test_openai_compat_stream_ignores_invalid_idle_timeout_env(monkeypatch):
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "abc")
    api = StreamAPI([chunk({"content": "ok"}, "stop")])
    response = await api.provider().chat_stream(messages=USER)
    assert response.content == "ok"
    assert api.timeouts[0]["read"] == DEFAULT_STREAM_IDLE_TIMEOUT_S


# --- idle timeout and stream closing, with a fake client ---


class FakeStream:
    """An SDK-like stream: yields chunks, then optionally stalls."""

    def __init__(self, chunks, *, stall_at=None):
        self.chunks = [ChatCompletionChunk.model_validate(c) for c in chunks]
        self.stall_at = stall_at
        self.closed = False
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index == self.stall_at:
            # What asyncio.wait_for raises when the idle timeout passes.
            raise asyncio.TimeoutError
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        self.index += 1
        return self.chunks[self.index - 1]

    async def close(self):
        self.closed = True


class FakeCompletions:
    def __init__(self, stream):
        self.stream = stream
        self.kwargs: dict = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.stream


def fake_client_provider(stream):
    completions = FakeCompletions(stream)
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    return OpenAICompatProvider(api_base="https://llm.test/v1", default_model="test-model", client=client), completions


@pytest.mark.parametrize("stall_at", [0, 2], ids=["first-chunk", "after-reasoning"])
async def test_stalled_stream_times_out_and_closes(monkeypatch, stall_at):
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "30")
    stream = FakeStream(
        [chunk({"reasoning_content": "thinking"}) for _ in range(4)] + [chunk({}, "stop")], stall_at=stall_at
    )
    provider, _ = fake_client_provider(stream)
    response = await provider.chat_stream(messages=USER)
    assert response.finish_reason == "error"
    assert response.error_kind == "timeout"
    assert response.content == "Error calling LLM: stream stalled for more than 30 seconds"
    assert stream.closed


async def test_every_chunk_gets_a_fresh_idle_timeout(monkeypatch):
    """nanobot test_silent_stream_renews_idle_timeout_on_every_event: the timeout
    is per chunk, never for the whole stream."""
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "0.3")
    timeouts: list[float] = []
    real_wait_for = asyncio.wait_for

    async def recording_wait_for(awaitable, timeout):
        timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout)

    stream = FakeStream([chunk({"reasoning_content": " "}) for _ in range(8)] + [chunk({"content": "done"}, "stop")])
    provider, completions = fake_client_provider(stream)
    monkeypatch.setattr("nanoclaw.providers.openai_compat.asyncio.wait_for", recording_wait_for)
    response = await provider.chat_stream(messages=USER)
    assert response.content == "done"
    assert timeouts == [0.3] * 10  # 9 chunks + end of stream
    assert completions.kwargs["timeout"] == 0.3
    assert stream.closed


async def test_stall_after_content_retries_in_new_segment(delays, monkeypatch):
    monkeypatch.setenv(STREAM_IDLE_TIMEOUT_ENV, "30")
    streams = [
        FakeStream([chunk({"content": "par"}), chunk({"content": "tial"})], stall_at=2),
        FakeStream([chunk({"content": "full"}), chunk({}, "stop")]),
    ]
    provider, completions = fake_client_provider(None)
    pending = list(streams)

    async def create(**kwargs):
        return pending.pop(0)

    completions.create = create
    rec = Recorder()
    response = await provider.chat_stream_with_retry(messages=USER, **rec.callbacks(recover=True))
    assert response.content == "full"
    assert rec.content == ["par", "tial", "full"]
    assert rec.recoveries == 1
    assert all(s.closed for s in streams)


# --- _parse_chunks (nanobot openai_compat_provider.py:1556) ---


def sdk(chunks):
    return [ChatCompletionChunk.model_validate(chunk(choices=False) | c) for c in chunks]


@pytest.mark.parametrize("as_sdk", [False, True], ids=["dict", "sdk"])
def test_parse_chunks_keeps_usage_only_chunk(as_sdk):
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 18,
                "total_tokens": 175,
                "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            },
        },
    ]
    result = OpenAICompatProvider._parse_chunks(sdk(chunks) if as_sdk else chunks)
    assert result.content == "Hello" and result.finish_reason == "stop"
    assert result.usage.input_tokens == 15 and result.usage.output_tokens == 18
    assert result.usage.total_tokens == 175 and result.usage.reported_tokens == 175
    assert result.usage.cache_read_tokens == 0 and result.usage.cache_write_tokens == 0


@pytest.mark.parametrize("as_sdk", [False, True], ids=["dict", "sdk"])
def test_parse_chunks_accumulates_reasoning(as_sdk):
    chunks = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "Step 1. "}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"reasoning_content": "Step 2."}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}]},
    ]
    result = OpenAICompatProvider._parse_chunks(sdk(chunks) if as_sdk else chunks)
    assert result.content == "answer"
    assert result.reasoning_content == "Step 1. Step 2."


def test_parse_chunks_reasoning_none_when_absent():
    result = OpenAICompatProvider._parse_chunks([chunk({"content": "hi"}, "stop")])
    assert result.reasoning_content is None


def test_parse_chunks_accepts_plain_text_chunks():
    result = OpenAICompatProvider._parse_chunks(["hello ", "world"])
    assert result.content == "hello world" and result.finish_reason == "stop"


@pytest.mark.parametrize("as_sdk", [False, True], ids=["dict", "sdk"])
def test_parse_chunks_deduplicates_parallel_tool_call_ids(as_sdk):
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "a", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "b", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    result = OpenAICompatProvider._parse_chunks(sdk(chunks) if as_sdk else chunks)
    ids = [call.id for call in result.tool_calls]
    assert ids[0] == "call_1" and ids[1] != "call_1" and len(set(ids)) == 2


def test_parse_chunks_uses_position_when_index_missing():
    chunks = [
        chunk(
            {
                "tool_calls": [
                    {"id": "x1", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "x2", "function": {"name": "b", "arguments": '{"k": 1}'}},
                ]
            }
        ),
        chunk({}, "tool_calls"),
    ]
    result = OpenAICompatProvider._parse_chunks(chunks)
    assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [("x1", "a", {}), ("x2", "b", {"k": 1})]


def test_parse_chunks_keeps_malformed_arguments_for_the_registry():
    chunks = [
        chunk({"tool_calls": [{"index": 0, "id": "m", "function": {"name": "f", "arguments": '{"a": '}}]}),
        chunk({}, "tool_calls"),
    ]
    assert OpenAICompatProvider._parse_chunks(chunks).tool_calls[0].arguments == '{"a": '


def test_parse_chunks_missing_finish_reason_defaults_to_stop():
    # chat_stream rejects such streams before parsing; the parser itself
    # defaults like nanobot's.
    assert OpenAICompatProvider._parse_chunks([chunk({"content": "x"})]).finish_reason == "stop"


def test_parse_chunks_top_level_text_and_cache_hit_usage():
    chunks = [
        {"choices": [], "content": "gateway text"},
        chunk({}, "length", usage=None),
        chunk(choices=False, usage={"prompt_tokens": 5, "completion_tokens": 2, "prompt_cache_hit_tokens": 3}),
    ]
    result = OpenAICompatProvider._parse_chunks(chunks)
    assert result.content == "gateway text" and result.finish_reason == "length"
    assert result.usage.cache_read_tokens == 3 and result.usage.total_tokens == 7


def test_parse_chunks_rejects_unknown_chunk_types():
    with pytest.raises(TypeError):
        OpenAICompatProvider._parse_chunks([42])


# --- FakeProvider streaming ---


async def test_fake_streams_scripted_reply_word_by_word():
    fake = FakeProvider([LLMResponse(content="Hello big world", reasoning_content="hmm")])
    rec = Recorder()
    response = await fake.chat_stream(messages=USER, **rec.callbacks())
    assert rec.content == ["Hello ", "big ", "world"]
    assert rec.thinking == ["hmm"]
    assert response.content == "Hello big world"
    assert fake.calls[0].streamed and fake.stream_calls == 1


async def test_fake_streams_tool_calls_as_deltas():
    call = ToolCallRequest("c1", "read_file", {"path": "a"})
    fake = FakeProvider([LLMResponse(content=None, tool_calls=[call], finish_reason="tool_calls")])
    rec = Recorder()
    response = await fake.chat_stream(messages=USER, **rec.callbacks())
    assert rec.content == []
    assert rec.tools == [{"index": 0, "call_id": "c1", "name": "read_file", "arguments_delta": '{"path": "a"}'}]
    assert response.tool_calls == [call]


async def test_fake_explicit_deltas_and_chat_compatibility():
    fake = FakeProvider(
        [
            ScriptedStream(content_deltas=["a", "b"], thinking_deltas=["t"]),
            ScriptedStream(content_deltas=["x"]),
            "plain",
        ]
    )
    rec = Recorder()
    first = await fake.chat_stream(messages=USER, **rec.callbacks())
    assert rec.content == ["a", "b"] and rec.thinking == ["t"]
    assert first.content == "ab"
    assert (await fake.chat(messages=USER)).content == "x"  # chat() skips the deltas
    assert (await fake.chat(messages=USER)).content == "plain"
    assert [call.streamed for call in fake.calls] == [True, False, False]


async def test_fake_error_reply_streams_nothing():
    fake = FakeProvider([LLMResponse(content="boom", finish_reason="error")])
    rec = Recorder()
    response = await fake.chat_stream_with_retry(messages=USER, retry_mode="persistent", **rec.callbacks())
    assert response.is_error and rec.content == []
    assert len(fake.calls) == 1  # one call per entry without retries=True
    assert fake.last_retry_mode == "persistent"


async def test_fake_stream_retry_policy_with_retries(delays):
    broken = ScriptedStream(content_deltas=["par"], response=LLMResponse(content="503", finish_reason="error"))
    fake = FakeProvider([broken, "unused"], retries=True)
    rec = Recorder()
    response = await fake.chat_stream_with_retry(messages=USER, **rec.callbacks())
    assert response.content == "503" and len(fake.calls) == 1  # text shown: no retry

    fake = FakeProvider([LLMResponse(content="503", finish_reason="error"), "ok"], retries=True)
    response = await fake.chat_stream_with_retry(messages=USER, **Recorder().callbacks())
    assert response.content == "ok" and fake.stream_calls == 2


async def test_fake_script_exhausted_when_streaming():
    response = await FakeProvider().chat_stream(messages=USER)
    assert response.is_error and "exhausted" in response.content
