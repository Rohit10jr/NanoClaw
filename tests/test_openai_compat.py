"""OpenAICompatProvider against the real openai SDK with a mocked HTTP layer."""

import json

import httpx2  # the openai SDK's HTTP library
import pytest
from openai import AsyncOpenAI

from nanoclaw.providers import LLMUsage, OpenAICompatProvider, provider_from_env

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]


def completion(message, finish_reason="stop", usage=None):
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish_reason}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def tool_call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class MockAPI:
    """Serves one canned reply and remembers the request body."""

    def __init__(self, status=200, body=None, exc=None, headers=None):
        self.status, self.body, self.exc, self.headers = status, body, exc, headers or {}
        self.requests: list[dict] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        if self.exc is not None:
            raise self.exc
        return httpx2.Response(self.status, json=self.body, headers=self.headers)

    def provider(self, **kwargs) -> OpenAICompatProvider:
        client = AsyncOpenAI(
            api_key="test-key",
            base_url="https://llm.test/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler)),
        )
        kwargs.setdefault("api_base", "https://llm.test/v1")
        return OpenAICompatProvider(default_model="test-model", client=client, **kwargs)


async def test_plain_text_reply_and_request_shape():
    api = MockAPI(body=completion({"content": "Hello!"}, usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}))
    resp = await api.provider().chat(messages=[{"role": "user", "content": "hi"}], max_tokens=50, temperature=0.2)

    assert resp.content == "Hello!"
    assert resp.finish_reason == "stop"
    assert resp.tool_calls == []
    assert resp.usage == LLMUsage.reported(input_tokens=12, output_tokens=3, total_tokens=15)
    assert resp.usage.total_tokens == 15

    sent = api.requests[0]
    assert sent["model"] == "test-model"
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["max_tokens"] == 50
    assert sent["temperature"] == 0.2
    assert "tools" not in sent and "tool_choice" not in sent


async def test_tools_are_sent_with_auto_choice_and_model_override():
    api = MockAPI(body=completion({"content": "ok"}))
    await api.provider().chat(messages=[{"role": "user", "content": "hi"}], tools=TOOLS, model="other-model")
    sent = api.requests[0]
    assert sent["model"] == "other-model"
    assert sent["tools"] == TOOLS
    assert sent["tool_choice"] == "auto"


async def test_tool_calls_are_parsed():
    api = MockAPI(
        body=completion(
            {"content": None, "tool_calls": [tool_call("call_1", "read_file", '{"path": "notes.md"}')]},
            finish_reason="tool_calls",
        )
    )
    resp = await api.provider().chat(messages=[{"role": "user", "content": "read notes"}], tools=TOOLS)

    assert resp.content is None
    assert resp.finish_reason == "tool_calls"
    assert resp.should_execute_tools
    [call] = resp.tool_calls
    assert (call.id, call.name, call.arguments) == ("call_1", "read_file", {"path": "notes.md"})


async def test_malformed_arguments_are_preserved_not_guessed():
    api = MockAPI(
        body=completion({"tool_calls": [tool_call("call_1", "read_file", '{"path": "no')]}, finish_reason="tool_calls")
    )
    resp = await api.provider().chat(messages=[{"role": "user", "content": "x"}], tools=TOOLS)
    assert resp.tool_calls[0].arguments == '{"path": "no'


async def test_duplicate_and_missing_tool_call_ids_are_made_unique():
    api = MockAPI(
        body=completion(
            {
                "tool_calls": [
                    tool_call("dup", "read_file", "{}"),
                    tool_call("dup", "read_file", "{}"),
                    tool_call("", "read_file", "{}"),
                ]
            },
            finish_reason="tool_calls",
        )
    )
    resp = await api.provider().chat(messages=[{"role": "user", "content": "x"}], tools=TOOLS)
    ids = [c.id for c in resp.tool_calls]
    assert ids[0] == "dup"
    assert len(set(ids)) == 3
    assert all(ids)


async def test_reasoning_content_is_kept_separate():
    api = MockAPI(body=completion({"content": "42", "reasoning_content": "thinking..."}))
    resp = await api.provider().chat(messages=[{"role": "user", "content": "x"}])
    assert (resp.content, resp.reasoning_content) == ("42", "thinking...")


async def test_round_trip_of_tool_history():
    """A full tool exchange in history is sent exactly as the API expects."""
    api = MockAPI(body=completion({"content": "done"}))
    history = [
        {"role": "user", "content": "read notes"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call("call_1", "read_file", '{"path": "notes.md"}')]},
        {"role": "tool", "tool_call_id": "call_1", "content": "file text"},
    ]
    await api.provider().chat(messages=history, tools=TOOLS)
    sent = api.requests[0]["messages"]
    assert sent[1]["content"] is None  # empty assistant content sanitized
    assert sent[2] == history[2]


async def test_history_is_repaired_before_sending():
    api = MockAPI(body=completion({"content": "ok"}))
    history = [
        {"role": "user", "content": "go", "timestamp": "2026-10-01"},  # unknown key
        {"role": "assistant", "content": "", "tool_calls": [tool_call("c1", "read_file", '{"path": ')]},
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "Error: ..."},
        {"role": "user", "content": "try again"},
        {"role": "user", "content": "please"},  # consecutive user messages
    ]
    await api.provider().chat(messages=history, tools=TOOLS)
    sent = api.requests[0]["messages"]

    assert sent[0] == {"role": "user", "content": "go"}
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert sent[1]["content"] is None
    assert sent[3] == {"role": "user", "content": "try again\n\nplease"}
    assert len(sent) == 4
    # The caller's history is untouched.
    assert history[1]["tool_calls"][0]["function"]["arguments"] == '{"path": '
    assert history[0]["timestamp"] == "2026-10-01"


async def test_http_error_becomes_error_response():
    api = MockAPI(status=401, body={"error": {"message": "Invalid API key", "type": "invalid_request_error"}})
    resp = await api.provider().chat(messages=[{"role": "user", "content": "x"}])
    assert resp.is_error
    assert resp.content == "Error calling LLM: HTTP 401: Invalid API key"


async def test_connection_error_becomes_error_response():
    api = MockAPI(exc=httpx2.ConnectError("refused"))
    resp = await api.provider().chat(messages=[{"role": "user", "content": "x"}])
    assert resp.is_error
    assert resp.content.startswith("Error calling LLM:")


@pytest.mark.parametrize(
    ("data", "content", "is_error"),
    [
        ({"choices": []}, "Error: API returned empty choices.", True),
        # Gateway shapes without choices (nanobot e4fc878).
        ({"choices": [], "content": "top"}, "top", False),
        ({"output_text": [{"type": "text", "text": "out"}]}, "out", False),
        (completion({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}), "ab", False),
        (completion({"content": [{"type": "thinking", "text": "hidden"}, {"type": "text", "text": "shown"}]}), "shown", False),
        # Bare strings in a parts list count as text (nanobot openai_compat_provider.py:1349).
        (completion({"content": ["x", {"type": "text", "text": "y"}]}), "xy", False),
    ],
)
def test_parse_edge_cases(data, content, is_error):
    resp = OpenAICompatProvider.parse(data)
    assert resp.content == content
    assert resp.is_error is is_error


def test_top_level_gateway_reply_keeps_usage_and_reasoning():
    resp = OpenAICompatProvider.parse(
        {
            "choices": [],
            "output_text": "hi",
            "reasoning_content": "hmm",
            "finish_reason": "length",
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }
    )
    assert (resp.content, resp.reasoning_content, resp.finish_reason) == ("hi", "hmm", "length")
    assert resp.usage == LLMUsage.reported(input_tokens=4, output_tokens=2)


def build(model, api_base="https://llm.test/v1", max_tokens=2048, temperature=0.7):
    provider = OpenAICompatProvider(api_key="k", api_base=api_base, default_model=model)
    return provider.build_kwargs(
        [{"role": "user", "content": "hello"}], max_tokens=max_tokens, temperature=temperature
    )


# Adapted from nanobot tests/providers/test_openai_compat_spec.py:902-1053.
@pytest.mark.parametrize(
    ("model", "expected_key"),
    [
        ("gpt-5.4", "max_completion_tokens"),
        ("gpt-5-chat", "max_completion_tokens"),
        ("o1-mini", "max_completion_tokens"),
        ("o3-mini", "max_completion_tokens"),
        ("o4-mini", "max_completion_tokens"),
        ("o1", "max_completion_tokens"),
        ("openai/gpt-5", "max_completion_tokens"),
        ("openai/o3-mini", "max_completion_tokens"),
        ("kimi-k3", "max_completion_tokens"),
        ("gpt-4", "max_tokens"),
        ("gpt-4o-mini", "max_tokens"),
        ("foo3-mini", "max_tokens"),
        ("foo4-mini", "max_tokens"),
        ("o10", "max_tokens"),
    ],
)
def test_max_token_key_by_model_name(model, expected_key):
    kwargs = build(model)
    other_key = "max_tokens" if expected_key == "max_completion_tokens" else "max_completion_tokens"
    assert kwargs[expected_key] == 2048
    assert other_key not in kwargs


@pytest.mark.parametrize(
    ("model", "supports"),
    [
        ("gpt-4o", True),
        ("gpt-4.1", True),
        ("llama3.2", True),
        ("gpt-5-chat", False),
        ("o3-mini", False),
        ("openai/gpt-5", False),
        ("gpt-6-sol", False),  # nanobot keeps it only for reasoning_effort="none"
        ("kimi-k3.1-pro", False),
        ("qwen-o1-preview", False),  # substring rule, as in nanobot
    ],
)
def test_temperature_only_for_models_that_accept_it(model, supports):
    kwargs = build(model, temperature=0.2)
    if supports:
        assert kwargs["temperature"] == 0.2
    else:
        assert "temperature" not in kwargs


@pytest.mark.parametrize("api_base", [None, "https://api.openai.com/v1"])
def test_direct_openai_always_uses_max_completion_tokens(api_base):
    # nanobot's "openai" spec sets supports_max_completion_tokens=True.
    kwargs = build("gpt-4o-mini", api_base=api_base)
    assert kwargs["max_completion_tokens"] == 2048
    assert "max_tokens" not in kwargs
    assert kwargs["temperature"] == 0.7


def test_gpt5_safe_parameters():
    kwargs = build("gpt-5-chat", api_base=None, max_tokens=4096)
    assert kwargs["model"] == "gpt-5-chat"
    assert kwargs["max_completion_tokens"] == 4096
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs


async def test_reasoning_model_request_on_the_wire():
    api = MockAPI(body=completion({"content": "ok"}))
    await api.provider().chat(messages=[{"role": "user", "content": "hi"}], model="o3-mini", max_tokens=50)
    sent = api.requests[0]
    assert sent["max_completion_tokens"] == 50
    assert "max_tokens" not in sent and "temperature" not in sent


def test_provider_from_env():
    p = provider_from_env(
        {"NANOCLAW_API_KEY": "k", "NANOCLAW_API_BASE": "http://localhost:11434/v1", "NANOCLAW_MODEL": "llama3.2"}
    )
    assert (p.api_key, p.api_base, p.default_model) == ("k", "http://localhost:11434/v1", "llama3.2")


def test_provider_from_env_defaults():
    p = provider_from_env({})
    assert p.api_key is None and p.api_base is None
    assert p.default_model == "gpt-4o-mini"


# --- error metadata for the retry policy (nanobot openai_compat_provider.py:1684) ---


async def test_rate_limit_error_carries_retry_metadata():
    api = MockAPI(
        status=429,
        body={"error": {"message": "Rate limit reached", "type": "requests", "code": "rate_limit_exceeded"}},
        headers={"retry-after": "5", "x-should-retry": "true"},
    )
    resp = await api.provider().chat(messages=[{"role": "user", "content": "x"}])
    assert resp.content == "Error calling LLM: HTTP 429: Rate limit reached"
    assert (resp.error_status_code, resp.error_type, resp.error_code) == (429, "requests", "rate_limit_exceeded")
    assert resp.error_retry_after_s == 5.0 and resp.retry_after == 5.0
    assert resp.error_should_retry is True
    assert OpenAICompatProvider.is_transient_response(resp)


async def test_quota_error_is_not_retried():
    api = MockAPI(
        status=429,
        body={"error": {"message": "You exceeded your current quota", "type": "insufficient_quota", "code": "insufficient_quota"}},
    )
    resp = await api.provider().chat(messages=[{"role": "user", "content": "x"}])
    assert resp.error_type == "insufficient_quota" and resp.error_should_retry is None
    assert not OpenAICompatProvider.is_transient_response(resp)
    assert OpenAICompatProvider.is_arrearage_response(resp)


async def test_retry_after_ms_and_should_retry_false_headers():
    api = MockAPI(status=500, body={"error": {"message": "oops"}}, headers={"retry-after-ms": "250", "x-should-retry": "false"})
    resp = await api.provider().chat(messages=[{"role": "user", "content": "x"}])
    assert resp.error_retry_after_s == 0.25
    assert resp.error_should_retry is False
    assert not OpenAICompatProvider.is_transient_response(resp)  # the header beats the 500


async def test_connection_error_metadata():
    api = MockAPI(exc=httpx2.ConnectError("refused"))
    resp = await api.provider().chat(messages=[{"role": "user", "content": "x"}])
    assert resp.error_kind == "connection" and resp.error_status_code is None
    assert OpenAICompatProvider.is_transient_response(resp)


async def test_timeout_error_metadata():
    api = MockAPI(exc=httpx2.ReadTimeout("slow"))
    resp = await api.provider().chat(messages=[{"role": "user", "content": "x"}])
    assert resp.error_kind == "timeout"
    assert OpenAICompatProvider.is_transient_response(resp)


def test_error_metadata_from_duck_typed_exception():
    """nanobot tests/providers/test_provider_error_metadata.py."""

    class FakeStatusError(Exception):
        pass

    err = FakeStatusError("boom")
    err.status_code = 409
    err.response = type("R", (), {"status_code": 409, "headers": {"retry-after-ms": "250", "x-should-retry": "false"}, "text": ""})()
    err.body = {"error": {"type": "rate_limit_exceeded", "code": "rate_limit_exceeded"}}
    resp = OpenAICompatProvider._error_response(err)
    assert (resp.error_status_code, resp.error_type, resp.error_code) == (409, "rate_limit_exceeded", "rate_limit_exceeded")
    assert resp.error_retry_after_s == 0.25 and resp.error_should_retry is False


def test_sdk_retries_are_disabled():
    """nanobot tests/providers/test_provider_sdk_retry_defaults.py: one retry policy, ours."""
    assert OpenAICompatProvider(api_key="k")._get_client().max_retries == 0


async def test_chat_with_retry_over_the_wire(monkeypatch):
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("nanoclaw.providers.base.asyncio.sleep", fake_sleep)
    replies = [
        httpx2.Response(503, json={"error": {"message": "overloaded"}}, headers={"retry-after": "2"}),
        httpx2.Response(200, json=completion({"content": "ok"})),
    ]
    api = MockAPI()
    api.handler = lambda request: replies.pop(0)
    resp = await api.provider().chat_with_retry(messages=[{"role": "user", "content": "x"}])
    assert resp.content == "ok"
    assert delays == [3.0]  # Retry-After + 1s


# --- usage with cache counts (nanobot tests/providers/test_cached_tokens.py) ---


@pytest.mark.parametrize(
    ("raw", "cache_read", "cache_write"),
    [
        ({"prompt_tokens": 2000, "completion_tokens": 300, "prompt_tokens_details": {"cached_tokens": 1200}}, 1200, None),
        ({"prompt_tokens": 2000, "completion_tokens": 300, "prompt_cache_hit_tokens": 1500, "prompt_cache_miss_tokens": 500}, 1500, None),
        ({"prompt_tokens": 2000, "completion_tokens": 300, "cached_tokens": 800}, 800, None),
        ({"prompt_tokens": 2000, "completion_tokens": 300, "prompt_tokens_details": {"cached_tokens": 0}}, 0, None),
        ({"prompt_tokens": 2000, "completion_tokens": 300, "prompt_tokens_details": None}, None, None),
        ({"prompt_tokens": 2000, "completion_tokens": 300}, None, None),
        (
            {"prompt_tokens": 2000, "completion_tokens": 300, "prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 50}},
            100,
            50,
        ),
        # The nested OpenAI field wins over a top-level one.
        ({"prompt_tokens": 2000, "completion_tokens": 300, "prompt_tokens_details": {"cached_tokens": 100}, "cached_tokens": 999}, 100, None),
    ],
)
def test_usage_cache_fields(raw, cache_read, cache_write):
    usage = OpenAICompatProvider._parse_usage(raw)
    assert (usage.input_tokens, usage.output_tokens) == (2000, 300)
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (cache_read, cache_write)
    assert usage.source == "reported" and usage.request_count == 1 and usage.context_tokens == 2000


def test_usage_keeps_a_larger_reported_total():
    usage = OpenAICompatProvider._parse_usage({"prompt_tokens": 15, "completion_tokens": 18, "total_tokens": 175})
    assert usage.total_tokens == 175 and usage.reported_tokens == 175


async def test_sdk_usage_object_round_trip():
    """The SDK's model_dump() gives prompt_tokens_details as a nested dict."""
    usage = {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55, "prompt_tokens_details": {"cached_tokens": 40, "audio_tokens": None}}
    resp = await MockAPI(body=completion({"content": "hi"}, usage=usage)).provider().chat(messages=[{"role": "user", "content": "x"}])
    assert resp.usage.cache_read_tokens == 40 and resp.usage.total_tokens == 55
