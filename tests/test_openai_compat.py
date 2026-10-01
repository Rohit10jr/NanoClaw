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

    def __init__(self, status=200, body=None, exc=None):
        self.status, self.body, self.exc = status, body, exc
        self.requests: list[dict] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        if self.exc is not None:
            raise self.exc
        return httpx2.Response(self.status, json=self.body)

    def provider(self, **kwargs) -> OpenAICompatProvider:
        client = AsyncOpenAI(
            api_key="test-key",
            base_url="https://llm.test/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler)),
        )
        return OpenAICompatProvider(default_model="test-model", client=client, **kwargs)


async def test_plain_text_reply_and_request_shape():
    api = MockAPI(body=completion({"content": "Hello!"}, usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}))
    resp = await api.provider().chat(messages=[{"role": "user", "content": "hi"}], max_tokens=50, temperature=0.2)

    assert resp.content == "Hello!"
    assert resp.finish_reason == "stop"
    assert resp.tool_calls == []
    assert resp.usage == LLMUsage(input_tokens=12, output_tokens=3)
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
        (completion({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}), "ab", False),
        (completion({"content": [{"type": "thinking", "text": "hidden"}, {"type": "text", "text": "shown"}]}), "shown", False),
    ],
)
def test_parse_edge_cases(data, content, is_error):
    resp = OpenAICompatProvider.parse(data)
    assert resp.content == content
    assert resp.is_error is is_error


def test_provider_from_env():
    p = provider_from_env(
        {"NANOCLAW_API_KEY": "k", "NANOCLAW_API_BASE": "http://localhost:11434/v1", "NANOCLAW_MODEL": "llama3.2"}
    )
    assert (p.api_key, p.api_base, p.default_model) == ("k", "http://localhost:11434/v1", "llama3.2")


def test_provider_from_env_defaults():
    p = provider_from_env({})
    assert p.api_key is None and p.api_base is None
    assert p.default_model == "gpt-4o-mini"
