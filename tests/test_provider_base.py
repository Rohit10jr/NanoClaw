import asyncio
import json

import pytest

from nanoclaw.providers import LLMProvider, LLMResponse, ToolCallRequest, parse_tool_arguments
from nanoclaw.providers.base import (
    enforce_role_alternation,
    sanitize_messages,
    tool_arguments_json_for_replay,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"path": "a.txt"}', {"path": "a.txt"}),
        ("", {}),
        ("   ", {}),
        (None, {}),
        ("null", "null"),
        ({"already": "parsed"}, {"already": "parsed"}),
        ('{"path": ', '{"path": '),  # malformed: kept for the registry to reject
        ("[1, 2]", [1, 2]),  # valid JSON but not an object: also left to the registry
    ],
)
def test_parse_tool_arguments(raw, expected):
    assert parse_tool_arguments(raw) == expected


def test_tool_call_serializes_arguments_as_json_string():
    call = ToolCallRequest(id="c1", name="read_file", arguments={"path": "ü.txt"})
    out = call.to_openai_tool_call()
    assert out == {
        "id": "c1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "ü.txt"}'},
    }
    assert json.loads(out["function"]["arguments"]) == {"path": "ü.txt"}


def test_tool_call_keeps_malformed_string_arguments_verbatim():
    call = ToolCallRequest(id="c1", name="x", arguments='{"bad": ')
    assert call.to_openai_tool_call()["function"]["arguments"] == '{"bad": '


def test_tool_call_name_validity():
    assert ToolCallRequest("c", "ok", {}).has_valid_name()
    assert not ToolCallRequest("c", "", {}).has_valid_name()


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [("tool_calls", True), ("stop", True), ("length", False), ("content_filter", False), ("error", False)],
)
def test_should_execute_tools_depends_on_finish_reason(finish_reason, expected):
    resp = LLMResponse(content=None, tool_calls=[ToolCallRequest("c", "t", {})], finish_reason=finish_reason)
    assert resp.should_execute_tools is expected


def test_should_execute_tools_false_without_calls():
    assert not LLMResponse(content="hi", finish_reason="tool_calls").should_execute_tools


def test_sanitize_messages_replaces_empty_content_without_mutating():
    messages = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "hi"},
    ]
    out = sanitize_messages(messages)
    assert [m["content"] for m in out] == ["(empty)", None, "(empty)", "hi"]
    assert messages[0]["content"] == ""


class _Raising(LLMProvider):
    def __init__(self, exc: BaseException):
        super().__init__()
        self.exc = exc

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7, tool_choice=None):
        raise self.exc


async def test_safe_chat_turns_exceptions_into_error_responses():
    resp = await _Raising(RuntimeError("boom")).safe_chat(messages=[])
    assert resp.is_error
    assert resp.content == "Error calling LLM: boom"


async def test_safe_chat_lets_cancellation_through():
    with pytest.raises(asyncio.CancelledError):
        await _Raising(asyncio.CancelledError()).safe_chat(messages=[])


# --- replaying history -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"path": "a"}, '{"path": "a"}'),
        ('{"path": "a"}', '{"path": "a"}'),
        ('{"path": ', "{}"),  # malformed: replaced so the API accepts the history
        ("[1, 2]", "{}"),
        ("", "{}"),
        (None, "{}"),
    ],
)
def test_tool_arguments_json_for_replay(raw, expected):
    assert tool_arguments_json_for_replay(raw) == expected


def test_role_alternation_merges_consecutive_user_messages():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "please answer"},
    ]
    out = enforce_role_alternation(msgs)
    assert out == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi\n\nplease answer"}]
    assert msgs[1]["content"] == "hi"  # input untouched


def test_role_alternation_drops_trailing_assistant():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert enforce_role_alternation(msgs) == [{"role": "user", "content": "hi"}]


def test_role_alternation_keeps_tool_exchanges():
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}]},
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "content": "2"},
        {"role": "user", "content": "y"},
    ]
    assert enforce_role_alternation(msgs) == msgs


def test_role_alternation_prefers_assistant_with_tool_calls():
    plain = {"role": "assistant", "content": "thinking"}
    with_tools = {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}]}
    tool = {"role": "tool", "tool_call_id": "a", "content": "1"}
    assert enforce_role_alternation([{"role": "user", "content": "x"}, plain, with_tools, tool])[1] == with_tools
