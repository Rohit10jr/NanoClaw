"""AgentRunner against a scripted FakeProvider and toy tools."""

import copy

import pytest

from nanoclaw.agent.messages import (
    BUDGET_EXHAUSTED_FINALIZATION_PROMPT,
    EMPTY_FINAL_RESPONSE_MESSAGE,
    FINALIZATION_RETRY_PROMPT,
    MODEL_ERROR_PLACEHOLDER,
    TRUNCATED_SUFFIX,
)
from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.tools import RETRY_HINT, ToolRegistry
from nanoclaw.providers import FakeProvider, LLMResponse, LLMUsage, ToolCallRequest
from toy_tools import AddTool, BigTool, BoomTool, EchoTool, SilentTool

USER = {"role": "user", "content": "please help"}


def call(name, args, call_id="c1"):
    return ToolCallRequest(id=call_id, name=name, arguments=args)


def wants_tools(*calls, content=None, finish_reason="tool_calls", usage=None):
    return LLMResponse(content=content, tool_calls=list(calls), finish_reason=finish_reason, usage=usage)


def make_spec(provider, *tools, **kwargs):
    registry = ToolRegistry()
    for tool in tools or (EchoTool(), AddTool(), BoomTool(), SilentTool(), BigTool()):
        registry.register(tool)
    return AgentRunSpec(initial_messages=[USER], tools=registry, provider=provider, **kwargs)


async def run(provider, *tools, **kwargs):
    return await AgentRunner().run(make_spec(provider, *tools, **kwargs))


# --- the happy paths -------------------------------------------------------


async def test_plain_answer_without_tools():
    fake = FakeProvider(["Hello!"])
    result = await run(fake)

    assert result.stop_reason == "completed"
    assert result.final_content == "Hello!"
    assert result.error is None
    assert result.messages == [USER, {"role": "assistant", "content": "Hello!"}]
    assert len(fake.calls) == 1


async def test_one_tool_call_then_answer_produces_exact_history():
    fake = FakeProvider([wants_tools(call("add", {"a": 2, "b": 3})), "The sum is 5."])
    result = await run(fake)

    assert result.stop_reason == "completed"
    assert result.final_content == "The sum is 5."
    assert result.tools_used == ["add"]
    assert result.messages == [
        USER,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "add", "content": "5"},
        {"role": "assistant", "content": "The sum is 5."},
    ]


async def test_second_request_contains_tool_result_and_tool_definitions():
    fake = FakeProvider([wants_tools(call("echo", {"text": "ping"})), "done"])
    await run(fake)

    first, second = fake.calls
    assert first.messages == [USER]
    assert [d["function"]["name"] for d in first.tools] == ["add", "big", "boom", "echo", "silent"]
    assert second.messages[-1] == {"role": "tool", "tool_call_id": "c1", "name": "echo", "content": "ping"}
    assert second.tools == first.tools


async def test_several_calls_in_one_reply_run_in_order():
    fake = FakeProvider(
        [
            wants_tools(call("echo", {"text": "one"}, "a"), call("echo", {"text": "two"}, "b")),
            "ok",
        ]
    )
    result = await run(fake)
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_msgs] == [("a", "one"), ("b", "two")]
    assert result.tools_used == ["echo", "echo"]


async def test_multiple_iterations():
    fake = FakeProvider(
        [
            wants_tools(call("add", {"a": 1, "b": 1}, "c1")),
            wants_tools(call("add", {"a": 2, "b": 2}, "c2")),
            "2 and 4",
        ]
    )
    result = await run(fake)
    assert result.stop_reason == "completed"
    assert result.tools_used == ["add", "add"]
    assert len(fake.calls) == 3


async def test_text_alongside_tool_calls_is_kept():
    fake = FakeProvider([wants_tools(call("echo", {"text": "x"}), content="Let me check."), "done"])
    result = await run(fake)
    assert result.messages[1]["content"] == "Let me check."


async def test_reasoning_content_is_kept_in_history():
    fake = FakeProvider([LLMResponse(content="42", reasoning_content="thinking")])
    result = await run(fake)
    assert result.messages[-1] == {"role": "assistant", "content": "42", "reasoning_content": "thinking"}


async def test_no_tools_sent_when_registry_is_empty():
    fake = FakeProvider(["hi"])
    await AgentRunner().run(AgentRunSpec(initial_messages=[USER], tools=ToolRegistry(), provider=fake))
    assert fake.calls[0].tools is None


# --- tool failures go back to the model ------------------------------------


@pytest.mark.parametrize(
    ("bad_call", "expected_start"),
    [
        (call("nope", {}), "Error: Tool 'nope' not found."),
        (call("echo", '{"text": "unterminated'), "Error: Tool 'echo' parameters must be a JSON object"),
        (call("add", {"a": 1}), "Error: Invalid parameters for tool 'add': missing required 'b'"),
        (call("boom", {}), "Error executing boom: kaboom"),
        (call("", {}), "Error: Tool '' not found."),
    ],
)
async def test_tool_failures_are_reported_and_the_model_recovers(bad_call, expected_start):
    fake = FakeProvider([wants_tools(bad_call), "Sorry, that failed."])
    result = await run(fake)

    tool_msg = result.messages[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"].startswith(expected_start)
    assert tool_msg["content"].endswith(RETRY_HINT)
    assert result.tools_used == []
    assert result.stop_reason == "completed"
    assert fake.calls[1].messages[-1] == tool_msg  # the model saw the error


# --- tool output handling ---------------------------------------------------


async def test_long_tool_output_is_truncated():
    fake = FakeProvider([wants_tools(call("big", {})), "ok"])
    result = await run(fake, max_tool_result_chars=10)
    assert result.messages[2]["content"] == "x" * 10 + TRUNCATED_SUFFIX


async def test_empty_tool_output_is_replaced_with_marker():
    fake = FakeProvider([wants_tools(call("silent", {})), "ok"])
    result = await run(fake)
    assert result.messages[2]["content"] == "(silent completed with no output)"
    assert result.tools_used == ["silent"]


# --- cut-off replies ---------------------------------------------------------


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
async def test_tool_calls_in_cut_off_reply_are_ignored(finish_reason):
    fake = FakeProvider([wants_tools(call("echo", {"text": "x"}), content="partial", finish_reason=finish_reason)])
    result = await run(fake)
    assert result.stop_reason == "completed"
    assert result.final_content == "partial"
    assert result.tools_used == []
    assert all(m["role"] != "tool" for m in result.messages)
    assert len(fake.calls) == 1


# --- empty replies -----------------------------------------------------------


async def test_empty_reply_is_retried():
    fake = FakeProvider(["", "Here you go."])
    result = await run(fake)
    assert result.stop_reason == "completed"
    assert result.final_content == "Here you go."
    assert len(fake.calls) == 2
    assert fake.calls[1].messages == [USER]  # plain retry: same request


async def test_second_empty_reply_triggers_finalization_without_tools():
    fake = FakeProvider(["", "  ", "Final answer."])
    result = await run(fake)

    assert result.final_content == "Final answer."
    assert result.stop_reason == "completed"
    finalize = fake.calls[2]
    assert finalize.tools is None
    assert finalize.messages[-1] == {"role": "user", "content": FINALIZATION_RETRY_PROMPT}
    # The finalization prompt is not saved into history.
    assert result.messages == [USER, {"role": "assistant", "content": "Final answer."}]


async def test_empty_after_finalization_gives_fallback():
    fake = FakeProvider(["", "", ""])
    result = await run(fake)
    assert result.stop_reason == "empty_final_response"
    assert result.final_content == EMPTY_FINAL_RESPONSE_MESSAGE
    assert result.error == EMPTY_FINAL_RESPONSE_MESSAGE
    assert result.messages[-1] == {"role": "assistant", "content": EMPTY_FINAL_RESPONSE_MESSAGE}


async def test_empty_retry_counter_resets_after_tools():
    fake = FakeProvider(["", wants_tools(call("echo", {"text": "x"})), "", "answer"])
    result = await run(fake)
    assert result.final_content == "answer"
    assert all(c.tools is not None for c in fake.calls)  # no finalization needed


# --- model errors ---------------------------------------------------------------


async def test_model_error_stops_with_placeholder_in_history():
    fake = FakeProvider([LLMResponse(content="Error calling LLM: HTTP 500: boom", finish_reason="error")])
    result = await run(fake)
    assert result.stop_reason == "error"
    assert result.final_content == result.error == "Error calling LLM: HTTP 500: boom"
    assert result.messages == [USER, {"role": "assistant", "content": MODEL_ERROR_PLACEHOLDER}]


async def test_model_error_after_tools_keeps_history_valid():
    fake = FakeProvider([wants_tools(call("echo", {"text": "x"})), LLMResponse(content=None, finish_reason="error")])
    result = await run(fake)
    assert result.stop_reason == "error"
    assert result.final_content == "Sorry, I encountered an error calling the AI model."
    assert [m["role"] for m in result.messages] == ["user", "assistant", "tool", "assistant"]


# --- iteration limit -------------------------------------------------------------


def looping_provider(n, final):
    script = [wants_tools(call("echo", {"text": str(i)}, f"c{i}")) for i in range(n)]
    return FakeProvider(script + [final])


async def test_max_iterations_asks_for_final_answer_without_tools():
    fake = looping_provider(3, "Summary of progress.")
    result = await run(fake, max_iterations=3)

    assert result.stop_reason == "max_iterations"
    assert result.final_content == "Summary of progress."
    assert len(fake.calls) == 4
    finalize = fake.calls[3]
    assert finalize.tools is None
    assert finalize.messages[-1] == {"role": "user", "content": BUDGET_EXHAUSTED_FINALIZATION_PROMPT}
    assert result.messages[-1] == {"role": "assistant", "content": "Summary of progress."}
    assert all(m.get("content") != BUDGET_EXHAUSTED_FINALIZATION_PROMPT for m in result.messages)


@pytest.mark.parametrize(
    "bad_final",
    [
        LLMResponse(content="x", finish_reason="error"),
        LLMResponse(content="", finish_reason="stop"),
        wants_tools(call("echo", {"text": "again"})),
    ],
)
async def test_max_iterations_falls_back_to_fixed_message(bad_final):
    fake = looping_provider(2, bad_final)
    result = await run(fake, max_iterations=2)
    assert result.stop_reason == "max_iterations"
    assert result.final_content.startswith("I reached the maximum number of tool call iterations (2)")


# --- bookkeeping -------------------------------------------------------------------


async def test_usage_is_summed_across_calls():
    fake = FakeProvider(
        [
            wants_tools(call("echo", {"text": "x"}), usage=LLMUsage(10, 2)),
            LLMResponse(content="done", usage=LLMUsage(15, 3)),
        ]
    )
    result = await run(fake)
    assert result.usage == LLMUsage(25, 5)


async def test_usage_is_none_when_provider_reports_none():
    result = await run(FakeProvider(["hi"]))
    assert result.usage is None


async def test_initial_messages_are_not_modified():
    fake = FakeProvider([wants_tools(call("echo", {"text": "x"})), "done"])
    spec = make_spec(fake)
    before = copy.deepcopy(spec.initial_messages)
    await AgentRunner().run(spec)
    assert spec.initial_messages == before


async def test_every_tool_call_has_exactly_one_result():
    fake = FakeProvider(
        [
            wants_tools(call("echo", {"text": "a"}, "a"), call("nope", {}, "b"), call("boom", {}, "c")),
            "done",
        ]
    )
    result = await run(fake)
    asked = [tc["id"] for m in result.messages if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
    answered = [m["tool_call_id"] for m in result.messages if m["role"] == "tool"]
    assert asked == answered == ["a", "b", "c"]
