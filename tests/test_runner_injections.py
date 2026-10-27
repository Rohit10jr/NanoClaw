"""Mid-turn injections, checkpoints and cancellation guards in AgentRunner.

Adapted from nanobot tests/agent/test_runner_injections.py (runner-level
tests only; the loop/queue tests belong to the loop) and the checkpoint
assertions of tests/agent/test_runner_core.py.
"""

import asyncio
from types import SimpleNamespace

import pytest

from nanoclaw.agent.hook import AgentHook, AgentHookContext, AgentRunHookContext
from nanoclaw.agent.messages import EMPTY_FINAL_RESPONSE_MESSAGE, MODEL_ERROR_PLACEHOLDER
from nanoclaw.agent.runner import _MAX_EMPTY_RETRIES, AgentRunner, AgentRunSpec
from nanoclaw.agent.tools import Tool, ToolRegistry
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from toy_tools import AddTool, EchoTool

USER = {"role": "user", "content": "hello"}


class ScriptedProvider(FakeProvider):
    """FakeProvider that runs ``on_call(n)`` (1-based) before answering call n."""

    def __init__(self, script, on_call=None):
        super().__init__(script)
        self.on_call = on_call

    async def chat_stream_with_retry(self, *args, **kwargs):
        if self.on_call is not None:
            await self.on_call(len(self.calls) + 1)
        return await super().chat_stream_with_retry(*args, **kwargs)


def queue_callback(queue: list):
    """An injection callback that drains ``queue`` (one snapshot per call)."""

    async def drain():
        items = list(queue)
        queue.clear()
        return items

    return drain


def tool_reply(*calls):
    return LLMResponse(content="", tool_calls=list(calls), finish_reason="tool_calls")


def call(name="add", args=None, call_id="c1"):
    return ToolCallRequest(id=call_id, name=name, arguments=args if args is not None else {"a": 1, "b": 2})


def spec(provider, *, tools=None, initial=None, **kwargs):
    registry = ToolRegistry()
    for tool in tools or (AddTool(), EchoTool()):
        registry.register(tool)
    return AgentRunSpec(
        initial_messages=list(initial or [USER]),
        tools=registry,
        provider=provider,
        hook=kwargs.pop("hook", AgentHook()),
        **kwargs,
    )


def sent(provider, index):
    """(role, content) of every message in request ``index``."""
    return [(m["role"], m.get("content")) for m in provider.calls[index].messages]


# --- _drain_injections ------------------------------------------------------


async def _drain(callback, *, terminal=False):
    key = "terminal_injection_callback" if terminal else "injection_callback"
    return await AgentRunner()._drain_injections(spec(FakeProvider(), **{key: callback}), terminal=terminal)


async def test_drain_without_callback_is_empty():
    assert await _drain(None) == []


async def test_drain_takes_content_from_objects():
    async def cb():
        return [SimpleNamespace(content="hello"), SimpleNamespace(content="world")]

    assert await _drain(cb) == [{"role": "user", "content": "hello"}, {"role": "user", "content": "world"}]


async def test_drain_keeps_the_entire_snapshot():
    async def cb():
        return [SimpleNamespace(content=f"msg{i}") for i in range(8)]

    assert await _drain(cb) == [{"role": "user", "content": f"msg{i}"} for i in range(8)]


async def test_drain_skips_blank_and_none_content():
    async def cb():
        return [
            SimpleNamespace(content=""),
            SimpleNamespace(content="   "),
            SimpleNamespace(content=None),
            "plain text",
            SimpleNamespace(content="valid"),
        ]

    assert await _drain(cb) == [{"role": "user", "content": "plain text"}, {"role": "user", "content": "valid"}]


async def test_drain_keeps_only_user_dicts_with_content_as_they_are():
    multimodal = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
    tagged = {"role": "user", "content": "valid", "_meta": {"id": "one"}}

    async def cb():
        return [
            {"role": "user", "content": ""},
            {"role": "user", "content": "   "},
            {"role": "user", "content": None},
            {"role": "user"},
            {"role": "assistant", "content": "should not be re-injected as user"},
            None,
            tagged,
            {"role": "user", "content": multimodal},
        ]

    result = await _drain(cb)
    assert result == [tagged, {"role": "user", "content": multimodal}]
    assert result[0] is tagged


async def test_drain_callback_error_counts_as_no_input():
    async def cb():
        raise RuntimeError("boom")

    assert await _drain(cb) == []


async def test_terminal_drain_uses_the_terminal_callback():
    async def cb():
        return ["from terminal"]

    assert await _drain(cb, terminal=True) == [{"role": "user", "content": "from terminal"}]


# --- injection points in the run --------------------------------------------


async def test_input_pending_before_the_first_request_is_sent_with_it():
    queue = ["also this"]
    fake = FakeProvider(["answer"])
    result = await AgentRunner().run(spec(fake, injection_callback=queue_callback(queue)))

    assert result.had_injections is True
    # Raw history keeps both messages; the model copy merges adjacent users.
    assert result.messages[:2] == [USER, {"role": "user", "content": "also this"}]
    assert sent(fake, 0) == [("user", "hello\n\nalso this")]


async def test_injection_after_tool_execution_reaches_the_next_request():
    queue: list = []

    class QueueingAdd(AddTool):
        async def execute(self, a, b):
            queue.append(SimpleNamespace(content="follow-up question"))
            return await super().execute(a, b)

    fake = FakeProvider([tool_reply(call()), "final answer"])
    result = await AgentRunner().run(
        spec(fake, tools=[QueueingAdd()], injection_callback=queue_callback(queue))
    )

    assert result.had_injections is True
    assert result.final_content == "final answer"
    assert len(fake.calls) == 2
    assert sent(fake, 1)[-1] == ("user", "follow-up question")
    assert sent(fake, 1)[-2][0] == "tool"


async def test_input_after_an_answer_keeps_the_answer_and_continues():
    queue: list = []

    async def on_call(n):
        if n == 1:
            queue.append(SimpleNamespace(content="follow-up question"))

    fake = ScriptedProvider(["first answer", "second answer"], on_call)
    result = await AgentRunner().run(spec(fake, injection_callback=queue_callback(queue)))

    assert result.final_content == "second answer"
    assert result.stop_reason == "completed"
    assert result.had_injections is True
    assert sent(fake, 1) == [
        ("user", "hello"),
        ("assistant", "first answer"),
        ("user", "follow-up question"),
    ]
    assert result.messages == [
        USER,
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow-up question"},
        {"role": "assistant", "content": "second answer"},
    ]


async def test_stream_stays_open_when_input_arrives_after_an_answer():
    queue: list = []
    ends: list[bool] = []

    class Streaming(AgentHook):
        def wants_streaming(self):
            return True

        async def on_stream_end(self, context, *, resuming):
            ends.append(resuming)

    async def on_call(n):
        if n == 1:
            queue.append(SimpleNamespace(content="quick follow-up"))

    fake = ScriptedProvider(["first answer", "second answer"], on_call)
    result = await AgentRunner().run(spec(fake, hook=Streaming(), injection_callback=queue_callback(queue)))

    assert result.final_content == "second answer"
    assert ends == [True, False]


async def test_injected_media_stays_raw_in_history_and_merges_for_the_model():
    image = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "text", "text": "look at this"},
    ]
    rounds = {"n": 0}

    async def inject():
        rounds["n"] += 1
        if rounds["n"] == 2:  # after the first answer
            return [{"role": "user", "content": image}, {"role": "user", "content": "and answer briefly"}]
        return []

    fake = FakeProvider(["first answer", "second answer"])
    result = await AgentRunner().run(spec(fake, injection_callback=inject))

    assert result.final_content == "second answer"
    users = [m for m in fake.calls[1].messages if m["role"] == "user"]
    assert len(users) == 2
    assert users[-1]["content"] == [*image, {"type": "text", "text": "and answer briefly"}]
    assert [m["content"] for m in result.messages[-3:-1]] == [image, "and answer briefly"]


async def test_injection_cycles_have_no_cap_but_the_iteration_budget():
    queue: list = []

    async def on_call(n):
        if n <= 7:
            queue.append(SimpleNamespace(content=f"msg-{n}"))

    fake = ScriptedProvider([f"answer-{i}" for i in range(1, 9)], on_call)
    result = await AgentRunner().run(spec(fake, max_iterations=20, injection_callback=queue_callback(queue)))

    assert result.had_injections is True
    assert len(fake.calls) == 8
    assert result.final_content == "answer-8"


async def test_no_injections_flag_is_false_by_default():
    result = await AgentRunner().run(spec(FakeProvider(["done"]), max_iterations=1))
    assert result.had_injections is False


async def test_run_hook_context_reports_had_injections():
    seen: list[AgentRunHookContext] = []

    class Capture(AgentHook):
        async def after_run(self, context):
            seen.append(context)

    queue = ["extra"]
    await AgentRunner().run(spec(FakeProvider(["done"]), hook=Capture(), injection_callback=queue_callback(queue)))
    assert seen[0].had_injections is True


async def test_answer_on_the_last_iteration_leaves_new_input_queued():
    queue: list = []

    async def on_call(n):
        queue.append(SimpleNamespace(content="too late"))

    fake = ScriptedProvider(["only answer"], on_call)
    result = await AgentRunner().run(spec(fake, max_iterations=1, injection_callback=queue_callback(queue)))

    assert result.final_content == "only answer"
    assert result.had_injections is False
    assert len(fake.calls) == 1
    assert [item.content for item in queue] == ["too late"]


# --- the terminal callback (waiting for subagents) ---------------------------


async def test_terminal_wait_runs_only_when_an_answer_would_end_the_turn():
    second_request_started = asyncio.Event()
    allow_final = asyncio.Event()
    terminal_started = asyncio.Event()
    release = asyncio.Event()
    delivered = False

    async def on_call(n):
        if n == 2:
            second_request_started.set()
            await allow_final.wait()

    async def nothing_pending():
        return []

    async def wait_at_terminal():
        nonlocal delivered
        if delivered:
            return []
        terminal_started.set()
        await release.wait()
        delivered = True
        return [SimpleNamespace(content="background result")]

    fake = ScriptedProvider([tool_reply(call()), "main work finished", "combined final answer"], on_call)
    task = asyncio.create_task(
        AgentRunner().run(
            spec(
                fake,
                max_iterations=5,
                injection_callback=nothing_pending,
                terminal_injection_callback=wait_at_terminal,
            )
        )
    )

    await asyncio.wait_for(second_request_started.wait(), timeout=1.0)
    assert not terminal_started.is_set()  # not after the tool round
    allow_final.set()
    await asyncio.wait_for(terminal_started.wait(), timeout=1.0)
    assert not task.done()
    release.set()
    result = await asyncio.wait_for(task, timeout=1.0)

    assert len(fake.calls) == 3
    assert result.had_injections is True
    assert result.final_content == "combined final answer"
    assert sent(fake, 2)[-2:] == [("assistant", "main work finished"), ("user", "background result")]


async def test_pending_input_is_taken_before_the_terminal_wait():
    terminal_calls = 0
    queue: list = []

    async def on_call(n):
        if n == 1:
            queue.append("pending")

    async def wait_at_terminal():
        nonlocal terminal_calls
        terminal_calls += 1
        return []

    fake = ScriptedProvider(["first", "second"], on_call)
    result = await AgentRunner().run(
        spec(fake, injection_callback=queue_callback(queue), terminal_injection_callback=wait_at_terminal)
    )

    assert result.final_content == "second"
    assert terminal_calls == 1  # only after the second answer


@pytest.mark.parametrize(
    "reply",
    [
        LLMResponse(content="cut off", finish_reason="length"),
        LLMResponse(content="boom", finish_reason="error"),
    ],
)
async def test_terminal_wait_is_skipped_for_non_answers(reply):
    terminal_calls = 0

    async def wait_at_terminal():
        nonlocal terminal_calls
        terminal_calls += 1
        return ["late"]

    result = await AgentRunner().run(spec(FakeProvider([reply]), terminal_injection_callback=wait_at_terminal))

    assert terminal_calls == 0
    assert result.had_injections is False


async def test_terminal_wait_is_skipped_on_the_last_iteration():
    terminal_calls = 0

    async def wait_at_terminal():
        nonlocal terminal_calls
        terminal_calls += 1
        return ["late"]

    result = await AgentRunner().run(
        spec(FakeProvider(["answer"]), max_iterations=1, terminal_injection_callback=wait_at_terminal)
    )
    assert terminal_calls == 0
    assert result.final_content == "answer"


# --- recovery after errors (and the D7 reset) --------------------------------


async def test_input_after_a_model_error_gets_an_answer():
    queue: list = []

    async def on_call(n):
        if n == 1:
            queue.append(SimpleNamespace(content="follow-up after LLM error"))

    fake = ScriptedProvider([LLMResponse(content=None, finish_reason="error"), "recovered answer"], on_call)
    result = await AgentRunner().run(
        spec(
            fake,
            initial=[USER, {"role": "assistant", "content": "previous"}, {"role": "user", "content": "trigger error"}],
            injection_callback=queue_callback(queue),
        )
    )

    assert result.had_injections is True
    assert result.final_content == "recovered answer"
    assert result.stop_reason == "completed"
    assert result.error is None
    # Drained right after the failed answer: no placeholder in between.
    assert [m["content"] for m in result.messages if m["role"] == "user"][-2:] == [
        "trigger error",
        "follow-up after LLM error",
    ]
    assert MODEL_ERROR_PLACEHOLDER not in [m["content"] for m in result.messages]


async def test_input_after_empty_replies_gets_an_answer():
    queue: list = []

    async def on_call(n):
        if n == _MAX_EMPTY_RETRIES + 1:
            queue.append(SimpleNamespace(content="follow-up after empty"))

    script = [""] * (_MAX_EMPTY_RETRIES + 1) + ["answer after empty"]
    fake = ScriptedProvider(script, on_call)
    result = await AgentRunner().run(spec(fake, max_iterations=10, injection_callback=queue_callback(queue)))

    assert result.had_injections is True
    assert result.final_content == "answer after empty"
    assert [m for m in result.messages if m.get("content") == "follow-up after empty"] == [
        {"role": "user", "content": "follow-up after empty"}
    ]


@pytest.mark.parametrize("failed_stop_reason", ["error", "empty_final_response"])
async def test_late_input_after_a_failure_resets_the_failure(failed_stop_reason):
    """nanobot 5825e15 (D7): a recovered turn must not report the old failure."""
    failed = (
        [LLMResponse(content="temporary failure", finish_reason="error")]
        if failed_stop_reason == "error"
        else [""] * (_MAX_EMPTY_RETRIES + 1)
    )
    queue: list = []
    observed: list[str] = []
    on_error_calls: list = []

    class InjectAfterFailure(AgentHook):
        async def after_iteration(self, context):
            if context.stop_reason == failed_stop_reason and context.error is not None:
                observed.append(context.error)
                queue.append(SimpleNamespace(content="Please retry now"))

        async def on_error(self, context):
            on_error_calls.append(context)

    fake = FakeProvider([*failed, "recovered answer"])
    result = await AgentRunner().run(
        spec(
            fake,
            hook=InjectAfterFailure(),
            max_iterations=_MAX_EMPTY_RETRIES + 2,
            injection_callback=queue_callback(queue),
        )
    )

    assert len(observed) == 1
    assert len(fake.calls) == len(failed) + 1
    assert result.had_injections is True
    assert result.final_content == "recovered answer"
    assert result.stop_reason == "completed"
    assert result.error is None
    assert on_error_calls == []
    if failed_stop_reason == "error":
        assert {"role": "assistant", "content": MODEL_ERROR_PLACEHOLDER} in result.messages
    else:
        assert {"role": "assistant", "content": EMPTY_FINAL_RESPONSE_MESSAGE} in result.messages


async def test_error_recovery_has_no_cap_but_the_iteration_budget():
    queue: list = []

    async def on_call(n):
        if n <= 7:
            queue.append(SimpleNamespace(content=f"msg-{n}"))

    fake = ScriptedProvider([LLMResponse(content=None, finish_reason="error")] * 8, on_call)
    result = await AgentRunner().run(spec(fake, max_iterations=20, injection_callback=queue_callback(queue)))

    assert result.had_injections is True
    assert len(fake.calls) == 8
    assert result.stop_reason == "error"


# --- the iteration budget ----------------------------------------------------


class InjectAfterIteration(AgentHook):
    def __init__(self, queue, when):
        super().__init__()
        self.queue, self.when, self.count = queue, when, 0

    async def after_iteration(self, context):
        self.count += 1
        if self.count == self.when:
            self.queue.append(SimpleNamespace(content="late follow-up after max iters"))


async def test_max_iterations_without_finalization_keeps_late_input_queued():
    queue: list = []
    fake = FakeProvider([tool_reply(call(call_id="c1")), tool_reply(call(call_id="c2"))])
    result = await AgentRunner().run(
        spec(
            fake,
            max_iterations=2,
            hook=InjectAfterIteration(queue, 2),
            injection_callback=queue_callback(queue),
            finalize_on_max_iterations=False,
        )
    )

    assert result.stop_reason == "max_iterations"
    assert result.had_injections is False
    assert [item.content for item in queue] == ["late follow-up after max iters"]


async def test_max_iterations_finalization_keeps_late_input_queued():
    queue: list = []
    fake = FakeProvider([tool_reply(call(call_id="c1")), tool_reply(call(call_id="c2")), "summary"])
    result = await AgentRunner().run(
        spec(
            fake,
            max_iterations=2,
            hook=InjectAfterIteration(queue, 2),
            injection_callback=queue_callback(queue),
        )
    )

    assert result.stop_reason == "max_iterations"
    assert result.had_injections is False
    assert len(fake.calls) == 3
    assert "late follow-up" not in str(fake.calls[-1].messages)
    assert [item.content for item in queue] == ["late follow-up after max iters"]
    assert not any(m.get("content") == "late follow-up after max iters" for m in result.messages)


# --- checkpoints ---------------------------------------------------------------


async def test_checkpoints_follow_each_finished_step():
    checkpoints: list[dict] = []

    async def checkpoint(payload):
        checkpoints.append(payload)

    fake = FakeProvider([tool_reply(call()), "done"])
    result = await AgentRunner().run(spec(fake, model="m", checkpoint_callback=checkpoint))

    assistant = result.messages[1]
    tool_message = result.messages[2]
    assert checkpoints == [
        {
            "phase": "awaiting_tools",
            "iteration": 0,
            "model": "m",
            "assistant_message": assistant,
            "completed_tool_results": [],
            "pending_tool_calls": assistant["tool_calls"],
        },
        {
            "phase": "tools_completed",
            "iteration": 0,
            "model": "m",
            "assistant_message": assistant,
            "completed_tool_results": [tool_message],
            "pending_tool_calls": [],
        },
        {
            "phase": "final_response",
            "iteration": 1,
            "model": "m",
            "assistant_message": {"role": "assistant", "content": "done"},
            "completed_tool_results": [],
            "pending_tool_calls": [],
        },
    ]


async def test_answer_kept_for_injected_input_is_checkpointed():
    checkpoints: list[dict] = []
    injections = [[], [{"role": "user", "content": "follow up"}], [], []]

    async def checkpoint(payload):
        checkpoints.append(payload)

    async def inject():
        return injections.pop(0)

    await AgentRunner().run(
        spec(FakeProvider(["first answer", "second answer"]), checkpoint_callback=checkpoint, injection_callback=inject)
    )

    assert [(c["phase"], c["iteration"], c["assistant_message"]["content"]) for c in checkpoints] == [
        ("final_response", 0, "first answer"),
        ("final_response", 1, "second answer"),
    ]


async def test_failures_emit_no_checkpoint():
    checkpoints: list[dict] = []

    async def checkpoint(payload):
        checkpoints.append(payload)

    await AgentRunner().run(
        spec(FakeProvider([LLMResponse(content="x", finish_reason="error")]), checkpoint_callback=checkpoint)
    )
    assert checkpoints == []


# --- cancellation -------------------------------------------------------------


class HangingTool(Tool):
    """Signals ``started`` and waits forever (until cancelled)."""

    def __init__(self, swallow: bool = False):
        self.started = asyncio.Event()
        self.swallow = swallow

    @property
    def name(self):
        return "hang"

    @property
    def description(self):
        return "Waits."

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if not self.swallow:
                raise
        return "swallowed"


async def test_cancel_during_tools_leaves_the_completed_steps_in_checkpoints():
    checkpoints: list[dict] = []

    async def checkpoint(payload):
        checkpoints.append(payload)

    hang = HangingTool()
    fake = FakeProvider([tool_reply(call()), tool_reply(call("hang", {}, "c2"))])
    task = asyncio.create_task(
        AgentRunner().run(spec(fake, tools=[AddTool(), hang], checkpoint_callback=checkpoint))
    )
    await asyncio.wait_for(hang.started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [c["phase"] for c in checkpoints] == ["awaiting_tools", "tools_completed", "awaiting_tools"]
    assert checkpoints[1]["completed_tool_results"] == [
        {"role": "tool", "tool_call_id": "c1", "name": "add", "content": "3"}
    ]
    assert [tc["id"] for tc in checkpoints[2]["pending_tool_calls"]] == ["c2"]


async def test_tool_that_swallows_cancellation_cannot_resume_model_work():
    hang = HangingTool(swallow=True)
    fake = FakeProvider([tool_reply(call("hang", {}, "c1")), "should never be requested"])
    task = asyncio.create_task(AgentRunner().run(spec(fake, tools=[hang])))
    await asyncio.wait_for(hang.started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(fake.calls) == 1


async def test_provider_that_swallows_cancellation_does_not_run_tools():
    started = asyncio.Event()
    executed: list = []

    class Recording(AddTool):
        async def execute(self, a, b):
            executed.append((a, b))
            return "3"

    async def on_call(n):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass  # swallowed: the provider returns its reply anyway

    fake = ScriptedProvider([tool_reply(call())], on_call)
    task = asyncio.create_task(AgentRunner().run(spec(fake, tools=[Recording()])))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert executed == []


async def test_cancelled_run_reports_partial_messages_to_run_hooks():
    seen: list[AgentRunHookContext] = []

    class Capture(AgentHook):
        async def on_finally(self, context):
            seen.append(context)

    hang = HangingTool()
    fake = FakeProvider([tool_reply(call()), tool_reply(call("hang", {}, "c2"))])
    task = asyncio.create_task(AgentRunner().run(spec(fake, tools=[AddTool(), hang], hook=Capture())))
    await asyncio.wait_for(hang.started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert seen[0].stop_reason == "cancelled"
    assert [m["role"] for m in seen[0].messages] == ["user", "assistant", "tool", "assistant"]


def test_hook_context_types_unchanged():
    # had_injections lives on the run context only (nanobot hook.py:47).
    assert "had_injections" in AgentRunHookContext.__slots__
    assert "had_injections" not in AgentHookContext.__slots__
