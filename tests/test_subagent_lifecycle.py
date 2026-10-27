"""SubagentManager lifecycle: spawn, run, announce, cancel, hook, tool wiring.

Ported from nanobot c4dac64 tests/agent/test_subagent_lifecycle.py,
tests/agent/tools/test_subagent_tools.py and tests/agent/test_subagent.py.
Left out: nanobot's deprecated constructor / ``set_provider`` paths (NanoClaw
never had them), ``llm_usage_source`` attribution, ToolLoader and file-tool
toggles (no counterparts), and the tests of nanobot's parallel tool batches
and ``_run_agent_loop`` signature. ``spec.runtime`` reads become NanoClaw's
flat ``spec.temperature`` / ``spec.provider``.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from nanoclaw.agent.hook import AgentHookContext
from nanoclaw.agent.runner import AgentRunResult
from nanoclaw.agent.subagent import SubagentManager, SubagentStatus, _SubagentHook
from nanoclaw.agent.tools.base import is_tool_error as is_tool_error_result
from nanoclaw.agent.tools.context import RequestContext, current_request_context, request_context
from nanoclaw.agent.tools.subagent import SubagentTool
from nanoclaw.bus.queue import MessageBus
from nanoclaw.bus.runtime_events import SubagentTaskChanged
from nanoclaw.config.schema import AgentDefaults, ExecToolConfig
from nanoclaw.providers import FakeProvider, LLMResponse, LLMUsage, ToolCallRequest
from nanoclaw.security.workspace_access import build_workspace_scope
from nanoclaw.utils.llm_runtime import LLMRuntime


def _manager(tmp_path, **kw) -> SubagentManager:
    kw.setdefault("bus", MessageBus())
    kw.setdefault("max_tool_result_chars", 16_000)
    return SubagentManager(tmp_path, **kw)


def _runtime(provider=None, *, model="test-model", temperature=0.1) -> LLMRuntime:
    return LLMRuntime(
        provider=provider or FakeProvider([]),
        model=model,
        max_tokens=4096,
        temperature=temperature,
        context_window_tokens=128_000,
    )


def _done(content="done"):
    return AgentRunResult(final_content=content, messages=[], stop_reason="completed")


def _make_hook_context(**overrides) -> AgentHookContext:
    defaults = dict(
        iteration=1, tool_calls=[], tool_events=[], messages=[], usage=None, error=None,
        stop_reason="completed", final_content="ok",
    )
    defaults.update(overrides)
    return AgentHookContext(**defaults)


async def _drain_subagent_tasks(sm: SubagentManager) -> None:
    tasks = list(sm._running_tasks.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)


def _blocked(block):
    async def _slow_run(spec):
        await block.wait()
        return _done()

    return _slow_run


async def test_close_cancels_tasks_before_closing_exec_sessions(tmp_path):
    sm = _manager(tmp_path)
    entered = asyncio.Event()

    async def run(spec):
        entered.set()
        await asyncio.Event().wait()

    sm.runner.run = run
    await sm.spawn("task", runtime=_runtime())
    task = next(iter(sm._running_tasks.values()))
    await entered.wait()

    async def close_exec_sessions() -> int:
        assert task.done()
        return 0

    sm._exec_session_manager.close_all = AsyncMock(side_effect=close_exec_sessions)
    await sm.close()
    assert task.cancelled()
    sm._exec_session_manager.close_all.assert_awaited_once()


def test_status_defaults():
    s = SubagentStatus(task_id="abc", label="test", task_description="do stuff", started_at=time.monotonic())
    assert s.phase == "queued"
    assert s.state == "queued"
    assert s.iteration == 0
    assert s.tool_events == []
    assert s.usage is None
    assert s.stop_reason is None
    assert s.error is None


# --- spawn ---------------------------------------------------------------------------------


async def test_spawn_returns_string_with_task_id(tmp_path):
    sm = _manager(tmp_path)
    sm.runner.run = AsyncMock(return_value=_done())
    result = await sm.spawn("do something", runtime=_runtime())
    assert "started" in result and "id:" in result
    await _drain_subagent_tasks(sm)


async def test_spawn_creates_task_in_running_tasks(tmp_path):
    sm = _manager(tmp_path)
    block = asyncio.Event()
    sm.runner.run = _blocked(block)
    await sm.spawn("task", runtime=_runtime(), session_key="s1")
    assert len(sm._running_tasks) == 1
    block.set()
    await _drain_subagent_tasks(sm)
    assert len(sm._running_tasks) == 0


async def test_spawn_creates_status(tmp_path):
    sm = _manager(tmp_path)
    sm.runner.run = AsyncMock(return_value=_done())
    assert not sm.runtime_statuses()
    await sm.spawn("my task", runtime=_runtime())
    statuses = sm.runtime_statuses()
    assert len(statuses) == 1
    assert next(iter(statuses.values())).task_description == "my task"
    await _drain_subagent_tasks(sm)
    assert [s.state for s in sm.statuses_for_session("cli:direct").values()] == ["done"]


async def test_spawn_registers_in_session_tasks(tmp_path):
    sm = _manager(tmp_path)
    block = asyncio.Event()
    sm.runner.run = _blocked(block)
    await sm.spawn("task", runtime=_runtime(), session_key="s1")
    assert len(sm.statuses_for_session("s1")) == 1
    block.set()
    await _drain_subagent_tasks(sm)
    assert [s.state for s in sm.statuses_for_session("s1").values()] == ["done"]


async def test_no_session_key_registers_channel_chat_owner(tmp_path):
    sm = _manager(tmp_path)
    block = asyncio.Event()
    sm.runner.run = _blocked(block)
    await sm.spawn("task", runtime=_runtime())
    assert len(sm.statuses_for_session("cli:direct")) == 1
    block.set()
    await _drain_subagent_tasks(sm)


async def test_label_defaults_to_truncated_task(tmp_path):
    sm = _manager(tmp_path)
    block = asyncio.Event()
    sm.runner.run = _blocked(block)
    await sm.spawn("A" * 50, runtime=_runtime(), session_key="s1")
    assert next(iter(sm.runtime_statuses().values())).label == "A" * 30 + "..."
    block.set()
    await _drain_subagent_tasks(sm)


async def test_custom_label(tmp_path):
    sm = _manager(tmp_path)
    block = asyncio.Event()
    sm.runner.run = _blocked(block)
    await sm.spawn("task", runtime=_runtime(), label="Custom Label", session_key="s1")
    assert next(iter(sm.runtime_statuses().values())).label == "Custom Label"
    block.set()
    await _drain_subagent_tasks(sm)


async def test_cleanup_callback_retains_terminal_status(tmp_path):
    sm = _manager(tmp_path)
    sm.runner.run = AsyncMock(return_value=_done())
    await sm.spawn("task", runtime=_runtime(), session_key="s1")
    await _drain_subagent_tasks(sm)
    assert len(sm._running_tasks) == 0
    assert [s.state for s in sm.statuses_for_session("s1").values()] == ["done"]


async def test_runtime_is_captured_before_background_task_starts(tmp_path):
    sm = _manager(tmp_path)
    runtime = _runtime(temperature=0.2)
    entered, release = asyncio.Event(), asyncio.Event()
    seen = {}

    async def observe(spec):
        seen["spec"] = (spec.provider, spec.temperature)
        request_ctx = current_request_context()
        seen["context_runtime"] = request_ctx.runtime if request_ctx else None
        seen["session_key"] = request_ctx.session_key if request_ctx else None
        entered.set()
        await release.wait()
        return _done()

    sm.runner.run = observe
    reply = await sm.spawn("task", runtime=runtime, session_key="s1")
    await asyncio.wait_for(entered.wait(), timeout=1)
    task_id = reply.split("(id: ")[1].split(")")[0]
    assert seen["spec"] == (runtime.provider, 0.2)
    assert seen["context_runtime"] is runtime
    assert seen["session_key"] == f"subagent:{task_id}"  # the child's own session
    release.set()
    await _drain_subagent_tasks(sm)


async def test_status_changes_are_published(tmp_path):
    sm = _manager(tmp_path)
    seen = []
    sm.bus.subscribe(lambda event: seen.append(event), SubagentTaskChanged)
    sm.runner.run = AsyncMock(return_value=_done())
    reply = await sm.spawn("task", runtime=_runtime(), session_key="s1", origin_channel="cli", origin_chat_id="c")
    await _drain_subagent_tasks(sm)
    while sm.bus.has_pending_events:
        await asyncio.sleep(0)
    task_id = reply.split("(id: ")[1].split(")")[0]
    assert len(seen) >= 3  # created, running, terminal
    assert {(e.task_id, e.context.channel, e.context.chat_id, e.context.session_key) for e in seen} == {
        (task_id, "cli", "c", "s1")
    }


# --- _run_subagent -------------------------------------------------------------------------


async def test_successful_run(tmp_path):
    sm = _manager(tmp_path)
    sm.runner.run = AsyncMock(return_value=_done("Task done!"))
    with patch.object(sm, "_announce_result", new_callable=AsyncMock) as mock_announce:
        await sm.spawn(task="do task", label="label", origin_channel="cli", origin_chat_id="direct", runtime=_runtime())
        await asyncio.gather(*sm._running_tasks.values(), return_exceptions=True)
        mock_announce.assert_called_once()
        assert mock_announce.call_args.args[-2] == "ok"


async def test_exception_run(tmp_path):
    sm = _manager(tmp_path)
    sm.runner.run = AsyncMock(side_effect=RuntimeError("LLM down"))
    with patch.object(sm, "_announce_result", new_callable=AsyncMock) as mock_announce:
        await sm.spawn(task="do task", label="label", origin_channel="cli", origin_chat_id="direct", runtime=_runtime())
        await asyncio.gather(*sm._running_tasks.values(), return_exceptions=True)
        status = next(iter(sm.runtime_statuses().values()))
        assert status.phase == "error"
        assert "LLM down" in status.error
        assert mock_announce.call_args.args[-2] == "error"


async def test_status_updated_on_success(tmp_path):
    sm = _manager(tmp_path)
    sm.runner.run = AsyncMock(return_value=_done("ok"))
    with patch.object(sm, "_announce_result", new_callable=AsyncMock):
        await sm.spawn(task="do task", label="label", origin_channel="cli", origin_chat_id="direct", runtime=_runtime())
        await asyncio.gather(*sm._running_tasks.values(), return_exceptions=True)
        status = next(iter(sm.runtime_statuses().values()))
        assert status.phase == "done"
        assert status.stop_reason == "completed"


# --- _announce_result ----------------------------------------------------------------------


async def _announced(tmp_path, *args, **kwargs):
    sm = _manager(tmp_path)
    published = []
    sm.bus.publish_inbound = AsyncMock(side_effect=lambda msg: published.append(msg))
    await sm._announce_result(*args, **kwargs)
    return published


async def test_announce_publishes_inbound_message(tmp_path):
    (msg,) = await _announced(tmp_path, "t1", "label", "task", "result text", {"channel": "cli", "chat_id": "direct"}, "ok")
    assert msg.channel == "system"
    assert msg.sender_id == "subagent"
    assert msg.metadata["injected_event"] == "subagent_result"
    assert msg.metadata["subagent_task_id"] == "t1"
    assert msg.metadata["subagent_message_receipts"] == {}


async def test_announce_session_key_override(tmp_path):
    origin = {"channel": "telegram", "chat_id": "123", "session_key": "s1"}
    (msg,) = await _announced(tmp_path, "t1", "label", "task", "result", origin, "ok")
    assert msg.session_key_override == "s1"


async def test_announce_session_key_override_fallback(tmp_path):
    (msg,) = await _announced(tmp_path, "t1", "label", "task", "result", {"channel": "telegram", "chat_id": "123"}, "ok")
    assert msg.session_key_override == "telegram:123"


@pytest.mark.parametrize(
    ("status", "text"),
    [("ok", "completed successfully"), ("error", "failed"), ("cancelled", "was cancelled"),
     ("incomplete", "stopped before completing the task")],
)
async def test_announce_status_text(tmp_path, status, text):
    (msg,) = await _announced(tmp_path, "t1", "label", "task", "r", {"channel": "cli", "chat_id": "direct"}, status)
    assert msg.content.startswith(f"[Subagent 'label' {text}]")


async def test_announce_origin_message_id_and_receipts(tmp_path):
    (msg,) = await _announced(
        tmp_path, "t1", "label", "task", "result", {"channel": "cli", "chat_id": "direct"}, "ok",
        origin_message_id="msg-123", receipts={"m1": "delivered"},
    )
    assert msg.metadata["origin_message_id"] == "msg-123"
    assert msg.metadata["subagent_message_receipts"] == {"m1": "delivered"}
    assert msg.content.endswith('\nMessage receipts: {"m1": "delivered"}')


# --- cancel_by_session / running counts ----------------------------------------------------


async def test_cancel_by_session_cancels_running_tasks(tmp_path):
    sm = _manager(tmp_path)
    block = asyncio.Event()
    sm.runner.run = _blocked(block)
    runtime = _runtime()
    await sm.spawn("task1", runtime=runtime, session_key="s1")
    await sm.spawn("task2", runtime=runtime, session_key="s1")
    assert len(sm.statuses_for_session("s1")) == 2
    assert await sm.cancel_by_session("s1") == 2
    block.set()
    await _drain_subagent_tasks(sm)


async def test_cancel_by_session_without_tasks_returns_zero(tmp_path):
    assert await _manager(tmp_path).cancel_by_session("nonexistent") == 0


async def test_cancels_active_and_queued_tasks(tmp_path):
    sm = _manager(tmp_path, max_concurrent_subagents=1)
    active_entered, queued_entered = asyncio.Event(), asyncio.Event()

    async def _blocked_run(spec):
        if spec.initial_messages[-1]["content"] == "active":
            active_entered.set()
        else:
            queued_entered.set()
        await asyncio.Event().wait()

    sm.runner.run = _blocked_run
    runtime = _runtime()
    await sm.spawn("active", runtime=runtime, session_key="s1")
    await asyncio.wait_for(active_entered.wait(), timeout=1.0)
    await sm.spawn("queued", runtime=runtime, session_key="s1")
    await asyncio.sleep(0)
    assert not queued_entered.is_set()
    assert await sm.cancel_by_session("s1") == 2
    await asyncio.sleep(0)
    assert not queued_entered.is_set()
    assert sm._running_tasks == {}
    assert [s.state for s in sm.statuses_for_session("s1").values()] == ["cancelled", "cancelled"]


async def test_already_done_not_counted(tmp_path):
    sm = _manager(tmp_path)
    sm.runner.run = AsyncMock(return_value=_done())
    await sm.spawn("task1", runtime=_runtime(), session_key="s1")
    await _drain_subagent_tasks(sm)
    assert await sm.cancel_by_session("s1") == 0


async def test_running_counts(tmp_path):
    sm = _manager(tmp_path)
    assert sm.get_running_count() == 0
    assert sm.get_running_count_by_session("nonexistent") == 0
    block = asyncio.Event()
    sm.runner.run = _blocked(block)
    runtime = _runtime()
    await sm.spawn("t1", runtime=runtime, session_key="s1")
    await sm.spawn("t2", runtime=runtime, session_key="s1")
    assert sm.get_running_count() == 2
    assert sm.get_running_count_by_session("s1") == 2
    block.set()
    await _drain_subagent_tasks(sm)
    assert sm.get_running_count() == 0


# --- _SubagentHook -------------------------------------------------------------------------


async def test_hook_after_iteration_updates_status():
    status = SubagentStatus(task_id="t1", label="test", task_description="do", started_at=time.monotonic())
    hook = _SubagentHook(status)
    usage = LLMUsage.reported(input_tokens=100, output_tokens=0)
    ctx = _make_hook_context(iteration=3, tool_events=[{"name": "read_file", "status": "ok", "detail": ""}], usage=usage)
    await hook.after_iteration(ctx)
    assert status.iteration == 3
    assert len(status.tool_events) == 1
    assert status.usage == usage


async def test_hook_after_iteration_sets_error():
    status = SubagentStatus(task_id="t1", label="test", task_description="do", started_at=time.monotonic())
    await _SubagentHook(status).after_iteration(_make_hook_context(error="something broke"))
    assert status.error == "something broke"


async def test_hook_remembers_the_last_text_and_running_tools():
    status = SubagentStatus(task_id="t1", label="test", task_description="do", started_at=time.monotonic())
    hook = _SubagentHook(status, max_result_chars=5)
    calls = [ToolCallRequest(id="1", name="read_file", arguments={})]
    await hook.before_execute_tools(_make_hook_context(response=LLMResponse(content="looking", tool_calls=calls), tool_calls=calls))
    assert (status.result, status.partial) == ("looki", True)
    assert status.tool_events == [{"name": "read_file", "status": "running"}]
    # An error reply or blank text is never remembered.
    await hook.after_iteration(_make_hook_context(response=LLMResponse(content="Rate limited"), error="Rate limited"))
    await hook.after_iteration(_make_hook_context(response=LLMResponse(content="  ")))
    assert status.result == "looki"


async def test_hook_check_active_runs_before_each_tool():
    stopped = []
    hook = _SubagentHook(check_active=lambda: stopped.append(True))
    await hook.before_execute_tool(_make_hook_context(), ToolCallRequest(id="1", name="x", arguments={}), None, {})
    assert stopped == [True]


# --- tool wiring (nanobot tests/agent/tools/test_subagent_tools.py) ---------------------------


async def test_run_inline_returns_result_without_announcement(tmp_path):
    manager = _manager(tmp_path)
    manager.runner.run = AsyncMock(return_value=AgentRunResult(messages=[], stop_reason="done", final_content="review result"))
    manager._announce_result = AsyncMock()
    result = await manager.run_inline(task="review this", session_key="test:c1", runtime=_runtime())
    assert result == "review result"
    manager._announce_result.assert_not_awaited()
    assert manager._running_tasks == {}
    assert [s.state for s in manager.statuses_for_session("test:c1").values()] == ["done"]


async def test_run_inline_returns_structured_error(tmp_path):
    manager = _manager(tmp_path)
    manager.runner.run = AsyncMock(
        return_value=AgentRunResult(messages=[], stop_reason="error", final_content=None, error="subagent failed")
    )
    result = await manager.run_inline(task="review this", session_key="test:c1", runtime=_runtime())
    assert result == "subagent failed"
    assert is_tool_error_result(result)
    assert manager._running_tasks == {}
    assert [s.state for s in manager.statuses_for_session("test:c1").values()] == ["error"]


async def test_subagent_exec_tool_receives_allowed_env_keys(tmp_path):
    mgr = _manager(tmp_path, exec_config=ExecToolConfig(allowed_env_keys=["GOPATH", "JAVA_HOME"]))
    mgr._announce_result = AsyncMock()
    mgr.runner.run = AsyncMock(return_value=AgentRunResult(messages=[], stop_reason="done", final_content="done"))
    await mgr.spawn(task="do task", label="label", origin_channel="test", origin_chat_id="c1", runtime=_runtime())
    await asyncio.gather(*mgr._running_tasks.values(), return_exceptions=True)
    spec = mgr.runner.run.await_args.args[0]
    exec_tool = spec.tools.get("exec")
    assert exec_tool is not None
    assert exec_tool.allowed_env_keys == ["GOPATH", "JAVA_HOME"]
    assert [s.state for s in mgr.statuses_for_session("test:c1").values()] == ["done"]


async def test_subagent_uses_configured_max_iterations(tmp_path):
    mgr = _manager(tmp_path, max_iterations=37)
    mgr._announce_result = AsyncMock()
    mgr.runner.run = AsyncMock(return_value=AgentRunResult(messages=[], stop_reason="done", final_content="done"))
    await mgr.spawn(task="do task", origin_channel="test", origin_chat_id="c1", runtime=_runtime())
    await asyncio.gather(*mgr._running_tasks.values(), return_exceptions=True)
    spec = mgr.runner.run.await_args.args[0]
    assert spec.max_iterations == 37
    assert (spec.finalize_on_max_iterations, spec.max_iterations_message) == (
        False, "Task stopped at its iteration limit before producing a final response."
    )


async def test_subagent_checkpoint_updates_status_and_preserves_omitted_fields(tmp_path):
    mgr = _manager(tmp_path)
    mgr._announce_result = AsyncMock()
    snapshots = []

    async def fake_run(spec):
        for payload in [
            {"phase": "awaiting_tools", "iteration": 2},
            {"phase": "tools_completed", "iteration": 3},
            {"iteration": 4},
            {"phase": "final_response"},
        ]:
            await spec.checkpoint_callback(payload)
            snapshots.extend(mgr.statuses_for_session("test:c1").values())
        return AgentRunResult(messages=[], stop_reason="done", final_content="done")

    mgr.runner.run = AsyncMock(side_effect=fake_run)
    await mgr.spawn(task="do task", session_key="test:c1", runtime=_runtime())
    await asyncio.gather(*mgr._running_tasks.values(), return_exceptions=True)
    assert [(s.state, s.phase, s.iteration) for s in snapshots] == [
        ("running", "awaiting_tools", 2),
        ("running", "tools_completed", 3),
        ("running", "tools_completed", 4),
        ("running", "final_response", 4),
    ]
    assert [s.state for s in mgr.statuses_for_session("test:c1").values()] == ["done"]


async def test_spawn_forwards_temperature_to_run_spec(tmp_path):
    mgr = _manager(tmp_path)
    mgr._announce_result = AsyncMock()
    parent_runtime = _runtime()
    seen = {}

    async def fake_run(spec):
        seen["temperature"] = spec.temperature
        return AgentRunResult(messages=[], stop_reason="done", final_content="done")

    mgr.runner.run = AsyncMock(side_effect=fake_run)
    await mgr.spawn(task="do task", runtime=parent_runtime, temperature=0.9)
    await asyncio.gather(*mgr._running_tasks.values(), return_exceptions=True)
    assert seen["temperature"] == 0.9
    assert parent_runtime.temperature == 0.1


async def test_background_spawn_waits_for_concurrency_capacity(tmp_path):
    mgr = _manager(tmp_path, max_concurrent_subagents=1)
    mgr._announce_result = AsyncMock()
    first_entered, second_entered, release_first, release_second = (asyncio.Event() for _ in range(4))

    async def fake_run(spec):
        if spec.initial_messages[-1]["content"] == "first task":
            first_entered.set()
            await release_first.wait()
        else:
            second_entered.set()
            await release_second.wait()
        return AgentRunResult(messages=[], stop_reason="done", final_content="done")

    mgr.runner.run = AsyncMock(side_effect=fake_run)
    tool = SubagentTool(mgr)
    with request_context(RequestContext(channel="test", chat_id="c1", session_key="test:c1", runtime=_runtime())):
        assert "started" in await tool.execute(action="create", task="first task")
        await asyncio.wait_for(first_entered.wait(), timeout=1.0)
        assert "started" in await tool.execute(action="create", task="second task")
        tasks = list(mgr._running_tasks.values())
        await asyncio.sleep(0)
        assert not second_entered.is_set()
        phases = {status.task_description: status.phase for status in mgr.runtime_statuses().values()}
        assert phases == {"first task": "initializing", "second task": "queued"}
    release_first.set()
    await asyncio.wait_for(second_entered.wait(), timeout=1.0)
    release_second.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)
    assert mgr._running_tasks == {}


async def test_subagent_tool_waits_for_inline_result():
    class Manager:
        def __init__(self):
            self.inline = AsyncMock(return_value="review result")
            self.spawn = AsyncMock(return_value="started")

        async def run_inline(self, **kwargs):
            return await self.inline(**kwargs)

    manager = Manager()
    tool = SubagentTool(manager)  # type: ignore[arg-type]
    with request_context(RequestContext(channel="test", chat_id="c1", session_key="test:c1", runtime=_runtime())):
        result = await tool.execute(action="create", task="review this", wait=True)
    assert result == "review result"
    manager.inline.assert_awaited_once()
    manager.spawn.assert_not_awaited()
    kwargs = manager.inline.await_args.kwargs
    assert (kwargs["session_key"], kwargs["origin_channel"], kwargs["origin_chat_id"]) == ("test:c1", "test", "c1")
    assert kwargs["session_policy"].persist is True


async def test_inline_spawn_waits_for_concurrency_capacity(tmp_path):
    manager = _manager(tmp_path, max_concurrent_subagents=1)
    first_entered, second_entered, release_first, release_second = (asyncio.Event() for _ in range(4))

    async def fake_run(spec):
        if spec.initial_messages[-1]["content"] == "first":
            first_entered.set()
            await release_first.wait()
        else:
            second_entered.set()
            await release_second.wait()
        return AgentRunResult(messages=[], stop_reason="done", final_content="done")

    manager.runner.run = AsyncMock(side_effect=fake_run)
    tool = SubagentTool(manager)
    with request_context(RequestContext(channel="test", chat_id="c1", session_key="test:c1", runtime=_runtime())):
        first = asyncio.create_task(tool.execute(action="create", task="first", wait=True))
        await asyncio.wait_for(first_entered.wait(), timeout=1.0)
        second = asyncio.create_task(tool.execute(action="create", task="second", wait=True))
        await asyncio.sleep(0)
        assert not second.done()
        assert not second_entered.is_set()
        assert manager.get_running_count() == 2
        release_first.set()
        assert await first == "done"
        await asyncio.wait_for(second_entered.wait(), timeout=1.0)
        release_second.set()
        assert await second == "done"
    assert manager.get_running_count() == 0
    assert [s.state for s in manager.statuses_for_session("test:c1").values()] == ["done", "done"]


async def test_cancel_by_session_cancels_inline_subagent(tmp_path):
    manager = _manager(tmp_path)
    entered = asyncio.Event()

    async def fake_run(spec):
        entered.set()
        await asyncio.Event().wait()

    manager.runner.run = AsyncMock(side_effect=fake_run)
    inline = asyncio.create_task(manager.run_inline(task="wait", session_key="test:c1", runtime=_runtime()))
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    assert await manager.cancel_by_session("test:c1") == 1
    result = await inline
    assert is_tool_error_result(result)
    assert result == "Task cancelled."
    assert manager._running_tasks == {}
    assert [s.state for s in manager.statuses_for_session("test:c1").values()] == ["cancelled"]


def test_subagent_defaults_match_agent_defaults(tmp_path):
    mgr = _manager(tmp_path)
    assert AgentDefaults().max_concurrent_subagents == 4
    assert mgr.max_concurrent_subagents == AgentDefaults().max_concurrent_subagents
    assert mgr.max_iterations == AgentDefaults().max_tool_iterations


# --- nanobot tests/agent/test_subagent.py ------------------------------------------------------


def test_subagent_tools_have_no_subagent_tool(tmp_path):
    tools = _manager(tmp_path)._build_tools()
    assert tools.has("read_file") and tools.has("write_file")
    assert not tools.has("spawn")
    assert not tools.has("subagent")


async def test_subagent_keeps_project_runtime_scope_with_agent_owned_tools(tmp_path):
    agent_workspace = tmp_path / "agent"
    project = tmp_path / "project"
    agent_workspace.mkdir()
    project.mkdir()
    manager = _manager(agent_workspace)
    manager.runner.run = AsyncMock(return_value=_done("ok"))
    manager._announce_result = AsyncMock()
    await manager.spawn(
        task="task", label="label", origin_channel="websocket", origin_chat_id="direct", runtime=_runtime(),
        workspace_scope=build_workspace_scope(project, "restricted"),
    )
    await asyncio.gather(*manager._running_tasks.values(), return_exceptions=True)
    spec = manager.runner.run.call_args.args[0]
    assert spec.workspace == project
    assert spec.tools.get("read_file")._workspace == agent_workspace


async def test_subagent_recovers_from_tool_error_in_same_run(tmp_path):
    provider = FakeProvider([
        LLMResponse(content="reading", tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"path": "missing.txt"})]),
        LLMResponse(content="recovered without restarting"),
    ])
    sm = _manager(tmp_path)
    result = await sm.run_inline(task="recover after a missing file", session_key="test:direct", runtime=_runtime(provider))
    assert result == "recovered without restarting"
    assert len(provider.calls) == 2
