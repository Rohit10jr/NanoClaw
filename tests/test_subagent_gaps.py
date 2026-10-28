"""Layer 19 behaviour the mutation checks found untested.

Child sessions and statuses (subagent_sessions.py, subagent_status.py,
session_activity.py), SubagentManager details (cancel, finish, runner and
hook arguments, ownership boundaries) and the tool's origin fields.
"""

import asyncio
import json
import math
import time

import pytest

from nanoclaw.agent import subagent_sessions as subagent_sessions_module
from nanoclaw.agent.hook import AgentHookContext
from nanoclaw.agent.runner import AgentRunResult
from nanoclaw.agent.session_activity import SessionActivity
from nanoclaw.agent.subagent import SubagentControlError, SubagentManager
from nanoclaw.agent.subagent_sessions import SubagentSessions
from nanoclaw.agent.subagent_status import SubagentSessionError, SubagentStatus
from nanoclaw.agent.tools.context import RequestContext, request_context
from nanoclaw.agent.tools.subagent import SubagentTool
from nanoclaw.bus import InboundMessage
from nanoclaw.bus.outbound_events import ProgressEvent
from nanoclaw.bus.queue import MessageBus
from nanoclaw.bus.runtime_events import SubagentTaskChanged
from nanoclaw.providers import FakeProvider, LLMResponse, LLMUsage, ToolCallRequest
from nanoclaw.session import Session, SessionManager, SessionPolicy
from nanoclaw.utils.llm_runtime import LLMRuntime


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    (path / "note.txt").write_text("evidence\n", encoding="utf-8")
    return path


def store(ws, tmp_path):
    return SessionManager(ws, sessions_root=tmp_path / "sessions")


def runtime(provider, **kwargs):
    kwargs.setdefault("context_window_tokens", 0)
    return LLMRuntime(provider=provider, model="m", max_tokens=100, temperature=0.1, **kwargs)


def task_id_of(reply):
    return reply.split("(id: ")[1].split(")")[0]


async def settled(sm):
    await asyncio.gather(*list(sm._running_tasks.values()), return_exceptions=True)
    await asyncio.sleep(0)


def status(task_id="t1", owner="cli:a", **kwargs):
    return SubagentStatus(task_id=task_id, label="l", task_description="d", started_at=time.monotonic(), owner=owner, **kwargs)


def child_file(sessions, task_id, owner="cli:a", **task):
    """A saved child session whose task status has the given fields."""
    payload = SubagentSessions._payload(status(task_id, owner))
    payload.update(task)
    sessions.save(
        Session(
            key=f"subagent:{task_id}",
            metadata={"session_type": "subagent", "parent_session_key": owner, "subagent_task": payload},
        )
    )


# --- subagent_status.py -----------------------------------------------------------


def test_status_as_dict_elapsed_usage_and_detached_receipts():
    running = status(receipts={"m1": "accepted"}, usage=LLMUsage(input_tokens=3, output_tokens=2))
    running.started_at = time.monotonic() - 5
    data = running.as_dict()
    assert data["elapsed_seconds"] >= 5  # a running task's clock keeps going
    assert data["usage"] == LLMUsage(input_tokens=3, output_tokens=2).to_dict()
    data["receipts"]["m1"] = "tampered"
    assert running.receipts == {"m1": "accepted"}
    future = status()
    future.started_at = time.monotonic() + 100
    assert future.as_dict()["elapsed_seconds"] == 0.0  # never negative
    assert "partial" in data and data["partial"] is False


# --- subagent_sessions.py: reading and validation --------------------------------


def test_contains_requires_the_subagent_session_type(ws, tmp_path):
    sessions = store(ws, tmp_path)
    children = SubagentSessions(sessions)
    # A session that names a parent but is not a subagent session is not a child.
    sessions.save(Session(key="subagent:fake", metadata={"parent_session_key": "cli:a"}))
    assert not children.contains("fake", "cli:a")
    child_file(sessions, "real")
    assert children.contains("real", "cli:a")


def test_snapshot_checks_the_owner_itself(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sessions.save(sessions.get_or_create("cli:a"))
    sessions.save(sessions.get_or_create("cli:b"))
    child_file(sessions, "t1")
    children = SubagentSessions(sessions)
    assert children.snapshot("t1", "cli:a") is not None
    assert children.snapshot("t1", "cli:b") is None


def test_payload_is_detached_from_the_status(ws, tmp_path):
    children = SubagentSessions(store(ws, tmp_path))
    task = status("t1", "cli:temp")
    children.create(task, policy=SessionPolicy(persist=False))
    task.receipts["m1"] = "accepted"  # not saved
    assert children.status("t1", "cli:temp").receipts == {}
    children.save(task)
    assert children.status("t1", "cli:temp").receipts == {"m1": "accepted"}


@pytest.mark.parametrize("elapsed", [True, math.inf, math.nan, "1"])
def test_invalid_durations_are_rejected(ws, tmp_path, elapsed):
    sessions = store(ws, tmp_path)
    child_file(sessions, "t1", elapsed_seconds=elapsed)
    with pytest.raises(SubagentSessionError, match="duration"):
        SubagentSessions(sessions).status("t1", "cli:a")


def test_a_saved_duration_is_restored(ws, tmp_path):
    sessions = store(ws, tmp_path)
    child_file(sessions, "done", elapsed_seconds=12.5, completed_at=time.time(), state="done")
    child_file(sessions, "running", elapsed_seconds=3.0, state="running")
    children = SubagentSessions(sessions)
    done = children.status("done", "cli:a")
    assert done.finished_at is not None and done.as_dict()["elapsed_seconds"] == 12.5
    running = children.status("running", "cli:a")
    assert running.finished_at is None and running.as_dict()["elapsed_seconds"] >= 3.0


def test_read_session_metadata_needs_a_metadata_line(ws, tmp_path):
    sessions = store(ws, tmp_path)
    path = sessions.get_session_path("subagent:odd")
    path.write_text(json.dumps({"key": "subagent:odd", "metadata": {"session_type": "subagent"}}) + "\n", encoding="utf-8")
    assert sessions.read_session_metadata("subagent:odd") is None
    assert all(h["key"] != "subagent:odd" for h in sessions.list_session_headers())


# --- subagent_sessions.py: recovery -------------------------------------------------


async def test_recover_interrupted_seals_only_active_tasks(ws, tmp_path):
    sessions = store(ws, tmp_path)
    child_file(sessions, "stopping", state="stopping", result="half done", revision=4)
    child_file(sessions, "queued", state="queued", revision=1)
    child_file(sessions, "done", state="done", result="all done", completed_at=time.time(), revision=7)
    # An unreadable child is logged and skipped; the others are still recovered.
    sessions.save(
        Session(
            key="subagent:broken",
            metadata={"session_type": "subagent", "parent_session_key": "cli:a", "subagent_task": "nope"},
        )
    )
    await SubagentSessions(sessions).interrupt_pending()

    children = SubagentSessions(SessionManager(ws, sessions_root=tmp_path / "sessions"))
    stopping = children.status("stopping", "cli:a")
    assert (stopping.state, stopping.stop_reason, stopping.partial, stopping.revision) == (
        "interrupted", "host_restarted", True, 5,
    )
    queued = children.status("queued", "cli:a")
    assert (queued.state, queued.partial, queued.revision) == ("interrupted", False, 2)
    done = children.status("done", "cli:a")
    assert (done.state, done.stop_reason, done.revision) == ("done", None, 7)


# --- subagent_sessions.py: temporary children ------------------------------------------


async def test_temporary_children_are_listed_after_terminal_eviction(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sm = SubagentManager(ws, bus=MessageBus(), session_manager=sessions)
    sm.MAX_TERMINAL = 0
    await sm.run_inline("t", session_key="cli:temp", runtime=runtime(FakeProvider(["answer"])), session_policy=SessionPolicy(persist=False))
    assert sm.runtime_statuses() == {}
    (status_,) = sm.statuses_for_session("cli:temp").values()
    assert (status_.state, status_.result) == ("done", "answer")


async def test_temporary_children_are_bounded_and_evicted_saves_publish_nothing(ws, tmp_path, monkeypatch):
    monkeypatch.setattr(subagent_sessions_module, "MAX_TRANSIENT", 1)
    sessions = store(ws, tmp_path)
    sm = SubagentManager(ws, bus=MessageBus(), session_manager=sessions)
    temp = SessionPolicy(persist=False)
    gate, entered = asyncio.Event(), asyncio.Event()

    class Blocked(FakeProvider):
        async def chat(self, *args, **kwargs):
            entered.set()
            await gate.wait()
            return LLMResponse(content="late")

    changes = []

    async def record(event):
        changes.append(event.task_id)

    sm.bus.subscribe(record, SubagentTaskChanged)
    first = asyncio.create_task(sm.run_inline("first", session_key="cli:temp", runtime=runtime(Blocked([])), session_policy=temp))
    await entered.wait()
    await sm.run_inline("second", session_key="cli:temp", runtime=runtime(FakeProvider(["two"])), session_policy=temp)
    first_id = next(tid for tid, s in sm.runtime_statuses().items() if s.task_description == "first")
    await sm.bus.drain()
    before = changes.count(first_id)
    gate.set()
    await first
    await sm.bus.drain()
    # The evicted child is gone from the store: its later saves are not published.
    assert changes.count(first_id) == before
    assert [s.task_description for s in sm.sessions.load("cli:temp").values()] == ["second"]


# --- SubagentManager: origin, cancel, inbox ---------------------------------------------


def tool_ctx(provider, **kwargs):
    kwargs.setdefault("session_key", "owner")
    return RequestContext("websocket", runtime=runtime(provider), chat_id="route", **kwargs)


async def test_task_records_its_origin_message_and_turn(ws):
    sm = SubagentManager(ws, bus=MessageBus())
    ctx = tool_ctx(FakeProvider(["done"]), message_id="m-1", turn_id="owner:42")
    with request_context(ctx):
        task_id = task_id_of(await SubagentTool(sm).execute(action="create", task="t"))
        await settled(sm)
        data = json.loads(await SubagentTool(sm).execute(action="check", task_id=task_id))
    assert (data["origin_message_id"], data["origin_turn_id"]) == ("m-1", "owner:42")
    report = await sm.bus.consume_inbound()
    assert report.metadata["origin_message_id"] == "m-1"


async def test_cancel_before_the_task_begins_still_reports(ws):
    sm = SubagentManager(ws, bus=MessageBus())
    task_id = task_id_of(await sm.spawn("t", session_key="cli:a", runtime=runtime(FakeProvider(["never"]))))
    # No yield since spawn: the execution has not begun.
    assert not sm._tasks[task_id].begun
    status_ = await sm.cancel(task_id, "cli:a")
    assert status_.state == "cancelled"
    await settled(sm)
    assert sm.bus.inbound_size == 1
    report = await sm.bus.consume_inbound()
    assert report.metadata["subagent_task_id"] == task_id and report.metadata["subagent_state"] == "cancelled"


def resistant(entered, release, reply):
    """A provider whose request ignores the first cancellation until ``release``."""

    class Provider(FakeProvider):
        calls = 0

        async def chat(self, *args, **kwargs):
            Provider.calls += 1
            if Provider.calls > 1:
                return LLMResponse(content="again")
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
                return reply

    return Provider([])


async def test_a_stopping_task_is_saved_as_stopping(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sm = SubagentManager(ws, bus=MessageBus(), session_manager=sessions)
    sm.CANCEL_WAIT_SECONDS = 0.02
    entered, release = asyncio.Event(), asyncio.Event()
    provider = resistant(entered, release, LLMResponse(content="late"))
    task_id = task_id_of(await sm.spawn("t", session_key="cli:a", runtime=runtime(provider)))
    await entered.wait()
    assert (await sm.cancel(task_id, "cli:a")).state == "stopping"
    header = sessions.read_session_metadata(f"subagent:{task_id}")
    assert header["metadata"]["subagent_task"]["state"] == "stopping"
    release.set()
    await settled(sm)
    assert sm.check(task_id, "cli:a").state == "cancelled"
    assert sm.sessions._live == {}  # a finished child is read from its file


async def test_a_message_is_never_delivered_after_cancel(ws):
    sm = SubagentManager(ws, bus=MessageBus())
    sm.CANCEL_WAIT_SECONDS = 0.02
    entered, release = asyncio.Event(), asyncio.Event()
    # The request swallows the cancellation and answers; the queued message must stay undelivered.
    provider = resistant(entered, release, LLMResponse(content="answer after cancel"))
    task_id = task_id_of(await sm.spawn("t", session_key="cli:a", runtime=runtime(provider)))
    await entered.wait()
    receipt = await sm.send(task_id, "cli:a", "follow-up")
    await sm.cancel(task_id, "cli:a")
    release.set()
    await settled(sm)
    status_ = sm.check(task_id, "cli:a")
    assert status_.state == "cancelled"
    assert status_.receipts == {receipt["message_id"]: "undelivered"}
    assert type(provider).calls == 1


async def test_delivered_receipts_are_saved_before_the_next_request(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sm = SubagentManager(ws, bus=MessageBus(), session_manager=sessions)
    seen = []

    class Provider(FakeProvider):
        async def chat(self, messages, *args, **kwargs):
            (task_id,) = sm._tasks
            if not seen:
                seen.append(await sm.send(task_id, "cli:a", "also check the date"))
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCallRequest(id="r1", name="read_file", arguments={"path": "note.txt"})],
                    finish_reason="tool_calls",
                )
            header = sessions.read_session_metadata(f"subagent:{task_id}")
            seen.append(header["metadata"]["subagent_task"]["receipts"])
            return LLMResponse(content="done")

    await sm.spawn("t", session_key="cli:a", runtime=runtime(Provider([])))
    await settled(sm)
    assert seen[1] == {seen[0]["message_id"]: "delivered"}


# --- SubagentManager: finish, hook, runner arguments ------------------------------------


async def test_terminal_status_is_bounded(ws):
    sm = SubagentManager(ws, bus=MessageBus(), max_tool_result_chars=10)

    async def run(spec):
        return AgentRunResult(final_content="x" * 50, messages=[], stop_reason="completed")

    sm.runner.run = run
    task_id = task_id_of(await sm.spawn("t", label="L" * 300, session_key="cli:a", runtime=runtime(FakeProvider([]))))
    await settled(sm)
    status_ = sm.check(task_id, "cli:a")
    assert status_.result == "x" * 10
    assert status_.label == "L" * 256


async def test_running_tools_are_observed_and_cleared_when_cancelled(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sm = SubagentManager(ws, bus=MessageBus(), session_manager=sessions, max_tool_result_chars=10)
    entered = asyncio.Event()

    async def run(spec):
        calls = [ToolCallRequest(id="c1", name="list_dir", arguments={})]
        context = AgentHookContext(
            iteration=0,
            messages=list(spec.initial_messages),
            response=LLMResponse(content="y" * 50, tool_calls=calls),
            tool_calls=calls,
        )
        await spec.hook.before_execute_tools(context)
        entered.set()
        await asyncio.Event().wait()

    sm.runner.run = run
    task_id = task_id_of(await sm.spawn("t", session_key="cli:a", runtime=runtime(FakeProvider([]))))
    await entered.wait()
    running = sm.check(task_id, "cli:a")
    assert running.result == "y" * 10 and running.partial  # the progress text, bounded
    saved = sessions.read_session_metadata(f"subagent:{task_id}")["metadata"]["subagent_task"]
    assert saved["tool_events"] == [{"name": "list_dir", "status": "running"}]
    await sm.cancel(task_id, "cli:a")
    await settled(sm)
    assert sm.check(task_id, "cli:a").tool_events == []


async def test_the_runners_transcript_and_summary_become_the_child_session(ws, tmp_path):
    from nanoclaw.session.summary import SessionSummaryCheckpoint

    sm = SubagentManager(ws, bus=MessageBus(), session_manager=store(ws, tmp_path))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "t"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "final"},
    ]

    async def run(spec):
        return AgentRunResult(
            final_content="final",
            messages=messages,
            stop_reason="completed",
            summary_checkpoint=SessionSummaryCheckpoint("summary", 3),
        )

    sm.runner.run = run
    task_id = task_id_of(await sm.spawn("t", session_key="cli:a", runtime=runtime(FakeProvider([]))))
    await settled(sm)
    _, child = sm.read_session(task_id, "cli:a")
    contents = [m["content"] for m in child.messages]
    # The summary replaces the two non-system messages before the boundary.
    assert contents[:2] == ["t", "first"] and contents[3:] == ["more", "final"]
    assert child.last_archived == 2


async def test_runner_and_compaction_arguments(ws, tmp_path):
    specs, summaries = [], []

    class Consolidator:
        async def summarize_transcript(self, accepted, previous, **kwargs):
            summaries.append(kwargs)
            return "summary"

    sm = SubagentManager(ws, bus=MessageBus(), max_tool_result_chars=123, session_manager=store(ws, tmp_path))
    sm.consolidator = Consolidator()

    async def run(spec):
        specs.append(spec)
        await spec.consolidate_history([{"role": "user", "content": "x"}], None)
        return AgentRunResult(final_content="ok", messages=[], stop_reason="completed")

    sm.runner.run = run
    rt = runtime(FakeProvider([]), context_window_tokens=50_000)
    saved_id = task_id_of(await sm.spawn("t", session_key="cli:a", runtime=rt))
    await settled(sm)
    await sm.run_inline("t", session_key="cli:temp", runtime=rt, session_policy=SessionPolicy(persist=False))
    saved, temporary = specs
    assert saved.max_tool_result_chars == temporary.max_tool_result_chars == 123
    assert saved.context_window_tokens == 50_000
    assert saved.workspace == ws and temporary.workspace is None  # a temporary child writes no tool-result files
    assert summaries[0]["persist"] is False and summaries[0]["session_key"] == f"subagent:{saved_id}"
    assert summaries[0]["runtime"] is rt


# --- SubagentManager: ownership boundaries and merged views ----------------------------------


async def test_a_removed_child_session_is_no_longer_readable(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sm = SubagentManager(ws, bus=MessageBus(), session_manager=sessions)
    task_id = task_id_of(await sm.spawn("t", session_key="cli:a", runtime=runtime(FakeProvider(["answer"]))))
    await settled(sm)
    assert sm.check(task_id, "cli:a").result == "answer"
    sessions.get_session_path(f"subagent:{task_id}").unlink()
    sessions.invalidate(f"subagent:{task_id}")
    with pytest.raises(SubagentControlError):
        sm.check(task_id, "cli:a")
    assert sm.statuses_for_session("cli:a") == {}


async def test_runtime_state_wins_over_an_older_saved_state_while_cleaning_up(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sm = SubagentManager(ws, bus=MessageBus(), session_manager=sessions)
    cleaning, release = asyncio.Event(), asyncio.Event()
    task_id = task_id_of(await sm.spawn("t", session_key="cli:a", runtime=runtime(FakeProvider(["answer"]))))
    record = sm._tasks[task_id]
    original = record.exec_manager.close_all

    async def cleanup():
        cleaning.set()
        await release.wait()
        return await original()

    record.exec_manager.close_all = cleanup
    await cleaning.wait()
    assert sessions.read_session_metadata(f"subagent:{task_id}")["metadata"]["subagent_task"]["state"] == "running"
    assert sm.statuses_for_session("cli:a")[task_id].state == "stopping"
    release.set()
    await settled(sm)
    assert sm.statuses_for_session("cli:a")[task_id].state == "done"


async def test_child_file_reads_do_not_touch_the_parents_read_state(ws, tmp_path):
    """The child has its own read state: its reads neither use nor replace
    the parent session's, so the parent's repeat read still gets the stub."""
    from nanoclaw.agent.loop import AgentLoop

    def call(name, args, call_id):
        return LLMResponse(
            content=None, tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=args)], finish_reason="tool_calls"
        )

    child_requests = []

    class Provider(FakeProvider):
        async def chat(self, messages, *args, **kwargs):
            if messages[0]["content"].startswith("# Subagent"):
                child_requests.append(messages)
                if len(child_requests) == 1:
                    return call("read_file", {"path": "note.txt"}, "k1")
                return LLMResponse(content="child done")
            return await super().chat(messages, *args, **kwargs)

    provider = Provider([
        call("read_file", {"path": "note.txt"}, "p1"),
        call("subagent", {"action": "create", "task": "read the note", "wait": True}, "p2"),
        call("read_file", {"path": "note.txt"}, "p3"),
        "all done",
    ])
    loop = AgentLoop(provider, ws, sessions=store(ws, tmp_path))
    await loop.process_direct("go")
    tool_result = child_requests[1][-1]
    assert tool_result["role"] == "tool" and "evidence" in tool_result["content"]
    parent_reread = provider.calls[-1].messages[-1]
    assert parent_reread["tool_call_id"] == "p3"
    assert parent_reread["content"].startswith("[File unchanged since last read")


# --- child sessions: queued children, session activity --------------------------------------


async def test_a_queued_child_is_saved_before_it_runs(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sm = SubagentManager(ws, bus=MessageBus(), session_manager=sessions, max_concurrent_subagents=1)
    gate = asyncio.Event()

    class Blocked(FakeProvider):
        async def chat(self, *args, **kwargs):
            await gate.wait()
            return LLMResponse(content="done")

    await sm.spawn("occupy", session_key="cli:a", runtime=runtime(Blocked([])))
    await asyncio.sleep(0.01)
    queued = task_id_of(await sm.spawn("waits", session_key="cli:a", runtime=runtime(Blocked([]))))
    await asyncio.sleep(0.01)
    header = SessionManager(ws, sessions_root=tmp_path / "sessions").read_session_metadata(f"subagent:{queued}")
    assert header is not None and header["metadata"]["subagent_task"]["state"] == "queued"
    gate.set()
    await settled(sm)


def tool_event(call_id, phase="start"):
    return ProgressEvent(tool_events=[{"version": 1, "phase": phase, "call_id": call_id, "name": "list_dir"}])


async def test_session_activity_keeps_tool_rows_only():
    activity = SessionActivity()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    await activity.before_iteration(AgentHookContext(iteration=0, messages=messages))
    assert activity.remember(ProgressEvent(content="thinking")) is False
    # Activity before the response exists is shown as its own display row...
    assert activity.remember(tool_event("c1")) is True
    rows = activity.transcript()
    assert rows[0] == {"role": "user", "content": "t"}
    assert rows[1]["role"] == "assistant" and rows[1]["content"] == ""
    assert [e["call_id"] for e in rows[1]["tool_events"]] == ["c1"]
    # ...and moves onto the response once it arrives; repeated phases are folded per call.
    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]})
    rows = activity.transcript()
    assert len(rows) == 2 and rows[1]["content"] is None
    assert [(e["call_id"], e["phase"]) for e in rows[1]["tool_events"]] == [("c1", "start")]
    activity.remember(tool_event("c1", "end"))
    activity.remember(tool_event("c2"))
    rows = activity.transcript()
    assert len(rows) == 2
    assert [(e["call_id"], e["phase"]) for e in rows[1]["tool_events"]] == [("c1", "end"), ("c2", "start")]
    assert "tool_events" not in messages[-1]  # the provider messages are never changed


# --- the subagent tool -------------------------------------------------------------------------


async def test_check_without_task_id_lists_only_retained_tasks(ws, tmp_path):
    sm = SubagentManager(ws, bus=MessageBus(), session_manager=store(ws, tmp_path))
    sm.MAX_TERMINAL = 0
    ctx = tool_ctx(FakeProvider(["done"]), session_key="cli:a")
    with request_context(ctx):
        task_id = task_id_of(await SubagentTool(sm).execute(action="create", task="t", label="café ✓"))
        await settled(sm)
        assert json.loads(await SubagentTool(sm).execute(action="check")) == {"tasks": []}
        single = await SubagentTool(sm).execute(action="check", task_id=task_id)
    assert '"label": "café ✓"' in single  # not ASCII-escaped


async def test_create_needs_a_session_identity(ws):
    sm = SubagentManager(ws, bus=MessageBus())
    ctx = RequestContext("", session_key=None, runtime=runtime(FakeProvider(["x"])), chat_id="")
    with request_context(ctx):
        result = await SubagentTool(sm).execute(action="create", task="t")
    assert result == "Error: create requires an active session identity"
    assert sm.runtime_statuses() == {}


async def test_a_temporary_session_always_waits(ws):
    sm = SubagentManager(ws, bus=MessageBus())
    ctx = tool_ctx(FakeProvider(["child answer"]), persist_session=False, can_receive_background_results=True)
    with request_context(ctx):
        result = await SubagentTool(sm).execute(action="create", task="t")
    assert result == "child answer"
    assert sm.bus.inbound_size == 0


async def test_create_passes_the_turns_workspace_scope(ws, tmp_path):
    from nanoclaw.security.workspace_access import bind_workspace_scope, build_workspace_scope, reset_workspace_scope

    project = tmp_path / "project"
    project.mkdir()
    sm = SubagentManager(ws, bus=MessageBus())
    specs = []

    async def run(spec):
        specs.append(spec)
        return AgentRunResult(final_content="ok", messages=[], stop_reason="completed")

    sm.runner.run = run
    token = bind_workspace_scope(build_workspace_scope(project, "restricted"))
    try:
        with request_context(tool_ctx(FakeProvider([]))):
            assert await SubagentTool(sm).execute(action="create", task="t", wait=True) == "ok"
    finally:
        reset_workspace_scope(token)
    assert specs[0].workspace == project.resolve()


async def test_a_temporary_turn_never_logs_the_childs_task(ws, tmp_path):
    from loguru import logger

    from nanoclaw.agent.loop import AgentLoop
    from nanoclaw.cli.log_control import set_nanoclaw_logs

    def call(name, args):
        return LLMResponse(content=None, tool_calls=[ToolCallRequest(id="c1", name=name, arguments=args)], finish_reason="tool_calls")

    class Provider(FakeProvider):
        async def chat(self, messages, *args, **kwargs):
            if messages[0]["content"].startswith("# Subagent"):
                return LLMResponse(content="child done")
            return await super().chat(messages, *args, **kwargs)

    lines = []
    sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    set_nanoclaw_logs(True)
    try:
        loop = AgentLoop(Provider([call("subagent", {"action": "create", "task": "classified errand"}), "ok"]), ws, sessions=store(ws, tmp_path))
        await loop.process_direct("go", session_key="cli:temp", ephemeral=True)
    finally:
        set_nanoclaw_logs(False)
        logger.remove(sink)
    assert any("Started subagent" in line for line in lines)
    assert not any("classified errand" in line for line in lines)


# --- loop wiring --------------------------------------------------------------------------------


def test_the_loop_gives_children_its_consolidator(ws, tmp_path):
    from nanoclaw.agent.loop import AgentLoop

    loop = AgentLoop(FakeProvider([]), ws, sessions=store(ws, tmp_path))
    assert loop.subagents.consolidator is loop.consolidator


async def test_a_running_child_is_pending_loop_work(ws, tmp_path):
    from nanoclaw.agent.loop import AgentLoop

    loop = AgentLoop(FakeProvider([]), ws, sessions=store(ws, tmp_path))
    gate = asyncio.Event()

    class Blocked(FakeProvider):
        async def chat(self, *args, **kwargs):
            await gate.wait()
            return LLMResponse(content="done")

    await loop.subagents.spawn("t", session_key="cli:a", runtime=runtime(Blocked([])))
    await asyncio.sleep(0.01)
    await loop.bus.drain()
    assert loop._has_pending_work()
    gate.set()
    await settled(loop.subagents)
    await loop.aclose()


async def test_the_loop_seals_interrupted_children_when_it_starts(ws, tmp_path):
    from bus_rig import Rig

    sessions = store(ws, tmp_path)
    child_file(sessions, "left", state="running")
    async with Rig(ws, tmp_path) as rig:
        await rig.until(lambda: SubagentSessions(store(ws, tmp_path)).status("left", "cli:a").state == "interrupted")


async def test_a_foreign_report_is_not_injected_into_a_running_turn(ws, tmp_path):
    from bus_rig import Rig, call

    async with Rig(ws, tmp_path, [call("slow", {"text": "g"}), "Done."]) as rig:
        await rig.send("work")
        await rig.slow.running("g").wait()
        report = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id="cli:a",
            content="[foreign report]",
            session_key_override="cli:a",
            metadata={"injected_event": "subagent_result", "subagent_task_id": "not-mine"},
        )
        await rig.loop.bus.publish_inbound(report)
        await asyncio.sleep(0.05)
        rig.slow.gate("g").set()
        await rig.settle(1)
    assert rig.texts() == ["Done."]
    assert not any("[foreign report]" in str(m.get("content")) for c in rig.provider.calls for m in c.messages)
    assert not any("[foreign report]" in str(m.get("content")) for m in rig.saved())


async def test_a_bus_turn_gives_its_child_the_message_and_turn_ids(ws, tmp_path):
    from bus_rig import Rig, call

    script = [
        call("subagent", {"action": "create", "task": "one"}), "Started one.", "Reported one.",
        call("subagent", {"action": "create", "task": "two"}), "Started two.", "Reported two.",
    ]
    async with Rig(ws, tmp_path, script, subagent_tool=True) as rig:
        original = rig.provider.chat

        async def chat(messages, *args, **kwargs):
            if messages[0]["content"].startswith("# Subagent"):
                return LLMResponse(content="child done")
            return await original(messages, *args, **kwargs)

        rig.provider.chat = chat
        await rig.send("first", message_id="m-1", webui_turn_id="w-1")
        await rig.settle(1)
        await rig.send("second", message_id="m-2")
        await rig.settle(2)
        statuses = {s.task_description: s for s in rig.loop.subagents.statuses_for_session("cli:a").values()}
    assert (statuses["one"].origin_message_id, statuses["one"].origin_turn_id) == ("m-1", "w-1")
    assert statuses["two"].origin_message_id == "m-2"
    assert statuses["two"].origin_turn_id.startswith("cli:a:") and statuses["two"].origin_turn_id != "cli:a:"
