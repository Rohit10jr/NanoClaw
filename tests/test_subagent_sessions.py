"""Private child sessions (layer 19): ``subagent:<task id>`` in the session store.

nanobot c4dac64 (agent/subagent_sessions.py) keeps each task's transcript and
status in a child session owned by the parent; NanoClaw stores it as a JSONL
session file through SessionManager (outside the workspace).
"""

import asyncio
import json

import pytest

from nanoclaw.agent.subagent import SubagentControlError, SubagentManager
from nanoclaw.agent.subagent_sessions import SubagentSessions
from nanoclaw.agent.subagent_status import SubagentSessionError, SubagentStatus
from nanoclaw.bus import InboundMessage
from nanoclaw.bus.queue import MessageBus
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
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


def manager(ws, sessions, **kwargs):
    return SubagentManager(ws, bus=MessageBus(), session_manager=sessions, **kwargs)


def runtime(provider):
    return LLMRuntime(provider=provider, model="m", max_tokens=100, temperature=0.1, context_window_tokens=0)


def read_call():
    return LLMResponse(
        content="Reading the note.",
        tool_calls=[ToolCallRequest(id="r1", name="read_file", arguments={"path": "note.txt"})],
        finish_reason="tool_calls",
    )


def task_id_of(reply):
    return reply.split("(id: ")[1].split(")")[0]


async def settled(sm):
    await asyncio.gather(*list(sm._running_tasks.values()), return_exceptions=True)
    await asyncio.sleep(0)


async def test_child_session_holds_transcript_and_status(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sm = manager(ws, sessions)
    reply = await sm.spawn("Read the note", label="Note", session_key="cli:a", runtime=runtime(FakeProvider([read_call(), "It says evidence."])))
    task_id = task_id_of(reply)
    await settled(sm)

    on_disk = SessionManager(ws, sessions_root=tmp_path / "sessions").get_existing(f"subagent:{task_id}")
    assert on_disk is not None
    meta = on_disk.metadata
    assert (meta["session_type"], meta["parent_session_key"]) == ("subagent", "cli:a")
    assert list(meta) == ["session_type", "subagent_task", "parent_session_key"]
    task = meta["subagent_task"]
    assert list(task) == [
        "task_id", "label", "task_description", "phase", "owner", "state", "receipts", "result", "partial",
        "iteration", "tool_events", "usage", "stop_reason", "error", "origin_message_id", "origin_turn_id",
        "created_at", "completed_at", "revision", "elapsed_seconds",
    ]
    assert (task["task_id"], task["label"], task["owner"], task["state"], task["phase"]) == (task_id, "Note", "cli:a", "done", "done")
    assert task["result"] == "It says evidence." and task["stop_reason"] == "completed" and task["iteration"] == 1
    assert task["completed_at"] is not None and task["revision"] > 2 and task["tool_events"] == []
    assert isinstance(task["usage"], dict)

    roles = [m["role"] for m in on_disk.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert on_disk.messages[0]["content"] == "Read the note"
    assert on_disk.messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    # The tool activity of a response is kept on it for display (nanobot SessionActivity).
    assert [e["phase"] for e in on_disk.messages[1]["tool_events"]] == ["end"]
    assert on_disk.messages[3]["content"] == "It says evidence."
    # The parent session was created for the child (nanobot create_child).
    assert sessions.get_session_path("cli:a").exists()
    assert sessions.sessions_dir.parent == (tmp_path / "sessions").resolve()


async def test_transcript_and_status_are_saved_while_the_child_runs(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sm = manager(ws, sessions)
    gate, waiting = asyncio.Event(), asyncio.Event()

    class Provider(FakeProvider):
        async def chat(self, messages, *args, **kwargs):
            if len(messages) > 2:  # after the tool result
                waiting.set()
                await gate.wait()
                return LLMResponse(content="done")
            return read_call()

    task_id = task_id_of(await sm.spawn("t", session_key="cli:a", runtime=runtime(Provider([]))))
    await asyncio.wait_for(waiting.wait(), 5)
    header = sessions.read_session_metadata(f"subagent:{task_id}")
    assert header["metadata"]["subagent_task"]["state"] == "running"
    assert header["metadata"]["subagent_task"]["result"] == "Reading the note."  # remembered progress text
    live = SessionManager(ws, sessions_root=tmp_path / "sessions").get_existing(f"subagent:{task_id}")
    assert [m["role"] for m in live.messages] == ["user", "assistant", "tool"]
    gate.set()
    await settled(sm)


async def test_status_survives_eviction_and_restart(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sm = manager(ws, sessions)
    sm.MAX_TERMINAL = 0
    task_id = task_id_of(await sm.spawn("t", session_key="cli:a", runtime=runtime(FakeProvider(["answer"]))))
    await settled(sm)
    assert sm.runtime_statuses() == {}
    assert sm.check(task_id, "cli:a").result == "answer"  # read back from the child session

    restarted = manager(ws, SessionManager(ws, sessions_root=tmp_path / "sessions"))
    status = restarted.check(task_id, "cli:a")
    assert (status.state, status.result, status.owner) == ("done", "answer", "cli:a")
    assert status.finished_at is not None and status.as_dict()["elapsed_seconds"] >= 0
    assert list(restarted.statuses_for_session("cli:a")) == [task_id]
    assert restarted.statuses_for_session("cli:a", include_history=False) == {}
    assert list((await restarted.read_statuses_for_session("cli:a"))) == [task_id]
    with pytest.raises(SubagentControlError):
        restarted.check(task_id, "cli:b")  # another session never sees it


async def test_read_session_is_detached_and_owned(ws, tmp_path):
    sm = manager(ws, store(ws, tmp_path))
    task_id = task_id_of(await sm.spawn("t", session_key="cli:a", runtime=runtime(FakeProvider(["answer"]))))
    await settled(sm)
    status, child = sm.read_session(task_id, "cli:a")
    assert status.state == "done" and [m["role"] for m in child.messages] == ["user", "assistant"]
    child.messages.clear()
    assert len(sm.read_session(task_id, "cli:a")[1].messages) == 2
    with pytest.raises(SubagentControlError):
        sm.read_session(task_id, "cli:b")


async def test_accepts_result_only_for_an_owned_child(ws, tmp_path):
    sm = manager(ws, store(ws, tmp_path))
    task_id = task_id_of(await sm.spawn("t", session_key="cli:a", runtime=runtime(FakeProvider(["answer"]))))
    await settled(sm)
    report = await sm.bus.consume_inbound()
    assert report.metadata["subagent_task_id"] == task_id
    assert sm.accepts_result(report, "cli:a")
    assert not sm.accepts_result(report, "cli:b")
    other = InboundMessage("system", "subagent", "cli:a", "x", metadata={"subagent_task_id": "unknown"})
    assert not sm.accepts_result(other, "cli:a")
    user = InboundMessage("cli", "user", "a", "hi", metadata={"subagent_task_id": "unknown"})
    assert sm.accepts_result(user, "cli:a")  # only subagent reports are checked
    assert SubagentManager(ws).accepts_result(other, "cli:a")  # no session store: nothing to check


async def test_a_report_for_a_missing_owner_is_not_published(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sm = manager(ws, sessions)
    gate = asyncio.Event()

    class Provider(FakeProvider):
        async def chat(self, *args, **kwargs):
            await gate.wait()
            return LLMResponse(content="late")

    await sm.spawn("t", session_key="cli:a", runtime=runtime(Provider([])))
    await asyncio.sleep(0.01)
    sessions.get_session_path("cli:a").unlink()  # the parent went away
    sessions.invalidate("cli:a")
    gate.set()
    await settled(sm)
    assert sm.bus.inbound_size == 0


async def test_recover_interrupted_marks_unfinished_tasks(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sm = manager(ws, sessions)
    gate = asyncio.Event()

    class Provider(FakeProvider):
        async def chat(self, *args, **kwargs):
            await gate.wait()

    task_id = task_id_of(await sm.spawn("t", session_key="cli:a", runtime=runtime(Provider([]))))
    await asyncio.sleep(0.01)
    receipt = await sm.send(task_id, "cli:a", "please hurry")
    # A new process finds the task still "running" on disk.
    restarted = manager(ws, SessionManager(ws, sessions_root=tmp_path / "sessions"))
    await restarted.recover_interrupted()
    status = restarted.check(task_id, "cli:a")
    assert (status.state, status.phase, status.stop_reason) == ("interrupted", "interrupted", "host_restarted")
    assert status.completed_at is not None and status.partial is False
    assert status.receipts == {receipt["message_id"]: "undelivered"}
    sm._closed = True  # the old manager must not report
    gate.set()
    await sm.close()


async def test_a_temporary_child_is_kept_in_memory_only(ws, tmp_path):
    sessions = store(ws, tmp_path)
    sm = manager(ws, sessions)
    result = await sm.run_inline(
        "t", session_key="cli:temp", runtime=runtime(FakeProvider(["answer"])), session_policy=SessionPolicy(persist=False)
    )
    assert result == "answer"
    assert list(sessions.sessions_dir.glob("*.jsonl")) == []
    (task_id,) = sm.statuses_for_session("cli:temp")
    status, child = sm.read_session(task_id, "cli:temp")
    assert status.state == "done" and child.messages[-1]["content"] == "answer"


def test_invalid_task_metadata_is_rejected(ws, tmp_path):
    sessions = store(ws, tmp_path)
    children = SubagentSessions(sessions)
    base = {"session_type": "subagent", "parent_session_key": "cli:a"}
    sessions.save(Session(key="subagent:t1", metadata={**base, "subagent_task": "nope"}))
    with pytest.raises(SubagentSessionError):
        children.status("t1", "cli:a")
    status = SubagentStatus(task_id="t2", label="l", task_description="d", started_at=0.0, owner="cli:a")
    payload = SubagentSessions._payload(status)
    sessions.save(Session(key="subagent:t2", metadata={**base, "subagent_task": {**payload, "elapsed_seconds": -1}}))
    with pytest.raises(SubagentSessionError, match="duration"):
        children.status("t2", "cli:a")
    sessions.save(Session(key="subagent:t3", metadata={**base, "subagent_task": {**payload, "task_id": "t2"}}))
    with pytest.raises(SubagentSessionError, match="ownership"):
        children.status("t3", "cli:a")
    assert children.status("t1", "cli:b") is None  # not this parent's child
    assert json.loads(json.dumps(payload)) == payload
