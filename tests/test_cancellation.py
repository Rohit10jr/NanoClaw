"""Layer 14: /stop cancels a session's turn; the cancelled turn is saved.

nanobot ``_cancel_active_tasks`` (loop.py:948), the cancel branch of
``_dispatch_one`` (loop.py:1716-1745: close the stream, restore the latest
runtime checkpoint, finish the turn even without one, 421bd6b) and
``restore_runtime_checkpoint`` (session/recovery.py:278).
"""

import asyncio
from datetime import datetime

import pytest

from bus_rig import Rig, call
from nanoclaw.bus.outbound_events import StreamDeltaEvent, StreamEndEvent
from nanoclaw.bus.runtime_events import TurnRunStatusChanged
from nanoclaw.providers import LLMResponse
from nanoclaw.session import Session, SessionManager
from nanoclaw.session.recovery import (
    INTERRUPTED_TOOL_MESSAGE,
    PENDING_USER_TURN_KEY,
    RUNTIME_CHECKPOINT_KEY,
    restore_runtime_checkpoint,
)


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    return path


def tool_call(call_id, name="slow", text="g"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": f'{{"text": "{text}"}}'}}


# --- /stop on the bus -------------------------------------------------------------


async def test_stop_during_a_tool_call_saves_the_interrupted_round(ws, tmp_path):
    async with Rig(ws, tmp_path, [call("slow", {"text": "g"}, content="Let me wait."), "never"]) as rig:
        await rig.send("go")
        await rig.slow.running("g").wait()
        await rig.send("/stop")
        await rig.settle(1)
        assert rig.texts() == ["Stopped 1 task(s)."]
        await rig.send("/stop")
        await rig.settle(2)
    assert rig.texts()[1] == "No active task to stop."
    assert rig.slow.cancelled == ["g"]
    saved = rig.saved()
    assert [m["role"] for m in saved] == ["user", "assistant", "tool"]
    assert saved[1]["content"] == "Let me wait." and saved[1]["tool_calls"][0]["id"] == "c1"
    assert saved[2]["tool_call_id"] == "c1" and saved[2]["content"] == INTERRUPTED_TOOL_MESSAGE
    assert saved[2]["_recovery_interrupted"] is True
    metadata = rig.session().metadata
    assert PENDING_USER_TURN_KEY not in metadata and RUNTIME_CHECKPOINT_KEY not in metadata
    assert len(rig.provider.calls) == 1  # the model never ran again
    # The turn still went idle (the chat closes its output on that).
    assert [e.status for e in rig.lifecycle if isinstance(e, TurnRunStatusChanged)][-1] == "idle"


async def test_the_next_turn_sees_the_saved_partial_work(ws, tmp_path):
    async with Rig(ws, tmp_path, [call("slow", {"text": "g"}), "Fresh start."]) as rig:
        await rig.send("go")
        await rig.slow.running("g").wait()
        await rig.send("/stop")
        await rig.settle(1)
        await rig.send("try again")
        await rig.settle(2)
    request = rig.provider.calls[-1].messages
    assert [m["role"] for m in request[1:]] == ["user", "assistant", "tool", "user"]
    assert request[3]["content"] == INTERRUPTED_TOOL_MESSAGE


async def test_only_the_latest_round_survives_a_cancel(ws, tmp_path):
    """As in nanobot: the checkpoint holds the latest round only, so earlier
    finished rounds of the cancelled turn are not saved (seen the same in
    nanobot c4dac64 with the comparison harness)."""
    script = [call("echo", {"text": "e"}, "c1"), call("slow", {"text": "g"}, "c2"), "never"]
    async with Rig(ws, tmp_path, script) as rig:
        await rig.send("go")
        await rig.slow.running("g").wait()
        await rig.send("/stop")
        await rig.settle(1)
    saved = rig.saved()
    assert [(m["role"], m.get("tool_call_id")) for m in saved] == [("user", None), ("assistant", None), ("tool", "c2")]


async def test_cancel_before_any_checkpoint_still_saves_the_turn(ws, tmp_path):
    """nanobot 421bd6b: finish the turn even without a checkpoint, so the user
    message stays and no "interrupted" reply is added later."""
    model_called, never = asyncio.Event(), asyncio.Event()
    async with Rig(ws, tmp_path, ["Second."]) as rig:
        original = rig.provider.chat

        async def hang_first(messages, *args, **kwargs):
            if not model_called.is_set():
                model_called.set()
                await never.wait()
            return await original(messages, *args, **kwargs)

        rig.provider.chat = hang_first
        await rig.send("first")
        await model_called.wait()
        await rig.send("/stop")
        await rig.settle(1)
        assert [m["role"] for m in rig.saved()] == ["user"]
        assert PENDING_USER_TURN_KEY not in rig.session().metadata
        await rig.send("second")
        await rig.settle(2)
    assert [m["content"].split("\n")[0] for m in rig.saved()] == ["first", "second", "Second."]


async def test_cancel_closes_an_open_stream(ws, tmp_path):
    """nanobot ``abort_stream``: the answer streamed and the turn then waits
    for its subagent; /stop ends the open stream segment."""
    gate, started = asyncio.Event(), asyncio.Event()
    async with Rig(ws, tmp_path, [call("subagent", {"action": "create", "task": "slow"}), "Started."], subagent_tool=True) as rig:
        original = rig.provider.chat

        async def chat(messages, *args, **kwargs):
            if messages[0]["content"].startswith("# Subagent"):
                started.set()
                await gate.wait()
                return LLMResponse(content="late")
            return await original(messages, *args, **kwargs)

        rig.provider.chat = chat
        await rig.send("go")
        await started.wait()
        await rig.until(lambda: any(isinstance(m.event, StreamDeltaEvent) and "Started" in m.content for m in rig.outbound))
        assert not isinstance(rig.outbound[-1].event, StreamEndEvent)
        await rig.send("/stop")
        await rig.settle(1)
    kinds = [type(m.event).__name__ if m.event else m.content for m in rig.outbound]
    assert kinds[-2:] == ["StreamEndEvent", "Stopped 2 task(s)."]  # the worker and the subagent
    assert rig.loop.subagents.get_running_count() == 0


async def test_stop_drops_queued_input_and_completes_queued_commands(ws, tmp_path):
    """nanobot ``_cancel_pending_messages``: queued messages go away; a queued
    session command still gets its (empty) completion."""
    async with Rig(ws, tmp_path, [call("slow", {"text": "g"})]) as rig:
        await rig.send("go")
        await rig.slow.running("g").wait()
        await rig.send("/compact")
        await rig.send("queued text")
        await rig.send("/stop")
        await rig.settle(2)
    assert rig.texts() == ["", "Stopped 1 task(s)."]
    assert all("queued text" not in m.get("content", "") for m in rig.saved())


async def test_ctrl_c_shutdown_saves_the_running_turn(ws, tmp_path):
    """Closing the loop (the chat's exit path) cancels workers the same way."""
    async with Rig(ws, tmp_path, [call("slow", {"text": "g"}, content="Working.")]) as rig:
        await rig.send("go")
        await rig.slow.running("g").wait()
    assert [m["role"] for m in rig.saved()] == ["user", "assistant", "tool"]


# --- restore_runtime_checkpoint -------------------------------------------------------


def checkpoint_session(**checkpoint):
    session = Session(key="cli:a")
    session.add_message("user", "go")
    session.metadata[PENDING_USER_TURN_KEY] = True
    session.metadata[RUNTIME_CHECKPOINT_KEY] = checkpoint
    return session


def test_restore_without_a_checkpoint_does_nothing():
    session = Session(key="cli:a")
    assert restore_runtime_checkpoint(session) is False
    assert session.messages == []


def test_restore_turns_pending_calls_into_interrupted_results():
    assistant = {"role": "assistant", "content": "Working.", "tool_calls": [tool_call("c1"), tool_call("c2", "echo")]}
    done = {"role": "tool", "tool_call_id": "c1", "name": "slow", "content": "slow:g"}
    session = checkpoint_session(
        phase="awaiting_tools",
        assistant_message=assistant,
        completed_tool_results=[done],
        pending_tool_calls=[tool_call("c2", "echo"), {"id": ""}, "junk"],
    )
    assert restore_runtime_checkpoint(session) is True
    rows = session.messages[1:]
    assert [(m["role"], m.get("tool_call_id"), m.get("name")) for m in rows] == [
        ("assistant", None, None),
        ("tool", "c1", "slow"),
        ("tool", "c2", "echo"),
    ]
    assert rows[2]["content"] == INTERRUPTED_TOOL_MESSAGE and rows[2]["_recovery_interrupted"] is True
    assert all(datetime.fromisoformat(m["timestamp"]) for m in rows)
    assert PENDING_USER_TURN_KEY not in session.metadata and RUNTIME_CHECKPOINT_KEY not in session.metadata


def test_restore_does_not_repeat_rows_already_saved():
    assistant = {"role": "assistant", "content": "Done.", "tool_calls": None}
    session = checkpoint_session(
        phase="final_response", assistant_message=assistant, completed_tool_results=[], pending_tool_calls=[]
    )
    session.messages.append(dict(assistant))
    assert restore_runtime_checkpoint(session) is True
    assert [m["content"] for m in session.messages] == ["go", "Done."]


async def test_a_checkpoint_left_by_a_crash_is_restored_at_the_next_turn(ws, tmp_path):
    """nanobot ``_restore_turn``: the checkpoint first, then the turn runs."""
    async with Rig(ws, tmp_path, ["Back again."]) as rig:
        session = rig.session()
        session.add_message("user", "go")
        session.metadata[PENDING_USER_TURN_KEY] = True
        session.metadata[RUNTIME_CHECKPOINT_KEY] = {
            "phase": "awaiting_tools",
            "assistant_message": {"role": "assistant", "content": "", "tool_calls": [tool_call("c1")]},
            "completed_tool_results": [],
            "pending_tool_calls": [tool_call("c1")],
        }
        rig.loop.sessions.save(session)
        await rig.send("hello")
        await rig.settle(1)
    roles = [m["role"] for m in rig.saved()]
    assert roles == ["user", "assistant", "tool", "user", "assistant"]
    assert rig.saved()[2]["content"] == INTERRUPTED_TOOL_MESSAGE


async def test_checkpoints_are_kept_while_the_turn_runs(ws, tmp_path):
    """The loop stores each checkpoint in the session (nanobot loop.py:1082) and
    clears it when the turn is saved."""
    async with Rig(ws, tmp_path, [call("slow", {"text": "g"}), "Done."]) as rig:
        await rig.send("go")
        await rig.slow.running("g").wait()
        stored = rig.session().metadata[RUNTIME_CHECKPOINT_KEY]
        assert stored["phase"] == "awaiting_tools"
        assert [c["id"] for c in stored["pending_tool_calls"]] == ["c1"]
        rig.slow.gate("g").set()
        await rig.settle(1)
    assert RUNTIME_CHECKPOINT_KEY not in rig.session().metadata


async def test_the_cancelled_turn_is_written_to_disk(ws, tmp_path):
    """The restored checkpoint is saved, not only kept in the cached session."""
    async with Rig(ws, tmp_path, [call("slow", {"text": "g"}, content="Let me wait.")]) as rig:
        await rig.send("go")
        await rig.slow.running("g").wait()
        await rig.send("/stop")
        await rig.settle(1)
        on_disk = SessionManager(ws, sessions_root=tmp_path / "sessions").get_or_create("cli:a")
    assert [m["role"] for m in on_disk.messages] == ["user", "assistant", "tool"]
    assert on_disk.messages[2]["content"] == INTERRUPTED_TOOL_MESSAGE
    assert RUNTIME_CHECKPOINT_KEY not in on_disk.metadata and PENDING_USER_TURN_KEY not in on_disk.metadata
