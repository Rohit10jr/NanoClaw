"""Layer 14: AgentLoop.run() on the message bus (nanobot agent/loop.py:1428-1770).

Session workers, mid-turn injection, the terminal wait for subagents, commands
on the bus, turn delivery (stream ids, final reply, lifecycle) and the session
lock shared with process_direct. Timing is driven by asyncio.Events (a gated
``slow`` tool); waits are tiny polling steps with a timeout.
"""

import asyncio

import pytest

import nanoclaw.agent.loop as loop_module
from bus_rig import Rig, call
from nanoclaw.agent.subagent import SubagentResult
from nanoclaw.bus import InboundMessage
from nanoclaw.bus.outbound_events import ProgressEvent, StreamDeltaEvent, StreamedResponseEvent, StreamEndEvent
from nanoclaw.bus.runtime_events import SessionTurnStarted, TurnCompleted, TurnRunStatusChanged
from nanoclaw.providers import LLMResponse
from nanoclaw.runtime_context import RUNTIME_CONTEXT_HISTORY_META, RUNTIME_CONTEXT_TAG


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    return path


def route_children(provider, child_reply="Sunny, 20C.", gate=None, started=None):
    """Answer subagent requests separately from the main agent's script."""
    original = provider.chat

    async def chat(messages, *args, **kwargs):
        if messages[0]["content"].startswith("# Subagent"):
            if started is not None:
                started.set()
            if gate is not None:
                await gate.wait()
            return LLMResponse(content=child_reply)
        return await original(messages, *args, **kwargs)

    provider.chat = chat


# --- one turn -----------------------------------------------------------------


async def test_single_turn_streams_and_publishes_the_final_reply(ws, tmp_path):
    async with Rig(ws, tmp_path, ["Hi there."]) as rig:
        await rig.send("hello")
        await rig.settle(1)
    deltas = [m for m in rig.outbound if isinstance(m.event, StreamDeltaEvent)]
    ends = [m for m in rig.outbound if isinstance(m.event, StreamEndEvent)]
    assert "".join(m.content for m in deltas) == "Hi there."
    assert len(ends) == 1 and not ends[0].event.resuming
    stream_ids = {m.event.stream_id for m in deltas + ends}
    assert len(stream_ids) == 1 and stream_ids.pop().startswith("cli:a:") and ends[0].event.stream_id.endswith(":0")
    (final,) = rig.replies()
    assert isinstance(final.event, StreamedResponseEvent) and final.content == "Hi there."
    assert (final.channel, final.chat_id) == ("cli", "a")
    assert final.metadata["_wants_stream"] is True and isinstance(final.metadata["latency_ms"], int)
    # Lifecycle facts for local subscribers (nanobot TurnDelivery started/running/complete/idle).
    kinds = [(type(e).__name__, getattr(e, "status", None)) for e in rig.lifecycle]
    assert kinds == [
        ("SessionTurnStarted", None),
        ("TurnRunStatusChanged", "running"),
        ("TurnCompleted", None),
        ("TurnRunStatusChanged", "idle"),
    ]
    assert all(e.context.session_key == "cli:a" for e in rig.lifecycle)
    completed = next(e for e in rig.lifecycle if isinstance(e, TurnCompleted))
    assert completed.outcome == "completed" and completed.latency_ms is not None
    assert [m["role"] for m in rig.saved()] == ["user", "assistant"]
    assert "pending_user_turn" not in rig.session().metadata


async def test_without_wants_stream_only_progress_and_the_reply_are_sent(ws, tmp_path):
    async with Rig(ws, tmp_path, [call("echo", {"text": "x"}, content="Checking."), "Done."]) as rig:
        await rig.send("go", stream=False)
        await rig.settle(1)
    kinds = [type(m.event).__name__ if m.event else None for m in rig.outbound]
    assert "StreamDeltaEvent" not in kinds and "StreamEndEvent" not in kinds
    progress = [m.content for m in rig.outbound if isinstance(m.event, ProgressEvent)]
    assert "Checking." in progress and 'echo("x")' in progress
    assert rig.outbound[-1].event is None and rig.outbound[-1].content == "Done."


async def test_a_turn_that_fails_gets_the_error_reply(ws, tmp_path):
    """nanobot TurnDelivery.fail; the unanswered message is closed next turn."""
    async with Rig(ws, tmp_path, []) as rig:
        original = rig.loop.context.build_messages
        failures = []

        def boom(*args, **kwargs):
            if not failures:
                failures.append(1)
                raise RuntimeError("prompt assembly failed")
            return original(*args, **kwargs)

        rig.loop.context.build_messages = boom
        await rig.send("hello")
        await rig.settle(1)
        assert rig.texts() == ["Sorry, I encountered an error."]
        completed = next(e for e in rig.lifecycle if isinstance(e, TurnCompleted))
        assert (completed.outcome, completed.failure_kind) == ("failed", "internal")
        await rig.send("again")
        await rig.settle(2)
    assert [(m["role"], m["content"][:5]) for m in rig.saved()[:3]] == [
        ("user", "hello"),
        ("assistant", "Error"),
        ("user", "again"),
    ]


# --- injection ------------------------------------------------------------------


async def test_message_queued_during_a_turn_is_injected(ws, tmp_path):
    async with Rig(ws, tmp_path, [call("slow", {"text": "g"}), "Done both."]) as rig:
        await rig.send("first")
        await rig.slow.running("g").wait()
        queued = await rig.send("second")
        rig.slow.gate("g").set()
        await rig.settle(1)
    assert rig.texts() == ["Done both."]
    assert len(rig.provider.calls) == 2
    request = rig.provider.calls[1].messages
    assert [m["role"] for m in request[1:]] == ["user", "assistant", "tool", "user"]
    assert request[-1]["content"].startswith("second") and RUNTIME_CONTEXT_TAG in request[-1]["content"]
    saved = rig.saved()
    assert [m["role"] for m in saved] == ["user", "assistant", "tool", "user", "assistant"]
    injected = saved[3]
    assert injected[RUNTIME_CONTEXT_HISTORY_META]  # saved as the model saw it, with its marker
    assert injected["timestamp"] == queued.timestamp.isoformat()  # when it arrived


async def test_commands_are_answered_at_once_or_queued_as_barriers(ws, tmp_path):
    """While a turn runs, /history is answered inline; /compact (a session-queue
    command) waits on the worker and stops the injection snapshot, so input
    after it is not folded into the running turn (nanobot loop.py:1007, :1140)."""
    script = [call("slow", {"text": "g"}), "Done.", "- summary", "After."]
    async with Rig(ws, tmp_path, script) as rig:
        await rig.send("work")
        await rig.slow.running("g").wait()
        await rig.send("/history")
        await rig.until(lambda: len(rig.replies()) == 1)
        await rig.send("/compact")
        await rig.send("after")
        rig.slow.gate("g").set()
        await rig.settle(4)
    assert rig.texts() == ["Last 1 message(s):\n👤 You: work", "Done.", "", "After."]
    assert len(rig.provider.calls) == 4  # the compaction, then "after" as its own turn
    assert rig.provider.calls[1].messages[-1]["role"] == "tool"


async def test_a_failed_injection_puts_the_snapshot_back_in_order(ws, tmp_path, monkeypatch):
    """nanobot loop.py:1204-1214: on a conversion failure the snapshot goes back
    to the queue ahead of later input; the next drain injects it in order."""
    async with Rig(ws, tmp_path, [call("slow", {"text": "g"}), "First answer.", "Second answer."]) as rig:
        original = rig.loop._injected_user_message
        failures = []

        def flaky(msg, session, **kwargs):
            if not failures:
                failures.append(msg.content)
                raise RuntimeError("conversion failed")
            return original(msg, session, **kwargs)

        monkeypatch.setattr(rig.loop, "_injected_user_message", flaky)
        await rig.send("first")
        await rig.slow.running("g").wait()
        await rig.send("second")
        await rig.send("third")
        rig.slow.gate("g").set()
        await rig.settle(1)
    assert failures == ["second"]
    last = rig.provider.calls[-1].messages
    assert last[-2] == {"role": "assistant", "content": "First answer."}
    # The two injected user messages, merged for the model in their order.
    assert last[-1]["role"] == "user" and last[-1]["content"].startswith("second")
    assert last[-1]["content"].index("second") < last[-1]["content"].index("third")
    assert rig.texts() == ["Second answer."]


# --- subagents ------------------------------------------------------------------


async def test_turn_waits_for_its_subagent_and_folds_the_report_in(ws, tmp_path):
    gate, started = asyncio.Event(), asyncio.Event()
    script = [call("spawn", {"task": "check weather", "label": "Weather"}), "On it.", "It is sunny."]
    async with Rig(ws, tmp_path, script, spawn=True) as rig:
        route_children(rig.provider, gate=gate, started=started)
        await rig.send("weather?")
        await started.wait()
        await rig.until(lambda: len(rig.provider.calls) == 2)
        assert rig.replies() == []  # the turn waits (nanobot _wait_for_pending)
        gate.set()
        await rig.settle(1)
    assert rig.texts() == ["It is sunny."]
    last = rig.provider.calls[-1].messages
    assert last[-2] == {"role": "assistant", "content": "On it."}
    assert last[-1]["content"].startswith("[Subagent 'Weather' completed successfully]")
    assert RUNTIME_CONTEXT_TAG not in last[-1]["content"]  # a report gets no runtime context
    report = rig.saved()[-2]
    assert report["role"] == "user" and report["injected_event"] == "subagent_result"
    assert report["_hidden_history"] == {"kind": "subagent_result", "subagent_task_id": report["subagent_task_id"]}
    # Not shown by /history (nanobot hides it from the display history).
    assert "Subagent" not in "\n".join(m.get("content", "") for m in rig.saved() if m["role"] == "assistant")


async def test_terminal_wait_gives_up_after_the_timeout(ws, tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "_SUBAGENT_TERMINAL_WAIT_SECONDS", 0.05)
    gate = asyncio.Event()
    async with Rig(ws, tmp_path, [call("spawn", {"task": "slow"}), "Started.", "Late report seen."], spawn=True) as rig:
        route_children(rig.provider, gate=gate)
        await rig.send("go")
        await rig.settle(1)
        assert rig.texts() == ["Started."]
        # The late report becomes a follow-up turn: not streamed, printed whole.
        gate.set()
        await rig.settle(2)
    follow_up = rig.replies()[-1]
    assert follow_up.content == "Late report seen." and follow_up.event is None
    assert follow_up.chat_id == "a" and follow_up.metadata == {}
    assert not rig.provider.calls[-1].streamed
    saved = rig.saved()
    assert saved[-2]["role"] == "assistant" and saved[-2]["injected_event"] == "subagent_result"


async def test_report_on_the_bus_round_trips(ws):
    report = SubagentResult("abcd1234", "Weather", "[report]", "incomplete", "cli:a", "cli", "a", "max_iterations", True)
    msg = report.to_inbound()
    assert (msg.channel, msg.sender_id, msg.chat_id, msg.session_key) == ("system", "subagent", "cli:a", "cli:a")
    assert not msg.is_user_input
    assert msg.metadata["subagent_state"] == "incomplete" and msg.metadata["subagent_partial"] is True
    assert SubagentResult.from_inbound(msg) == report


# --- commands, sessions, locks ----------------------------------------------------


async def test_status_while_running_counts_the_working_turn(ws, tmp_path):
    async with Rig(ws, tmp_path, [call("slow", {"text": "g"}), "Finished."]) as rig:
        await rig.send("work")
        await rig.slow.running("g").wait()
        await rig.send("/status")
        await rig.until(lambda: len(rig.replies()) == 1)
        assert "⚡ Tasks: 1 active" in rig.texts()[0]
        rig.slow.gate("g").set()
        await rig.settle(2)
    assert rig.texts()[1] == "Finished."


async def test_two_sessions_run_at_the_same_time(ws, tmp_path):
    script = [call("slow", {"text": "a"}), call("slow", {"text": "b"}, "c2"), "B done.", "A done."]
    async with Rig(ws, tmp_path, script) as rig:
        await rig.send("alpha", chat="a")
        await rig.slow.running("a").wait()
        await rig.send("beta", chat="b")
        await rig.slow.running("b").wait()
        assert set(rig.loop._active_tasks) == {"cli:a", "cli:b"}
        rig.slow.gate("b").set()
        await rig.until(lambda: len(rig.replies("b")) == 1)
        assert rig.replies("a") == []
        rig.slow.gate("a").set()
        await rig.settle(2)
    assert rig.texts("b") == ["B done."] and rig.texts("a") == ["A done."]
    assert {m.chat_id for m in rig.outbound} == {"a", "b"}


async def test_process_direct_waits_for_a_bus_turn_of_the_same_session(ws, tmp_path):
    """Direct calls share the dispatch lock with bus turns (nanobot loop.py:2700)."""
    async with Rig(ws, tmp_path, [call("slow", {"text": "g"}), "Bus done.", "Direct done."]) as rig:
        await rig.send("bus")
        await rig.slow.running("g").wait()
        direct = asyncio.create_task(rig.loop.process_direct("direct", session_key="cli:a"))
        await asyncio.sleep(0.05)
        assert not direct.done()
        rig.slow.gate("g").set()
        result = await asyncio.wait_for(direct, 5)
    assert result.final_content == "Direct done."
    assert [m["content"].split("\n")[0] for m in rig.saved() if m["role"] == "user"] == ["bus", "direct"]


async def test_new_session_takes_the_session_lock(ws, tmp_path):
    """Research D6 (nanobot 9650f4f): clearing waits for the running turn."""
    async with Rig(ws, tmp_path, [call("slow", {"text": "g"}), "Done."]) as rig:
        await rig.send("work")
        await rig.slow.running("g").wait()
        reset = asyncio.create_task(rig.loop.new_session("cli:a"))
        await asyncio.sleep(0.05)
        assert not reset.done()
        rig.slow.gate("g").set()
        await asyncio.wait_for(reset, 5)
        await rig.settle(1)
    assert rig.saved() == []


async def test_new_on_an_idle_session_runs_on_its_worker_and_replies(ws, tmp_path):
    """/new reaches the worker when the session is idle; it holds the lock and
    must not cancel its own worker (deliberate difference, see
    AgentLoop.cancel_active_tasks)."""
    async with Rig(ws, tmp_path, ["Hi."]) as rig:
        await rig.send("hello")
        await rig.settle(1)
        await rig.send("/new")
        await rig.send("hello again")
        await rig.settle(3)
    assert rig.texts() == ["Hi.", "New session started.", "FakeProvider: script exhausted"]
    assert [m["content"].split("\n")[0] for m in rig.saved()][:1] == ["hello again"]


async def test_command_replies_on_the_worker_are_saved_hidden(ws, tmp_path):
    """nanobot loop.py:2185: shortcut commands save a ``_command`` pair, kept
    out of the model's history; /new is not saved."""
    async with Rig(ws, tmp_path, ["Hi."]) as rig:
        await rig.send("/help")
        await rig.settle(1)
        await rig.send("hello")
        await rig.settle(2)
    saved = rig.saved()
    assert [(m["role"], m.get("_command")) for m in saved] == [
        ("user", True),
        ("assistant", True),
        ("user", None),
        ("assistant", None),
    ]
    assert [m["role"] for m in rig.provider.calls[0].messages] == ["system", "user"]


async def test_stop_and_aclose_end_run(ws, tmp_path):
    rig = Rig(ws, tmp_path, [call("slow", {"text": "g"})])
    async with rig:
        await rig.send("work")
        await rig.slow.running("g").wait()
    assert rig.slow.cancelled == ["g"]  # aclose cancelled the worker
    assert rig.loop._active_tasks == {} and rig.loop._pending_queues == {}
    # The interrupted turn was saved like a /stop (its checkpoint restored).
    assert [m["role"] for m in rig.saved()] == ["user", "assistant", "tool"]


async def test_system_message_without_override_routes_to_the_named_chat(ws, tmp_path):
    """A system message's ``chat_id`` names ``channel:chat`` for the reply
    (nanobot turn_delivery.py:184). Without an override its session is
    ``system:<chat_id>``, as in nanobot (loop.py:1001, :1881)."""
    async with Rig(ws, tmp_path, ["Noted."]) as rig:
        msg = InboundMessage(channel="system", sender_id="subagent", chat_id="cli:b", content="[report]")
        await rig.loop.bus.publish_inbound(msg)
        await rig.settle(1)
    (reply,) = rig.replies()
    assert (reply.channel, reply.chat_id, reply.content) == ("cli", "b", "Noted.")
    assert rig.saved("system:cli:b")[0]["content"] == "[report]"
    assert not any(isinstance(e, SessionTurnStarted) for e in rig.lifecycle)  # no lifecycle for system turns
    assert any(isinstance(e, TurnRunStatusChanged) and e.status == "idle" for e in rig.lifecycle)


# --- mutation-check gaps (layer 14) -------------------------------------------------


async def test_each_streamed_segment_gets_its_own_stream_id(ws, tmp_path):
    """nanobot TurnDelivery: a segment that ends without ``merge_next`` moves the
    stream id on, so text before tools and the answer are separate messages."""
    async with Rig(ws, tmp_path, [call("echo", {"text": "x"}, content="Checking."), "Done."]) as rig:
        await rig.send("go")
        await rig.settle(1)
    ends = [m.event for m in rig.outbound if isinstance(m.event, StreamEndEvent)]
    deltas = [m.event for m in rig.outbound if isinstance(m.event, StreamDeltaEvent)]
    assert [e.stream_id.rsplit(":", 1)[1] for e in ends] == ["0", "1"]
    assert {d.stream_id for d in deltas if d.content == "Checking."} == {ends[0].stream_id}
    assert {d.stream_id for d in deltas if d.content == "Done."} == {ends[1].stream_id}


async def test_a_system_turn_publishes_no_progress(ws, tmp_path):
    """Progress is a lifecycle notification: only user turns publish it
    (nanobot bus/notification_delivery.py:34)."""
    script = [call("echo", {"text": "x"}, content="Checking."), "Noted."]
    async with Rig(ws, tmp_path, script) as rig:
        msg = InboundMessage(channel="system", sender_id="subagent", chat_id="cli:b", content="[report]")
        await rig.loop.bus.publish_inbound(msg)
        await rig.settle(1)
    assert rig.texts() == ["Noted."]
    assert not any(isinstance(m.event, ProgressEvent) for m in rig.outbound)


async def test_input_a_command_produces_while_busy_joins_the_running_turn(ws, tmp_path):
    """nanobot loop.py:849: a command answered inline that returns agent input
    has that input queued on the session (here: injected into the running turn)."""
    async with Rig(ws, tmp_path, [call("slow", {"text": "g"}), "Done."]) as rig:

        async def again(ctx):
            return InboundMessage(
                channel=ctx.msg.channel,
                sender_id=ctx.msg.sender_id,
                chat_id=ctx.msg.chat_id,
                content="again please",
                metadata={"original_command": "/again"},
            )

        rig.loop.commands.exact("/again", again)
        await rig.send("work")
        await rig.slow.running("g").wait()
        await rig.send("/again")
        await rig.until(lambda: rig.loop._pending_queues["cli:a"].qsize() == 1)
        rig.slow.gate("g").set()
        await rig.settle(1)
    assert rig.texts() == ["Done."]
    assert rig.provider.calls[1].messages[-1]["content"].startswith("again please")


async def test_the_worker_keeps_serving_after_one_message_fails(ws, tmp_path, monkeypatch):
    """nanobot loop.py:1594: a failure on one message is logged and the worker
    continues FIFO with the rest of the session's inbox."""
    async with Rig(ws, tmp_path, ["Second done."]) as rig:
        original = rig.loop._dispatch_one

        async def flaky(msg, pending):
            if msg.content == "boom":
                raise RuntimeError("dispatch failed")
            return await original(msg, pending)

        monkeypatch.setattr(rig.loop, "_dispatch_one", flaky)
        # Both arrive before the worker starts, so they share its inbox.
        await rig.send("boom")
        await rig.send("second")
        await rig.settle(1)
    assert rig.texts() == ["Second done."]


async def test_cancelling_a_worker_before_it_started_drops_its_inbox(ws, tmp_path):
    """A worker cancelled before its first step never ran its own cleanup;
    cancel_active_tasks drops its inbox so later input starts a new worker,
    and a queued session command still gets its completion."""
    async with Rig(ws, tmp_path, ["Fresh."]) as rig:
        loop = rig.loop
        first = InboundMessage(channel="cli", sender_id="user", chat_id="a", content="never")
        compact = InboundMessage(channel="cli", sender_id="user", chat_id="a", content="/compact")
        await loop._enqueue_session_message(first)
        await loop._enqueue_session_message(compact)
        assert await loop.cancel_active_tasks("cli:a") == 1
        assert loop._pending_queues == {} and loop._active_tasks == {}
        await rig.until(lambda: rig.texts() == [""])  # the queued /compact's empty completion
        await rig.send("hello")
        await rig.settle(2)
    assert rig.texts() == ["", "Fresh."]
    assert all("never" not in m.get("content", "") for m in rig.saved())
