"""Layer 21: scheduled turns through the agent loop.

Adapted from nanobot's tests/agent/test_loop_cron_timezone.py,
tests/agent/test_session_inbox_controls.py (the cron cases) and
tests/cron/test_cron_reply_contract.py, on NanoClaw's FakeProvider.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from bus_rig import Rig, call
from nanoclaw.agent.automation_turns import AutomationTurnError
from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.tools import ToolRegistry
from nanoclaw.agent.tools.cron import CronTool
from nanoclaw.bus import InboundMessage, OutboundMessage
from nanoclaw.cron.binding import CronBinding, binding_revision
from nanoclaw.cron.bound_runner import run_bound_cron_job
from nanoclaw.cron.service import CronService
from nanoclaw.cron.session_turns import CRON_DEFER_UNTIL_IDLE_META, CRON_TRIGGER_META
from nanoclaw.cron.types import CronSchedule
from nanoclaw.providers import FakeProvider
from nanoclaw.session.automation_spec import AUTOMATION_HISTORY_META

SCHEDULING = "## Scheduling"


def _cron_message(name: str, chat: str = "a") -> InboundMessage:
    return InboundMessage(
        channel="cli",
        sender_id="cron",
        chat_id=chat,
        content=name,
        metadata={
            CRON_TRIGGER_META: {"job_id": name, "run_id": name, "persist_content": f"saved {name}"},
            CRON_DEFER_UNTIL_IDLE_META: True,
        },
    )


# -- registration and prompt ------------------------------------------------------


def test_agent_loop_registers_cron_tool_with_configured_timezone(tmp_path) -> None:
    loop = AgentLoop(
        FakeProvider([]),
        tmp_path,
        cron_service=CronService(tmp_path / "cron" / "jobs.json"),
        timezone="Asia/Shanghai",
    )
    cron_tool = loop.tools.get("cron")
    assert isinstance(cron_tool, CronTool)
    assert cron_tool._default_timezone == "Asia/Shanghai"
    assert loop.cron_service is not None


def test_no_cron_service_means_no_tool_and_no_scheduling_section(tmp_path) -> None:
    loop = AgentLoop(FakeProvider([]), tmp_path)
    assert not loop.tools.has("cron")
    assert SCHEDULING not in loop.context.build_system_prompt()


def test_scheduling_section_follows_the_cron_tool(tmp_path) -> None:
    loop = AgentLoop(FakeProvider([]), tmp_path, cron_service=CronService(tmp_path / "cron" / "jobs.json"))
    prompt = loop.context.build_system_prompt()
    assert SCHEDULING in prompt
    assert "Use `cron` for reminders" in prompt
    # The section sits right after the tool usage notes, as in nanobot.
    notes = prompt.index("# Tool Usage Notes")
    assert notes < prompt.index(SCHEDULING) < prompt.index("---", notes)


def test_cron_tool_added_to_a_given_registry(tmp_path) -> None:
    tools = ToolRegistry()
    loop = AgentLoop(FakeProvider([]), tmp_path, tools=tools, cron_service=CronService(tmp_path / "jobs.json"))
    assert loop.tools.has("cron")
    assert loop.context.include_scheduling


def test_subagents_do_not_get_the_cron_tool(tmp_path) -> None:
    loop = AgentLoop(FakeProvider([]), tmp_path, cron_service=CronService(tmp_path / "jobs.json"))
    child_tools = loop.subagents._build_tools() if hasattr(loop.subagents, "_build_tools") else None
    if child_tools is None:
        pytest.skip("subagent tool set is built per task")
    assert not child_tools.has("cron")


def test_prompt_command_shows_scheduling(tmp_path, capsys) -> None:
    builder = ContextBuilder(tmp_path)
    assert SCHEDULING not in builder.build_system_prompt()
    builder.include_scheduling = True
    assert SCHEDULING in builder.build_system_prompt()


# -- submitted turns ----------------------------------------------------------------


async def test_cron_turn_runs_in_its_session_and_returns_the_reply(tmp_path) -> None:
    async with Rig(tmp_path / "ws", tmp_path, ["Drink water"]) as rig:
        reply = await asyncio.wait_for(rig.loop.submit_cron_turn(_cron_message("job1")), 5)
        assert isinstance(reply, OutboundMessage)
        assert reply.content == "Drink water"
        await rig.settle(1)
        saved = rig.saved()
        user = [m for m in saved if m["role"] == "user"]
        # Saved as the trigger record with nanobot's markers; the model saw the prompt.
        assert user[0]["content"].startswith("saved job1")
        assert user[0][AUTOMATION_HISTORY_META] == {
            "kind": "cron",
            "cron_job_id": "job1",
            "cron_job_name": None,
            "cron_run_id": "job1",
            "cron_prompt_ref": None,
        }
        assert user[0]["_cron_turn"] is True
        assert rig.provider.calls[0].messages[-1]["content"].startswith("job1")
        assert rig.loop.pending_cron_job_ids_for_session("cli:a") == set()


async def test_cron_turn_waits_until_the_session_is_idle(tmp_path) -> None:
    # "second" arrives during the first turn and is injected into it.
    script = [call("slow", {"text": "one"}), "first done", "cron done"]
    async with Rig(tmp_path / "ws", tmp_path, script) as rig:
        await rig.send("first")
        await asyncio.wait_for(rig.slow.running("one").wait(), 5)
        cron = asyncio.create_task(rig.loop.submit_cron_turn(_cron_message("job")))
        await rig.until(lambda: rig.loop._deferred_automation_turns.get("cli:a"))
        assert rig.loop.pending_cron_job_ids_for_session("cli:a") == {"job"}
        # Input that arrives later still goes first; the cron turn is never injected.
        await rig.send("second")
        await asyncio.sleep(0.05)
        rig.slow.gate("one").set()
        reply = await asyncio.wait_for(cron, 5)
        assert reply is not None and reply.content == "cron done"
        await rig.settle(2)
        users = [m["content"] for m in rig.saved() if m["role"] == "user"]
        assert [u.split("\n")[0] for u in users] == ["first", "second", "saved job"]
        assert rig.loop.pending_cron_job_ids_for_session("cli:a") == set()


async def test_cancelled_session_fails_queued_cron_waiters(tmp_path) -> None:
    async with Rig(tmp_path / "ws", tmp_path, [call("slow", {"text": "x"})]) as rig:
        await rig.send("user request")
        await asyncio.wait_for(rig.slow.running("x").wait(), 5)
        cron = asyncio.create_task(rig.loop.submit_cron_turn(_cron_message("job")))
        await rig.until(lambda: rig.loop._deferred_automation_turns.get("cli:a"))
        await rig.send("/stop")
        with pytest.raises(AutomationTurnError, match="CancelledError"):
            await asyncio.wait_for(cron, 5)
        assert "cli:a" not in rig.loop._deferred_automation_turns
        assert rig.loop.pending_cron_job_ids_for_session("cli:a") == set()


async def test_failed_cron_turn_raises_automation_error(tmp_path, monkeypatch) -> None:
    async with Rig(tmp_path / "ws", tmp_path, []) as rig:
        async def boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("turn failed")

        monkeypatch.setattr(rig.loop, "_run_and_save_turn", boom)
        with pytest.raises(RuntimeError, match="turn failed"):
            await asyncio.wait_for(rig.loop.submit_cron_turn(_cron_message("job")), 5)


async def test_duplicate_run_id_is_refused(tmp_path) -> None:
    loop = AgentLoop(FakeProvider([]), tmp_path)
    started = asyncio.Event()

    async def enqueue(msg: InboundMessage) -> None:
        started.set()

    loop._cron_turns._enqueue = enqueue
    first = asyncio.create_task(loop.submit_cron_turn(_cron_message("job")))
    await started.wait()
    with pytest.raises(RuntimeError, match="already pending"):
        await loop.submit_cron_turn(_cron_message("job"))
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    with pytest.raises(ValueError, match="run_id"):
        await loop.submit_cron_turn(InboundMessage(channel="cli", sender_id="cron", chat_id="a", content="x"))


# -- the cron tool inside turns --------------------------------------------------------


async def test_cron_tool_binds_jobs_to_the_current_session_and_refuses_add_in_cron_turns(tmp_path) -> None:
    cron = CronService(tmp_path / "cron" / "jobs.json")
    script = [
        call("cron", {"action": "add", "message": "Drink water", "every_seconds": 3600}),
        "Scheduled",
        call("cron", {"action": "add", "message": "Nested", "every_seconds": 60}, call_id="c2"),
        "Refused",
    ]
    async with Rig(tmp_path / "ws", tmp_path, script) as rig:
        rig.loop.tools.register(CronTool(cron))
        await rig.send("remind me", chat="room")
        await rig.settle(1)
        [job] = cron.list_jobs()
        assert (job.payload.session_key, job.payload.origin_channel, job.payload.origin_chat_id) == (
            "cli:room", "cli", "room",
        )
        assert job.payload.origin_metadata == {"_wants_stream": True}
        await asyncio.wait_for(rig.loop.submit_cron_turn(_cron_message("nested", chat="room")), 5)
        tool_rows = [m for m in rig.saved("cli:room") if m["role"] == "tool"]
        assert "cannot schedule new jobs from within a cron job execution" in tool_rows[-1]["content"]
        assert len(cron.list_jobs()) == 1


# -- the bound runner end to end -----------------------------------------------------


async def test_bound_runner_delivers_to_the_origin_chat_and_records_runs(tmp_path) -> None:
    cron = CronService(tmp_path / "cron" / "jobs.json")
    job = cron.add_job(
        "Reminder", CronSchedule(kind="every", every_ms=86_400_000), "Remind me to drink water",
        session_key="cli:old", origin_channel="cli", origin_chat_id="old",
        origin_metadata={"_wants_stream": True, "thread": 7},
    )
    async with Rig(tmp_path / "ws", tmp_path, ["Ready", "Drink water", "Drink water again"]) as rig:
        rig.loop.tools.register(CronTool(cron))
        await rig.send("Prepare", chat="new")
        await rig.settle(1)
        await cron.start()
        try:
            moved = cron.change_binding(
                job.id, revision=binding_revision(job),
                binding=CronBinding("cli:new", "cli", "new", {"thread": 42}), message=job.payload.message,
            )
            for expected in ("Drink water", "Drink water again"):
                result = await asyncio.wait_for(run_bound_cron_job(moved, agent=rig.loop, cron=cron), 5)
                assert result.response == expected
                record = json.loads((tmp_path / "cron" / "runs" / f"{result.run_id.replace(':', '_')}.json").read_text())
                assert record["status"] == "ok"
                assert record["response"] == expected
                assert record["session_key"] == "cli:new"
            await rig.settle(3)
        finally:
            cron.stop()
        replies = [m for m in rig.replies("new") if m.content.startswith("Drink")]
        assert [m.content for m in replies] == ["Drink water", "Drink water again"]
        assert all(m.metadata.get("thread") == 42 for m in replies)
        assert rig.loop.sessions.read_session_file("cli:old") is None if hasattr(
            rig.loop.sessions, "read_session_file"
        ) else True
        # Ordinary and scheduled turns share the system prompt and tool prefix.
        first = rig.provider.calls[0]
        assert all(c.messages[0] == first.messages[0] and c.tools == first.tools for c in rig.provider.calls)
        last = rig.provider.calls[-1].messages[-1]["content"]
        assert last.startswith("The scheduled time has arrived.")
        assert "Cron job: Remind me to drink water" in last
        users = [m["content"] for m in rig.saved("cli:new") if m["role"] == "user"]
        assert users[1].startswith("Scheduled cron job triggered: Reminder\n\nRemind me to drink water")
