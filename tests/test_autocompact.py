"""Idle auto-compaction in the agent loop (layer 21).

Adapted from nanobot tests/agent/test_auto_compact.py (c4dac64): the TTL
config, the idle tick and its throttle, which sessions are compacted (expired,
unarchived, not Dream / subagent / busy), the session-lock recheck, summary
hand-off to the next turn and its persistence across restarts.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanoclaw.__main__ import _make_loop as make_cli_loop
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.bus import InboundMessage
from nanoclaw.config import Config
from nanoclaw.events import ContextCompactionEvent
from nanoclaw.providers import FakeProvider
from nanoclaw.session import SessionManager


def make_loop(tmp_path, *script, ttl: int = 15, interval: int = 0) -> AgentLoop:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return AgentLoop(
        FakeProvider(list(script)),
        ws,
        sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"),
        session_ttl_minutes=ttl,
        idle_compact_check_interval_seconds=interval,
    )


def add_turns(session, turns: int, *, prefix: str = "msg") -> None:
    for i in range(turns):
        session.add_message("user", f"{prefix} user {i}")
        session.add_message("assistant", f"{prefix} assistant {i}")


def idle_session(loop: AgentLoop, key: str = "cli:test", turns: int = 3, minutes: int = 20, prefix: str = "old"):
    session = loop.sessions.get_or_create(key)
    add_turns(session, turns, prefix=prefix)
    session.updated_at = datetime.now() - timedelta(minutes=minutes)
    loop.sessions.save(session)
    return session


async def drain(loop: AgentLoop) -> None:
    while loop._background_tasks:
        await asyncio.gather(*tuple(loop._background_tasks), return_exceptions=True)
    await asyncio.sleep(0)


async def scan(loop: AgentLoop, active=()) -> None:
    await loop.auto_compact.check_expired(loop.schedule_background, loop.runtime_for_session, active_session_keys=active)
    await drain(loop)


def outbound_events(loop: AgentLoop) -> list:
    return [loop.bus.outbound.get_nowait().event for _ in range(loop.bus.outbound_size)]


# -- config -------------------------------------------------------------------------


class TestSessionTTLConfig:
    def test_default_ttl_is_fifteen_minutes(self):
        assert Config().agents.defaults.session_ttl_minutes == 15

    def test_explicit_zero_disables_ttl(self):
        config = Config.model_validate({"agents": {"defaults": {"idleCompactAfterMinutes": 0}}})
        assert config.agents.defaults.session_ttl_minutes == 0

    def test_user_friendly_alias_is_supported(self):
        config = Config.model_validate({"agents": {"defaults": {"idleCompactAfterMinutes": 30}}})
        assert config.agents.defaults.session_ttl_minutes == 30

    def test_legacy_alias_is_still_supported(self):
        config = Config.model_validate({"agents": {"defaults": {"sessionTtlMinutes": 45}}})
        assert config.agents.defaults.session_ttl_minutes == 45

    def test_serializes_with_user_friendly_alias(self):
        dumped = Config().model_dump(mode="json", by_alias=True)["agents"]["defaults"]
        assert dumped["idleCompactAfterMinutes"] == 15
        assert "sessionTtlMinutes" not in dumped

    def test_negative_ttl_is_rejected(self):
        with pytest.raises(ValueError):
            Config.model_validate({"agents": {"defaults": {"idleCompactAfterMinutes": -1}}})

    def test_idle_scan_interval_defaults_to_sixty_seconds(self):
        assert Config().agents.defaults.idle_compact_check_interval_seconds == 60

    def test_idle_scan_interval_uses_camel_case_config_key(self):
        config = Config.model_validate({"agents": {"defaults": {"idleCompactCheckIntervalSeconds": 5}}})
        assert config.agents.defaults.idle_compact_check_interval_seconds == 5

    def test_cli_loop_gets_the_config_values(self, tmp_path):
        config = Config.model_validate(
            {"agents": {"defaults": {"idleCompactAfterMinutes": 7, "idleCompactCheckIntervalSeconds": 3}}}
        )
        loop = make_cli_loop(FakeProvider([]), tmp_path, config=config)
        assert loop.auto_compact._ttl == 7
        assert loop._idle_compact_check_interval_s == 3

    def test_loop_default_is_off(self, tmp_path):
        loop = AgentLoop(FakeProvider([]), tmp_path)
        assert loop.auto_compact._ttl == 0


# -- the idle tick ---------------------------------------------------------------------


class TestIdleScanThrottling:
    async def test_configured_idle_scan_interval_throttles_checks(self, tmp_path, monkeypatch):
        ticks = iter((1_000.0, 1_000.0, 1_009.999, 1_010.0))
        monkeypatch.setattr(
            "nanoclaw.agent.loop.time", SimpleNamespace(time=time.time, time_ns=time.time_ns, monotonic=lambda: next(ticks))
        )
        loop = make_loop(tmp_path, interval=10)
        loop.auto_compact.check_expired = AsyncMock()

        await loop._check_expired_sessions_if_due()
        loop.auto_compact.check_expired.assert_called_once()
        await loop._check_expired_sessions_if_due()
        loop.auto_compact.check_expired.assert_called_once()
        await loop._check_expired_sessions_if_due()
        assert loop.auto_compact.check_expired.call_count == 2

    async def test_zero_idle_scan_interval_checks_every_tick(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "nanoclaw.agent.loop.time", SimpleNamespace(time=time.time, time_ns=time.time_ns, monotonic=lambda: 1_000.0)
        )
        loop = make_loop(tmp_path)
        loop.auto_compact.check_expired = AsyncMock()
        await loop._check_expired_sessions_if_due()
        await loop._check_expired_sessions_if_due()
        assert loop.auto_compact.check_expired.call_count == 2

    async def test_busy_sessions_are_passed_as_active(self, tmp_path):
        loop = make_loop(tmp_path)
        loop.auto_compact.check_expired = AsyncMock()
        loop._pending_queues["cli:busy"] = asyncio.Queue()
        await loop._check_expired_sessions_if_due()
        assert set(loop.auto_compact.check_expired.await_args.kwargs["active_session_keys"]) == {"cli:busy"}

    async def test_run_scans_on_the_idle_tick(self, tmp_path, monkeypatch):
        """``run``'s 1 s inbound timeout is the hook (nanobot loop.py:1440)."""
        loop = make_loop(tmp_path)
        scanned = asyncio.Event()

        async def check_expired(*_args, **_kwargs):
            scanned.set()

        loop.auto_compact.check_expired = check_expired

        async def idle_once():
            loop._running = False  # one tick, then run ends
            raise TimeoutError

        monkeypatch.setattr(loop.bus, "consume_inbound", idle_once)
        await asyncio.wait_for(loop.run(), 5)
        assert scanned.is_set()

    async def test_idle_compaction_in_a_running_loop(self, tmp_path):
        """End to end: an idle session is summarized by the running bus loop."""
        loop = make_loop(tmp_path, "Idle summary from the model.")
        idle_session(loop)
        task = asyncio.create_task(loop.run())
        try:
            async with asyncio.timeout(5):
                while "_last_summary" not in loop.sessions.get_or_create("cli:test").metadata:
                    await asyncio.sleep(0.05)
        finally:
            loop.stop()
            await asyncio.wait_for(task, 5)
        session = loop.sessions.get_or_create("cli:test")
        assert session.metadata["_last_summary"]["text"] == "Idle summary from the model."
        assert session.get_history() == []


# -- which sessions ----------------------------------------------------------------------


class TestProactiveAutoCompact:
    async def test_no_check_when_ttl_disabled(self, tmp_path):
        loop = make_loop(tmp_path, ttl=0)
        idle_session(loop, minutes=30)
        loop.consolidator.archive_session = AsyncMock(return_value="S.")
        await scan(loop)
        loop.consolidator.archive_session.assert_not_awaited()

    async def test_proactive_archive_on_idle_tick(self, tmp_path):
        loop = make_loop(tmp_path)
        idle_session(loop, turns=5)
        loop.consolidator.archive_session = AsyncMock(return_value="User chatted about old things.")
        await scan(loop)
        after = loop.sessions.get_or_create("cli:test")
        assert len(after.messages) == 11  # ten messages plus the summary marker
        assert after.get_history() == []
        assert loop.auto_compact._summaries["cli:test"]["text"] == "User chatted about old things."

    async def test_only_expired_sessions_are_archived(self, tmp_path):
        loop = make_loop(tmp_path)
        idle_session(loop, "cli:old")
        recent = loop.sessions.get_or_create("cli:recent")
        add_turns(recent, 1)
        loop.sessions.save(recent)
        loop.consolidator.archive_session = AsyncMock(return_value="S.")
        await scan(loop)
        assert [c.args[0].key for c in loop.consolidator.archive_session.await_args_list] == ["cli:old"]

    async def test_proactive_archive_skips_dream_sessions(self, tmp_path):
        loop = make_loop(tmp_path)
        idle_session(loop, "dream:20260602-155256")
        loop.consolidator.archive_session = AsyncMock(return_value="S.")
        await scan(loop)
        loop.consolidator.archive_session.assert_not_awaited()
        assert "dream:20260602-155256" not in loop.auto_compact._archiving

    async def test_idle_compaction_skips_subagent_sessions(self, tmp_path):
        """nanobot 688976b (D13): a child's private session is never compacted."""
        loop = make_loop(tmp_path)
        idle_session(loop, "cli:parent")
        idle_session(loop, "subagent:abc123")
        loop.consolidator.archive_session = AsyncMock(return_value="Conversation summary.")
        await loop._check_expired_sessions_if_due()
        await drain(loop)
        assert [c.args[0].key for c in loop.consolidator.archive_session.await_args_list] == ["cli:parent"]
        child = loop.sessions.get_or_create("subagent:abc123")
        assert child.last_archived == 0 and "_last_summary" not in child.metadata

    async def test_skip_expired_session_with_active_task(self, tmp_path):
        loop = make_loop(tmp_path)
        idle_session(loop)
        loop.consolidator.archive_session = AsyncMock(return_value="S.")
        await scan(loop, active={"cli:test"})
        loop.consolidator.archive_session.assert_not_awaited()
        # The next tick, with the task done, archives it.
        await scan(loop)
        loop.consolidator.archive_session.assert_awaited_once()

    async def test_no_duplicate_archive(self, tmp_path):
        loop = make_loop(tmp_path)
        idle_session(loop)
        started, release = asyncio.Event(), asyncio.Event()
        count = 0

        async def slow_compact(key, **_kwargs):
            nonlocal count
            count += 1
            started.set()
            await release.wait()
            return "Summary."

        loop.consolidator.compact_idle_session = slow_compact
        await loop.auto_compact.check_expired(loop.schedule_background, loop.runtime_for_session)
        await started.wait()
        await loop.auto_compact.check_expired(loop.schedule_background, loop.runtime_for_session)
        assert count == 1
        release.set()
        await drain(loop)

    async def test_archive_error_does_not_block(self, tmp_path):
        loop = make_loop(tmp_path)
        idle_session(loop)
        loop.consolidator.compact_idle_session = AsyncMock(side_effect=RuntimeError("LLM down"))
        await scan(loop)
        assert "cli:test" not in loop.auto_compact._archiving
        loop.consolidator.compact_idle_session = AsyncMock(return_value="S.")
        await scan(loop)
        loop.consolidator.compact_idle_session.assert_awaited_once()

    async def test_empty_session_does_not_schedule(self, tmp_path):
        loop = make_loop(tmp_path)
        idle_session(loop, turns=0)
        loop.consolidator.compact_idle_session = AsyncMock(return_value="S.")
        await scan(loop)
        await scan(loop)
        loop.consolidator.compact_idle_session.assert_not_awaited()

    async def test_command_only_tail_does_not_schedule(self, tmp_path):
        loop = make_loop(tmp_path)
        session = loop.sessions.get_or_create("cli:test")
        session.add_message("user", "/status", _command=True)
        session.add_message("assistant", "Ready.", _command=True)
        session.updated_at = datetime.now() - timedelta(minutes=20)
        loop.sessions.save(session)
        loop.consolidator.compact_idle_session = AsyncMock(return_value="S.")
        await scan(loop)
        loop.consolidator.compact_idle_session.assert_not_awaited()

    async def test_no_reschedule_after_successful_archive(self, tmp_path):
        loop = make_loop(tmp_path)
        idle_session(loop, turns=5)
        loop.consolidator.archive_session = AsyncMock(return_value="S.")
        await scan(loop)
        await scan(loop)
        assert loop.consolidator.archive_session.await_count == 1
        # A new ordinary message, idle again: compacted again.
        session = loop.sessions.get_or_create("cli:test")
        session.add_message("user", "Another topic.")
        session.updated_at = datetime.now() - timedelta(minutes=20)
        loop.sessions.save(session)
        await scan(loop)
        assert loop.consolidator.archive_session.await_count == 2

    async def test_unparseable_updated_at_does_not_stop_the_scan(self, tmp_path):
        loop = make_loop(tmp_path)
        idle_session(loop, "cli:a")
        idle_session(loop, "cli:b")
        path = loop.sessions.get_session_path("cli:a")
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] = lines[0].replace('"updated_at": "', '"updated_at": "garbage-', 1)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        loop.sessions.invalidate("cli:a")
        loop.consolidator.archive_session = AsyncMock(return_value="S.")
        await scan(loop)
        assert [c.args[0].key for c in loop.consolidator.archive_session.await_args_list] == ["cli:b"]


# -- events and lock -------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["succeeded", "failed", "cancelled"])
async def test_idle_compaction_emits_its_real_trigger_for_each_phase(tmp_path, outcome):
    loop = make_loop(tmp_path)
    idle_session(loop, "cli:idle-reason", turns=1)
    if outcome == "succeeded":
        loop.consolidator.archive_session = AsyncMock(return_value="Idle summary.")
    else:
        error = asyncio.CancelledError() if outcome == "cancelled" else RuntimeError("summary failed")
        loop.consolidator.archive_session = AsyncMock(side_effect=error)
    await scan(loop)
    events = outbound_events(loop)
    assert all(isinstance(e, ContextCompactionEvent) for e in events)
    assert [e.phase for e in events] == ["started", outcome]
    assert [e.reason for e in events] == ["idle", "idle"]
    assert events[0].compaction_id == events[1].compaction_id
    assert all(e.notify is False for e in events)
    await loop.aclose()


@pytest.mark.parametrize("key", ["heartbeat", "telegram:123"])
async def test_sessions_without_a_direct_route_get_no_notices(tmp_path, key):
    """nanobot routes idle notices only to direct clients (cli, websocket) or a
    saved channel route; heartbeat and Dream are maintenance sessions."""
    loop = make_loop(tmp_path)
    idle_session(loop, key)
    loop.consolidator.archive_session = AsyncMock(return_value="S.")
    await scan(loop)
    assert loop.sessions.get_or_create(key).metadata["_last_summary"]["text"] == "S."
    assert loop.bus.outbound.empty()


async def test_idle_compaction_rechecks_activity_under_the_lock(tmp_path):
    """A turn that ran while the archive waited for the lock wins (nanobot 9650f4f)."""
    loop = make_loop(tmp_path, "fresh reply")
    idle_session(loop)
    lock = loop._get_session_lock("cli:test")
    await lock.acquire()
    loop.consolidator.archive_session = AsyncMock(return_value="S.")
    await loop.auto_compact.check_expired(loop.schedule_background, loop.runtime_for_session)
    assert "cli:test" in loop.auto_compact._archiving
    session = loop.sessions.get_or_create("cli:test")
    session.add_message("user", "back again")
    loop.sessions.save(session)
    lock.release()
    await drain(loop)
    loop.consolidator.archive_session.assert_not_awaited()
    assert "cli:test" not in loop.auto_compact._archiving


async def test_turn_waits_for_a_running_archive_and_starts_from_its_summary(tmp_path):
    loop = make_loop(tmp_path, "new reply")
    idle_session(loop, turns=1)
    entered, release = asyncio.Event(), asyncio.Event()

    async def archive(_session, **_kwargs):
        entered.set()
        await release.wait()
        return "Old session summary."

    loop.consolidator.archive_session = archive
    await loop.auto_compact.check_expired(loop.schedule_background, loop.runtime_for_session)
    await asyncio.wait_for(entered.wait(), 5)
    turn = asyncio.create_task(loop.process_direct("new input", session_key="cli:test"))
    await asyncio.sleep(0.05)
    assert not turn.done()  # the archive holds the session lock
    release.set()
    result = await asyncio.wait_for(turn, 5)
    assert result.final_content == "new reply"
    system = loop.provider.calls[-1].messages[0]["content"]
    assert "Old session summary." in system
    assert [m["role"] for m in loop.provider.calls[-1].messages[1:]] == ["user"]
    saved = loop.sessions.get_or_create("cli:test")
    assert saved.metadata["_last_summary"]["text"] == "Old session summary."
    assert saved.messages[-1]["content"] == "new reply"
    await loop.aclose()


# -- summary hand-off and persistence ----------------------------------------------------------


class TestSummaryPersistence:
    async def test_summary_recovered_after_restart(self, tmp_path):
        loop = make_loop(tmp_path)
        idle_session(loop)
        loop.consolidator.archive_session = AsyncMock(return_value="Persisted summary.")
        await scan(loop)
        await loop.aclose()

        restarted = make_loop(tmp_path, "hi again")
        assert restarted.auto_compact._summaries == {}
        await restarted.process_direct("hello", session_key="cli:test")
        assert "Persisted summary." in restarted.provider.calls[0].messages[0]["content"]
        await restarted.aclose()

    async def test_hot_summary_is_used_once_then_the_saved_one(self, tmp_path):
        loop = make_loop(tmp_path, "one", "two")
        idle_session(loop)
        loop.consolidator.archive_session = AsyncMock(return_value="Hot summary.")
        await scan(loop)
        assert "cli:test" in loop.auto_compact._summaries
        await loop.process_direct("first", session_key="cli:test")
        assert "cli:test" not in loop.auto_compact._summaries
        await loop.process_direct("second", session_key="cli:test")
        for call in loop.provider.calls:
            assert "Hot summary." in call.messages[0]["content"]
        await loop.aclose()

    async def test_new_command_drops_the_summary(self, tmp_path):
        """``/new`` clears ``_last_summary`` and, as nanobot's turn stages do,
        the summary kept in memory: the next turn starts with none."""
        loop = make_loop(tmp_path, "fresh")
        idle_session(loop)
        loop.consolidator.archive_session = AsyncMock(return_value="OLD-SUMMARY")
        await scan(loop)
        assert "cli:test" in loop.auto_compact._summaries
        await loop.process_direct("/new", session_key="cli:test")
        await loop.wait_for_background_tasks()
        assert "cli:test" not in loop.auto_compact._summaries
        assert "_last_summary" not in loop.sessions.get_or_create("cli:test").metadata
        await loop.process_direct("hello", session_key="cli:test")
        assert "OLD-SUMMARY" not in loop.provider.calls[-1].messages[0]["content"]
        await loop.aclose()

    async def test_session_can_be_compacted_again_after_new_messages(self, tmp_path, monkeypatch):
        loop = make_loop(tmp_path, "second reply")
        idle_session(loop, turns=5, prefix="first")
        loop.consolidator.archive_session = AsyncMock(return_value="S1.")
        await loop.auto_compact._archive("cli:test", runtime=loop.runtime)
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="test", content="second topic")
        async with loop._get_session_lock("cli:test"):
            await loop._process_message(msg)
        clock = MagicMock(wraps=datetime)
        clock.now.return_value = datetime.now() + timedelta(minutes=20)
        clock.fromisoformat = datetime.fromisoformat
        monkeypatch.setattr("nanoclaw.agent.autocompact.datetime", clock)
        loop.consolidator.archive_session = AsyncMock(return_value="S2.")
        await loop.auto_compact._archive("cli:test", runtime=loop.runtime)
        loop.consolidator.archive_session.assert_awaited_once()
        assert loop.sessions.get_or_create("cli:test").metadata["_last_summary"]["text"] == "S2."
        await loop.aclose()


# -- the session store helpers ---------------------------------------------------------------------


def test_list_sessions_and_has_unarchived_messages(tmp_path):
    sessions = SessionManager(tmp_path / "ws", sessions_root=tmp_path / "sessions")
    a = sessions.get_or_create("cli:a")
    add_turns(a, 1)
    sessions.save(a)
    b = sessions.get_or_create("cli:b")
    sessions.save(b)
    rows = {r["key"]: r for r in sessions.list_sessions()}
    assert set(rows) == {"cli:a", "cli:b"}
    assert datetime.fromisoformat(rows["cli:a"]["updated_at"]) == a.updated_at
    assert sessions.has_unarchived_messages("cli:a") is True
    assert sessions.has_unarchived_messages("cli:b") is False
    assert sessions.has_unarchived_messages("cli:missing") is False
    a.commit_summary_checkpoint("S.")
    sessions.save(a)
    assert sessions.has_unarchived_messages("cli:a") is False
