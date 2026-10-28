"""Layer 21 gaps found by mutation checks (cron, gateway, idle compaction,
git memory history and prompt overrides, heartbeat and evaluator).

Each test pins a behavior that a deliberate code change left undetected by
the rest of the suite.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from rich.console import Console

from bus_rig import Rig, call
from nanoclaw.agent.automation_turns import AutomationTurnError
from nanoclaw.agent.cron_turns import CronTurnCoordinator
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.tools.context import RequestContext, request_context
from nanoclaw.agent.tools.cron import CronTool
from nanoclaw.agent.tools.mcp import MCPProvider
from nanoclaw.bus import InboundMessage, OutboundMessage
from nanoclaw.cli.gateway import Gateway, _pick_heartbeat_target_from_sessions
from nanoclaw.config import Config
from nanoclaw.cron.binding import CronBinding, CronBindingError, binding_revision
from nanoclaw.cron.bound_runner import DELIVERY_RUN_META, run_bound_cron_job
from nanoclaw.cron.service import CronService, _compute_next_run, _resume_next_run
from nanoclaw.cron.session_turns import (
    CRON_AUTOMATION_SPEC,
    CRON_DEFER_UNTIL_IDLE_META,
    CRON_TRIGGER_META,
    cron_run_id,
    defer_cron_until_session_idle,
    is_bound_cron_job,
)
from nanoclaw.cron.types import (
    CronJob,
    CronJobState,
    CronPayload,
    CronRunRecord,
    CronRunResult,
    CronSchedule,
    CronStore,
)
from nanoclaw.providers import FakeProvider
from nanoclaw.session import SessionManager
from nanoclaw.session.automation_spec import automation_history_overrides_for_spec
from nanoclaw.utils.templates import sync_workspace_templates

_BOUND = {"session_key": "websocket:chat-1", "origin_channel": "websocket", "origin_chat_id": "chat-1"}

# -- cron: next-run computation ------------------------------------------------------------


def test_one_shot_in_the_past_has_no_next_run() -> None:
    now = 2_000_000
    assert _compute_next_run(CronSchedule(kind="at", at_ms=now - 1), now) is None
    assert _compute_next_run(CronSchedule(kind="at", at_ms=now), now) is None
    assert _compute_next_run(CronSchedule(kind="at", at_ms=now + 1), now) == now + 1


def test_cron_next_run_uses_the_schedule_timezone() -> None:
    # 2026-01-15 12:00 UTC = 07:00 in New York (EST, UTC-5).
    now = int(datetime(2026, 1, 15, 12, 0, tzinfo=ZoneInfo("UTC")).timestamp() * 1000)
    nxt = _compute_next_run(CronSchedule(kind="cron", expr="0 9 * * *", tz="America/New_York"), now)
    assert datetime.fromtimestamp(nxt / 1000, tz=ZoneInfo("UTC")) == datetime(2026, 1, 15, 14, 0, tzinfo=ZoneInfo("UTC"))
    nxt = _compute_next_run(CronSchedule(kind="cron", expr="0 9 * * *", tz="Asia/Tokyo"), now)
    # 12:00 UTC = 21:00 Tokyo; next 09:00 Tokyo is 2026-01-16 00:00 UTC.
    assert datetime.fromtimestamp(nxt / 1000, tz=ZoneInfo("UTC")) == datetime(2026, 1, 16, 0, 0, tzinfo=ZoneInfo("UTC"))


def test_cron_next_run_across_a_dst_change() -> None:
    # New York springs forward on 2026-03-08: 09:00 local is 14:00 UTC before, 13:00 UTC after.
    now = int(datetime(2026, 3, 7, 15, 0, tzinfo=ZoneInfo("UTC")).timestamp() * 1000)
    nxt = _compute_next_run(CronSchedule(kind="cron", expr="0 9 * * *", tz="America/New_York"), now)
    assert datetime.fromtimestamp(nxt / 1000, tz=ZoneInfo("UTC")) == datetime(2026, 3, 8, 13, 0, tzinfo=ZoneInfo("UTC"))


def test_non_positive_interval_has_no_next_run() -> None:
    assert _compute_next_run(CronSchedule(kind="every", every_ms=-1000), 5_000) is None
    assert _compute_next_run(CronSchedule(kind="every", every_ms=0), 5_000) is None
    assert _compute_next_run(CronSchedule(kind="every", every_ms=1000), 5_000) == 6_000


def test_resume_keeps_a_pending_interval_run_up_to_one_interval_away() -> None:
    every = CronSchedule(kind="every", every_ms=60_000)
    now = 1_000_000
    # Exactly one interval away is still a pending run.
    assert _resume_next_run(every, now + 60_000, now) == now + 60_000
    assert _resume_next_run(every, now + 1, now) == now + 1
    # Further than one interval (e.g. the interval was shortened): recomputed from now.
    assert _resume_next_run(every, now + 60_001, now) == now + 60_000
    assert _resume_next_run(every, now + 10 * 60_000, now) == now + 60_000
    # Due or overdue: recomputed too.
    assert _resume_next_run(every, now, now) == now + 60_000


def test_tz_is_only_valid_for_cron_schedules(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    for schedule in (
        CronSchedule(kind="every", every_ms=60_000, tz="UTC"),
        CronSchedule(kind="at", at_ms=int(time.time() * 1000) + 60_000, tz="UTC"),
    ):
        with pytest.raises(ValueError, match="tz can only be used with cron schedules"):
            service.add_job("x", schedule, "hi", **_BOUND)


# -- cron: store persistence, action-log merge, start-up recompute, ticks --------------------

_ORIGIN = {"session_key": "websocket:chat-1", "origin_channel": "websocket", "origin_chat_id": "chat-1"}


def _bound_job(job_id: str = "j1", **state) -> CronJob:
    return CronJob(
        id=job_id,
        name=job_id,
        schedule=CronSchedule(kind="every", every_ms=60_000),
        payload=CronPayload(kind="agent_turn", message="hi", **_ORIGIN),
        state=CronJobState(**state),
    )


async def test_stale_action_for_a_moved_job_is_ignored(tmp_path) -> None:
    path = tmp_path / "cron" / "jobs.json"
    owner = CronService(path)
    await owner.start()
    try:
        job = owner.add_job("j", CronSchedule(kind="every", every_ms=60_000), "original", **_BOUND)
        before = deepcopy(job)
        moved = owner.change_binding(
            job.id,
            revision=binding_revision(job),
            binding=CronBinding("websocket:chat-2", "websocket", "chat-2", {}),
            message="moved",
        )
        assert moved.payload.binding_version == 1
        # Another process saw the job before the move and edits it afterwards.
        before.payload.message = "stale edit"
        assert before.payload.binding_version == 0
        owner._append_action("update", asdict(before))
        current = owner.get_job(job.id)
        assert current.payload.message == "moved"
        assert current.payload.session_key == "websocket:chat-2"
    finally:
        owner.stop()


async def test_store_round_trip_keeps_run_id_binding_version_and_delete_after_run(tmp_path) -> None:
    path = tmp_path / "cron" / "jobs.json"

    async def on_job(job: CronJob) -> CronRunResult:
        return CronRunResult(run_id="run-42", response="done")

    owner = CronService(path, on_job=on_job)
    await owner.start()
    try:
        job = owner.add_job("j", CronSchedule(kind="every", every_ms=60_000), "hi", **_BOUND)
        owner.change_binding(
            job.id,
            revision=binding_revision(job),
            binding=CronBinding("websocket:chat-2", "websocket", "chat-2", {}),
            message="hi",
        )
        assert await owner.run_job(job.id)
        once = owner.add_job(
            "once", CronSchedule(kind="at", at_ms=int(time.time() * 1000) + 3_600_000), "later",
            delete_after_run=True, **_BOUND,
        )
    finally:
        owner.stop()

    reloaded = CronService(path)
    again = reloaded.get_job(job.id)
    assert again.payload.binding_version == 1
    assert again.state.run_history[-1].run_id == "run-42"
    assert reloaded.get_job(once.id).delete_after_run is True
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {j["id"]: j for j in data["jobs"]}
    assert by_id[job.id]["payload"]["bindingVersion"] == 1
    assert by_id[job.id]["state"]["runHistory"][-1]["runId"] == "run-42"
    assert by_id[once.id]["deleteAfterRun"] is True


async def test_start_does_not_schedule_disabled_jobs_and_wake_ignores_them(tmp_path) -> None:
    path = tmp_path / "cron" / "jobs.json"
    seed = CronService(path)
    disabled = _bound_job("off")
    disabled.enabled = False
    hand_edited = _bound_job("off2", next_run_at_ms=int(time.time() * 1000) + 30_000)
    hand_edited.enabled = False
    seed._store = CronStore(jobs=[disabled, hand_edited])
    seed._save_store()

    service = CronService(path)
    await service.start()
    try:
        assert service.get_job("off").state.next_run_at_ms is None
        # A stale next run on a disabled job never becomes the next wake-up.
        assert service.status()["next_wake_at_ms"] is None
    finally:
        service.stop()


async def test_tick_with_an_unsaved_store_only_persists_it(tmp_path) -> None:
    calls: list[str] = []

    async def on_job(job: CronJob) -> None:
        calls.append(job.id)

    path = tmp_path / "cron" / "jobs.json"
    service = CronService(path, on_job=on_job)
    service._store = CronStore(jobs=[_bound_job("due", next_run_at_ms=1)])
    service._store_dirty = True  # a previous tick ran a job but failed to save
    await service._on_timer()
    assert calls == []  # nothing runs before the earlier result is durable
    assert service._store_dirty is False
    assert json.loads(path.read_text(encoding="utf-8"))["jobs"][0]["id"] == "due"
    await service._on_timer()
    assert calls == ["due"]


async def test_job_disabled_by_an_earlier_callback_in_the_same_tick_does_not_run(tmp_path) -> None:
    calls: list[str] = []
    path = tmp_path / "cron" / "jobs.json"

    async def on_job(job: CronJob) -> None:
        calls.append(job.id)
        if job.id == "a":
            service.enable_job("b", enabled=False)

    service = CronService(path, on_job=on_job)
    service._store = CronStore(jobs=[_bound_job("a", next_run_at_ms=1), _bound_job("b", next_run_at_ms=2)])
    service._save_store()
    await service._on_timer()
    assert calls == ["a"]


async def test_cancelling_the_cron_task_propagates_instead_of_recording_an_error(tmp_path) -> None:
    started = asyncio.Event()

    async def on_job(job: CronJob) -> None:
        started.set()
        await asyncio.Event().wait()

    service = CronService(tmp_path / "cron" / "jobs.json", on_job=on_job)
    service._store = CronStore(jobs=[_bound_job("j")])
    service._save_store()
    task = asyncio.create_task(service.run_job("j", force=True))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert service.get_job("j").state.last_status is None


async def test_success_clears_the_previous_error_and_last_run_is_the_start(tmp_path, monkeypatch) -> None:
    clock = {"now": 1_000_000}
    monkeypatch.setattr("nanoclaw.cron.service._now_ms", lambda: clock["now"])
    outcomes = [RuntimeError("boom"), None]

    async def on_job(job: CronJob) -> CronRunResult | None:
        clock["now"] += 5_000  # the run takes five seconds
        outcome = outcomes.pop(0)
        if outcome is not None:
            raise outcome
        return CronRunResult(run_id="r-ok", response="fine")

    service = CronService(tmp_path / "cron" / "jobs.json", on_job=on_job)
    service._store = CronStore(jobs=[_bound_job("j")])
    service._save_store()
    assert await service.run_job("j", force=True)
    job = service.get_job("j")
    assert (job.state.last_status, job.state.last_error) == ("error", "boom")
    start = clock["now"]
    assert await service.run_job("j", force=True)
    job = service.get_job("j")
    assert (job.state.last_status, job.state.last_error) == ("ok", None)
    assert job.state.last_run_at_ms == start
    last = job.state.run_history[-1]
    assert (last.run_at_ms, last.duration_ms, last.error, last.status) == (start, 5_000, None, "ok")
    assert last.run_id == "r-ok"
    assert last.session_key == "websocket:chat-1"


async def test_run_history_keeps_the_newest_twenty(tmp_path) -> None:
    runs = []

    async def on_job(job: CronJob) -> CronRunResult:
        runs.append(job.id)
        return CronRunResult(run_id=f"r{len(runs)}", response="")

    service = CronService(tmp_path / "cron" / "jobs.json", on_job=on_job)
    service._store = CronStore(jobs=[_bound_job("j")])
    service._save_store()
    for _ in range(25):
        assert await service.run_job("j", force=True)
    history = service.get_job("j").state.run_history
    assert [r.run_id for r in history] == [f"r{i}" for i in range(6, 26)]


async def test_change_binding_refusals(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    target = CronBinding("websocket:chat-2", "websocket", "chat-2", {"k": "v"})

    def reason(job_id: str, revision: str, message: str = "msg") -> str:
        with pytest.raises(CronBindingError) as info:
            service.change_binding(job_id, revision=revision, binding=target, message=message)
        return info.value.reason

    service._store = CronStore(jobs=[_bound_job("j")])
    service._save_store()
    job = service.get_job("j")
    assert reason("j", binding_revision(job)) == "busy"  # not the live owner
    await service.start()
    try:
        job = service.get_job("j")
        rev = binding_revision(job)
        service._active_executions = 1  # a run (or a turn waiting in an inbox) owns a snapshot
        assert reason("j", rev) == "busy"
        service._active_executions = 0
        service._store_dirty = True
        assert reason("j", rev) == "busy"
        service._store_dirty = False
        assert reason("missing", rev) == "unavailable"
        assert reason("j", "0" * 64) == "conflict"
        assert reason("j", rev, "   ") == "empty"
        assert service.get_job("j").payload.session_key == "websocket:chat-1"
        moved = service.change_binding("j", revision=rev, binding=target, message="msg")
        assert (moved.payload.session_key, moved.payload.origin_chat_id) == ("websocket:chat-2", "chat-2")
        assert moved.payload.origin_metadata == {"k": "v"}
        # The revision changes with the move, so a second move from the old one conflicts.
        assert reason("j", rev) == "conflict"
    finally:
        service.stop()


async def test_change_binding_keeps_earlier_runs_attributed_to_the_old_chat(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    old_run = CronRunRecord(run_at_ms=1, status="ok")  # recorded before runs carried a session key
    service._store = CronStore(jobs=[_bound_job("j", run_history=[old_run])])
    service._save_store()
    await service.start()
    try:
        job = service.get_job("j")
        target = CronBinding("websocket:chat-2", "websocket", "chat-2", {})
        moved = service.change_binding("j", revision=binding_revision(job), binding=target, message="hi")
        assert moved.state.run_history[0].session_key == "websocket:chat-1"
    finally:
        service.stop()


async def test_a_schedule_state_change_between_review_and_move_is_not_a_conflict(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    reviewed = _bound_job("j", next_run_at_ms=1)  # overdue in the saved store
    service._store = CronStore(jobs=[reviewed])
    service._save_store()
    rev = binding_revision(reviewed)
    await service.start()  # recomputes the next run: run state only
    try:
        assert service.get_job("j").state.next_run_at_ms != 1
        target = CronBinding("websocket:chat-2", "websocket", "chat-2", {})
        moved = service.change_binding("j", revision=rev, binding=target, message="hi")
        assert moved.payload.session_key == "websocket:chat-2"
    finally:
        service.stop()


def test_list_jobs_is_ordered_by_next_run_with_unscheduled_last(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    unscheduled = _bound_job("none")
    unscheduled.enabled = False
    service._store = CronStore(
        jobs=[unscheduled, _bound_job("late", next_run_at_ms=3_000), _bound_job("soon", next_run_at_ms=1_000)]
    )
    service._save_store()
    assert [j.id for j in service.list_jobs(include_disabled=True)] == ["soon", "late", "none"]
    assert [j.id for j in service.list_jobs()] == ["soon", "late"]


def test_bound_jobs_are_listed_per_session(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    other = _bound_job("other")
    other.payload.session_key = "websocket:chat-9"
    service._store = CronStore(jobs=[_bound_job("mine"), other])
    service._save_store()
    assert [j.id for j in service.list_bound_cron_jobs_for_session("websocket:chat-1")] == ["mine"]
    assert [j.id for j in service.list_bound_cron_jobs_for_session("websocket:chat-9")] == ["other"]


def test_enabling_an_unbound_agent_job_keeps_it_disabled(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    unbound = _bound_job("u")
    unbound.payload.session_key = None
    service._store = CronStore(jobs=[unbound])
    service._save_store()
    job = service.enable_job("u", enabled=True)
    assert job.enabled is False
    assert job.state.next_run_at_ms is None
    assert job.state.last_status == "error"
    assert "bound session" in job.state.last_error


async def test_status_reports_whether_the_service_runs(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    assert service.status()["enabled"] is False
    await service.start()
    try:
        assert service.status()["enabled"] is True
    finally:
        service.stop()
    assert service.status()["enabled"] is False


# -- cron: bound runner ------------------------------------------------------------------------


class _CapturingAgent:
    def __init__(self) -> None:
        self.messages: list[InboundMessage] = []

    async def submit_cron_turn(self, msg: InboundMessage) -> OutboundMessage | None:
        self.messages.append(msg)
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="reply")


async def test_bound_runner_inbound_message_shape(tmp_path) -> None:
    cron = CronService(tmp_path / "cron" / "jobs.json")
    job = _bound_job("j")
    job.payload.origin_metadata = {"_wants_stream": True, "thread": 7}
    # The owning session need not be named after the delivery chat.
    job.payload.session_key = "websocket:chat-1:thread-7"
    agent = _CapturingAgent()
    result = await run_bound_cron_job(job, agent=agent, cron=cron)
    (msg,) = agent.messages
    assert msg.sender_id == "cron"
    assert (msg.channel, msg.chat_id, msg.session_key) == ("websocket", "chat-1", "websocket:chat-1:thread-7")
    assert "_wants_stream" not in msg.metadata
    assert msg.metadata["thread"] == 7
    assert msg.metadata[CRON_DEFER_UNTIL_IDLE_META] is True
    assert msg.metadata[DELIVERY_RUN_META] == result.run_id
    assert msg.metadata[CRON_TRIGGER_META]["run_id"] == result.run_id
    # The job's saved routing is not modified by the turn's metadata.
    assert job.payload.origin_metadata == {"_wants_stream": True, "thread": 7}


async def test_bound_runner_needs_a_session_key(tmp_path) -> None:
    cron = CronService(tmp_path / "cron" / "jobs.json")
    job = _bound_job("j")
    job.payload.session_key = None
    agent = _CapturingAgent()
    with pytest.raises(ValueError, match="missing payload.session_key"):
        await run_bound_cron_job(job, agent=agent, cron=cron)
    assert agent.messages == []
    assert not (tmp_path / "cron" / "runs").exists()


def _run_records(tmp_path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((tmp_path / "cron" / "runs").glob("*.json"))]


async def test_bound_runner_records_queued_then_error(tmp_path) -> None:
    cron = CronService(tmp_path / "cron" / "jobs.json")
    seen: list[str] = []

    class _FailingAgent:
        async def submit_cron_turn(self, msg: InboundMessage) -> OutboundMessage | None:
            seen.extend(r["status"] for r in _run_records(tmp_path))
            raise RuntimeError("model down")

    with pytest.raises(RuntimeError, match="model down"):
        await run_bound_cron_job(_bound_job("j"), agent=_FailingAgent(), cron=cron)
    assert seen == ["queued"]  # recorded before the turn is submitted
    (record,) = _run_records(tmp_path)
    assert (record["status"], record["error"]) == ("error", "model down")
    assert "response" not in record


# -- cron: session-turn metadata and the coordinator ------------------------------------------


def test_cron_turn_metadata_helpers() -> None:
    trigger = {"job_id": "j", "job_name": "n", "run_id": "r", "prompt_ref": {"id": "x"}, "persist_content": "  "}
    meta = {CRON_TRIGGER_META: trigger, CRON_DEFER_UNTIL_IDLE_META: True}
    text, extra = automation_history_overrides_for_spec(meta, CRON_AUTOMATION_SPEC)
    assert text is None  # blank saved text falls back to the prompt
    assert extra["cron_run_id"] == "r" and extra["cron_job_id"] == "j"
    assert extra["cron_job_name"] == "n" and extra["cron_prompt_ref"] == {"id": "x"}
    assert defer_cron_until_session_idle(meta) is True
    assert defer_cron_until_session_idle({CRON_DEFER_UNTIL_IDLE_META: True}) is False  # not a cron turn
    assert defer_cron_until_session_idle({CRON_TRIGGER_META: trigger}) is False
    assert defer_cron_until_session_idle({CRON_TRIGGER_META: trigger, CRON_DEFER_UNTIL_IDLE_META: 1}) is False
    assert cron_run_id(meta) == "r"
    assert cron_run_id({CRON_TRIGGER_META: {**trigger, "run_id": ""}}) is None


def test_is_bound_cron_job_needs_complete_new_style_routing() -> None:
    assert is_bound_cron_job(_bound_job()) is True
    for change in (
        {"origin_chat_id": None},
        {"origin_channel": None},
        {"session_key": None},
        {"deliver": True},
        {"channel": "telegram"},
        {"to": "42"},
        {"channel_meta": {"x": 1}},
        {"kind": "system_event"},
    ):
        job = _bound_job()
        for key, value in change.items():
            setattr(job.payload, key, value)
        assert is_bound_cron_job(job) is False, change


async def test_coordinator_wraps_a_turn_error() -> None:
    coordinator: CronTurnCoordinator

    async def enqueue(msg: InboundMessage) -> None:
        coordinator.complete(msg, error=RuntimeError("tool crashed"))

    coordinator = CronTurnCoordinator(enqueue=enqueue)
    msg = InboundMessage(
        channel="cli", sender_id="cron", chat_id="a", content="x",
        metadata={CRON_TRIGGER_META: {"job_id": "j", "run_id": "r"}},
    )
    with pytest.raises(AutomationTurnError, match="tool crashed"):
        await coordinator.submit(msg)
    with pytest.raises(ValueError, match="run_id"):
        await coordinator.submit(dataclasses.replace(msg, metadata={CRON_TRIGGER_META: {"job_id": "j", "run_id": ""}}))


def _cron_msg(job_id: str, run_id: str | None = None, chat: str = "a", *, defer: bool = True) -> InboundMessage:
    metadata: dict = {CRON_TRIGGER_META: {"job_id": job_id, "run_id": run_id or job_id, "persist_content": f"saved {job_id}"}}
    if defer:
        metadata[CRON_DEFER_UNTIL_IDLE_META] = True
    return InboundMessage(channel="cli", sender_id="cron", chat_id=chat, content=job_id, metadata=metadata)


async def test_coordinator_bookkeeping() -> None:
    gate = asyncio.Event()
    queued: list[InboundMessage] = []

    async def enqueue(msg: InboundMessage) -> None:
        queued.append(msg)
        await gate.wait()
        coordinator.complete(msg, response=OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="ok"))

    coordinator = CronTurnCoordinator(enqueue=enqueue)
    task = asyncio.create_task(coordinator.submit(_cron_msg("job-a", "run-a")))
    while not queued:
        await asyncio.sleep(0)
    # Pending IDs are job IDs, per session.
    assert coordinator.pending_job_ids_for_session("cli:a") == {"job-a"}
    assert coordinator.pending_job_ids_for_session("cli:b") == set()
    # A deferred turn that was never submitted counts too, under the session it waits for.
    deferred = _cron_msg("job-b", "run-b", chat="b")
    assert coordinator.defer_if_active(deferred, session_key="cli:other", active_session_keys=["cli:other"])
    assert coordinator.pending_job_ids_for_session("cli:other") == {"job-b"}
    assert coordinator.deferred_queues["cli:other"][0].session_key == "cli:other"
    assert not coordinator.defer_if_active(_cron_msg("job-c"), session_key="cli:a", active_session_keys=[])
    gate.set()
    assert (await task).content == "ok"
    # The run ID is free again once its turn finished.
    queued.clear()
    assert (await coordinator.submit(_cron_msg("job-a", "run-a"))).content == "ok"
    assert coordinator.pending_job_ids_for_session("cli:a") == set()


async def test_deferred_cron_turns_run_in_submission_order(tmp_path) -> None:
    script = [call("slow", {"text": "one"}), "first done", "cron one", "cron two"]
    async with Rig(tmp_path / "ws", tmp_path, script) as rig:
        await rig.send("first")
        await asyncio.wait_for(rig.slow.running("one").wait(), 5)
        c1 = asyncio.create_task(rig.loop.submit_cron_turn(_cron_msg("j1")))
        await rig.until(lambda: len(rig.loop._deferred_automation_turns.get("cli:a", [])) == 1)
        c2 = asyncio.create_task(rig.loop.submit_cron_turn(_cron_msg("j2")))
        await rig.until(lambda: len(rig.loop._deferred_automation_turns.get("cli:a", [])) == 2)
        rig.slow.gate("one").set()
        assert (await asyncio.wait_for(c1, 5)).content == "cron one"
        assert (await asyncio.wait_for(c2, 5)).content == "cron two"
        users = [m["content"].split("\n")[0] for m in rig.saved() if m["role"] == "user"]
        assert users == ["first", "saved j1", "saved j2"]


async def test_a_queued_cron_turn_is_never_injected_into_a_running_turn(tmp_path) -> None:
    # Without the defer flag the cron turn queues behind the running turn like
    # user input, but it still runs as its own turn with its own reply.
    script = [call("slow", {"text": "one"}), "first done", "cron done"]
    async with Rig(tmp_path / "ws", tmp_path, script) as rig:
        await rig.send("first")
        await asyncio.wait_for(rig.slow.running("one").wait(), 5)
        cron = asyncio.create_task(rig.loop.submit_cron_turn(_cron_msg("job", defer=False)))
        await rig.until(lambda: rig.loop._pending_queues["cli:a"].qsize() == 1)
        rig.slow.gate("one").set()
        assert (await asyncio.wait_for(cron, 5)).content == "cron done"
        users = [m["content"].split("\n")[0] for m in rig.saved() if m["role"] == "user"]
        assert users == ["first", "saved job"]


async def test_stopping_a_running_cron_turn_fails_its_waiter(tmp_path) -> None:
    async with Rig(tmp_path / "ws", tmp_path, [call("slow", {"text": "x"})]) as rig:
        cron = asyncio.create_task(rig.loop.submit_cron_turn(_cron_msg("job")))
        await asyncio.wait_for(rig.slow.running("x").wait(), 5)
        await rig.send("/stop")
        with pytest.raises(AutomationTurnError):
            await asyncio.wait_for(cron, 5)
        assert rig.loop.pending_cron_job_ids_for_session("cli:a") == set()


def test_make_loop_gives_the_cron_tool_the_configured_timezone(tmp_path) -> None:
    from nanoclaw.__main__ import _make_loop
    from nanoclaw.config import Config

    config = Config.model_validate({"agents": {"defaults": {"timezone": "Asia/Tokyo"}}})
    loop = _make_loop(FakeProvider([]), tmp_path, config=config)
    assert loop.timezone == "Asia/Tokyo"
    assert loop.tools.get("cron")._default_timezone == "Asia/Tokyo"


# -- cron tool ---------------------------------------------------------------------------------


def _tool(tmp_path, tz: str = "Asia/Tokyo") -> CronTool:
    return CronTool(CronService(tmp_path / "cron" / "jobs.json"), default_timezone=tz)


_CTX = RequestContext(channel="telegram", chat_id="chat-1", session_key="telegram:chat-1")


def test_cron_tool_add_needs_a_chat_id(tmp_path) -> None:
    tool = _tool(tmp_path)
    with request_context(RequestContext(channel="telegram", chat_id="", session_key="telegram:chat-1")):
        result = tool._add_job(None, "Water", 60, None, None, None)
    assert result == "Error: scheduled cron jobs must be created from a chat session"
    assert tool._cron.list_jobs(include_disabled=True) == []


def test_cron_tool_add_names_and_one_shot_cleanup(tmp_path) -> None:
    tool = _tool(tmp_path)
    message = "Remind me to call the dentist about the appointment next week"
    future = (datetime.now(ZoneInfo("UTC")) + timedelta(days=1)).isoformat()
    with request_context(_CTX):
        assert tool._add_job(None, message, 3600, None, None, None).startswith("Created job")
        assert tool._add_job("Once", "ping", None, None, None, future).startswith("Created job")
    by_name = {j.name: j for j in tool._cron.list_jobs()}
    assert message[:30] in by_name
    assert by_name[message[:30]].delete_after_run is False
    assert by_name["Once"].delete_after_run is True


def test_cron_tool_rejects_an_unknown_timezone_with_a_tool_error(tmp_path) -> None:
    tool = _tool(tmp_path)
    with request_context(_CTX):
        result = tool._add_job(None, "x", None, "0 9 * * *", "Bad/Zone", None)
    assert result == "Error: unknown timezone 'Bad/Zone'"
    assert tool._cron.list_jobs(include_disabled=True) == []


def test_cron_tool_update_interval_job_to_cron_uses_the_default_timezone(tmp_path) -> None:
    tool = _tool(tmp_path)
    job = tool._cron.add_job("j", CronSchedule(kind="every", every_ms=60_000), "hi", **_BOUND)
    assert tool._update_job(job.id, None, None, None, "0 9 * * *", None, None).startswith("Updated")
    assert tool._cron.get_job(job.id).schedule.tz == "Asia/Tokyo"


def test_cron_tool_update_with_an_invalid_cron_expression_is_a_tool_error(tmp_path) -> None:
    tool = _tool(tmp_path)
    job = tool._cron.add_job("j", CronSchedule(kind="every", every_ms=60_000), "hi", **_BOUND)
    result = tool._update_job(job.id, None, None, None, "not a cron", None, None)
    assert result.startswith("Error: invalid cron expression 'not a cron'")
    assert tool._cron.get_job(job.id).schedule.kind == "every"


def test_cron_tool_can_disable_but_not_enable_an_unbound_job(tmp_path) -> None:
    tool = _tool(tmp_path)
    unbound = _bound_job("u")
    unbound.payload.session_key = None
    tool._cron._store = CronStore(jobs=[unbound])
    tool._cron._save_store()
    assert tool._set_job_enabled("u", enabled=False) == "Disabled job u"
    assert tool._set_job_enabled("u", enabled=True) == "Error: job has no linked chat; recreate it from a chat session"


def test_atomic_write_removes_its_temporary_file_on_failure(tmp_path, monkeypatch) -> None:
    import os

    from nanoclaw.utils.atomic_file import atomic_write_text

    target = tmp_path / "jobs.json"

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(target, "{}")
    assert list(tmp_path.iterdir()) == []


# -- gateway: heartbeat target -------------------------------------------------------------------


def test_heartbeat_target_never_picks_a_cli_chat() -> None:
    target = _pick_heartbeat_target_from_sessions(
        enabled_channels=["cli", "telegram"], archived_keys=[], sessions=[{"key": "cli:other"}, {"key": "telegram:9"}]
    )
    assert target == ("telegram", "9")
    target = _pick_heartbeat_target_from_sessions(enabled_channels=["cli"], archived_keys=[], sessions=[{"key": "cli:other"}])
    assert target == ("cli", "direct")


# -- gateway: health, readiness, shutdown -------------------------------------------------------


def test_health_constants_match_nanobot() -> None:
    import inspect

    from nanoclaw.cli.gateway import (
        GATEWAY_HEALTH_MAX_CONNECTIONS,
        GATEWAY_HEALTH_READ_TIMEOUT_SECONDS,
        make_health_handler,
    )

    assert (GATEWAY_HEALTH_MAX_CONNECTIONS, GATEWAY_HEALTH_READ_TIMEOUT_SECONDS) == (64, 2.0)
    params = inspect.signature(make_health_handler).parameters
    assert params["max_connections"].default == 64
    assert params["read_timeout"].default == 2.0


class _Channels:
    def __init__(self, status: dict, enabled: tuple[str, ...] = ()) -> None:
        self._status = status
        self.enabled_channels = enabled

    def get_status(self) -> dict:
        return self._status


def test_readiness_websocket_enabled_but_not_reported_is_degraded() -> None:
    from nanoclaw.cli.gateway import _gateway_readiness_payload

    ready, payload = _gateway_readiness_payload(_Channels({}, ("websocket",)))
    assert ready is False
    assert payload["websocket"] == "unavailable" and payload["status"] == "degraded"
    # Running must be exactly True.
    for status in ({"state": "starting"}, {"running": "yes", "state": "x"}, {"running": 1}):
        ready, payload = _gateway_readiness_payload(_Channels({"websocket": status}))
        assert ready is False, status
    ready, _ = _gateway_readiness_payload(_Channels({"telegram": {"running": True}}, ("telegram",)))
    assert ready is True


def test_health_url_and_loopback_detection() -> None:
    from nanoclaw.cli.gateway import gateway_health_url
    from nanoclaw.security.network import is_loopback_host

    assert gateway_health_url("::1", 18790) == "http://[::1]:18790/health"
    assert gateway_health_url("fe80::1", 1) == "http://[fe80::1]:1/health"
    assert gateway_health_url("[::1]", 1) == "http://[::1]:1/health"
    assert gateway_health_url("0.0.0.0", 1) == "http://127.0.0.1:1/health"
    for host in ("localhost", "LOCALHOST.", " localhost ", "[::1]", "::1", "127.0.0.2"):
        assert is_loopback_host(host) is True, host
    for host in ("0.0.0.0", "example.com", "[fe80::1]", "10.0.0.1"):
        assert is_loopback_host(host) is False, host


async def test_close_recancels_a_task_that_swallowed_the_first_cancel() -> None:
    from nanoclaw.cli.gateway import _close_gateway_runtime

    events: list[str] = []

    async def swallow_once() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            events.append("swallowed")
        await asyncio.sleep(3600)

    class _Closable:
        async def aclose(self) -> None:
            pass

    task = asyncio.create_task(swallow_once())
    await asyncio.sleep(0)
    await _close_gateway_runtime(_Closable(), _Closable(), [task], None, task_wait_timeout=0.05)
    for _ in range(5):
        await asyncio.sleep(0)
    assert events == ["swallowed"]
    assert task.cancelled()


# -- gateway: system jobs and the Dream job -----------------------------------------------------


def _gw_agent(tmp_path, *script) -> AgentLoop:
    ws = tmp_path / "ws"
    sync_workspace_templates(ws)
    return AgentLoop(
        FakeProvider(list(script)),
        ws,
        sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"),
        cron_service=CronService(ws / "cron" / "jobs.json"),
    )


class _NoMCP(MCPProvider):
    def __init__(self) -> None:
        super().__init__({}, MagicMock())
        self.connected = 0

    async def connect(self) -> None:
        self.connected += 1

    async def aclose(self) -> None:
        pass


def _gateway(agent: AgentLoop, config: Config | None = None) -> Gateway:
    return Gateway(config or Config(), agent, _NoMCP(), console=Console(record=True, width=200), health_server_enabled=False)


def test_legacy_dream_cron_uses_the_configured_timezone(tmp_path) -> None:
    config = Config.model_validate({"agents": {"defaults": {"timezone": "Asia/Tokyo", "dream": {"cron": "0 3 * * *"}}}})
    gateway = _gateway(_gw_agent(tmp_path), config)
    gateway.register_system_jobs()
    assert gateway.cron.get_job("dream").schedule.tz == "Asia/Tokyo"


async def test_dream_job_runs_with_dream_tools_scope_and_runtime(tmp_path, monkeypatch) -> None:
    agent = _gw_agent(tmp_path)
    agent.memory.append_history("- likes tea")
    sentinel_runtime = object()
    monkeypatch.setattr(agent, "dream_runtime", lambda: sentinel_runtime)
    seen: dict = {}

    async def fake_process_direct(prompt, **kwargs):
        seen.update(kwargs, prompt=prompt)
        return SimpleNamespace(stop_reason="completed", final_content="")

    monkeypatch.setattr(agent, "process_direct", fake_process_direct)
    gateway = _gateway(agent)
    await gateway.run_dream_job()
    assert seen["runtime"] is sentinel_runtime
    assert seen["workspace_scope"] is agent.agent_workspace_scope
    assert seen["ephemeral"] is True
    assert seen["session_key"].startswith("dream:")
    tools = seen["tools"]
    assert tools is not None and tools.has("write_file") and not tools.has("cron")
    assert gateway.mcp_provider.connected == 1


# -- gateway: the run ---------------------------------------------------------------------------


async def test_gateway_run_drains_scheduled_bus_events_before_returning(tmp_path) -> None:
    from nanoclaw.events import AgentEvent

    agent = _gw_agent(tmp_path)
    gateway = _gateway(agent)
    handled: list[str] = []

    async def slow_handler(event) -> None:
        await asyncio.sleep(0.2)
        handled.append(type(event).__name__)

    agent.bus.subscribe(slow_handler)

    async def publish_then_stop() -> None:
        agent.bus.publish_nowait(AgentEvent())
        gateway.shutdown_event.set()
        await asyncio.sleep(3600)

    gateway.extra_tasks.append(("publisher", publish_then_stop))
    await asyncio.wait_for(gateway.run(), 10)
    assert handled == ["AgentEvent"]


async def test_gateway_startup_failure_is_raised(tmp_path) -> None:
    agent = _gw_agent(tmp_path)
    store = agent.workspace / "cron" / "jobs.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("{not json", encoding="utf-8")
    gateway = _gateway(agent)
    with pytest.raises(RuntimeError, match="corrupt"):
        await asyncio.wait_for(gateway.run(), 10)
    assert gateway.crashed is True


async def test_gateway_run_cancelled_from_outside_propagates_and_restores_signals(tmp_path) -> None:
    import signal

    agent = _gw_agent(tmp_path)
    gateway = _gateway(agent)
    run = asyncio.create_task(gateway.run())
    async with asyncio.timeout(5):
        while not agent._running:
            await asyncio.sleep(0.02)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run, 10)
    # The gateway's SIGINT/SIGTERM handlers are gone again.
    loop = asyncio.get_running_loop()
    assert loop.remove_signal_handler(signal.SIGINT) is False
    assert loop.remove_signal_handler(signal.SIGTERM) is False


async def test_run_agent_closes_mcp_when_the_loop_ends(tmp_path, monkeypatch) -> None:
    agent = _gw_agent(tmp_path)
    gateway = _gateway(agent)
    closed = []

    async def aclose() -> None:
        closed.append(1)

    async def failing_run() -> None:
        raise RuntimeError("loop died")

    monkeypatch.setattr(gateway.mcp_provider, "aclose", aclose)
    monkeypatch.setattr(agent, "run", failing_run)
    with pytest.raises(RuntimeError, match="loop died"):
        await gateway._run_agent()
    assert closed == [1]


def test_run_gateway_exit_status_reports_a_crash(tmp_path, monkeypatch) -> None:
    from nanoclaw.cli import gateway as gw

    def free_port() -> int:
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    async def crashing_run(self) -> None:
        self.crashed = True

    monkeypatch.setattr(gw.Gateway, "run", crashing_run)
    status = gw.run_gateway(
        Config(), lambda: (_gw_agent(tmp_path), _NoMCP()), port=free_port(), console=Console(record=True)
    )
    assert status == 1


def test_gateway_verbose_flag_logs_at_debug(tmp_path, monkeypatch) -> None:
    import nanoclaw.__main__ as cli
    from nanoclaw.cli import gateway as gw

    levels: list[str] = []
    shown: list[bool] = []
    monkeypatch.setattr(cli, "configure_console_logging", lambda stream, level="INFO": levels.append(level))
    # Keep the process-wide loguru switch untouched for the other tests.
    monkeypatch.setattr(cli, "set_nanoclaw_logs", shown.append)
    monkeypatch.setattr(gw, "run_gateway", lambda *a, **k: 0)
    assert cli.main(["gateway", "-w", str(tmp_path / "ws"), "-v"], provider=FakeProvider([])) == 0
    assert cli.main(["gateway", "-w", str(tmp_path / "ws")], provider=FakeProvider([])) == 0
    assert levels == ["DEBUG", "INFO"]
    assert shown == [True, True]


# -- idle auto-compaction ----------------------------------------------------------------------


async def test_archive_never_compacts_a_dream_session(tmp_path) -> None:
    from unittest.mock import AsyncMock

    from nanoclaw.agent.autocompact import AutoCompact

    sessions = SessionManager(tmp_path / "ws", sessions_root=tmp_path / "sessions")
    session = sessions.get_or_create("dream:1")
    session.add_message("user", "x")
    session.updated_at = datetime.now() - timedelta(minutes=60)
    sessions.save(session)
    consolidator = SimpleNamespace(compact_idle_session=AsyncMock(return_value="summary"))
    ac = AutoCompact(sessions, consolidator, session_ttl_minutes=15, session_lock=lambda key: asyncio.Lock())
    ac._archiving.add("dream:1")
    await ac._archive("dream:1", runtime=SimpleNamespace())
    consolidator.compact_idle_session.assert_not_called()
    assert "dream:1" not in ac._archiving


async def test_idle_compactions_run_as_tracked_background_tasks(tmp_path) -> None:
    agent = _gw_agent(tmp_path)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def archive() -> None:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    agent.schedule_background(archive())
    await started.wait()
    assert agent._has_pending_work()
    await agent.aclose()  # cancels what is still running
    assert cancelled.is_set()


def test_list_sessions_prefers_the_cached_copys_times(tmp_path) -> None:
    sessions = SessionManager(tmp_path / "ws", sessions_root=tmp_path / "sessions")
    session = sessions.get_or_create("cli:a")
    session.add_message("user", "hi")
    sessions.save(session)
    newer = datetime.now() + timedelta(hours=1)
    session.updated_at = newer  # changed in memory, not saved yet
    (info,) = sessions.list_sessions()
    assert info["key"] == "cli:a"
    assert info["updated_at"] == newer.isoformat()


# -- git memory history --------------------------------------------------------------------------


def test_working_tree_summary_truncates_a_huge_diff_but_keeps_the_counts(tmp_path) -> None:
    from nanoclaw.agent.memory import MemoryStore

    store = MemoryStore(tmp_path)
    store.write_memory("# Memory\n")
    store.git.init()
    store.git.auto_commit("initial")
    store.write_memory("# Memory\n" + "".join(f"- fact number {i} with some padding text\n" for i in range(400)))
    summary = store.dream_content_diff()
    assert "memory/MEMORY.md: +400 -0" in summary
    assert "400 insertions(+)" in summary
    assert summary.endswith("\n...[diff truncated]\n```")
    block = summary.split("```diff\n", 1)[1]
    assert len(block) < 6000 + 100


def test_show_commit_diff_accepts_a_short_sha_prefix_and_soul_changes_are_tracked(tmp_path) -> None:
    from nanoclaw.agent.memory import MemoryStore

    store = MemoryStore(tmp_path)
    store.soul_file.write_text("# Soul\n- calm\n", encoding="utf-8")
    store.git.init()
    store.soul_file.write_text("# Soul\n- calm\n- curious\n", encoding="utf-8")
    diff = store.dream_content_diff()
    assert "SOUL.md: +1 -0" in diff
    sha = store.git.auto_commit(MemoryStore.build_dream_commit_message("dream: manual run", diff))
    found = store.git.show_commit_diff(sha[:4])
    assert found is not None
    commit, patch = found
    assert commit.sha == sha
    assert "+- curious" in patch


async def test_run_dream_always_compacts_the_history_journal(tmp_path, monkeypatch) -> None:
    agent = _gw_agent(tmp_path, "Nothing worth keeping.")
    calls: list[int] = []
    monkeypatch.setattr(agent.memory, "compact_history", lambda: calls.append(1))
    await agent.run_dream()  # nothing to process
    agent.memory.append_history("- small talk")
    await agent.run_dream()  # a real run
    assert calls == [1, 1]


def test_changed_files_are_listed_once() -> None:
    from nanoclaw.command.builtin import _extract_changed_files

    diff = (
        "diff --git a/USER.md b/USER.md\n+x\n"
        "diff --git a/memory/MEMORY.md b/memory/MEMORY.md\n+y\n"
        "diff --git a/USER.md b/USER.md\n+z\n"
    )
    assert _extract_changed_files(diff) == ["USER.md", "memory/MEMORY.md"]


async def test_dream_prompt_init_is_case_insensitive(tmp_path) -> None:
    from nanoclaw.agent.memory import MemoryStore
    from nanoclaw.command.builtin import cmd_dream_prompt
    from nanoclaw.command.router import CommandContext

    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="direct", content="/dream-prompt INIT")
    store = MemoryStore(tmp_path)
    loop = SimpleNamespace(context=SimpleNamespace(memory=store))
    ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/dream-prompt INIT", args=" INIT ", loop=loop)
    out = await cmd_dream_prompt(ctx)
    assert out.content.startswith("Created Dream memory instructions at `prompts/dream.md`.")
    assert store.has_dream_prompt_override()


async def test_bare_evaluator_prompt_command_is_routed(tmp_path) -> None:
    from nanoclaw.command import CommandRouter, register_builtin_commands
    from nanoclaw.command.router import CommandContext

    router = CommandRouter()
    register_builtin_commands(router)
    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="direct", content="/evaluator-prompt")
    loop = SimpleNamespace(context=SimpleNamespace(memory=SimpleNamespace(workspace=tmp_path)))
    ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/evaluator-prompt", args="", loop=loop)
    out = await router.dispatch(ctx)
    assert out.content.startswith("Heartbeat evaluator prompt: nanoclaw default")


def test_template_sync_survives_a_git_failure(tmp_path, monkeypatch) -> None:
    from nanoclaw.utils.gitstore import GitStore

    def broken_init(self) -> bool:
        raise RuntimeError("no git for you")

    monkeypatch.setattr(GitStore, "init", broken_init)
    added = sync_workspace_templates(tmp_path / "ws")
    assert "AGENTS.md" in added and "HEARTBEAT.md" in added
    assert not (tmp_path / "ws" / ".git").exists()


# -- heartbeat -----------------------------------------------------------------------------------


def test_heartbeat_tasks_after_a_closed_multiline_comment_count() -> None:
    from nanoclaw.cli.gateway import _heartbeat_has_active_tasks

    assert _heartbeat_has_active_tasks("## Active Tasks\n<!--\nnote\n-->\n- water the plants\n") is True
    assert _heartbeat_has_active_tasks("## Active Tasks\n<!--\nnote -->\n- water the plants\n") is True


async def test_heartbeat_needs_an_existing_target_session(tmp_path, monkeypatch) -> None:
    agent = _gw_agent(tmp_path, "Report.")
    (agent.workspace / "HEARTBEAT.md").write_text("## Active Tasks\n\n- check\n", encoding="utf-8")
    gateway = _gateway(agent)
    monkeypatch.setattr(gateway, "pick_heartbeat_target", lambda: ("telegram", "ghost"))
    assert await gateway.run_heartbeat_job() is None
    assert agent.provider.calls == []
    assert agent.sessions.get_existing("telegram:ghost") is None


async def test_heartbeat_turn_connects_mcp_and_runs_for_the_target_chat(tmp_path, monkeypatch) -> None:
    agent = _gw_agent(tmp_path)
    (agent.workspace / "HEARTBEAT.md").write_text("## Active Tasks\n\n- check\n", encoding="utf-8")
    target = agent.sessions.get_or_create("telegram:u1")
    target.add_message("user", "hello")
    agent.sessions.save(target)
    gateway = _gateway(agent)
    gateway.enabled_channels = ["telegram"]
    seen: dict = {}

    async def fake_process_direct(prompt, **kwargs):
        seen.update(kwargs, connected=gateway.mcp_provider.connected)
        return SimpleNamespace(final_content="")

    monkeypatch.setattr(agent, "process_direct", fake_process_direct)
    assert await gateway.run_heartbeat_job() is None
    assert seen["connected"] == 1  # MCP is up before the turn starts
    assert (seen["session_key"], seen["channel"], seen["chat_id"]) == ("heartbeat", "telegram", "u1")


async def test_delivery_record_details(tmp_path) -> None:
    agent = _gw_agent(tmp_path)
    gateway = _gateway(agent)
    await gateway.deliver_to_channel(OutboundMessage(channel="telegram", chat_id="u1", content="   "), record=True)
    assert agent.sessions.get_existing("telegram:u1") is None  # blank text is published, never recorded
    await gateway.deliver_to_channel(
        OutboundMessage(channel="telegram", chat_id="u1", content="Chart attached.", media=["/tmp/chart.png"]),
        record=True,
    )
    agent.sessions.invalidate("telegram:u1")
    saved = agent.sessions.get_existing("telegram:u1")  # read back from disk
    assert saved is not None
    (message,) = saved.messages
    assert (message["role"], message["content"], message["_channel_delivery"]) == ("assistant", "Chart attached.", True)
    assert message["media"] == ["/tmp/chart.png"]
    assert agent.bus.outbound.qsize() == 2


def test_a_reply_to_a_heartbeat_delivery_replays_it() -> None:
    from nanoclaw.session import Session

    session = Session(key="telegram:u1")
    session.add_message("assistant", "Disk is 95% full.", _channel_delivery=True)
    session.add_message("user", "which disk?")
    assert [m["content"] for m in session.get_history()] == ["Disk is 95% full.", "which disk?"]
    plain = Session(key="telegram:u2")
    plain.add_message("assistant", "stray reply")
    plain.add_message("user", "hi")
    assert [m["content"] for m in plain.get_history()] == ["hi"]


def test_dream_session_keys_need_the_colon() -> None:
    from nanoclaw.session.keys import is_dream_session, is_internal_session

    assert is_dream_session("dream:20260101-000000") is True
    assert is_dream_session("dreamy:chat") is False
    assert is_internal_session("dreamteam") is False


async def test_evaluator_missing_decision_and_internal_errors_use_the_default() -> None:
    from nanoclaw.providers import LLMResponse, ToolCallRequest
    from nanoclaw.utils.evaluator import evaluate_response

    no_decision = LLMResponse(
        content="",
        tool_calls=[ToolCallRequest(id="e1", name="evaluate_notification", arguments={"reason": "unsure"})],
    )
    assert await evaluate_response("r", "t", FakeProvider([no_decision]), None, "gate", default_notify=False) is False

    class _Raising(FakeProvider):
        async def chat_stream_with_retry(self, *args, **kwargs):
            raise RuntimeError("unexpected")

    assert await evaluate_response("r", "t", _Raising([]), None, "gate", default_notify=False) is False
    assert await evaluate_response("r", "t", _Raising([]), None, "gate", default_notify=True) is True
