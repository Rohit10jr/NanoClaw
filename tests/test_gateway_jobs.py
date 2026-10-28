"""The gateway's system jobs (layer 21.4): scheduled Dream and the heartbeat.

Ported from nanobot c4dac64: tests/cli/test_commands.py (HEARTBEAT.md parsing,
heartbeat target, disabled-Dream cursor, empty heartbeat response, unbound
cron skipped), tests/config/test_dream_config.py, tests/agent/test_evaluator.py
and tests/command/test_builtin_evaluator_prompt.py. The job-body and
in-process tests are NanoClaw's: a real AgentLoop with a scripted model, the
real CronService and a fake clock.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.memory import MemoryStore
from nanoclaw.agent.tools.mcp import MCPProvider
from nanoclaw.bus import InboundMessage
from nanoclaw.cli import gateway as gw
from nanoclaw.cli.gateway import (
    _HEARTBEAT_PREAMBLE,
    Gateway,
    _advance_dream_cursor_if_behind,
    _heartbeat_has_active_tasks,
    _pick_heartbeat_target_from_sessions,
)
from nanoclaw.command.builtin import build_help_text, builtin_command_palette, cmd_evaluator_prompt
from nanoclaw.command.router import CommandContext
from nanoclaw.config import Config
from nanoclaw.config.schema import DreamConfig
from nanoclaw.cron.service import CronJobSkippedError, CronService
from nanoclaw.cron.types import CronJob, CronPayload, CronSchedule
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.session import SessionManager
from nanoclaw.session.keys import HEARTBEAT_SESSION_KEY, is_internal_session
from nanoclaw.utils.evaluator import (
    EVALUATOR_PROMPT_MAX_CHARS,
    _EVALUATE_TOOL,
    default_evaluator_prompt,
    evaluate_response,
    evaluator_prompt_file,
    resolve_evaluator_prompt,
)
from nanoclaw.utils.templates import load_bundled_template, render_template, sync_workspace_templates

# -- HEARTBEAT.md (nanobot test_commands.py:1895) ------------------------------------------


@pytest.mark.parametrize(
    "content, expected",
    [
        ("", False),
        ("# Title\n\n## Active Tasks\n", False),
        ("<!--\nmulti-line\ncomment\n-->\n", False),  # block comment, not tasks
        ("<!-- single line -->\n", False),
        ("## Active Tasks\n\n- water the plants\n", True),
        ("## Active Tasks\n\n### Garden\n\n- water the plants\n", True),
        ("## Notes\n\nsome random note\n", False),
        ("stray text before any heading\n## Active Tasks\n\n- task\n", True),
        ("stray text before any heading\n", False),
        # NanoClaw additions
        ("## Active Tasks\n<!-- a\nb -->\n", False),
        ("## Active Tasks\n<!-- a --> - task after comment\n", False),
        ("## ACTIVE TASKS (daily)\n- task\n", True),
        ("## Active Tasks\n## Done\n- old task\n", False),
        ("# Active Tasks\n- top-level heading does not open the section\n", False),
    ],
)
def test_heartbeat_has_active_tasks(content, expected):
    assert _heartbeat_has_active_tasks(content) is expected


def test_heartbeat_skips_bundled_template():
    assert _heartbeat_has_active_tasks(load_bundled_template("HEARTBEAT.md")) is False


def test_new_workspace_gets_heartbeat_md(tmp_path):
    added = sync_workspace_templates(tmp_path)
    assert "HEARTBEAT.md" in added
    text = (tmp_path / "HEARTBEAT.md").read_text(encoding="utf-8")
    assert text.startswith("# Heartbeat Tasks\n")
    assert "your nanoclaw agent" in text and "## Active Tasks" in text


# -- heartbeat target (nanobot test_commands.py:1920) ----------------------------------------


def test_heartbeat_target_skips_archived_sessions():
    target = _pick_heartbeat_target_from_sessions(
        enabled_channels=["websocket"],
        archived_keys=["websocket:archived"],
        sessions=[{"key": "websocket:archived"}, {"key": "websocket:active"}],
    )
    assert target == ("websocket", "active")


@pytest.mark.parametrize(
    "keys, enabled, expected",
    [
        ([], ["telegram"], ("cli", "direct")),
        (["cli:direct", "cli:other"], ["telegram"], ("cli", "direct")),
        (["cli:direct"], ["cli"], ("cli", "direct")),  # the CLI is never a target
        (["system:x", "heartbeat", "dream:1"], ["system"], ("cli", "direct")),
        (["telegram:1"], [], ("cli", "direct")),  # channel not enabled
        (["telegram:"], ["telegram"], ("cli", "direct")),  # empty chat id
        (["cli:direct", "telegram:42", "websocket:a"], ["telegram", "websocket"], ("telegram", "42")),
        (["telegram:a:b"], ["telegram"], ("telegram", "a:b")),
    ],
)
def test_heartbeat_target_selection(keys, enabled, expected):
    target = _pick_heartbeat_target_from_sessions(
        enabled_channels=enabled, archived_keys=[], sessions=[{"key": k} for k in keys]
    )
    assert target == expected


def test_gateway_target_is_the_most_recently_updated_session(tmp_path):
    agent = make_agent(tmp_path)
    for key, stamp in (("telegram:old", "2026-01-01T10:00:00"), ("telegram:new", "2026-01-02T10:00:00")):
        session = agent.sessions.get_or_create(key)
        session.add_message("user", "hi")
        agent.sessions.save(session)
        agent.sessions.invalidate(key)
        path = agent.sessions.get_session_path(key)
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] = lines[0].replace(lines[0].split('"updated_at": "')[1].split('"')[0], stamp)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    gateway = make_gateway(agent)
    assert gateway.pick_heartbeat_target() == ("cli", "direct")  # no channels yet (layers 23/24)
    gateway.enabled_channels = ["telegram"]
    assert gateway.pick_heartbeat_target() == ("telegram", "new")


def test_session_keys():
    assert HEARTBEAT_SESSION_KEY == "heartbeat"
    assert is_internal_session("heartbeat") and is_internal_session("dream:20260101-000000")
    assert not is_internal_session("telegram:heartbeat")


# -- DreamConfig (nanobot tests/config/test_dream_config.py) ---------------------------------


def test_dream_config_defaults_to_interval_hours() -> None:
    cfg = DreamConfig()
    assert cfg.enabled is True
    assert cfg.interval_h == 2
    assert cfg.cron is None


def test_dream_config_builds_every_schedule_from_interval() -> None:
    schedule = DreamConfig(interval_h=3).build_schedule("UTC")
    assert schedule.kind == "every"
    assert schedule.every_ms == 3 * 3_600_000
    assert schedule.expr is None
    assert DreamConfig(interval_h=3).describe_schedule() == "every 3h"


def test_dream_config_honors_legacy_cron_override() -> None:
    cfg = DreamConfig.model_validate({"cron": "0 */4 * * *"})
    schedule = cfg.build_schedule("UTC")
    assert schedule.kind == "cron"
    assert schedule.expr == "0 */4 * * *"
    assert schedule.tz == "UTC"
    assert cfg.describe_schedule() == "cron 0 */4 * * * (legacy)"


def test_dream_config_dump_preserves_legacy_cron_override() -> None:
    dumped = DreamConfig.model_validate({"intervalH": 5, "cron": "0 */4 * * *"}).model_dump(by_alias=True)
    assert dumped["intervalH"] == 5
    assert dumped["cron"] == "0 */4 * * *"


def test_dream_config_dump_omits_empty_legacy_cron() -> None:
    assert "cron" not in DreamConfig().model_dump(by_alias=True)


def test_dream_config_uses_model_override_name_and_accepts_legacy_model() -> None:
    cfg = DreamConfig.model_validate({"model": "openrouter/sonnet"})
    dumped = cfg.model_dump(by_alias=True)
    assert cfg.model_override == "openrouter/sonnet"
    assert dumped["modelOverride"] == "openrouter/sonnet"
    assert "model" not in dumped


def test_dream_interval_must_be_at_least_one_hour() -> None:
    with pytest.raises(ValueError):
        DreamConfig(interval_h=0)


# -- evaluator (nanobot tests/agent/test_evaluator.py) ----------------------------------------


def _eval_tool_call(should_notify, reason: str = "") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCallRequest(
                id="eval_1",
                name="evaluate_notification",
                arguments={"should_notify": should_notify, "reason": reason},
            )
        ],
    )


class _FailingProvider(FakeProvider):
    async def chat_stream(self, *args, **kwargs) -> LLMResponse:
        raise RuntimeError("provider down")


_EVAL_PROMPT = "You are a notification gate. Call evaluate_notification."


def test_resolve_evaluator_prompt_uses_workspace_override(tmp_path) -> None:
    path = evaluator_prompt_file(tmp_path)
    path.parent.mkdir()
    path.write_text("Custom evaluator prompt.\n", encoding="utf-8")
    assert resolve_evaluator_prompt(tmp_path) == "Custom evaluator prompt."


def test_resolve_evaluator_prompt_uses_default_for_empty_override(tmp_path) -> None:
    path = evaluator_prompt_file(tmp_path)
    path.parent.mkdir()
    path.write_text("  \n", encoding="utf-8")
    assert resolve_evaluator_prompt(tmp_path) == default_evaluator_prompt()


def test_resolve_evaluator_prompt_uses_default_for_undecodable_override(tmp_path) -> None:
    path = evaluator_prompt_file(tmp_path)
    path.parent.mkdir()
    path.write_bytes("Custom evaluator prompt.".encode("utf-16"))
    assert resolve_evaluator_prompt(tmp_path) == default_evaluator_prompt()


def test_resolve_evaluator_prompt_caps_workspace_override(tmp_path) -> None:
    path = evaluator_prompt_file(tmp_path)
    path.parent.mkdir()
    path.write_text("x" * (EVALUATOR_PROMPT_MAX_CHARS + 1), encoding="utf-8")
    prompt = resolve_evaluator_prompt(tmp_path)
    assert prompt.startswith("x" * EVALUATOR_PROMPT_MAX_CHARS)
    assert prompt.endswith("... (truncated)")


def test_default_evaluator_prompt_is_nanobots() -> None:
    prompt = default_evaluator_prompt()
    assert prompt.startswith("You are a notification gate for a background agent.")
    assert prompt.endswith("The user should never see the agent reasoning about whether to speak.")


async def test_should_notify_true() -> None:
    provider = FakeProvider([_eval_tool_call(True, "user asked to be reminded")])
    result = await evaluate_response(
        "Task completed with results", "check emails", provider, "m", evaluator_prompt=_EVAL_PROMPT
    )
    assert result is True


async def test_should_notify_false() -> None:
    provider = FakeProvider([_eval_tool_call(False, "routine check, nothing new")])
    result = await evaluate_response(
        "All clear, no updates", "check status", provider, "m", evaluator_prompt=_EVAL_PROMPT
    )
    assert result is False


async def test_evaluator_request() -> None:
    provider = FakeProvider([_eval_tool_call(True)])
    await evaluate_response("the answer", "the task", provider, "eval-model", evaluator_prompt=_EVAL_PROMPT)
    (call,) = provider.calls
    assert call.model == "eval-model"
    assert (call.max_tokens, call.temperature) == (4096, 0.0)
    assert call.tools == _EVALUATE_TOOL
    assert call.messages == [
        {"role": "system", "content": _EVAL_PROMPT},
        {
            "role": "user",
            "content": render_template("agent/evaluator.md", part="user", task_context="the task", response="the answer"),
        },
    ]
    assert call.messages[1]["content"] == "## Original task\nthe task\n\n## Agent response\nthe answer\n"


@pytest.mark.parametrize("decision", ["false", "true", 0, 1, None, [], {"notify": True}])
@pytest.mark.parametrize("default_notify", [False, True])
async def test_invalid_notification_decision_uses_default(decision, default_notify) -> None:
    response = LLMResponse(
        content="",
        tool_calls=[ToolCallRequest(id="eval_1", name="evaluate_notification", arguments={"should_notify": decision})],
    )
    result = await evaluate_response(
        "All clear", "check status", FakeProvider([response]), "m",
        evaluator_prompt=_EVAL_PROMPT, default_notify=default_notify,
    )
    assert result is default_notify


@pytest.mark.parametrize("default_notify", [False, True])
async def test_fallback_on_error(default_notify) -> None:
    result = await evaluate_response(
        "some response", "some task", _FailingProvider(), "m",
        evaluator_prompt=_EVAL_PROMPT, default_notify=default_notify,
    )
    assert result is default_notify


@pytest.mark.parametrize("default_notify", [False, True])
async def test_no_tool_call_fallback(default_notify) -> None:
    provider = FakeProvider([LLMResponse(content="I think you should notify", tool_calls=[])])
    result = await evaluate_response(
        "some response", "some task", provider, "m", evaluator_prompt=_EVAL_PROMPT, default_notify=default_notify
    )
    assert result is default_notify


async def test_tool_call_under_error_finish_reason_is_ignored() -> None:
    response = _eval_tool_call(True)
    response.finish_reason = "length"
    result = await evaluate_response("x", "y", FakeProvider([response]), "m", evaluator_prompt=_EVAL_PROMPT)
    assert result is False


# -- /evaluator-prompt (nanobot tests/command/test_builtin_evaluator_prompt.py) ---------------


def _make_ctx(tmp_path, raw: str = "/evaluator-prompt", args: str = "") -> CommandContext:
    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="direct", content=raw)
    loop = SimpleNamespace(context=SimpleNamespace(memory=SimpleNamespace(workspace=tmp_path)))
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, args=args, loop=loop)


async def test_evaluator_prompt_reports_default_prompt(tmp_path) -> None:
    out = await cmd_evaluator_prompt(_make_ctx(tmp_path))
    assert "Heartbeat evaluator prompt: nanoclaw default" in out.content
    assert "prompts/evaluator.md" in out.content
    assert str(tmp_path) not in out.content
    assert "/evaluator-prompt init" in out.content
    assert out.metadata["render_as"] == "text"


async def test_evaluator_prompt_init_copies_default_prompt(tmp_path) -> None:
    out = await cmd_evaluator_prompt(_make_ctx(tmp_path, "/evaluator-prompt init", "init"))
    prompt_file = tmp_path / "prompts" / "evaluator.md"
    assert "Created heartbeat evaluator prompt" in out.content
    assert "prompts/evaluator.md" in out.content
    assert str(tmp_path) not in out.content
    assert "evaluate_notification" in out.content
    assert prompt_file.read_text(encoding="utf-8") == default_evaluator_prompt() + "\n"


async def test_evaluator_prompt_init_does_not_overwrite_existing_prompt(tmp_path) -> None:
    prompt_file = tmp_path / "prompts" / "evaluator.md"
    prompt_file.parent.mkdir()
    prompt_file.write_text("custom", encoding="utf-8")
    out = await cmd_evaluator_prompt(_make_ctx(tmp_path, "/evaluator-prompt init", "init"))
    assert "already exists" in out.content
    assert "prompts/evaluator.md" in out.content
    assert str(tmp_path) not in out.content
    assert prompt_file.read_text(encoding="utf-8") == "custom"


async def test_evaluator_prompt_handles_undecodable_existing_prompt(tmp_path) -> None:
    prompt_file = tmp_path / "prompts" / "evaluator.md"
    prompt_file.parent.mkdir()
    original = "custom".encode("utf-16")
    prompt_file.write_bytes(original)
    status = await cmd_evaluator_prompt(_make_ctx(tmp_path))
    init = await cmd_evaluator_prompt(_make_ctx(tmp_path, "/evaluator-prompt init", "init"))
    assert "Heartbeat evaluator prompt: nanoclaw default" in status.content
    assert "already exists" in init.content
    assert prompt_file.read_bytes() == original


async def test_evaluator_prompt_init_recreates_empty_prompt(tmp_path) -> None:
    prompt_file = tmp_path / "prompts" / "evaluator.md"
    prompt_file.parent.mkdir()
    prompt_file.write_text("  \n", encoding="utf-8")
    out = await cmd_evaluator_prompt(_make_ctx(tmp_path, "/evaluator-prompt init", "init"))
    assert "Created heartbeat evaluator prompt" in out.content
    assert prompt_file.read_text(encoding="utf-8") == default_evaluator_prompt() + "\n"


async def test_evaluator_prompt_reports_override(tmp_path) -> None:
    prompt_file = tmp_path / "prompts" / "evaluator.md"
    prompt_file.parent.mkdir()
    prompt_file.write_text("custom", encoding="utf-8")
    out = await cmd_evaluator_prompt(_make_ctx(tmp_path))
    assert "Heartbeat evaluator prompt: custom for this workspace" in out.content
    assert "prompts/evaluator.md" in out.content


async def test_evaluator_prompt_rejects_unknown_args(tmp_path) -> None:
    out = await cmd_evaluator_prompt(_make_ctx(tmp_path, "/evaluator-prompt nope", "nope"))
    assert out.content == "Usage: /evaluator-prompt [init]"


def test_evaluator_prompt_command_in_help_and_palette() -> None:
    entry = next(item for item in builtin_command_palette() if item["command"] == "/evaluator-prompt")
    assert entry["arg_hint"] == "[init]"
    assert entry["lifecycle"] == "side_channel"
    assert entry["accepts_args"] is True
    assert "/evaluator-prompt [init]" in build_help_text()


def test_prompts_readme_explains_the_evaluator_override() -> None:
    readme = load_bundled_template("prompts/README.md")
    assert "## Heartbeat evaluator" in readme
    assert "/evaluator-prompt init" in readme


# -- helpers for the gateway tests ----------------------------------------------------------


def make_agent(tmp_path, *script) -> AgentLoop:
    ws = tmp_path / "ws"
    sync_workspace_templates(ws)
    return AgentLoop(
        FakeProvider(list(script)),
        ws,
        sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"),
        cron_service=CronService(ws / "cron" / "jobs.json"),
    )


class _TrackingMCP(MCPProvider):
    def __init__(self) -> None:
        super().__init__({}, MagicMock())
        self.connected = 0

    async def connect(self) -> None:
        self.connected += 1

    async def aclose(self) -> None:
        pass


def make_gateway(agent: AgentLoop, config: Config | None = None, console: Console | None = None) -> Gateway:
    return Gateway(
        config or Config(),
        agent,
        _TrackingMCP(),
        console=console or Console(record=True, width=200),
        health_server_enabled=False,
    )


def _write_memory_call(content: str) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="w1", name="write_file", arguments={"path": "memory/MEMORY.md", "content": content})],
        finish_reason="tool_calls",
    )


def _add_target_session(agent: AgentLoop, key: str = "telegram:u1") -> None:
    session = agent.sessions.get_or_create(key)
    session.add_message("user", "hello")
    session.add_message("assistant", "hi")
    agent.sessions.save(session)


def _write_tasks(agent: AgentLoop, tasks: str = "- Check repository health\n") -> str:
    content = f"## Active Tasks\n\n{tasks}"
    (agent.workspace / "HEARTBEAT.md").write_text(content, encoding="utf-8")
    return content


def _outbound(agent: AgentLoop) -> list:
    items = []
    while not agent.bus.outbound.empty():
        items.append(agent.bus.outbound.get_nowait())
    return items


# -- the cron service and system jobs ------------------------------------------------------


def test_gateway_reuses_the_loops_cron_service(tmp_path):
    agent = make_agent(tmp_path)
    gateway = make_gateway(agent)
    assert gateway.cron is agent.cron_service
    assert gateway.cron.on_job == gateway.on_cron_job
    assert agent.tools.has("cron")


def test_gateway_creates_a_cron_service_when_the_loop_has_none(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    agent = AgentLoop(FakeProvider([]), ws, sessions=SessionManager(ws, sessions_root=tmp_path / "s"))
    gateway = make_gateway(agent)
    assert isinstance(gateway.cron, CronService)
    assert gateway.cron.store_path == ws / "cron" / "jobs.json"


def test_make_loop_builds_exactly_one_cron_service(tmp_path):
    from nanoclaw.__main__ import _make_loop

    loop = _make_loop(FakeProvider([]), tmp_path, config=Config())
    gateway = make_gateway(loop)
    assert gateway.cron is loop.cron_service
    assert loop.tools.get("cron")._cron is gateway.cron


def test_register_system_jobs_defaults(tmp_path):
    agent = make_agent(tmp_path)
    console = Console(record=True, width=200)
    gateway = make_gateway(agent, console=console)
    gateway.print_banner()
    gateway.register_system_jobs()
    text = console.export_text()
    assert "Heartbeat: every 1800s" in text
    assert "Dream: every 2h" in text
    assert "Cron: 2 scheduled jobs" in text
    dream, heartbeat = gateway.cron.get_job("dream"), gateway.cron.get_job("heartbeat")
    assert dream.payload.kind == heartbeat.payload.kind == "system_event"
    assert dream.schedule == CronSchedule(kind="every", every_ms=2 * 3_600_000)
    tz = Config().agents.defaults.timezone
    assert heartbeat.schedule == CronSchedule(kind="every", every_ms=1_800_000, tz=tz)
    assert gateway.cron.remove_job("dream") == "protected"
    assert gateway.cron.remove_job("heartbeat") == "protected"


def test_register_system_jobs_is_idempotent_and_keeps_the_pending_run(tmp_path):
    agent = make_agent(tmp_path)
    gateway = make_gateway(agent)
    gateway.register_system_jobs()
    first = gateway.cron.get_job("dream").state.next_run_at_ms
    again = make_gateway(make_agent(tmp_path))
    again.register_system_jobs()
    assert sorted(j.id for j in again.cron.list_jobs()) == ["dream", "heartbeat"]
    assert again.cron.get_job("dream").state.next_run_at_ms == first


def test_register_system_jobs_legacy_cron_and_custom_interval(tmp_path):
    config = Config()
    config.agents.defaults.dream.cron = "0 */4 * * *"
    config.gateway.heartbeat.interval_s = 60
    console = Console(record=True, width=200)
    gateway = make_gateway(make_agent(tmp_path), config, console)
    gateway.print_banner()
    gateway.register_system_jobs()
    text = console.export_text()
    assert "Dream: cron 0 */4 * * * (legacy)" in text
    assert "Heartbeat: every 60s" in text
    assert gateway.cron.get_job("dream").schedule.expr == "0 */4 * * *"
    assert gateway.cron.get_job("heartbeat").schedule.every_ms == 60_000


def test_disabled_jobs_are_removed_and_the_dream_cursor_catches_up(tmp_path):
    agent = make_agent(tmp_path)
    make_gateway(agent).register_system_jobs()
    agent.memory.append_history("first")
    agent.memory.append_history("second")
    config = Config()
    config.agents.defaults.dream.enabled = False
    config.gateway.heartbeat.enabled = False
    console = Console(record=True, width=200)
    gateway = make_gateway(agent, config, console)
    gateway.print_banner()
    gateway.register_system_jobs()
    text = console.export_text()
    assert "Dream: disabled" in text and "Heartbeat: disabled" in text
    assert "Cron:" not in text
    assert gateway.cron.list_jobs(include_disabled=True) == []
    assert agent.memory.get_last_dream_cursor() == 2


def test_disabled_dream_cursor_only_advances_when_behind(tmp_path) -> None:
    store = MemoryStore(tmp_path)
    store.append_history("first")
    store.append_history("second")
    _advance_dream_cursor_if_behind(store)
    assert store.get_last_dream_cursor() == 2
    store.set_last_dream_cursor(10)
    _advance_dream_cursor_if_behind(store)
    assert store.get_last_dream_cursor() == 10


async def test_unbound_agent_cron_is_skipped(tmp_path):
    gateway = make_gateway(make_agent(tmp_path))
    job = CronJob(id="j1", name="remind me", payload=CronPayload(kind="agent_turn", message="hi"))
    with pytest.raises(CronJobSkippedError, match="unbound agent cron job must be recreated from a chat session"):
        await gateway.on_cron_job(job)


async def test_bound_cron_job_runs_through_the_bound_runner(tmp_path, monkeypatch):
    gateway = make_gateway(make_agent(tmp_path))
    seen = {}

    async def fake_run(job, *, agent, cron):
        seen.update(job=job, agent=agent, cron=cron)
        return "ran"

    monkeypatch.setattr(gw, "is_bound_cron_job", lambda job: True)
    monkeypatch.setattr("nanoclaw.cron.bound_runner.run_bound_cron_job", fake_run)
    job = CronJob(id="j2", name="bound")
    assert await gateway.on_cron_job(job) == "ran"
    assert seen == {"job": job, "agent": gateway.agent, "cron": gateway.cron}


# -- the Dream job ------------------------------------------------------------------------


async def test_dream_job_commits_with_the_scheduled_message(tmp_path):
    agent = make_agent(tmp_path, _write_memory_call("# Memory\n- likes tea\n"), "Updated.")
    agent.memory.append_history("- [permanent] likes tea")
    gateway = make_gateway(agent)
    assert await gateway.on_cron_job(CronJob(id="dream", name="dream")) is None
    latest = agent.memory.git.log()[0]
    assert latest.message.startswith("dream: periodic memory consolidation\n\nmemory/MEMORY.md: +2 -")
    assert agent.memory.get_last_dream_cursor() == 1
    assert gateway.mcp_provider.connected == 1
    assert not [k for k in (h["key"] for h in agent.sessions.list_sessions()) if k.startswith("dream:")]


async def test_dream_job_without_changes_makes_no_commit(tmp_path):
    agent = make_agent(tmp_path, "Nothing to change.")
    agent.memory.append_history("- small talk")
    before = [c.sha for c in agent.memory.git.log()]
    await make_gateway(agent).on_cron_job(CronJob(id="dream", name="dream"))
    assert [c.sha for c in agent.memory.git.log()] == before
    assert agent.memory.get_last_dream_cursor() == 1


async def test_dream_job_with_nothing_to_process(tmp_path):
    agent = make_agent(tmp_path)
    gateway = make_gateway(agent)
    before = [c.sha for c in agent.memory.git.log()]
    assert await gateway.on_cron_job(CronJob(id="dream", name="dream")) is None
    assert agent.provider.calls == []
    assert gateway.mcp_provider.connected == 0
    assert [c.sha for c in agent.memory.git.log()] == before


async def test_failed_dream_job_keeps_the_cursor_and_still_commits_edits(tmp_path):
    error = LLMResponse(content="boom", finish_reason="error")
    agent = make_agent(tmp_path, _write_memory_call("# Memory\n- partial\n"), error)
    agent.memory.append_history("- [permanent] partial")
    await make_gateway(agent).on_cron_job(CronJob(id="dream", name="dream"))
    assert agent.memory.get_last_dream_cursor() == 0
    assert agent.memory.git.log()[0].message.startswith("dream: periodic memory consolidation")


async def test_dream_job_exception_is_logged_not_raised(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    agent.memory.append_history("x")
    compacted = []
    monkeypatch.setattr(agent.memory, "build_dream_prompt", MagicMock(side_effect=RuntimeError("bad")))
    monkeypatch.setattr(agent.memory, "compact_history", lambda: compacted.append(True))
    assert await make_gateway(agent).on_cron_job(CronJob(id="dream", name="dream")) is None
    assert compacted == [True]


# -- the heartbeat job --------------------------------------------------------------------


async def test_heartbeat_without_heartbeat_md_does_nothing(tmp_path):
    agent = make_agent(tmp_path)
    (agent.workspace / "HEARTBEAT.md").unlink()
    gateway = make_gateway(agent)
    gateway.enabled_channels = ["telegram"]
    _add_target_session(agent)
    assert await gateway.on_cron_job(CronJob(id="heartbeat", name="heartbeat")) is None
    assert agent.provider.calls == []


async def test_heartbeat_with_the_bundled_template_does_nothing(tmp_path):
    agent = make_agent(tmp_path)
    gateway = make_gateway(agent)
    gateway.enabled_channels = ["telegram"]
    _add_target_session(agent)
    assert await gateway.on_cron_job(CronJob(id="heartbeat", name="heartbeat")) is None
    assert agent.provider.calls == []


async def test_heartbeat_with_only_the_cli_does_nothing(tmp_path):
    agent = make_agent(tmp_path)
    _write_tasks(agent)
    _add_target_session(agent, "cli:direct")
    _add_target_session(agent)  # a telegram session, but no channel is enabled
    assert await make_gateway(agent).on_cron_job(CronJob(id="heartbeat", name="heartbeat")) is None
    assert agent.provider.calls == []
    assert agent.sessions.get_existing(HEARTBEAT_SESSION_KEY) is None


async def test_heartbeat_empty_response_is_not_evaluated(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)

    async def empty_turn(*_args, **_kwargs):
        return SimpleNamespace(final_content="")

    monkeypatch.setattr(agent, "process_direct", empty_turn)
    _write_tasks(agent)
    _add_target_session(agent)
    gateway = make_gateway(agent)
    gateway.enabled_channels = ["telegram"]

    async def unexpected(*_args, **_kwargs):
        raise AssertionError("empty heartbeat response must not be evaluated")

    monkeypatch.setattr(gw, "evaluate_response", unexpected)
    assert await gateway.on_cron_job(CronJob(id="heartbeat", name="heartbeat")) is None


async def test_heartbeat_delivers_and_records_when_the_evaluator_says_so(tmp_path):
    agent = make_agent(tmp_path, "The build is red: 3 failing tests.", _eval_tool_call(True, "actionable"))
    content = _write_tasks(agent)
    _add_target_session(agent)
    gateway = make_gateway(agent)
    gateway.enabled_channels = ["telegram"]

    result = await gateway.on_cron_job(CronJob(id="heartbeat", name="heartbeat"))

    assert result == "The build is red: 3 failing tests."
    turn, evaluation = agent.provider.calls
    expected_prompt = (
        _HEARTBEAT_PREAMBLE
        + "You are executing periodic heartbeat tasks. Read the active tasks below, perform each one, "
        f"and report what you did:\n\n{content}"
    )
    assert turn.messages[-1]["content"].startswith(expected_prompt)
    assert evaluation.tools == _EVALUATE_TOOL
    assert evaluation.messages[0]["content"] == default_evaluator_prompt()
    assert evaluation.messages[1]["content"].startswith(f"## Original task\n{expected_prompt}")
    # The heartbeat turn ran in its own session.
    hb = agent.sessions.get_existing(HEARTBEAT_SESSION_KEY)
    assert [m["role"] for m in hb.messages] == ["user", "assistant"]
    # Delivered on the bus and recorded into the target session.
    (out,) = _outbound(agent)
    assert (out.channel, out.chat_id, out.content) == ("telegram", "u1", "The build is red: 3 failing tests.")
    target = agent.sessions.get_existing("telegram:u1")
    assert target.messages[-1] == {
        "role": "assistant",
        "content": "The build is red: 3 failing tests.",
        "timestamp": target.messages[-1]["timestamp"],
        "_channel_delivery": True,
    }
    # A reply to the delivery replays it as context.
    target.add_message("user", "which tests?")
    history = target.get_history()
    assert [m["content"] for m in history[-2:]] == ["The build is red: 3 failing tests.", "which tests?"]


async def test_heartbeat_silenced_by_the_evaluator(tmp_path):
    agent = make_agent(tmp_path, "All clear.", _eval_tool_call(False, "routine"))
    _write_tasks(agent)
    _add_target_session(agent)
    gateway = make_gateway(agent)
    gateway.enabled_channels = ["telegram"]
    assert await gateway.on_cron_job(CronJob(id="heartbeat", name="heartbeat")) == "All clear."
    assert _outbound(agent) == []
    assert len(agent.sessions.get_existing("telegram:u1").messages) == 2


async def test_heartbeat_evaluator_failure_fails_closed(tmp_path):
    agent = make_agent(tmp_path, "Something happened.", LLMResponse(content="no tool", tool_calls=[]))
    _write_tasks(agent)
    _add_target_session(agent)
    gateway = make_gateway(agent)
    gateway.enabled_channels = ["telegram"]
    assert await gateway.on_cron_job(CronJob(id="heartbeat", name="heartbeat")) == "Something happened."
    assert _outbound(agent) == []


async def test_heartbeat_uses_the_workspace_evaluator_prompt(tmp_path):
    agent = make_agent(tmp_path, "Report.", _eval_tool_call(False))
    _write_tasks(agent)
    _add_target_session(agent)
    evaluator_prompt_file(agent.workspace).write_text("Custom gate. Call evaluate_notification.", encoding="utf-8")
    gateway = make_gateway(agent)
    gateway.enabled_channels = ["telegram"]
    await gateway.on_cron_job(CronJob(id="heartbeat", name="heartbeat"))
    assert agent.provider.calls[-1].messages[0]["content"] == "Custom gate. Call evaluate_notification."


async def test_cli_delivery_is_published_but_not_recorded(tmp_path):
    agent = make_agent(tmp_path)
    gateway = make_gateway(agent)
    from nanoclaw.bus import OutboundMessage

    await gateway.deliver_to_channel(OutboundMessage(channel="cli", chat_id="direct", content="x"), record=True)
    assert agent.sessions.get_existing("cli:direct") is None
    assert len(_outbound(agent)) == 1


# -- in-process gateway with a fake clock ------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.offset_ms = 0

    def __call__(self) -> int:
        return int(time.time() * 1000) + self.offset_ms


async def _until(predicate, timeout: float = 10) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


async def _run_gateway_for_jobs(gateway: Gateway, clock: _Clock, monkeypatch) -> None:
    monkeypatch.setattr("nanoclaw.cron.service._now_ms", clock)
    gateway.cron.max_sleep_ms = 50
    gateway.register_system_jobs()
    run = asyncio.create_task(gateway.run())
    try:
        await _until(lambda: gateway.cron._running)
        assert gateway.cron.get_job("dream").state.last_run_at_ms is None
        clock.offset_ms = 3 * 3_600_000  # past both the heartbeat (30 min) and Dream (2 h)
        await _until(
            lambda: all(gateway.cron.get_job(j).state.last_status for j in ("dream", "heartbeat"))
        )
    finally:
        gateway.shutdown_event.set()
        await asyncio.wait_for(run, 10)
    assert gateway.cron._running is False  # stopped on shutdown


async def test_gateway_fires_dream_and_heartbeat_without_a_target(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, _write_memory_call("# Memory\n- likes tea\n"), "Updated.")
    agent.memory.append_history("- [permanent] likes tea")
    _write_tasks(agent)
    gateway = make_gateway(agent)
    await _run_gateway_for_jobs(gateway, _Clock(), monkeypatch)

    assert gateway.cron.get_job("dream").state.last_status == "ok"
    assert gateway.cron.get_job("heartbeat").state.last_status == "ok"
    assert agent.memory.git.log()[0].message.startswith("dream: periodic memory consolidation")
    assert len(agent.provider.calls) == 2  # Dream only: the heartbeat found no target
    assert agent.sessions.get_existing(HEARTBEAT_SESSION_KEY) is None


async def test_gateway_fires_heartbeat_to_a_channel_session(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, "Dream: nothing to keep.", "Disk is 95% full.", _eval_tool_call(True))
    agent.memory.append_history("- small talk")
    _write_tasks(agent, "- Check disk space\n")
    _add_target_session(agent)
    gateway = make_gateway(agent)
    gateway.enabled_channels = ["telegram"]
    seen = []
    real_evaluate = gw.evaluate_response

    async def spy(**kwargs):
        seen.append(kwargs)
        return await real_evaluate(**kwargs)

    monkeypatch.setattr(gw, "evaluate_response", spy)
    await _run_gateway_for_jobs(gateway, _Clock(), monkeypatch)

    assert len(seen) == 1 and seen[0]["response"] == "Disk is 95% full." and seen[0]["default_notify"] is False
    target = agent.sessions.get_existing("telegram:u1")
    assert target.messages[-1]["content"] == "Disk is 95% full."
    assert target.messages[-1]["_channel_delivery"] is True
    assert gateway.cron.get_job("heartbeat").state.run_history[-1].status == "ok"
