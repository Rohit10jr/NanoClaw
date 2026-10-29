"""Exec sessions wired into the agent (task 16.3).

The agent loop and the subagent manager own their exec sessions: /stop and
/new end a conversation's sessions, closing the loop ends all of them, each
subagent's processes end with it, and ``config.tools.exec`` reaches the exec
tools. The three ``aclose`` tests and the subagent close / cancel tests are
adapted from nanobot c4dac64 (tests/agent/test_task_cancel.py,
tests/agent/test_subagent_lifecycle.py); the rest are NanoClaw's own.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from conftest import WINDOWS, pid_running
from nanoclaw.__main__ import _make_loop
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.subagent import SubagentManager
from nanoclaw.agent.tools import is_tool_error
from nanoclaw.agent.tools.exec_session import ExecSessionManager
from nanoclaw.bus import InboundMessage
from nanoclaw.command.builtin import cmd_stop
from nanoclaw.command.router import CommandContext
from nanoclaw.config import Config
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.session import SessionManager
from nanoclaw.utils.llm_runtime import LLMRuntime

EXEC_TOOLS = {"exec", "exec_session", "list_exec_sessions"}


def call(name, args, call_id="c1"):
    return LLMResponse(
        content=None, tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=args)], finish_reason="tool_calls"
    )


class Sleeper:
    """A 30-second Python command that writes its pid to a file first.

    The same text runs in bash and in PowerShell (where _spawn adds ``&``
    before the quoted interpreter path), so the tests below run on every OS;
    the pid file is how they find the process (``pid_running``).
    """

    def __init__(self, folder: Path):
        self.pid_file = folder / f"sleeper-{random.randrange(10**9)}.pid"
        code = (
            f"import os,pathlib,time; pathlib.Path(r'{self.pid_file}').write_text(str(os.getpid())); "
            "time.sleep(30)"
        )
        self.command = f'"{sys.executable}" -c "{code}"'

    def pid(self, timeout: float = 10.0) -> int:
        """The sleeper's pid, once it has written it; fails if it never started."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with contextlib.suppress(OSError, ValueError):
                return int(self.pid_file.read_text())
            time.sleep(0.02)
        raise AssertionError(f"the sleeper never started ({self.pid_file} is missing)")


def running(sleeper: Sleeper) -> bool:
    return pid_running(sleeper.pid())


async def wait_until(predicate, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            return False
        await asyncio.sleep(0.02)
    return True


# Long enough for the sleeper to write its pid before exec returns (Python
# starts more slowly on Windows), short enough to return while it still runs.
YIELD_MS = 2000 if WINDOWS else 300


def started_session(sleeper: Sleeper, call_id: str = "c1") -> LLMResponse:
    """An exec call that returns while ``sleeper`` still runs (a session)."""
    return call("exec", {"command": sleeper.command, "yield_time_ms": YIELD_MS}, call_id)


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    return path


def make_loop(ws, tmp_path, script=(), **kwargs) -> AgentLoop:
    sessions = SessionManager(ws, sessions_root=tmp_path / "sessions")
    return AgentLoop(FakeProvider(list(script)), ws, sessions=sessions, **kwargs)


def runtime(provider) -> LLMRuntime:
    return LLMRuntime(provider=provider, model="m", max_tokens=100, temperature=0.1, context_window_tokens=0)


# --- the loop's manager ------------------------------------------------------


def test_agent_loops_do_not_share_exec_session_managers(tmp_path):
    """nanobot tests/agent/test_exec_session_isolation.py."""
    loop_a = AgentLoop(FakeProvider([]), tmp_path / "a")
    loop_b = AgentLoop(FakeProvider([]), tmp_path / "b")

    assert loop_a.tools.get("exec")._session_manager is loop_a._exec_session_manager
    assert loop_a.tools.get("exec_session")._manager is loop_a._exec_session_manager
    assert loop_a.tools.get("list_exec_sessions")._manager is loop_a._exec_session_manager
    assert loop_b.tools.get("exec")._session_manager is loop_b._exec_session_manager
    assert loop_a._exec_session_manager is not loop_b._exec_session_manager


async def test_aclose_closes_exec_sessions_after_background_work(tmp_path):
    """nanobot test_shutdown_drains_background_failure_and_logs_it_once, adapted:
    NanoClaw cancels command background work (/dream) instead of draining it;
    resources close only after it ended, each once."""
    loop = AgentLoop(FakeProvider([]), tmp_path)
    loop.subagents.close = AsyncMock()
    loop._exec_session_manager.close_all = AsyncMock()
    started = asyncio.Event()

    async def background():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(background())
    loop._background_tasks.add(task)
    await started.wait()
    await loop.aclose()

    assert task.cancelled()
    assert not loop._background_tasks
    loop.subagents.close.assert_awaited_once()
    loop._exec_session_manager.close_all.assert_awaited_once()


async def test_aclose_cancels_active_turn_before_resources(tmp_path):
    """nanobot test_aclose_cancels_active_turn_before_resources."""
    loop = AgentLoop(FakeProvider([]), tmp_path)
    events: list[str] = []

    async def active_turn():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            events.append("turn_cancelled")
            raise

    task = asyncio.create_task(active_turn())
    await asyncio.sleep(0)
    loop._active_tasks["test:c1"] = {task}

    async def close_subagents():
        events.append("resources_closed")

    async def close_exec():
        events.append("exec_closed")
        return 0

    loop.subagents.close = close_subagents
    loop._exec_session_manager.close_all = close_exec
    await loop.aclose()

    assert events == ["turn_cancelled", "resources_closed", "exec_closed"]
    assert task.cancelled()


async def test_aclose_serializes_duplicate_cleanup(tmp_path):
    """nanobot test_aclose_serializes_duplicate_cleanup."""
    loop = AgentLoop(FakeProvider([]), tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    concurrent = 0
    max_concurrent = 0

    async def close_subagents():
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        entered.set()
        await release.wait()
        concurrent -= 1

    loop.subagents.close = close_subagents
    loop._exec_session_manager.close_all = AsyncMock()
    first = asyncio.create_task(loop.aclose())
    await entered.wait()
    second = asyncio.create_task(loop.aclose())
    await asyncio.sleep(0)
    assert not second.done()
    release.set()
    await asyncio.gather(first, second)

    assert max_concurrent == 1
    assert loop._exec_session_manager.close_all.await_count == 2


async def test_exec_close_runs_even_when_subagent_close_fails(tmp_path):
    """nanobot loop.py:1817: every cleanup step runs; a single error is re-raised."""
    loop = AgentLoop(FakeProvider([]), tmp_path)
    loop.subagents.close = AsyncMock(side_effect=RuntimeError("subagents"))
    loop._exec_session_manager.close_all = AsyncMock()
    with pytest.raises(RuntimeError, match="subagents"):
        await loop.aclose()
    loop._exec_session_manager.close_all.assert_awaited_once()


async def test_aclose_ends_a_running_exec_session(ws, tmp_path):
    sleeper = Sleeper(ws)
    loop = make_loop(ws, tmp_path, [started_session(sleeper), "started"])
    result = await loop.process_direct("go")
    tool_result = next(m["content"] for m in result.messages if m["role"] == "tool")
    assert "Process running. session_id:" in tool_result
    assert running(sleeper)
    assert loop._has_pending_work()

    await loop.aclose()

    assert await wait_until(lambda: not running(sleeper))
    assert not loop._exec_session_manager.has_pending_work


async def test_new_ends_only_that_sessions_exec_sessions(ws, tmp_path):
    mine, other = Sleeper(ws), Sleeper(ws)
    loop = make_loop(ws, tmp_path, [started_session(mine), "a", started_session(other), "b"])
    try:
        await loop.process_direct("go", session_key="cli:a")
        await loop.process_direct("go", session_key="cli:b")
        assert running(mine) and running(other)

        reply = await loop.process_direct("/new", session_key="cli:a")

        assert reply.final_content == "New session started."
        assert await wait_until(lambda: not running(mine))
        assert running(other)
        assert [s.owner_session_key for s in await loop._exec_session_manager.list(owner_session_key="cli:b")] == [
            "cli:b"
        ]
    finally:
        await loop.aclose()


async def test_stop_counts_and_ends_exec_sessions(ws, tmp_path):
    """nanobot ``_cancel_active_tasks`` adds ``terminate_by_owner``'s count
    (loop.py:983), so /stop reports the ended sessions."""
    sleeper = Sleeper(ws)
    loop = make_loop(ws, tmp_path, [started_session(sleeper), "started"])
    try:
        await loop.process_direct("go", session_key="cli:direct")
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/stop")
        out = await cmd_stop(CommandContext(msg=msg, session=None, key="cli:direct", raw="/stop", loop=loop))
        assert out.content == "Stopped 1 task(s)."
        assert await wait_until(lambda: not running(sleeper))
        out = await cmd_stop(CommandContext(msg=msg, session=None, key="cli:direct", raw="/stop", loop=loop))
        assert out.content == "No active task to stop."
    finally:
        await loop.aclose()


async def test_cancel_active_tasks_terminates_the_owners_exec_sessions(ws, tmp_path):
    loop = make_loop(ws, tmp_path)
    loop._exec_session_manager.terminate_by_owner = AsyncMock(return_value=2)
    assert await loop.cancel_active_tasks("cli:x") == 2
    loop._exec_session_manager.terminate_by_owner.assert_awaited_once_with("cli:x")


# --- subagents ---------------------------------------------------------------


async def test_subagent_close_cancels_tasks_before_closing_exec_sessions(ws):
    """nanobot tests/agent/test_subagent_lifecycle.py."""
    sm = SubagentManager(ws)
    entered = asyncio.Event()

    async def run(spec):
        entered.set()
        await asyncio.Event().wait()

    sm.runner.run = run
    await sm.spawn("task", runtime=runtime(FakeProvider([])))
    task = next(iter(sm._running_tasks.values()))
    await entered.wait()

    async def close_exec_sessions() -> int:
        assert task.done()
        return 0

    sm._exec_session_manager.close_all = AsyncMock(side_effect=close_exec_sessions)

    await sm.close()

    assert task.cancelled()
    sm._exec_session_manager.close_all.assert_awaited_once()
    assert not sm.has_pending_work


async def test_subagent_cancel_by_session_leaves_the_shared_manager_alone(ws):
    """nanobot tests/agent/test_task_cancel.py: children own their sessions, so
    cancelling a conversation's children never sweeps the manager's own."""
    sm = SubagentManager(ws)
    mock_exec_mgr = AsyncMock(spec=ExecSessionManager)
    mock_exec_mgr.terminate_by_owner = AsyncMock(return_value=0)
    sm._exec_session_manager = mock_exec_mgr

    await sm.cancel_by_session("test:c1")

    mock_exec_mgr.terminate_by_owner.assert_not_awaited()


async def test_each_child_gets_its_own_exec_manager(ws):
    sm = SubagentManager(ws)
    seen = []

    async def run(spec):
        seen.append(spec.tools.get("exec")._session_manager)
        assert spec.tools.get("exec_session")._manager is seen[-1]
        assert spec.tools.get("list_exec_sessions")._manager is seen[-1]
        from nanoclaw.agent.runner import AgentRunResult

        return AgentRunResult(final_content="done", messages=[], stop_reason="completed")

    sm.runner.run = run
    await sm.run_inline("one", runtime=runtime(FakeProvider([])))
    await sm.run_inline("two", runtime=runtime(FakeProvider([])))
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert sm._exec_session_manager not in seen
    assert all(m._closed for m in seen)  # closed when the child ended
    assert not sm._exec_managers and not sm.has_pending_work


async def test_child_processes_end_when_the_child_finishes(ws):
    sleeper = Sleeper(ws)
    sm = SubagentManager(ws)
    provider = FakeProvider([started_session(sleeper), "child done"])
    seen = []
    original = sm._cleanup_exec

    async def spy(task_id):
        seen.append(bool(running(sleeper)))
        return await original(task_id)

    sm._cleanup_exec = spy
    result = await sm.run_inline("t", runtime=runtime(provider))

    assert result == "child done"
    assert seen == [True]  # still running when the child's run ended
    assert await wait_until(lambda: not running(sleeper))
    assert not sm.has_pending_work


async def test_background_child_processes_end_before_its_report(ws):
    sleeper = Sleeper(ws)
    sm = SubagentManager(ws)
    await sm.spawn("t", runtime=runtime(FakeProvider([started_session(sleeper), "child done"])))
    report = await asyncio.wait_for(sm.bus.consume_inbound(), 5)
    assert "child done" in report.content
    assert not running(sleeper)


class BlockingProvider(FakeProvider):
    """Replays its script, then waits forever (a child that stays busy)."""

    def __init__(self, script):
        super().__init__(script)
        self.blocked = asyncio.Event()

    async def chat(self, *args, **kwargs):
        if not self._script:
            self.blocked.set()
            await asyncio.Event().wait()
        return await super().chat(*args, **kwargs)


@pytest.mark.parametrize("how", ["cancel_by_session", "close"])
async def test_cancelled_child_processes_end(ws, how):
    sleeper = Sleeper(ws)
    sm = SubagentManager(ws)
    provider = BlockingProvider([started_session(sleeper)])
    await sm.spawn("t", session_key="cli:a", runtime=runtime(provider))
    await asyncio.wait_for(provider.blocked.wait(), 5)
    assert running(sleeper)

    if how == "close":
        await sm.close()
    else:
        assert await sm.cancel_by_session("cli:a") == 1

    assert not running(sleeper)
    assert not sm.has_pending_work
    assert sm.bus.inbound.empty()  # a cancelled child does not report


async def test_failed_child_cleanup_is_an_error_and_retried_on_close(ws):
    """nanobot test_cleanup_failure_survives_status_eviction_and_shutdown_retries,
    reduced: the result is the cleanup error and close() tries again."""
    sm = SubagentManager(ws)
    created = []
    original_new = sm._new_exec_manager

    def new_manager(task_id):
        manager = original_new(task_id)
        original_close = manager.close_all
        manager.close_all = AsyncMock(side_effect=[RuntimeError("cleanup failed"), 0])
        created.append((manager, original_close))
        return manager

    sm._new_exec_manager = new_manager
    result = await sm.run_inline("t", runtime=runtime(FakeProvider(["child done"])))

    assert is_tool_error(result)
    assert result.startswith("Error cleaning up task processes: cleanup failed")
    assert sm.has_pending_work
    await sm.close()
    assert created[0][0].close_all.await_count == 2
    assert not sm.has_pending_work


# --- config.tools.exec -------------------------------------------------------


def exec_config_json():
    return {
        "maxRuntime": 120,
        "maxRuntimeLimit": 600,
        "idleTimeout": 7,
        "idleAction": "terminate",
        "pathPrepend": "/opt/pre",
        "pathAppend": "/opt/post",
        "allowedEnvKeys": ["MY_KEY"],
        "allowPatterns": [r"^git\b"],
        "denyPatterns": [r"\bshutdown\b"],
    }


def assert_configured(tool):
    assert tool.max_runtime == 120 and tool.timeout == 120
    assert tool.max_runtime_limit == 600
    assert tool.idle_timeout == 7
    assert tool.idle_action == "terminate"
    assert tool.path_prepend == "/opt/pre" and tool.path_append == "/opt/post"
    assert tool.allowed_env_keys == ["MY_KEY"]
    assert tool.allow_patterns == [r"^git\b"]
    assert tool.deny_patterns[0] == r"\bshutdown\b"  # then the built-in ones


def test_config_reaches_the_exec_tools_of_the_loop_and_subagents(ws):
    config = Config.model_validate({"tools": {"exec": exec_config_json()}})
    loop = _make_loop(FakeProvider([]), ws, config=config)
    assert_configured(loop.tools.get("exec"))
    assert_configured(loop.subagents._build_tools().get("exec"))
    # The schema shows the configured budget, as in nanobot.
    params = loop.tools.get("exec").parameters["properties"]
    assert "600" in params["max_runtime"]["description"]


def test_legacy_timeout_reaches_the_exec_tool(ws):
    config = Config.model_validate({"tools": {"exec": {"timeout": 45}}})
    assert _make_loop(FakeProvider([]), ws, config=config).tools.get("exec").max_runtime == 45


def test_defaults_match_exec_tool_defaults(ws):
    tool = _make_loop(FakeProvider([]), ws, config=Config()).tools.get("exec")
    assert (tool.max_runtime, tool.max_runtime_limit, tool.idle_timeout, tool.idle_action) == (
        3600, 86400, 300, "report"
    )


def test_exec_disabled_registers_no_exec_tools(ws):
    config = Config.model_validate({"tools": {"exec": {"enable": False}}})
    loop = _make_loop(FakeProvider([]), ws, config=config)
    assert not EXEC_TOOLS & set(loop.tools.tool_names)
    assert not EXEC_TOOLS & set(loop.subagents._build_tools().tool_names)
    assert {"read_file", "spawn", "web_fetch"} <= set(loop.tools.tool_names)


async def test_configured_deny_pattern_blocks_a_command(ws, tmp_path):
    config = Config.model_validate({"tools": {"exec": {"denyPatterns": [r"\bforbidden\b"]}}})
    loop = _make_loop(FakeProvider([call("exec", {"command": "echo forbidden"}), "ok"]), ws, config=config)
    try:
        result = await loop.process_direct("go")
        tool_result = next(m["content"] for m in result.messages if m["role"] == "tool")
        assert "blocked by deny pattern filter" in tool_result
    finally:
        await loop.aclose()


# --- nanoclaw ask, as a real process -----------------------------------------


def test_ask_leaves_no_exec_process_behind(tmp_path):
    """``nanoclaw ask`` whose model starts a long command that outlives the
    exec call: once the command exits, the process is gone."""
    sleeper = Sleeper(tmp_path)
    script = textwrap.dedent(
        f"""
        import sys
        from nanoclaw.__main__ import main
        from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest

        call = LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(
                id="c1", name="exec",
                arguments={{"command": {sleeper.command!r}, "yield_time_ms": {YIELD_MS}}},
            )],
            finish_reason="tool_calls",
        )
        sys.exit(main(["ask", "start it", "-w", {str(tmp_path / "ws")!r}], provider=FakeProvider([call, "started"])))
        """
    )
    env = {**os.environ, "HOME": str(tmp_path), "NANOCLAW_CONFIG": str(tmp_path / "config.json")}
    done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60)

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "started"
    assert not pid_running(sleeper.pid())  # it really started, and is gone



# --- nanoclaw chat -----------------------------------------------------------


def make_chat(ws, tmp_path, script, inputs):
    import io

    from rich.console import Console

    from nanoclaw.cli.chat import Chat

    loop = make_loop(ws, tmp_path, script)
    pending = list(inputs)

    async def read_input():
        if not pending:
            raise EOFError
        item = pending.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return await item()
        return item

    console = Console(file=io.StringIO(), width=120, color_system=None, force_terminal=False)
    return Chat(loop, console=console, read_input=read_input, render_markdown=False), console.file


@pytest.mark.parametrize("leave", ["exit", KeyboardInterrupt()], ids=["exit", "ctrl-c"])
async def test_leaving_the_chat_ends_exec_sessions(ws, tmp_path, leave):
    sleeper = Sleeper(ws)
    seen = []

    async def check_then_leave():
        seen.append(bool(running(sleeper)))
        if isinstance(leave, BaseException):
            raise leave
        return leave

    chat, _ = make_chat(ws, tmp_path, [started_session(sleeper), "started"], ["go", check_then_leave])
    await chat.run()
    assert seen == [True]
    assert not running(sleeper)


async def test_chat_stop_ends_the_sessions_exec_sessions(ws, tmp_path):
    sleeper = Sleeper(ws)
    seen = []

    async def check():
        seen.append(bool(running(sleeper)))
        return "exit"

    chat, out = make_chat(ws, tmp_path, [started_session(sleeper), "started"], ["go", "/stop", check])
    await chat.run()
    assert seen == [False]
    assert "Stopped 1 task(s)." in out.getvalue()
