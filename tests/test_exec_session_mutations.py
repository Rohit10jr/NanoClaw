"""Focused exec-session, shell and wiring checks found by mutation testing (layer 16)."""

from __future__ import annotations

import asyncio
import os
import re
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nanoclaw.agent.tools import exec_session
from nanoclaw.agent.tools.exec_session import (
    ExecSessionManager,
    ExecSessionTool,
    _BoundedOutputBuffer,
    _ExecSession,
    _SessionPoll,
    format_session_poll,
)
from nanoclaw.agent.subagent import SubagentManager
from nanoclaw.agent.tools.shell import ExecTool
from nanoclaw.providers import FakeProvider
from nanoclaw.utils.llm_runtime import LLMRuntime


def _fake_process() -> tuple[SimpleNamespace, asyncio.Event]:
    exited = asyncio.Event()
    process = SimpleNamespace(pid=12345, stdout=None, stderr=None, stdin=None, returncode=None)

    async def wait() -> int:
        await exited.wait()
        return process.returncode

    process.wait = wait
    return process, exited


async def _cancel_session_tasks(session: _ExecSession) -> None:
    tasks = [session._stdout_task, session._stderr_task, session._completion_task]
    tasks.extend(task for task in (session._timeout_task, session._kill_task) if task is not None)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _session_id(output: str) -> str:
    match = re.search(r"session_id:\s*([0-9a-f]+)", output)
    assert match, output
    return match.group(1)


# -- output buffer ------------------------------------------------------------


def test_bounded_buffer_trims_the_oldest_tail_chars_exactly():
    buffer = _BoundedOutputBuffer(10)
    buffer.append("012345")
    buffer.append("6789ABCDEF")
    buffer.append("XY")  # trims part of the retained tail chunk

    assert buffer.retained_chars == 10
    assert buffer.drain() == ("01234DEFXY", 8)

    buffer.append("0123456789ABCDEF")
    buffer.append("Z")  # one char over the budget
    assert buffer.drain() == ("01234CDEFZ", 7)


def test_bounded_buffer_keeps_a_fresh_head_after_drain():
    buffer = _BoundedOutputBuffer(10)
    buffer.append("0123456789AB")
    assert buffer.drain() == ("01234789AB", 2)

    buffer.append("abcdefghijKL")
    assert buffer.drain() == ("abcdehijKL", 2)


# -- poll ---------------------------------------------------------------------


async def test_poll_wait_is_capped_at_max_yield(monkeypatch):
    monkeypatch.setattr(exec_session, "MAX_YIELD_MS", 50)
    process, _ = _fake_process()
    session = _ExecSession(session_id="cap", process=process, command="x", cwd=".", timeout=None)
    try:
        started = time.monotonic()
        poll = await session.poll(5000, 1000)
        assert time.monotonic() - started < 2
        assert not poll.done
    finally:
        await _cancel_session_tasks(session)


# -- format_session_poll ---------------------------------------------------------


def test_format_reports_a_stopping_session():
    text = format_session_poll("abc", _SessionPoll(output="", done=False, exit_code=None, stopping=True))
    assert "Process stopping. session_id: abc" in text


def test_format_reports_no_silence_without_an_idle_limit():
    poll = _SessionPoll(output="", done=False, exit_code=None, output_idle_s=50.0, idle_timeout=0)
    assert "Output silent" not in format_session_poll("abc", poll)
    poll.idle_timeout = 10
    assert "Output silent for 50.0s" in format_session_poll("abc", poll)


def test_format_reports_unlimited_runtime():
    poll = _SessionPoll(output="", done=False, exit_code=None, remaining_s=None)
    assert "Runtime remaining: unlimited" in format_session_poll("abc", poll)
    poll.remaining_s = 12.34
    assert "Runtime remaining: 12.3s" in format_session_poll("abc", poll)


# -- supervision --------------------------------------------------------------------


async def test_idle_terminate_applies_without_a_runtime_budget(tmp_path):
    manager = ExecSessionManager()
    tool = ExecTool(
        working_dir=str(tmp_path), max_runtime=0, idle_timeout=1, idle_action="terminate",
        exec_session_manager=manager,
    )
    try:
        initial = await tool.execute(command="sleep 30", yield_time_ms=0)
        result = await ExecSessionTool(manager=manager).execute(
            session_id=_session_id(initial), until_exit=True, timeout_ms=8000,
        )
        assert "Reason: idle_timeout." in result
    finally:
        await manager.close_all()


async def test_failed_kill_is_recorded_as_cleanup_error(monkeypatch):
    async def failing_kill(process):
        raise RuntimeError("kill boom")

    monkeypatch.setattr(ExecTool, "_kill_process", staticmethod(failing_kill))
    process, _ = _fake_process()
    session = _ExecSession(session_id="killfail", process=process, command="x", cwd=".", timeout=None)
    try:
        with pytest.raises(RuntimeError, match="kill boom"):
            await session.kill()
        assert session.cleanup_error == "kill boom"
        assert session.has_pending_work
    finally:
        await _cancel_session_tasks(session)


async def test_failed_supervision_is_recorded_and_ends_waits():
    process, _ = _fake_process()

    async def failing_wait() -> int:
        raise RuntimeError("wait boom")

    process.wait = failing_wait
    session = _ExecSession(session_id="supfail", process=process, command="x", cwd=".", timeout=None)
    try:
        with pytest.raises(RuntimeError, match="cleanup failed: wait boom"):
            await asyncio.wait_for(session.wait_exit(), timeout=2)
        assert session.cleanup_error == "wait boom"
    finally:
        await _cancel_session_tasks(session)


async def test_stop_with_pipe_still_held_reports_incomplete_output(monkeypatch):
    stdout = asyncio.StreamReader()
    stdout.feed_data(b"partial\n")  # no EOF: a descendant still holds the pipe
    process, exited = _fake_process()
    process.stdout = stdout

    async def kill(proc):
        proc.returncode = -9
        exited.set()

    monkeypatch.setattr(ExecTool, "_kill_process", staticmethod(kill))
    session = _ExecSession(session_id="held", process=process, command="x", cwd=".", timeout=None)
    try:
        await asyncio.wait_for(session.kill(), timeout=5)
        poll = await session.poll(0, 1000)
        assert poll.done and poll.finish_reason == "cancelled"
        assert poll.output_incomplete
        assert "partial" in poll.output
        assert "output may be incomplete" in format_session_poll("held", poll)
    finally:
        await _cancel_session_tasks(session)


# -- stdin ----------------------------------------------------------------------------


class _BrokenStdin:
    def write(self, data: bytes) -> None:
        raise BrokenPipeError

    async def drain(self) -> None:  # pragma: no cover - write raises first
        pass


async def test_stdin_errors_are_reported_as_text():
    process, exited = _fake_process()
    session = _ExecSession(session_id="stdin", process=process, command="x", cwd=".", timeout=None)
    try:
        assert await session.write("hi") == "session stdin is not available"
        assert await session.close_stdin() == "session stdin is not available"
        process.stdin = _BrokenStdin()
        assert await session.write("hi") == "session stdin is closed"
        process.returncode = 0
        assert await session.write("hi") == "session has already exited"
        assert await session.close_stdin() == "session has already exited"
    finally:
        await _cancel_session_tasks(session)


async def test_manager_raises_stdin_errors():
    manager = ExecSessionManager()
    process, _ = _fake_process()
    session = _ExecSession(session_id="nostdin", process=process, command="x", cwd=".", timeout=None)
    manager._sessions["nostdin"] = session
    try:
        for chars, close in (("hi", False), (None, True)):
            with pytest.raises(RuntimeError, match="session stdin is not available"):
                await manager.write(
                    session_id="nostdin", chars=chars, close_stdin=close, terminate=False,
                    yield_time_ms=0, max_output_chars=1000,
                )
    finally:
        await _cancel_session_tasks(session)


# -- manager capacity and owners ------------------------------------------------------


class _GatedManager(ExecSessionManager):
    """Spawning waits for ``gate``, so a launch stays in the startup reservations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gate = asyncio.Event()
        self.spawning = asyncio.Event()

    async def _spawn(self, *args, **kwargs):
        self.spawning.set()
        await self.gate.wait()
        return await super()._spawn(*args, **kwargs)


async def _start(manager: ExecSessionManager, owner: str | None) -> tuple[str, _SessionPoll]:
    return await manager.start(
        command="true", cwd=".", env={"PATH": "/usr/bin:/bin"}, timeout=10, shell_program=None,
        login=False, yield_time_ms=0, max_output_chars=1000, owner_session_key=owner,
    )


async def test_capacity_counts_sessions_still_starting():
    manager = _GatedManager(max_sessions=1)
    first = asyncio.create_task(_start(manager, "a"))
    try:
        await asyncio.wait_for(manager.spawning.wait(), timeout=2)
        with pytest.raises(RuntimeError, match=r"maximum exec sessions reached \(1\)"):
            await asyncio.wait_for(_start(manager, "a"), timeout=2)
    finally:
        manager.gate.set()
        await asyncio.wait_for(first, timeout=5)
        await manager.close_all()


async def test_terminate_by_owner_leaves_other_owners_launches_alone():
    manager = _GatedManager()
    other = asyncio.create_task(_start(manager, "a"))
    try:
        await asyncio.wait_for(manager.spawning.wait(), timeout=2)
        assert await manager.terminate_by_owner("b") == 0
        assert not other.done()
        manager.gate.set()
        session_id, _ = await asyncio.wait_for(other, timeout=5)
        assert session_id
    finally:
        manager.gate.set()
        await asyncio.gather(other, return_exceptions=True)
        await manager.close_all()


# -- exec_session tool ------------------------------------------------------------------


async def test_terminate_rejects_a_timeout():
    tool = ExecSessionTool(manager=ExecSessionManager())
    result = await tool.execute(session_id="x", terminate=True, timeout_ms=100)
    assert "terminate must be used alone" in result


def test_clamp_applies_the_minimum():
    assert exec_session.clamp_session_int(-5, 1, 0, 10) == 0
    assert exec_session.clamp_session_int(50, 1, 0, 10) == 10
    assert exec_session.clamp_session_int(None, 7, 0, 10) == 7


async def test_wait_for_matches_text_split_across_polls(tmp_path):
    manager = ExecSessionManager()
    tool = ExecTool(working_dir=str(tmp_path), exec_session_manager=manager)
    try:
        # The first chunk comes after a delay so that exec's own (immediate)
        # read can't consume "ab" under load: both halves must reach the wait.
        initial = await tool.execute(
            command="sleep 1; printf 'ab'; sleep 0.8; printf 'cd'; sleep 30", yield_time_ms=0,
        )
        assert "ab" not in initial
        # A missed match would end at the timeout with "Wait target not
        # observed", so no wall-clock assertion is needed (it failed under load).
        result = await ExecSessionTool(manager=manager).execute(
            session_id=_session_id(initial), wait_for="bc", timeout_ms=10000,
        )
        assert "Wait target not observed" not in result
    finally:
        await manager.close_all()


async def test_until_exit_defaults_to_the_long_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(exec_session, "DEFAULT_WAIT_FOR_MS", 200)
    manager = ExecSessionManager()
    tool = ExecTool(working_dir=str(tmp_path), exec_session_manager=manager)
    try:
        initial = await tool.execute(command="sleep 1; echo finished", yield_time_ms=0)
        result = await ExecSessionTool(manager=manager).execute(session_id=_session_id(initial), until_exit=True)
        assert "finished" in result and "Exit code: 0" in result
    finally:
        await manager.close_all()


async def test_wait_counts_session_and_response_truncation(tmp_path):
    manager = ExecSessionManager()
    tool = ExecTool(working_dir=str(tmp_path), exec_session_manager=manager)
    try:
        initial = await tool.execute(
            command="while [ ! -e go ]; do sleep 0.01; done; head -c 120000 /dev/zero | tr '\\0' x",
            yield_time_ms=0,
        )
        (tmp_path / "go").touch()
        result = await ExecSessionTool(manager=manager).execute(
            session_id=_session_id(initial), until_exit=True, timeout_ms=5000,
        )
        # 70,000 dropped by the session buffer, 40,000 more by the response budget.
        assert "(110,000 chars truncated from output)" in result
    finally:
        await manager.close_all()


async def test_list_reports_a_stopping_session():
    manager = ExecSessionManager()
    process, _ = _fake_process()
    session = _ExecSession(session_id="stopping1", process=process, command="x", cwd=".", timeout=None)
    session._stop_reason = "cancelled"
    manager._sessions["stopping1"] = session
    try:
        listed = await exec_session.ListExecSessionsTool(manager=manager, session_key_getter=lambda: None).execute()
        assert "stopping1 | stopping |" in listed
    finally:
        await _cancel_session_tasks(session)


async def test_list_reports_silence_long_commands_and_cleanup_errors():
    manager = ExecSessionManager()
    process, _ = _fake_process()
    session = _ExecSession(
        session_id="listme", process=process, command="echo " + "x" * 200, cwd="/w", timeout=None,
        idle_timeout=1, started_at=time.monotonic() - 5,
    )
    session.cleanup_error = "kill failed"
    manager._sessions["listme"] = session
    try:
        listed = await exec_session.ListExecSessionsTool(manager=manager, session_key_getter=lambda: None).execute()
        line, cleanup = listed.splitlines()
        assert "| silent=true |" in line
        assert line.endswith("| cwd=/w | " + ("echo " + "x" * 200)[:119] + "...")
        assert cleanup == "Cleanup failed: kill failed"
    finally:
        await _cancel_session_tasks(session)


def test_filter_strips_csi_ending_in_at_sign():
    from nanoclaw.agent.tools.shell_output import ShellOutputFilter

    assert ShellOutputFilter().feed("a\x1b[2@b") == "ab"


# -- shell.py session path ------------------------------------------------------------


def test_argv_command_gets_the_composed_path(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    prepared = ExecTool(path_prepend="/pre", path_append="/post")._prepare_command(["echo", "hi"])
    assert prepared.env["PATH"] == os.pathsep.join(["/pre", "/usr/bin:/bin", "/post"])


async def test_tiny_output_limit_is_raised_to_the_minimum(tmp_path):
    manager = ExecSessionManager()
    tool = ExecTool(working_dir=str(tmp_path), exec_session_manager=manager)
    try:
        result = await tool.execute(command="head -c 3000 /dev/zero | tr '\\0' x", max_output_chars=5)
        assert "(2,000 chars truncated from output)" in result
    finally:
        await manager.close_all()


async def test_login_flag_applies_only_to_bash_and_zsh():
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        await ExecTool._spawn("echo hi", "/tmp", {}, shell_program="/bin/sh", login=True)
        assert mock_exec.call_args[0] == ("/bin/sh", "-c", "echo hi")
        await ExecTool._spawn("echo hi", "/tmp", {}, shell_program="/bin/bash", login=True)
        assert mock_exec.call_args[0] == ("/bin/bash", "-l", "-c", "echo hi")


# -- subagent wiring --------------------------------------------------------------------


def _runtime(provider) -> LLMRuntime:
    return LLMRuntime(provider=provider, model="m", max_tokens=100, temperature=0.1, context_window_tokens=0)


async def test_failed_child_cleanup_replaces_the_answer(tmp_path):
    sm = SubagentManager(tmp_path)
    original_new = sm._new_exec_manager

    def new_manager(task_id):
        manager = original_new(task_id)
        manager.close_all = AsyncMock(side_effect=[RuntimeError("cleanup failed"), 0])
        return manager

    sm._new_exec_manager = new_manager
    result = await sm.run_inline("t", runtime=_runtime(FakeProvider(["child done"])))
    assert "Partial result" not in result and "child done" not in result
    assert result.endswith("Error cleaning up task processes: cleanup failed")
    await sm.close()


async def test_subagent_close_raises_a_failed_cleanup(tmp_path):
    sm = SubagentManager(tmp_path)
    sm._exec_session_manager.close_all = AsyncMock(side_effect=RuntimeError("close boom"))
    with pytest.raises(RuntimeError, match="close boom"):
        await sm.close()


async def test_child_cancelled_before_it_starts_releases_its_exec_manager(tmp_path):
    sm = SubagentManager(tmp_path)
    await sm.spawn("t", runtime=_runtime(FakeProvider([])))
    task = next(iter(sm._running_tasks.values()))
    task.cancel()  # before its first step: _run_subagent never runs
    await asyncio.gather(task, return_exceptions=True)
    for _ in range(50):
        if not sm._exec_managers:
            break
        await asyncio.sleep(0.01)
    assert not sm._exec_managers
    assert not sm.has_pending_work


# -- Windows branches (mocked) ---------------------------------------------------------------


async def test_windows_falls_back_to_powershell_by_name():
    with (
        patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
        patch("nanoclaw.agent.tools.shell.shutil.which", return_value=None),
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
    ):
        await ExecTool._spawn("dir", r"C:\work", {"PATH": ""})
    assert mock_exec.call_args[0][0] == "powershell"


async def test_windows_cmd_runs_through_the_chosen_cmd():
    env = {"COMSPEC": r"C:\Windows\system32\cmd.exe", "PATH": ""}
    with (
        patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
        patch("asyncio.create_subprocess_shell", new_callable=AsyncMock) as mock_shell,
    ):
        await ExecTool._spawn("echo hi", r"C:\work", env, shell_program=r"D:\Tools\CMD.EXE")
    assert mock_shell.call_args[1]["env"]["COMSPEC"] == r"D:\Tools\CMD.EXE"
    assert env["COMSPEC"] == r"C:\Windows\system32\cmd.exe"  # the caller's env is untouched


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('"C:\\Tools\\x.exe"-v', '"C:\\Tools\\x.exe"-v'),  # not a separate token
        ('"\\\\server\\share\\tool" -v', '& "\\\\server\\share\\tool" -v'),
        ('"tool.EXE" -v', '& "tool.EXE" -v'),
        ('  "C:\\x.exe"', '  & "C:\\x.exe"'),
        ('"just text"', '"just text"'),
    ],
)
def test_normalize_powershell_command(command, expected):
    assert ExecTool._normalize_powershell_command(command) == expected


def test_windows_shell_names(monkeypatch):
    monkeypatch.setenv("COMSPEC", r"D:\Custom\cmd.exe")
    with (
        patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
        patch("nanoclaw.agent.tools.shell.shutil.which", side_effect=lambda name: rf"C:\bin\{name}"),
    ):
        assert ExecTool._resolve_shell("cmd.exe") == (r"D:\Custom\cmd.exe", None)
        assert ExecTool._resolve_shell("CMD") == (r"D:\Custom\cmd.exe", None)
        assert ExecTool._resolve_shell("PWSH") == (r"C:\bin\PWSH", None)
        _, error = ExecTool._resolve_shell("bin/pwsh")
        assert "shell must be a shell name or absolute path" in error


def test_windows_absolute_shell_paths(tmp_path):
    shell = tmp_path / "PowerShell.EXE"
    shell.write_text("")
    with patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True):
        assert ExecTool._resolve_shell(str(shell)) == (str(shell), None)
        _, error = ExecTool._resolve_shell(str(tmp_path / "pwsh.exe"))
        assert "shell is not found" in error


def test_windows_env_forwards_comspec_and_allowed_keys(monkeypatch):
    monkeypatch.setenv("COMSPEC", r"D:\Custom\cmd.exe")
    monkeypatch.setenv("MY_TOOL_HOME", r"D:\tool")
    with patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True):
        env = ExecTool(allowed_env_keys=["MY_TOOL_HOME", "NOT_SET_ANYWHERE_123"])._build_env()
    assert env["COMSPEC"] == r"D:\Custom\cmd.exe"
    assert env["MY_TOOL_HOME"] == r"D:\tool"
    assert "NOT_SET_ANYWHERE_123" not in env
