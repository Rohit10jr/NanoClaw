"""Registration and exec delegation for ripgrep.

Ported from nanobot tests/tools/test_rg_tool.py (c4dac64). nanobot's loader
tests choose between ``rg`` and ``grep``/``find_files``; NanoClaw has neither
of the latter, so here ``rg`` is simply registered or not. Left out: the
loader's plugin and creation-failure cases (NanoClaw has no loader), the
``read_only`` check (no concurrency flags yet) and the bwrap/seatbelt case
(no sandbox until layer 18).
"""

import asyncio
import os
import re
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.subagent import SubagentManager
from nanoclaw.agent.tools import build_default_tools
from nanoclaw.agent.tools.exec_session import ExecSessionManager
from nanoclaw.agent.tools.rg import RgTool
from nanoclaw.agent.tools.shell import ExecTool
from nanoclaw.config.schema import ExecToolConfig
from nanoclaw.providers import FakeProvider
from nanoclaw.session import SessionManager
from nanoclaw.utils.tool_hints import format_tool_hints
from nanoclaw.providers import ToolCallRequest

requires_rg = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")


def _fake_which(installed: bool):
    def which(command, *args, **kwargs):
        return "/usr/bin/rg" if installed else None

    return which


@pytest.mark.parametrize("installed", [True, False])
@pytest.mark.parametrize("exec_enabled", [True, False])
def test_registers_rg_only_when_installed_and_exec_enabled(tmp_path, monkeypatch, installed, exec_enabled):
    monkeypatch.setattr("nanoclaw.agent.tools.rg.shutil.which", _fake_which(installed))
    registry = build_default_tools(tmp_path, exec_config=ExecToolConfig(enable=exec_enabled))
    assert ("rg" in registry.tool_names) == (installed and exec_enabled)
    assert ("exec" in registry.tool_names) == exec_enabled
    assert {"grep", "find_files"}.isdisjoint(registry.tool_names)


@pytest.mark.parametrize("installed", [True, False])
def test_main_agent_and_subagents_register_rg(tmp_path, monkeypatch, installed):
    monkeypatch.setattr("nanoclaw.agent.tools.rg.shutil.which", _fake_which(installed))
    ws = tmp_path / "ws"
    ws.mkdir()
    loop = AgentLoop(FakeProvider([]), ws, sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"))
    assert ("rg" in loop.tools.tool_names) == installed
    assert ("rg" in loop.subagents._build_tools().tool_names) == installed
    assert "rg" not in loop.memory.build_dream_tools().tool_names


def test_subagents_follow_exec_enable(tmp_path, monkeypatch):
    monkeypatch.setattr("nanoclaw.agent.tools.rg.shutil.which", _fake_which(True))
    manager = SubagentManager(tmp_path, exec_config=ExecToolConfig(enable=False))
    assert "rg" not in manager._build_tools().tool_names


def test_registered_rg_gets_the_exec_settings(tmp_path, monkeypatch):
    monkeypatch.setattr("nanoclaw.agent.tools.rg.shutil.which", _fake_which(True))
    cfg = ExecToolConfig(max_runtime=30, path_prepend="/opt/bin", deny_patterns=["secret"])
    registry = build_default_tools(tmp_path, restrict_to_workspace=True, exec_config=cfg)
    tool = registry.get("rg")
    assert isinstance(tool, RgTool)
    assert tool.restrict_to_workspace is True
    assert tool.max_runtime == 30
    assert tool.path_prepend == "/opt/bin"
    assert "secret" in tool.deny_patterns
    assert tool.working_dir == str(tmp_path)


def test_detection_uses_exec_path_configuration(tmp_path, monkeypatch):
    config = ExecToolConfig()
    config.path_prepend = str(tmp_path / "before")
    config.path_append = str(tmp_path / "after")
    captured = {}

    def which(command, *, path):
        captured.update(command=command, path=path)
        return "rg"

    monkeypatch.setattr("nanoclaw.agent.tools.rg.shutil.which", which)
    assert RgTool.enabled(config)
    assert captured["command"] == "rg"
    assert captured["path"] == os.pathsep.join([
        config.path_prepend, os.environ.get("PATH", ""), config.path_append,
    ])


async def test_rg_forwards_native_arguments_and_exec_options(monkeypatch):
    execute = AsyncMock(return_value="native output")
    monkeypatch.setattr(ExecTool, "execute", execute)
    arguments = ["--engine=pcre2", "--sort", "path", "--json", "(?<=hello)world", "."]
    result = await RgTool().execute(
        args=arguments, working_dir="project", timeout=12,
        yield_time_ms=25, max_output_chars=1500,
    )
    assert result == "native output"
    execute.assert_awaited_once_with(
        command=["rg", *arguments], working_dir="project", timeout=12,
        yield_time_ms=25, max_output_chars=1500,
    )


async def test_rg_drops_shell_options(monkeypatch):
    execute = AsyncMock(return_value="ok")
    monkeypatch.setattr(ExecTool, "execute", execute)
    await RgTool().execute(args=["x"], command="rm -rf /", cmd="id", shell="zsh", login=True)
    execute.assert_awaited_once_with(command=["rg", "x"])


async def test_empty_arguments_are_an_error():
    result = await RgTool().execute(args=[])
    assert result.is_error
    assert result == "Error: Provide rg arguments, such as --help or --files."


def test_schema_exposes_arguments_and_exec_controls():
    tool = RgTool()
    schema = tool.parameters
    assert schema["properties"]["args"]["type"] == "array"
    assert schema["properties"]["args"]["items"] == {"type": "string"}
    assert schema["required"] == ["args"]
    assert {"working_dir", "timeout", "yield_time_ms"} <= schema["properties"].keys()
    assert {"command", "cmd", "shell", "login"}.isdisjoint(schema["properties"])
    assert "command" in ExecTool().parameters["properties"]
    assert tool.name == "rg"
    assert tool.description == (
        "Search file contents and discover files with ripgrep. "
        "Accepts native rg arguments as an array of strings."
    )
    assert tool.validate_params({"args": []})  # minItems: 1
    assert not tool.validate_params({"args": ["-n", "x"]})


def test_tool_hint():
    hint = format_tool_hints([ToolCallRequest("c1", "rg", {"args": ["-n", "needle", "src"]})])
    assert hint.startswith("rg ")
    assert "needle" in hint


@requires_rg
async def test_native_search_and_file_discovery_via_exec(tmp_path):
    (tmp_path / "source.py").write_text("before\nneedle\nafter\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("unrelated\n", encoding="utf-8")
    tool = RgTool(working_dir=str(tmp_path), restrict_to_workspace=True)
    result = await tool.execute(args=["-n", "-C1", "-g", "*.py", "needle", "."])
    assert "2:needle" in result
    assert "before" in result and "after" in result
    assert "Exit code: 0" in result
    files = await tool.execute(args=["--files", "-g", "*.py", "."])
    assert "source.py" in files
    absent = await tool.execute(args=["absent", "."])
    assert "Exit code: 1" in absent


async def test_exec_workspace_policy_applies(tmp_path):
    result = await RgTool(working_dir=str(tmp_path), restrict_to_workspace=True).execute(
        args=["--files"], working_dir=str(tmp_path.parent),
    )
    assert "outside the configured workspace" in result


@requires_rg
async def test_native_arguments_preserve_spaces_and_shell_characters(tmp_path):
    text = 'hello world | $HOME > output; "quoted" & $(echo hi)'
    (tmp_path / "space name.txt").write_text(text + "\n", encoding="utf-8")
    tool = RgTool(working_dir=str(tmp_path))
    result = await tool.execute(args=["-n", "-F", "--", text, "space name.txt"])
    assert f"1:{text}" in result
    assert "Exit code: 0" in result


@requires_rg
async def test_native_search_session(tmp_path):
    (tmp_path / "source.txt").write_text("needle\n", encoding="utf-8")
    manager = ExecSessionManager()
    tool = RgTool(working_dir=str(tmp_path), exec_session_manager=manager)
    try:
        result = await tool.execute(args=["-n", "needle", "."], yield_time_ms=0)
        session = re.search(r"session_id:\s*([0-9a-f]+)", result)
        if session:
            poll = await manager.write(
                session_id=session.group(1), chars=None, close_stdin=False, terminate=False,
                yield_time_ms=10000, max_output_chars=1000,
            )
            result += poll.output
            assert poll.exit_code == 0
        assert "1:needle" in result
    finally:
        await manager.close_all()


async def test_direct_process_preserves_arguments_and_output_limit(tmp_path):
    tool = ExecTool(working_dir=str(tmp_path))
    result = await tool.execute(
        command=[sys.executable, "-c", "print('x' * 4000)"], max_output_chars=1000,
    )
    assert "chars truncated" in result
    assert "Exit code: 0" in result


async def test_direct_process_timeout(tmp_path):
    result = await ExecTool(working_dir=str(tmp_path)).execute(
        command=[sys.executable, "-c", "import time; time.sleep(20)"], timeout=1,
        yield_time_ms=5000,
    )
    assert "Reason: max_runtime" in result
    assert "Exit code:" in result


async def test_direct_process_cancellation_reaps_child(tmp_path, monkeypatch):
    spawned = asyncio.Event()
    processes = []
    original_spawn = ExecTool._spawn

    async def capture(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        processes.append(process)
        spawned.set()
        return process

    monkeypatch.setattr(ExecTool, "_spawn", staticmethod(capture))
    task = asyncio.create_task(ExecTool(working_dir=str(tmp_path)).execute(
        command=[sys.executable, "-c", "import time; time.sleep(20)"],
    ))
    await asyncio.wait_for(spawned.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert isinstance(processes[0].returncode, int)


@requires_rg
async def test_direct_search_uses_configured_path(tmp_path, monkeypatch):
    installed = shutil.which("rg")
    assert installed
    # Keep installed launchers with their adjacent files (e.g. Scoop's .shim).
    bin_dir = os.path.dirname(installed)
    monkeypatch.setenv("PATH", "")
    result = await RgTool(working_dir=str(tmp_path), path_prepend=bin_dir).execute(
        args=["--version"],
    )
    assert "ripgrep" in result
    assert "Exit code: 0" in result


async def test_missing_executable_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    result = await RgTool(working_dir=str(tmp_path)).execute(args=["--version"])
    assert result.is_error
    assert "rg" in result


def test_guard_runs_only_in_restricted_mode(tmp_path):
    # rg runs no shell, so unrestricted it skips the guard, as in nanobot
    # (exec runs it on every command): searching for deny-listed words or an
    # internal URL is allowed.
    tool = RgTool(working_dir=str(tmp_path))
    for args in (["rm -rf", "."], ["x; shutdown", "."], ["http://169.254.169.254/latest", "."]):
        prepared = tool._prepare_command(["rg", *args])
        assert not isinstance(prepared, str), args
        assert prepared.command == ["rg", *args]
    prepared = tool._prepare_command(["rg", "-F", "hello | $HOME", "."])
    assert not isinstance(prepared, str)
    assert prepared.command == ["rg", "-F", "hello | $HOME", "."]
    assert Path(prepared.cwd) == tmp_path

    # Restricted: the full guard on shlex.join(["rg", *args]), as nanobot.
    restricted = RgTool(working_dir=str(tmp_path), restrict_to_workspace=True)
    blocked = restricted._prepare_command(["rg", "x; shutdown", "."])
    assert isinstance(blocked, str) and "blocked" in blocked.lower()
    outside = restricted._prepare_command(["rg", "x", "/etc"])
    assert isinstance(outside, str) and "blocked" in outside.lower()


@pytest.mark.asyncio
async def test_search_for_deny_listed_text_runs_unrestricted(tmp_path):
    if shutil.which("rg") is None:
        pytest.skip("ripgrep is not installed")
    (tmp_path / "notes.txt").write_text("never run rm -rf here\n", encoding="utf-8")
    result = await RgTool(working_dir=str(tmp_path)).execute(args=["-F", "rm -rf", "."])
    assert "blocked" not in result
    assert "never run rm -rf here" in result
