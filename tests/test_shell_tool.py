"""ExecTool against a real shell: bash, or PowerShell on Windows.

Tests that run a command give Windows a PowerShell variant; the few marked
``requires_posix_shell`` check something only a POSIX shell does (reasons in
docs/WINDOWS.md). The guard tests run everywhere. Windows output lines end in
CRLF, so outputs are compared after ``lf()``."""

import re
import sys
import time

import pytest

from conftest import WINDOWS, requires_posix_shell
from nanoclaw.agent.tools import is_tool_error
from nanoclaw.agent.tools.shell import ExecTool


@pytest.fixture
def tool(tmp_path):
    return ExecTool(working_dir=tmp_path)


# Since layer 16 every command runs as an exec session and ends with the
# session's elapsed time (nanobot exec_session.format_session_poll).
ELAPSED = r"\nElapsed: \d+\.\ds"
PWD = "(Get-Location).Path" if WINDOWS else "pwd"


def lf(text: str) -> str:
    """Windows line endings to LF (a no-op on POSIX)."""
    return text.replace("\r\n", "\n")


async def test_stdout_and_exit_code(tool):
    assert re.fullmatch("hello\n\nExit code: 0" + ELAPSED, lf(await tool.execute(command="echo hello")))


async def test_stderr_and_nonzero_exit_is_not_a_tool_error(tool):
    command = "[Console]::Error.WriteLine('oops'); exit 3" if WINDOWS else "echo oops >&2; exit 3"
    out = await tool.execute(command=command)
    assert not is_tool_error(out)
    assert re.fullmatch("STDERR:\noops\n\nExit code: 3" + ELAPSED, lf(out))


async def test_runs_in_workspace_and_working_dir(tool, tmp_path):
    (tmp_path / "sub").mkdir()
    assert (await tool.execute(command=PWD)).startswith(str(tmp_path))
    assert (await tool.execute(command=PWD, working_dir="sub")).startswith(str(tmp_path / "sub"))


async def test_missing_working_dir_is_an_error(tool):
    out = await tool.execute(command="pwd", working_dir="nope")
    assert is_tool_error(out) and out.startswith("Error executing command:")


@requires_posix_shell(
    "a `&` background job and exit code -9 (SIGKILL); the Windows tree kill is tested in "
    "test_shell_reap (real) and test_exec_platform (mocked)"
)
async def test_timeout_kills_the_command_and_its_children(tool, tmp_path):
    marker = tmp_path / "late.txt"
    start = time.monotonic()
    out = await tool.execute(command=f"(sleep 3; touch {marker}) & sleep 5", timeout=1, yield_time_ms=3000)
    assert is_tool_error(out)
    assert re.fullmatch(
        "Error: Command timed out; session was terminated. Reason: max_runtime.\nExit code: -9" + ELAPSED, out
    )
    assert time.monotonic() - start < 3
    time.sleep(2.5)
    assert not marker.exists()  # the background child was killed too


def test_timeout_resolution(tmp_path):
    # nanobot shell.py _resolve_timeout: the model's request is used as given
    # within 1..max_runtime_limit and rejected outside it, never silently capped.
    assert ExecTool()._resolve_timeout(None) == 3600  # default max_runtime: 1 hour
    assert ExecTool(timeout=60)._resolve_timeout(None) == 60
    assert ExecTool(timeout=60)._resolve_timeout(5) == 5
    assert ExecTool(timeout=60)._resolve_timeout(9999) == 9999
    assert ExecTool(max_runtime=0)._resolve_timeout(None) is None  # 0 = no limit
    assert ExecTool(timeout=0)._resolve_timeout(None) is None
    with pytest.raises(ValueError, match="between 1 and 86400 seconds"):
        ExecTool()._resolve_timeout(86401)
    with pytest.raises(ValueError, match="between 1 and 100 seconds"):
        ExecTool(max_runtime_limit=100)._resolve_timeout(0)
    with pytest.raises(ValueError, match="integer number of seconds"):
        ExecTool()._resolve_timeout(True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ExecTool(timeout=5, max_runtime=5)


async def test_timeout_and_max_runtime_together_is_an_error(tool):
    out = await tool.execute(command="echo hi", timeout=5, max_runtime=5)
    assert is_tool_error(out) and out == "Error: timeout and max_runtime are mutually exclusive."


async def test_runtime_over_the_limit_is_an_error(tool):
    out = await tool.execute(command="echo hi", max_runtime=86401)
    assert is_tool_error(out) and out == "Error: max_runtime must be between 1 and 86400 seconds"


async def test_zero_timeout_means_no_limit(tmp_path):
    command = "Start-Sleep -Milliseconds 200; echo done" if WINDOWS else "sleep 0.2; echo done"
    out = await ExecTool(working_dir=tmp_path, timeout=0).execute(command=command)
    assert re.fullmatch("done\n\nExit code: 0" + ELAPSED, lf(out))


@requires_posix_shell("the exact truncation counts assume LF line endings; Windows prints CRLF")
async def test_long_output_keeps_head_and_tail(tool):
    out = await tool.execute(command="python3 -c \"print('a' * 6000 + 'b' * 6000)\"")
    # nanobot: the first and last 5,000 characters, then a note of what was cut.
    assert out.startswith("a" * 5000 + "b" * 4999 + "\n\n(2,001 chars truncated from output)\nExit code: 0")
    assert re.search(ELAPSED + "$", out)


async def test_secrets_are_not_passed_to_commands(tool, monkeypatch):
    monkeypatch.setenv("NANOCLAW_API_KEY", "sk-secret")
    if WINDOWS:
        command = (
            'echo "key=[$env:NANOCLAW_API_KEY]"; '
            'Get-ChildItem env: | ForEach-Object { "$($_.Name)=$($_.Value)" }'
        )
    else:
        command = 'echo "key=[$NANOCLAW_API_KEY]"; env | sort'
    out = await tool.execute(command=command)
    assert "key=[]" in out
    assert "sk-secret" not in out
    assert ("systemroot=" if WINDOWS else "home=") in out.lower()


async def test_commands_can_still_find_programs(tool):
    # POSIX: no PATH is passed; bash supplies a default one. Windows passes PATH.
    command = "where.exe cmd" if WINDOWS else "ls / > /dev/null && which python3"
    assert "Exit code: 0" in await tool.execute(command=command)


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "rm -r build",
        "rm -r -f build",
        "cd x && RM -RF /",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=disk.img",
        "echo hi > /dev/sda",
        "sudo shutdown now",
        "reboot",
        ":(){ :|:& };:",
    ],
)
async def test_deny_list_is_always_applied(tool, command):
    out = await tool.execute(command=command)
    assert out == "Error: Command blocked by deny pattern filter"


@pytest.mark.parametrize("command", ["rm file.txt", "echo format", "git status --format=short", "ls -rf"])
async def test_deny_list_allows_ordinary_commands(tool, command):
    assert "blocked" not in await tool.execute(command=command)


@pytest.mark.parametrize(
    "command",
    [
        "Remove-Item -Recurse -Force junk",
        "cmd /c rd /s /q junk",
        # A quoted .exe path: PowerShell needs the `&` that _spawn adds.
        f"\"{sys.executable}\" -c \"import shutil; shutil.rmtree('junk')\"",
    ]
    if WINDOWS
    else [
        "rm --recursive --force junk",
        "find junk -delete",
        "python3 -c \"import shutil; shutil.rmtree('junk')\"",
    ],
)
async def test_deny_list_is_only_a_speed_bump(tool, tmp_path, command):
    """Documented limitation (same as nanobot): other spellings get through."""
    (tmp_path / "junk").mkdir()
    out = await tool.execute(command=command)
    assert "blocked" not in out
    assert not (tmp_path / "junk").exists()


async def test_extra_deny_patterns(tmp_path):
    tool = ExecTool(working_dir=tmp_path, deny_patterns=[r"\bcurl\b"])
    assert await tool.execute(command="curl example.com") == "Error: Command blocked by deny pattern filter"


async def test_missing_command(tool):
    assert await tool.execute(command="") == "Error: Missing command. Provide command or cmd."


@pytest.mark.parametrize(
    "command",
    [
        "echo x > memory/history.jsonl",
        "echo x >> memory/history.jsonl",
        "echo 5 > memory/.dream_cursor",
        "echo x | tee -a memory/history.jsonl",
        "cp other.jsonl memory/history.jsonl",
        "mv a b memory/.dream_cursor",
        "dd if=/dev/zero of=memory/history.jsonl",
        "sed -i 's/a/b/' memory/history.jsonl",
    ],
)
async def test_writes_to_memory_journal_are_blocked(tmp_path, command):
    result = await ExecTool(working_dir=str(tmp_path)).execute(command=command)
    assert result == "Error: Command blocked by deny pattern filter"


@pytest.mark.parametrize("command", ["cat memory/history.jsonl", "grep fact memory/history.jsonl", "wc -l memory/.dream_cursor"])
async def test_reading_the_memory_journal_is_allowed(tmp_path, command):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "history.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "memory" / ".dream_cursor").write_text("1", encoding="utf-8")
    result = await ExecTool(working_dir=str(tmp_path)).execute(command=command)
    assert "blocked" not in result
