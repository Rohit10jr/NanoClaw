"""ExecTool against real bash."""

import time

import pytest

from nanoclaw.agent.tools import is_tool_error
from nanoclaw.agent.tools.shell import ExecTool


@pytest.fixture
def tool(tmp_path):
    return ExecTool(working_dir=tmp_path)


async def test_stdout_and_exit_code(tool):
    assert await tool.execute(command="echo hello") == "hello\n\n\nExit code: 0"


async def test_stderr_and_nonzero_exit_is_not_a_tool_error(tool):
    out = await tool.execute(command="echo oops >&2; exit 3")
    assert not is_tool_error(out)
    assert out == "STDERR:\noops\n\n\nExit code: 3"


async def test_runs_in_workspace_and_working_dir(tool, tmp_path):
    (tmp_path / "sub").mkdir()
    assert (await tool.execute(command="pwd")).startswith(str(tmp_path))
    assert (await tool.execute(command="pwd", working_dir="sub")).startswith(str(tmp_path / "sub"))


async def test_missing_working_dir_is_an_error(tool):
    out = await tool.execute(command="pwd", working_dir="nope")
    assert is_tool_error(out) and out.startswith("Error executing command:")


async def test_timeout_kills_the_command_and_its_children(tool, tmp_path):
    marker = tmp_path / "late.txt"
    start = time.monotonic()
    out = await tool.execute(command=f"(sleep 3; touch {marker}) & sleep 5", timeout=1)
    assert out == "Error: Command timed out after 1 seconds"
    assert time.monotonic() - start < 3
    time.sleep(2.5)
    assert not marker.exists()  # the background child was killed too


def test_timeout_resolution(tmp_path):
    assert ExecTool(timeout=60)._resolve_timeout(None) == 60
    assert ExecTool(timeout=60)._resolve_timeout(5) == 5
    assert ExecTool(timeout=60)._resolve_timeout(9999) == 600  # model's choice is capped
    assert ExecTool(timeout=1200)._resolve_timeout(None) == 1200  # configured default may exceed the cap
    assert ExecTool(timeout=0)._resolve_timeout(None) is None  # 0 = no limit


async def test_zero_timeout_means_no_limit(tmp_path):
    out = await ExecTool(working_dir=tmp_path, timeout=0).execute(command="sleep 0.2; echo done")
    assert out == "done\n\n\nExit code: 0"


async def test_long_output_keeps_head_and_tail(tool):
    out = await tool.execute(command="python3 -c \"print('a' * 6000 + 'b' * 6000)\"")
    assert out.startswith("a" * 5000)
    assert "chars truncated) ..." in out
    assert out.endswith("\n\nExit code: 0")


async def test_secrets_are_not_passed_to_commands(tool, monkeypatch):
    monkeypatch.setenv("NANOCLAW_API_KEY", "sk-secret")
    out = await tool.execute(command='echo "key=[$NANOCLAW_API_KEY]"; env | sort')
    assert "key=[]" in out
    assert "sk-secret" not in out
    assert "HOME=" in out


async def test_commands_can_still_find_programs(tool):
    # No PATH is passed; bash supplies a default one.
    assert "Exit code: 0" in await tool.execute(command="ls / > /dev/null && which python3")


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
    assert await tool.execute(command="") == "Error: Missing command. Provide command."


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
    (tmp_path / "memory" / "history.jsonl").write_text("{}\n")
    (tmp_path / "memory" / ".dream_cursor").write_text("1")
    result = await ExecTool(working_dir=str(tmp_path)).execute(command=command)
    assert "blocked" not in result
