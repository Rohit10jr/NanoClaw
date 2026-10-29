"""Tests for cross-platform shell execution.

Ported from nanobot tests/tools/test_exec_platform.py (c4dac64), task 16.4.
Verifies that ExecTool selects the correct shell, environment and path-append
strategy per platform, without actually running platform-specific binaries
(all subprocess calls are mocked), so the Windows parts run on Linux too.
nanobot's TestSandboxPlatform is left out: NanoClaw has no OS sandbox yet
(layer 18). The tests after TestWindowsRealExec are NanoClaw's own (taskkill,
Job-object ownership, the job-creation fallback).
"""

import asyncio
import os
import re
import shutil
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanoclaw.agent.tools.exec_session import ExecSessionManager, ExecSessionTool
from nanoclaw.agent.tools.shell import ExecTool

_WINDOWS_ENV_KEYS = {
    "APPDATA", "LOCALAPPDATA", "ProgramData",
    "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432",
}


class _CompletedProcess:
    """A finished subprocess with the stream/wait protocol used by the collector."""

    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> None:
        self.pid = 12345
        self.returncode = returncode
        self.stdin = None
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stdout.feed_eof()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()

    async def wait(self) -> int:
        await asyncio.sleep(0)
        return self.returncode


class _FakeWindowsJob:
    creation_flags = 0

    def assign_and_resume(self, pid: int) -> None:
        pass

    def release(self) -> None:
        pass

    def terminate(self) -> None:
        pass


# ---------------------------------------------------------------------------
# _build_env
# ---------------------------------------------------------------------------

class TestBuildEnvUnix:

    def test_expected_keys(self):
        with patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False):
            env = ExecTool()._build_env()
        expected = {"HOME", "LANG", "TERM", "PYTHONUNBUFFERED"}
        assert expected <= set(env)
        if sys.platform != "win32":
            assert set(env) == expected

    def test_home_from_environ(self, monkeypatch):
        monkeypatch.setenv("HOME", "/Users/dev")
        with patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False):
            env = ExecTool()._build_env()
        assert env["HOME"] == "/Users/dev"

    def test_secrets_excluded(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("NANOCLAW_TOKEN", "tok-secret")
        with patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False):
            env = ExecTool()._build_env()
        assert "OPENAI_API_KEY" not in env
        assert "NANOCLAW_TOKEN" not in env
        for v in env.values():
            assert "secret" not in v.lower()


class TestBuildEnvWindows:

    _EXPECTED_KEYS = {
        "SYSTEMROOT", "COMSPEC", "USERPROFILE", "HOMEDRIVE",
        "HOMEPATH", "TEMP", "TMP", "PATHEXT", "PATH", "PYTHONUNBUFFERED",
        *_WINDOWS_ENV_KEYS,
    }

    def test_expected_keys(self):
        with patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True):
            env = ExecTool()._build_env()
        assert set(env) == self._EXPECTED_KEYS

    def test_secrets_excluded(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("NANOCLAW_TOKEN", "tok-secret")
        with patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True):
            env = ExecTool()._build_env()
        assert "OPENAI_API_KEY" not in env
        assert "NANOCLAW_TOKEN" not in env
        for v in env.values():
            assert "secret" not in v.lower()

    def test_path_has_sensible_default(self):
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch.dict("os.environ", {}, clear=True),
        ):
            env = ExecTool()._build_env()
        assert "system32" in env["PATH"].lower()

    def test_systemroot_forwarded(self, monkeypatch):
        monkeypatch.setenv("SYSTEMROOT", r"D:\Windows")
        with patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True):
            env = ExecTool()._build_env()
        assert env["SYSTEMROOT"] == r"D:\Windows"


# ---------------------------------------------------------------------------
# argument-vector PATH
# ---------------------------------------------------------------------------

class TestArgumentVectorPath:

    def test_uses_parent_path_for_executable_lookup(self):
        parent_path = os.pathsep.join(("parent-bin", "system-bin"))
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False),
            patch.dict(
                "os.environ",
                {"PATH": parent_path, "NANOCLAW_SECRET_TOKEN": "super-secret-value"},
                clear=True,
            ),
        ):
            prepared = ExecTool()._prepare_command(["rg", "--version"])

        assert not isinstance(prepared, str)
        assert prepared.env["PATH"] == parent_path
        assert "NANOCLAW_SECRET_TOKEN" not in prepared.env


# ---------------------------------------------------------------------------
# _spawn
# ---------------------------------------------------------------------------

class TestSpawnUnix:

    @pytest.mark.asyncio
    async def test_uses_bash(self):
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = AsyncMock()
            await ExecTool._spawn("echo hi", "/tmp", {"HOME": "/tmp"})

        args = mock_exec.call_args[0]
        assert "bash" in args[0]
        assert "-l" not in args
        assert "-c" in args
        assert "echo hi" in args

        kwargs = mock_exec.call_args[1]
        assert kwargs["stdin"] == asyncio.subprocess.DEVNULL

    @pytest.mark.asyncio
    async def test_process_tree_starts_new_session(self):
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = AsyncMock()
            await ExecTool._spawn(
                "echo hi",
                "/tmp",
                {"HOME": "/tmp"},
                process_tree=True,
            )

        assert mock_exec.call_args.kwargs["start_new_session"] is True


class TestSpawnWindows:

    @pytest.mark.asyncio
    async def test_job_assignment_failure_kills_suspended_process(self):
        env = {"PATH": ""}
        process = AsyncMock()
        process.pid = 123
        process.returncode = None
        process.kill = MagicMock()
        process.wait.return_value = -9
        job = MagicMock(spec=_FakeWindowsJob)
        job.creation_flags = 0x4
        job.assign_and_resume.side_effect = OSError("OpenProcess failed")

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("nanoclaw.agent.tools.shell.sys", MagicMock(platform="win32")),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch.object(ExecTool, "_create_windows_job", return_value=job),
            pytest.raises(OSError, match="OpenProcess failed"),
        ):
            mock_exec.return_value = process
            await ExecTool._spawn("echo hi", r"C:\work", env, process_tree=True)

        job.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        process.wait.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_single_line_uses_powershell(self):
        """Single-line commands on Windows now route through PowerShell."""
        env = {"COMSPEC": r"C:\Windows\system32\cmd.exe", "PATH": ""}
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = AsyncMock()
            await ExecTool._spawn("dir", r"C:\work", env)

        args = mock_exec.call_args[0]
        assert any(shell in args[0].lower() for shell in ("pwsh", "powershell"))
        assert "-NoProfile" in args
        assert "-NonInteractive" in args
        assert "-Command" in args
        assert "dir" in args[-1]

        kwargs = mock_exec.call_args[1]
        assert kwargs["stdin"] == asyncio.subprocess.DEVNULL

    @pytest.mark.asyncio
    async def test_single_line_passes_cwd_and_env(self):
        """PowerShell should receive cwd and env from the caller."""
        env = {"PATH": "/usr/bin"}
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = AsyncMock()
            await ExecTool._spawn("echo hi", r"C:\work", env)

        kwargs = mock_exec.call_args[1]
        assert kwargs["cwd"] == r"C:\work"
        assert kwargs["env"] == env

    @pytest.mark.asyncio
    async def test_multiline_uses_powershell(self):
        env = {"PATH": ""}
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = AsyncMock()
            await ExecTool._spawn('python -c "print(1)\nprint(2)"', r"C:\work", env)

        args = mock_exec.call_args[0]
        assert any(shell in args[0].lower() for shell in ("pwsh", "powershell"))
        assert "-NoProfile" in args
        assert "-NonInteractive" in args
        assert "-Command" in args
        assert "print(1)" in args[-1]
        assert "print(2)" in args[-1]

        kwargs = mock_exec.call_args[1]
        assert kwargs["cwd"] == r"C:\work"
        assert kwargs["env"] == env

    @pytest.mark.asyncio
    async def test_explicit_cmd_shell_uses_raw_shell_string(self):
        """Explicit shell='cmd' should preserve raw cmd.exe quoting semantics."""
        env = {"COMSPEC": r"C:\Windows\system32\cmd.exe", "PATH": ""}
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("asyncio.create_subprocess_shell", new_callable=AsyncMock) as mock_shell,
        ):
            mock_shell.return_value = AsyncMock()
            await ExecTool._spawn(
                'echo "a & b"', r"C:\work", env,
                shell_program=r"C:\Windows\system32\cmd.exe",
            )

        args = mock_shell.call_args[0]
        assert args == ('echo "a & b"',)
        kwargs = mock_shell.call_args[1]
        assert kwargs["cwd"] == r"C:\work"
        assert kwargs["env"] == {
            "COMSPEC": r"C:\Windows\system32\cmd.exe",
            "PATH": "",
        }

    @pytest.mark.asyncio
    async def test_powershell_preserves_last_native_exit_code(self):
        """PowerShell -Command should forward native process exit codes."""
        env = {"PATH": ""}
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = AsyncMock()
            await ExecTool._spawn("cmd /c exit 7", r"C:\work", env)

        command = mock_exec.call_args[0][-1]
        assert "cmd /c exit 7" in command
        assert "if ($LASTEXITCODE -ne $null) { exit $LASTEXITCODE }" in command

    @pytest.mark.asyncio
    async def test_powershell_configures_utf8_io(self):
        """PowerShell should use UTF-8 for captured output, native input, and redirections."""
        env = {"PATH": ""}
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = AsyncMock()
            await ExecTool._spawn("Write-Output 'café 你好'", r"C:\work", env)

        command = mock_exec.call_args[0][-1]
        assert "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)" in command
        assert (
            "if ($PSVersionTable.PSVersion.Major -lt 6) { "
            "$OutputEncoding = [Console]::OutputEncoding }"
        ) in command
        assert "$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'" in command

    @pytest.mark.asyncio
    async def test_powershell_invokes_quoted_windows_executable_path(self):
        """PowerShell needs & before quoted executable paths with arguments."""
        env = {"PATH": ""}
        command = r'"D:\Program Files\Python\python.exe" -u -c "print(1)"'
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = AsyncMock()
            await ExecTool._spawn(command, r"C:\work", env)

        powershell_command = mock_exec.call_args[0][-1]
        assert f"\n& {command}\n" in powershell_command

    @pytest.mark.asyncio
    async def test_powershell_invokes_quoted_windows_executable_without_arguments(self):
        """A quoted executable path is still a command when it has no arguments."""
        env = {"PATH": ""}
        command = r'"D:\Program Files\Git\cmd\git.exe"'
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = AsyncMock()
            await ExecTool._spawn(command, r"C:\work", env)

        powershell_command = mock_exec.call_args[0][-1]
        assert f"\n& {command}\n" in powershell_command

    @pytest.mark.asyncio
    async def test_prefers_pwsh_when_available(self):
        env = {"PATH": ""}

        def fake_which(command):
            if command == "pwsh":
                return r"C:\Program Files\PowerShell\7\pwsh.exe"
            if command == "powershell":
                return r"C:\Windows\system32\WindowsPowerShell\v1.0\powershell.exe"
            return None

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("nanoclaw.agent.tools.shell.shutil.which", side_effect=fake_which),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = AsyncMock()
            await ExecTool._spawn("dir", r"C:\work", env)

        args = mock_exec.call_args[0]
        assert "pwsh" in args[0].lower()


# ---------------------------------------------------------------------------
# path_append
# ---------------------------------------------------------------------------

class TestPathAppendPlatform:

    @pytest.mark.asyncio
    async def test_unix_uses_env_var_in_fixed_export(self):
        """On Unix, path_append must not be interpolated into shell source."""
        mock_proc = _CompletedProcess(b"ok")

        captured_cmd = None
        captured_env = {}

        async def capture_spawn(
            cmd, cwd, env, shell_program=None, login=True, *, stdin=None, process_tree=False,
        ):
            nonlocal captured_cmd
            captured_cmd = cmd
            captured_env.update(env)
            return mock_proc

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False),
            patch("nanoclaw.agent.tools.shell.os.pathsep", ":"),
            patch.object(ExecTool, "_spawn", side_effect=capture_spawn),
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            tool = ExecTool(path_append="/opt/bin; echo INJECTED")
            await tool.execute(command="ls")

        assert captured_cmd == 'export PATH="$PATH:$NANOBOT_PATH_APPEND"; ls'
        assert captured_env["NANOBOT_PATH_APPEND"] == "/opt/bin; echo INJECTED"
        assert "INJECTED" not in captured_cmd

    @pytest.mark.asyncio
    async def test_unix_path_prepend_uses_env_var_in_fixed_export(self):
        """On Unix, path_prepend must not be interpolated into shell source."""
        mock_proc = _CompletedProcess(b"ok")

        captured_cmd = None
        captured_env = {}

        async def capture_spawn(
            cmd, cwd, env, shell_program=None, login=True, *, stdin=None, process_tree=False,
        ):
            nonlocal captured_cmd
            captured_cmd = cmd
            captured_env.update(env)
            return mock_proc

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False),
            patch("nanoclaw.agent.tools.shell.os.pathsep", ":"),
            patch.object(ExecTool, "_spawn", side_effect=capture_spawn),
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            tool = ExecTool(path_prepend="/venv/bin; echo INJECTED")
            await tool.execute(command="python --version")

        assert captured_cmd == 'export PATH="$NANOBOT_PATH_PREPEND:$PATH"; python --version'
        assert captured_env["NANOBOT_PATH_PREPEND"] == "/venv/bin; echo INJECTED"
        assert "INJECTED" not in captured_cmd

    @pytest.mark.asyncio
    async def test_unix_path_prepend_and_append_order(self):
        mock_proc = _CompletedProcess(b"ok")

        captured_cmd = None
        captured_env = {}

        async def capture_spawn(
            cmd, cwd, env, shell_program=None, login=True, *, stdin=None, process_tree=False,
        ):
            nonlocal captured_cmd
            captured_cmd = cmd
            captured_env.update(env)
            return mock_proc

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False),
            patch("nanoclaw.agent.tools.shell.os.pathsep", ":"),
            patch.object(ExecTool, "_spawn", side_effect=capture_spawn),
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            tool = ExecTool(path_prepend="/venv/bin", path_append="/usr/sbin")
            await tool.execute(command="python --version")

        assert captured_cmd == (
            'export PATH="$NANOBOT_PATH_PREPEND:$PATH:$NANOBOT_PATH_APPEND"; python --version'
        )
        assert captured_env["NANOBOT_PATH_PREPEND"] == "/venv/bin"
        assert captured_env["NANOBOT_PATH_APPEND"] == "/usr/sbin"

    @pytest.mark.asyncio
    async def test_windows_modifies_env(self):
        """On Windows, path_append is appended to PATH in the env dict."""
        mock_proc = _CompletedProcess(b"ok")

        captured_env = {}

        async def capture_spawn(
            cmd, cwd, env, shell_program=None, login=True, *, stdin=None, process_tree=False,
        ):
            captured_env.update(env)
            return mock_proc

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("nanoclaw.agent.tools.shell.os.pathsep", ";"),
            patch.object(ExecTool, "_spawn", side_effect=capture_spawn),
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            tool = ExecTool(path_append=r"C:\tools\bin")
            await tool.execute(command="dir")

        assert captured_env["PATH"].endswith(r";C:\tools\bin")

    @pytest.mark.asyncio
    async def test_windows_path_prepend_and_append_order(self):
        mock_proc = _CompletedProcess(b"ok")

        captured_env = {}

        async def capture_spawn(
            cmd, cwd, env, shell_program=None, login=True, *, stdin=None, process_tree=False,
        ):
            captured_env.update(env)
            return mock_proc

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("nanoclaw.agent.tools.shell.os.pathsep", ";"),
            patch.object(ExecTool, "_build_env", return_value={"PATH": r"C:\Windows\System32"}),
            patch.object(ExecTool, "_spawn", side_effect=capture_spawn),
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            tool = ExecTool(path_prepend=r"C:\venv\Scripts", path_append=r"C:\tools\bin")
            await tool.execute(command="python --version")

        assert captured_env["PATH"] == (
            r"C:\venv\Scripts;C:\Windows\System32;C:\tools\bin"
        )


# ---------------------------------------------------------------------------
# end-to-end (mocked subprocess, full execute path)
# ---------------------------------------------------------------------------

class TestExecuteEndToEnd:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "yield_time_ms, stdin",
        [(None, asyncio.subprocess.DEVNULL), (1000, asyncio.subprocess.PIPE)],
    )
    async def test_windows_full_path(self, yield_time_ms, stdin):
        """Full execute() flow on Windows: env, spawn, output formatting."""
        mock_proc = _CompletedProcess(b"hello world\r\n")

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch.object(ExecTool, "_spawn", return_value=mock_proc) as mock_spawn,
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            tool = ExecTool()
            result = await tool.execute(command="echo hello world", yield_time_ms=yield_time_ms)

        assert "hello world" in result
        assert "Exit code: 0" in result
        assert "session_id:" not in result
        assert mock_spawn.call_args.kwargs.get("stdin", asyncio.subprocess.DEVNULL) == stdin
        assert mock_spawn.call_args.kwargs["process_tree"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "yield_time_ms, stdin",
        [(None, asyncio.subprocess.DEVNULL), (1000, asyncio.subprocess.PIPE)],
    )
    async def test_unix_full_path(self, yield_time_ms, stdin):
        """Full execute() flow on Unix: env, spawn, output formatting."""
        mock_proc = _CompletedProcess(b"hello world\n")

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False),
            patch.object(ExecTool, "_spawn", return_value=mock_proc) as mock_spawn,
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            tool = ExecTool()
            result = await tool.execute(command="echo hello world", yield_time_ms=yield_time_ms)

        assert "hello world" in result
        assert "Exit code: 0" in result
        assert "session_id:" not in result
        assert mock_spawn.call_args.kwargs.get("stdin", asyncio.subprocess.DEVNULL) == stdin
        assert mock_spawn.call_args.kwargs["process_tree"] is True

    @pytest.mark.asyncio
    async def test_execute_defaults_to_non_login_shell(self):
        """The public execute path must not silently request a login shell."""
        captured_login = []

        async def capture_spawn(
            cmd, cwd, env, shell_program=None, login=None, *, stdin=None, process_tree=False,
        ):
            captured_login.append(login)
            return _CompletedProcess(b"ok\n")

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False),
            patch.object(ExecTool, "_spawn", side_effect=capture_spawn),
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            tool = ExecTool()
            await tool.execute(command="echo ok")
            await tool.execute(command="echo ok", login=True)

        assert captured_login == [False, True]


# ---------------------------------------------------------------------------
# _extract_absolute_paths - UNC path support
# ---------------------------------------------------------------------------

class TestExtractAbsolutePaths:
    """Tests for Windows UNC path extraction in shell commands."""

    def test_windows_drive_path(self):
        """Test extraction of standard Windows drive paths."""
        cmd = r"dir C:\Users\Public"
        paths = ExecTool._extract_absolute_paths(cmd)
        assert r"C:\Users\Public" in paths

    def test_windows_drive_path_root(self):
        """Test extraction of Windows drive root paths."""
        cmd = r"dir C:\temp"
        paths = ExecTool._extract_absolute_paths(cmd)
        assert any("C:\\" in p for p in paths)

    def test_unc_path_simple(self):
        """Test extraction of simple UNC paths."""
        cmd = r"dir \\server\share"
        paths = ExecTool._extract_absolute_paths(cmd)
        assert r"\\server\share" in paths

    def test_unc_path_with_subdirs(self):
        """Test extraction of UNC paths with subdirectories."""
        cmd = r"copy \\server\share\folder\file.txt D:\backup"
        paths = ExecTool._extract_absolute_paths(cmd)
        assert r"\\server\share\folder\file.txt" in paths
        assert r"D:\backup" in paths

    def test_unc_path_in_quotes(self):
        """Test extraction of UNC paths enclosed in quotes."""
        cmd = r'type "\\server\share\docs\readme.txt"'
        paths = ExecTool._extract_absolute_paths(cmd)
        assert r"\\server\share\docs\readme.txt" in paths

    def test_mixed_paths(self):
        """Test extraction of mixed UNC, drive, and POSIX paths."""
        cmd = r'copy \\server\data\file.txt C:\local\temp && ls /tmp'
        paths = ExecTool._extract_absolute_paths(cmd)
        assert r"\\server\data\file.txt" in paths
        assert any("C:\\" in p for p in paths)
        assert "/tmp" in paths

    def test_home_path(self):
        """Test extraction of home directory shortcuts."""
        cmd = "cat ~/config.txt"
        paths = ExecTool._extract_absolute_paths(cmd)
        assert "~/config.txt" in paths

    def test_no_paths(self):
        """Test command with no absolute paths."""
        cmd = "echo hello"
        paths = ExecTool._extract_absolute_paths(cmd)
        assert paths == []


# ---------------------------------------------------------------------------
# Windows multi-line command PowerShell fallback
# ---------------------------------------------------------------------------

class TestWindowsMultilineExec:
    """Verify commands on Windows route through PowerShell (now the default)."""

    @pytest.mark.asyncio
    async def test_multiline_python_uses_powershell(self):
        mock_proc = _CompletedProcess(b"1\n2\n")

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch.object(ExecTool, "_create_windows_job", side_effect=_FakeWindowsJob),
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            mock_exec.return_value = mock_proc
            tool = ExecTool()
            result = await tool.execute(command='python -c "print(1)\nprint(2)"')

        assert "1" in result
        assert "2" in result
        assert "Exit code: 0" in result
        args = mock_exec.call_args[0]
        assert any(shell in args[0].lower() for shell in ("pwsh", "powershell"))

    @pytest.mark.asyncio
    async def test_multiline_node_uses_powershell(self):
        mock_proc = _CompletedProcess(b"1\n")

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch.object(ExecTool, "_create_windows_job", side_effect=_FakeWindowsJob),
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            mock_exec.return_value = mock_proc
            tool = ExecTool()
            result = await tool.execute(command='node -e "console.log(1)\nconsole.log(2)"')

        assert "1" in result
        args = mock_exec.call_args[0]
        assert any(shell in args[0].lower() for shell in ("pwsh", "powershell"))

    @pytest.mark.asyncio
    async def test_single_line_uses_powershell(self):
        """Single-line commands also route through PowerShell now."""
        mock_proc = _CompletedProcess(b"1\n")

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch.object(ExecTool, "_spawn", return_value=mock_proc) as mock_spawn,
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            tool = ExecTool()
            result = await tool.execute(command='python -c "print(1)"')

        assert "1" in result
        mock_spawn.assert_called_once()

    @pytest.mark.asyncio
    async def test_unix_unchanged(self):
        mock_proc = _CompletedProcess(b"1\n2\n")

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False),
            patch.object(ExecTool, "_spawn", return_value=mock_proc) as mock_spawn,
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            tool = ExecTool()
            result = await tool.execute(command='python -c "print(1)\nprint(2)"')

        assert "1" in result
        mock_spawn.assert_called_once()


# ---------------------------------------------------------------------------
# _resolve_shell — Windows support
# ---------------------------------------------------------------------------

class TestResolveShellWindows:
    """shell parameter is now accepted on Windows."""

    @pytest.mark.asyncio
    async def test_shell_powershell_accepted(self):
        """shell='powershell' should resolve and route through PowerShell."""
        mock_proc = _CompletedProcess(b"hello\n")

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch.object(ExecTool, "_create_windows_job", side_effect=_FakeWindowsJob),
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            mock_exec.return_value = mock_proc
            tool = ExecTool()
            result = await tool.execute(command="echo hello", shell="powershell")

        assert "hello" in result
        args = mock_exec.call_args[0]
        assert "powershell" in args[0].lower()
        assert "-NonInteractive" in args

    @pytest.mark.asyncio
    async def test_shell_cmd_accepted(self):
        """shell='cmd' should preserve the command string for cmd.exe parsing."""
        mock_proc = _CompletedProcess(b'"a & b"\n')

        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("asyncio.create_subprocess_shell", new_callable=AsyncMock) as mock_shell,
            patch.object(ExecTool, "_create_windows_job", side_effect=_FakeWindowsJob),
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            mock_shell.return_value = mock_proc
            tool = ExecTool()
            result = await tool.execute(command='echo "a & b"', shell="cmd")

        assert '"a & b"' in result
        args = mock_shell.call_args[0]
        assert args == ('echo "a & b"',)

    @pytest.mark.asyncio
    async def test_shell_bash_rejected_on_windows(self):
        """shell='bash' should still be rejected on Windows."""
        with patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True):
            tool = ExecTool()
            result = await tool.execute(command="echo hello", shell="bash")

        assert "Error: unsupported shell" in result
        assert "Allowed: powershell, pwsh, cmd" in result


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="requires Windows",
)
class TestWindowsRealExec:

    async def _execute(self, command, shell=None, **kwargs):
        manager = ExecSessionManager()
        try:
            initial = await ExecTool(exec_session_manager=manager, **kwargs).execute(command=command, shell=shell)
            match = re.search(r"session_id:\s*([0-9a-f]+)", initial)
            if match is None:
                return initial
            final = await ExecSessionTool(manager=manager).execute(
                session_id=match.group(1), until_exit=True,
            )
            return initial + "\n" + final
        finally:
            await manager.close_all()

    @pytest.mark.skipif(shutil.which("pwsh") is None, reason="requires PowerShell 7")
    @pytest.mark.asyncio
    async def test_single_line_and_separator_uses_pwsh(self):
        result = await self._execute(command="echo before && echo after", timeout=10)

        assert "before" in result
        assert "after" in result
        assert "Exit code: 0" in result

    @pytest.mark.asyncio
    async def test_explicit_cmd_preserves_embedded_quotes(self):
        result = await self._execute(command='echo "a & b"', shell="cmd", timeout=10)

        assert '"a & b"' in result
        assert r'\"a & b\"' not in result
        assert "Exit code: 0" in result

    @pytest.mark.asyncio
    async def test_default_powershell_preserves_native_exit_code(self):
        result = await self._execute(command="cmd /c exit 7", timeout=10)

        assert "Exit code: 7" in result

    @pytest.mark.asyncio
    async def test_windows_powershell_output_and_redirection_are_utf8(self, tmp_path):
        result = await self._execute(
            working_dir=str(tmp_path), timeout=180,
            command=(
                "Write-Output 'file café λ 你好' > marker.txt; "
                "Write-Output 'café λ 你好'; "
                "[Console]::Error.WriteLine('warn λ 你好')"
            ),
            shell="powershell",
        )

        assert "café λ 你好" in result
        assert "warn λ 你好" in result
        assert "\x00" not in result
        assert "Exit code: 0" in result

        data = (tmp_path / "marker.txt").read_bytes()
        assert b"\x00" not in data
        assert data.decode("utf-8-sig").strip() == "file café λ 你好"

    @pytest.mark.asyncio
    async def test_windows_powershell_native_pipeline_input_is_utf8(self):
        python = sys.executable.replace("'", "''")
        result = await self._execute(
            timeout=180,
            command=(
                f"[string][char]0x4F1A | & '{python}' "
                '-c "import sys; print(sys.stdin.buffer.read().hex())"'
            ),
            shell="powershell",
        )

        assert "e4bc9a0d0a" in result
        assert "Exit code: 0" in result

    @pytest.mark.asyncio
    async def test_windows_powershell_session_output_is_utf8(self):
        manager = ExecSessionManager()
        try:
            result = await ExecTool(timeout=180, exec_session_manager=manager).execute(
                command="Start-Sleep -Milliseconds 1500; Write-Output 'café λ 你好'",
                shell="powershell",
                yield_time_ms=1000,
            )

            if "session_id:" in result:
                session_id = result.split("session_id:", 1)[1].splitlines()[0].strip()
                poll_result = await ExecSessionTool(manager=manager).execute(
                    session_id=session_id,
                    input="",
                    wait_for="café λ 你好",
                    timeout_ms=120_000,
                )
                result += "\n" + poll_result
                if "Process running." in poll_result:
                    final_result = await ExecSessionTool(manager=manager).execute(
                        session_id=session_id,
                        input="",
                        timeout_ms=30_000,
                    )
                    result += "\n" + final_result
                    assert "Process running." not in final_result

            assert "café λ 你好" in result
            assert "Exit code: 0" in result
            assert "\x00" not in result
        finally:
            await manager.close_all()


# ---------------------------------------------------------------------------
# NanoClaw: killing a Windows process tree, Job-object ownership
# ---------------------------------------------------------------------------


def _live_process(pid: int = 4321) -> MagicMock:
    process = MagicMock()
    process.pid = pid
    process.returncode = None
    process.wait = AsyncMock(return_value=1)
    return process


class TestKillProcessTreeWindows:

    @pytest.mark.asyncio
    async def test_without_owner_runs_taskkill_on_the_tree(self):
        process = _live_process()
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("nanoclaw.agent.tools.shell.subprocess.run") as run,
            patch("nanoclaw.agent.tools.shell._reap_pid") as reap,
        ):
            await ExecTool._kill_process_tree(process)

        assert run.call_args.args[0] == ["taskkill", "/PID", "4321", "/T", "/F"]
        assert run.call_args.kwargs["check"] is False
        process.kill.assert_called_once_with()  # then the root itself, as nanobot
        process.wait.assert_awaited_once_with()
        reap.assert_called_once_with(4321)

    @pytest.mark.asyncio
    async def test_taskkill_failure_still_kills_the_root(self):
        process = _live_process()
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("nanoclaw.agent.tools.shell.subprocess.run", side_effect=OSError("taskkill missing")),
            patch("nanoclaw.agent.tools.shell._reap_pid"),
        ):
            await ExecTool._kill_process_tree(process)

        process.kill.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_no_taskkill_after_the_root_exited(self):
        process = _live_process()
        process.returncode = 0
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("nanoclaw.agent.tools.shell.subprocess.run") as run,
            patch("nanoclaw.agent.tools.shell._reap_pid"),
        ):
            await ExecTool._kill_process_tree(process)

        run.assert_not_called()
        process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_owner_terminates_the_job_instead_of_taskkill(self):
        process = _live_process()
        job = MagicMock(spec=_FakeWindowsJob)
        setattr(process, "_nanoclaw_process_tree_owner", job)
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("nanoclaw.agent.tools.shell.subprocess.run") as run,
            patch("nanoclaw.agent.tools.shell._reap_pid"),
        ):
            await ExecTool._kill_process_tree(process)

        job.terminate.assert_called_once_with()
        run.assert_not_called()
        assert ExecTool._process_tree_owner(process) is None

    def test_release_drops_the_owner(self):
        process = _live_process()
        job = MagicMock(spec=_FakeWindowsJob)
        setattr(process, "_nanoclaw_process_tree_owner", job)

        ExecTool._release_process_tree(process)

        job.release.assert_called_once_with()
        assert ExecTool._process_tree_owner(process) is None


class TestWindowsJobOwnership:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("command", ["dir", ["rg", "--version"]], ids=["shell", "argv"])
    async def test_process_tree_starts_suspended_in_a_job(self, command):
        job = MagicMock(spec=_FakeWindowsJob)
        job.creation_flags = 0x4
        process = _live_process()
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("nanoclaw.agent.tools.shell.sys", MagicMock(platform="win32")),
            patch("nanoclaw.agent.tools.shell.shutil.which", return_value=r"C:\bin\tool.exe"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process) as mock_exec,
            patch.object(ExecTool, "_create_windows_job", return_value=job),
        ):
            spawned = await ExecTool._spawn(command, r"C:\work", {"PATH": ""}, process_tree=True)

        assert mock_exec.call_args.kwargs["creationflags"] == 0x4
        job.assign_and_resume.assert_called_once_with(4321)
        assert ExecTool._process_tree_owner(spawned) is job

    @pytest.mark.asyncio
    async def test_cmd_process_tree_is_owned_too(self):
        job = MagicMock(spec=_FakeWindowsJob)
        job.creation_flags = 0x4
        process = _live_process()
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("nanoclaw.agent.tools.shell.sys", MagicMock(platform="win32")),
            patch("asyncio.create_subprocess_shell", new_callable=AsyncMock, return_value=process) as mock_shell,
            patch.object(ExecTool, "_create_windows_job", return_value=job),
        ):
            spawned = await ExecTool._spawn(
                "dir", r"C:\work", {"PATH": ""}, r"C:\Windows\system32\cmd.exe", process_tree=True,
            )

        assert mock_shell.call_args.kwargs["creationflags"] == 0x4
        assert ExecTool._process_tree_owner(spawned) is job

    @pytest.mark.asyncio
    async def test_job_creation_failure_falls_back_to_taskkill(self):
        """NanoClaw difference: nanobot fails the command when Windows refuses a job."""
        process = _live_process()
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("nanoclaw.agent.tools.shell.sys", MagicMock(platform="win32")),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process) as mock_exec,
            patch.object(ExecTool, "_create_windows_job", side_effect=OSError("CreateJobObjectW failed")),
            patch("nanoclaw.agent.tools.shell.subprocess.run") as run,
            patch("nanoclaw.agent.tools.shell._reap_pid"),
        ):
            spawned = await ExecTool._spawn("dir", r"C:\work", {"PATH": ""}, process_tree=True)
            assert mock_exec.call_args.kwargs["creationflags"] == 0
            assert ExecTool._process_tree_owner(spawned) is None
            await ExecTool._kill_process_tree(spawned)

        assert run.call_args.args[0] == ["taskkill", "/PID", "4321", "/T", "/F"]

    def test_shell_parameter_describes_the_platform_shells(self):
        with patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True):
            windows = ExecTool().parameters["properties"]["shell"]["description"]
        with patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False):
            posix = ExecTool().parameters["properties"]["shell"]["description"]
        assert windows == "Shell override; omit for PowerShell, or pass 'cmd' for cmd.exe."
        assert posix == "Shell override; omit for bash, or pass 'sh' or 'zsh'."

    @pytest.mark.skipif(sys.platform == "win32", reason="ctypes.WinDLL exists only on Windows")
    def test_job_module_needs_windows(self):
        with pytest.raises(AttributeError):
            ExecTool._create_windows_job()
