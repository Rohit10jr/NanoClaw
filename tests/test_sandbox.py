"""Tests for the bubblewrap sandbox (layer 18, task 18.1).

TestBwrapBackend and TestUnknownBackend are ported from nanobot
tests/tools/test_sandbox.py (c4dac64); the seatbelt tests are left out
(decision 11) and the media-folder test is replaced by one showing there is no
media bind. TestSandboxPlatform is nanobot tests/tools/test_exec_platform.py
``TestSandboxPlatform`` (bwrap only); the bind-root guard tests are nanobot
tests/tools/test_exec_security.py; the rg test is nanobot
tests/tools/test_rg_tool.py. The rest (forced restriction, config wiring, live
runs) are NanoClaw's own. Live tests run only where ``bwrap`` works.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from nanoclaw.agent.tools import build_default_tools
from nanoclaw.agent.tools.rg import RgTool
from nanoclaw.agent.tools.sandbox import wrap_command
from nanoclaw.agent.tools.shell import ExecTool, _PreparedCommand
from nanoclaw.config.schema import ExecToolConfig
from tests.test_exec_platform import _CompletedProcess


def _parse(cmd: str) -> list[str]:
    """Split a wrapped command back into tokens for assertion."""
    return shlex.split(cmd)


def _bwrap_usable() -> bool:
    if shutil.which("bwrap") is None:
        return False
    try:
        probe = subprocess.run(
            ["bwrap", "--ro-bind", "/", "/", "true"], capture_output=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


BWRAP_USABLE = _bwrap_usable()
needs_bwrap = pytest.mark.skipif(
    not BWRAP_USABLE, reason="bwrap is not installed or cannot create namespaces here"
)


class TestBwrapBackend:
    def test_basic_structure(self, tmp_path):
        ws = str(tmp_path / "project")
        result = wrap_command("bwrap", "echo hi", ws, ws)
        tokens = _parse(result)

        assert tokens[0] == "bwrap"
        assert "--new-session" in tokens
        assert "--die-with-parent" in tokens
        assert "--ro-bind" in tokens
        assert "--proc" in tokens
        assert "--dev" in tokens
        assert "--tmpfs" in tokens

        sep = tokens.index("--")
        assert tokens[sep + 1:] == ["sh", "-c", "echo hi"]

    def test_workspace_bind_mounted_rw(self, tmp_path):
        ws = str(tmp_path / "project")
        result = wrap_command("bwrap", "ls", ws, ws)
        tokens = _parse(result)

        bind_idx = [i for i, t in enumerate(tokens) if t == "--bind"]
        assert any(tokens[i + 1] == ws and tokens[i + 2] == ws for i in bind_idx)

    def test_home_env_points_to_workspace(self, tmp_path):
        ws = str(tmp_path / "project")
        result = wrap_command("bwrap", "echo $HOME", ws, ws)
        tokens = _parse(result)

        setenv_idx = [i for i, t in enumerate(tokens) if t == "--setenv"]
        assert any(
            tokens[i + 1] == "HOME" and tokens[i + 2] == str(tmp_path / "project")
            for i in setenv_idx
        )

    def test_parent_dir_masked_with_tmpfs(self, tmp_path):
        ws = tmp_path / "project"
        result = wrap_command("bwrap", "ls", str(ws), str(ws))
        tokens = _parse(result)

        tmpfs_indices = [i for i, t in enumerate(tokens) if t == "--tmpfs"]
        tmpfs_targets = {tokens[i + 1] for i in tmpfs_indices}
        assert str(ws.parent) in tmpfs_targets

    def test_tmp_dir_mounted_as_tmpfs(self, tmp_path):
        """Regression coverage for nanobot #1948: commands need writable scratch space."""
        ws = tmp_path / "project"
        result = wrap_command("bwrap", "touch /tmp/probe", str(ws), str(ws))
        tokens = _parse(result)

        tmpfs_indices = [i for i, t in enumerate(tokens) if t == "--tmpfs"]
        tmpfs_targets = {tokens[i + 1] for i in tmpfs_indices}
        assert "/tmp" in tmpfs_targets

    def test_parent_mask_precedes_workspace_recreation(self, tmp_path):
        ws = tmp_path / "project"
        result = wrap_command("bwrap", "ls", str(ws), str(ws))
        tokens = _parse(result)

        parent_mask = next(
            i for i, t in enumerate(tokens) if t == "--tmpfs" and tokens[i + 1] == str(ws.parent)
        )
        workspace_dir = next(i for i, t in enumerate(tokens) if t == "--dir" and tokens[i + 1] == str(ws))
        workspace_bind = next(
            i for i, t in enumerate(tokens)
            if t == "--bind" and tokens[i + 1] == str(ws) and tokens[i + 2] == str(ws)
        )
        chdir = tokens.index("--chdir")

        assert parent_mask < workspace_dir < workspace_bind < chdir

    def test_cwd_inside_workspace(self, tmp_path):
        ws = tmp_path / "project"
        sub = ws / "src" / "lib"
        result = wrap_command("bwrap", "pwd", str(ws), str(sub))
        tokens = _parse(result)

        chdir_idx = tokens.index("--chdir")
        assert tokens[chdir_idx + 1] == str(sub)

    def test_cwd_outside_workspace_falls_back(self, tmp_path):
        ws = tmp_path / "project"
        outside = tmp_path / "other"
        result = wrap_command("bwrap", "pwd", str(ws), str(outside))
        tokens = _parse(result)

        chdir_idx = tokens.index("--chdir")
        assert tokens[chdir_idx + 1] == str(ws.resolve())

    def test_command_with_special_characters(self, tmp_path):
        ws = str(tmp_path / "project")
        cmd = "echo 'hello world' && cat \"file with spaces.txt\""
        result = wrap_command("bwrap", cmd, ws, ws)
        tokens = _parse(result)

        sep = tokens.index("--")
        assert tokens[sep + 1:] == ["sh", "-c", cmd]

    def test_system_dirs_ro_bound(self, tmp_path):
        ws = str(tmp_path / "project")
        result = wrap_command("bwrap", "ls", ws, ws)
        tokens = _parse(result)

        ro_bind_indices = [i for i, t in enumerate(tokens) if t == "--ro-bind"]
        ro_targets = {tokens[i + 1] for i in ro_bind_indices}
        assert "/usr" in ro_targets

    def test_optional_dirs_use_ro_bind_try(self, tmp_path):
        ws = str(tmp_path / "project")
        result = wrap_command("bwrap", "ls", ws, ws)
        tokens = _parse(result)

        try_indices = [i for i, t in enumerate(tokens) if t == "--ro-bind-try"]
        try_targets = {tokens[i + 1] for i in try_indices}
        assert "/bin" in try_targets
        assert "/etc/ssl/certs" in try_targets

    def test_no_media_dir_bind(self, tmp_path):
        """NanoClaw difference: no media folder, so only system folders are ro-bound by default."""
        ws = str(tmp_path / "project")
        tokens = _parse(wrap_command("bwrap", "ls", ws, ws))

        try_indices = [i for i, t in enumerate(tokens) if t == "--ro-bind-try"]
        assert all(tokens[i + 1].startswith(("/bin", "/lib", "/etc/")) for i in try_indices)

    def test_custom_read_only_binds_use_ro_bind_try(self, tmp_path):
        ws = tmp_path / "project"
        tool_bin = tmp_path / "home" / ".local" / "bin"

        result = wrap_command("bwrap", "uv --version", str(ws), str(ws), sandbox_ro_binds=[str(tool_bin)])
        tokens = _parse(result)

        try_indices = [i for i, t in enumerate(tokens) if t == "--ro-bind-try"]
        try_pairs = {(tokens[i + 1], tokens[i + 2]) for i in try_indices}
        assert (str(tool_bin.resolve(strict=False)), str(tool_bin.resolve(strict=False))) in try_pairs

    def test_custom_read_write_binds_use_bind_try(self, tmp_path):
        ws = tmp_path / "project"
        cache_dir = tmp_path / "cache"

        result = wrap_command("bwrap", "touch cache/file", str(ws), str(ws), sandbox_rw_binds=[str(cache_dir)])
        tokens = _parse(result)

        bind_try_indices = [i for i, t in enumerate(tokens) if t == "--bind-try"]
        bind_try_pairs = {(tokens[i + 1], tokens[i + 2]) for i in bind_try_indices}
        resolved = str(cache_dir.resolve(strict=False))
        assert (resolved, resolved) in bind_try_pairs

    def test_custom_relative_bind_paths_are_ignored(self, tmp_path):
        ws = tmp_path / "project"

        result = wrap_command(
            "bwrap",
            "ls",
            str(ws),
            str(ws),
            sandbox_ro_binds=["relative/bin"],
            sandbox_rw_binds=["relative/cache"],
        )
        tokens = _parse(result)

        assert "relative/bin" not in tokens
        assert "relative/cache" not in tokens

    def test_custom_workspace_parent_binds_are_ignored(self, tmp_path):
        ws = tmp_path / "private" / "project"
        parent = ws.parent.resolve(strict=False)

        result = wrap_command(
            "bwrap",
            "cat ../config.json",
            str(ws),
            str(ws),
            sandbox_ro_binds=[str(parent)],
            sandbox_rw_binds=[str(parent)],
        )
        tokens = _parse(result)

        ro_try_indices = [i for i, token in enumerate(tokens) if token == "--ro-bind-try"]
        ro_try_pairs = {(tokens[i + 1], tokens[i + 2]) for i in ro_try_indices}
        bind_try_indices = [i for i, token in enumerate(tokens) if token == "--bind-try"]
        bind_try_pairs = {(tokens[i + 1], tokens[i + 2]) for i in bind_try_indices}
        assert (str(parent), str(parent)) not in ro_try_pairs
        assert (str(parent), str(parent)) not in bind_try_pairs

    def test_duplicate_and_blank_binds_collapse(self, tmp_path):
        ws = tmp_path / "project"
        cache = tmp_path / "cache"
        tokens = _parse(
            wrap_command(
                "bwrap", "ls", str(ws), str(ws),
                sandbox_rw_binds=[str(cache), f"{cache}/", "  ", str(cache / "x" / "..")],
            )
        )
        assert tokens.count("--bind-try") == 1


class TestUnknownBackend:
    def test_raises_value_error(self, tmp_path):
        ws = str(tmp_path / "project")
        with pytest.raises(ValueError, match="Unknown sandbox backend"):
            wrap_command("nonexistent", "ls", ws, ws)

    def test_empty_string_raises(self, tmp_path):
        ws = str(tmp_path / "project")
        with pytest.raises(ValueError):
            wrap_command("", "ls", ws, ws)

    def test_seatbelt_is_unknown(self, tmp_path):
        """Decision 11: bubblewrap only."""
        ws = str(tmp_path / "project")
        with pytest.raises(ValueError, match=r"Unknown sandbox backend 'seatbelt'. Available: \['bwrap'\]"):
            wrap_command("seatbelt", "ls", ws, ws)


class TestSandboxPlatform:
    async def test_sandbox_skipped_on_windows(self):
        """A configured sandbox is ignored on Windows, with nanobot's warning."""
        mock_proc = _CompletedProcess(b"ok")
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", True),
            patch("nanoclaw.agent.tools.shell.logger.warning") as warn,
            patch.object(ExecTool, "_spawn", return_value=mock_proc) as mock_spawn,
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            tool = ExecTool(sandbox="bwrap")
            result = await tool.execute(command="dir")

        assert "ok" in result
        spawned_cmd = mock_spawn.call_args[0][0]
        assert "bwrap" not in spawned_cmd
        warn.assert_called_once_with("Sandbox '{}' is not supported on Windows; running unsandboxed", "bwrap")

    async def test_sandbox_applied_on_unix(self):
        mock_proc = _CompletedProcess(b"sandboxed")
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False),
            patch("nanoclaw.agent.tools.shell.wrap_command", return_value="bwrap -- sh -c ls") as mock_wrap,
            patch.object(ExecTool, "_spawn", return_value=mock_proc) as mock_spawn,
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            tool = ExecTool(sandbox="bwrap", working_dir="/workspace")
            await tool.execute(command="ls")

        mock_wrap.assert_called_once()
        assert "bwrap" in mock_spawn.call_args[0][0]

    async def test_sandbox_receives_configured_bind_roots(self, tmp_path):
        mock_proc = _CompletedProcess(b"sandboxed")
        tool_bin = tmp_path / "tool-bin"
        tool_cache = tmp_path / "tool-cache"
        with (
            patch("nanoclaw.agent.tools.shell._IS_WINDOWS", False),
            patch("nanoclaw.agent.tools.shell.wrap_command", return_value="bwrap -- sh -c ls") as mock_wrap,
            patch.object(ExecTool, "_spawn", return_value=mock_proc),
            patch.object(ExecTool, "_guard_command", return_value=None),
        ):
            tool = ExecTool(
                sandbox="bwrap",
                working_dir="/workspace",
                sandbox_ro_binds=[str(tool_bin)],
                sandbox_rw_binds=[str(tool_cache)],
            )
            await tool.execute(command="ls")

        kwargs = mock_wrap.call_args.kwargs
        assert kwargs["sandbox_ro_binds"] == [str(tool_bin.resolve(strict=False))]
        assert kwargs["sandbox_rw_binds"] == [str(tool_cache.resolve(strict=False))]

    def test_string_command_wrapped_and_cwd_is_workspace(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nanoclaw.agent.tools.shell._IS_WINDOWS", False)
        (tmp_path / "sub").mkdir()
        prepared = ExecTool(working_dir=str(tmp_path), sandbox="bwrap")._prepare_command("ls -la", "sub")
        assert isinstance(prepared, _PreparedCommand)
        assert isinstance(prepared.command, str)
        tokens = shlex.split(prepared.command)
        assert tokens[0] == "bwrap"
        assert tokens[tokens.index("--chdir") + 1] == str(tmp_path.resolve() / "sub")
        assert tokens[-3:] == ["sh", "-c", "ls -la"]
        assert prepared.cwd == str(tmp_path.resolve())

    def test_unknown_backend_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nanoclaw.agent.tools.shell._IS_WINDOWS", False)
        with pytest.raises(ValueError, match="Unknown sandbox backend"):
            ExecTool(working_dir=str(tmp_path), sandbox="seatbelt")._prepare_command("ls")

    def test_rg_direct_search_preserves_sandbox_and_arguments(self, tmp_path, monkeypatch):
        """nanobot test_rg_tool.py: argv commands are wrapped and split back into argv."""
        monkeypatch.setattr("nanoclaw.agent.tools.shell._IS_WINDOWS", False)
        args = ["rg", "-F", "hello | $HOME; 'world'", "."]
        prepared = RgTool(working_dir=str(tmp_path), sandbox="bwrap")._prepare_command(args)
        assert isinstance(prepared, _PreparedCommand)
        assert isinstance(prepared.command, list)
        assert prepared.command[0] == "bwrap"
        assert prepared.command[-2] == "-c"
        assert shlex.split(prepared.command[-1].split("\n")[-1]) == args


class TestSandboxForcesRestriction:
    def test_sandbox_forces_exec_workspace_guard(self, tmp_path, monkeypatch):
        """A sandbox implies restricted mode (nanobot sandbox_restricts_workspace)."""
        monkeypatch.setattr("nanoclaw.agent.tools.shell._IS_WINDOWS", False)
        tool = ExecTool(working_dir=str(tmp_path), sandbox="bwrap")
        assert tool.restrict_to_workspace is False
        result = tool._prepare_command("cat /etc/passwd")
        assert isinstance(result, str) and "path outside working dir" in result
        result = tool._prepare_command("ls", "/etc")
        assert isinstance(result, str) and "outside the configured workspace" in result

    def test_sandbox_forces_guard_even_on_windows(self, tmp_path, monkeypatch):
        """nanobot forces restriction from the configured sandbox, even where it is not applied."""
        monkeypatch.setattr("nanoclaw.agent.tools.shell._IS_WINDOWS", True)
        tool = ExecTool(working_dir=str(tmp_path), sandbox="bwrap")
        result = tool._prepare_command("ls", "/etc")
        assert isinstance(result, str) and "outside the configured workspace" in result

    def test_build_default_tools_restricts_file_tools(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("secret")
        registry = build_default_tools(ws, exec_config=ExecToolConfig(sandbox="bwrap"))
        read = registry.get("read_file")
        with pytest.raises(PermissionError):
            read._resolve_read(str(outside))
        exec_tool = registry.get("exec")
        assert exec_tool.sandbox == "bwrap"
        assert exec_tool.restrict_to_workspace is False

    def test_bind_roots_do_not_grant_file_tool_access(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()
        cfg = ExecToolConfig(sandbox="bwrap", sandbox_rw_binds=[str(cache)])
        registry = build_default_tools(ws, exec_config=cfg)
        with pytest.raises(PermissionError):
            registry.get("write_file")._resolve_write(str(cache / "x"))
        assert registry.get("exec").sandbox_rw_binds == [cache.resolve()]

    def test_no_sandbox_leaves_file_tools_unrestricted(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        outside = tmp_path / "x.txt"
        registry = build_default_tools(ws, exec_config=ExecToolConfig())
        assert registry.get("read_file")._resolve_read(str(outside)) == outside.resolve()

    def test_config_json_names(self):
        cfg = ExecToolConfig.model_validate(
            {"sandbox": "bwrap", "sandboxRoBinds": ["/opt/tools"], "sandboxRwBinds": ["/var/cache/x"]}
        )
        assert (cfg.sandbox, cfg.sandbox_ro_binds, cfg.sandbox_rw_binds) == (
            "bwrap", ["/opt/tools"], ["/var/cache/x"]
        )


class TestSandboxBindRootsInGuard:
    """nanobot tests/tools/test_exec_security.py (bwrap parametrization)."""

    def test_exec_allows_absolute_path_inside_sandbox_ro_bind(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        tool_bin = tmp_path / "home" / ".local" / "bin"
        tool_bin.mkdir(parents=True)
        uv = tool_bin / "uv"
        uv.write_text("#!/bin/sh\n")
        monkeypatch.setattr("nanoclaw.agent.tools.shell._IS_WINDOWS", False)
        tool = ExecTool(
            working_dir=str(workspace), restrict_to_workspace=True, sandbox="bwrap", sandbox_ro_binds=[str(tool_bin)]
        )
        blocked = tool._guard_command(
            f"{uv} --version", str(workspace), restrict_to_workspace=True, workspace_root=str(workspace)
        )
        assert blocked is None

    def test_exec_allows_absolute_path_inside_sandbox_rw_bind(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("nanoclaw.agent.tools.shell._IS_WINDOWS", False)
        tool = ExecTool(
            working_dir=str(workspace), restrict_to_workspace=True, sandbox="bwrap", sandbox_rw_binds=[str(cache_dir)]
        )
        blocked = tool._guard_command(
            f"touch {cache_dir / 'stamp'}", str(workspace), restrict_to_workspace=True, workspace_root=str(workspace)
        )
        assert blocked is None

    def test_exec_bind_roots_do_not_widen_guard_without_sandbox(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        tool_bin = tmp_path / "home" / ".local" / "bin"
        tool_bin.mkdir(parents=True)
        uv = tool_bin / "uv"
        uv.write_text("#!/bin/sh\n")
        tool = ExecTool(
            working_dir=str(workspace), restrict_to_workspace=True, sandbox="", sandbox_ro_binds=[str(tool_bin)]
        )
        blocked = tool._guard_command(
            f"{uv} --version", str(workspace), restrict_to_workspace=True, workspace_root=str(workspace)
        )
        assert blocked is not None
        assert "path outside working dir" in blocked

    def test_exec_bind_roots_do_not_widen_guard_on_windows(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        cache = tmp_path / "cache"
        monkeypatch.setattr("nanoclaw.agent.tools.shell._IS_WINDOWS", True)
        tool = ExecTool(working_dir=str(workspace), sandbox="bwrap", sandbox_rw_binds=[str(cache)])
        assert tool._active_sandbox_bind_roots(workspace) == []

    def test_exec_sandbox_bind_parent_does_not_widen_workspace_guard(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        secret = tmp_path / "config.json"
        secret.write_text("secret")
        monkeypatch.setattr("nanoclaw.agent.tools.shell._IS_WINDOWS", False)
        tool = ExecTool(
            working_dir=str(workspace), restrict_to_workspace=True, sandbox="bwrap", sandbox_ro_binds=[str(tmp_path)]
        )
        blocked = tool._guard_command(
            f"cat {secret}", str(workspace), restrict_to_workspace=True, workspace_root=str(workspace)
        )
        assert blocked is not None
        assert "path outside working dir" in blocked


@needs_bwrap
class TestBwrapLive:
    """Run real commands through bubblewrap.

    The text guard is switched off so these show the OS boundary itself.
    """

    @pytest.fixture(autouse=True)
    def _no_guard(self):
        with patch.object(ExecTool, "_guard_command", return_value=None):
            yield

    async def test_workspace_writable_parent_hidden(self, tmp_path):
        ws = tmp_path / "project"
        ws.mkdir()
        (tmp_path / "config.json").write_text("secret")
        tool = ExecTool(working_dir=str(ws), sandbox="bwrap")
        result = await tool.execute(
            command="echo hi > made.txt && pwd && echo HOME=$HOME && ls -A .. && ls /home 2>&1; true"
        )
        assert (ws / "made.txt").read_text() == "hi\n"
        assert str(ws) in result
        assert f"HOME={ws}" in result
        assert "config.json" not in result

    async def test_system_is_read_only_and_tmp_private(self, tmp_path):
        ws = tmp_path / "project"
        ws.mkdir()
        tool = ExecTool(working_dir=str(ws), sandbox="bwrap")
        result = await tool.execute(command="touch /usr/nanoclaw-probe 2>&1; echo x > /tmp/probe && cat /tmp/probe")
        assert "Read-only file system" in result or "Permission denied" in result
        assert not Path("/usr/nanoclaw-probe").exists()

    async def test_rw_bind_writable_ro_bind_read_only(self, tmp_path):
        ws = tmp_path / "project"
        ws.mkdir()
        ro = tmp_path / "tools"
        ro.mkdir()
        (ro / "tool.txt").write_text("tool")
        rw = tmp_path / "cache"
        rw.mkdir()
        tool = ExecTool(
            working_dir=str(ws), sandbox="bwrap", sandbox_ro_binds=[str(ro)], sandbox_rw_binds=[str(rw)]
        )
        result = await tool.execute(command=f"cat {ro}/tool.txt; touch {ro}/new 2>&1; echo ok > {rw}/out")
        assert "tool" in result
        assert not (ro / "new").exists()
        assert (rw / "out").read_text() == "ok\n"

    async def test_argv_command_runs(self, tmp_path):
        ws = tmp_path / "project"
        ws.mkdir()
        tool = ExecTool(working_dir=str(ws), sandbox="bwrap")
        result = await tool.execute(command=["sh", "-c", "echo argv-$((1+1))"])
        assert "argv-2" in result


# --- A configured sandbox under a bound project scope (layer 18 integration) ---


def _bound_full_project(tmp_path):
    from nanoclaw.security.workspace_access import bind_workspace_scope, project_workspace_scope

    workspace = tmp_path / "ws"
    project = tmp_path / "proj"
    workspace.mkdir()
    project.mkdir()
    scope = project_workspace_scope(project, workspace=workspace, restrict_to_workspace=False)
    return workspace, project, scope, bind_workspace_scope(scope)


@pytest.mark.asyncio
async def test_sandbox_restricts_file_tools_under_an_unrestricted_project_scope(tmp_path):
    from nanoclaw.security.workspace_access import reset_workspace_scope

    workspace, project, _scope, token = _bound_full_project(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        tools = build_default_tools(workspace, exec_config=ExecToolConfig(sandbox="bwrap"))
        blocked = await tools.get("read_file").execute(path=str(outside))
        assert "outside allowed directory" in blocked
        await tools.get("write_file").execute(path="inside.txt", content="ok")
        assert (project / "inside.txt").read_text(encoding="utf-8") == "ok"

        # Without a sandbox the same full-access scope reads anywhere.
        open_tools = build_default_tools(workspace, exec_config=ExecToolConfig())
        assert "secret" in await open_tools.get("read_file").execute(path=str(outside))
    finally:
        reset_workspace_scope(token)


def test_sandboxed_exec_under_a_project_scope_uses_the_project_as_root(tmp_path):
    from nanoclaw.security.workspace_access import reset_workspace_scope

    workspace, project, _scope, token = _bound_full_project(tmp_path)
    try:
        tool = ExecTool(working_dir=str(workspace), sandbox="bwrap")
        prepared = tool._prepare_command("ls", None)
        assert isinstance(prepared, _PreparedCommand)
        assert prepared.cwd == str(project.resolve())
        assert "--bind" in prepared.command and str(project.resolve()) in prepared.command
        # The full-access scope is restricted by the sandbox: paths outside are refused.
        blocked = tool._prepare_command("cat /etc/hostname", None)
        assert isinstance(blocked, str) and "outside" in blocked
    finally:
        reset_workspace_scope(token)


def test_runtime_block_names_the_configured_sandbox(tmp_path):
    from nanoclaw.agent.loop import AgentLoop
    from nanoclaw.providers import FakeProvider
    from nanoclaw.session import SessionManager

    line = "Shell sandbox configured; extra binds do not grant file-tool access."

    def access_text(exec_config):
        agent = tmp_path / "agent"
        agent.mkdir(exist_ok=True)
        loop = AgentLoop(
            FakeProvider([]), agent,
            sessions=SessionManager(agent, sessions_root=tmp_path / "s"),
            exec_config=exec_config,
        )
        session = loop.sessions.get_or_create("cli:direct")
        blocks = loop._runtime_blocks("hi", session, ephemeral=False)
        return next(b.content for b in blocks if b.source == "workspace_access")

    sandboxed = access_text(ExecToolConfig(sandbox="bwrap"))
    assert line in sandboxed
    # The sandbox forces the restriction.
    assert "File access: project and tool-specific allowed paths only." in sandboxed
    plain = access_text(ExecToolConfig())
    assert line not in plain
    assert "File access: no workspace restriction." in plain
