"""The project folder (layer 18, task 18.2): WorkspaceScope, tools, prompt, CLI.

The scope tests are ported from nanobot tests/agent/test_workspace_scope.py
and the runtime-block tests from tests/agent/tools/test_runtime_control.py
(``workspace_context``); the loop and CLI tests are NanoClaw's.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from nanoclaw.__main__ import main
from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.subagent import SubagentManager
from nanoclaw.agent.tools.apply_patch import ApplyPatchTool
from nanoclaw.agent.tools.context import RequestContext, request_context
from nanoclaw.agent.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from nanoclaw.agent.tools.rg import RgTool
from nanoclaw.agent.tools.shell import ExecTool
from nanoclaw.agent.tools.spawn import SpawnTool
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.runtime_context import RUNTIME_CONTEXT_END, RUNTIME_CONTEXT_TAG
from nanoclaw.security.workspace_access import (
    WorkspaceScopeError,
    bind_workspace_scope,
    build_workspace_scope,
    current_tool_workspace,
    current_workspace_scope,
    default_workspace_scope,
    project_workspace_scope,
    reset_workspace_scope,
    validate_workspace_scope_payload,
    workspace_access_block,
    workspace_sandbox_status,
)
from nanoclaw.session import SessionManager
from nanoclaw.utils.templates import sync_workspace_templates
from tests.conftest import requires_posix_shell, symlink_or_skip

PY = f'"{sys.executable}"'


def scope_for(project: Path, mode: str, default: Path, restrict: bool = False):
    return validate_workspace_scope_payload(
        {"project_path": str(project), "access_mode": mode},
        default_workspace=default,
        default_restrict_to_workspace=restrict,
    )


class bound:
    """``with bound(scope):`` binds a scope like the loop does for a turn."""

    def __init__(self, scope):
        self.scope = scope

    def __enter__(self):
        self.token = bind_workspace_scope(self.scope)
        return self.scope

    def __exit__(self, *exc):
        reset_workspace_scope(self.token)


def calls(*items):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(items)],
        finish_reason="tool_calls",
    )


@pytest.fixture
def layout(tmp_path):
    """An agent workspace and a separate project folder."""
    agent = tmp_path / "agent"
    project = tmp_path / "project"
    agent.mkdir()
    project.mkdir()
    return agent, project


# --- the scope ------------------------------------------------------------------------------------


def test_workspace_scope_defaults_match_legacy_config(tmp_path):
    unrestricted = default_workspace_scope(tmp_path, restrict_to_workspace=False)
    restricted = default_workspace_scope(tmp_path, restrict_to_workspace=True)
    assert unrestricted.project_path == tmp_path.resolve()
    assert unrestricted.access_mode == "full"
    assert unrestricted.restrict_to_workspace is False
    assert restricted.access_mode == "restricted"
    assert restricted.restrict_to_workspace is True


def test_workspace_scope_rejects_invalid_project_path(tmp_path):
    with pytest.raises(WorkspaceScopeError, match="absolute"):
        scope_for(Path("relative/project"), "restricted", tmp_path)
    with pytest.raises(WorkspaceScopeError, match="existing directory"):
        scope_for(tmp_path / "missing", "restricted", tmp_path)
    (tmp_path / "file.txt").write_text("x")
    with pytest.raises(WorkspaceScopeError, match="existing directory"):
        scope_for(tmp_path / "file.txt", "restricted", tmp_path)
    with pytest.raises(WorkspaceScopeError, match="restricted or full"):
        scope_for(tmp_path, "readonly", tmp_path)


def test_workspace_scope_accepts_home_relative_project_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "Desktop" / "Photos"
    project.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    scope = validate_workspace_scope_payload(
        {"project_path": "~/Desktop/Photos", "access_mode": "restricted"},
        default_workspace=tmp_path,
        default_restrict_to_workspace=False,
    )
    assert scope.project_path == project.resolve()
    assert scope.metadata()["project_path"] == str(project.resolve())


def test_access_mode_aliases_and_payload(tmp_path):
    assert build_workspace_scope(tmp_path, "restrict").access_mode == "restricted"
    assert build_workspace_scope(tmp_path, "Full_Access").access_mode == "full"
    payload = build_workspace_scope(tmp_path, "restricted").payload()
    assert payload["project_name"] == tmp_path.name and payload["restrict_to_workspace"] is True
    assert payload["sandbox_status"]["level"] == "application"


def test_project_scope_for_the_cli(tmp_path, monkeypatch, layout):
    agent, project = layout
    assert project_workspace_scope(None, workspace=agent, restrict_to_workspace=True).project_path == agent.resolve()
    monkeypatch.chdir(project.parent)
    scope = project_workspace_scope(Path("project"), workspace=agent, restrict_to_workspace=True)
    assert scope.project_path == project.resolve() and scope.access_mode == "restricted"
    assert project_workspace_scope(project, workspace=agent, restrict_to_workspace=False).access_mode == "full"
    with pytest.raises(WorkspaceScopeError):
        project_workspace_scope(Path("missing"), workspace=agent, restrict_to_workspace=False)


def test_sandbox_status_from_environment(tmp_path):
    status = workspace_sandbox_status(
        restrict_to_workspace=True,
        workspace=tmp_path,
        environ={"NANOCLAW_WORKSPACE_SANDBOX_ENFORCED": "1", "NANOCLAW_WORKSPACE_SANDBOX_PROVIDER": "bwrap"},
    )
    assert (status.level, status.enforced, status.provider_label) == ("system", True, "Bubblewrap")
    assert workspace_sandbox_status(restrict_to_workspace=False, workspace=tmp_path, environ={}).level == "off"
    assert workspace_sandbox_status(restrict_to_workspace=True, workspace=tmp_path, environ={}).level == "application"


def test_current_tool_workspace_falls_back_to_the_tool_settings(tmp_path, layout):
    agent, project = layout
    access = current_tool_workspace(agent, restrict_to_workspace=True)
    assert access.project_path == agent and access.allowed_root == agent and access.scope is None
    with bound(scope_for(project, "full", agent, restrict=True)) as scope:
        access = current_tool_workspace(agent, restrict_to_workspace=True)
        assert access.project_path == project.resolve() and access.allowed_root is None and access.scope is scope
        assert current_tool_workspace(agent, sandbox_restricts_workspace=True).allowed_root == project.resolve()
    assert current_workspace_scope() is None


# --- the system prompt ----------------------------------------------------------------------------


@pytest.mark.parametrize("access_mode", ["restricted", "full"])
def test_selected_project_is_visible_to_the_model(layout, access_mode):
    agent, project = layout
    scope = scope_for(project, access_mode, agent)
    prompt = ContextBuilder(agent).build_system_prompt(workspace=scope.project_path)
    assert prompt.index("# Tool Usage Notes") < prompt.index("# Current Project")
    assert f"Working directory: {project.resolve()}" in prompt
    assert "Use it as the default root for project files" in prompt


def test_no_current_project_section_for_the_agent_workspace(layout):
    agent, _ = layout
    builder = ContextBuilder(agent)
    assert "# Current Project" not in builder.build_system_prompt()
    assert "# Current Project" not in builder.build_system_prompt(workspace=agent)
    assert builder.build_system_prompt() == builder.build_system_prompt(workspace=agent)


def test_agents_md_from_the_project_profile_from_the_workspace(layout):
    agent, project = layout
    sync_workspace_templates(agent)
    (agent / "AGENTS.md").write_text("Agent workspace instructions.", encoding="utf-8")
    (agent / "USER.md").write_text("The user is Ada.", encoding="utf-8")
    (agent / "memory" / "MEMORY.md").write_text("# Long-term Memory\n\nAda likes tea.", encoding="utf-8")
    (project / "AGENTS.md").write_text("Run make test before committing.", encoding="utf-8")
    (project / "SOUL.md").write_text("Project soul must not load.", encoding="utf-8")
    (project / "USER.md").write_text("Project user must not load.", encoding="utf-8")
    prompt = ContextBuilder(agent).build_system_prompt(workspace=project)
    assert "## AGENTS.md\n\nRun make test before committing." in prompt
    assert "Agent workspace instructions." not in prompt
    assert "The user is Ada." in prompt and "Ada likes tea." in prompt
    assert "must not load" not in prompt
    # Identity names the agent's own files by absolute path.
    assert f"NanoClaw's agent workspace is at: {agent.resolve()}" in prompt
    assert f"- History log: {agent.resolve()}/memory/history.jsonl" in prompt


def test_project_without_agents_md_has_none(layout):
    agent, project = layout
    sync_workspace_templates(agent)
    (agent / "AGENTS.md").write_text("Agent workspace instructions.", encoding="utf-8")
    prompt = ContextBuilder(agent).build_system_prompt(workspace=project)
    assert "## AGENTS.md" not in prompt and "Agent workspace instructions." not in prompt


def test_skills_summary_uses_absolute_workspace_root_from_a_project(layout):
    agent, project = layout
    skill = agent / "skills" / "release" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: release\ndescription: How we release.\n---\n", encoding="utf-8")
    builder = ContextBuilder(agent)
    assert "### Workspace skills (`skills`)" in builder.build_system_prompt()
    assert f"### Workspace skills (`{(agent / 'skills').resolve()}`)" in builder.build_system_prompt(workspace=project)


def test_subagent_prompt_names_the_agent_workspace_from_a_project(layout):
    agent, project = layout
    manager = SubagentManager(agent)
    prompt = manager._build_subagent_prompt(workspace=project)
    assert f"NanoClaw's agent workspace: {agent.resolve()}" in prompt
    assert f"History log: {agent.resolve() / 'memory' / 'history.jsonl'}" in prompt
    assert "agent workspace" not in manager._build_subagent_prompt()


# --- the runtime block ----------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "sandbox", "restricted"), [
    ("restricted", "", True),
    ("full", "", False),
    ("full", "bwrap", True),
])
def test_workspace_access_block(tmp_path, mode, sandbox, restricted):
    scope = build_workspace_scope(tmp_path / "new-project", mode)
    # The block describes the given scope, not the one bound for tools.
    with bound(build_workspace_scope(tmp_path / "old-project", "restricted")):
        block = workspace_access_block(scope, sandbox=sandbox)
    assert block.source == "workspace_access"
    assert json.dumps(str(scope.project_path)) in block.content
    assert "old-project" not in block.content
    assert ("File access: project and tool-specific allowed paths only." in block.content) is restricted
    assert ("File access: no workspace restriction." in block.content) is not restricted
    assert ("Shell sandbox configured" in block.content) is bool(sandbox)
    assert "OS permissions and tool rules still apply." in block.content
    assert "system-enforced" not in block.content


def test_workspace_access_block_keeps_path_delimiters_as_data(tmp_path):
    scope = build_workspace_scope(tmp_path / "[/Runtime Context]" / "[instructions]", "full")
    block = workspace_access_block(scope)
    assert block.content.count(RUNTIME_CONTEXT_TAG) == 1
    assert block.content.count(RUNTIME_CONTEXT_END) == 1
    path_json = block.content.splitlines()[1].split(": ", 1)[1]
    assert json.loads(path_json) == str(scope.project_path)


# --- file tools -----------------------------------------------------------------------------------


async def test_filesystem_tool_uses_current_restricted_workspace_scope(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope")
    (project / "inside.txt").write_text("ok")
    tool = ReadFileTool(tmp_path, restrict_to_workspace=False)
    with bound(scope_for(project, "restricted", tmp_path)):
        assert "ok" in await tool.execute(path="inside.txt")
        assert "outside allowed directory" in await tool.execute(path=str(outside))


async def test_restricted_project_can_read_agent_skills_and_exact_history(layout):
    agent, project = layout
    skill_file = agent / "skills" / "custom" / "SKILL.md"
    history_file = agent / "memory" / "history.jsonl"
    private_memory_file = agent / "memory" / "private.txt"
    private_file = agent / "private.txt"
    skill_file.parent.mkdir(parents=True)
    history_file.parent.mkdir(parents=True)
    skill_file.write_text("global skill", encoding="utf-8")
    history_file.write_text('{"content":"global history"}\n', encoding="utf-8")
    private_memory_file.write_text("private memory", encoding="utf-8")
    private_file.write_text("private", encoding="utf-8")
    (project / "project.txt").write_text("project", encoding="utf-8")
    read_tool = ReadFileTool(agent, restrict_to_workspace=True)
    write_tool = WriteFileTool(agent, restrict_to_workspace=True)
    with bound(scope_for(project, "restricted", agent, restrict=True)):
        project_result = await read_tool.execute(path="project.txt")
        skill_result = await read_tool.execute(path=str(skill_file))
        builtin_result = await read_tool.execute(path="skills/weather/SKILL.md")
        history_result = await read_tool.execute(path=str(history_file))
        private_memory_result = await read_tool.execute(path=str(private_memory_file))
        private_result = await read_tool.execute(path=str(private_file))
        write_result = await write_tool.execute(path=str(skill_file), content="changed")
        history_write_result = await write_tool.execute(path=str(history_file), content="changed")
    assert "project" in project_result
    assert "global skill" in skill_result
    assert "weather" in builtin_result.lower()
    assert "global history" in history_result
    assert "outside allowed directory" in private_memory_result
    assert "outside allowed directory" in private_result
    assert "outside allowed directory" in write_result
    assert "outside allowed directory" in history_write_result
    assert skill_file.read_text(encoding="utf-8") == "global skill"
    assert history_file.read_text(encoding="utf-8") == '{"content":"global history"}\n'


async def test_restricted_project_reads_history_from_linked_agent_workspace(tmp_path):
    real_agent = tmp_path / "real-agent"
    linked_agent = tmp_path / "agent-link"
    project = tmp_path / "project"
    history_file = real_agent / "memory" / "history.jsonl"
    history_file.parent.mkdir(parents=True)
    project.mkdir()
    history_file.write_text('{"content":"linked history"}\n', encoding="utf-8")
    symlink_or_skip(linked_agent, real_agent)
    tool = ReadFileTool(linked_agent, restrict_to_workspace=True)
    with bound(scope_for(project, "restricted", linked_agent, restrict=True)):
        result = await tool.execute(path=str(history_file.resolve()))
    assert "linked history" in result


async def test_filesystem_write_tool_full_scope_allows_outside_project(tmp_path):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    tool = WriteFileTool(tmp_path, restrict_to_workspace=True)
    with bound(scope_for(project, "full", tmp_path, restrict=True)):
        result = await tool.execute(path=str(outside / "outside.txt"), content="ok")
    assert "Successfully wrote" in result
    assert (outside / "outside.txt").read_text(encoding="utf-8") == "ok"


async def test_relative_writes_list_and_patch_land_in_the_project(layout):
    agent, project = layout
    write_tool = WriteFileTool(agent)
    with bound(scope_for(project, "full", agent)):
        assert "Successfully wrote" in await write_tool.execute(path="src/app.py", content="x = 1\n")
        listing = await ListDirTool(agent).execute(path=".")
        patched = await ApplyPatchTool(agent).execute(
            edits=[{"path": "src/app.py", "action": "replace", "old_text": "x = 1", "new_text": "x = 2"}]
        )
    assert (project / "src" / "app.py").read_text(encoding="utf-8") == "x = 2\n"
    assert not (agent / "src").exists()
    assert "src" in listing
    assert "src/app.py" in patched  # shown relative to the project


async def test_restricted_patch_outside_the_project_is_refused(layout):
    agent, project = layout
    (agent / "notes.txt").write_text("a\n")
    edit = {"path": str(agent / "notes.txt"), "action": "replace", "old_text": "a", "new_text": "b"}
    with bound(scope_for(project, "restricted", agent)):
        result = await ApplyPatchTool(agent, restrict_to_workspace=True).execute(edits=[edit])
    assert "outside allowed directory" in result
    assert (agent / "notes.txt").read_text() == "a\n"


# --- exec and rg ----------------------------------------------------------------------------------


@requires_posix_shell("writes the file with a shell redirect")
async def test_exec_tool_uses_scope_project_as_default_cwd(tmp_path):
    # nanobot runs its Python by name; an absolute interpreter path outside
    # the project would trip the restricted-mode path check, so: echo.
    project = tmp_path / "project"
    project.mkdir()
    tool = ExecTool(working_dir=str(tmp_path), restrict_to_workspace=False, timeout=5)
    with bound(scope_for(project, "restricted", tmp_path)):
        result = await tool.execute(command="echo ok > scoped-marker.txt")
    assert "Exit code: 0" in result
    assert (project / "scoped-marker.txt").read_text().strip() == "ok"


async def test_exec_tool_resolves_relative_working_dir_from_scope_project(tmp_path):
    project = tmp_path / "project"
    subdir = project / "subdir"
    subdir.mkdir(parents=True)
    tool = ExecTool(working_dir=str(tmp_path), restrict_to_workspace=False, timeout=5)
    with bound(scope_for(project, "full", tmp_path)):
        result = await tool.execute(
            command=f"{PY} -c \"from pathlib import Path; print(Path.cwd())\"", working_dir="subdir"
        )
    assert "Exit code: 0" in result
    assert str(subdir.resolve()) in result


async def test_exec_full_scope_allows_explicit_cwd_outside_project(tmp_path):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    tool = ExecTool(working_dir=str(tmp_path), restrict_to_workspace=True, timeout=5)
    with bound(scope_for(project, "full", tmp_path, restrict=True)):
        result = await tool.execute(
            command=f"{PY} -c \"from pathlib import Path; Path('outside-marker.txt').write_text('ok')\"",
            working_dir=str(outside),
        )
    assert "Exit code: 0" in result
    assert (outside / "outside-marker.txt").read_text() == "ok"


async def test_exec_restricted_scope_keeps_cwd_and_paths_in_the_project(layout):
    agent, project = layout
    tool = ExecTool(working_dir=str(agent), restrict_to_workspace=False, timeout=5)
    with bound(scope_for(project, "restricted", agent)):
        outside_cwd = await tool.execute(command="echo hi", working_dir=str(agent))
        outside_path = await tool.execute(command=f"cat {agent / 'SOUL.md'}")
    assert "working_dir is outside the configured workspace" in outside_cwd
    assert "path outside working dir" in outside_path


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
async def test_rg_searches_the_project_by_default(layout):
    agent, project = layout
    (project / "code.py").write_text("needle_in_project\n")
    (agent / "notes.md").write_text("needle_in_agent\n")
    tool = RgTool(working_dir=str(agent), timeout=5)
    with bound(scope_for(project, "restricted", agent)):
        result = await tool.execute(args=["-n", "needle_"])
    assert "needle_in_project" in result and "needle_in_agent" not in result


# --- subagents ------------------------------------------------------------------------------------


async def test_spawn_tool_forwards_current_workspace_scope(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    scope = scope_for(project, "restricted", tmp_path)

    class Manager:
        seen = None

        async def spawn(self, **kwargs):
            Manager.seen = kwargs
            return "spawned"

    tool = SpawnTool(Manager())  # type: ignore[arg-type]
    with bound(scope), request_context(RequestContext(channel="cli", session_key="cli:direct", runtime=object())):  # type: ignore[arg-type]
        result = await tool.execute(task="inspect")
    assert result == "spawned"
    assert Manager.seen["workspace_scope"] == scope


async def test_subagent_works_in_the_project(layout, tmp_path):
    agent, project = layout
    child_script = [calls(("write_file", {"path": "child.txt", "content": "from child"})), "Wrote it."]
    main_script = [calls(("spawn", {"task": "write child.txt", "wait": True})), "Done."]

    class Routing(FakeProvider):
        def __init__(self):
            super().__init__(main_script)
            self.child = FakeProvider(child_script)

        async def chat(self, messages, *args, **kwargs):
            if messages[0]["content"].startswith("# Subagent"):
                return await self.child.chat(messages, *args, **kwargs)
            return await super().chat(messages, *args, **kwargs)

    provider = Routing()
    loop = AgentLoop(provider, agent, sessions=SessionManager(agent, sessions_root=tmp_path / "s"), project=project)
    result = await loop.process_direct("go")
    assert result.final_content == "Done."
    assert (project / "child.txt").read_text() == "from child"
    assert f"NanoClaw's agent workspace: {agent.resolve()}" in provider.child.calls[0].messages[0]["content"]


# --- the loop -------------------------------------------------------------------------------------


async def test_loop_without_project_sends_the_workspace_block(layout, tmp_path):
    agent, _ = layout
    fake = FakeProvider(["ok"])
    loop = AgentLoop(fake, agent, sessions=SessionManager(agent, sessions_root=tmp_path / "s"))
    await loop.process_direct("hello")
    system, user = fake.calls[0].messages
    assert "# Current Project" not in system["content"]
    assert f"Current project (JSON path): {json.dumps(str(agent.resolve()))}" in user["content"]
    assert "File access: no workspace restriction." in user["content"]


async def test_loop_with_project_binds_the_scope_per_turn(layout, tmp_path):
    agent, project = layout
    fake = FakeProvider([calls(("write_file", {"path": "out.txt", "content": "hi"})), "ok", "again"])
    loop = AgentLoop(
        fake,
        agent,
        sessions=SessionManager(agent, sessions_root=tmp_path / "s"),
        project=project,
        restrict_to_workspace=True,
    )
    await loop.process_direct("write")
    assert (project / "out.txt").read_text() == "hi"
    assert current_workspace_scope() is None  # reset after the turn
    system = fake.calls[0].messages[0]["content"]
    current = fake.calls[0].messages[-1]["content"]
    assert f"Working directory: {project.resolve()}" in system
    assert f"Current project (JSON path): {json.dumps(str(project.resolve()))}" in current
    assert "File access: project and tool-specific allowed paths only." in current
    # The second request of the turn keeps the first one's prefix (prompt cache).
    assert fake.calls[1].messages[: len(fake.calls[0].messages)] == fake.calls[0].messages
    await loop.process_direct("again")
    assert fake.calls[2].messages[0] == fake.calls[0].messages[0]
    assert fake.calls[2].messages[-1]["content"].count("Current project (JSON path):") == 1


def test_loop_rejects_a_missing_project(layout):
    agent, project = layout
    with pytest.raises(WorkspaceScopeError):
        AgentLoop(FakeProvider([]), agent, project=project / "missing")


async def test_dream_stays_on_the_agent_workspace(layout, tmp_path):
    agent, project = layout
    sync_workspace_templates(agent)
    loop = AgentLoop(
        FakeProvider([]), agent, sessions=SessionManager(agent, sessions_root=tmp_path / "s"), project=project
    )
    loop.memory.append_history("- [durable] the user prefers tabs")
    loop.provider = fake = FakeProvider([
        calls(("edit_file", {"path": "USER.md", "old_text": "", "new_text": "Prefers tabs.\n"})),
        calls(("read_file", {"path": "USER.md"})),
        "done",
    ])
    await loop.run_dream()
    system, current = fake.calls[0].messages[0]["content"], fake.calls[0].messages[-1]["content"]
    assert "# Current Project" not in system
    assert json.dumps(str(agent.resolve())) in current
    assert not (project / "USER.md").exists()
    assert "USER.md" not in [p.name for p in project.iterdir()]


# --- the CLI --------------------------------------------------------------------------------------


@requires_posix_shell("the exec call runs pwd")
def test_cli_ask_project_writes_into_the_project(capsys, layout):
    agent, project = layout
    (project / "AGENTS.md").write_text("Project rules: tabs only.", encoding="utf-8")
    fake = FakeProvider([
        calls(("write_file", {"path": "hello.txt", "content": "hi from ask"}), ("exec", {"command": "pwd"})),
        "Saved.",
    ])
    assert main(["ask", "save it", "-w", str(agent), "--project", str(project)], provider=fake) == 0
    assert capsys.readouterr().out.strip() == "Saved."
    assert (project / "hello.txt").read_text(encoding="utf-8") == "hi from ask"
    assert not (agent / "hello.txt").exists()
    system, user = fake.calls[0].messages
    assert "Project rules: tabs only." in system["content"]
    assert f"Working directory: {project.resolve()}" in system["content"]
    assert f"Current project (JSON path): {json.dumps(str(project.resolve()))}" in user["content"]
    exec_result = fake.calls[1].messages[-1]["content"]
    assert str(project.resolve()) in exec_result
    # SOUL.md / USER.md were created in the workspace, not the project.
    assert (agent / "SOUL.md").exists() and not (project / "SOUL.md").exists()


def test_cli_ask_project_from_environment(capsys, layout, monkeypatch):
    agent, project = layout
    monkeypatch.setenv("NANOCLAW_PROJECT", str(project))
    fake = FakeProvider([calls(("write_file", {"path": "env.txt", "content": "x"})), "ok"])
    assert main(["ask", "go", "-w", str(agent)], provider=fake) == 0
    assert (project / "env.txt").exists()


def test_cli_project_flag_overrides_the_environment(capsys, layout, monkeypatch, tmp_path):
    agent, project = layout
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("NANOCLAW_PROJECT", str(other))
    assert main(["prompt", "-w", str(agent), "-p", str(project)]) == 0
    out = capsys.readouterr().out
    assert f"Working directory: {project.resolve()}" in out and str(other) not in out


def test_cli_project_relative_to_the_current_directory(capsys, layout, monkeypatch):
    agent, project = layout
    monkeypatch.chdir(project.parent)
    assert main(["prompt", "-w", str(agent), "--project", "project"]) == 0
    assert f"# Current Project\n\nWorking directory: {project.resolve()}" in capsys.readouterr().out


def test_cli_prompt_without_project_has_no_section(capsys, layout):
    agent, _ = layout
    assert main(["prompt", "-w", str(agent)]) == 0
    assert "# Current Project" not in capsys.readouterr().out


@pytest.mark.parametrize("command", [["ask", "hi"], ["chat"], ["prompt"]])
def test_cli_rejects_a_missing_or_file_project(layout, command, capsys):
    agent, project = layout
    with pytest.raises(SystemExit, match="existing directory"):
        main([*command, "-w", str(agent), "--project", str(project / "missing")], provider=FakeProvider([]))
    (project / "file.txt").write_text("x")
    with pytest.raises(SystemExit, match="existing directory"):
        main([*command, "-w", str(agent), "--project", str(project / "file.txt")], provider=FakeProvider([]))
    assert not (agent / "SOUL.md").exists()  # checked before the workspace templates are written


def test_cli_rejects_a_bad_project_from_the_environment(layout, monkeypatch):
    agent, project = layout
    monkeypatch.setenv("NANOCLAW_PROJECT", str(project / "missing"))
    with pytest.raises(SystemExit, match="existing directory"):
        main(["ask", "hi", "-w", str(agent)], provider=FakeProvider([]))


def test_cli_restrict_with_project_keeps_tools_in_the_project(capsys, layout):
    agent, project = layout
    fake = FakeProvider([calls(("read_file", {"path": str(agent / "SOUL.md")})), "ok"])
    assert main(["ask", "go", "-w", str(agent), "-p", str(project), "--restrict"], provider=fake) == 0
    assert "outside allowed directory" in fake.calls[1].messages[-1]["content"]
    assert "File access: project and tool-specific allowed paths only." in fake.calls[0].messages[-1]["content"]
