"""Gaps found by the layer 25 mutation checks (skills, Agent Plugins, ``my``).

Each test pins behaviour that a mutation of the layer 25 code changed
without failing the layer's own tests: skill snapshots and their scope,
summary roots, requirement and metadata details, the web UI skill payloads,
Agent Plugin validation, containment, fingerprints and activation markers,
the ``nanoclaw plugins`` CLI, and the ``my`` tool's guards and snapshot.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nanoclaw.__main__ import main
from nanoclaw.agent import plugins as agent_plugins
from nanoclaw.agent import skills as agent_skills
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.plugins import (
    AGENT_PLUGIN_MCP_SCHEMA,
    AGENT_PLUGIN_SCHEMA,
    agent_plugin_mcp_servers,
    discover_agent_plugins,
    enabled_agent_plugin_skill_dirs,
    enabled_agent_plugin_skills,
    set_agent_plugin_enabled,
)
from nanoclaw.agent.skills import SkillsLoader, parse_skill_metadata
from nanoclaw.agent.tools import ToolRegistry
from nanoclaw.agent.tools.context import RequestContext, request_context
from nanoclaw.agent.tools.filesystem import ReadFileTool
from nanoclaw.agent.tools.runtime_control import AgentRuntimeControl
from nanoclaw.agent.tools.self import MyTool, MyToolConfig
from nanoclaw.agent.tools.web import WebFetchTool, WebSearchTool
from nanoclaw.config.loader import save_config
from nanoclaw.config.schema import Config, ExecToolConfig, ModelPresetConfig
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.security.workspace_access import (
    bind_workspace_scope,
    reset_workspace_scope,
    validate_workspace_scope_payload,
)
from nanoclaw.session import SessionManager
from nanoclaw.webui.context import WebUIContext
from nanoclaw.webui.rest_api import register_rest_routes
from nanoclaw.webui.skills_api import (
    set_webui_skill_enabled,
    webui_skill_detail_payload,
    webui_skills_payload,
)


@pytest.fixture(autouse=True)
def _isolate_plugin_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_plugins, "get_config_path", lambda: tmp_path / "config" / "config.json")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _plugin(workspace: Path, name: str = "demo", **fields: object) -> Path:
    root = workspace / "plugins" / name
    _write_json(root / "plugin.json", {"$schema": AGENT_PLUGIN_SCHEMA, "name": name, **fields})
    return root


def _skill(base: Path, name: str, frontmatter: str | None = None, body: str = "") -> Path:
    path = base / name
    path.mkdir(parents=True)
    metadata = frontmatter if frontmatter is not None else f"name: {name}\ndescription: {name} description."
    (path / "SKILL.md").write_text(f"---\n{metadata}\n---\n\n{body}\n", encoding="utf-8")
    return path


def _marker(workspace: Path, name: str = "demo") -> Path:
    return agent_plugins._plugin_data_dir(workspace, name, create=False) / "enabled"


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:  # pragma: no cover - Windows without symlink rights
        pytest.skip(f"symlinks unavailable: {exc}")


def _count_discoveries(loader: SkillsLoader, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    original = loader._discover_skills

    def discover():
        calls[0] += 1
        return original()

    monkeypatch.setattr(loader, "_discover_skills", discover)
    return calls


# --- area 1: skills -------------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    path = tmp_path / "ws"
    (path / "skills").mkdir(parents=True)
    return path


@pytest.fixture
def builtin(tmp_path: Path) -> Path:
    path = tmp_path / "builtin"
    path.mkdir()
    return path


@pytest.mark.parametrize("call", ["list", "summary", "explicit", "always"])
def test_each_public_skills_call_discovers_once(ws, builtin, monkeypatch, call):
    """Every public call opens its own snapshot, so nested lookups reuse one discovery."""
    for name in ("alpha", "beta", "gamma"):
        _skill(ws / "skills", name, f"name: {name}\ndescription: {name} d.\nalways: true")
    loader = SkillsLoader(ws, builtin_skills_dir=builtin)
    calls = _count_discoveries(loader, monkeypatch)
    if call == "list":
        assert len(loader.list_skills(filter_unavailable=True)) == 3
    elif call == "summary":
        assert "**gamma** — gamma d." in loader.build_skills_summary()
    elif call == "explicit":
        # Always-on skills are left out of the block, so nothing is returned,
        # but the lookups still share one snapshot.
        assert loader.build_explicit_skill_runtime_context("Use $alpha and $beta") is None
    else:
        assert sorted(loader.get_always_skills()) == ["alpha", "beta", "gamma"]
    assert calls[0] == 1


def test_plugin_skill_is_served_from_the_captured_bytes(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path)
    skill_file = _skill(plugin / "skills", "shared", body="Approved body.") / "SKILL.md"
    set_agent_plugin_enabled(tmp_path, "demo", True)
    loader = SkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "no-builtin")
    original_read = Path.read_text

    def read_text(path, *args, **kwargs):
        if Path(path) == skill_file.resolve():
            raise AssertionError("a plugin skill must not be re-read from disk")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert "Approved body." in (loader.load_skill("shared") or "")

    def no_parse(_content):
        raise AssertionError("captured metadata must be reused")

    monkeypatch.setattr(agent_skills, "parse_skill_metadata", no_parse)
    with loader.snapshot():
        assert [entry["name"] for entry in loader.list_skills()] == ["shared"]
        assert loader.get_skill_metadata("shared") == {"name": "shared", "description": "shared description."}


def test_summary_roots_with_a_project_folder_and_group_order(tmp_path):
    ws = tmp_path / "ws"
    _skill(ws / "skills", "mine")
    plugin = _plugin(ws)
    _skill(plugin / "skills", "plug")
    set_agent_plugin_enabled(ws, "demo", True)
    (tmp_path / "x").mkdir()
    builtin = tmp_path / "builtin"
    _skill(builtin, "built")
    loader = SkillsLoader(ws, builtin_skills_dir=tmp_path / "x" / ".." / "builtin")

    same = loader.build_skills_summary()
    assert "### Workspace skills (`skills`)" in same
    assert "### Agent Plugin skills (`plugins`)" in same
    assert f"### Built-in skills (`{builtin.resolve()}`)" in same
    assert ".." not in same
    assert same.index("Workspace skills") < same.index("Agent Plugin skills") < same.index("Built-in skills")

    project = tmp_path / "project"
    project.mkdir()
    other = loader.build_skills_summary(workspace=project)
    assert f"### Workspace skills (`{(ws / 'skills').resolve()}`)" in other
    assert f"### Agent Plugin skills (`{ws.resolve() / 'plugins'}`)" in other
    assert "`mine/SKILL.md`" in other and "`demo/skills/plug/SKILL.md`" in other


def test_requirement_details(ws, builtin, monkeypatch):
    monkeypatch.setenv("L25_SET", "1")
    monkeypatch.setattr("nanoclaw.agent.skills.shutil.which", lambda name: "/bin/x" if name == "have-it" else None)
    meta = json.dumps({"nanobot": {"requires": {"bins": ["have-it", "missing-one"], "env": ["  ", "L25_SET"]}}})
    _skill(ws / "skills", "needs", f"name: needs\ndescription: d.\nmetadata: '{meta}'")
    blank = json.dumps({"nanobot": {"requires": {"env": ["  "]}}})
    _skill(ws / "skills", "blank-env", f"name: blank-env\ndescription: d.\nmetadata: '{blank}'")
    loader = SkillsLoader(ws, builtin_skills_dir=builtin)
    assert loader.get_skill_availability("needs") == (False, "CLI: missing-one")
    assert loader.get_skill_availability("blank-env") == (True, "")
    assert loader.get_skill_requirements("blank-env")["env"] == []


def test_skill_metadata_details(ws, builtin):
    _skill(ws / "skills", "nodesc", 'name: nodesc\ndescription: ""')
    both = json.dumps({"nanoclaw": {"always": True}, "nanobot": {"always": False}})
    _skill(ws / "skills", "both", f"name: both\ndescription: d.\nmetadata: '{both}'")
    loader = SkillsLoader(ws, builtin_skills_dir=builtin)
    assert loader.get_skill_description("nodesc") == "nodesc"
    assert loader.get_always_skills() == ["both"]


def test_parse_skill_metadata_keys_and_crlf():
    assert parse_skill_metadata("---\n1: one\nname: x\n---\n") == {"1": "one", "name": "x"}
    assert parse_skill_metadata("---\r\nname: x\r\ndescription: d\r\n---\r\nbody") == {"name": "x", "description": "d"}


def test_webui_skill_payload_fields(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    monkeypatch.delenv("L25_MISSING", raising=False)
    meta = json.dumps({"nanobot": {"requires": {"env": ["L25_MISSING"]}}})
    _skill(ws / "skills", "blocked", f"name: blocked\ndescription: d.\nmetadata: '{meta}'")
    _skill(ws / "skills", "padded", 'name: padded\ndescription: "  Padded text.  "')
    _skill(ws / "skills", "blank", 'name: blank\ndescription: "   "')
    plugin = _plugin(ws)
    _skill(plugin / "skills", "plug")
    set_agent_plugin_enabled(ws, "demo", True)

    skills = {item["name"]: item for item in webui_skills_payload(ws)["skills"]}
    assert skills["blocked"]["available"] is False
    assert skills["blocked"]["unavailable_reason"] == "ENV: L25_MISSING"
    assert skills["padded"]["description"] == "Padded text."
    assert skills["blank"]["description"] == "blank"
    assert skills["plug"]["source"] == "plugin" and skills["plug"]["deletable"] is False


def test_webui_skill_install_options_details(tmp_path):
    ws = tmp_path / "ws"
    install = [
        {"kind": "brew", "package": "not-a-formula"},
        {"kind": "apt", "package": "   "},
        {"kind": "brew", "formula": "a b"},
        {"id": "custom", "kind": "apt", "package": "pkg", "label": "Use apt"},
    ]
    # ``metadata`` as a JSON string, under the NanoClaw-only ``nanoclaw`` key.
    meta = json.dumps({"nanoclaw": {"install": install}, "nanobot": {"install": []}})
    _skill(ws / "skills", "tools", f"name: tools\ndescription: d.\nmetadata: '{meta}'")
    detail = webui_skill_detail_payload(ws, "tools")
    assert detail is not None
    assert detail["install_options"] == [
        {"id": "brew", "kind": "brew", "label": "Install with brew", "command": "brew install 'a b'"},
        {"id": "custom", "kind": "apt", "label": "Use apt", "command": "sudo apt-get install -y pkg"},
    ]


@pytest.mark.parametrize("install", [5, True])
def test_webui_skill_install_options_ignore_a_non_list(tmp_path, install):
    ws = tmp_path / "ws"
    meta = json.dumps({"nanobot": {"install": install}})
    _skill(ws / "skills", "tools", f"name: tools\ndescription: d.\nmetadata: '{meta}'")
    detail = webui_skill_detail_payload(ws, "tools")
    assert detail is not None and detail["install_options"] == []


def test_skill_toggle_with_both_key_spellings_writes_the_camel_case_one(tmp_path):
    ws = tmp_path / "ws"
    _skill(ws / "skills", "custom-skill")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"agents": {"defaults": {"disabled_skills": ["a"], "disabledSkills": ["a"]}}}))
    set_webui_skill_enabled(ws, "custom-skill", enabled=False, disabled_skills=set(), config_path=path)
    defaults = json.loads(path.read_text())["agents"]["defaults"]
    assert defaults["disabled_skills"] == ["a"]
    assert defaults["disabledSkills"] == ["a", "custom-skill"]


async def test_skill_update_mutation_accepts_any_case(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    _skill(ws / "skills", "custom-skill")
    config_path = tmp_path / "config.json"
    monkeypatch.setattr("nanoclaw.config.loader._current_config_path", config_path)
    config = Config()
    save_config(config, config_path)
    ctx = WebUIContext(
        config=config, sessions=SessionManager(ws), history=SimpleNamespace(), loop=None, workspace=ws,
        check_auth=lambda request: True,
    )
    from aiohttp import web

    register_rest_routes(web.Application(), ctx)
    response = await ctx.mutations["skill.update"]({"name": "custom-skill", "enabled": "FALSE"})
    assert response.status == 200
    assert config.agents.defaults.disabled_skills == ["custom-skill"]


# --- area 2: Agent Plugins ------------------------------------------------------------------------


def test_plugin_schema_constants():
    assert AGENT_PLUGIN_SCHEMA == "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
    assert AGENT_PLUGIN_MCP_SCHEMA == "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"


@pytest.mark.parametrize(
    ("name", "valid"),
    [("a" * 64, True), ("a" * 65, False), ("a..b", False), ("a--b", False), ("abc-", False), ("abc.", False),
     ("a.b-c", True), (123, False)],
)
def test_plugin_name_rules(tmp_path, name, valid):
    _write_json(tmp_path / "plugins" / "candidate" / "plugin.json", {"$schema": AGENT_PLUGIN_SCHEMA, "name": name})
    assert [plugin.name for plugin in discover_agent_plugins(tmp_path)] == ([name] if valid else [])


def test_plugin_manifest_fields(tmp_path):
    _plugin(
        tmp_path,
        description="  Adds numbers.  ",
        extensions={"dev.nanobot": {
            "category": "Math", "permissions": ["", "  ", " Runs a server "], "accentColor": "#123456zz",
        }},
    )
    plugin = discover_agent_plugins(tmp_path)[0]
    assert plugin.description == "Adds numbers."
    assert plugin.category == "Math"
    assert plugin.permissions == ("Runs a server",)
    assert plugin.accent_color is None


@pytest.mark.parametrize(
    ("logo", "filename", "data", "expected"),
    [
        # Without ``./`` (dropping the first two characters would find assets/icon.png).
        ("xxassets/icon.png", "icon.png", b"\x89PNG\r\n\x1a\nx", None),
        ("./assets/icon.png", "icon.png", b"not a png", None),
        ("./assets/icon.png", "icon.png", b"\x89PNG\r\n\x1a\n" + b"x" * (256 * 1024), None),
        ("./assets/icon.jpg", "icon.jpg", b"\xff\xd8\xffjpg", "data:image/jpeg;base64,/9j/anBn"),
        ("./assets/icon.webp", "icon.webp", b"RIFF\0\0\0\0WEBPx", "data:image/webp;base64,UklGRgAAAABXRUJQeA=="),
        ("./assets/icon.webp", "icon.webp", b"RIFF\0\0\0\0WAVEx", None),
    ],
)
def test_plugin_logo_rules(tmp_path, logo, filename, data, expected):
    plugin = _plugin(tmp_path, extensions={"dev.nanobot": {"logo": logo}})
    (plugin / "assets").mkdir()
    (plugin / "assets" / filename).write_bytes(data)
    assert discover_agent_plugins(tmp_path)[0].logo == expected


def test_fingerprint_covers_paths_and_link_targets(tmp_path):
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "a.txt").write_text("same")
    first = agent_plugins._package_fingerprint(root)
    (root / "a.txt").rename(root / "b.txt")
    assert agent_plugins._package_fingerprint(root) != first

    (root / "c.txt").write_text("same")
    _symlink(root / "link", root / "b.txt")
    linked_b = agent_plugins._package_fingerprint(root)
    (root / "link").unlink()
    _symlink(root / "link", root / "c.txt")
    assert agent_plugins._package_fingerprint(root) != linked_b


def test_fingerprint_fails_on_unreadable_files(tmp_path, monkeypatch):
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "a.txt").write_text("x")

    def unreadable(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    assert agent_plugins._package_fingerprint(root) is None


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_special_files_block_and_withdraw_activation(tmp_path, capsys):
    plugin = _plugin(tmp_path)
    _skill(plugin / "skills", "demo-skill")
    set_agent_plugin_enabled(tmp_path, "demo", True)
    assert discover_agent_plugins(tmp_path)[0].enabled is True
    os.mkfifo(plugin / "pipe")
    assert agent_plugins._package_fingerprint(plugin) is None
    assert discover_agent_plugins(tmp_path)[0].enabled is False
    assert not _marker(tmp_path).exists()
    with pytest.raises(RuntimeError, match="changed while it was being enabled"):
        set_agent_plugin_enabled(tmp_path, "demo", True)
    assert main(["plugins", "enable", "demo", "--yes", "-w", str(tmp_path)]) == 1
    assert "cannot be read (unreadable or special files)" in capsys.readouterr().err


def test_a_changed_package_stays_disabled_after_the_change_is_reverted(tmp_path):
    plugin = _plugin(tmp_path)
    skill_file = _skill(plugin / "skills", "demo-skill") / "SKILL.md"
    set_agent_plugin_enabled(tmp_path, "demo", True)
    original = skill_file.read_text()
    skill_file.write_text(original + "changed\n")
    assert discover_agent_plugins(tmp_path)[0].enabled is False
    assert not _marker(tmp_path).exists()
    skill_file.write_text(original)
    assert discover_agent_plugins(tmp_path)[0].enabled is False


def test_activation_is_per_workspace(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    for workspace in (first, second):
        _plugin(workspace)
        set_agent_plugin_enabled(workspace, "demo", True)
    assert discover_agent_plugins(first)[0].enabled is True
    assert discover_agent_plugins(second)[0].enabled is True
    assert _marker(first) != _marker(second)


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes")
def test_plugin_data_directories_are_private(tmp_path):
    _plugin(tmp_path)
    set_agent_plugin_enabled(tmp_path, "demo", True)
    data = agent_plugins._plugin_data_dir(tmp_path, "demo", create=False)
    for path in (data, data.parent, data.parent.parent):
        assert path.stat().st_mode & 0o777 == 0o700


def test_checking_a_disabled_plugin_keeps_the_skill_cache(tmp_path):
    plugin = _plugin(tmp_path)
    _skill(plugin / "skills", "demo-skill")
    set_agent_plugin_enabled(tmp_path, "demo", True)
    _plugin(tmp_path, "other")
    assert len(enabled_agent_plugin_skill_dirs(tmp_path)) == 1
    discover_agent_plugins(tmp_path)
    assert agent_plugins._skill_cache_key(tmp_path) in agent_plugins._SKILL_CACHE


def test_a_read_check_withdraws_a_changed_package(tmp_path):
    plugin = _plugin(tmp_path)
    skill = _skill(plugin / "skills", "demo-skill")
    set_agent_plugin_enabled(tmp_path, "demo", True)
    assert enabled_agent_plugin_skill_dirs(tmp_path, requested_path=skill / "SKILL.md") == (skill.resolve(),)
    (skill / "SKILL.md").write_text("changed")
    assert enabled_agent_plugin_skill_dirs(tmp_path, requested_path=skill / "SKILL.md") == ()
    assert not _marker(tmp_path).exists()


def test_read_allowance_is_the_requested_skill_folder_only(tmp_path):
    plugin = _plugin(tmp_path)
    first = _skill(plugin / "skills", "first").resolve()
    second = _skill(plugin / "skills", "second").resolve()
    set_agent_plugin_enabled(tmp_path, "demo", True)
    assert enabled_agent_plugin_skill_dirs(tmp_path) == (first, second)
    assert enabled_agent_plugin_skill_dirs(tmp_path, requested_path=first / "SKILL.md") == (first,)


def test_read_allowance_resolves_the_requested_path(tmp_path):
    workspace = tmp_path / "real"
    plugin = _plugin(workspace)
    skill = _skill(plugin / "skills", "demo-skill").resolve()
    set_agent_plugin_enabled(workspace, "demo", True)
    alias = tmp_path / "alias"
    _symlink(alias, workspace, directory=True)
    requested = alias / "plugins" / "demo" / "skills" / "demo-skill" / "SKILL.md"
    assert enabled_agent_plugin_skill_dirs(workspace, requested_path=requested) == (skill,)


def test_skill_cache_is_per_config(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path)
    _skill(plugin / "skills", "demo-skill")
    set_agent_plugin_enabled(tmp_path, "demo", True)
    assert len(enabled_agent_plugin_skill_dirs(tmp_path)) == 1
    monkeypatch.setattr(agent_plugins, "get_config_path", lambda: tmp_path / "other-config" / "config.json")
    assert enabled_agent_plugin_skill_dirs(tmp_path) == ()


def test_plugin_folders_resolve_inside_the_workspace_and_package(tmp_path):
    # ``plugins`` and ``skills`` may be links inside their parent folder.
    real = tmp_path / "real-plugins"
    plugin = real / "demo"
    _write_json(plugin / "plugin.json", {"$schema": AGENT_PLUGIN_SCHEMA, "name": "demo"})
    _skill(plugin / "real-skills", "inner")
    _symlink(plugin / "skills", plugin / "real-skills", directory=True)
    _symlink(tmp_path / "plugins", real, directory=True)
    set_agent_plugin_enabled(tmp_path, "demo", True)
    assert [name for name, _path in enabled_agent_plugin_skills(tmp_path)] == ["inner"]


def test_a_plugin_folder_linked_outside_plugins_is_ignored(tmp_path):
    outside = tmp_path / "elsewhere" / "demo"
    _write_json(outside / "plugin.json", {"$schema": AGENT_PLUGIN_SCHEMA, "name": "demo"})
    (tmp_path / "plugins").mkdir()
    _symlink(tmp_path / "plugins" / "demo", outside, directory=True)
    assert discover_agent_plugins(tmp_path) == []


def test_plugin_skill_links(tmp_path, capsys):
    plugin = _plugin(tmp_path)
    skills = plugin / "skills"
    skills.mkdir()
    # A skill folder linked to a folder of the package outside skills/: ignored.
    _skill(plugin / "elsewhere", "stray")
    _symlink(skills / "stray", plugin / "elsewhere" / "stray", directory=True)
    # An alias of another skill folder: its SKILL.md names the target, so it is invalid.
    _skill(skills, "real")
    _symlink(skills / "alias", skills / "real", directory=True)
    # SKILL.md linked to a file inside the package: read through the link.
    (skills / "linked").mkdir()
    (plugin / "linked.md").write_text("---\nname: linked\ndescription: Linked.\n---\n")
    _symlink(skills / "linked" / "SKILL.md", plugin / "linked.md")
    # SKILL.md linked outside the package: ignored.
    (tmp_path / "outside.md").write_text("---\nname: outlinked\ndescription: Out.\n---\n")
    (skills / "outlinked").mkdir()
    _symlink(skills / "outlinked" / "SKILL.md", tmp_path / "outside.md")
    # Named so that the sorted order differs from creation order.
    _skill(skills, "a-first")

    set_agent_plugin_enabled(tmp_path, "demo", True)
    assert [name for name, _path in enabled_agent_plugin_skills(tmp_path)] == ["a-first", "linked", "real"]
    assert main(["plugins", "list", "-w", str(tmp_path)]) == 0
    assert "skills: a-first, linked, real\n" in capsys.readouterr().out


def _servers(workspace: Path, servers: object, **payload: object) -> dict:
    plugin = workspace / "plugins" / "demo"
    if not (plugin / "plugin.json").exists():
        _plugin(workspace)
    _write_json(plugin / "mcp.json", {"$schema": AGENT_PLUGIN_MCP_SCHEMA, "mcpServers": servers, **payload})
    set_agent_plugin_enabled(workspace, "demo", True)
    return agent_plugin_mcp_servers(workspace)


@pytest.mark.parametrize(
    "payload",
    [{"extra": 1}, {"$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"}],
)
def test_mcp_component_shape(tmp_path, payload):
    assert _servers(tmp_path, {"main": {"type": "stdio", "command": "echo"}}, **payload) == {}


def test_mcp_servers_must_be_an_object(tmp_path):
    assert _servers(tmp_path, [{"type": "stdio", "command": "echo"}]) == {}


@pytest.mark.parametrize("bad_name", ["", "x" * 129, "a\nb"])
def test_mcp_server_names(tmp_path, bad_name):
    servers = _servers(tmp_path, {"good": {"type": "stdio", "command": "echo"}, bad_name: {"type": "stdio", "command": "echo"}})
    assert set(servers) == {"demo"}


@pytest.mark.parametrize(
    "server",
    [
        ["echo"],
        {"type": "stdio", "command": "echo", "url": "http://example.com/mcp"},
        {"type": "sse", "command": "echo"},
        {"type": "stdio", "command": "echo", "env": {"PLUGIN_ROOT": "/elsewhere"}},
        {"type": "stdio", "command": ""},
        {"type": "stdio", "command": "./../outside.sh"},
        {"type": "stdio", "command": "./missing"},
        {"type": "stdio", "command": "./bin"},
        {"type": "stdio", "command": "python -m server"},
        {"type": "stdio", "command": "echo", "cwd": "./../"},
        {"type": "stdio", "command": "echo", "cwd": "./missing"},
        {"type": "stdio", "command": "echo", "cwd": "./file.txt"},
        {"type": "stdio", "command": "echo", "cwd": "${PLUGIN_ROOT}bin"},
        {"type": "stdio", "command": "echo", "cwd": "${PLUGIN_ROOT}/../.."},
        {"type": "stdio", "command": "echo", "cwd": "${PLUGIN_ROOT}/file.txt"},
        {"type": "stdio", "command": "echo", "cwd": "/"},
    ],
)
def test_invalid_mcp_servers_are_ignored(tmp_path, server):
    plugin = _plugin(tmp_path)
    (plugin / "bin").mkdir()
    (plugin / "file.txt").write_text("x")
    (tmp_path / "plugins" / "outside.sh").write_text("#!/bin/sh\n")
    assert _servers(tmp_path, {"main": server}) == {}


def test_mcp_server_expansion_environment_and_cwd(tmp_path):
    plugin = _plugin(tmp_path).resolve()
    (plugin / "sub").mkdir()
    servers = _servers(
        tmp_path,
        {
            "b": {
                "type": "stdio", "command": "echo",
                "args": ["${PLUGIN_DATA}/x", "${PLUGIN_ROOT}/y"],
                "env": {"A": "${PLUGIN_DATA}/z"},
                "cwd": "${PLUGIN_DATA}/state",
            },
            "a": {"type": "stdio", "command": "echo", "cwd": "./sub"},
        },
    )
    data = agent_plugins._plugin_data_dir(tmp_path, "demo", create=False)
    server = servers["demo--b"]
    assert server.args == [f"{data}/x", f"{plugin}/y"]
    assert server.env == {
        "A": f"{data}/z", "PYTHONDONTWRITEBYTECODE": "1", "PLUGIN_ROOT": str(plugin), "PLUGIN_DATA": str(data),
    }
    assert server.cwd == str(data / "state") and (data / "state").is_dir()
    if os.name == "posix":
        assert (data / "state").stat().st_mode & 0o777 == 0o700
    assert servers["demo--a"].cwd == str(plugin / "sub")
    assert discover_agent_plugins(tmp_path)[0].mcp_servers == ("a", "b")


@pytest.mark.parametrize("content", ['["not", "an", "object"]', "{not json"])
def test_bad_manifest_json_is_ignored(tmp_path, content):
    root = tmp_path / "plugins" / "demo"
    root.mkdir(parents=True)
    (root / "plugin.json").write_text(content)
    assert discover_agent_plugins(tmp_path) == []


def test_cli_review_reads_cr_line_endings_and_accepts_upper_case_yes(tmp_path, capsys, monkeypatch):
    plugin = _plugin(tmp_path)
    (plugin / "skills" / "old-mac").mkdir(parents=True)
    (plugin / "skills" / "old-mac" / "SKILL.md").write_bytes(b"---\rname: old-mac\rdescription: CR only.\r---\r")
    monkeypatch.setattr("builtins.input", lambda _text: "Y")
    assert main(["plugins", "enable", "demo", "-w", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "skills:      old-mac" in out and "ignored skills" not in out
    assert discover_agent_plugins(tmp_path)[0].enabled is True
    assert [name for name, _path in enabled_agent_plugin_skills(tmp_path)] == ["old-mac"]


@pytest.mark.parametrize("action", ["enable", "disable"])
def test_cli_reports_enable_errors_with_exit_code_1(tmp_path, capsys, monkeypatch, action):
    _plugin(tmp_path)

    def broken(*_args):
        raise RuntimeError("Agent Plugin data directory escapes its parent")

    monkeypatch.setattr("nanoclaw.cli.plugins.set_agent_plugin_enabled", broken)
    args = ["plugins", action, "demo", "-w", str(tmp_path)] + (["--yes"] if action == "enable" else [])
    assert main(args) == 1
    assert f"Could not {action} demo: Agent Plugin data directory escapes its parent" in capsys.readouterr().err


def test_cli_plugins_needs_an_action(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["plugins"])
    assert exc.value.code == 2


def _scope(project: Path, workspace: Path):
    return validate_workspace_scope_payload(
        {"project_path": str(project), "access_mode": "restricted"},
        default_workspace=workspace,
        default_restrict_to_workspace=True,
    )


async def test_restricted_relative_read_of_a_plugin_skill(tmp_path):
    workspace = tmp_path / "agent"
    project = tmp_path / "project"
    project.mkdir()
    skill = _skill(_plugin(workspace) / "skills", "demo-skill", body="Plugin text.")
    set_agent_plugin_enabled(workspace, "demo", True)
    token = bind_workspace_scope(_scope(project, workspace))
    try:
        result = await ReadFileTool(workspace, restrict_to_workspace=True).execute(
            path="../agent/plugins/demo/skills/demo-skill/SKILL.md"
        )
    finally:
        reset_workspace_scope(token)
    assert "Plugin text." in result
    assert skill.exists()


async def test_plugin_read_allowance_is_checked_only_when_restricted(tmp_path, monkeypatch):
    workspace = tmp_path / "agent"
    workspace.mkdir()
    (workspace / "notes.md").write_text("notes")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("unrestricted reads need no plugin check")

    monkeypatch.setattr(agent_plugins, "enabled_agent_plugin_skill_dirs", unexpected)
    result = await ReadFileTool(workspace).execute(path=str(workspace / "notes.md"))
    assert "notes" in result


async def test_a_broken_plugin_data_folder_does_not_break_restricted_reads(tmp_path, monkeypatch):
    workspace = tmp_path / "agent"
    project = tmp_path / "project"
    project.mkdir()
    workspace.mkdir()
    (project / "notes.md").write_text("project notes")

    def broken(*_args, **_kwargs):
        raise RuntimeError("Agent Plugin data directory escapes its parent")

    monkeypatch.setattr(agent_plugins, "enabled_agent_plugin_skill_dirs", broken)
    token = bind_workspace_scope(_scope(project, workspace))
    try:
        result = await ReadFileTool(workspace, restrict_to_workspace=True).execute(path=str(project / "notes.md"))
    finally:
        reset_workspace_scope(token)
    assert "project notes" in result


# --- area 3: the ``my`` tool ----------------------------------------------------------------------


def _mock_loop(**overrides):
    loop = MagicMock()
    loop.model = "anthropic/claude-sonnet-4-6"
    loop.provider = FakeProvider()
    loop.max_iterations = 40
    loop.context_window_tokens = 65_536
    loop.workspace = Path("/tmp/workspace")
    loop.restrict_to_workspace = False
    loop.exec_config = ExecToolConfig()
    loop.provider_retry_mode = "standard"
    loop.max_tool_result_chars = 16000
    loop.model_preset = None
    loop.model_presets = {}
    loop.set_runtime_model.side_effect = lambda value: setattr(loop, "model", value)
    loop.set_runtime_max_iterations.side_effect = lambda value: setattr(loop, "max_iterations", value)
    loop.set_runtime_context_window.side_effect = lambda value: setattr(loop, "context_window_tokens", value)
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    loop.tools = registry
    for key, value in overrides.items():
        setattr(loop, key, value)
    return loop


def _tool(loop=None, **kwargs) -> MyTool:
    return MyTool(runtime_control=AgentRuntimeControl(loop or _mock_loop()), **kwargs)


@pytest.mark.parametrize("key", ["auth", "API_KEY", "Token", "__hidden"])
async def test_protected_top_level_keys_cannot_be_set(key):
    tool = _tool()
    result = await tool.execute(action="set", key=key, value="x")
    assert result == f"Error: '{key}' is protected and cannot be modified"
    assert tool._runtime_control.snapshot().scratchpad == {}


@pytest.mark.parametrize("key", ["tool_names", "exec_config", "web_config", "model_presets", "workspace_sandbox"])
async def test_read_only_keys_report_read_only_even_for_secret_leaves(key):
    result = await _tool().execute(action="set", key=f"{key}.token", value="x")
    assert result == f"Error: '{key}.token' is read-only and cannot be modified"


@pytest.mark.parametrize(
    ("key", "value", "ok"),
    [("max_iterations", 100, True), ("max_iterations", 101, False),
     ("context_window_tokens", 1_000_000, True), ("context_window_tokens", 1_000_001, False),
     ("model", "", False)],
)
async def test_restricted_value_limits(key, value, ok):
    loop = _mock_loop()
    before = getattr(loop, key)
    result = await _tool(loop).execute(action="set", key=key, value=value)
    assert result.startswith("Set ") is ok
    assert getattr(loop, key) == (value if ok else before)
    if key == "model":
        assert result == "Error: 'model' must be at least 1 characters"


async def test_a_deep_copy_keeps_read_only_mode():
    tool = copy.deepcopy(_tool(modify_allowed=False))
    assert await tool.execute(action="set", key="max_iterations", value=5) == (
        "Error: set is disabled (tools.my.allow_set is false)"
    )


def test_audit_prefers_the_session_key():
    ctx = RequestContext(channel="telegram", chat_id="42", session_key="telegram:42:topic")
    with patch("nanoclaw.agent.tools.self.logger.info") as info, request_context(ctx):
        _tool()._audit("modify", "x")
    info.assert_called_once_with("self.{} | {} | session:{}", "modify", "x", "telegram:42:topic")


async def test_check_path_guards():
    loop = _mock_loop(model_presets={"TOKEN": ModelPresetConfig(model="m")})
    tool = _tool(loop)
    assert await tool.execute(action="check", key="model_presets.__class__") == (
        "Error: '__class__' is not accessible"
    )
    assert await tool.execute(action="check", key="model_presets.TOKEN") == "Error: 'TOKEN' is not accessible"
    assert await tool.execute(action="check", key="max_iterations.x") == "Error: 'x' not found"
    assert "max_iterations: 40" in await tool.execute(action="check", key="")


async def test_set_dotted_path_guards():
    tool = _tool()
    assert await tool.execute(action="set", key="missing.leaf", value=1) == "Error: 'missing' not found in mapping"
    assert await tool.execute(action="set", key="workspace.leaf", value=1) == (
        "Error: 'workspace.leaf' is read-only and cannot be modified"
    )
    assert tool._runtime_control.snapshot().scratchpad == {}


def test_value_formatting():
    fmt = MyTool._format_value
    assert fmt(MappingProxyType({}), "k") == "k: {}"
    assert fmt(MappingProxyType({})) == "{}"
    five = {f"k{i}": i for i in range(5)}
    assert fmt(five, "m") == f"m: {five!r}"
    long_values = {f"k{i}": "v" * 50 for i in range(5)}
    assert fmt(long_values, "m") == "m: {k0, k1, k2, k3, k4}"
    six = {f"k{i}": i for i in range(6)}
    assert fmt(six, "m") == "m: {k0, k1, k2, k3, k4, k5}"
    many = {f"k{i}": i for i in range(16)}
    assert fmt(many, "m") == "m: {" + ", ".join(f"k{i}" for i in range(15)) + ", ...}"
    assert fmt(list(range(20)), "l") == f"l: {list(range(20))!r}"
    assert fmt(list(range(21)), "l") == "l: [21 items]"


async def test_model_preset_name_is_trimmed(tmp_path):
    loop = AgentLoop(
        FakeProvider(), tmp_path, model="test-model", my_config=MyToolConfig(allow_set=True),
        model_presets={"fast": ModelPresetConfig(model="fast-model")},
    )
    result = await loop.tools.execute("my", {"action": "set", "key": "model_preset", "value": "  fast  "})
    assert result.startswith("Set model_preset = 'fast' (was None)")


async def test_max_tool_result_chars_is_applied():
    loop = _mock_loop()
    assert await _tool(loop).execute(action="set", key="max_tool_result_chars", value=5000) == (
        "Set max_tool_result_chars = 5000 (was 16000)"
    )
    assert loop.max_tool_result_chars == 5000


def test_json_safe_values():
    def nested(levels: int) -> object:
        value: object = 1
        for _ in range(levels):
            value = [value]
        return value

    assert MyTool._validate_json_safe(nested(10)) is None
    assert MyTool._validate_json_safe(nested(11)) is not None
    assert MyTool._validate_json_safe([1, {1, 2}]) == "list[1] contains unsupported type set"
    assert MyTool._validate_json_safe((1, 2)) == "unsupported type tuple"


def test_scratchpad_snapshot_is_a_deep_copy():
    control = AgentRuntimeControl(_mock_loop())
    control.set_scratchpad("k", {"a": [1]}, max_keys=64)
    snapshot = control.snapshot()
    snapshot.scratchpad["k"]["a"].append(2)  # type: ignore[index,union-attr]
    snapshot.scratchpad["new"] = 1
    assert control.snapshot().scratchpad == {"k": {"a": [1]}}


def test_web_and_exec_config_snapshot():
    fetch_only = ToolRegistry()
    fetch_only.register(WebFetchTool())
    assert AgentRuntimeControl(_mock_loop(tools=fetch_only)).snapshot().web_config["enable"] is True

    search = ToolRegistry()
    search.register(WebSearchTool(max_results=3, timeout=12.0))
    exec_config = ExecToolConfig(
        max_runtime=100, idle_timeout=7, sandbox="bwrap", sandbox_ro_binds=["/ro"], sandbox_rw_binds=["/rw"],
    )
    presets = {"fast": ModelPresetConfig(model="m", context_window_tokens=8192)}
    snapshot = AgentRuntimeControl(_mock_loop(tools=search, exec_config=exec_config, model_presets=presets)).snapshot()
    timeout = snapshot.web_config["search"]["timeout"]  # type: ignore[index]
    assert snapshot.web_config["search"] == {  # type: ignore[comparison-overlap]
        "provider": "duckduckgo", "base_url": "", "max_results": 3, "timeout": 12,
    }
    assert type(timeout) is int
    assert snapshot.exec_config["sandbox"] == "bwrap"
    assert snapshot.exec_config["timeout"] == 100
    assert snapshot.exec_config["sandbox_ro_binds"] == ["/ro"]
    assert snapshot.exec_config["sandbox_rw_binds"] == ["/rw"]
    assert snapshot.model_presets["fast"]["context_window_tokens"] == 8192


async def test_the_turn_passes_the_sender_to_the_my_tool(tmp_path):
    fake = FakeProvider([
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest("c1", "my", {"action": "check", "key": "request.sender_id"}),
        ], finish_reason="tool_calls"),
        "done",
    ])
    loop = AgentLoop(fake, tmp_path, model="test-model")
    await loop.process_direct("who am I", session_key="cli:s")
    tool_message = fake.calls[1].messages[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == "request.sender_id: 'user'"
