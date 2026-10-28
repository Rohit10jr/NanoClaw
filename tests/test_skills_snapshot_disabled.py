"""Skill snapshots, disabled skills and the web UI skills API (layer 25, task 25.1).

Ported from nanobot c4dac64 tests/agent/test_skills_loader.py (the disabled,
explicit-reference and snapshot cases), tests/command/test_skill_command.py
(``test_skill_command_excludes_disabled``) and tests/webui/test_skills_api.py
(all but the plugin and delete cases: NanoClaw has no ``skill.delete``),
plus the NanoClaw wiring: config, CLI, loop, subagents and the REST routes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from nanoclaw.__main__ import main
from nanoclaw.agent import skills as agent_skills
from nanoclaw.agent import skill_metadata
from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.skills import SkillsLoader
from nanoclaw.bus.events import InboundMessage
from nanoclaw.command.builtin import cmd_skill
from nanoclaw.command.router import CommandContext
from nanoclaw.config.loader import load_config, save_config
from nanoclaw.config.schema import Config
from nanoclaw.providers import FakeProvider
from nanoclaw.session import SessionManager
from nanoclaw.webui.context import WebUIContext
from nanoclaw.webui.rest_api import register_api_fallback, register_rest_routes
from nanoclaw.webui.skills_api import (
    SkillManagementError,
    set_webui_skill_enabled,
    webui_skill_detail_payload,
    webui_skills_payload,
)


def _write_skill(base: Path, name: str, *, metadata_json: dict | None = None, body: str = "# Skill\n") -> Path:
    """``base / name / SKILL.md`` with optional nanobot metadata JSON (nanobot's helper)."""
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    lines = ["---"]
    if metadata_json is not None:
        payload = json.dumps({"nanobot": metadata_json}, separators=(",", ":"))
        lines.append(f"metadata: {payload}")
    lines.extend(["---", "", body])
    path = skill_dir / "SKILL.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _named_skill(workspace: Path, name: str, *, metadata: str = "") -> Path:
    """nanobot tests/webui/test_skills_api.py ``_write_skill``."""
    directory = workspace / "skills" / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} description.\n{metadata}---\n",
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "ws"
    (path / "skills").mkdir(parents=True)
    return path


@pytest.fixture
def builtin(tmp_path: Path) -> Path:
    path = tmp_path / "builtin"
    path.mkdir()
    return path


# --- skill_metadata ---------------------------------------------------------------------------


def test_skill_metadata_names_are_reexported_from_skills():
    assert agent_skills.BUILTIN_SKILLS_DIR is skill_metadata.BUILTIN_SKILLS_DIR
    assert agent_skills.parse_skill_metadata is skill_metadata.parse_skill_metadata
    assert agent_skills.valid_skill_metadata is skill_metadata.valid_skill_metadata
    assert skill_metadata.SKILL_NAME.fullmatch("my-skill") and not skill_metadata.SKILL_NAME.fullmatch("a--b")
    assert skill_metadata.SKILL_FRONTMATTER.match("---\nname: a\n---\nbody")


# --- disabled skills (nanobot test_skills_loader.py) -----------------------------------------


def test_disabled_skills_excluded_from_list(workspace, builtin):
    _write_skill(workspace / "skills", "alpha", body="# Alpha")
    beta_path = _write_skill(workspace / "skills", "beta", body="# Beta")
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin, disabled_skills={"alpha"})
    entries = loader.list_skills(filter_unavailable=False)
    assert len(entries) == 1
    assert entries[0]["name"] == "beta"
    assert entries[0]["path"] == str(beta_path)
    assert loader.load_skill("alpha") is None


def test_disabled_skills_empty_set_no_effect(workspace, builtin):
    _write_skill(workspace / "skills", "alpha", body="# Alpha")
    _write_skill(workspace / "skills", "beta", body="# Beta")
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin, disabled_skills=set())
    assert len(loader.list_skills(filter_unavailable=False)) == 2


def test_disabled_skills_excluded_from_build_skills_summary(workspace, builtin):
    _write_skill(workspace / "skills", "alpha", body="# Alpha")
    _write_skill(workspace / "skills", "beta", body="# Beta")
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin, disabled_skills={"alpha"})
    summary = loader.build_skills_summary()
    assert "alpha" not in summary
    assert "beta" in summary


def test_disabled_skills_excluded_from_get_always_skills(workspace, builtin):
    _write_skill(workspace / "skills", "alpha", metadata_json={"always": True}, body="# Alpha")
    _write_skill(workspace / "skills", "beta", metadata_json={"always": True}, body="# Beta")
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin, disabled_skills={"alpha"})
    always = loader.get_always_skills()
    assert "alpha" not in always
    assert "beta" in always


def test_disabling_a_workspace_override_does_not_bring_back_the_builtin(workspace, builtin):
    # The name is dropped after discovery, so the shadowed built-in stays hidden.
    _write_skill(workspace / "skills", "weather", body="# Mine")
    _write_skill(builtin, "weather", body="# Built-in")
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin, disabled_skills={"weather"})
    assert loader.list_skills(filter_unavailable=False) == []
    assert loader.load_skill("weather") is None


def test_explicit_skill_references_resolve_available_enabled_names_in_order(workspace, builtin, monkeypatch):
    _write_skill(workspace / "skills", "alpha", body="# Alpha")
    _write_skill(workspace / "skills", "beta", body="# Beta")
    _write_skill(
        workspace / "skills", "blocked", metadata_json={"requires": {"env": ["MISSING_SKILL_TEST_ENV"]}}, body="# Blocked"
    )
    monkeypatch.delenv("MISSING_SKILL_TEST_ENV", raising=False)
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin, disabled_skills={"beta"})
    invoked = loader.get_explicitly_invoked_skills("Use $alpha, then $unknown, $alpha again, $beta, and $blocked.")
    assert invoked == ["alpha"]
    assert loader.build_explicit_skill_runtime_context("Use $beta") is None


def test_multiple_explicit_skills_share_one_ordered_runtime_context(workspace, builtin):
    _write_skill(workspace / "skills", "alpha", body="Alpha instructions")
    _write_skill(workspace / "skills", "beta", body="Beta instructions")
    _write_skill(workspace / "skills", "always", metadata_json={"always": True}, body="Always instructions")
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    context = loader.build_explicit_skill_runtime_context("Use $beta, then $alpha, $beta again, and $always.")
    assert context is not None
    assert context.source == "explicit_skills"
    assert context.content.count("### Skill: beta") == 1
    assert context.content.count("### Skill: alpha") == 1
    assert context.content.index("### Skill: beta") < context.content.index("### Skill: alpha")
    assert "### Skill: always" not in context.content


def test_plain_message_skips_explicit_skill_discovery(tmp_path, monkeypatch):
    loader = SkillsLoader(tmp_path)

    def unexpected_discovery():
        pytest.fail("ordinary chat must not scan skills for explicit references")

    monkeypatch.setattr(loader, "_discover_skills", unexpected_discovery)
    assert loader.build_explicit_skill_runtime_context("Explain this project") is None
    assert loader.get_explicitly_invoked_skills("") == []


# --- snapshot -----------------------------------------------------------------------------------


@pytest.mark.parametrize("explicit", [False, True])
def test_prompt_reads_skills_once_and_refreshes_on_next_build(tmp_path, monkeypatch, explicit):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    skills = workspace / "skills"
    alpha = _write_skill(skills, "alpha", metadata_json={"always": True}, body="First instructions")
    beta = _write_skill(skills, "beta", metadata_json={"requires": {"bins": ["prompt-test-cli"]}})
    builder = ContextBuilder(workspace)
    builder.skills = SkillsLoader(workspace, builtin_skills_dir=builtin)
    documents = {alpha, beta}
    reads: list[Path] = []
    parses: list[str] = []
    original_read = Path.read_text
    original_parse = agent_skills.parse_skill_metadata
    original_discover = builder.skills._discover_skills
    discoveries = 0
    command_available = False
    checks: list[str] = []

    def discover():
        nonlocal discoveries
        discoveries += 1
        return original_discover()

    def read(path, *args, **kwargs):
        if path in documents:
            reads.append(path)
        return original_read(path, *args, **kwargs)

    def parse(content):
        parses.append(content)
        return original_parse(content)

    def which(command):
        checks.append(command)
        return "installed-command" if command_available else None

    def build_prompt():
        if explicit:
            messages = builder.build_messages(history=[], current_message="Use $alpha and $beta")
            return "\n".join(message["content"] for message in messages)
        return builder.build_system_prompt()

    monkeypatch.setattr(builder.skills, "_discover_skills", discover)
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(agent_skills, "parse_skill_metadata", parse)
    monkeypatch.setattr("nanoclaw.agent.skills.shutil.which", which)
    first = build_prompt()
    assert "First instructions" in first
    assert "**beta**" not in first
    assert discoveries == 1
    assert all(reads.count(path) == 1 for path in documents)
    assert len(parses) == len(documents)
    assert checks == ["prompt-test-cli"]

    alpha.write_text(original_read(alpha).replace("First instructions", "Updated instructions"), encoding="utf-8")
    beta.write_text(original_read(beta).replace("---\n", "---\ndescription: Updated description\n", 1), encoding="utf-8")
    command_available = True
    documents.add(_write_skill(skills, "gamma", body="New skill"))
    reads.clear()
    parses.clear()
    checks.clear()
    second = build_prompt()
    assert "Updated instructions" in second and "First instructions" not in second
    assert "Updated description" in second
    assert "**beta**" in second
    assert "**gamma**" in second
    assert discoveries == 2
    assert all(reads.count(path) == 1 for path in documents)
    assert len(parses) == len(documents)
    assert checks == ["prompt-test-cli"]


def test_snapshot_nests_and_is_cleared_on_exit(workspace, builtin):
    path = _write_skill(workspace / "skills", "alpha", body="Old")
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    with loader.snapshot():
        assert "Old" in (loader.load_skill("alpha") or "")
        path.write_text(path.read_text().replace("Old", "New"))
        with loader.snapshot():
            assert "Old" in (loader.load_skill("alpha") or "")
        assert "Old" in (loader.load_skill("alpha") or "")
    assert loader._snapshot.get() is None
    assert "New" in (loader.load_skill("alpha") or "")


def test_without_snapshot_each_call_reads_fresh(workspace, builtin):
    path = _write_skill(workspace / "skills", "alpha", body="Old")
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    assert "Old" in (loader.load_skill("alpha") or "")
    path.write_text(path.read_text().replace("Old", "New"))
    assert "New" in (loader.load_skill("alpha") or "")


async def test_snapshot_is_per_task(workspace, builtin):
    # A ContextVar: a snapshot opened in one task is not seen by another.
    import asyncio

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    opened = asyncio.Event()
    release = asyncio.Event()

    async def holder():
        with loader.snapshot():
            opened.set()
            await release.wait()

    task = asyncio.create_task(holder())
    await opened.wait()
    assert loader._snapshot.get() is None
    release.set()
    await task


def test_get_skill_requirements(workspace, builtin, monkeypatch):
    monkeypatch.setenv("SKILL_REQ_SET", "1")
    monkeypatch.delenv("SKILL_REQ_UNSET", raising=False)
    _write_skill(
        workspace / "skills",
        "needs",
        metadata_json={"requires": {"bins": ["python3", "definitely-missing-bin-x"], "env": ["SKILL_REQ_SET", "SKILL_REQ_UNSET"]}},
    )
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    assert loader.get_skill_requirements("needs") == {
        "bins": ["python3", "definitely-missing-bin-x"],
        "env": ["SKILL_REQ_SET", "SKILL_REQ_UNSET"],
        "missing_bins": ["definitely-missing-bin-x"],
        "missing_env": ["SKILL_REQ_UNSET"],
    }


# --- wiring: config, context, loop, subagents, /skill, CLI ---------------------------------------


def test_config_field_name_and_default():
    assert Config().agents.defaults.disabled_skills == []
    config = Config.model_validate({"agents": {"defaults": {"disabledSkills": ["weather"]}}})
    assert config.agents.defaults.disabled_skills == ["weather"]
    assert config.model_dump(by_alias=True)["agents"]["defaults"]["disabledSkills"] == ["weather"]


def test_context_builder_and_loop_pass_disabled_skills(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert ContextBuilder(workspace).skills.disabled_skills == set()
    builder = ContextBuilder(workspace, disabled_skills=["weather"])
    assert builder.skills.disabled_skills == {"weather"}
    assert "**weather**" not in builder.build_system_prompt()
    assert "**weather**" in ContextBuilder(workspace).build_system_prompt()
    loop = AgentLoop(FakeProvider([]), workspace, sessions=SessionManager(workspace), disabled_skills=["weather"])
    assert loop.context.skills.disabled_skills == {"weather"}
    assert loop.subagents.disabled_skills == {"weather"}
    assert "**weather**" not in loop.subagents._build_subagent_prompt()
    loop.subagents.disabled_skills = set()
    assert "**weather**" in loop.subagents._build_subagent_prompt()


async def test_skill_command_excludes_disabled(tmp_path):
    ws_skills = tmp_path / "skills"
    ws_skills.mkdir()
    (ws_skills / "alpha").mkdir()
    (ws_skills / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: Alpha skill\n---\n")
    (ws_skills / "beta").mkdir()
    (ws_skills / "beta" / "SKILL.md").write_text("---\nname: beta\ndescription: Beta skill\n---\n")
    loop = AgentLoop(FakeProvider([]), tmp_path, sessions=SessionManager(tmp_path))
    loop.context.skills.disabled_skills = {"alpha"}
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/skill")
    out = await cmd_skill(CommandContext(msg=msg, session=None, key=msg.session_key, raw="/skill", loop=loop))
    assert "alpha" not in out.content
    assert "**beta** — Beta skill" in out.content


def test_cli_skills_marks_disabled_and_prompt_hides_them(capsys, tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"agents": {"defaults": {"disabledSkills": ["weather", "summarize"]}}}))
    monkeypatch.setenv("NANOCLAW_CONFIG", str(config_path))
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: None if name == "summarize" else f"/bin/{name}")
    assert main(["skills", "-w", str(workspace)]) == 0
    out = capsys.readouterr().out
    assert "- **weather** — Get current weather and forecasts. (disabled)\n" in out
    assert "- **summarize** — " in out and "(disabled; unavailable: CLI: summarize)\n" in out
    assert "- **memory** — " in out and "**memory** — Search" in out
    assert main(["prompt", "-w", str(workspace)]) == 0
    prompt = capsys.readouterr().out
    assert "**weather**" not in prompt
    assert "**memory**" in prompt


def test_make_loop_reads_disabled_skills_from_config(tmp_path):
    from nanoclaw.__main__ import _make_loop

    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = Config.model_validate({"agents": {"defaults": {"disabledSkills": ["tmux"]}}})
    loop = _make_loop(FakeProvider([]), workspace, config=config)
    assert loop.context.skills.disabled_skills == {"tmux"}
    assert loop.subagents.disabled_skills == {"tmux"}


# --- cron built-in skill ------------------------------------------------------------------------


def test_cron_skill_is_bundled_with_the_product_name_adapted():
    text = (agent_skills.BUILTIN_SKILLS_DIR / "cron" / "SKILL.md").read_text(encoding="utf-8")
    meta = agent_skills.parse_skill_metadata(text)
    assert meta == {"name": "cron", "description": "Schedule reminders and recurring tasks."}
    assert "nanoclaw sends the agent's final reply" in text
    # Deliberate difference: no `message` tool or WebUI chat change in NanoClaw.
    assert "`message` tool" not in text and "WebUI" not in text
    assert "cannot send to other chats or attach files" in text
    assert "cron(action=\"history\", job_id=\"...\", limit=5)" in text


def test_cron_skill_is_in_the_prompt(tmp_path):
    prompt = ContextBuilder(tmp_path).build_system_prompt()
    assert "- **cron** — Schedule reminders and recurring tasks.  `cron/SKILL.md`" in prompt


# --- web UI skills API (nanobot tests/webui/test_skills_api.py) ----------------------------------


def test_disabled_skills_remain_visible_and_loadable(tmp_path):
    _named_skill(tmp_path, "custom-skill")
    payload = webui_skills_payload(tmp_path, disabled_skills={"custom-skill"})
    skill = next(item for item in payload["skills"] if item["name"] == "custom-skill")
    assert skill["enabled"] is False
    assert skill["deletable"] is True
    detail = webui_skill_detail_payload(tmp_path, "custom-skill", disabled_skills={"custom-skill"})
    assert detail is not None
    assert detail["enabled"] is False
    assert "custom-skill description" in detail["raw_markdown"]


def test_skill_detail_exposes_copyable_install_commands(tmp_path):
    _named_skill(
        tmp_path,
        "custom-skill",
        metadata=(
            'metadata: {"nanobot":{"requires":{"bins":["demo"]},'
            '"install":[{"id":"brew","kind":"brew","formula":"acme/demo",'
            '"label":"Install demo"}]}}\n'
        ),
    )
    detail = webui_skill_detail_payload(tmp_path, "custom-skill")
    assert detail is not None
    assert detail["install_options"] == [
        {"id": "brew", "kind": "brew", "label": "Install demo", "command": "brew install acme/demo"}
    ]
    assert detail["requirements"]["bins"] == ["demo"]


def test_skill_detail_install_options_shapes(tmp_path):
    _named_skill(
        tmp_path,
        "custom-skill",
        metadata=(
            'metadata: {"nanobot":{"install":[{"kind":"apt","package":"a b"},{"kind":"pip","package":"x"},'
            '"junk",{"kind":"brew"}]}}\n'
        ),
    )
    detail = webui_skill_detail_payload(tmp_path, "custom-skill")
    assert detail is not None
    assert detail["install_options"] == [
        {"id": "apt", "kind": "apt", "label": "Install with apt", "command": "sudo apt-get install -y 'a b'"}
    ]
    assert webui_skill_detail_payload(tmp_path, "missing") is None


def test_set_webui_skill_enabled_persists_and_updates_runtime(tmp_path, monkeypatch):
    # nanobot's test stubs load/save_config; NanoClaw edits the raw file, so
    # this one watches the file instead.
    _named_skill(tmp_path, "custom-skill")
    path = tmp_path / "config.json"
    monkeypatch.setattr("nanoclaw.config.loader._current_config_path", path)
    disabled: set[str] = set()
    action = set_webui_skill_enabled(tmp_path, "custom-skill", enabled=False, disabled_skills=disabled)
    assert action == {"name": "custom-skill", "enabled": False, "deleted": False}
    assert json.loads(path.read_text()) == {"agents": {"defaults": {"disabledSkills": ["custom-skill"]}}}
    assert load_config(path).agents.defaults.disabled_skills == ["custom-skill"]
    assert disabled == {"custom-skill"}
    # No change: nothing is written.
    path.write_text(path.read_text() + "\n")
    before = path.read_text()
    set_webui_skill_enabled(tmp_path, "custom-skill", enabled=False, disabled_skills=disabled)
    assert path.read_text() == before


def test_skill_toggle_keeps_unknown_keys_order_and_env_references(tmp_path):
    # Deliberate difference: nanobot re-saves through its schema, dropping
    # keys it does not model; NanoClaw changes only disabledSkills.
    _named_skill(tmp_path, "custom-skill")
    path = tmp_path / "config.json"
    original = {
        "zetaNanobotOnly": {"keep": [1, 2, {"deep": "x"}]},
        "providers": {"openai": {"apiKey": "${SOME_KEY}"}},
        "agents": {"defaults": {"unknownKnob": True, "model": "openai/gpt-x"}},
        "channels": {"slack": {"enabled": False, "botToken": "${SLACK_TOKEN}"}},
    }
    path.write_text(json.dumps(original, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    set_webui_skill_enabled(tmp_path, "custom-skill", enabled=False, disabled_skills=set(), config_path=path)
    expected = json.loads(json.dumps(original))
    expected["agents"]["defaults"]["disabledSkills"] = ["custom-skill"]
    assert path.read_text(encoding="utf-8") == json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    assert list(json.loads(path.read_text())) == list(original)  # key order kept
    set_webui_skill_enabled(tmp_path, "custom-skill", enabled=True, disabled_skills=set(), config_path=path)
    expected["agents"]["defaults"]["disabledSkills"] = []
    assert path.read_text(encoding="utf-8") == json.dumps(expected, indent=2, ensure_ascii=False) + "\n"


def test_skill_toggle_keeps_a_snake_case_key(tmp_path):
    _named_skill(tmp_path, "custom-skill")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"agents": {"defaults": {"disabled_skills": ["weather"]}}}))
    set_webui_skill_enabled(tmp_path, "custom-skill", enabled=False, disabled_skills=set(), config_path=path)
    assert json.loads(path.read_text()) == {"agents": {"defaults": {"disabled_skills": ["custom-skill", "weather"]}}}


def test_skill_toggle_adds_the_path_when_missing(tmp_path):
    _named_skill(tmp_path, "custom-skill")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"other": 1}))
    set_webui_skill_enabled(tmp_path, "custom-skill", enabled=False, disabled_skills=set(), config_path=path)
    assert json.loads(path.read_text()) == {"other": 1, "agents": {"defaults": {"disabledSkills": ["custom-skill"]}}}


def test_skill_state_uses_explicit_gateway_config(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _named_skill(workspace, "custom-skill")
    default_path = tmp_path / "default.json"
    gateway_path = tmp_path / "gateway.json"
    save_config(Config(), default_path)
    save_config(Config(), gateway_path)
    monkeypatch.setattr("nanoclaw.config.loader._current_config_path", default_path)
    set_webui_skill_enabled(workspace, "custom-skill", enabled=False, disabled_skills=set(), config_path=gateway_path)
    assert load_config(default_path).agents.defaults.disabled_skills == []
    assert load_config(gateway_path).agents.defaults.disabled_skills == ["custom-skill"]
    set_webui_skill_enabled(workspace, "custom-skill", enabled=True, disabled_skills=set(), config_path=gateway_path)
    assert load_config(gateway_path).agents.defaults.disabled_skills == []


@pytest.mark.parametrize(("name", "status"), [("", 400), ("a/b", 400), ("a\\b", 400), ("missing", 404)])
def test_set_webui_skill_enabled_rejects_bad_names(tmp_path, name, status):
    with pytest.raises(SkillManagementError) as err:
        set_webui_skill_enabled(tmp_path, name, enabled=False, disabled_skills=set(), config_path=tmp_path / "c.json")
    assert err.value.status == status
    assert not (tmp_path / "c.json").exists()


def test_set_webui_skill_enabled_keeps_env_references(tmp_path):
    _named_skill(tmp_path, "custom-skill")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"providers": {"openai": {"apiKey": "${SOME_KEY}"}}}))
    set_webui_skill_enabled(tmp_path, "custom-skill", enabled=False, disabled_skills=set(), config_path=path)
    data = json.loads(path.read_text())
    assert data["providers"]["openai"]["apiKey"] == "${SOME_KEY}"
    assert data["agents"]["defaults"]["disabledSkills"] == ["custom-skill"]


# --- REST routes and the skill.update mutation ---------------------------------------------------

TOKEN = "nbwt_test"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def rest_ctx(tmp_path, monkeypatch) -> WebUIContext:
    workspace = tmp_path / "ws"
    _named_skill(workspace, "custom-skill")
    config_path = tmp_path / "config.json"
    monkeypatch.setattr("nanoclaw.config.loader._current_config_path", config_path)
    config = Config.model_validate({"agents": {"defaults": {"disabledSkills": ["weather"]}}})
    save_config(config, config_path)
    loop = AgentLoop(FakeProvider([]), workspace, sessions=SessionManager(workspace), disabled_skills=["weather"])
    return WebUIContext(
        config=config,
        sessions=loop.sessions,
        history=SimpleNamespace(),
        loop=loop,
        workspace=workspace,
        check_auth=lambda request: request.headers.get("Authorization") == f"Bearer {TOKEN}",
    )


@pytest.fixture
async def rest(rest_ctx):
    app = web.Application()
    register_rest_routes(app, rest_ctx)
    register_api_fallback(app)
    async with TestClient(TestServer(app)) as client:
        yield client


async def test_rest_skills_marks_configured_disabled_skills(rest):
    skills = {s["name"]: s for s in (await (await rest.get("/api/webui/skills", headers=AUTH)).json())["skills"]}
    assert skills["weather"]["enabled"] is False
    assert skills["custom-skill"]["enabled"] is True
    assert list(skills)[0] == "custom-skill"  # workspace skills first


async def test_rest_skill_detail(rest):
    assert (await rest.get("/api/webui/skills/weather")).status == 401
    resp = await rest.get("/api/webui/skills/weather", headers=AUTH)
    assert resp.status == 200
    body = await resp.json()
    assert body["name"] == "weather" and body["enabled"] is False and body["source"] == "builtin"
    assert body["requirements"]["bins"] == ["curl"]
    assert body["raw_markdown"].startswith("---\nname: weather")
    assert str(Path.home()) not in json.dumps({k: v for k, v in body.items() if k != "raw_markdown"})
    assert (await rest.get("/api/webui/skills/nope", headers=AUTH)).status == 404
    assert (await rest.get("/api/webui/skills/a%5Cb", headers=AUTH)).status == 400


async def test_rest_skill_update_path_needs_the_socket(rest):
    resp = await rest.post("/api/webui/skills/update", headers=AUTH)
    assert resp.status == 405


async def test_skill_update_mutation_saves_and_applies(rest, rest_ctx):
    update = rest_ctx.mutations["skill.update"]
    response = await update({"name": "custom-skill", "enabled": False})
    assert response.status == 200
    body = json.loads(response.text)
    assert body["last_action"] == {"name": "custom-skill", "enabled": False, "deleted": False}
    states = {s["name"]: s["enabled"] for s in body["skills"]}
    assert states["custom-skill"] is False and states["weather"] is False
    assert load_config().agents.defaults.disabled_skills == ["custom-skill", "weather"]
    assert rest_ctx.config.agents.defaults.disabled_skills == ["custom-skill", "weather"]
    assert rest_ctx.loop.context.skills.disabled_skills == {"custom-skill", "weather"}
    assert rest_ctx.loop.subagents.disabled_skills == {"custom-skill", "weather"}
    assert "custom-skill" not in rest_ctx.loop.context.build_system_prompt()

    response = await update({"name": "weather", "enabled": True})
    assert response.status == 200
    assert load_config().agents.defaults.disabled_skills == ["custom-skill"]
    assert "**weather**" in rest_ctx.loop.context.build_system_prompt()


async def test_skill_update_mutation_errors(rest, rest_ctx):
    update = rest_ctx.mutations["skill.update"]
    response = await update({"name": "custom-skill", "enabled": "maybe"})
    assert response.status == 400 and response.text == "enabled must be true or false"
    response = await update({"name": "custom-skill"})
    assert response.status == 400
    response = await update({"name": "missing", "enabled": True})
    assert response.status == 404 and response.text == "skill not found"
    response = await update({"name": "a/b", "enabled": True})
    assert response.status == 400 and response.text == "invalid skill name"
    assert load_config().agents.defaults.disabled_skills == ["weather"]


async def test_skill_update_mutation_without_a_loop(rest, rest_ctx):
    rest_ctx.loop = None
    response = await rest_ctx.mutations["skill.update"]({"name": "custom-skill", "enabled": "false"})
    assert response.status == 200
    assert rest_ctx.config.agents.defaults.disabled_skills == ["custom-skill", "weather"]
