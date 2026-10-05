"""Skills (nanobot agent/skills.py) and how the agent uses them."""

import pytest

from nanoclaw.__main__ import main
from nanoclaw.agent.context import SECTION_SEPARATOR, ContextBuilder
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.memory import MemoryStore
from nanoclaw.agent.skills import BUILTIN_SKILLS_DIR, SkillsLoader, parse_skill_metadata, valid_skill_metadata
from nanoclaw.agent.tools import ToolRegistry
from nanoclaw.agent.tools.filesystem import ReadFileTool
from nanoclaw.providers import FakeProvider
from nanoclaw.runtime_context import RuntimeContextBlock
from nanoclaw.session import SessionManager
from nanoclaw.utils.templates import sync_workspace_templates


def write_skill(root, name, description="Does things.", body="Step 1.", extra=""):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n{body}\n")
    return folder / "SKILL.md"


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    return path


@pytest.fixture
def builtin(tmp_path):
    path = tmp_path / "builtin"
    path.mkdir()
    return path


def loader(ws, builtin):
    return SkillsLoader(ws, builtin_skills_dir=builtin)


# --- frontmatter ----------------------------------------------------------------------------


def test_parse_skill_metadata():
    assert parse_skill_metadata("---\nname: a\ndescription: b\n---\nbody") == {"name": "a", "description": "b"}
    assert parse_skill_metadata("---\r\nname: a\r\n---\r\nbody") == {"name": "a"}
    assert parse_skill_metadata("no frontmatter") is None
    assert parse_skill_metadata("---\nname: [unclosed\n---\n") is None
    assert parse_skill_metadata("---\n- a list\n---\n") is None


@pytest.mark.parametrize(
    ("meta", "name", "ok"),
    [
        ({"name": "my-skill", "description": "x"}, "my-skill", True),
        ({"name": "other", "description": "x"}, "my-skill", False),  # must match the folder
        ({"name": "My_Skill", "description": "x"}, "My_Skill", False),
        ({"name": "a--b", "description": "x"}, "a--b", False),
        ({"name": "ok", "description": "  "}, "ok", False),
        ({"name": "ok", "description": "x" * 1025}, "ok", False),
        ({"name": "x" * 65, "description": "d"}, "x" * 65, False),
    ],
)
def test_valid_skill_metadata(meta, name, ok):
    assert valid_skill_metadata(meta, name) is ok


# --- discovery --------------------------------------------------------------------------------


def test_workspace_skills_come_first_and_override_builtin(ws, builtin):
    write_skill(ws / "skills", "mine")
    write_skill(ws / "skills", "shared", description="Workspace version.")
    write_skill(builtin, "shared", description="Built-in version.")
    write_skill(builtin, "stock")
    (ws / "skills" / "not-a-skill").mkdir()
    (ws / "skills" / "loose.md").write_text("x")
    skills = loader(ws, builtin).list_skills()
    by_name = {s["name"]: s["source"] for s in skills}
    assert by_name == {"mine": "workspace", "shared": "workspace", "stock": "builtin"}
    assert [s["source"] for s in skills][-1] == "builtin"
    assert loader(ws, builtin).get_skill_description("shared") == "Workspace version."


def test_requirements(ws, builtin, monkeypatch):
    write_skill(builtin, "needs-bin", extra='metadata: {"nanobot":{"requires":{"bins":["definitely-not-installed-x"]}}}\n')
    write_skill(builtin, "needs-env", extra='metadata: {"nanobot":{"requires":{"env":["NANOCLAW_TEST_TOKEN"]}}}\n')
    write_skill(builtin, "has-bin", extra='metadata: {"nanobot":{"requires":{"bins":["sh"]}}}\n')
    sl = loader(ws, builtin)
    assert {s["name"] for s in sl.list_skills()} == {"has-bin"}
    assert len(sl.list_skills(filter_unavailable=False)) == 3
    assert sl.get_skill_availability("needs-bin") == (False, "CLI: definitely-not-installed-x")
    monkeypatch.setenv("NANOCLAW_TEST_TOKEN", "t")
    assert sl.get_skill_availability("needs-env") == (True, "")


@pytest.mark.parametrize(
    "metadata",
    [
        'metadata: {"nanobot":{"requires":{"bins":["definitely-not-installed-x"]}}}',
        'metadata: \'{"openclaw":{"requires":{"bins":["definitely-not-installed-x"]}}}\'',
        'metadata: {"nanoclaw":{"requires":{"bins":["definitely-not-installed-x"]}}}',
    ],
)
def test_metadata_forms(ws, builtin, metadata):
    write_skill(builtin, "s", extra=metadata + "\n")
    assert loader(ws, builtin).get_skill_availability("s")[0] is False


def test_wrong_metadata_shapes_are_ignored(ws, builtin):
    write_skill(builtin, "a", extra='metadata: "not json"\n')
    write_skill(builtin, "b", extra='metadata: {"nanobot":{"requires":"curl"}}\n')
    write_skill(builtin, "c", extra='metadata: {"nanobot":{"requires":{"bins":"curl"}}}\n')
    assert {s["name"] for s in loader(ws, builtin).list_skills()} == {"a", "b", "c"}


def test_description_falls_back_to_the_name(ws, builtin):
    (builtin / "bare").mkdir()
    (builtin / "bare" / "SKILL.md").write_text("Just instructions, no frontmatter.")
    assert loader(ws, builtin).get_skill_description("bare") == "bare"


# --- loading ----------------------------------------------------------------------------------


def test_load_skills_for_context_strips_frontmatter(ws, builtin):
    write_skill(builtin, "one", body="First.")
    write_skill(builtin, "two", body="Second.")
    sl = loader(ws, builtin)
    assert sl.load_skill("missing") is None
    assert sl.load_skills_for_context(["one", "missing", "two"]) == (
        "### Skill: one\n\nFirst.\n\n---\n\n### Skill: two\n\nSecond."
    )


def test_always_skills(ws, builtin):
    write_skill(builtin, "top", extra="always: true\n")
    write_skill(builtin, "nested", extra='metadata: {"nanobot":{"always":true}}\n')
    write_skill(builtin, "blocked", extra='metadata: {"nanobot":{"always":true,"requires":{"bins":["definitely-not-installed-x"]}}}\n')
    write_skill(builtin, "normal")
    assert sorted(loader(ws, builtin).get_always_skills()) == ["nested", "top"]


def test_explicit_skill_references(ws, builtin):
    write_skill(builtin, "weather", body="Use curl.")
    write_skill(builtin, "github")
    write_skill(builtin, "off", extra='metadata: {"nanobot":{"requires":{"bins":["definitely-not-installed-x"]}}}\n')
    sl = loader(ws, builtin)
    assert sl.get_explicitly_invoked_skills("try $github and $weather, then $github again") == ["github", "weather"]
    assert sl.get_explicitly_invoked_skills("price is US$weather or a$github") == []
    assert sl.get_explicitly_invoked_skills("$unknown $off") == []
    block = sl.build_explicit_skill_runtime_context("what's the $weather?")
    assert block == RuntimeContextBlock(
        "explicit_skills",
        "[Active Skills — instructions for this user turn]\n### Skill: weather\n\nUse curl.\n[/Active Skills]",
    )
    assert sl.build_explicit_skill_runtime_context("no skills here") is None


def test_explicit_reference_to_an_always_skill_adds_nothing(ws, builtin):
    write_skill(builtin, "core", extra="always: true\n")
    assert loader(ws, builtin).build_explicit_skill_runtime_context("$core") is None


def test_skills_summary(ws, builtin):
    write_skill(ws / "skills", "mine", description="My own.")
    write_skill(builtin, "weather", description="Forecasts.")
    write_skill(builtin, "gh", description="GitHub.", extra='metadata: {"nanobot":{"requires":{"bins":["definitely-not-installed-x"],"env":["NANOCLAW_NOPE"]}}}\n')
    summary = loader(ws, builtin).build_skills_summary(exclude={"weather"})
    assert summary == (
        "### Workspace skills (`skills`)\n- **mine** — My own.  `mine/SKILL.md`\n\n"
        "### Built-in skills (`skills`)\n"
        "- **gh** — GitHub. (unavailable: CLI: definitely-not-installed-x, ENV: NANOCLAW_NOPE)  `gh/SKILL.md`"
    )
    assert loader(ws, ws / "nothing").build_skills_summary() == "### Workspace skills (`skills`)\n- **mine** — My own.  `mine/SKILL.md`"
    assert SkillsLoader(ws / "empty", builtin_skills_dir=ws / "none").build_skills_summary() == ""


# --- built-in skills shipped with NanoClaw ------------------------------------------------------


def test_bundled_skills_are_valid():
    names = sorted(p.name for p in BUILTIN_SKILLS_DIR.iterdir() if (p / "SKILL.md").exists())
    assert names == ["github", "memory", "skill-creator", "summarize", "tmux", "weather"]
    for name in names:
        meta = parse_skill_metadata((BUILTIN_SKILLS_DIR / name / "SKILL.md").read_text())
        assert valid_skill_metadata(meta, name), name
    assert (BUILTIN_SKILLS_DIR / "skill-creator" / "scripts" / "init_skill.py").exists()


# --- read_file finds built-in skills under skills/ --------------------------------------------------


async def test_read_file_falls_back_to_builtin_skills(ws):
    tool = ReadFileTool(ws)
    assert "name: weather" in await tool.execute(path="skills/weather/SKILL.md")
    # A workspace copy wins.
    write_skill(ws / "skills", "weather", description="Local copy.")
    assert "Local copy." in await tool.execute(path="skills/weather/SKILL.md")
    # No escaping the built-in folder, and absolute paths are not mapped.
    assert (await tool.execute(path="skills/../../etc/hostname")).startswith("Error: File not found")
    # nanoclaw/agent/skills.py exists, but is outside the built-in skills folder.
    assert (await tool.execute(path="skills/../agent/skills.py")).startswith("Error: File not found")
    assert (await tool.execute(path=str(ws / "skills" / "memory" / "SKILL.md"))).startswith("Error: File not found")


# --- in the system prompt and the turn -----------------------------------------------------------------


def test_prompt_sections(ws, builtin):
    sync_workspace_templates(ws)
    write_skill(builtin, "core", body="Always do X.", extra="always: true\n")
    write_skill(builtin, "weather", description="Forecasts.")
    builder = ContextBuilder(ws)
    builder.skills = loader(ws, builtin)
    sections = builder.build_system_prompt(channel="cli").split(SECTION_SEPARATOR)
    assert sections[-2] == "# Active Skills\n\n### Skill: core\n\nAlways do X."
    assert sections[-1].startswith("# Skills\n\nThe following skills extend your capabilities.")
    assert "- **weather** — Forecasts.  `weather/SKILL.md`" in sections[-1]
    assert "core" not in sections[-1]  # always-on skills are not listed twice


def test_current_message_gets_the_named_skill(ws, builtin):
    write_skill(builtin, "weather", body="Use curl.")
    builder = ContextBuilder(ws)
    builder.skills = loader(ws, builtin)
    msg = builder.build_current_message("$weather in Paris?")
    assert msg["content"].startswith("$weather in Paris?\n\n[Active Skills — instructions for this user turn]")
    assert msg["_meta"]["runtime_context"]["sources"] == ["explicit_skills"]
    # A block the caller already added is not repeated.
    block = builder.skills.build_explicit_skill_runtime_context("$weather")
    msg = builder.build_current_message("$weather in Paris?", runtime_context_blocks=[block])
    assert msg["content"].count("[Active Skills") == 1


async def test_turn_saves_the_named_skill_with_the_message(ws, tmp_path):
    fake = FakeProvider(["Sunny."])
    loop = AgentLoop(fake, ws, tools=ToolRegistry(), sessions=SessionManager(ws, sessions_root=tmp_path / "s"))
    await loop.process_direct("$weather in Paris?")
    user = loop.sessions.get_or_create("cli:direct").messages[0]
    assert user["_runtime_context"]["sources"] == ["time", "explicit_skills"]
    assert "### Skill: weather" in user["content"]
    sent = fake.calls[0].messages[-1]["content"]
    assert sent.count("[Active Skills") == 1 and sent.index("Current Time") < sent.index("[Active Skills")


# --- Dream and the workspace ------------------------------------------------------------------------------


def test_dream_prompt_includes_the_skills_rules(ws):
    store = MemoryStore(ws)
    store.append_history("- [durable] deploys with make release, twice now")
    prompt, _ = store.build_dream_prompt()
    assert "## Skills\n\nCreate a skill only when a workflow has appeared at least twice" in prompt
    assert str(BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md") in prompt
    assert "| `skills/<name>/SKILL.md` |" in prompt


async def test_dream_can_read_builtin_skills_by_absolute_path(ws):
    # Dream's instructions name the skill-creator skill by its absolute path.
    tools = MemoryStore(ws).build_dream_tools()
    out = await tools.execute("read_file", {"path": str(BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md")})
    assert "name: skill-creator" in out


def test_workspace_gets_a_skills_folder(tmp_path):
    sync_workspace_templates(tmp_path / "ws")
    assert (tmp_path / "ws" / "skills").is_dir()


def test_cli_skills_command(capsys, ws):
    write_skill(ws / "skills", "mine", description="My own.")
    assert main(["skills", "-w", str(ws)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Available skills (7):\n\n- **mine** — My own.\n")
    assert "- **weather** — Get current weather and forecasts (no API key required)." in out
