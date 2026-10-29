"""``nanoclaw plugins list/enable/disable`` and an end-to-end Agent Plugin.

nanobot enables plugins from its web UI only; NanoClaw's CLI makes the same
``set_agent_plugin_enabled`` call after showing what the package brings
(nanoclaw/cli/plugins.py). The end-to-end test runs ``nanoclaw ask`` with a
plugin that has a skill and a stdio MCP server (tests/fixtures/mcp_demo_server.py):
both reach the model only after ``enable``, and both disappear when the
package changes.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from nanoclaw.__main__ import main
from nanoclaw.agent import plugins as agent_plugins
from nanoclaw.agent.plugins import (
    AGENT_PLUGIN_MCP_SCHEMA,
    AGENT_PLUGIN_SCHEMA,
    discover_agent_plugins,
    set_agent_plugin_enabled,
)
from nanoclaw.cli import plugins as plugins_cli
from nanoclaw.config.loader import get_config_path
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest

SERVER = Path(__file__).parent / "fixtures" / "mcp_demo_server.py"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _demo_plugin(workspace: Path, *, server: bool = True) -> Path:
    root = workspace / "plugins" / "demo"
    _write_json(
        root / "plugin.json",
        {
            "$schema": AGENT_PLUGIN_SCHEMA,
            "name": "demo",
            "description": "Adds numbers.",
            "repository": "https://example.com/demo",
            "extensions": {
                "dev.nanobot": {"displayName": "Demo Tools", "permissions": ["Runs a local Python MCP server"]}
            },
        },
    )
    skill = root / "skills" / "demo-math"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo-math\ndescription: Add numbers with the demo server.\n---\n\nUse mcp_demo_add.\n",
        encoding="utf-8",
    )
    bad = root / "skills" / "broken"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\nname: other\ndescription: Mismatch.\n---\n", encoding="utf-8")
    if server:
        shutil.copy(SERVER, root / "server.py")
        _write_json(
            root / "mcp.json",
            {
                "$schema": AGENT_PLUGIN_MCP_SCHEMA,
                "mcpServers": {
                    "math": {"type": "stdio", "command": "python", "args": ["${PLUGIN_ROOT}/server.py"]}
                },
            },
        )
    return root


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    path = tmp_path / "ws"
    path.mkdir()
    return path


def test_list_without_plugins(ws, capsys):
    assert main(["plugins", "list", "-w", str(ws)]) == 0
    assert f"No Agent Plugins in {ws / 'plugins'}." in capsys.readouterr().out


def test_list_shows_state_skills_and_servers(ws, capsys):
    _demo_plugin(ws)
    assert main(["plugins", "list", "-w", str(ws)]) == 0
    out = capsys.readouterr().out
    assert "- **demo** Demo Tools [disabled] — Adds numbers." in out
    assert "skills: demo-math\n" in out
    assert "MCP servers: math" in out
    set_agent_plugin_enabled(ws, "demo", True)
    assert main(["plugins", "list", "-w", str(ws)]) == 0
    assert "[enabled]" in capsys.readouterr().out


def test_enable_reviews_and_asks(ws, capsys, monkeypatch):
    root = _demo_plugin(ws)
    prompts: list[str] = []

    def answer(text):
        prompts.append(text)
        return "n"

    monkeypatch.setattr("builtins.input", answer)
    assert main(["plugins", "enable", "demo", "-w", str(ws)]) == 1
    out = capsys.readouterr().out
    fingerprint = agent_plugins._package_fingerprint(root.resolve())
    assert f"fingerprint: sha256:{fingerprint}" in out
    assert "permissions: Runs a local Python MCP server" in out
    assert "skills:      demo-math" in out
    assert "ignored skills (invalid SKILL.md): broken" in out
    assert f"math: python {root.resolve()}/server.py" in out
    assert "repository:  https://example.com/demo" in out
    assert prompts == ["Enable demo? [y/N] "]
    assert "Not enabled." in out
    assert discover_agent_plugins(ws)[0].enabled is False

    monkeypatch.setattr("builtins.input", lambda _text: "y")
    assert main(["plugins", "enable", "demo", "-w", str(ws)]) == 0
    assert "demo enabled." in capsys.readouterr().out
    assert discover_agent_plugins(ws)[0].enabled is True
    marker = next((get_config_path().parent / "plugin-data").glob("*/demo/enabled"))
    assert json.loads(marker.read_text())["fingerprint"] == fingerprint
    if os.name == "posix":
        assert marker.stat().st_mode & 0o777 == 0o600

    assert main(["plugins", "enable", "demo", "-w", str(ws)]) == 0
    assert "demo is already enabled." in capsys.readouterr().out


def test_enable_without_a_terminal_answer_refuses(ws, capsys, monkeypatch):
    _demo_plugin(ws, server=False)

    def eof(_text):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert main(["plugins", "enable", "demo", "-w", str(ws)]) == 1
    assert discover_agent_plugins(ws)[0].enabled is False


def test_enable_yes_skips_the_question_and_disable(ws, capsys, monkeypatch):
    _demo_plugin(ws, server=False)
    monkeypatch.setattr("builtins.input", lambda _text: pytest.fail("asked"))
    assert main(["plugins", "enable", "demo", "--yes", "-w", str(ws)]) == 0
    out = capsys.readouterr().out
    assert "MCP servers: (none)" in out and "demo enabled." in out
    assert discover_agent_plugins(ws)[0].enabled is True
    assert main(["plugins", "disable", "demo", "-w", str(ws)]) == 0
    assert "demo disabled." in capsys.readouterr().out
    assert discover_agent_plugins(ws)[0].enabled is False


def test_unknown_plugin(ws, capsys):
    assert main(["plugins", "enable", "nope", "-y", "-w", str(ws)]) == 1
    assert main(["plugins", "disable", "nope", "-w", str(ws)]) == 1
    err = capsys.readouterr().err
    assert err.count("Unknown Agent Plugin 'nope'") == 2


def test_enable_fails_when_the_package_changes_during_review(ws, capsys, monkeypatch):
    root = _demo_plugin(ws, server=False)

    def edit_while_asking(_text):
        (root / "skills" / "demo-math" / "SKILL.md").write_text(
            "---\nname: demo-math\ndescription: Something else.\n---\n", encoding="utf-8"
        )
        return "y"

    monkeypatch.setattr("builtins.input", edit_while_asking)
    assert main(["plugins", "enable", "demo", "-w", str(ws)]) == 1
    assert "changed while it was being reviewed" in capsys.readouterr().err
    assert discover_agent_plugins(ws)[0].enabled is False


def _ask(ws: Path, script: list) -> FakeProvider:
    fake = FakeProvider(script)
    assert main(["ask", "add 2 and 3", "-w", str(ws)], provider=fake) == 0
    return fake


def _tool_names(fake: FakeProvider) -> list[str]:
    return [d["function"]["name"] for d in fake.calls[0].tools or []]


def test_end_to_end_plugin_skill_and_mcp_server(ws, capfd, monkeypatch):
    # The plugin's server runs as ``python`` from PATH (a bare command name,
    # as Agent Plugins require); point PATH at this interpreter.
    monkeypatch.setenv("PATH", os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")]))
    root = _demo_plugin(ws)

    before = _ask(ws, ["No tools."])
    prompt = before.calls[0].messages[0]["content"]
    assert "demo-math" not in prompt and "Agent Plugin skills" not in prompt
    assert not any(name.startswith("mcp_") for name in _tool_names(before))

    assert main(["plugins", "enable", "demo", "--yes", "-w", str(ws)]) == 0
    capfd.readouterr()
    enabled = _ask(
        ws,
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest("c1", "mcp_demo_add", {"a": 2, "b": 3})],
                finish_reason="tool_calls",
            ),
            "The sum is 5.",
        ],
    )
    assert capfd.readouterr().out.strip() == "The sum is 5."
    prompt = enabled.calls[0].messages[0]["content"]
    assert "### Agent Plugin skills (`plugins`)" in prompt
    assert "- **demo-math** — Add numbers with the demo server.  `demo/skills/demo-math/SKILL.md`" in prompt
    assert "broken" not in prompt
    assert "mcp_demo_add" in _tool_names(enabled)
    tool_message = enabled.calls[1].messages[-1]
    assert tool_message["role"] == "tool" and tool_message["content"] == "5"

    # Any change to the package withdraws the approval: skill and server go.
    with (root / "server.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# changed\n")
    after = _ask(ws, ["No tools."])
    prompt = after.calls[0].messages[0]["content"]
    assert "demo-math" not in prompt and "Agent Plugin skills" not in prompt
    assert not any(name.startswith("mcp_") for name in _tool_names(after))
    assert main(["plugins", "list", "-w", str(ws)]) == 0
    assert "[disabled]" in capfd.readouterr().out


def test_mcp_provider_merges_plugin_servers_user_config_wins(ws, tmp_path):
    from nanoclaw.agent.tools.mcp import configured_servers
    from nanoclaw.config import Config

    _demo_plugin(ws)
    config = Config.model_validate({"tools": {"mcpServers": {"other": {"command": "echo"}}}})
    assert set(configured_servers(config, ws)) == {"other"}
    set_agent_plugin_enabled(ws, "demo", True)
    assert set(configured_servers(config, ws)) == {"other", "demo"}
    config = Config.model_validate({"tools": {"mcpServers": {"demo": {"command": "echo", "args": ["mine"]}}}})
    assert configured_servers(config, ws)["demo"].args == ["mine"]


def test_plugins_help_lists_actions(capsys):
    with pytest.raises(SystemExit):
        main(["plugins", "--help"])
    out = capsys.readouterr().out
    assert "list" in out and "enable" in out and "disable" in out
    assert plugins_cli.__all__ == ["add_plugins_parser", "plugins_command"]
