"""MCP configuration and wiring: ``tools.mcpServers``, ``config show``
masking, ``nanoclaw ask`` connecting and closing the servers, subagents
without MCP tools, and the options NanoClaw accepts but does not support."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nanoclaw.__main__ import main
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.tools import mcp as mcp_module
from nanoclaw.agent.tools.mcp import MCPProvider, connect_mcp_servers
from nanoclaw.agent.tools.registry import ToolRegistry
from nanoclaw.config import Config
from nanoclaw.config.loader import load_config, mask_secrets
from nanoclaw.config.schema import MCPServerConfig
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest

SERVER = Path(__file__).parent / "fixtures" / "mcp_demo_server.py"


def test_mcp_servers_load_from_camel_case_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "tools": {
                    "mcpServers": {
                        "fs": {"command": "npx", "args": ["-y", "server-fs"], "env": {"TOKEN": "x"}},
                        "remote": {
                            "url": "https://mcp.example.com/mcp",
                            "headers": {"Authorization": "Bearer abc"},
                            "toolTimeout": 60,
                            "enabledTools": ["search"],
                            "useProxy": True,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    servers = load_config(path).tools.mcp_servers
    assert servers["fs"].command == "npx" and servers["fs"].args == ["-y", "server-fs"]
    assert servers["fs"].enabled_tools == ["*"] and servers["fs"].tool_timeout == 30
    remote = servers["remote"]
    assert (remote.tool_timeout, remote.enabled_tools, remote.use_proxy) == (60, ["search"], True)
    assert remote.headers == {"Authorization": "Bearer abc"}
    dumped = Config().model_dump(mode="json", by_alias=True)
    assert dumped["tools"]["mcpServers"] == {}
    assert set(MCPServerConfig().model_dump(by_alias=True)) == {
        "type", "auth", "command", "args", "env", "cwd", "url", "headers",
        "useProxy", "toolTimeout", "enabledTools",
    }


def test_config_show_masks_mcp_headers_and_env():
    data = {
        "tools": {
            "mcpServers": {
                "remote": {"url": "https://x", "headers": {"Authorization": "Bearer sk-abcdefghijkl"}},
                "local": {"command": "npx", "env": {"GITHUB_TOKEN": "ghp_123"}, "args": ["a"]},
            }
        }
    }
    masked = mask_secrets(data)["tools"]["mcpServers"]
    assert masked["remote"]["headers"] == {"Authorization": "****ijkl"}
    assert masked["local"]["env"] == {"GITHUB_TOKEN": "****"}
    assert masked["local"]["args"] == ["a"] and masked["remote"]["url"] == "https://x"


def _write_config(tmp_path: Path, servers: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tools": {"mcpServers": servers}}), encoding="utf-8")
    return path


def test_cli_ask_connects_mcp_servers_and_closes_them(tmp_path, capfd, monkeypatch):
    # capfd, not capsys: the MCP SDK hands sys.stderr to the server process
    # (its stderr), which needs a real file descriptor.
    path = _write_config(
        tmp_path,
        {
            "demo": {"command": sys.executable, "args": [str(SERVER)], "enabledTools": ["add"]},
            "missing": {"command": str(tmp_path / "no-such-server")},
        },
    )
    closed: list[set[str]] = []
    real_aclose = MCPProvider.aclose

    async def tracking_aclose(self):
        closed.append(self.connected_server_names)
        await real_aclose(self)

    monkeypatch.setattr(MCPProvider, "aclose", tracking_aclose)
    fake = FakeProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest("c1", "mcp_demo_add", {"a": 40, "b": 2})],
                finish_reason="tool_calls",
            ),
            "The sum is 42.",
        ]
    )
    assert main(["ask", "add", "-c", str(path), "-w", str(tmp_path / "ws")], provider=fake) == 0
    assert capfd.readouterr().out.strip() == "The sum is 42."
    names = [d["function"]["name"] for d in fake.calls[0].tools]
    assert names[-1] == "mcp_demo_add"  # after the built-ins
    assert names[:-1] == sorted(names[:-1])
    tool_message = fake.calls[1].messages[-1]
    assert tool_message["role"] == "tool" and tool_message["content"] == "42"
    assert closed == [{"demo"}]  # the failing server was skipped, the other closed at exit


def test_subagents_get_no_mcp_tools(tmp_path):
    """nanobot builds subagent tools with ToolLoader, which skips the mcp module
    (agent/tools/loader.py:22); MCP tools live only in the main registry."""
    loop = AgentLoop(FakeProvider([]), tmp_path)
    loop.tools.register(
        mcp_module.MCPToolWrapper(
            object(), "demo", type("D", (), {"name": "x", "description": "", "inputSchema": None})()
        )
    )
    assert "mcp_demo_x" in loop.tools
    assert not any(name.startswith("mcp_") for name in loop.subagents._build_tools().tool_names)


async def test_oauth_servers_are_skipped(monkeypatch):
    probed: list[str] = []

    async def probe(url):
        probed.append(url)
        return True

    monkeypatch.setattr(mcp_module, "validate_url_target", lambda _url: (True, ""))
    monkeypatch.setattr(mcp_module, "_probe_http_url", probe)
    registry = ToolRegistry()
    connections = await connect_mcp_servers(
        {"remote": MCPServerConfig(url="https://mcp.example.com/mcp", auth="oauth")}, registry
    )
    assert connections == {} and probed == []


async def test_use_proxy_is_ignored_for_stdio():
    """nanobot refuses useProxy on a stdio server (it has a shared proxy to
    misapply); NanoClaw ignores the setting and starts the server."""
    registry = ToolRegistry()
    connections = await connect_mcp_servers(
        {"demo": MCPServerConfig(command=sys.executable, args=[str(SERVER)], use_proxy=True, enabled_tools=["add"])},
        registry,
    )
    try:
        assert registry.tool_names == ["mcp_demo_add"]
    finally:
        for connection in connections.values():
            await connection.aclose()


def test_use_proxy_defaults_off():
    assert MCPServerConfig().use_proxy is False


def _record_mcp_lifecycle(monkeypatch) -> list[str]:
    events: list[str] = []

    async def connect(self):
        events.append("mcp.connect")

    async def aclose(self):
        events.append("mcp.aclose")

    real_close = AgentLoop.close

    async def loop_close(self):
        events.append("loop.close")
        await real_close(self)

    monkeypatch.setattr(MCPProvider, "connect", connect)
    monkeypatch.setattr(MCPProvider, "aclose", aclose)
    monkeypatch.setattr(AgentLoop, "close", loop_close)
    return events


def test_cli_ask_closes_mcp_after_the_loop(tmp_path, monkeypatch):
    events = _record_mcp_lifecycle(monkeypatch)
    path = _write_config(tmp_path, {})
    assert main(["ask", "hi", "-c", str(path), "-w", str(tmp_path / "ws")], provider=FakeProvider(["ok"])) == 0
    assert events == ["mcp.connect", "loop.close", "mcp.aclose"]


def test_cli_chat_connects_mcp_before_the_first_turn_and_closes_it(tmp_path, monkeypatch):
    import io

    from nanoclaw.cli import chat as chat_module

    events = _record_mcp_lifecycle(monkeypatch)
    real_run = chat_module.Chat.run

    async def run(self):
        events.append("chat.run")
        await real_run(self)

    monkeypatch.setattr(chat_module.Chat, "run", run)
    monkeypatch.setattr(chat_module, "install_exit_signal_handlers", lambda console: None)
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello\nexit\n"))
    path = _write_config(tmp_path, {})
    assert main(["chat", "-c", str(path), "-w", str(tmp_path / "ws"), "--no-markdown"], provider=FakeProvider(["Hi!"])) == 0
    assert [e for e in events if e != "loop.close"] == ["mcp.connect", "chat.run", "mcp.aclose"]
