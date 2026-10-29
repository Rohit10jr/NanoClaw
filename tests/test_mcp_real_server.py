"""MCP against a real server: tests/fixtures/mcp_demo_server.py over stdio
(a child process) and over streamable HTTP (a child process on 127.0.0.1).

The SSRF guard blocks loopback, so the HTTP test lifts it for 127.0.0.1 the
way nanobot's tests/agent/test_mcp_reconnect_crash.py does.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from nanoclaw.agent.tools import build_default_tools, is_tool_error
from nanoclaw.agent.tools import mcp as mcp_module
from nanoclaw.agent.tools.mcp import MCPProvider, connect_mcp_servers
from nanoclaw.agent.tools.registry import ToolRegistry
from nanoclaw.config.schema import MCPServerConfig
from nanoclaw.security import network as security_network

SERVER = Path(__file__).parent / "fixtures" / "mcp_demo_server.py"
LONG = "a_tool_with_a_rather_long_name_that_needs_the_hash_suffix_to_fit"


def _stdio(**kwargs) -> MCPServerConfig:
    return MCPServerConfig(command=sys.executable, args=[str(SERVER)], **kwargs)


async def _connect(cfg: MCPServerConfig, name: str = "demo") -> tuple[ToolRegistry, dict]:
    registry = ToolRegistry()
    connections = await connect_mcp_servers({name: cfg}, registry)
    return registry, connections


async def _close(connections: dict) -> None:
    for connection in connections.values():
        await connection.aclose()


async def test_stdio_server_lists_and_calls_tools():
    registry, connections = await _connect(_stdio())
    try:
        assert set(connections) == {"demo"}
        names = registry.tool_names
        assert {"mcp_demo_add", "mcp_demo_echo", "mcp_demo_fail", "mcp_demo_slow"} <= set(names)
        long_name = mcp_module._sanitize_mcp_tool_name(f"mcp_demo_{LONG}")
        assert long_name in names and len(long_name) == 64

        assert await registry.execute("mcp_demo_add", {"a": 2, "b": "3"}) == "5"
        assert await registry.execute("mcp_demo_echo", {"text": "hi", "repeat": 2}) == "hi hi"
        assert await registry.execute(long_name, {}) == "long ok"
        assert await registry.execute("mcp_demo_two_blocks", {}) == "first\nsecond"
        assert await registry.execute("mcp_demo_tiny_image", {}) == (
            "(MCP tool returned an image that could not be stored)"
        )
        assert (
            await registry.execute("mcp_demo_describe_point", {"point": {"x": 1, "y": 2}, "label": None})
            == "point: (1, 2)"
        )
    finally:
        await _close(connections)


async def test_stdio_server_error_result_and_invalid_arguments():
    registry, connections = await _connect(_stdio())
    try:
        failed = await registry.execute("mcp_demo_fail", {"message": "boom"})
        assert is_tool_error(failed)
        assert "boom" in failed

        invalid = await registry.execute("mcp_demo_add", {"a": "x"})
        assert is_tool_error(invalid)
        assert invalid.startswith("Error: Invalid parameters for tool 'mcp_demo_add'")
    finally:
        await _close(connections)


async def test_stdio_server_tool_timeout():
    registry, connections = await _connect(_stdio(tool_timeout=1))
    try:
        result = await registry.execute("mcp_demo_slow", {"seconds": 5})
        assert is_tool_error(result)
        assert result.startswith("(MCP tool call timed out after 1s)")
        # The session still works after a timed-out call.
        assert await registry.execute("mcp_demo_add", {"a": 1, "b": 1}) == "2"
    finally:
        await _close(connections)


async def test_stdio_server_schema_is_normalized():
    registry, connections = await _connect(_stdio())
    try:
        params = registry.get("mcp_demo_describe_point").parameters
        assert params["properties"]["label"]["nullable"] is True
        assert params["properties"]["label"]["type"] == "string"
        assert params["properties"]["point"]["$ref"].startswith("#/$defs/")
        assert params["required"] == ["point"]
    finally:
        await _close(connections)


async def test_stdio_server_enabled_tools_filter():
    registry, connections = await _connect(_stdio(enabled_tools=["add", "mcp_demo_echo"]))
    try:
        assert registry.tool_names == ["mcp_demo_add", "mcp_demo_echo"]
    finally:
        await _close(connections)


async def test_bad_server_is_skipped_and_others_connect():
    registry = ToolRegistry()
    connections = await connect_mcp_servers(
        {
            "broken": MCPServerConfig(command=sys.executable, args=["-c", "import sys; sys.exit(3)"]),
            "demo": _stdio(enabled_tools=["add"]),
        },
        registry,
    )
    try:
        assert set(connections) == {"demo"}
        assert registry.tool_names == ["mcp_demo_add"]
    finally:
        await _close(connections)


async def test_provider_adds_tools_after_builtins_and_removes_them_on_close(tmp_path):
    registry = build_default_tools(tmp_path)
    builtin = [d["function"]["name"] for d in registry.get_definitions()]
    provider = MCPProvider({"demo": _stdio(enabled_tools=["add", "echo"])}, registry)
    await provider.connect()
    try:
        names = [d["function"]["name"] for d in registry.get_definitions()]
        assert names == builtin + ["mcp_demo_add", "mcp_demo_echo"]
    finally:
        await provider.aclose()
    assert [d["function"]["name"] for d in registry.get_definitions()] == builtin


# -- streamable HTTP ------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def http_server_url():
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, str(SERVER), "http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/mcp"
    deadline = time.monotonic() + 15
    ready = False
    while time.monotonic() < deadline and not ready:
        try:
            httpx.get(url, timeout=1.0, headers={"Accept": "text/event-stream"})
            ready = True
        except httpx.HTTPError:
            time.sleep(0.1)
    if not ready:
        process.kill()
        pytest.skip("MCP HTTP server did not start")
    yield url
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture(autouse=True)
def _own_resolver_lock(monkeypatch: pytest.MonkeyPatch):
    """PinnedDNSAsyncTransport's class-level asyncio.Lock binds to the first
    event loop that waits on it; give this module's loops their own (as
    nanobot's test_mcp_reconnect_crash.py does) so later tests are unaffected."""

    class TestPinnedDNSAsyncTransport(security_network.PinnedDNSAsyncTransport):
        _resolver_lock = asyncio.Lock()

    monkeypatch.setattr(mcp_module, "PinnedDNSAsyncTransport", TestPinnedDNSAsyncTransport)


@pytest.fixture
def allow_loopback(monkeypatch: pytest.MonkeyPatch):
    """Lift the SSRF guard for 127.0.0.1 only (nanobot's tests patch the same names)."""
    monkeypatch.setattr(mcp_module, "validate_url_target", lambda url: (True, ""))
    monkeypatch.setattr(mcp_module, "resolve_url_target", lambda url: (True, "", ("127.0.0.1",)))
    monkeypatch.setattr(security_network, "resolve_url_target", lambda url: (True, "", ("127.0.0.1",)))
    monkeypatch.setattr(mcp_module, "env_proxy_applies_to_url", lambda url: False)
    monkeypatch.setattr(mcp_module, "httpx_env_proxy_mounts", lambda: {})


async def test_streamable_http_server_lists_and_calls_tools(http_server_url, allow_loopback):
    registry, connections = await _connect(MCPServerConfig(url=http_server_url, enabled_tools=["add", "fail"]))
    try:
        assert set(connections) == {"demo"}
        assert registry.tool_names == ["mcp_demo_add", "mcp_demo_fail"]
        assert await registry.execute("mcp_demo_add", {"a": 20, "b": 22}) == "42"
        assert is_tool_error(await registry.execute("mcp_demo_fail", {"message": "nope"}))
    finally:
        await _close(connections)


async def test_streamable_http_loopback_is_blocked_without_whitelist(http_server_url):
    security_network.configure_ssrf_whitelist([])
    registry, connections = await _connect(MCPServerConfig(url=http_server_url))
    assert connections == {}
    assert registry.tool_names == []


async def test_streamable_http_loopback_allowed_by_ssrf_whitelist(http_server_url, monkeypatch):
    """``tools.ssrfWhitelist`` lets a trusted local server through, as for web_fetch."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    security_network.configure_ssrf_whitelist(["127.0.0.1/32"])
    try:
        registry, connections = await _connect(MCPServerConfig(url=http_server_url, enabled_tools=["echo"]))
        try:
            assert await asyncio.wait_for(registry.execute("mcp_demo_echo", {"text": "ok"}), 10) == "ok"
        finally:
            await _close(connections)
    finally:
        security_network.configure_ssrf_whitelist([])
