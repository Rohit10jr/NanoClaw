"""Tests for the application-owned MCP provider lifecycle.

Ported from nanobot tests/agent/test_mcp_connection.py (c4dac64), without hot
reload, OAuth, the shared proxy and runtime status. The concurrent reconnect
test uses tools where nanobot's uses resource wrappers.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import anyio
import pytest
from mcp import types as mcp_types
from mcp.shared.exceptions import McpError
from mcp.shared.message import SessionMessage
from mcp.types import ErrorData

from nanoclaw.agent.tools import mcp as mcp_runtime
from nanoclaw.agent.tools.base import Tool
from nanoclaw.agent.tools.mcp import MCPProvider, MCPToolWrapper
from nanoclaw.agent.tools.registry import ToolRegistry
from nanoclaw.config.schema import MCPServerConfig


def _mcp_notification(method: str, params: dict[str, Any] | None = None) -> SessionMessage:
    return SessionMessage(
        message=mcp_types.JSONRPCMessage(
            mcp_types.JSONRPCNotification(
                jsonrpc="2.0",
                method=method,
                params=params,
            )
        )
    )


def test_mcp_progress_detection_accepts_flattened_sdk_message_shape():
    malformed = SimpleNamespace(
        message=SimpleNamespace(
            method="notifications/progress",
            params={"progress": 20, "total": 600},
        )
    )
    valid = SimpleNamespace(
        message=SimpleNamespace(
            method="notifications/progress",
            params={"progressToken": "req-1", "progress": 25},
        )
    )

    assert mcp_runtime._is_malformed_mcp_progress_notification(malformed) is True
    assert mcp_runtime._is_malformed_mcp_progress_notification(valid) is False


class _FakeMcpTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "fake MCP tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **_kwargs: Any) -> str:
        return "ok"


def _stdio_server(command: str = "test-mcp") -> MCPServerConfig:
    return MCPServerConfig(type="stdio", command=command)


def _make_provider(
    *,
    mcp_servers: dict[str, MCPServerConfig] | None = None,
) -> tuple[MCPProvider, ToolRegistry]:
    registry = ToolRegistry()
    provider = MCPProvider(
        mcp_servers if mcp_servers is not None else {"test": _stdio_server()},
        registry,
    )
    return provider, registry


@pytest.mark.asyncio
async def test_mcp_read_filter_drops_progress_notifications_without_progress_token():
    send, receive = anyio.create_memory_object_stream(4)
    malformed_progress = _mcp_notification(
        "notifications/progress",
        {"progress": 20, "total": 600, "message": "Polling"},
    )
    tool_change = _mcp_notification("notifications/tools/list_changed")
    valid_progress = _mcp_notification(
        "notifications/progress",
        {"progressToken": "req-1", "progress": 25, "total": 600, "message": "Polling"},
    )

    await send.send(malformed_progress)
    await send.send(tool_change)
    await send.send(valid_progress)
    await send.aclose()

    wrapped = mcp_runtime._filter_malformed_mcp_progress_notifications(receive, "brightdata")
    forwarded = []
    async with wrapped:
        async for message in wrapped:
            forwarded.append(message)

    assert forwarded == [tool_change, valid_progress]


@pytest.mark.asyncio
async def test_owned_mcp_connection_closes_from_its_owner_task():
    close_requested = asyncio.Event()
    ready = asyncio.Event()
    tasks: dict[str, asyncio.Task] = {}

    async def own_connection() -> None:
        tasks["open"] = asyncio.current_task()  # type: ignore[assignment]
        ready.set()
        await close_requested.wait()
        tasks["close"] = asyncio.current_task()  # type: ignore[assignment]

    owner = asyncio.create_task(own_connection())
    connection = mcp_runtime._OwnedMCPConnection(owner, close_requested)
    await ready.wait()

    await connection.aclose()

    assert tasks["open"] is owner
    assert tasks["close"] is owner
    assert tasks["close"] is not asyncio.current_task()


@pytest.mark.asyncio
async def test_connect_mcp_retries_when_no_servers_connect(tmp_path, monkeypatch: pytest.MonkeyPatch):
    provider, _registry = _make_provider()
    attempts = 0

    async def _fake_connect(_servers, _registry):
        nonlocal attempts
        attempts += 1
        return {}

    monkeypatch.setattr("nanoclaw.agent.tools.mcp.connect_mcp_servers", _fake_connect)

    await provider.connect()
    await provider.connect()

    assert attempts == 2
    assert provider.connected_server_names == set()


@pytest.mark.asyncio
async def test_mcp_provider_closes_connections_independently_from_agent_loop(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider, registry = _make_provider(
        mcp_servers={"playwright": _stdio_server("playwright")}
    )
    owner_tasks: list[asyncio.Task | None] = []
    closed_tasks: list[asyncio.Task | None] = []

    class _OwnerCheckedStack:
        def __init__(self) -> None:
            self.owner = asyncio.current_task()
            owner_tasks.append(self.owner)

        async def aclose(self) -> None:
            closed_tasks.append(asyncio.current_task())
            assert asyncio.current_task() is self.owner

    async def _fake_connect(servers, _registry):
        stacks = {name: _OwnerCheckedStack() for name in servers}
        return stacks

    monkeypatch.setattr("nanoclaw.agent.tools.mcp.connect_mcp_servers", _fake_connect)

    await provider.connect()
    registry.register(_FakeMcpTool("mcp_playwright_search"))
    await provider.aclose()

    assert owner_tasks
    assert closed_tasks == owner_tasks
    assert provider.connected_server_names == set()
    assert registry.get("mcp_playwright_search") is None


@pytest.mark.asyncio
async def test_close_server_ignores_server_cancelled_error(tmp_path):
    provider, _registry = _make_provider()

    class _ServerCancelledStack:
        async def aclose(self) -> None:
            raise asyncio.CancelledError()

    provider._connections = {"test": _ServerCancelledStack()}

    await provider._close_server("test")

    assert provider.connected_server_names == set()


@pytest.mark.asyncio
async def test_provider_close_continues_after_server_cancelled_error(tmp_path):
    provider, _registry = _make_provider()
    closed: list[str] = []

    class _ServerCancelledStack:
        async def aclose(self) -> None:
            raise asyncio.CancelledError()

    class _TrackedStack:
        async def aclose(self) -> None:
            closed.append("second")

    provider._connections = {
        "first": _ServerCancelledStack(),
        "second": _TrackedStack(),
    }

    await provider.aclose()

    assert closed == ["second"]
    assert provider.connected_server_names == set()


@pytest.mark.asyncio
async def test_provider_close_finishes_other_connections_before_propagating_cancellation(
    tmp_path,
):
    provider, _registry = _make_provider()
    started = asyncio.Event()
    closed: list[str] = []

    class _BlockingStack:
        async def aclose(self) -> None:
            started.set()
            await asyncio.Event().wait()

    class _TrackedStack:
        async def aclose(self) -> None:
            closed.append("second")

    provider._connections = {
        "first": _BlockingStack(),
        "second": _TrackedStack(),
    }
    task = asyncio.create_task(provider.aclose())
    await asyncio.wait_for(started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed == ["second"]
    assert provider.connected_server_names == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_all", [False, True], ids=["single", "all"])
async def test_mcp_cleanup_re_raises_external_cancellation(tmp_path, close_all: bool):
    provider, _registry = _make_provider()
    started = asyncio.Event()

    class _BlockingStack:
        async def aclose(self) -> None:
            started.set()
            await asyncio.Event().wait()

    provider._connections = {"test": _BlockingStack()}

    if close_all:
        task = asyncio.create_task(provider.aclose())
    else:
        task = asyncio.create_task(provider._close_server("test"))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_mcp_tool_reconnects_after_session_terminated(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider, registry = _make_provider(
        mcp_servers={"remote": _stdio_server("remote")}
    )
    closed: list[str] = []
    sessions: list[Any] = []
    connect_count = 0

    async def _mark_closed(name: str) -> None:
        closed.append(name)

    class _FakeSession:
        def __init__(self, index: int) -> None:
            self.index = index
            self.call_count = 0

        async def call_tool(self, _name: str, arguments: dict[str, Any]) -> Any:
            self.call_count += 1
            assert arguments == {"symbol": "AAPL"}
            if self.index == 1:
                raise McpError(ErrorData(code=-32000, message="Session terminated"))
            return SimpleNamespace(
                content=[mcp_types.TextContent(type="text", text="recovered")]
            )

    async def _fake_connect(servers, registry):
        nonlocal connect_count
        stacks = {}
        for name in servers:
            connect_count += 1
            session = _FakeSession(connect_count)
            sessions.append(session)
            tool_def = SimpleNamespace(
                name="quote",
                description="quote tool",
                inputSchema={"type": "object", "properties": {}},
            )
            registry.register(MCPToolWrapper(session, name, tool_def, tool_timeout=5))
            stack = AsyncExitStack()
            await stack.__aenter__()
            stack.push_async_callback(_mark_closed, name)
            stacks[name] = stack
        return stacks

    monkeypatch.setattr("nanoclaw.agent.tools.mcp.connect_mcp_servers", _fake_connect)

    await provider.connect()
    old_tool = registry.get("mcp_remote_quote")
    assert isinstance(old_tool, MCPToolWrapper)

    output = await old_tool.execute(symbol="AAPL")

    assert output == "recovered"
    assert connect_count == 2
    assert closed == ["remote"]
    assert sessions[0].call_count == 1
    assert sessions[1].call_count == 1
    assert provider.connected_server_names == {"remote"}
    assert registry.get("mcp_remote_quote") is not old_tool


@pytest.mark.asyncio
async def test_mcp_reconnect_handler_uses_sanitized_server_prefix(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider, registry = _make_provider(
        mcp_servers={"remote_": _stdio_server("remote")}
    )
    connect_count = 0

    class _FakeSession:
        def __init__(self, index: int) -> None:
            self.index = index

        async def call_tool(self, _name: str, arguments: dict[str, Any]) -> Any:
            assert arguments == {}
            if self.index == 1:
                raise McpError(ErrorData(code=-32000, message="Session terminated"))
            return SimpleNamespace(
                content=[mcp_types.TextContent(type="text", text="recovered")]
            )

    async def _fake_connect(servers, registry):
        nonlocal connect_count
        stacks = {}
        for name in servers:
            connect_count += 1
            tool_def = SimpleNamespace(
                name="quote",
                description="quote tool",
                inputSchema={"type": "object", "properties": {}},
            )
            registry.register(MCPToolWrapper(_FakeSession(connect_count), name, tool_def))
            stack = AsyncExitStack()
            await stack.__aenter__()
            stacks[name] = stack
        return stacks

    monkeypatch.setattr("nanoclaw.agent.tools.mcp.connect_mcp_servers", _fake_connect)

    await provider.connect()
    old_tool = registry.get("mcp_remote_quote")
    assert isinstance(old_tool, MCPToolWrapper)

    output = await old_tool.execute()

    assert output == "recovered"
    assert connect_count == 2
    assert registry.get("mcp_remote_quote") is not old_tool


@pytest.mark.asyncio
async def test_concurrent_mcp_reconnect_reuses_fresh_session(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider, registry = _make_provider(
        mcp_servers={"remote": _stdio_server("remote")}
    )
    closed: list[str] = []
    connect_count = 0

    async def _mark_closed(name: str) -> None:
        closed.append(name)

    class _DeadSession:
        async def call_tool(self, _name: str, arguments: dict[str, Any]) -> Any:
            raise McpError(ErrorData(code=-32000, message="Session terminated"))

    class _LiveSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            await asyncio.sleep(0)
            return SimpleNamespace(content=[mcp_types.TextContent(type="text", text=f"fresh:{name}")])

    async def _fake_connect(servers, registry):
        nonlocal connect_count
        stacks = {}
        for name in servers:
            connect_count += 1
            session = _DeadSession() if connect_count == 1 else _LiveSession()
            for tool_name in ("alpha", "beta"):
                tool_def = SimpleNamespace(
                    name=tool_name,
                    description=f"{tool_name} tool",
                    inputSchema={"type": "object", "properties": {}},
                )
                registry.register(MCPToolWrapper(session, name, tool_def))
            stack = AsyncExitStack()
            await stack.__aenter__()
            stack.push_async_callback(_mark_closed, name)
            stacks[name] = stack
        return stacks

    monkeypatch.setattr("nanoclaw.agent.tools.mcp.connect_mcp_servers", _fake_connect)

    await provider.connect()
    old_alpha = registry.get("mcp_remote_alpha")
    old_beta = registry.get("mcp_remote_beta")
    assert isinstance(old_alpha, MCPToolWrapper)
    assert isinstance(old_beta, MCPToolWrapper)

    outputs = await asyncio.gather(old_alpha.execute(), old_beta.execute())

    assert outputs == ["fresh:alpha", "fresh:beta"]
    assert connect_count == 2
    assert closed == ["remote"]


