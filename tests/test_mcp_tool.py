"""MCP tools, ported from nanobot tests/tools/test_mcp_tool.py (c4dac64).

Left out with the features (docs/notes/03-tools.md): resource and prompt
wrappers, OAuth, the shared proxy (``network=``), runtime status, image
artifacts. A fake ``mcp`` module stands in for the SDK here; the real SDK and
a real stdio server are used in test_mcp_real_server.py.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote
from urllib.request import getproxies_environment

import httpx
import pytest

import nanoclaw.agent.tools.mcp as mcp_mod
from nanoclaw.agent.tools.mcp import (
    MCPProvider,
    MCPToolWrapper,
    _normalize_windows_stdio_command,
    _sanitize_mcp_tool_name,
    _sanitize_name,
    connect_mcp_servers,
)
from nanoclaw.agent.tools.base import is_tool_error
from nanoclaw.agent.tools.registry import ToolRegistry
from nanoclaw.config.schema import MCPServerConfig

_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


class _FakeTextContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeTextResourceContents:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeBlobResourceContents:
    def __init__(self, blob: bytes) -> None:
        self.blob = blob


class _FakeImageContent:
    def __init__(self, data: str, mime_type: str = "image/png") -> None:
        self.data = data
        self.mimeType = mime_type


@pytest.fixture
def fake_mcp_runtime() -> dict[str, object | None]:
    return {"session": None}


@pytest.fixture(autouse=True)
def _nanoclaw_logs():
    """NanoClaw's package logger is off until the CLI turns it on; these tests read it."""
    mcp_mod.logger.enable("nanoclaw")
    yield
    mcp_mod.logger.disable("nanoclaw")


@pytest.fixture(autouse=True)
def _clear_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*_PROXY_ENV_VARS, "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    # getproxies() falls back to OS-level proxy settings (e.g. the Windows
    # registry) when no environment variables are set; keep tests hermetic.
    monkeypatch.setattr("nanoclaw.security.network.getproxies", getproxies_environment)


@pytest.fixture(autouse=True)
def _fake_mcp_module(
    monkeypatch: pytest.MonkeyPatch, fake_mcp_runtime: dict[str, object | None]
) -> None:
    mod = ModuleType("mcp")
    mod.types = SimpleNamespace(
        TextContent=_FakeTextContent,
        TextResourceContents=_FakeTextResourceContents,
        BlobResourceContents=_FakeBlobResourceContents,
        ImageContent=_FakeImageContent,
        PaginatedRequestParams=SimpleNamespace,
    )

    class _FakeStdioServerParameters:
        def __init__(
            self,
            command: str,
            args: list[str],
            env: dict | None = None,
            cwd: str | None = None,
        ) -> None:
            self.command = command
            self.args = args
            self.env = env
            self.cwd = cwd

    class _FakeClientSession:
        def __init__(self, _read: object, _write: object) -> None:
            self._session = fake_mcp_runtime["session"]

        async def __aenter__(self) -> object:
            return self._session

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    @asynccontextmanager
    async def _fake_stdio_client(_params: object):
        yield object(), object()

    @asynccontextmanager
    async def _fake_sse_client(_url: str, httpx_client_factory=None):
        yield object(), object()

    @asynccontextmanager
    async def _fake_streamable_http_client(_url: str, http_client=None):
        yield object(), object(), object()

    mod.ClientSession = _FakeClientSession
    mod.StdioServerParameters = _FakeStdioServerParameters
    monkeypatch.setitem(sys.modules, "mcp", mod)

    client_mod = ModuleType("mcp.client")
    stdio_mod = ModuleType("mcp.client.stdio")
    stdio_mod.stdio_client = _fake_stdio_client
    sse_mod = ModuleType("mcp.client.sse")
    sse_mod.sse_client = _fake_sse_client
    streamable_http_mod = ModuleType("mcp.client.streamable_http")
    streamable_http_mod.streamable_http_client = _fake_streamable_http_client

    monkeypatch.setitem(sys.modules, "mcp.client", client_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.sse", sse_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", streamable_http_mod)

    shared_mod = ModuleType("mcp.shared")
    exc_mod = ModuleType("mcp.shared.exceptions")

    class _FakeMcpError(Exception):
        def __init__(self, code: int = -1, message: str = "error"):
            self.error = SimpleNamespace(code=code, message=message)
            super().__init__(message)

    exc_mod.McpError = _FakeMcpError
    monkeypatch.setitem(sys.modules, "mcp.shared", shared_mod)
    monkeypatch.setitem(sys.modules, "mcp.shared.exceptions", exc_mod)


def _make_wrapper(session: object, *, timeout: float = 0.1) -> MCPToolWrapper:
    tool_def = SimpleNamespace(
        name="demo",
        description="demo tool",
        inputSchema={"type": "object", "properties": {}},
    )
    return MCPToolWrapper(session, "test", tool_def, tool_timeout=timeout)


@pytest.mark.asyncio
async def test_mcp_provider_connect_propagates_external_cancellation(monkeypatch) -> None:
    started = asyncio.Event()

    async def connect_mcp_servers(_servers: dict, _registry: ToolRegistry) -> dict:
        started.set()
        await asyncio.sleep(60)
        return {}

    provider = MCPProvider(
        {"test": MCPServerConfig(command="fake")},
        ToolRegistry(),
    )
    monkeypatch.setattr(mcp_mod, "connect_mcp_servers", connect_mcp_servers)

    task = asyncio.create_task(provider.connect())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.connected_server_names == set()


@pytest.mark.parametrize( "value_schema, params, error",
    [
        pytest.param(
            True, {"value": {"count": "42", "enabled": False}}, None,
            id="true-property-preserves-value",
        ),
        pytest.param(
            False, {"value": None}, "value is not allowed by schema",
            id="false-property-rejects-null",
        ),
        pytest.param(False, {}, None, id="false-property-can-be-absent"),
        pytest.param(
            {"type": "array", "items": True},
            {"value": ["42", False, None, {"nested": [1]}]},
            None,
            id="true-items-preserve-values",
        ),
        pytest.param(
            {"type": "array", "items": False},
            {"value": [1]},
            "value[0] is not allowed by schema",
            id="false-items-reject-element",
        ),
        pytest.param(
            {"type": "array", "items": False}, {"value": []}, None,
            id="false-items-allow-empty-array",
        ),
        pytest.param(
            {"type": "object", "properties": {"blocked": False}},
            {"value": {"blocked": "x"}},
            "value.blocked is not allowed by schema",
            id="nested-false-property-reports-path",
        ),
    ],
)
async def test_registry_executes_mcp_tools_with_boolean_subschemas(
    value_schema, params, error,
) -> None:
    session = SimpleNamespace(
        call_tool=AsyncMock(return_value=SimpleNamespace(content=[_FakeTextContent("ok")])),
    )
    tool_def = SimpleNamespace(
        name="demo",
        description="demo tool",
        inputSchema={"type": "object", "properties": {"value": value_schema}},
    )
    wrapper = MCPToolWrapper(session, "test", tool_def)
    registry = ToolRegistry()
    registry.register(wrapper)

    result = await registry.execute(wrapper.name, params)

    if error is None:
        assert result == "ok"
        session.call_tool.assert_awaited_once_with("demo", arguments=params)
    else:
        assert is_tool_error(result)
        assert f"Invalid parameters for tool '{wrapper.name}': {error}" in result
        session.call_tool.assert_not_awaited()


@pytest.mark.parametrize("types", [["integer", "string"], ["string", "integer"]])
@pytest.mark.parametrize("value", ["00123", "doc-A", 42])
async def test_registry_preserves_mcp_type_union_arguments(types, value) -> None:
    session = SimpleNamespace(
        call_tool=AsyncMock(return_value=SimpleNamespace(content=[_FakeTextContent("ok")])),
    )
    tool_def = SimpleNamespace(
        name="lookup",
        description="Look up an id without changing its type or value.",
        inputSchema={
            "type": "object",
            "properties": {"id": {"type": types}},
            "required": ["id"],
        },
    )
    wrapper = MCPToolWrapper(session, "test", tool_def)
    registry = ToolRegistry()
    registry.register(wrapper)

    result = await registry.execute(wrapper.name, {"id": value})

    assert result == "ok"
    session.call_tool.assert_awaited_once_with("lookup", arguments={"id": value})
    assert type(session.call_tool.call_args.kwargs["arguments"]["id"]) is type(value)


def test_wrapper_preserves_non_nullable_unions() -> None:
    tool_def = SimpleNamespace(
        name="demo",
        description="demo tool",
        inputSchema={
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [{"type": "string"}, {"type": "integer"}],
                }
            },
        },
    )

    wrapper = MCPToolWrapper(SimpleNamespace(call_tool=None), "test", tool_def)

    assert wrapper.parameters["properties"]["value"]["anyOf"] == [
        {"type": "string"},
        {"type": "integer"},
    ]


def test_wrapper_normalizes_nullable_property_type_union() -> None:
    tool_def = SimpleNamespace(
        name="demo",
        description="demo tool",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
            },
        },
    )

    wrapper = MCPToolWrapper(SimpleNamespace(call_tool=None), "test", tool_def)

    assert wrapper.parameters["properties"]["name"] == {"type": "string", "nullable": True}


def test_wrapper_normalizes_nullable_property_anyof() -> None:
    tool_def = SimpleNamespace(
        name="demo",
        description="demo tool",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "optional name",
                },
            },
        },
    )

    wrapper = MCPToolWrapper(SimpleNamespace(call_tool=None), "test", tool_def)

    assert wrapper.parameters["properties"]["name"] == {
        "type": "string",
        "description": "optional name",
        "nullable": True,
    }


def test_wrapper_hoists_recursive_local_refs_into_defs() -> None:
    recursive_items_ref = "#/properties/filter/properties/items"
    tool_def = SimpleNamespace(
        name="search_dataset",
        description="search tool",
        inputSchema={
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {"$ref": recursive_items_ref},
                        }
                    },
                    "required": ["items"],
                }
            },
        },
    )

    wrapper = MCPToolWrapper(SimpleNamespace(call_tool=None), "test", tool_def)

    generated_ref = wrapper.parameters["properties"]["filter"]["properties"]["items"][
        "items"
    ]["$ref"]
    assert generated_ref.startswith("#/$defs/ref_")
    generated_name = generated_ref.removeprefix("#/$defs/")
    generated_schema = wrapper.parameters["$defs"][generated_name]
    assert generated_schema["type"] == "array"
    assert generated_schema["items"]["$ref"] == generated_ref


def test_wrapper_hoists_root_self_ref_into_defs() -> None:
    tool_def = SimpleNamespace(
        name="tree",
        description="tree tool",
        inputSchema={
            "type": "object",
            "properties": {
                "children": {"type": "array", "items": {"$ref": "#"}},
            },
        },
    )

    wrapper = MCPToolWrapper(SimpleNamespace(call_tool=None), "test", tool_def)

    generated_ref = wrapper.parameters["properties"]["children"]["items"]["$ref"]
    assert generated_ref.startswith("#/$defs/ref_")
    generated_name = generated_ref.removeprefix("#/$defs/")
    assert wrapper.parameters["$defs"][generated_name]["properties"]["children"]["items"] == {
        "$ref": generated_ref
    }


def test_wrapper_preserves_existing_defs_refs() -> None:
    tool_def = SimpleNamespace(
        name="demo",
        description="demo tool",
        inputSchema={
            "type": "object",
            "$defs": {"value": {"type": "string"}},
            "properties": {"value": {"$ref": "#/$defs/value"}},
        },
    )

    wrapper = MCPToolWrapper(SimpleNamespace(call_tool=None), "test", tool_def)

    assert wrapper.parameters["properties"]["value"]["$ref"] == "#/$defs/value"
    assert wrapper.parameters["$defs"]["value"]["type"] == "string"


def test_wrapper_resolves_uri_encoded_json_pointer() -> None:
    tool_def = SimpleNamespace(
        name="demo",
        description="demo tool",
        inputSchema={
            "type": "object",
            "properties": {
                "space name/value": {"type": "string"},
                "alias": {"$ref": "#/properties/space%20name~1value"},
            },
        },
    )

    wrapper = MCPToolWrapper(SimpleNamespace(call_tool=None), "test", tool_def)

    generated_ref = wrapper.parameters["properties"]["alias"]["$ref"]
    assert generated_ref.startswith("#/$defs/ref_")
    generated_name = generated_ref.removeprefix("#/$defs/")
    assert wrapper.parameters["$defs"][generated_name] == {"type": "string"}


def test_normalize_windows_stdio_command_is_noop_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_mod.os, "name", "posix", raising=False)

    command, args, env = _normalize_windows_stdio_command(
        "npx",
        ["-y", "chrome-devtools-mcp@latest"],
        {"FOO": "bar"},
    )

    assert command == "npx"
    assert args == ["-y", "chrome-devtools-mcp@latest"]
    assert env == {"FOO": "bar"}


def test_normalize_windows_stdio_command_wraps_npx_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_mod.os, "name", "nt", raising=False)
    monkeypatch.setattr(
        mcp_mod.shutil,
        "which",
        lambda command, path=None: r"C:\Program Files\nodejs\npx.cmd",
    )
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")

    command, args, env = _normalize_windows_stdio_command(
        "npx",
        ["-y", "chrome-devtools-mcp@latest"],
        None,
    )

    assert command == r"C:\Windows\System32\cmd.exe"
    assert args == ["/d", "/c", "npx", "-y", "chrome-devtools-mcp@latest"]
    assert env is None


def test_normalize_windows_stdio_command_wraps_resolved_cmd_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_mod.os, "name", "nt", raising=False)

    def _fake_which(command: str, path: str | None = None) -> str:
        assert command == "custom-launcher"
        assert path == r"C:\Tools"
        return r"C:\Tools\custom-launcher.cmd"

    monkeypatch.setattr(mcp_mod.shutil, "which", _fake_which)
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")

    command, args, _env = _normalize_windows_stdio_command(
        "custom-launcher",
        ["serve"],
        {"PATH": r"C:\Tools"},
    )

    assert command == r"C:\Windows\System32\cmd.exe"
    assert args == ["/d", "/c", "custom-launcher", "serve"]


def test_normalize_windows_stdio_command_keeps_real_executables_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_mod.os, "name", "nt", raising=False)

    command, args, env = _normalize_windows_stdio_command(
        "python.exe",
        ["-m", "http.server"],
        {"FOO": "bar"},
    )

    assert command == "python.exe"
    assert args == ["-m", "http.server"]
    assert env == {"FOO": "bar"}


def test_normalize_windows_stdio_command_skips_existing_shells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_mod.os, "name", "nt", raising=False)

    command, args, env = _normalize_windows_stdio_command(
        "cmd.exe",
        ["/c", "echo", "hello"],
        None,
    )

    assert command == "cmd.exe"
    assert args == ["/c", "echo", "hello"]
    assert env is None


@pytest.mark.asyncio
async def test_execute_returns_text_blocks() -> None:
    async def call_tool(_name: str, arguments: dict) -> object:
        assert arguments == {"value": 1}
        return SimpleNamespace(content=[_FakeTextContent("hello"), 42])

    wrapper = _make_wrapper(SimpleNamespace(call_tool=call_tool))

    result = await wrapper.execute(value=1)

    assert result == "hello\n42"


@pytest.mark.asyncio
async def test_execute_wraps_mcp_is_error_result() -> None:
    async def call_tool(_name: str, arguments: dict) -> object:
        return SimpleNamespace(
            content=[_FakeTextContent("Error: server-side MCP failure")],
            isError=True,
        )

    wrapper = _make_wrapper(SimpleNamespace(call_tool=call_tool))

    result = await wrapper.execute()

    assert result == "Error: server-side MCP failure"
    assert is_tool_error(result)


@pytest.mark.asyncio
async def test_execute_contains_malformed_success_result() -> None:
    async def call_tool(_name: str, arguments: dict) -> object:
        return SimpleNamespace(content=None)

    wrapper = _make_wrapper(SimpleNamespace(call_tool=call_tool))

    result = await wrapper.execute()

    assert result == "(MCP tool returned malformed content: TypeError)"
    assert is_tool_error(result)


@pytest.mark.asyncio
async def test_registry_adds_error_hint_to_malformed_mcp_result() -> None:
    async def call_tool(_name: str, arguments: dict) -> object:
        return SimpleNamespace(content=None)

    wrapper = _make_wrapper(SimpleNamespace(call_tool=call_tool))
    registry = ToolRegistry()
    registry.register(wrapper)

    result = await registry.execute(wrapper.name, {})

    assert is_tool_error(result)
    assert "MCP tool returned malformed content" in result
    assert (
        "[Consider this result in the context of the user's goal "
        "and decide whether and how to proceed.]"
    ) in result


@pytest.mark.asyncio
async def test_execute_preserves_success_text_that_starts_with_error() -> None:
    async def call_tool(_name: str, arguments: dict) -> object:
        return SimpleNamespace(
            content=[_FakeTextContent("Error: generated report successfully")],
            isError=False,
        )

    wrapper = _make_wrapper(SimpleNamespace(call_tool=call_tool))

    result = await wrapper.execute()

    assert result == "Error: generated report successfully"
    assert not is_tool_error(result)


# NanoClaw: nanobot saves image blocks as artifacts for its message tool
# (test_execute_persists_image_block_as_artifact); NanoClaw has neither, so
# every image becomes the note nanobot writes when saving fails, and the
# base64 never reaches the model.
@pytest.mark.asyncio
async def test_execute_replaces_image_block_with_note() -> None:
    async def call_tool(_name: str, arguments: dict) -> object:
        return SimpleNamespace(
            content=[_FakeTextContent("here you go"), _FakeImageContent(_PNG_B64, "image/png")]
        )

    wrapper = _make_wrapper(SimpleNamespace(call_tool=call_tool))

    result = await wrapper.execute(prompt="a cat")

    assert result == "here you go\n(MCP tool returned an image that could not be stored)"
    assert _PNG_B64 not in result
    assert not is_tool_error(result)


# Smallest valid 1x1 PNG, base64 without the data: prefix.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


@pytest.mark.asyncio
async def test_execute_notes_unstorable_image_block() -> None:
    async def call_tool(_name: str, arguments: dict) -> object:
        return SimpleNamespace(content=[_FakeImageContent("not-valid-base64!!", "image/png")])

    wrapper = _make_wrapper(SimpleNamespace(call_tool=call_tool))

    result = await wrapper.execute()

    assert result == "(MCP tool returned an image that could not be stored)"


@pytest.mark.asyncio
async def test_execute_returns_timeout_message() -> None:
    async def call_tool(_name: str, arguments: dict) -> object:
        await asyncio.sleep(1)
        return SimpleNamespace(content=[])

    wrapper = _make_wrapper(SimpleNamespace(call_tool=call_tool), timeout=0.01)

    result = await wrapper.execute()

    assert result == "(MCP tool call timed out after 0.01s)"
    assert is_tool_error(result)


@pytest.mark.asyncio
async def test_execute_handles_server_cancelled_error() -> None:
    async def call_tool(_name: str, arguments: dict) -> object:
        raise asyncio.CancelledError()

    wrapper = _make_wrapper(SimpleNamespace(call_tool=call_tool))

    result = await wrapper.execute()

    assert result == "(MCP tool call was cancelled)"
    assert is_tool_error(result)


@pytest.mark.asyncio
async def test_execute_re_raises_external_cancellation() -> None:
    started = asyncio.Event()

    async def call_tool(_name: str, arguments: dict) -> object:
        started.set()
        await asyncio.sleep(60)
        return SimpleNamespace(content=[])

    wrapper = _make_wrapper(SimpleNamespace(call_tool=call_tool), timeout=10)
    task = asyncio.create_task(wrapper.execute())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_execute_handles_generic_exception() -> None:
    async def call_tool(_name: str, arguments: dict) -> object:
        raise RuntimeError("boom")

    wrapper = _make_wrapper(SimpleNamespace(call_tool=call_tool))

    result = await wrapper.execute()

    assert result == "(MCP tool call failed: RuntimeError)"
    assert is_tool_error(result)


def _make_tool_def(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"{name} tool",
        inputSchema={"type": "object", "properties": {}},
    )


def _make_fake_session(tool_names: list[str]) -> SimpleNamespace:
    async def initialize() -> SimpleNamespace:
        return SimpleNamespace(capabilities=SimpleNamespace(tools=SimpleNamespace()))

    async def list_tools() -> SimpleNamespace:
        return SimpleNamespace(tools=[_make_tool_def(name) for name in tool_names], nextCursor=None)

    return SimpleNamespace(initialize=initialize, list_tools=list_tools)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled_tools", "expected_names"),
    [
        pytest.param(["*"], ["first", "second"], id="wildcard"),
        pytest.param(["second"], ["second"], id="raw-allowlist"),
        pytest.param(["mcp_test_second"], ["second"], id="wrapped-allowlist"),
        pytest.param([], [], id="deny-all"),
    ],
)
async def test_connect_mcp_servers_loads_all_tool_pages(
    fake_mcp_runtime: dict[str, object | None],
    enabled_tools: list[str],
    expected_names: list[str],
) -> None:
    cursors: list[str | None] = []
    pages = {
        None: SimpleNamespace(tools=[_make_tool_def("first")], nextCursor=""),
        "": SimpleNamespace(tools=[], nextCursor="opaque:+/="),
        "opaque:+/=": SimpleNamespace(tools=[_make_tool_def("second")], nextCursor=None),
    }

    async def list_tools(*, params: SimpleNamespace | None = None) -> SimpleNamespace:
        cursor = params.cursor if params is not None else None
        cursors.append(cursor)
        return pages[cursor]

    session = _make_fake_session([])
    session.list_tools = list_tools
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[_FakeTextContent("second page result")])
    )
    fake_mcp_runtime["session"] = session
    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {"test": MCPServerConfig(command="fake", enabled_tools=enabled_tools)}, registry,
    )
    try:
        assert set(stacks) == {"test"}
        assert cursors == [None, "", "opaque:+/="]
        expected = [f"mcp_test_{name}" for name in expected_names]
        assert registry.tool_names == expected
        assert [tool["function"]["name"] for tool in registry.get_definitions()] == expected
        if expected_names:
            assert await registry.execute("mcp_test_second", {}) == "second page result"
            session.call_tool.assert_awaited_once_with("second", arguments={})
        else:
            session.call_tool.assert_not_awaited()
    finally:
        for stack in stacks.values():
            await stack.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["request-error", "repeated-cursor"])
async def test_connect_mcp_servers_pagination_failure_registers_no_tools(
    fake_mcp_runtime: dict[str, object | None], failure: str,
) -> None:
    cursors: list[str | None] = []

    async def list_tools(*, params: SimpleNamespace | None = None) -> SimpleNamespace:
        cursor = params.cursor if params is not None else None
        cursors.append(cursor)
        if cursor is None:
            return SimpleNamespace(tools=[_make_tool_def("first")], nextCursor="next")
        if failure == "request-error":
            raise RuntimeError("second page failed")
        if len(cursors) > 2:
            raise AssertionError("A repeated cursor must not be requested again")
        return SimpleNamespace(tools=[_make_tool_def("second")], nextCursor="next")

    session = _make_fake_session([])
    session.list_tools = list_tools
    fake_mcp_runtime["session"] = session
    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {"test": MCPServerConfig(command="fake")}, registry,
    )
    try:
        assert cursors == [None, "next"]
        assert stacks == {}
        assert registry.tool_names == []
        assert registry.get_definitions() == []
    finally:
        for stack in stacks.values():
            await stack.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled_tool",
    [
        pytest.param("demo", id="raw_names"),
        pytest.param("mcp_test_demo", id="wrapped_names"),
    ],
)
async def test_connect_mcp_servers_enabled_tools_accepts_raw_and_wrapped_names(
    fake_mcp_runtime: dict[str,
    object | None],
    enabled_tool,
) -> None:
    fake_mcp_runtime["session"] = _make_fake_session(["demo", "other"])
    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {"test": MCPServerConfig(command="fake", enabled_tools=[enabled_tool])},
        registry,
    )
    for stack in stacks.values():
        await stack.aclose()

    assert registry.tool_names == ["mcp_test_demo"]


@pytest.mark.asyncio
async def test_connect_mcp_servers_enabled_tools_defaults_to_all(
    fake_mcp_runtime: dict[str, object | None],
) -> None:
    fake_mcp_runtime["session"] = _make_fake_session(["demo", "other"])
    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {"test": MCPServerConfig(command="fake")},
        registry,
    )
    for stack in stacks.values():
        await stack.aclose()

    assert registry.tool_names == ["mcp_test_demo", "mcp_test_other"]


@pytest.mark.asyncio
async def test_connect_mcp_servers_enabled_tools_supports_limited_wrapped_names(
    fake_mcp_runtime: dict[str, object | None],
) -> None:
    long_tool_name = "tool-" + "very-long-name-" * 8
    wrapped_name = _sanitize_mcp_tool_name(f"mcp_test_{long_tool_name}")
    assert len(wrapped_name) == 64

    fake_mcp_runtime["session"] = _make_fake_session([long_tool_name, "other"])
    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {"test": MCPServerConfig(command="fake", enabled_tools=[wrapped_name])},
        registry,
    )
    for stack in stacks.values():
        await stack.aclose()

    assert registry.tool_names == [wrapped_name]


@pytest.mark.asyncio
async def test_connect_mcp_servers_enabled_tools_empty_list_registers_none(
    fake_mcp_runtime: dict[str, object | None],
) -> None:
    fake_mcp_runtime["session"] = _make_fake_session(["demo", "other"])
    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {"test": MCPServerConfig(command="fake", enabled_tools=[])},
        registry,
    )
    for stack in stacks.values():
        await stack.aclose()

    assert registry.tool_names == []


@pytest.mark.asyncio
async def test_connect_mcp_servers_enabled_tools_empty_list_blocks_resources_and_prompts(
    fake_mcp_runtime: dict[str, object | None],
) -> None:
    """enabledTools: [] (deny-all) must also block resource and prompt registration."""
    fake_mcp_runtime["session"] = _make_fake_session_with_capabilities(
        tool_names=["demo"],
        resource_names=["secret_data"],
        prompt_names=["admin_prompt"],
    )
    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {"test": MCPServerConfig(command="fake", enabled_tools=[])},
        registry,
    )
    for stack in stacks.values():
        await stack.aclose()

    assert registry.tool_names == []
    # Resources and prompts must also be blocked
    assert not any("secret_data" in name for name in registry.tool_names)
    assert not any("admin_prompt" in name for name in registry.tool_names)


@pytest.mark.asyncio
async def test_connect_mcp_servers_enabled_tools_specific_list_blocks_resources_and_prompts(
    fake_mcp_runtime: dict[str, object | None],
) -> None:
    """enabledTools with specific tool names must not leak resources or prompts."""
    fake_mcp_runtime["session"] = _make_fake_session_with_capabilities(
        tool_names=["demo", "other"],
        resource_names=["secret_data"],
        prompt_names=["admin_prompt"],
    )
    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {"test": MCPServerConfig(command="fake", enabled_tools=["demo"])},
        registry,
    )
    for stack in stacks.values():
        await stack.aclose()

    # Only the allowed tool should be registered
    assert "mcp_test_demo" in registry.tool_names
    assert "mcp_test_other" not in registry.tool_names
    # Resources and prompts must not leak
    assert not any("secret_data" in name for name in registry.tool_names)
    assert not any("admin_prompt" in name for name in registry.tool_names)


@pytest.mark.asyncio
async def test_connect_mcp_servers_enabled_tools_warns_on_unknown_entries(
    fake_mcp_runtime: dict[str, object | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mcp_runtime["session"] = _make_fake_session(["demo"])
    registry = ToolRegistry()
    warnings: list[str] = []

    def _warning(message: str, *args: object) -> None:
        warnings.append(message.format(*args))

    monkeypatch.setattr("nanoclaw.agent.tools.mcp.logger.warning", _warning)

    stacks = await connect_mcp_servers(
        {"test": MCPServerConfig(command="fake", enabled_tools=["unknown"])},
        registry,
    )
    for stack in stacks.values():
        await stack.aclose()

    assert registry.tool_names == []
    assert warnings
    assert "enabledTools entries not found: unknown" in warnings[-1]
    assert "Available raw names: demo" in warnings[-1]
    assert "Available wrapped names: mcp_test_demo" in warnings[-1]


@pytest.mark.asyncio
async def test_connect_mcp_servers_logs_stdio_pollution_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []

    @asynccontextmanager
    async def _broken_stdio_client(_params: object):
        raise RuntimeError("Parse error: Unexpected token 'INFO' before JSON-RPC headers")
        yield  # pragma: no cover

    monkeypatch.setattr(sys.modules["mcp.client.stdio"], "stdio_client", _broken_stdio_client)
    sink = mcp_mod.logger.add(
        lambda message: messages.append(message.record["message"]), level="ERROR"
    )

    registry = ToolRegistry()
    try:
        stacks = await connect_mcp_servers(
            {"gh": MCPServerConfig(command="github-mcp")}, registry
        )
    finally:
        mcp_mod.logger.remove(sink)

    assert stacks == {}
    assert messages
    assert "stdio protocol pollution" in messages[-1]
    assert "stdout" in messages[-1]
    assert "stderr" in messages[-1]


def test_transient_connection_group_logs_brief_warning_and_debug_trace() -> None:
    records: list[dict] = []
    sink = mcp_mod.logger.add(lambda message: records.append(message.record), level="DEBUG")
    error = ExceptionGroup("transport failed", [httpx.ConnectError("")])
    try:
        mcp_mod._log_mcp_connection_failure("notion", error)
    finally:
        mcp_mod.logger.remove(sink)

    warning = next(record for record in records if record["level"].name == "WARNING")
    debug = next(record for record in records if record["level"].name == "DEBUG")
    assert warning["exception"] is None
    assert "transient connection failure" in warning["message"]
    assert debug["exception"] is not None
    assert not any(record["level"].name == "ERROR" for record in records)


def test_unexpected_connection_failure_keeps_error_trace() -> None:
    records: list[dict] = []
    sink = mcp_mod.logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        mcp_mod._log_mcp_connection_failure("notion", RuntimeError("boom"))
    finally:
        mcp_mod.logger.remove(sink)

    error = next(record for record in records if record["level"].name == "ERROR")
    assert error["exception"] is not None
    assert not any(record["level"].name == "WARNING" for record in records)


@pytest.mark.parametrize("transient", [False, True])
def test_connection_failure_retains_useful_trace_without_local_values(transient: bool) -> None:
    messages: list[str] = []
    records: list[dict] = []

    def capture(message):
        messages.append(str(message))
        records.append(message.record)

    sink = mcp_mod.logger.add(
        capture, level="DEBUG", format="{level} {message}", diagnose=True, backtrace=True,
    )

    def fail_transport():
        local_credential = "unrelated-local-credential"
        error = httpx.ConnectError if transient else RuntimeError
        raise error(f"handshake failed (credential length {len(local_credential)})")

    try:
        try:
            fail_transport()
        except Exception as exc:
            mcp_mod._log_mcp_connection_failure("remote", exc, " check server configuration")
    finally:
        mcp_mod.logger.remove(sink)

    output = "\n".join(messages)
    assert "unrelated-local-credential" not in output
    assert "handshake failed (credential length 26)" in output
    assert "in fail_transport" in output
    assert "Traceback (most recent call last)" in output
    detail = next(record for record in records if record["exception"] is not None)
    assert detail["exception"].traceback is None
    assert detail["exception"].value.__traceback__ is None
    assert detail["exception"].value.__context__ is None
    assert detail["level"].name == ("DEBUG" if transient else "ERROR")
    if transient:
        assert "ConnectError" in output
        assert "transient connection failure" in output
    else:
        assert "RuntimeError" in output
        assert "check server configuration" in output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        MCPServerConfig(url="http://127.0.0.1:9/sse"),
        MCPServerConfig(type="streamableHttp", url="http://127.0.0.1:9/mcp"),
        MCPServerConfig(url="http://169.254.169.254/sse"),
        MCPServerConfig(type="streamableHttp", url="http://169.254.169.254/mcp"),
    ],
)
@pytest.mark.parametrize("use_proxy", [False, True])
async def test_connect_mcp_servers_rejects_unsafe_http_urls_before_probe(
    use_proxy: bool,
    config: MCPServerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_connections: list[tuple[object, ...]] = []
    warnings: list[str] = []

    async def _open_connection(*args: object, **_kwargs: object):
        attempted_connections.append(args)
        raise AssertionError("unsafe MCP URL should be rejected before TCP probe")

    def _warning(message: str, *args: object) -> None:
        warnings.append(message.format(*args))

    monkeypatch.setattr(mcp_mod.asyncio, "open_connection", _open_connection)
    monkeypatch.setattr("nanoclaw.agent.tools.mcp.logger.warning", _warning)

    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {"local": config.model_copy(update={"use_proxy": use_proxy})},
        registry,
    )

    assert stacks == {}
    assert registry.tool_names == []
    assert attempted_connections == []
    assert any("blocked unsafe URL" in warning for warning in warnings)


@pytest.mark.asyncio
async def test_validate_mcp_request_url_rejects_loopback_without_whitelist() -> None:
    from nanoclaw.security.network import configure_ssrf_whitelist

    configure_ssrf_whitelist([])
    request = httpx.Request("GET", "http://127.0.0.1/private")

    with pytest.raises(httpx.RequestError, match="Blocked unsafe MCP URL"):
        await mcp_mod._validate_mcp_request_url(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        MCPServerConfig(type="sse", url="https://mcp.example.com/sse"),
        MCPServerConfig(type="streamableHttp", url="https://mcp.example.com/mcp"),
    ],
)
async def test_connect_mcp_servers_env_proxy_adds_proxy_mounts_and_keeps_pinned_transport(
    config: MCPServerConfig,
    fake_mcp_runtime: dict[str, object | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp_runtime["session"] = _make_fake_session(["demo"])
    client_kwargs: list[dict[str, object]] = []

    async def _reachable(_url: str, *, proxy: str | None = None) -> bool:
        return True

    def _validate(_url: str) -> tuple[bool, str]:
        return True, ""

    class FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            client_kwargs.append(kwargs)

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    @asynccontextmanager
    async def _capturing_sse_client(_url: str, httpx_client_factory=None):
        assert httpx_client_factory is not None
        async with httpx_client_factory():
            pass
        yield object(), object()

    @asynccontextmanager
    async def _capturing_streamable_http_client(_url: str, http_client=None):
        assert http_client is not None
        yield object(), object(), object()

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,::1")
    monkeypatch.setattr(mcp_mod, "validate_url_target", _validate)
    monkeypatch.setattr(mcp_mod, "_probe_http_url", _reachable)
    monkeypatch.setattr(
        mcp_mod,
        "PinnedDNSAsyncTransport",
        lambda: httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
    )
    monkeypatch.setattr(
        "nanoclaw.security.network.httpx.AsyncHTTPTransport",
        lambda **_kwargs: httpx.MockTransport(
            lambda request: httpx.Response(200, request=request)
        ),
    )
    monkeypatch.setattr(mcp_mod.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(sys.modules["mcp.client.sse"], "sse_client", _capturing_sse_client)
    monkeypatch.setattr(
        sys.modules["mcp.client.streamable_http"],
        "streamable_http_client",
        _capturing_streamable_http_client,
    )

    registry = ToolRegistry()
    stacks = await connect_mcp_servers({"remote": config}, registry)
    for stack in stacks.values():
        await stack.aclose()

    assert client_kwargs
    assert all("transport" in kwargs for kwargs in client_kwargs)
    assert all("mounts" in kwargs for kwargs in client_kwargs)


def test_mcp_http_clients_no_proxy_env_keeps_pinned_direct_route(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("NO_PROXY", "mcp.example.com")
    monkeypatch.setattr(
        mcp_mod,
        "PinnedDNSAsyncTransport",
        lambda: httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
    )
    monkeypatch.setattr(
        "nanoclaw.security.network.httpx.AsyncHTTPTransport",
        lambda **_kwargs: httpx.MockTransport(
            lambda request: httpx.Response(200, request=request)
        ),
    )

    kwargs = mcp_mod._pinned_transport_kwargs()

    assert "transport" in kwargs
    assert any(transport is None for transport in kwargs["mounts"].values())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "expected_transport"),
    [
        (MCPServerConfig(type="sse", url="https://mcp.example.com/sse"), "sse"),
        (
            MCPServerConfig(type="streamableHttp", url="https://mcp.example.com/mcp"),
            "streamableHttp",
        ),
    ],
)
@pytest.mark.parametrize("use_proxy", [False, True])
async def test_connect_mcp_servers_http_clients_reject_unsafe_redirect_targets(
    use_proxy: bool,
    config: MCPServerConfig,
    expected_transport: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_urls: list[str] = []
    sent_urls: list[str] = []
    used_transports: list[str] = []

    def _validate(url: str, **_kwargs: object) -> tuple[bool, str]:
        checked_urls.append(url)
        if url == "http://127.0.0.1/private":
            return False, "loopback blocked"
        return True, ""

    async def _reachable(_url: str, *, proxy: str | None = None) -> bool:
        return True

    def _handler(request: httpx.Request) -> httpx.Response:
        sent_urls.append(str(request.url))
        if str(request.url) == "https://example.com/start":
            return httpx.Response(
                302,
                headers={"Location": "http://127.0.0.1/private"},
                request=request,
            )
        raise AssertionError("unsafe redirect target should be blocked before transport")

    original_async_client = httpx.AsyncClient

    def _async_client_with_mock_transport(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs.setdefault("transport", httpx.MockTransport(_handler))
        return original_async_client(*args, **kwargs)

    @asynccontextmanager
    async def _fake_sse_client(_url: str, httpx_client_factory=None):
        assert httpx_client_factory is not None
        used_transports.append("sse")
        async with httpx_client_factory() as client:
            await client.get("https://example.com/start")
        yield object(), object()

    @asynccontextmanager
    async def _fake_streamable_http_client(_url: str, http_client=None):
        assert http_client is not None
        used_transports.append("streamableHttp")
        await http_client.get("https://example.com/start")
        yield object(), object(), object()

    monkeypatch.setattr(mcp_mod, "validate_url_target", _validate)
    monkeypatch.setattr(mcp_mod, "_probe_http_url", _reachable)
    monkeypatch.setattr(
        mcp_mod,
        "PinnedDNSAsyncTransport",
        lambda **_kwargs: httpx.MockTransport(_handler),
    )
    monkeypatch.setattr(
        mcp_mod.httpx,
        "AsyncHTTPTransport",
        lambda **_kwargs: httpx.MockTransport(_handler),
    )
    monkeypatch.setattr(mcp_mod.httpx, "AsyncClient", _async_client_with_mock_transport)
    monkeypatch.setattr(sys.modules["mcp.client.sse"], "sse_client", _fake_sse_client)
    monkeypatch.setattr(
        sys.modules["mcp.client.streamable_http"],
        "streamable_http_client",
        _fake_streamable_http_client,
    )

    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {"remote": config.model_copy(update={"use_proxy": use_proxy})},
        registry,
    )

    assert stacks == {}
    assert registry.tool_names == []
    assert used_transports == [expected_transport]
    assert checked_urls == [
        config.url,
        "https://example.com/start",
        "http://127.0.0.1/private",
    ]
    assert sent_urls == ["https://example.com/start"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["exception", "cancellation"])
async def test_connect_mcp_servers_one_failure_does_not_block_others(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    bad_session = _make_fake_session([])

    async def _cancel_initialize() -> None:
        raise asyncio.CancelledError("cancelled by SDK")

    if failure_mode == "cancellation":
        bad_session.initialize = _cancel_initialize
    sessions = {
        "bad": bad_session,
        "good": _make_fake_session(["demo"]),
    }

    class _SelectiveClientSession:
        def __init__(self, read: object, _write: object) -> None:
            self._session = sessions[read]

        async def __aenter__(self) -> object:
            return self._session

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    @asynccontextmanager
    async def _selective_stdio_client(params: object):
        if params.command == "bad" and failure_mode == "exception":
            raise RuntimeError("boom")
        yield params.command, object()

    monkeypatch.setattr(sys.modules["mcp"], "ClientSession", _SelectiveClientSession)
    monkeypatch.setattr(sys.modules["mcp.client.stdio"], "stdio_client", _selective_stdio_client)

    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {
            "bad": MCPServerConfig(command="bad"),
            "good": MCPServerConfig(command="good"),
        },
        registry,
    )
    for stack in stacks.values():
        await stack.aclose()

    assert registry.tool_names == ["mcp_good_demo"]
    assert set(stacks) == {"good"}


@pytest.mark.asyncio
async def test_connect_mcp_servers_propagates_external_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    closed = asyncio.Event()

    @asynccontextmanager
    async def _blocking_stdio_client(_params: object):
        try:
            started.set()
            await asyncio.Event().wait()
            yield object(), object()
        finally:
            closed.set()

    monkeypatch.setattr(sys.modules["mcp.client.stdio"], "stdio_client", _blocking_stdio_client)

    task = asyncio.create_task(
        connect_mcp_servers({"slow": MCPServerConfig(command="slow")}, ToolRegistry())
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(closed.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_connect_mcp_servers_rolls_back_completed_batch_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slow_started = asyncio.Event()
    closed: list[str] = []
    sessions = {"fast": _make_fake_session(["demo"])}

    class _SelectiveClientSession:
        def __init__(self, read: object, _write: object) -> None:
            self._session = sessions[str(read)]

        async def __aenter__(self) -> object:
            return self._session

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    @asynccontextmanager
    async def _selective_stdio_client(params: object):
        command = str(params.command)
        try:
            if command == "slow":
                slow_started.set()
                await asyncio.Event().wait()
            yield command, object()
        finally:
            closed.append(command)

    monkeypatch.setattr(sys.modules["mcp"], "ClientSession", _SelectiveClientSession)
    monkeypatch.setattr(sys.modules["mcp.client.stdio"], "stdio_client", _selective_stdio_client)

    registry = ToolRegistry()
    task = asyncio.create_task(
        connect_mcp_servers(
            {
                "fast": MCPServerConfig(command="fast"),
                "slow": MCPServerConfig(command="slow"),
            },
            registry,
        )
    )
    await asyncio.wait_for(slow_started.wait(), timeout=1.0)
    assert registry.tool_names == ["mcp_fast_demo"]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert registry.tool_names == []
    assert sorted(closed) == ["fast", "slow"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_timeout", "expected_read"),
    [
        (30, 300.0),  # default: match the MCP SDK's SSE read timeout
        (600, 600.0),  # long-running tools: read must not cut the call short
    ],
)
async def test_connect_mcp_servers_streamable_http_uses_finite_timeout(
    fake_mcp_runtime: dict[str, object | None],
    monkeypatch: pytest.MonkeyPatch,
    tool_timeout: int,
    expected_read: float,
) -> None:
    fake_mcp_runtime["session"] = _make_fake_session(["demo"])
    captured: dict[str, object] = {}

    async def _reachable(_url: str, *, proxy: str | None = None) -> bool:
        return True

    def _validate(_url: str) -> tuple[bool, str]:
        return True, ""

    @asynccontextmanager
    async def _capturing_streamable_http_client(_url: str, http_client=None):
        captured["timeout"] = http_client.timeout
        yield object(), object(), object()

    monkeypatch.setattr(mcp_mod, "validate_url_target", _validate)
    monkeypatch.setattr(mcp_mod, "_probe_http_url", _reachable)
    monkeypatch.setattr(
        mcp_mod,
        "PinnedDNSAsyncTransport",
        lambda: httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
    )
    monkeypatch.setattr(
        sys.modules["mcp.client.streamable_http"],
        "streamable_http_client",
        _capturing_streamable_http_client,
    )

    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {
            "test": MCPServerConfig(
                url="https://mcp.example.com/mcp", tool_timeout=tool_timeout
            )
        },
        registry,
    )
    for stack in stacks.values():
        await stack.aclose()

    timeout = captured["timeout"]
    assert timeout.connect == 10.0
    assert timeout.read == expected_read
    assert timeout.write == 30.0
    assert timeout.pool == 30.0


@pytest.mark.asyncio
async def test_connect_mcp_servers_wraps_windows_stdio_launchers(
    fake_mcp_runtime: dict[str, object | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp_runtime["session"] = _make_fake_session(["demo"])
    captured: dict[str, object] = {}

    @asynccontextmanager
    async def _capturing_stdio_client(params: object):
        captured["command"] = params.command
        captured["args"] = params.args
        captured["env"] = params.env
        yield object(), object()

    monkeypatch.setattr(mcp_mod.os, "name", "nt", raising=False)
    monkeypatch.setattr(
        mcp_mod.shutil,
        "which",
        lambda command, path=None: r"C:\Program Files\nodejs\npx.cmd",
    )
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")
    monkeypatch.setattr(sys.modules["mcp.client.stdio"], "stdio_client", _capturing_stdio_client)

    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {
            "test": MCPServerConfig(
                command="npx",
                args=["-y", "chrome-devtools-mcp@latest"],
            )
        },
        registry,
    )
    for stack in stacks.values():
        await stack.aclose()

    assert captured["command"] == r"C:\Windows\System32\cmd.exe"
    assert captured["args"] == ["/d", "/c", "npx", "-y", "chrome-devtools-mcp@latest"]
    assert captured["env"] is None


@pytest.mark.asyncio
async def test_connect_mcp_servers_passes_stdio_cwd(
    fake_mcp_runtime: dict[str, object | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp_runtime["session"] = _make_fake_session(["demo"])
    captured: dict[str, object] = {}

    @asynccontextmanager
    async def _capturing_stdio_client(params: object):
        captured["cwd"] = params.cwd
        yield object(), object()

    monkeypatch.setattr(sys.modules["mcp.client.stdio"], "stdio_client", _capturing_stdio_client)

    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {"test": MCPServerConfig(command="fake", cwd="/tmp/nanobot-mcp-test")},
        registry,
    )
    for stack in stacks.values():
        await stack.aclose()

    assert captured["cwd"] == "/tmp/nanobot-mcp-test"


# Fake session that also offers resources and prompts (registered by nanobot
# only under enabledTools ["*"]; never by NanoClaw).


def _make_fake_session_with_capabilities(
    tool_names: list[str],
    resource_names: list[str] | None = None,
    prompt_names: list[str] | None = None,
) -> SimpleNamespace:
    async def initialize() -> SimpleNamespace:
        return SimpleNamespace(capabilities=SimpleNamespace(tools=SimpleNamespace()))

    async def list_tools() -> SimpleNamespace:
        return SimpleNamespace(tools=[_make_tool_def(name) for name in tool_names], nextCursor=None)

    async def list_resources() -> SimpleNamespace:
        resources = []
        for rname in resource_names or []:
            resources.append(
                SimpleNamespace(
                    name=rname,
                    uri=f"file:///{rname}",
                    description=f"{rname} resource",
                )
            )
        return SimpleNamespace(resources=resources, nextCursor=None)

    async def list_prompts() -> SimpleNamespace:
        prompts = []
        for pname in prompt_names or []:
            prompts.append(
                SimpleNamespace(
                    name=pname,
                    description=f"{pname} prompt",
                    arguments=None,
                )
            )
        return SimpleNamespace(prompts=prompts, nextCursor=None)

    return SimpleNamespace(
        initialize=initialize,
        list_tools=list_tools,
        list_resources=list_resources,
        list_prompts=list_prompts,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("catalog", ["resources", "prompts"])
async def test_connect_accepts_servers_without_tools_capability(
    fake_mcp_runtime: dict[str, object | None], catalog: str,
) -> None:
    from mcp.shared.exceptions import McpError

    session = _make_fake_session_with_capabilities(
        tool_names=[],
        resource_names=["document"] if catalog == "resources" else [],
        prompt_names=["review"] if catalog == "prompts" else [],
    )
    session.initialize = AsyncMock(
        return_value=SimpleNamespace(capabilities=SimpleNamespace(tools=None)),
    )
    session.list_tools = AsyncMock(side_effect=McpError(code=-32601, message="Method not found"))
    fake_mcp_runtime["session"] = session
    registry = ToolRegistry()
    stacks = await connect_mcp_servers({"test": MCPServerConfig(command="fake")}, registry)
    try:
        # NanoClaw registers tools only: the server connects with none.
        assert set(stacks) == {"test"}
        assert registry.tool_names == []
        session.list_tools.assert_not_awaited()
    finally:
        for stack in stacks.values():
            await stack.aclose()


# ---------------------------------------------------------------------------
# _sanitize_name tests
# ---------------------------------------------------------------------------


def test_sanitize_name_replaces_spaces() -> None:
    assert _sanitize_name("PostgreSQL System Information") == "PostgreSQL_System_Information"


def test_sanitize_name_replaces_special_characters() -> None:
    assert _sanitize_name("foo.bar@baz!") == "foo_bar_baz_"


def test_sanitize_name_collapses_consecutive_underscores() -> None:
    assert _sanitize_name("a   b") == "a_b"


def test_sanitize_name_preserves_valid_characters() -> None:
    assert _sanitize_name("my-tool_v2") == "my-tool_v2"


def test_sanitize_name_noop_for_already_clean_names() -> None:
    assert _sanitize_name("mcp_server_tool") == "mcp_server_tool"


# ---------------------------------------------------------------------------
# Wrapper sanitization tests
# ---------------------------------------------------------------------------


def test_tool_wrapper_sanitizes_name() -> None:
    tool_def = SimpleNamespace(
        name="My Tool",
        description="tool with spaces",
        inputSchema={"type": "object", "properties": {}},
    )
    wrapper = MCPToolWrapper(SimpleNamespace(call_tool=None), "srv", tool_def)
    assert wrapper.name == "mcp_srv_My_Tool"


def test_tool_wrapper_preserves_original_name_for_mcp_call() -> None:
    tool_def = SimpleNamespace(
        name="My Tool",
        description="tool with spaces",
        inputSchema={"type": "object", "properties": {}},
    )
    wrapper = MCPToolWrapper(SimpleNamespace(call_tool=None), "srv", tool_def)
    # The sanitized API-facing name differs from the original MCP name
    assert wrapper.name == "mcp_srv_My_Tool"
    assert wrapper._original_name == "My Tool"


@pytest.mark.asyncio
async def test_connect_mcp_servers_enabled_tools_matches_sanitized_name(
    fake_mcp_runtime: dict[str, object | None],
) -> None:
    fake_mcp_runtime["session"] = _make_fake_session_with_capabilities(
        tool_names=["My Tool", "other"],
    )
    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {"test": MCPServerConfig(command="fake", enabled_tools=["mcp_test_My_Tool"])},
        registry,
    )
    for stack in stacks.values():
        await stack.aclose()

    assert registry.tool_names == ["mcp_test_My_Tool"]


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://user:secret@host.example/sse", "https://host.example/..."),
        ("https://host.example:8443/mcp?token=abc#frag", "https://host.example:8443/..."),
        ("https://user:secret@[::1]:8443/sse?token=abc", "https://[::1]:8443/..."),
        ("https://host.example/sse", "https://host.example/..."),
        ("https://host.example", "https://host.example"),
        ("https://host.example/", "https://host.example/"),
    ],
)
def test_redact_url_strips_credentials_and_query(url: str, expected: str) -> None:
    assert mcp_mod._redact_url(url) == expected

def test_mcp_tool_name_keeps_short_name():
    name = _sanitize_mcp_tool_name("mcp_myserver_resource_myres")
    assert name == "mcp_myserver_resource_myres"


def test_mcp_tool_name_limits_long_name():
    long_name = "mcp_" + "a" * 100
    name = _sanitize_mcp_tool_name(long_name)

    assert len(name) <= 64
    assert name.startswith("mcp_")


def test_long_server_name_tools_are_matched_by_server_name() -> None:
    server_name = "very-long-server-name-" * 4
    tool_def = SimpleNamespace(
        name="search",
        description="search tool",
        inputSchema={"type": "object", "properties": {}},
    )
    other_tool_def = SimpleNamespace(
        name="search",
        description="other search tool",
        inputSchema={"type": "object", "properties": {}},
    )
    wrapper = MCPToolWrapper(SimpleNamespace(call_tool=None), server_name, tool_def)
    other_wrapper = MCPToolWrapper(SimpleNamespace(call_tool=None), "other", other_tool_def)
    registry = ToolRegistry()
    registry.register(wrapper)
    registry.register(other_wrapper)

    assert len(wrapper.name) == 64
    assert not wrapper.name.startswith(mcp_mod._tool_prefix(server_name))

    provider = MCPProvider(
        {server_name: MCPServerConfig(command="fake")},
        registry,
    )
    provider._attach_reconnect_handlers({server_name})
    assert wrapper._reconnect is not None
    assert other_wrapper._reconnect is None

    removed = mcp_mod._unregister_server_tools(registry, server_name)

    assert removed == 1
    assert wrapper.name not in registry.tool_names
    assert other_wrapper.name in registry.tool_names


@pytest.mark.parametrize("params, error", [
    ({"team": "nanobot", "query": "bug"}, None),
    ({"team": "nanobot", "customView": "saved-view"}, None),
    ({"query": "bug"}, "missing required team"),
    ({"team": "nanobot", "customView": ""}, "customView must be at least 1 chars"),
])
async def test_optional_mcp_filters_reach_server_unchanged(params, error):
    async def call_tool(name, arguments):
        assert name == "list_issues"
        assert not ({"query", "customView"} <= arguments.keys())
        return SimpleNamespace(content=[_FakeTextContent("ok")])

    session = SimpleNamespace(call_tool=AsyncMock(side_effect=call_tool))
    wrapper = MCPToolWrapper(session, "linear", SimpleNamespace(
        name="list_issues",
        description="Search issues or use a saved view",
        inputSchema={
            "type": "object",
            "properties": {
                "team": {"type": "string"},
                "query": {"type": "string"},
                "customView": {"type": "string", "minLength": 1},
            },
            "required": ["team"],
        },
    ))
    registry = ToolRegistry()
    registry.register(wrapper)

    result = await registry.execute(wrapper.name, params)

    if error:
        assert is_tool_error(result)
        assert error in result
        session.call_tool.assert_not_awaited()
    else:
        assert result == "ok"
        session.call_tool.assert_awaited_once_with("list_issues", arguments=params)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["sse", "streamableHttp"])
@pytest.mark.parametrize("use_proxy", [False, True], ids=["environment-route", "use-proxy-ignored"])
async def test_mcp_http_requests_use_selected_transport_without_route_fallback(
    fake_mcp_runtime,
    monkeypatch: pytest.MonkeyPatch,
    transport: str,
    use_proxy: bool,
) -> None:
    fake_mcp_runtime["session"] = _make_fake_session(["demo"])
    routes: list[str] = []
    constructed_proxies: list[str] = []
    shared_proxy = "http://shared.example:8080"

    def recording_transport(route: str) -> httpx.MockTransport:
        def respond(request: httpx.Request) -> httpx.Response:
            routes.append(route)
            return httpx.Response(200, json={"ok": True}, request=request)
        return httpx.MockTransport(respond)

    def proxy_transport(*, proxy: str) -> httpx.MockTransport:
        constructed_proxies.append(proxy)
        return recording_transport("shared")

    async def probe(_url):
        return True

    @asynccontextmanager
    async def sse_client(url, *, httpx_client_factory):
        async with httpx_client_factory() as client:
            response = await client.get(url)
            assert response.json() == {"ok": True}
        yield object(), object()

    @asynccontextmanager
    async def streamable_client(url, *, http_client):
        response = await http_client.get(url)
        assert response.json() == {"ok": True}
        yield object(), object(), object()

    monkeypatch.setenv("HTTPS_PROXY", "http://environment.example:8888")
    monkeypatch.setattr(mcp_mod, "validate_url_target", lambda _url: (True, ""))
    monkeypatch.setattr(mcp_mod, "_probe_http_url", probe)
    monkeypatch.setattr(mcp_mod, "PinnedDNSAsyncTransport", lambda: recording_transport("direct"))
    monkeypatch.setattr(
        mcp_mod, "httpx_env_proxy_mounts", lambda: {"all://": recording_transport("environment")},
    )
    monkeypatch.setattr(mcp_mod.httpx, "AsyncHTTPTransport", proxy_transport)
    monkeypatch.setattr(sys.modules["mcp.client.sse"], "sse_client", sse_client)
    monkeypatch.setattr(sys.modules["mcp.client.streamable_http"], "streamable_http_client", streamable_client)

    registry = ToolRegistry()
    connections = await connect_mcp_servers(
        {"remote": MCPServerConfig(
            type=transport, url="https://93.184.216.34/mcp", use_proxy=use_proxy,
        )},
        registry,
    )
    try:
        assert set(connections) == {"remote"}
        assert registry.has("mcp_remote_demo")
        # NanoClaw has no shared proxy: useProxy is ignored and the
        # environment route is used either way.
        assert routes == ["environment"]
        assert constructed_proxies == []
    finally:
        for connection in connections.values():
            await connection.aclose()


# ---------------------------------------------------------------------------
# Pinned by mutation checks (layer 17)
# ---------------------------------------------------------------------------


def _record_transports(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    used: list[str] = []

    @asynccontextmanager
    async def stdio_client(_params):
        used.append("stdio")
        yield object(), object()

    @asynccontextmanager
    async def sse_client(_url, httpx_client_factory=None):
        used.append("sse")
        yield object(), object()

    @asynccontextmanager
    async def streamable_http_client(_url, http_client=None):
        used.append("streamableHttp")
        yield object(), object(), object()

    async def reachable(_url):
        return True

    monkeypatch.setattr(sys.modules["mcp.client.stdio"], "stdio_client", stdio_client)
    monkeypatch.setattr(sys.modules["mcp.client.sse"], "sse_client", sse_client)
    monkeypatch.setattr(sys.modules["mcp.client.streamable_http"], "streamable_http_client", streamable_http_client)
    monkeypatch.setattr(mcp_mod, "validate_url_target", lambda _url: (True, ""))
    monkeypatch.setattr(mcp_mod, "_probe_http_url", reachable)
    return used


def _capture_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    warnings: list[str] = []
    monkeypatch.setattr(
        "nanoclaw.agent.tools.mcp.logger.warning", lambda message, *args: warnings.append(message.format(*args))
    )
    return warnings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(MCPServerConfig(command="fake"), "stdio", id="command"),
        pytest.param(MCPServerConfig(url="https://mcp.example.com/sse"), "sse", id="sse-url"),
        pytest.param(MCPServerConfig(url="https://mcp.example.com/sse/"), "sse", id="sse-url-slash"),
        pytest.param(MCPServerConfig(url="https://mcp.example.com/mcp"), "streamableHttp", id="http-url"),
        pytest.param(MCPServerConfig(type="sse", url="https://mcp.example.com/mcp"), "sse", id="explicit"),
    ],
)
async def test_connect_mcp_servers_picks_the_transport(
    fake_mcp_runtime: dict[str, object | None], monkeypatch: pytest.MonkeyPatch, config, expected,
) -> None:
    used = _record_transports(monkeypatch)
    fake_mcp_runtime["session"] = _make_fake_session(["demo"])
    stacks = await connect_mcp_servers({"test": config}, ToolRegistry())
    for stack in stacks.values():
        await stack.aclose()
    assert used == [expected]
    assert set(stacks) == {"test"}


@pytest.mark.asyncio
async def test_connect_mcp_servers_skips_server_without_command_or_url(
    fake_mcp_runtime: dict[str, object | None], monkeypatch: pytest.MonkeyPatch,
) -> None:
    used = _record_transports(monkeypatch)
    warnings = _capture_warnings(monkeypatch)
    fake_mcp_runtime["session"] = _make_fake_session(["demo"])
    stacks = await connect_mcp_servers({"empty": MCPServerConfig()}, ToolRegistry())
    assert stacks == {}
    assert used == []
    assert "MCP server 'empty': no command or url configured, skipping" in warnings


@pytest.mark.asyncio
async def test_connect_mcp_servers_warns_that_use_proxy_is_ignored(
    fake_mcp_runtime: dict[str, object | None], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _record_transports(monkeypatch)
    warnings = _capture_warnings(monkeypatch)
    fake_mcp_runtime["session"] = _make_fake_session(["demo"])
    stacks = await connect_mcp_servers({"p": MCPServerConfig(command="fake", use_proxy=True)}, ToolRegistry())
    for stack in stacks.values():
        await stack.aclose()
    assert set(stacks) == {"p"}
    assert any("useProxy is not supported by NanoClaw" in w for w in warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["sse", "streamableHttp"])
async def test_connect_mcp_servers_skips_unreachable_http_server(
    fake_mcp_runtime: dict[str, object | None], monkeypatch: pytest.MonkeyPatch, transport: str,
) -> None:
    used = _record_transports(monkeypatch)
    warnings = _capture_warnings(monkeypatch)
    probed: list[str] = []

    async def unreachable(url):
        probed.append(url)
        return False

    monkeypatch.setattr(mcp_mod, "_probe_http_url", unreachable)
    fake_mcp_runtime["session"] = _make_fake_session(["demo"])
    stacks = await connect_mcp_servers(
        {"remote": MCPServerConfig(type=transport, url="https://mcp.example.com/x?token=s")}, ToolRegistry()
    )
    assert stacks == {}
    assert used == []
    assert probed == ["https://mcp.example.com/x?token=s"]
    assert "MCP server 'remote': https://mcp.example.com/... unreachable, skipping" in warnings


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["sse", "streamableHttp"])
async def test_connect_mcp_servers_sends_configured_headers(
    fake_mcp_runtime: dict[str, object | None], monkeypatch: pytest.MonkeyPatch, transport: str,
) -> None:
    _record_transports(monkeypatch)
    client_kwargs: list[dict[str, object]] = []

    class FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            client_kwargs.append(kwargs)

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

    @asynccontextmanager
    async def sse_client(_url, httpx_client_factory=None):
        httpx_client_factory(headers={"X-Factory": "1"})
        yield object(), object()

    monkeypatch.setattr(mcp_mod.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(sys.modules["mcp.client.sse"], "sse_client", sse_client)
    fake_mcp_runtime["session"] = _make_fake_session(["demo"])
    stacks = await connect_mcp_servers(
        {"remote": MCPServerConfig(type=transport, url="https://mcp.example.com/x", headers={"Authorization": "Bearer t"})},
        ToolRegistry(),
    )
    for stack in stacks.values():
        await stack.aclose()
    assert len(client_kwargs) == 1
    if transport == "sse":
        assert client_kwargs[0]["headers"] == {
            "Accept": "application/json, text/event-stream",
            "Authorization": "Bearer t",
            "X-Factory": "1",
        }
    else:
        assert client_kwargs[0]["headers"] == {"Authorization": "Bearer t"}


@pytest.mark.asyncio
async def test_connect_mcp_servers_filters_the_read_stream(
    fake_mcp_runtime: dict[str, object | None], monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def __aiter__(self):
            return self

    raw = Stream()
    seen: list[object] = []

    @asynccontextmanager
    async def stdio_client(_params):
        yield raw, object()

    class ClientSession:
        def __init__(self, read, _write):
            seen.append(read)

        async def __aenter__(self):
            return fake_mcp_runtime["session"]

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(sys.modules["mcp.client.stdio"], "stdio_client", stdio_client)
    monkeypatch.setattr(sys.modules["mcp"], "ClientSession", ClientSession)
    fake_mcp_runtime["session"] = _make_fake_session(["demo"])
    stacks = await connect_mcp_servers({"test": MCPServerConfig(command="fake")}, ToolRegistry())
    for stack in stacks.values():
        await stack.aclose()
    assert len(seen) == 1
    assert isinstance(seen[0], mcp_mod._MalformedProgressNotificationFilter)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled_tools",
    [
        pytest.param(["demo", "mcp_test_other"], id="raw-and-wrapped-found"),
        pytest.param(["*", "missing"], id="wildcard"),
    ],
)
async def test_connect_mcp_servers_no_enabled_tools_warning_when_all_match(
    fake_mcp_runtime: dict[str, object | None], monkeypatch: pytest.MonkeyPatch, enabled_tools,
) -> None:
    fake_mcp_runtime["session"] = _make_fake_session(["demo", "other"])
    warnings = _capture_warnings(monkeypatch)
    registry = ToolRegistry()
    stacks = await connect_mcp_servers(
        {"test": MCPServerConfig(command="fake", enabled_tools=enabled_tools)}, registry
    )
    for stack in stacks.values():
        await stack.aclose()
    assert registry.tool_names == ["mcp_test_demo", "mcp_test_other"]
    assert not [w for w in warnings if "enabledTools" in w]


def test_mcp_tool_name_limit_boundary_and_hash_of_full_name() -> None:
    exact = "mcp_" + "a" * 60
    assert _sanitize_mcp_tool_name(exact) == exact
    first = _sanitize_mcp_tool_name("mcp_" + "b" * 60 + "_first")
    second = _sanitize_mcp_tool_name("mcp_" + "b" * 60 + "_second")
    assert len(first) == len(second) == 64
    assert first[:55] == second[:55] == ("mcp_" + "b" * 60)[:55]
    assert first != second


def test_wrapper_description_falls_back_to_tool_name() -> None:
    tool_def = SimpleNamespace(name="demo", description=None, inputSchema=None)
    wrapper = MCPToolWrapper(SimpleNamespace(call_tool=None), "test", tool_def)
    assert wrapper.description == "demo"
    assert wrapper.parameters == {"type": "object", "properties": {}, "required": []}


def test_wrapper_replaces_non_object_schema() -> None:
    tool_def = SimpleNamespace(name="demo", description="d", inputSchema=["not", "a", "schema"])
    wrapper = MCPToolWrapper(SimpleNamespace(call_tool=None), "test", tool_def)
    assert wrapper.parameters == {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_transient_retry_waits_one_second(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def call_tool(_name: str, arguments: dict) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError("reset")
        return SimpleNamespace(content=[_FakeTextContent("ok")])

    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay, *args, **kwargs):
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(mcp_mod.asyncio, "sleep", fake_sleep)
    result = await _make_wrapper(SimpleNamespace(call_tool=call_tool), timeout=5).execute()
    assert result == "ok"
    assert delays == [1]


def test_connection_failure_group_is_transient_only_if_every_member_is() -> None:
    assert mcp_mod._is_transient_connection_failure(
        BaseExceptionGroup("g", [ConnectionResetError(), httpx.ConnectError("x")])
    )
    assert not mcp_mod._is_transient_connection_failure(
        BaseExceptionGroup("g", [ConnectionResetError(), ValueError("x")])
    )


def test_session_terminated_reads_the_error_message() -> None:
    exc = RuntimeError("request failed")
    exc.error = SimpleNamespace(code=32600, message="Session terminated")
    assert mcp_mod._is_session_terminated(exc)
    assert mcp_mod._is_session_terminated(RuntimeError("Connection closed"))
    assert not mcp_mod._is_session_terminated(RuntimeError("request failed"))


@pytest.mark.asyncio
async def test_execute_empty_content_reports_no_output() -> None:
    async def call_tool(_name: str, arguments: dict) -> object:
        return SimpleNamespace(content=[])

    assert await _make_wrapper(SimpleNamespace(call_tool=call_tool)).execute() == "(no output)"


def test_is_image_block_accepts_only_image_blobs() -> None:
    class Embedded:
        def __init__(self, resource):
            self.resource = resource

    class Blob:
        def __init__(self, mime):
            self.mimeType = mime

    class Text:
        mimeType = "image/png"

    types = SimpleNamespace(ImageContent=_FakeImageContent, EmbeddedResource=Embedded, BlobResourceContents=Blob)
    assert mcp_mod._is_image_block(_FakeImageContent("x"), types)
    assert mcp_mod._is_image_block(Embedded(Blob("image/png")), types)
    assert not mcp_mod._is_image_block(Embedded(Blob("application/pdf")), types)
    assert not mcp_mod._is_image_block(Embedded(Blob(None)), types)
    assert not mcp_mod._is_image_block(Embedded(Text()), types)


def _params(schema: dict) -> dict:
    return MCPToolWrapper(
        SimpleNamespace(call_tool=None), "test", SimpleNamespace(name="demo", description="d", inputSchema=schema)
    ).parameters


def test_wrapper_keeps_existing_defs_next_to_hoisted_refs() -> None:
    params = _params({
        "type": "object",
        "$defs": {"kept": {"type": "integer"}},
        "properties": {"a": {"type": "string"}, "b": {"$ref": "#/properties/a"}, "c": {"$ref": "#/$defs/kept"}},
    })
    name = params["properties"]["b"]["$ref"].removeprefix("#/$defs/")
    assert params["$defs"] == {"kept": {"type": "integer"}, name: {"type": "string"}}


def test_wrapper_hoisted_ref_name_avoids_existing_defs() -> None:
    import hashlib

    ref = "#/properties/a"
    taken = f"ref_{hashlib.sha256(ref.encode()).hexdigest()[:12]}"
    params = _params({
        "type": "object",
        "$defs": {taken: {"type": "boolean"}},
        "properties": {"a": {"type": "string"}, "b": {"$ref": ref}},
    })
    assert params["properties"]["b"]["$ref"] == f"#/$defs/{taken}_"
    assert params["$defs"][taken] == {"type": "boolean"}
    assert params["$defs"][f"{taken}_"] == {"type": "string"}


def test_wrapper_normalizes_nullable_oneof_items_and_defs() -> None:
    params = _params({
        "type": "object",
        "$defs": {"maybe": {"type": ["integer", "null"]}},
        "properties": {
            "choice": {"oneOf": [{"type": "string"}, {"type": "null"}]},
            "list": {"type": "array", "items": {"type": ["string", "null"]}},
            "single": {"anyOf": [{"type": "string"}]},
            "nested": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    })
    props = params["properties"]
    assert props["choice"] == {"type": "string", "nullable": True}
    assert props["list"]["items"] == {"type": "string", "nullable": True}
    assert props["single"] == {"anyOf": [{"type": "string"}]}
    assert props["nested"] == {"type": "object", "properties": {"x": {"type": "string"}}, "required": []}
    assert params["$defs"]["maybe"] == {"type": "integer", "nullable": True}
    assert params["required"] == []


@pytest.mark.parametrize(
    ("command", "env", "expected_command"),
    [
        pytest.param("npx", None, r"C:\os\cmd.exe", id="launcher-not-on-path"),
        pytest.param("tool.cmd", None, r"C:\os\cmd.exe", id="cmd-script-not-on-path"),
        pytest.param("run.BAT", None, r"C:\os\cmd.exe", id="bat-script"),
        pytest.param("npx", {"COMSPEC": r"D:\server\cmd.exe"}, r"D:\server\cmd.exe", id="server-comspec"),
    ],
)
def test_normalize_windows_stdio_command_wraps_scripts(
    monkeypatch: pytest.MonkeyPatch, command, env, expected_command,
) -> None:
    monkeypatch.setattr(mcp_mod.os, "name", "nt", raising=False)
    monkeypatch.setattr(mcp_mod.shutil, "which", lambda command, path=None: None)
    monkeypatch.setenv("COMSPEC", r"C:\os\cmd.exe")

    wrapped, args, out_env = _normalize_windows_stdio_command(command, ["serve"], env)

    assert wrapped == expected_command
    assert args == ["/d", "/c", command, "serve"]
    assert out_env == env


@pytest.mark.parametrize("command", ["python.exe", "tool.COM", "cmd", "powershell", "pwsh.exe"])
def test_normalize_windows_stdio_command_runs_executables_and_shells_directly(
    monkeypatch: pytest.MonkeyPatch, command: str,
) -> None:
    """Even when PATH would resolve the name to a .cmd shim."""
    monkeypatch.setattr(mcp_mod.os, "name", "nt", raising=False)
    monkeypatch.setattr(mcp_mod.shutil, "which", lambda command, path=None: r"C:\shims\shim.cmd")

    assert _normalize_windows_stdio_command(command, ["x"], None) == (command, ["x"], None)


@pytest.mark.asyncio
async def test_failed_server_is_closed_before_connect_returns(
    fake_mcp_runtime: dict[str, object | None], monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    @asynccontextmanager
    async def stdio_client(_params):
        try:
            yield object(), object()
        finally:
            await asyncio.sleep(0.01)
            closed.append(True)

    async def initialize():
        raise RuntimeError("initialize failed")

    monkeypatch.setattr(sys.modules["mcp.client.stdio"], "stdio_client", stdio_client)
    fake_mcp_runtime["session"] = SimpleNamespace(initialize=initialize)
    stacks = await connect_mcp_servers({"bad": MCPServerConfig(command="fake")}, ToolRegistry())
    assert stacks == {}
    assert closed == [True]


@pytest.mark.asyncio
async def test_close_connection_ignores_exception_groups() -> None:
    class Connection:
        async def aclose(self):
            raise BaseExceptionGroup("cleanup", [RuntimeError("x")])

    await mcp_mod._close_mcp_connection("s", Connection())


