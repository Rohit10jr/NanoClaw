"""The ``my`` tool (layer 25.3).

Ported from nanobot tests/agent/tools/test_self_tool.py (mock loop),
tests/agent/tools/test_runtime_control.py (real loop) and
tests/agent/test_self_model_preset.py (the tool's model-preset cases), plus
NanoClaw checks per action and for the registration and config.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.tools import ToolRegistry, build_default_tools
from nanoclaw.agent.tools.context import RequestContext, request_context
from nanoclaw.agent.tools.runtime_control import (
    RUNTIME_COMMAND_KEYS,
    RUNTIME_SNAPSHOT_KEYS,
    AgentRuntimeControl,
    RuntimeControl,
)
from nanoclaw.agent.tools.self import MyTool, MyToolConfig
from nanoclaw.agent.tools.web import WebSearchTool
from nanoclaw.config.schema import Config, ExecToolConfig, ModelPresetConfig, ToolsConfig
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.security.workspace_access import (
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)
from nanoclaw.session.model_selection import model_preset_from_metadata

# ---------------------------------------------------------------------------
# Helpers (nanobot test_self_tool.py:24)
# ---------------------------------------------------------------------------


def _make_mock_loop(**overrides):
    """A lightweight mock AgentLoop with the attributes MyTool reads."""
    loop = MagicMock()
    loop.model = "anthropic/claude-sonnet-4-6"
    loop.provider = FakeProvider()
    loop.max_iterations = 40
    loop.context_window_tokens = 65_536
    loop.workspace = Path("/tmp/workspace")
    loop.restrict_to_workspace = False
    loop.exec_config = ExecToolConfig()
    loop.channels_config = MagicMock()
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
    loop.subagents = MagicMock()
    for k, v in overrides.items():
        setattr(loop, k, v)
    return loop


def _make_tool(loop=None):
    if loop is None:
        loop = _make_mock_loop()
    return MyTool(runtime_control=AgentRuntimeControl(loop))


def _make_loop(
    tmp_path: Path,
    *,
    allow_set: bool = False,
    restricted: bool = False,
    sandbox: str = "",
    presets: dict[str, ModelPresetConfig] | None = None,
    active_preset: str | None = None,
    provider: FakeProvider | None = None,
) -> AgentLoop:
    """A real loop (nanobot test_runtime_control.py:37 / test_self_model_preset.py:26)."""
    return AgentLoop(
        provider or FakeProvider(),
        tmp_path,
        model="test-model",
        context_window_tokens=1000,
        restrict_to_workspace=restricted,
        exec_config=ExecToolConfig(sandbox=sandbox),
        model_presets=presets,
        model_preset=active_preset,
        my_config=MyToolConfig(allow_set=allow_set),
    )


def _my_tool(loop: AgentLoop) -> MyTool:
    tool = loop.tools.get("my")
    assert isinstance(tool, MyTool)
    return tool


# ---------------------------------------------------------------------------
# check — no key (summary)
# ---------------------------------------------------------------------------


class TestInspectSummary:
    async def test_inspect_returns_current_state_and_description_keys(self):
        result = await _make_tool().execute(action="check")
        assert "max_iterations: 40" in result
        assert "context_window_tokens: 65536" in result
        assert "model" in result
        assert "workspace" in result
        assert "provider_retry_mode" in result
        assert "max_tool_result_chars" in result

    async def test_inspect_includes_scratchpad(self):
        tool = _make_tool()
        tool._runtime_control.set_scratchpad("task", "review", max_keys=64)
        assert "task" in await tool.execute(action="check")

    async def test_summary_lines_in_nanobot_order(self):
        result = await _make_tool().execute(action="check")
        keys = [line.split(":", 1)[0] for line in result.splitlines()]
        assert keys == [
            "max_iterations", "context_window_tokens", "model", "model_preset", "workspace",
            "workspace_sandbox", "provider_retry_mode", "max_tool_result_chars", "web_config", "exec_config",
        ]


class TestInspectSingleKey:
    async def test_inspect_simple_value(self):
        assert "40" in await _make_tool().execute(action="check", key="max_iterations")

    async def test_inspect_blocked_returns_error(self):
        assert "not accessible" in await _make_tool().execute(action="check", key="bus")

    async def test_inspect_dunder_blocked(self):
        tool = _make_tool()
        for attr in ("__class__", "__dict__", "__bases__", "__subclasses__", "__mro__"):
            assert "not accessible" in await tool.execute(action="check", key=attr)

    async def test_inspect_nonexistent_returns_not_found(self):
        assert "not found" in await _make_tool().execute(action="check", key="nonexistent_attr_xyz")


class TestInspectPathNavigation:
    async def test_inspect_config_subfield(self):
        assert "True" in await _make_tool().execute(action="check", key="web_config.enable")

    async def test_inspect_blocked_in_path(self):
        assert "not accessible" in await _make_tool().execute(action="check", key="bus.foo")

    async def test_inspect_tools_returns_blocked(self):
        assert "not accessible" in await _make_tool().execute(action="check", key="tools")

    async def test_inspect_nested_web_search_has_no_secret_fields(self):
        result = await _make_tool().execute(action="check", key="web_config.search")
        assert "duckduckgo" in result
        assert "api_key" not in result.lower()

    async def test_web_config_disabled_without_web_tools(self):
        tool = _make_tool(_make_mock_loop(tools=ToolRegistry()))
        assert await tool.execute(action="check", key="web_config.enable") == "web_config.enable: False"


# ---------------------------------------------------------------------------
# set — restricted (with validation)
# ---------------------------------------------------------------------------


class TestModifyRestricted:
    async def test_modify_restricted_valid(self):
        tool = _make_tool()
        assert "Set max_iterations = 80" in await tool.execute(action="set", key="max_iterations", value=80)
        assert tool._runtime_control.snapshot().max_iterations == 80

    async def test_modify_restricted_out_of_range(self):
        tool = _make_tool()
        assert "Error" in await tool.execute(action="set", key="max_iterations", value=0)
        assert tool._runtime_control.snapshot().max_iterations == 40

    async def test_modify_restricted_max_exceeded(self):
        assert "Error" in await _make_tool().execute(action="set", key="max_iterations", value=999)

    async def test_modify_restricted_wrong_type(self):
        assert "Error" in await _make_tool().execute(action="set", key="max_iterations", value="not_an_int")

    async def test_modify_restricted_bool_rejected(self):
        assert "Error" in await _make_tool().execute(action="set", key="max_iterations", value=True)

    async def test_modify_string_int_coerced(self):
        tool = _make_tool()
        assert "Set max_iterations" in await tool.execute(action="set", key="max_iterations", value="80")
        assert tool._runtime_control.snapshot().max_iterations == 80

    async def test_modify_context_window_valid(self):
        loop = _make_mock_loop()
        tool = _make_tool(loop)
        result = await tool.execute(action="set", key="context_window_tokens", value=131072)
        assert "Set context_window_tokens" in result
        assert loop.context_window_tokens == 131072
        loop.set_runtime_context_window.assert_called_once_with(131072)

    async def test_modify_none_value_for_restricted_int(self):
        assert "Error" in await _make_tool().execute(action="set", key="max_iterations", value=None)


class TestModifyBlocked:
    @pytest.mark.parametrize("key", ["bus", "provider", "_running", "__class__"])
    async def test_modify_protected(self, key):
        assert "protected" in await _make_tool().execute(action="set", key=key, value="hacked")

    @pytest.mark.parametrize("leaf", ["__class__", "__globals__"])
    async def test_modify_dotpath_leaf_denied(self, leaf):
        result = await _make_tool().execute(action="set", key=f"provider_retry_mode.{leaf}", value="evil")
        assert "not accessible" in result


class TestRuntimeSettingsAndScratchpad:
    async def test_set_updates_runtime_setting(self):
        tool = _make_tool()
        assert "Set provider_retry_mode" in await tool.execute(
            action="set", key="provider_retry_mode", value="persistent"
        )
        assert tool._runtime_control.snapshot().provider_retry_mode == "persistent"

    async def test_modify_new_key_stores_in_scratchpad(self):
        tool = _make_tool()
        assert "my_custom_var" in await tool.execute(action="set", key="my_custom_var", value="hello")
        assert tool._runtime_control.snapshot().scratchpad["my_custom_var"] == "hello"

    async def test_modify_rejects_callable(self):
        assert "callable" in await _make_tool().execute(action="set", key="evil", value=lambda: None)

    async def test_modify_rejects_complex_objects(self):
        assert "Error" in await _make_tool().execute(action="set", key="obj", value=Path("/tmp"))

    async def test_modify_allows_list(self):
        tool = _make_tool()
        assert await tool.execute(action="set", key="items", value=[1, 2, 3]) == "Set scratchpad.items = [1, 2, 3]"
        assert tool._runtime_control.snapshot().scratchpad["items"] == [1, 2, 3]

    async def test_modify_allows_dict(self):
        tool = _make_tool()
        assert await tool.execute(action="set", key="data", value={"a": 1}) == "Set scratchpad.data = {'a': 1}"

    async def test_modify_whitespace_key_rejected(self):
        assert "cannot be empty or whitespace" in await _make_tool().execute(action="set", key="   ", value="t")

    async def test_modify_nested_dict_with_object_rejected(self):
        assert "Error" in await _make_tool().execute(action="set", key="evil", value={"nested": object()})

    async def test_modify_deep_nesting_rejected(self):
        deep: dict = {"level": 0}
        current = deep
        for i in range(1, 15):
            current["child"] = {"level": i}
            current = current["child"]
        assert "nesting too deep" in await _make_tool().execute(action="set", key="deep", value=deep)

    async def test_modify_dict_with_non_str_key_rejected(self):
        assert "key must be str" in await _make_tool().execute(action="set", key="evil", value={42: "value"})

    async def test_modify_existing_attr_type_mismatch_rejected(self):
        tool = _make_tool()
        result = await tool.execute(action="set", key="provider_retry_mode", value=42)
        assert "Error" in result and "str" in result
        assert tool._runtime_control.snapshot().provider_retry_mode == "standard"

    async def test_modify_existing_int_attr_wrong_type_rejected(self):
        tool = _make_tool()
        assert "Error" in await tool.execute(action="set", key="max_tool_result_chars", value="big")
        assert tool._runtime_control.snapshot().max_tool_result_chars == 16000

    async def test_max_tool_result_chars_set(self):
        tool = _make_tool()
        result = await tool.execute(action="set", key="max_tool_result_chars", value=8000)
        assert result == "Set max_tool_result_chars = 8000 (was 16000)"


class TestProtectedRuntimeState:
    @pytest.mark.parametrize(
        "key",
        [
            "tools", "subagents", "runner", "sessions", "consolidator", "dream", "auto_compact",
            "context", "commands", "_pending_queues", "_session_locks", "_active_tasks",
            "_background_tasks", "__wrapped__", "__closure__", "func_globals", "func_code",
        ],
    )
    async def test_modify_protected(self, key):
        assert "protected" in await _make_tool().execute(action="set", key=key, value={})

    async def test_modify_workspace_preserves_display_compatibility(self):
        tool = _make_tool()
        assert "Set workspace" in await tool.execute(action="set", key="workspace", value="/new/path")
        assert tool._runtime_control.snapshot().workspace == "/new/path"

    async def test_modify_workspace_requires_str(self):
        result = await _make_tool().execute(action="set", key="workspace", value=3)
        assert result == "Error: 'workspace' expects str, got int"

    @pytest.mark.parametrize("key", ["tool_names", "model_presets", "request", "web_config", "exec_config"])
    async def test_read_only_top_level(self, key):
        assert "read-only" in await _make_tool().execute(action="set", key=key, value=[])


class TestValidateJsonSafe:
    def test_single_list_passes(self):
        assert MyTool._validate_json_safe(list(range(500))) is None

    def test_deeply_nested_within_limit(self):
        assert MyTool._validate_json_safe({"level1": {"level2": {"level3": list(range(100))}}}) is None


async def test_unknown_action():
    tool = MyTool(runtime_control=AgentRuntimeControl(_make_mock_loop()), modify_allowed=True)
    assert "Unknown action" in await tool.execute(action="explode")


async def test_legacy_action_names_still_work():
    """nanobot accepts its old ``inspect`` / ``modify`` names."""
    tool = _make_tool()
    assert await tool.execute(action="inspect", key="max_iterations") == "max_iterations: 40"
    assert await tool.execute(action="modify", key="note", value=1) == "Set scratchpad.note = 1"


class TestScratchpadLimits:
    async def test_scratchpad_rejects_at_max_keys(self):
        tool = _make_tool()
        for i in range(64):
            tool._runtime_control.set_scratchpad(f"key_{i}", i, max_keys=64)
        result = await tool.execute(action="set", key="overflow", value="data")
        assert result == "Error: scratchpad is full (max 64 keys). Remove unused keys first."
        assert "overflow" not in tool._runtime_control.snapshot().scratchpad

    async def test_scratchpad_allows_update_existing_key_at_max(self):
        tool = _make_tool()
        for i in range(64):
            tool._runtime_control.set_scratchpad(f"key_{i}", i, max_keys=64)
        assert "Error" not in await tool.execute(action="set", key="key_0", value="updated")
        assert tool._runtime_control.snapshot().scratchpad["key_0"] == "updated"


class TestReadOnlyMode:
    def _make_readonly_tool(self):
        return MyTool(runtime_control=AgentRuntimeControl(_make_mock_loop()), modify_allowed=False)

    async def test_inspect_allowed_in_readonly(self):
        assert "40" in await self._make_readonly_tool().execute(action="check", key="max_iterations")

    async def test_modify_blocked_in_readonly(self):
        result = await self._make_readonly_tool().execute(action="set", key="max_iterations", value=80)
        assert result == "Error: set is disabled (tools.my.allow_set is false)"
        assert result.is_error

    def test_description_shows_readonly(self):
        assert "READ-ONLY MODE" in self._make_readonly_tool().description
        assert "READ-ONLY MODE" not in _make_tool().description


class TestScratchpadInspection:
    async def test_inspect_scratchpad_value_after_modify(self):
        tool = _make_tool()
        await tool.execute(action="set", key="user_prefers_concise", value=True)
        assert "True" in await tool.execute(action="check", key="user_prefers_concise")

    async def test_inspect_scratchpad_dict(self):
        tool = _make_tool()
        await tool.execute(action="set", key="task_meta", value={"step": 2, "total": 5})
        result = await tool.execute(action="check", key="task_meta")
        assert "step" in result and "2" in result

    async def test_inspect_scratchpad_whole_and_empty(self):
        tool = _make_tool()
        assert await tool.execute(action="check", key="scratchpad") == "scratchpad is empty"
        await tool.execute(action="set", key="a", value=1)
        assert await tool.execute(action="check", key="scratchpad") == "scratchpad: {'a': 1}"


class TestSensitiveSubFieldBlocking:
    @pytest.mark.parametrize(
        "key", ["web_config.search.api_key", "some_config.password", "vault.secret", "auth_data.token"]
    )
    async def test_inspect_sensitive_blocked(self, key):
        assert "not accessible" in await _make_tool().execute(action="check", key=key)

    async def test_modify_api_key_blocked(self):
        result = await _make_tool().execute(action="set", key="web_config.search.api_key", value="evil")
        assert "read-only" in result or "not accessible" in result

    async def test_modify_password_blocked(self):
        result = await _make_tool().execute(action="set", key="some_config.password", value="evil")
        assert "not accessible" in result

    async def test_modify_sensitive_top_level_blocked(self):
        assert "protected" in await _make_tool().execute(action="set", key="api_key", value="evil")


class TestSecurityAttributeProtection:
    async def test_modify_restrict_to_workspace_blocked(self):
        assert "protected" in await _make_tool().execute(action="set", key="restrict_to_workspace", value=True)

    @pytest.mark.parametrize("key", ["exec_config", "web_config", "exec_config.enable", "web_config.enable"])
    async def test_modify_config_read_only(self, key):
        assert "read-only" in await _make_tool().execute(action="set", key=key, value=False)

    async def test_modify_channels_config_blocked(self):
        assert "protected" in await _make_tool().execute(action="set", key="channels_config", value={})

    async def test_inspect_restrict_to_workspace_blocked(self):
        assert "not accessible" in await _make_tool().execute(action="check", key="restrict_to_workspace")

    @pytest.mark.parametrize("key", ["exec_config", "web_config"])
    async def test_inspect_config_allowed(self, key):
        assert "Error" not in await _make_tool().execute(action="check", key=key)

    async def test_modify_model_presets_dotpath_blocked(self):
        presets = {"fast": ModelPresetConfig(model="fast-model")}
        tool = _make_tool(_make_mock_loop(model_presets=presets))
        result = await tool.execute(action="set", key="model_presets.other", value={"model": "other-model"})
        assert "read-only" in result
        assert presets == {"fast": ModelPresetConfig(model="fast-model")}

    async def test_inspect_read_only_model_preset_dotpath(self):
        presets = MappingProxyType({"fast": ModelPresetConfig(model="fast-model")})
        tool = _make_tool(_make_mock_loop(model_presets=presets))
        assert await tool.execute(action="check", key="model_presets.fast.model") == (
            "model_presets.fast.model: 'fast-model'"
        )


class TestRequestContext:
    async def test_check_exposes_current_routing_metadata_on_demand(self):
        tool = _make_tool()
        ctx = RequestContext(channel="feishu", chat_id="oc_abc123", sender_id="ou_user456")
        with request_context(ctx):
            assert await tool.execute(action="check", key="request.channel") == "request.channel: 'feishu'"
            assert await tool.execute(action="check", key="request.chat_id") == "request.chat_id: 'oc_abc123'"
            assert await tool.execute(action="check", key="request.sender_id") == (
                "request.sender_id: 'ou_user456'"
            )
            assert await tool.execute(action="check", key="request.message_id") == (
                "Error: 'request.message_id' not found"
            )
            summary = await tool.execute(action="check")
        assert "oc_abc123" not in summary
        assert "ou_user456" not in summary

    async def test_request_without_context_is_an_error(self):
        result = await _make_tool().execute(action="check", key="request")
        assert result == "Error: current request context is unavailable"
        assert result.is_error

    async def test_request_routing_metadata_is_read_only(self):
        assert "read-only" in await _make_tool().execute(action="set", key="request.chat_id", value="x")

    def test_audit_reads_bound_session(self):
        tool = _make_tool()
        ctx = RequestContext(channel="feishu", chat_id="oc_abc123", session_key="feishu:oc_abc123")
        with patch("nanoclaw.agent.tools.self.logger.info") as info, request_context(ctx):
            tool._audit("modify", "temperature = 0.2")
        info.assert_called_once_with(
            "self.{} | {} | session:{}", "modify", "temperature = 0.2", "feishu:oc_abc123"
        )

    async def test_runtime_values_come_from_the_turn(self):
        """Inside a turn, model/model_preset/context_window_tokens are the session's runtime."""
        loop = _make_mock_loop()
        tool = _make_tool(loop)
        runtime = MagicMock(model=None, provider=FakeProvider(), model_preset="fast", context_window_tokens=4096)
        with request_context(RequestContext(channel="cli", session_key="cli:x", runtime=runtime)):
            assert await tool.execute(action="check", key="model") == "model: 'fake-model'"
            assert await tool.execute(action="check", key="model_preset") == "model_preset: 'fast'"
            summary = await tool.execute(action="check")
        assert "context_window_tokens: 4096" in summary
        assert "model_preset: 'fast'" in summary


# ---------------------------------------------------------------------------
# Real loop (nanobot test_runtime_control.py)
# ---------------------------------------------------------------------------


def test_agent_loop_assembles_my_tool_with_runtime_control(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    tool = _my_tool(loop)
    assert isinstance(tool._runtime_control, RuntimeControl)
    assert isinstance(tool._runtime_control, AgentRuntimeControl)
    assert tool._runtime_control is not loop


def test_my_tool_config(tmp_path: Path) -> None:
    assert MyTool.config_key == "my"
    assert MyTool.config_cls() is MyToolConfig
    assert ToolsConfig().my.enable is True
    assert ToolsConfig().my.allow_set is False
    config = Config.model_validate({"tools": {"my": {"enable": False, "allowSet": True}}})
    assert config.tools.my == MyToolConfig(enable=False, allow_set=True)
    loop = AgentLoop(FakeProvider(), tmp_path, my_config=config.tools.my)
    assert not loop.tools.has("my")


def test_my_tool_is_main_agent_only(tmp_path: Path) -> None:
    """Subagents build their own tools without ``my`` (nanobot test_subagent.py:45)."""
    loop = _make_loop(tmp_path)
    assert loop.tools.has("my")
    assert not build_default_tools(tmp_path).has("my")


def test_a_registry_handed_in_is_used_as_given(tmp_path: Path) -> None:
    """NanoClaw difference: nanobot's loader adds ``my`` to a given registry too."""
    loop = AgentLoop(FakeProvider(), tmp_path, tools=ToolRegistry())
    assert loop.tools.tool_names == []


def test_runtime_snapshot_has_exact_allowlist(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    loop.unlisted_secret = "loop-secret"
    values = _my_tool(loop)._runtime_control.snapshot().as_mapping()
    expected = frozenset({
        "model", "model_preset", "model_presets", "max_iterations", "context_window_tokens", "workspace",
        "workspace_sandbox", "provider_retry_mode", "max_tool_result_chars", "tool_names", "web_config",
        "exec_config",
    })
    assert RUNTIME_SNAPSHOT_KEYS == expected
    assert frozenset(values) == expected
    assert RUNTIME_COMMAND_KEYS == frozenset({
        "model", "model_preset", "max_iterations", "context_window_tokens", "provider_retry_mode",
        "max_tool_result_chars", "workspace",
    })
    assert "provider" not in values and "sessions" not in values
    assert "restrict_to_workspace" not in values
    assert "loop-secret" not in repr(values)


def test_runtime_snapshot_is_detached_from_mutable_config(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    control = _my_tool(loop)._runtime_control
    snapshot = control.snapshot()
    search = snapshot.web_config["search"]
    assert isinstance(search, dict)
    search["provider"] = "mutated"
    snapshot.exec_config["allow_patterns"] = ["mutated"]
    snapshot.tool_names.append("mutated")
    refreshed = control.snapshot()
    assert refreshed.web_config["search"]["provider"] == "duckduckgo"  # type: ignore[index]
    assert refreshed.exec_config["allow_patterns"] == loop.exec_config.allow_patterns
    assert "mutated" not in refreshed.tool_names


async def test_unlisted_loop_attributes_cannot_be_read_or_modified(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, allow_set=True)
    loop.unlisted_control_plane = "internal-secret"
    original = loop.workspace_scope.project_path
    tool = _my_tool(loop)
    inspected = await tool.execute(action="check", key="unlisted_control_plane")
    modified = await tool.execute(action="set", key="unlisted_control_plane", value="scratch-value")
    nested = await tool.execute(action="set", key="workspace_scope.project_path", value="elsewhere")
    assert "internal-secret" not in inspected and "not found" in inspected
    assert modified == "Set scratchpad.unlisted_control_plane = 'scratch-value'"
    assert loop.unlisted_control_plane == "internal-secret"
    assert "Error" in nested
    assert loop.workspace_scope.project_path == original


async def test_default_allow_set_and_public_parameter_schema_are_unchanged(tmp_path: Path) -> None:
    loop = AgentLoop(FakeProvider(), tmp_path)
    tool = _my_tool(loop)
    assert tool.parameters == {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["check", "set"], "description": "Action to perform"},
            "key": {
                "type": "string",
                "description": (
                    "Dot-path for check; top-level runtime or scratchpad key for set. "
                    "Read-only routing metadata: request.channel, request.chat_id, request.sender_id."
                ),
            },
            "value": {
                "description": "New value for set; match the setting's type. Scratchpad accepts JSON values.",
            },
        },
        "required": ["action"],
    }
    assert "READ-ONLY MODE" in tool.description
    result = await tool.execute(action="set", key="max_iterations", value=80)
    assert result == "Error: set is disabled (tools.my.allow_set is false)"
    assert loop.max_iterations != 80


async def test_allowlisted_commands_preserve_runtime_side_effects(tmp_path: Path) -> None:
    loop = AgentLoop(FakeProvider(), tmp_path, my_config=MyToolConfig(allow_set=True))
    tool = _my_tool(loop)
    assert await tool.execute(action="set", key="max_iterations", value=80) == "Set max_iterations = 80 (was 200)"
    assert await tool.execute(action="set", key="provider_retry_mode", value="persistent") == (
        "Set provider_retry_mode = 'persistent' (was 'standard')"
    )
    assert await tool.execute(action="set", key="preference", value={"concise": True}) == (
        "Set scratchpad.preference = {'concise': True}"
    )
    assert loop.max_iterations == 80
    assert loop.subagents.max_iterations == 80
    assert loop.provider_retry_mode == "persistent"
    assert tool._runtime_control.snapshot().scratchpad == {"preference": {"concise": True}}


async def test_registry_exposes_unchanged_my_tool_actions(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, allow_set=True)
    assert await loop.tools.execute("my", {"action": "check", "key": "model"}) == "model: 'test-model'"
    assert await loop.tools.execute("my", {"action": "set", "key": "max_iterations", "value": 80}) == (
        "Set max_iterations = 80 (was 200)"
    )
    assert loop.max_iterations == 80


async def test_model_without_name_reports_provider_default(tmp_path: Path) -> None:
    loop = AgentLoop(FakeProvider(), tmp_path)
    assert await loop.tools.execute("my", {"action": "check", "key": "model"}) == "model: 'fake-model'"


async def test_workspace_display_command_cannot_change_path_enforcement(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, allow_set=True)
    tool = _my_tool(loop)
    assert "Set workspace" in await tool.execute(action="set", key="workspace", value="elsewhere")
    assert tool._runtime_control.snapshot().workspace == "elsewhere"
    assert loop.workspace == tmp_path


@pytest.mark.parametrize(("mode", "sandbox", "restricted"), [
    ("restricted", "", True),
    ("full", "", False),
    ("full", "bwrap", True),
])
async def test_workspace_inspection_matches_registered_file_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, sandbox: str, restricted: bool,
) -> None:
    monkeypatch.delenv("NANOCLAW_WORKSPACE_SANDBOX_ENFORCED", raising=False)
    monkeypatch.delenv("NANOCLAW_WORKSPACE_SANDBOX_PROVIDER", raising=False)
    agent = tmp_path / "agent"
    agent.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / "marker.txt").write_text("project marker", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside marker", encoding="utf-8")
    loop = _make_loop(agent, allow_set=True, restricted=True, sandbox=sandbox)
    tool = _my_tool(loop)
    definitions = json.dumps(loop.tools.get_definitions(), sort_keys=True)
    token = bind_workspace_scope(build_workspace_scope(project, mode))
    try:
        await tool.execute(action="set", key="workspace", value=str(outside.parent))
        status = tool._runtime_control.snapshot().workspace_sandbox
        assert status["workspace_root"] == str(project.resolve())
        assert status["restrict_to_workspace"] is restricted
        assert status["level"] == ("application" if restricted else "off")
        assert status["enforced"] is False
        assert repr(str(project.resolve())) in await tool.execute(action="check", key="workspace_sandbox")
        assert repr(str(project.resolve())) in await tool.execute(action="check")
        assert await tool.execute(action="check", key="workspace_sandbox.restrict_to_workspace") == (
            f"workspace_sandbox.restrict_to_workspace: {restricted!r}"
        )
        for key in ("workspace_sandbox", "workspace_sandbox.restrict_to_workspace"):
            assert "read-only" in await tool.execute(action="set", key=key, value=False)
        assert "not accessible" in await tool.execute(action="check", key="restrict_to_workspace")
        assert "project marker" in await loop.tools.execute("read_file", {"path": "marker.txt"})
        result = await loop.tools.execute("read_file", {"path": str(outside)})
        assert ("outside" in result and "marker" not in result) if restricted else "outside marker" in result
        assert json.dumps(loop.tools.get_definitions(), sort_keys=True) == definitions
        status["workspace_root"] = "changed snapshot"
        assert tool._runtime_control.snapshot().workspace_sandbox["workspace_root"] == str(project.resolve())
    finally:
        reset_workspace_scope(token)


@pytest.mark.parametrize("restricted", [False, True])
def test_workspace_inspection_uses_instance_default_without_turn(tmp_path: Path, restricted: bool) -> None:
    status = _my_tool(_make_loop(tmp_path, restricted=restricted))._runtime_control.snapshot().workspace_sandbox
    assert status["workspace_root"] == str(tmp_path.resolve())
    assert status["restrict_to_workspace"] is restricted


async def test_workspace_inspection_is_isolated_between_concurrent_turns(tmp_path: Path) -> None:
    tool = _my_tool(_make_loop(tmp_path))
    ready = asyncio.Event()

    async def inspect(project: Path, mode: str, *, signal: bool) -> dict[str, object]:
        token = bind_workspace_scope(build_workspace_scope(project, mode))
        try:
            if signal:
                ready.set()
            await ready.wait()
            await asyncio.sleep(0)
            result = await tool.execute(action="check", key="workspace_sandbox.workspace_root")
            assert repr(str(project.resolve())) in result
            return tool._runtime_control.snapshot().workspace_sandbox
        finally:
            reset_workspace_scope(token)

    first, second = await asyncio.gather(
        inspect(tmp_path / "first", "restricted", signal=False),
        inspect(tmp_path / "second", "full", signal=True),
    )
    assert first["restrict_to_workspace"] is True
    assert second["restrict_to_workspace"] is False
    assert tool._runtime_control.snapshot().workspace_sandbox["workspace_root"] == str(tmp_path.resolve())


async def test_workspace_inspection_reaches_model_from_turn(tmp_path: Path) -> None:
    """A turn's ``my`` call sees the turn's project and its routing (test_runtime_control.py:370)."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "marker.txt").write_text("selected project", encoding="utf-8")
    fake = FakeProvider([
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest("inspect", "my", {"action": "check", "key": "workspace_sandbox"}),
            ToolCallRequest("route", "my", {"action": "check", "key": "request"}),
            ToolCallRequest("read", "read_file", {"path": "marker.txt"}),
        ], finish_reason="tool_calls"),
        "done",
    ])
    loop = AgentLoop(fake, tmp_path / "agent", project=project, restrict_to_workspace=True)
    response = await loop.process_direct("Check the project", channel="cli", chat_id="inspection")
    assert response is not None
    results = {m["tool_call_id"]: m["content"] for m in fake.calls[1].messages if m["role"] == "tool"}
    assert repr(str(project.resolve())) in results["inspect"]
    assert "'restrict_to_workspace': True" in results["inspect"]
    assert "'channel': 'cli'" in results["route"] and "'chat_id': 'inspection'" in results["route"]
    assert "selected project" in results["read"]
    assert fake.calls[0].tools == fake.calls[1].tools


# ---------------------------------------------------------------------------
# model presets (nanobot test_self_model_preset.py)
# ---------------------------------------------------------------------------


def test_self_tool_inspect_shows_model_preset(tmp_path) -> None:
    loop = _make_loop(tmp_path, presets={"fast": ModelPresetConfig(model="openai/gpt-4.1")}, active_preset="fast")
    assert "model_preset: 'fast'" in _my_tool(loop)._inspect_all()


async def test_self_tool_set_model_preset_via_modify(tmp_path) -> None:
    loop = _make_loop(tmp_path, presets={"fast": ModelPresetConfig(model="openai/gpt-4.1")})
    result = await _my_tool(loop)._modify("model_preset", "fast")
    assert "Error" not in result
    assert loop.model_preset == "fast"
    assert loop.model == "openai/gpt-4.1"
    assert result == (
        "Set model_preset = 'fast' (was None); model is now 'openai/gpt-4.1'; "
        "context_window_tokens is now 128000"
    )


async def test_self_tool_set_model_preset_switches_back_to_default(tmp_path) -> None:
    presets = {
        "default": ModelPresetConfig(model="test-model", context_window_tokens=1000),
        "fast": ModelPresetConfig(model="openai/gpt-4.1", context_window_tokens=32_768),
    }
    loop = _make_loop(tmp_path, presets=presets, active_preset="fast")
    loop.set_model_preset("fast")
    result = await _my_tool(loop)._modify("model_preset", "default")
    assert "Error" not in result
    assert "model is now 'test-model'" in result
    assert loop.model_preset == "default"
    assert loop.model == "test-model"
    assert loop.context_window_tokens == 1000


async def test_self_tool_set_model_preset_unknown_lists_available(tmp_path) -> None:
    presets = {"default": ModelPresetConfig(model="test-model"), "fast": ModelPresetConfig(model="openai/gpt-4.1")}
    loop = _make_loop(tmp_path, presets=presets)
    result = await _my_tool(loop)._modify("model_preset", "missing")
    assert result == "Error: model_preset 'missing' not found. Available: default, fast."
    assert loop.model_preset is None
    assert loop.model == "test-model"


@pytest.mark.parametrize("value", ["", "  ", 3, None])
async def test_self_tool_model_preset_needs_a_name(tmp_path, value) -> None:
    result = await _my_tool(_make_loop(tmp_path))._modify("model_preset", value)
    assert result == "Error: 'model_preset' must be a non-empty string"


async def test_self_tool_sets_model_preset_for_current_session(tmp_path) -> None:
    presets = {"default": ModelPresetConfig(model="test-model"), "fast": ModelPresetConfig(model="openai/gpt-4.1")}
    loop = _make_loop(tmp_path, presets=presets)
    with request_context(RequestContext(channel="cli", chat_id="one", session_key="cli:one")):
        result = await _my_tool(loop)._modify("model_preset", "FAST")
    assert result == (
        "Set model_preset = 'FAST' for the next turn; model will be 'openai/gpt-4.1'; "
        "context_window_tokens will be 128000"
    )
    assert model_preset_from_metadata(loop.sessions.get_or_create("cli:one").metadata) == "fast"
    assert loop.model_preset is None
    assert loop.model == "test-model"


async def test_session_preset_from_my_is_used_by_the_next_turn(tmp_path) -> None:
    """The same effect as ``/model fast`` (layer 20): the session's next turn runs on the preset."""
    fake = FakeProvider([
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest("c1", "my", {"action": "set", "key": "model_preset", "value": "fast"}),
        ], finish_reason="tool_calls"),
        "switched",
        "next",
    ])
    loop = AgentLoop(
        fake, tmp_path, model="test-model", my_config=MyToolConfig(allow_set=True),
        model_presets={"fast": ModelPresetConfig(model="fast-model")},
    )
    await loop.process_direct("use fast", session_key="cli:s")
    await loop.process_direct("again", session_key="cli:s")
    assert [call.model for call in fake.calls] == ["test-model", "test-model", "fast-model"]
    assert loop.model == "test-model"


async def test_self_tool_reports_session_preset_provider_configuration_error(tmp_path) -> None:
    loop = _make_loop(tmp_path)
    loop.set_session_model_preset = AsyncMock(side_effect=ValueError("No API key configured for provider 'openai'."))
    with request_context(RequestContext(channel="cli", chat_id="one", session_key="cli:one")):
        result = await _my_tool(loop)._modify("model_preset", "broken")
    assert result == "Error: No API key configured for provider 'openai'."


@pytest.mark.parametrize(("key", "value"), [("model", "other-model"), ("context_window_tokens", 8_192)])
async def test_self_tool_rejects_instance_runtime_changes_in_session(tmp_path, key, value) -> None:
    loop = _make_loop(tmp_path)
    session = loop.sessions.get_or_create("cli:one")
    with request_context(RequestContext(
        channel="cli", chat_id="one", session_key=session.key, runtime=await loop.runtime_for_session(session),
    )):
        result = await _my_tool(loop)._modify(key, value)
    other = await loop.runtime_for_session(loop.sessions.get_or_create("cli:two"))
    assert "instance-wide and disabled" in result
    assert "model_preset" in result
    assert other.model == "test-model"
    assert other.context_window_tokens == 1000


async def test_self_tool_set_model_clears_active_preset(tmp_path) -> None:
    loop = _make_loop(tmp_path, presets={"fast": ModelPresetConfig(model="openai/gpt-4.1")}, active_preset="fast")
    loop.set_model_preset("fast")
    result = await _my_tool(loop)._modify("model", "anthropic/claude-opus-4-5")
    assert result == "Set model = 'anthropic/claude-opus-4-5' (was 'openai/gpt-4.1')"
    assert loop.model_preset is None
    assert loop.model == "anthropic/claude-opus-4-5"


async def test_self_tool_sets_context_window_outside_a_session(tmp_path) -> None:
    loop = _make_loop(tmp_path)
    result = await _my_tool(loop)._modify("context_window_tokens", 65_536)
    assert result == "Set context_window_tokens = 65536 (was 1000)"
    assert loop.context_window_tokens == 65_536
    assert "must be >= 4096" in await _my_tool(loop)._modify("context_window_tokens", 100)


def test_make_loop_passes_tools_my(tmp_path) -> None:
    from nanoclaw.__main__ import _make_loop as cli_make_loop

    config = Config.model_validate({"tools": {"my": {"allowSet": True}}})
    tool = _my_tool(cli_make_loop(FakeProvider(), tmp_path, config=config))
    assert tool._modify_allowed is True
    config = Config.model_validate({"tools": {"my": {"enable": False}}})
    assert not cli_make_loop(FakeProvider(), tmp_path, config=config).tools.has("my")


def test_tool_hint_uses_the_generic_form():
    """nanobot has no hint format for ``my`` (utils/tool_hints.py): the fallback shows the action."""
    from nanoclaw.utils.tool_hints import format_tool_hints

    calls = [
        ToolCallRequest("1", "my", {"action": "check", "key": "model"}),
        ToolCallRequest("2", "my", {"action": "set", "key": "max_iterations", "value": 80}),
    ]
    assert format_tool_hints(calls) == 'my("check"), my("set")'


def test_my_skill_is_bundled_and_names_only_existing_actions():
    from nanoclaw.agent.skills import BUILTIN_SKILLS_DIR

    text = (BUILTIN_SKILLS_DIR / "my" / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\nname: my\n")
    assert (BUILTIN_SKILLS_DIR / "my" / "references" / "examples.md").exists()
    for call in ('my(action="check"', 'my(action="set", key="model_preset"', 'key="max_iterations", value=80'):
        assert call in text
