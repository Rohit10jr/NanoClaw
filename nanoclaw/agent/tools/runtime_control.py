"""The runtime state the ``my`` tool may read and change.

Mirrors nanobot/agent/tools/runtime_control.py: ``MyTool`` never touches the
agent loop directly. ``AgentRuntimeControl`` hands it a detached,
allowlisted ``RuntimeSnapshot`` and a short list of setters, so a key that is
not listed here (the bus, providers, sessions, security settings) can't be
read or changed through the tool.

Differences from nanobot:

- NanoClaw has no ``tools.web`` config: ``web_config`` is built from the web
  tools the loop registered (``enable`` is whether they are registered) with
  nanobot's field names and NanoClaw's fixed values (DuckDuckGo, no proxy, no
  Jina reader), the same view the WebUI settings show.
- A loop without a model (the provider's default model) reports that
  default, as nanobot's loop always has a model string.
- nanobot's ``workspace_context`` runtime-context provider lives here; in
  NanoClaw the loop already adds that block (``workspace_access_block``,
  since layer 21), so it is not repeated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

from nanoclaw.security.workspace_access import current_tool_workspace, workspace_sandbox_status

if TYPE_CHECKING:
    from nanoclaw.agent.tools.registry import ToolRegistry
    from nanoclaw.config.schema import ExecToolConfig, ModelPresetConfig
    from nanoclaw.providers.base import LLMProvider
    from nanoclaw.utils.llm_runtime import LLMRuntime


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


RUNTIME_SNAPSHOT_KEYS = frozenset({
    "model",
    "model_preset",
    "model_presets",
    "max_iterations",
    "context_window_tokens",
    "workspace",
    "workspace_sandbox",
    "provider_retry_mode",
    "max_tool_result_chars",
    "tool_names",
    "web_config",
    "exec_config",
})

RUNTIME_COMMAND_KEYS = frozenset({
    "model",
    "model_preset",
    "max_iterations",
    "context_window_tokens",
    "provider_retry_mode",
    "max_tool_result_chars",
    "workspace",
})


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Detached, allowlisted values available to self-inspection (nanobot :56)."""

    model: str
    model_preset: str | None
    model_presets: dict[str, dict[str, object]]
    max_iterations: int
    context_window_tokens: int
    workspace: Path | str
    provider_retry_mode: str
    max_tool_result_chars: int
    tool_names: list[str]
    web_config: dict[str, object]
    exec_config: dict[str, object]
    scratchpad: dict[str, JsonValue]
    workspace_sandbox: dict[str, object]

    def as_mapping(self) -> Mapping[str, object]:
        """Return the fixed public names understood by ``MyTool``."""
        values: dict[str, object] = {
            "model": self.model,
            "model_preset": self.model_preset,
            "model_presets": self.model_presets,
            "max_iterations": self.max_iterations,
            "context_window_tokens": self.context_window_tokens,
            "workspace": self.workspace,
            "workspace_sandbox": self.workspace_sandbox,
            "provider_retry_mode": self.provider_retry_mode,
            "max_tool_result_chars": self.max_tool_result_chars,
            "tool_names": self.tool_names,
            "web_config": self.web_config,
            "exec_config": self.exec_config,
        }
        assert values.keys() == RUNTIME_SNAPSHOT_KEYS
        return values


@runtime_checkable
class RuntimeControl(Protocol):
    """The complete runtime capability exposed to ``MyTool`` (nanobot :93)."""

    def snapshot(self) -> RuntimeSnapshot: ...

    def set_model(self, model: str) -> LLMRuntime: ...

    async def set_model_preset(self, name: str, *, session_key: str | None) -> LLMRuntime: ...

    def set_max_iterations(self, value: int) -> None: ...

    def set_context_window_tokens(self, value: int) -> LLMRuntime: ...

    def set_provider_retry_mode(self, value: str) -> None: ...

    def set_max_tool_result_chars(self, value: int) -> None: ...

    def set_workspace_display(self, value: str) -> None: ...

    def set_scratchpad(self, key: str, value: JsonValue, *, max_keys: int) -> None: ...


class _RuntimeControlTarget(Protocol):
    """Narrow structural dependency required by ``AgentRuntimeControl`` (nanobot :123)."""

    max_iterations: int
    provider_retry_mode: str
    max_tool_result_chars: int
    exec_config: ExecToolConfig
    restrict_to_workspace: bool
    workspace: Path
    tools: ToolRegistry

    @property
    def provider(self) -> LLMProvider: ...

    @property
    def model(self) -> str | None: ...

    @property
    def model_preset(self) -> str | None: ...

    @property
    def model_presets(self) -> Mapping[str, ModelPresetConfig]: ...

    @property
    def context_window_tokens(self) -> int: ...

    def set_runtime_model(self, model: str) -> LLMRuntime: ...

    def set_runtime_context_window(self, context_window_tokens: int) -> LLMRuntime: ...

    def set_runtime_max_iterations(self, value: int) -> None: ...

    def set_model_preset(self, name: str | None) -> LLMRuntime: ...

    async def set_session_model_preset(self, session_key: str, name: str) -> LLMRuntime: ...


class AgentRuntimeControl:
    """Allowlisted adapter from agent-loop state to ``RuntimeControl`` (nanobot :160)."""

    def __init__(self, target: _RuntimeControlTarget) -> None:
        self.__target = target
        self.__scratchpad: dict[str, JsonValue] = {}
        self.__workspace_display: str | None = None

    def _workspace_sandbox(self) -> dict[str, object]:
        """The bound turn's project and restriction, else the instance default."""
        target = self.__target
        workspace = current_tool_workspace(
            target.workspace,
            restrict_to_workspace=target.restrict_to_workspace,
            sandbox_restricts_workspace=bool(target.exec_config.sandbox),
        )
        assert workspace.project_path is not None
        return workspace_sandbox_status(
            workspace=workspace.project_path,
            restrict_to_workspace=workspace.restrict_to_workspace,
        ).as_dict()

    def snapshot(self) -> RuntimeSnapshot:
        target = self.__target
        return RuntimeSnapshot(
            model=effective_model(target.model, target.provider),
            model_preset=target.model_preset,
            model_presets=_snapshot_model_presets(target.model_presets),
            max_iterations=target.max_iterations,
            context_window_tokens=target.context_window_tokens,
            workspace=(
                self.__workspace_display
                if self.__workspace_display is not None
                else target.workspace
            ),
            provider_retry_mode=target.provider_retry_mode,
            max_tool_result_chars=target.max_tool_result_chars,
            tool_names=list(target.tools.tool_names),
            web_config=_snapshot_web_config(target.tools),
            exec_config=_snapshot_exec_config(target.exec_config),
            scratchpad=_snapshot_json_mapping(self.__scratchpad),
            workspace_sandbox=self._workspace_sandbox(),
        )

    def set_model(self, model: str) -> LLMRuntime:
        return self.__target.set_runtime_model(model)

    async def set_model_preset(self, name: str, *, session_key: str | None) -> LLMRuntime:
        if session_key is not None:
            return await self.__target.set_session_model_preset(session_key, name)
        return self.__target.set_model_preset(name)

    def set_max_iterations(self, value: int) -> None:
        self.__target.set_runtime_max_iterations(value)

    def set_context_window_tokens(self, value: int) -> LLMRuntime:
        return self.__target.set_runtime_context_window(value)

    def set_provider_retry_mode(self, value: str) -> None:
        self.__target.provider_retry_mode = value

    def set_max_tool_result_chars(self, value: int) -> None:
        self.__target.max_tool_result_chars = value

    def set_workspace_display(self, value: str) -> None:
        """Preserve MyTool display compatibility without changing path enforcement."""
        self.__workspace_display = value

    def set_scratchpad(self, key: str, value: JsonValue, *, max_keys: int) -> None:
        if key not in self.__scratchpad and len(self.__scratchpad) >= max_keys:
            raise ValueError(f"scratchpad is full (max {max_keys} keys)")
        self.__scratchpad[key] = value


def effective_model(model: str | None, provider: Any) -> str:
    """The model a runtime calls: its own, else the provider's default (NanoClaw only)."""
    if model:
        return model
    default = getattr(provider, "default_model", None)
    return default if isinstance(default, str) else ""


def _snapshot_model_presets(presets: Mapping[str, ModelPresetConfig]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "model": preset.model,
            "provider": preset.provider,
            "max_tokens": preset.max_tokens,
            "context_window_tokens": preset.context_window_tokens,
            "temperature": preset.temperature,
            "reasoning_effort": preset.reasoning_effort,
        }
        for name, preset in presets.items()
    }


def _snapshot_web_config(tools: ToolRegistry) -> dict[str, object]:
    """nanobot's ``tools.web`` view (runtime_control.py:281) of NanoClaw's fixed web tools."""
    search = tools.get("web_search")
    timeout = getattr(search, "timeout", 30)
    if isinstance(timeout, float) and timeout.is_integer():
        timeout = int(timeout)  # nanobot's config field is an int
    return {
        "enable": tools.has("web_search") or tools.has("web_fetch"),
        "use_proxy": False,
        "user_agent": None,
        "search": {
            "provider": "duckduckgo",
            "base_url": "",
            "max_results": getattr(search, "max_results", 5),
            "timeout": timeout,
        },
        "fetch": {
            "use_jina_reader": False,
        },
    }


def _snapshot_exec_config(config: ExecToolConfig) -> dict[str, object]:
    return {
        "enable": config.enable,
        "max_runtime": config.max_runtime,
        "max_runtime_limit": config.max_runtime_limit,
        "idle_timeout": config.idle_timeout,
        "idle_action": config.idle_action,
        "timeout": config.max_runtime,
        "path_prepend": config.path_prepend,
        "path_append": config.path_append,
        "sandbox": config.sandbox,
        "sandbox_ro_binds": list(config.sandbox_ro_binds),
        "sandbox_rw_binds": list(config.sandbox_rw_binds),
        "allowed_env_keys": list(config.allowed_env_keys),
        "allow_patterns": list(config.allow_patterns),
        "deny_patterns": list(config.deny_patterns),
    }


def _snapshot_json_mapping(values: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: _snapshot_json_value(value) for key, value in values.items()}


def _snapshot_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return [_snapshot_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _snapshot_json_value(item) for key, item in value.items()}
    return value
