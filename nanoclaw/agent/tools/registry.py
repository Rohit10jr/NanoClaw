"""Tool registry. Mirrors nanobot/agent/tools/registry.py."""

from __future__ import annotations

import json
from typing import Any

from nanoclaw.agent.tools.base import Tool, ToolResult, is_tool_error

# Appended to every tool error so the model weighs it against the user's goal
# instead of stopping. Neutral since nanobot 841d2b8: the old "try a different
# approach" pushed the model into retrying even when that was wrong.
# (nanobot registry.py:14 ``_ERROR_HINT``)
RETRY_HINT = (
    "\n\n[Consider this result in the context of the user's goal "
    "and decide whether and how to proceed.]"
)


def with_retry_hint(text: str) -> ToolResult:
    """Mark ``text`` as an error and append the hint exactly once.

    (nanobot registry.py:20 ``with_tool_error_hint``)
    """
    return ToolResult.error(text if text.endswith(RETRY_HINT) else text + RETRY_HINT)


class ToolRegistry:
    """Holds the tools available to the agent and runs them by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._cached_definitions: list[dict[str, Any]] | None = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._cached_definitions = None

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._cached_definitions = None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Tool definitions in a stable order (nanobot registry.py:97).

        Built-in tools come first, sorted by name, then MCP tools
        (``mcp_*``), sorted by name. The built-ins stay a fixed prefix when
        an MCP server connects or drops, so a provider's prompt cache over
        the tool list keeps its hit up to that point. Cached until the next
        ``register``/``unregister``.
        """
        if self._cached_definitions is None:
            builtins: list[dict[str, Any]] = []
            mcp_tools: list[dict[str, Any]] = []
            for name in sorted(self._tools):
                (mcp_tools if name.startswith("mcp_") else builtins).append(self._tools[name].to_schema())
            self._cached_definitions = builtins + mcp_tools
        return self._cached_definitions

    @staticmethod
    def _lookup_key(name: str) -> str:
        return "".join(ch.lower() for ch in name if ch.isalnum())

    def _suggest_name(self, name: str) -> str | None:
        """Suggest a tool whose name differs only in case or punctuation.

        Used only for the error message; calls never run under a guessed name.
        """
        key = self._lookup_key(name)
        if not key:
            return None
        matches = [t for t in self._tools if self._lookup_key(t) == key]
        return matches[0] if len(matches) == 1 else None

    def prepare_call(self, name: str, params: Any) -> tuple[Tool | None, Any, str | None]:
        """Find the tool and check its arguments. Returns ``(tool, params, error)``."""
        tool = self.get(name) if isinstance(name, str) else None
        if tool is None:
            suggestion = self._suggest_name(str(name or ""))
            hint = f" Did you mean '{suggestion}'? Tool names must match exactly." if suggestion else ""
            available = ", ".join(self.tool_names) or "(none)"
            return None, params, f"Error: Tool '{name}' not found.{hint} Available: {available}"

        params = self._coerce_params(tool, params)
        if not isinstance(params, dict):
            return tool, params, (
                f"Error: Tool '{name}' parameters must be a JSON object, got "
                f"{type(params).__name__}. Use named parameters like "
                'tool_name(param1="value1", param2="value2") matching the tool schema.'
            )

        params = tool.cast_params(params)
        errors = tool.validate_params(params)
        if errors:
            return tool, params, f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
        return tool, params, None

    @staticmethod
    def _coerce_argument_value(value: Any) -> Any:
        """Parse a JSON object/array string; leave anything else unchanged."""
        if value is None:
            return {}
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return {}
        if not stripped.startswith(("{", "[")):
            return value
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value

    @classmethod
    def _coerce_params(cls, tool: Tool, params: Any) -> Any:
        """Unwrap ``{"arguments": {...}}``, which some models send instead of the arguments.

        Not unwrapped when the tool really has a parameter called ``arguments``.
        """
        params = cls._coerce_argument_value(params)
        if not isinstance(params, dict) or set(params) != {"arguments"}:
            return params
        if "arguments" in (tool.parameters or {}).get("properties", {}):
            return params
        return cls._coerce_argument_value(params["arguments"])

    async def execute(self, name: str, params: Any) -> str | ToolResult:
        """Run a tool by name. Never raises: every failure becomes an error result."""
        tool, params, error = self.prepare_call(name, params)
        if error:
            return with_retry_hint(error)
        assert tool is not None
        try:
            result = await tool.execute(**params)
        except Exception as exc:
            return with_retry_hint(f"Error executing {name}: {exc}")
        if is_tool_error(result):
            return with_retry_hint(str(result))
        return result
