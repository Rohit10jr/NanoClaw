"""Tool registry. Mirrors nanobot/agent/tools/registry.py."""

from __future__ import annotations

from typing import Any

from nanoclaw.agent.tools.base import Tool, ToolResult, is_tool_error

# Appended to every tool error so the model treats it as feedback, not a dead end.
RETRY_HINT = "\n\n[Analyze the error above and try a different approach.]"


def with_retry_hint(text: str) -> ToolResult:
    """Mark ``text`` as an error and append the hint exactly once."""
    return ToolResult.error(text if text.endswith(RETRY_HINT) else text + RETRY_HINT)


class ToolRegistry:
    """Holds the tools available to the agent and runs them by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

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
        """Tool definitions sorted by name, so the request is stable across turns."""
        return [self._tools[name].to_schema() for name in sorted(self._tools)]

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

        if not isinstance(params, dict):
            return tool, params, (
                f"Error: Tool '{name}' parameters must be a JSON object, got "
                f"{type(params).__name__}. Use named parameters like "
                'tool_name(param1="value1", param2="value2") matching the tool schema.'
            )

        # Layer 2 checks only required keys; full JSON Schema validation is layer 3.
        required = (tool.parameters or {}).get("required") or []
        missing = [key for key in required if key not in params]
        if missing:
            details = "; ".join(f"missing required {key}" for key in missing)
            return tool, params, f"Error: Invalid parameters for tool '{name}': {details}"

        return tool, params, None

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
