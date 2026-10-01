"""The Tool interface. Mirrors nanobot/agent/tools/base.py."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ToolResult(str):
    """A tool's output: a plain string that can also be marked as an error.

    Tools report expected failures ("file not found") by returning
    ``ToolResult.error(...)`` rather than raising, so the model sees a clear
    message and can try something else.
    """

    is_error: bool

    def __new__(cls, content: str, *, is_error: bool = False) -> ToolResult:
        obj = str.__new__(cls, content)
        obj.is_error = is_error
        return obj

    @classmethod
    def error(cls, content: str) -> ToolResult:
        return cls(content, is_error=True)


def is_tool_error(result: Any) -> bool:
    return isinstance(result, ToolResult) and result.is_error


class Tool(ABC):
    """A capability the model can call: read a file, run a command, ..."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name the model uses in tool calls."""

    @property
    @abstractmethod
    def description(self) -> str:
        """What the tool does; the model reads this to decide when to use it."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema (type "object") for the tool's arguments."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str | ToolResult:
        """Run the tool. Return text, or ``ToolResult.error(...)`` on failure."""

    def to_schema(self) -> dict[str, Any]:
        """OpenAI function-tool definition sent to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
