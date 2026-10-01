"""Small tools for testing the registry and runner."""

from typing import Any

from nanoclaw.agent.tools import Tool, ToolResult


class _Simple(Tool):
    _name = ""
    _description = ""
    _parameters: dict[str, Any] = {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters


class EchoTool(_Simple):
    _name = "echo"
    _description = "Return the given text."
    _parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, text: str) -> str:
        return text


class AddTool(_Simple):
    _name = "add"
    _description = "Add two numbers."
    _parameters = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }

    async def execute(self, a: float, b: float) -> str:
        return str(a + b)


class BoomTool(_Simple):
    _name = "boom"
    _description = "Always raises."

    async def execute(self) -> str:
        raise RuntimeError("kaboom")


class FailTool(_Simple):
    _name = "fail"
    _description = "Always returns an error result."

    async def execute(self) -> ToolResult:
        return ToolResult.error("Error: file not found")


class SilentTool(_Simple):
    _name = "silent"
    _description = "Returns nothing."

    async def execute(self) -> str:
        return ""


class BigTool(_Simple):
    _name = "big"
    _description = "Returns a lot of text."

    async def execute(self) -> str:
        return "x" * 100
