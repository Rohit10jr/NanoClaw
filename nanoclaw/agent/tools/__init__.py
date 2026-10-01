"""Tools the model can call."""

from __future__ import annotations

from pathlib import Path

from nanoclaw.agent.tools.base import Tool, ToolResult, is_tool_error
from nanoclaw.agent.tools.registry import RETRY_HINT, ToolRegistry


def build_default_tools(workspace: Path) -> ToolRegistry:
    """The standard tool set, rooted at ``workspace``.

    nanobot discovers tools by scanning its tools package (agent/tools/loader.py);
    NanoClaw lists them explicitly.
    """
    from nanoclaw.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
    from nanoclaw.agent.tools.shell import ExecTool
    from nanoclaw.agent.tools.web import WebFetchTool, WebSearchTool

    registry = ToolRegistry()
    for tool in (
        ReadFileTool(workspace),
        WriteFileTool(workspace),
        EditFileTool(workspace),
        ListDirTool(workspace),
        ExecTool(working_dir=workspace),
        WebFetchTool(),
        WebSearchTool(),
    ):
        registry.register(tool)
    return registry


__all__ = ["RETRY_HINT", "Tool", "ToolRegistry", "ToolResult", "build_default_tools", "is_tool_error"]
