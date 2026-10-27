"""Tools the model can call."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nanoclaw.agent.tools.base import Tool, ToolResult, is_tool_error
from nanoclaw.agent.tools.registry import RETRY_HINT, ToolRegistry

if TYPE_CHECKING:
    from nanoclaw.agent.tools.exec_session import ExecSessionManager


def build_default_tools(
    workspace: Path,
    *,
    restrict_to_workspace: bool = False,
    exec_session_manager: ExecSessionManager | None = None,
) -> ToolRegistry:
    """The standard tool set, rooted at ``workspace``.

    ``restrict_to_workspace`` keeps file tools and ``exec`` inside the
    workspace (nanobot ``tools.restrictToWorkspace``, off by default).

    ``exec``, ``exec_session`` and ``list_exec_sessions`` share one
    ``exec_session_manager`` (default: the module-level
    ``DEFAULT_EXEC_SESSION_MANAGER``, as in nanobot when none is given).

    nanobot discovers tools by scanning its tools package (agent/tools/loader.py);
    NanoClaw lists them explicitly.
    """
    from nanoclaw.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
    from nanoclaw.agent.tools.exec_session import ExecSessionTool, ListExecSessionsTool
    from nanoclaw.agent.tools.shell import ExecTool
    from nanoclaw.agent.tools.web import WebFetchTool, WebSearchTool

    registry = ToolRegistry()
    for tool in (
        ReadFileTool(workspace, restrict_to_workspace=restrict_to_workspace),
        WriteFileTool(workspace, restrict_to_workspace=restrict_to_workspace),
        EditFileTool(workspace, restrict_to_workspace=restrict_to_workspace),
        ListDirTool(workspace, restrict_to_workspace=restrict_to_workspace),
        ExecTool(
            working_dir=workspace,
            restrict_to_workspace=restrict_to_workspace,
            exec_session_manager=exec_session_manager,
        ),
        ExecSessionTool(manager=exec_session_manager),
        ListExecSessionsTool(manager=exec_session_manager),
        WebFetchTool(),
        WebSearchTool(),
    ):
        registry.register(tool)
    return registry


__all__ = ["RETRY_HINT", "Tool", "ToolRegistry", "ToolResult", "build_default_tools", "is_tool_error"]
