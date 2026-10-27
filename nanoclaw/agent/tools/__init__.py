"""Tools the model can call."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nanoclaw.agent.tools.base import Tool, ToolResult, is_tool_error
from nanoclaw.agent.tools.registry import RETRY_HINT, ToolRegistry

if TYPE_CHECKING:
    from nanoclaw.agent.tools.exec_session import ExecSessionManager
    from nanoclaw.config.schema import ExecToolConfig


def build_default_tools(
    workspace: Path,
    *,
    restrict_to_workspace: bool = False,
    exec_session_manager: ExecSessionManager | None = None,
    exec_config: ExecToolConfig | None = None,
) -> ToolRegistry:
    """The standard tool set, rooted at ``workspace``.

    ``restrict_to_workspace`` keeps file tools and ``exec`` inside the
    workspace (nanobot ``tools.restrictToWorkspace``, off by default).

    ``exec``, ``exec_session`` and ``list_exec_sessions`` share one
    ``exec_session_manager`` (default: the module-level
    ``DEFAULT_EXEC_SESSION_MANAGER``, as in nanobot when none is given).
    ``exec_config`` is ``config.tools.exec`` (default: its defaults); with
    ``enable`` off none of the three is registered (nanobot ``enabled``,
    shell.py:169, exec_session.py:788, :947). Its sandbox fields are not
    read yet (layer 18).

    nanobot discovers tools by scanning its tools package (agent/tools/loader.py);
    NanoClaw lists them explicitly.
    """
    from nanoclaw.agent.tools.apply_patch import ApplyPatchTool
    from nanoclaw.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
    from nanoclaw.agent.tools.exec_session import ExecSessionTool, ListExecSessionsTool
    from nanoclaw.agent.tools.shell import ExecTool
    from nanoclaw.agent.tools.web import WebFetchTool, WebSearchTool
    from nanoclaw.config.schema import ExecToolConfig

    cfg = exec_config if exec_config is not None else ExecToolConfig()
    tools: list[Tool] = [
        ReadFileTool(workspace, restrict_to_workspace=restrict_to_workspace),
        WriteFileTool(workspace, restrict_to_workspace=restrict_to_workspace),
        EditFileTool(workspace, restrict_to_workspace=restrict_to_workspace),
        ApplyPatchTool(workspace, restrict_to_workspace=restrict_to_workspace),
        ListDirTool(workspace, restrict_to_workspace=restrict_to_workspace),
    ]
    if cfg.enable:
        # nanobot ExecTool.create (shell.py:173).
        tools += [
            ExecTool(
                working_dir=workspace,
                max_runtime=cfg.max_runtime,
                max_runtime_limit=cfg.max_runtime_limit,
                idle_timeout=cfg.idle_timeout,
                idle_action=cfg.idle_action,
                restrict_to_workspace=restrict_to_workspace,
                path_prepend=cfg.path_prepend,
                path_append=cfg.path_append,
                allowed_env_keys=list(cfg.allowed_env_keys),
                allow_patterns=list(cfg.allow_patterns),
                deny_patterns=list(cfg.deny_patterns),
                exec_session_manager=exec_session_manager,
            ),
            ExecSessionTool(manager=exec_session_manager),
            ListExecSessionsTool(manager=exec_session_manager),
        ]
    tools += [WebFetchTool(), WebSearchTool()]
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


__all__ = ["RETRY_HINT", "Tool", "ToolRegistry", "ToolResult", "build_default_tools", "is_tool_error"]
