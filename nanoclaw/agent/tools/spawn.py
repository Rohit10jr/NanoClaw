"""The ``spawn`` tool: hand a task to a subagent. Mirrors nanobot/agent/tools/spawn.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanoclaw.agent.tools.base import Tool, ToolResult
from nanoclaw.agent.tools.context import current_request_context
from nanoclaw.security.workspace_access import current_workspace_scope
from nanoclaw.agent.tools.schema import object_schema

if TYPE_CHECKING:
    from nanoclaw.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """Starts a subagent in the background, or waits for it with ``wait=true``."""

    def __init__(self, manager: SubagentManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "Set wait=true for a consultation whose result must inform the current turn. "
            "The subagent will complete the task and report back when done. "
            "For deliverables or existing projects, inspect the workspace first "
            "and use a dedicated subdirectory when helpful."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return object_schema(
            task={"type": "string", "description": "The task for the subagent to complete"},
            label={"type": "string", "description": "Optional short label for the task (for display)"},
            temperature={
                "type": "number",
                "description": (
                    "Optional sampling temperature for the subagent "
                    "(0.0 = deterministic, higher = more creative). "
                    "Defaults to the provider's configured temperature."
                ),
                "minimum": 0.0,
                "maximum": 2.0,
            },
            wait={
                "type": "boolean",
                "description": (
                    "Wait for the subagent and return its result directly. Use this for a "
                    "blocking consultation that must inform the current turn. Defaults to "
                    "false for background execution."
                ),
                "default": False,
            },
            required=["task"],
        )

    async def execute(
        self,
        task: str,
        label: str | None = None,
        temperature: float | None = None,
        wait: bool = False,
        **kwargs: Any,
    ) -> str:
        request = current_request_context()
        if request is None or request.runtime is None:
            return ToolResult.error("Error: spawn requires an active model runtime")
        method = self._manager.run_inline if wait else self._manager.spawn
        return await method(
            task=task,
            runtime=request.runtime,
            label=label,
            origin_channel=request.channel,
            origin_chat_id=request.chat_id,
            session_key=request.session_key,
            temperature=temperature,
            # The child works on this turn's project (nanobot tools/subagent.py:144).
            workspace_scope=current_workspace_scope(),
        )
