"""The ``subagent`` tool: create, message, cancel and inspect session-owned tasks.

Mirrors nanobot/agent/tools/subagent.py (c4dac64): same name, schema,
description and result texts. Replaces layer 8's ``spawn`` tool.

nanobot also marks the tool ``concurrency_safe`` so several ``subagent``
calls in one model reply run at the same time; NanoClaw's runner runs tools
one after another (docs/notes/08-subagents.md, difference #7).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from nanoclaw.agent.subagent import SubagentControlError
from nanoclaw.agent.tools.base import Tool, ToolResult
from nanoclaw.agent.tools.context import current_request_context, current_request_session_key
from nanoclaw.agent.tools.schema import object_schema
from nanoclaw.security.workspace_access import current_workspace_scope
from nanoclaw.session.manager import SessionPolicy

if TYPE_CHECKING:
    from nanoclaw.agent.subagent import SubagentManager


class SubagentTool(Tool):
    """Manage the current session's private subagent tasks."""

    def __init__(self, manager: SubagentManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "subagent"

    @property
    def description(self) -> str:
        return (
            "Manage the current session's private subagent tasks. "
            "Background results arrive automatically; avoid repeated polling. "
            "Check with a known task_id to read older results. "
            "Partial results or iteration limits do not mean completion."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return object_schema(
            action={
                "type": "string",
                "description": (
                    "create starts a task; send queues a message; cancel requests an idempotent stop; "
                    "check returns task status, receipts, and results."
                ),
                "enum": ["create", "send", "cancel", "check"],
            },
            task={
                "type": "string",
                "description": "Required for create: the task for the subagent to complete.",
                "minLength": 1,
            },
            label={"type": "string", "description": "Display label for create."},
            temperature={
                "type": "number",
                "description": "Sampling temperature for create; inherits the parent's runtime if omitted.",
                "minimum": 0.0,
                "maximum": 2.0,
            },
            wait={
                "type": "boolean",
                "description": (
                    "For create: true waits for the result; false runs in background with a completion "
                    "notification. Temporary sessions and hosts unable to receive background results always wait."
                ),
                "default": False,
            },
            task_id={
                "type": "string",
                "description": (
                    "Full task ID returned by create. Required for send/cancel; optional for check. "
                    "Omit for check to list the current session's active and recently finished tasks."
                ),
                "minLength": 1,
            },
            message={
                "type": "string",
                "description": (
                    "Required for send: text, at most 8192 UTF-8 bytes. Accepted means queued, not delivered. "
                    "Delivery receipts mean injected into the task transcript, not that the task acted on the message."
                ),
                "maxLength": 8192,
            },
            required=["action"],
        )

    async def execute(
        self,
        action: str,
        task_id: str | None = None,
        message: str | None = None,
        task: str | None = None,
        label: str | None = None,
        temperature: float | None = None,
        wait: bool = False,
        **kwargs: Any,
    ) -> str:
        if action == "create":
            if not task or not task.strip():
                return ToolResult.error("Error: create requires a non-empty task")
            return await self._create_task(task, label, temperature, wait)
        owner = current_request_session_key()
        try:
            if action == "check":
                if not owner:
                    raise SubagentControlError("task unavailable")
                if task_id is not None:
                    return json.dumps(self._manager.check(task_id, owner).as_dict(), ensure_ascii=False)
                return json.dumps(
                    {
                        "tasks": [
                            status.as_dict()
                            for status in self._manager.statuses_for_session(owner, include_history=False).values()
                        ],
                    },
                    ensure_ascii=False,
                )
            if action in {"send", "cancel"}:
                if not task_id:
                    return ToolResult.error(f"Error: {action} requires task_id")
                if action == "send":
                    return json.dumps(await self._manager.send(task_id, owner, message))
                status = await self._manager.cancel(task_id, owner)
                return json.dumps(status.as_dict(), ensure_ascii=False)
            return ToolResult.error("Error: unknown subagent action")
        except SubagentControlError as exc:
            return ToolResult.error(f"Error: {exc}")

    async def _create_task(self, task: str, label: str | None, temperature: float | None, wait: bool) -> str:
        request_ctx = current_request_context()
        if request_ctx is None or request_ctx.runtime is None:
            return ToolResult.error("Error: create requires an active model runtime")
        origin_channel = request_ctx.channel
        origin_chat_id = request_ctx.chat_id
        session_key = request_ctx.session_key or (
            f"{origin_channel}:{origin_chat_id}" if origin_channel and origin_chat_id else None
        )
        if not session_key:
            return ToolResult.error("Error: create requires an active session identity")
        inline = wait or not request_ctx.can_receive_background_results or not request_ctx.persist_session
        method = self._manager.run_inline if inline else self._manager.spawn
        return await method(
            task=task,
            runtime=request_ctx.runtime,
            label=label,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            session_key=session_key,
            origin_message_id=request_ctx.message_id,
            origin_turn_id=request_ctx.turn_id,
            temperature=temperature,
            # The child works on this turn's project (nanobot tools/subagent.py:144).
            workspace_scope=current_workspace_scope(),
            session_policy=SessionPolicy(
                persist=request_ctx.persist_session,
                log_content=request_ctx.log_content,
            ),
        )
