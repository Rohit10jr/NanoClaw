"""Subagents: child agents that work on one task for the main agent.

Mirrors nanobot/agent/subagent.py (``SubagentManager``). A child starts fresh:
a short system prompt, the task as its only message, and the normal tools
without ``spawn`` (no subagents inside subagents). At most
``max_concurrent_subagents`` run at once; the rest wait in a queue.

Two ways to run one (the ``spawn`` tool's ``wait`` argument):

- ``run_inline``: the caller waits and gets the child's final answer.
- ``spawn``: the child runs in the background. When it finishes, a report
  (``subagent_announce.md``) is put on ``results``. nanobot publishes it on
  its message bus; NanoClaw has no bus, so the loop reads this queue and runs
  the report as a follow-up turn (``AgentLoop.process_subagent_result``).

Left out (see docs/notes/08-subagents.md): hooks that log each child tool
call and update live status, compaction during a child's run, project-folder
scopes, and background shell sessions shared by children.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.skills import SkillsLoader
from nanoclaw.agent.tools import ToolRegistry, ToolResult, build_default_tools
from nanoclaw.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.utils.templates import render_template

# nanobot config AgentDefaults.max_concurrent_subagents.
DEFAULT_MAX_CONCURRENT_SUBAGENTS = 4
NO_FINAL_RESPONSE_MESSAGE = "Task completed but no final response was generated."
EXECUTION_FAILED_MESSAGE = "Error: subagent execution failed."
# The child's last message when its iterations run out (nanobot subagent.py:848),
# and the result if even that is missing (nanobot subagent.py:881).
ITERATION_LIMIT_MESSAGE = "Task stopped at its iteration limit before producing a final response."
ITERATION_LIMIT_RESULT = "Iteration limit reached."
SUBAGENT_RESULT_EVENT = "subagent_result"


@dataclass(slots=True)
class SubagentStatus:
    """What a subagent is doing. (nanobot ``SubagentStatus``, without the live
    iteration, tool-event and usage fields that its hooks fill in.)"""

    task_id: str
    label: str
    task_description: str
    started_at: float  # time.monotonic()
    # queued | initializing | done | incomplete | error  (nanobot outcome states,
    # without "cancelled": a cancelled child here never reports)
    phase: str = "initializing"
    stop_reason: str | None = None
    error: str | None = None
    # The result is the child's last progress text from an unfinished run.
    partial: bool = False


@dataclass(frozen=True)
class SubagentResult:
    """A finished background subagent's report for the main agent.

    nanobot sends the same as an ``InboundMessage`` on channel ``system`` from
    sender ``subagent``, with ``injected_event`` and ``subagent_task_id`` metadata.
    """

    task_id: str
    label: str
    content: str
    status: str  # "ok" | "incomplete" | "error"
    session_key: str | None
    channel: str


class SubagentManager:
    """Starts subagents, limits how many run at once, and collects their results."""

    def __init__(
        self,
        workspace: Path,
        *,
        max_tool_result_chars: int = 16_000,
        max_iterations: int = 20,
        max_concurrent_subagents: int = DEFAULT_MAX_CONCURRENT_SUBAGENTS,
        restrict_to_workspace: bool = False,
    ) -> None:
        self.workspace = workspace
        self.restrict_to_workspace = restrict_to_workspace
        self.max_tool_result_chars = max_tool_result_chars
        self.max_iterations = max_iterations
        self.max_concurrent_subagents = max_concurrent_subagents
        self._run_slots = asyncio.Semaphore(max_concurrent_subagents)
        self.runner = AgentRunner()
        self.results: asyncio.Queue[SubagentResult] = asyncio.Queue()
        self._running_tasks: dict[str, asyncio.Task[str]] = {}
        self._task_statuses: dict[str, SubagentStatus] = {}
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}

    def _build_tools(self) -> ToolRegistry:
        """The child's own tool set: the standard tools, without ``spawn``, with
        the main agent's workspace restriction (nanobot ``_subagent_tools_config``)."""
        return build_default_tools(self.workspace, restrict_to_workspace=self.restrict_to_workspace)

    @staticmethod
    def _display_label(task: str, label: str | None) -> str:
        return label or task[:30] + ("..." if len(task) > 30 else "")

    def _register(self, task_id: str, session_key: str | None, task: asyncio.Task[str]) -> None:
        self._running_tasks[task_id] = task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

    def _forget(self, task_id: str, session_key: str | None) -> None:
        self._running_tasks.pop(task_id, None)
        self._task_statuses.pop(task_id, None)
        if session_key and (ids := self._session_tasks.get(session_key)):
            ids.discard(task_id)
            if not ids:
                del self._session_tasks[session_key]

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        session_key: str | None = None,
        temperature: float | None = None,
        *,
        runtime: LLMRuntime,
    ) -> str:
        """Start a subagent in the background; its report goes to ``results``."""
        if temperature is not None:
            runtime = dataclasses.replace(runtime, temperature=temperature)
        task_id = str(uuid.uuid4())[:8]
        display_label = self._display_label(task, label)
        status = SubagentStatus(task_id, display_label, task, time.monotonic())
        self._task_statuses[task_id] = status
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin_channel, session_key, status, runtime)
        )
        self._register(task_id, session_key, bg_task)
        bg_task.add_done_callback(lambda _: self._forget(task_id, session_key))
        logger.info("Started subagent [{}]: {}", task_id, display_label)
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def run_inline(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        session_key: str | None = None,
        temperature: float | None = None,
        *,
        runtime: LLMRuntime,
    ) -> str:
        """Run a subagent and return its final answer (an error result if it failed)."""
        if temperature is not None:
            runtime = dataclasses.replace(runtime, temperature=temperature)
        task_id = str(uuid.uuid4())[:8]
        display_label = self._display_label(task, label)
        status = SubagentStatus(task_id, display_label, task, time.monotonic())
        self._task_statuses[task_id] = status
        inline_task = asyncio.create_task(
            self._run_subagent(
                task_id, task, display_label, origin_channel, session_key, status, runtime, announce=False
            )
        )
        self._register(task_id, session_key, inline_task)
        logger.info("Started subagent [{}]: {}", task_id, display_label)
        try:
            result = await inline_task
            # nanobot subagent.py:678-682
            if status.phase == "error":
                error = status.error or result
                return ToolResult.error(error + (f"\nPartial result:\n{result}" if error != result else ""))
            if status.phase == "incomplete":
                return ToolResult.error(f"Task incomplete ({status.stop_reason}).\n{result}")
            return result
        finally:
            self._forget(task_id, session_key)

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        channel: str,
        session_key: str | None,
        status: SubagentStatus,
        runtime: LLMRuntime,
        *,
        announce: bool = True,
    ) -> str:
        """Wait for a free slot, then run the task."""
        status.phase = "queued"
        async with self._run_slots:
            status.phase = "initializing"
            return await self._run_admitted_subagent(
                task_id, task, label, channel, session_key, status, runtime, announce=announce
            )

    async def _run_admitted_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        channel: str,
        session_key: str | None,
        status: SubagentStatus,
        runtime: LLMRuntime,
        *,
        announce: bool = True,
    ) -> str:
        """Run the child agent and report the result."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)
        try:
            tools = self._build_tools()
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": self._build_subagent_prompt()},
                {"role": "user", "content": task},
            ]
            token = bind_request_context(RequestContext(channel=channel, session_key=session_key, runtime=runtime))
            try:
                result = await self.runner.run(
                    AgentRunSpec(
                        initial_messages=messages,
                        tools=tools,
                        provider=runtime.provider,
                        model=runtime.model,
                        max_iterations=self.max_iterations,
                        max_tool_result_chars=self.max_tool_result_chars,
                        workspace=self.workspace,
                        max_tokens=runtime.max_tokens,
                        temperature=runtime.temperature,
                        session_key=session_key,
                        # Out of iterations: no extra summary request, just this message.
                        max_iterations_message=ITERATION_LIMIT_MESSAGE,
                        finalize_on_max_iterations=False,
                    )
                )
            finally:
                reset_request_context(token)

            # The run's outcome (nanobot subagent.py:877-883).
            error: str | None = result.error
            if result.stop_reason in {"error", "empty_final_response"}:
                phase, final_result = "error", result.error or EXECUTION_FAILED_MESSAGE
            elif result.stop_reason == "max_iterations":
                phase, final_result, error = "incomplete", result.final_content or ITERATION_LIMIT_RESULT, None
            else:
                phase, final_result = "done", result.final_content or NO_FINAL_RESPONSE_MESSAGE
            # An unfinished run reports the child's last progress text instead
            # (nanobot subagent.py:129-136, from what _SubagentHook remembered).
            partial = None if phase == "done" else self._last_progress_text(result.messages[len(messages):])
            if partial is not None:
                final_result = partial
            status.phase = phase
            status.stop_reason = result.stop_reason
            status.error = error[: self.max_tool_result_chars] if error else None
            status.partial = partial is not None

        except Exception as e:
            # nanobot subagent.py:733-735
            logger.opt(exception=True).error("Subagent [{}] failed", task_id)
            status.phase = "error"
            status.stop_reason = "error"
            status.error = str(e)[: self.max_tool_result_chars]
            final_result = f"Error: {e}"
        if announce:
            await self._announce_result(task_id, label, task, final_result, channel, session_key, status)
        return final_result

    def _last_progress_text(self, run_messages: list[dict[str, Any]]) -> str | None:
        """The last text the child wrote alongside tool calls, cut to the result limit.

        nanobot's ``_SubagentHook._remember_output`` (subagent.py:170) records
        each model reply's text as it arrives. Replies without tool calls
        either end the run as ``completed`` or were added by the runner (the
        iteration-limit, error and empty-answer messages), so only replies
        with tool calls count here.
        """
        for message in reversed(run_messages):
            content = message.get("content")
            if message.get("role") == "assistant" and message.get("tool_calls") and isinstance(content, str):
                if content.strip():
                    return content[: self.max_tool_result_chars]
        return None

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        channel: str,
        session_key: str | None,
        status: SubagentStatus,
    ) -> None:
        """Hand the report to the main agent (nanobot: publish on the message bus).

        Mirrors nanobot subagent.py:885 ``_announce_result``.
        """
        report_status = "ok" if status.phase == "done" else status.phase
        status_text = {
            "ok": "completed successfully",
            "incomplete": "stopped before completing the task",
        }.get(report_status, "failed")
        content = render_template(
            "agent/subagent_announce.md",
            label=label,
            status_text=status_text,
            task=task,
            result=result,
            stop_reason=status.stop_reason,
            error=status.error,
            partial_result=status.partial,
        )
        await self.results.put(
            SubagentResult(
                task_id=task_id,
                label=label,
                content=content,
                status=report_status,
                session_key=session_key,
                channel=channel,
            )
        )
        logger.debug("Subagent [{}] announced result to {}", task_id, session_key or channel)

    def _build_subagent_prompt(self) -> str:
        """The child's short system prompt."""
        workspace = self.workspace.expanduser().resolve()
        skills_summary = SkillsLoader(self.workspace).build_skills_summary()
        return render_template(
            "agent/subagent_system.md",
            workspace=str(workspace),
            agent_workspace=str(workspace),
            history_log="memory/history.jsonl",
            skills_summary=skills_summary or "",
        )

    def running_tasks(self) -> list[asyncio.Task[str]]:
        """The subagent tasks that haven't finished."""
        return [task for task in self._running_tasks.values() if not task.done()]

    def statuses_for_session(self, session_key: str | None) -> dict[str, SubagentStatus]:
        """Status of the subagents started from one conversation."""
        if not session_key:
            return {}
        task_ids = self._session_tasks.get(session_key, set())
        return {tid: status for tid, status in self._task_statuses.items() if tid in task_ids}

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel one conversation's subagents. Returns how many were cancelled."""
        tasks = [
            self._running_tasks[tid]
            for tid in self._session_tasks.get(session_key, [])
            if tid in self._running_tasks and not self._running_tasks[tid].done()
        ]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    async def close(self) -> None:
        """Cancel all running subagents."""
        tasks = self.running_tasks()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_running_count(self) -> int:
        return len(self._running_tasks)

    def get_running_count_by_session(self, session_key: str) -> int:
        tids = self._session_tasks.get(session_key, set())
        return sum(1 for tid in tids if tid in self._running_tasks and not self._running_tasks[tid].done())
