"""Subagents: child agents that work on one task for the main agent.

Mirrors nanobot/agent/subagent.py (``SubagentManager``). A child starts fresh:
a short system prompt, the task as its only message, and the normal tools
without ``spawn`` (no subagents inside subagents). At most
``max_concurrent_subagents`` run at once; the rest wait in a queue.

Two ways to run one (the ``spawn`` tool's ``wait`` argument):

- ``run_inline``: the caller waits and gets the child's final answer.
- ``spawn``: the child runs in the background. When it finishes, a report
  (``subagent_announce.md``) is published on the message bus as a system
  :class:`InboundMessage` from sender ``subagent``, as nanobot does
  (subagent.py:885). The agent loop injects it into the session's running
  turn, or runs it as a follow-up turn when the session is idle (since layer
  14; before, the report always waited for a follow-up turn).

A child's requests go through the same context governance as the main
agent's (since layer 15): when its transcript outgrows the context window it
is compacted mid-run with the loop's ``Consolidator``, journaling nothing
(nanobot subagent.py:826-860).

Each child owns its exec sessions (since layer 16): its ``exec`` /
``exec_session`` / ``list_exec_sessions`` share a manager of its own, whose
processes are ended when the child finishes, fails or is cancelled, before
its report goes out (nanobot subagent.py:106, :313, :736).

Left out (see docs/notes/08-subagents.md): hooks that log each child tool
call and update live status, a saved transcript per child (so no summary
marker is committed anywhere) and project-folder scopes.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.skills import SkillsLoader
from nanoclaw.agent.tools import ToolRegistry, ToolResult, build_default_tools
from nanoclaw.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanoclaw.agent.tools.exec_session import ExecSessionManager
from nanoclaw.bus.events import InboundMessage
from nanoclaw.bus.queue import MessageBus
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.utils.templates import render_template

if TYPE_CHECKING:
    from nanoclaw.agent.memory import Consolidator
    from nanoclaw.config.schema import ExecToolConfig

# nanobot config AgentDefaults.max_concurrent_subagents.
DEFAULT_MAX_CONCURRENT_SUBAGENTS = 4
NO_FINAL_RESPONSE_MESSAGE = "Task completed but no final response was generated."
EXECUTION_FAILED_MESSAGE = "Error: subagent execution failed."
# The child's last message when its iterations run out (nanobot subagent.py:848),
# and the result if even that is missing (nanobot subagent.py:881).
ITERATION_LIMIT_MESSAGE = "Task stopped at its iteration limit before producing a final response."
ITERATION_LIMIT_RESULT = "Iteration limit reached."
SUBAGENT_RESULT_EVENT = "subagent_result"
SUBAGENT_SENDER_ID = "subagent"
# Report metadata (nanobot subagent.py:927); the label is NanoClaw's own, for
# ``nanoclaw ask``'s "[subagent 'X' finished]" line.
META_SUBAGENT_LABEL = "subagent_label"


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

    On the bus it travels as an ``InboundMessage`` on channel ``system`` from
    sender ``subagent``, with ``injected_event`` and ``subagent_task_id``
    metadata (nanobot subagent.py:927-947); ``to_inbound`` / ``from_inbound``
    convert.
    """

    task_id: str
    label: str
    content: str
    status: str  # "ok" | "incomplete" | "error"
    session_key: str | None
    channel: str
    chat_id: str = "direct"
    stop_reason: str | None = None
    partial: bool = False

    def to_inbound(self) -> InboundMessage:
        """The report as nanobot publishes it (subagent.py:922-947)."""
        return InboundMessage(
            channel="system",
            sender_id=SUBAGENT_SENDER_ID,
            chat_id=f"{self.channel}:{self.chat_id}",
            content=self.content,
            session_key_override=self.session_key or f"{self.channel}:{self.chat_id}",
            metadata={
                "injected_event": SUBAGENT_RESULT_EVENT,
                "subagent_task_id": self.task_id,
                "subagent_state": "done" if self.status == "ok" else self.status,
                "subagent_stop_reason": self.stop_reason,
                "subagent_partial": self.partial,
                META_SUBAGENT_LABEL: self.label,
            },
        )

    @classmethod
    def from_inbound(cls, msg: InboundMessage) -> SubagentResult:
        meta = msg.metadata
        channel, _, chat_id = msg.chat_id.partition(":")
        state = meta.get("subagent_state")
        return cls(
            task_id=str(meta.get("subagent_task_id") or ""),
            label=str(meta.get(META_SUBAGENT_LABEL) or ""),
            content=msg.content,
            status="ok" if state == "done" else str(state or "ok"),
            session_key=msg.session_key_override,
            channel=channel or "cli",
            chat_id=chat_id or "direct",
            stop_reason=meta.get("subagent_stop_reason"),
            partial=bool(meta.get("subagent_partial")),
        )


def is_subagent_report(msg: InboundMessage) -> bool:
    """A background subagent's report (nanobot loop.py:1180, :2235)."""
    return msg.sender_id == SUBAGENT_SENDER_ID and msg.metadata.get("injected_event") == SUBAGENT_RESULT_EVENT


class SubagentManager:
    """Starts subagents, limits how many run at once, and publishes their reports."""

    # How long ``close`` waits for exec cleanup (nanobot subagent.py:288).
    CANCEL_WAIT_SECONDS = 5.0

    def __init__(
        self,
        workspace: Path,
        *,
        max_tool_result_chars: int = 16_000,
        max_iterations: int = 200,  # nanobot max_tool_iterations (config/schema.py:180)
        max_concurrent_subagents: int = DEFAULT_MAX_CONCURRENT_SUBAGENTS,
        restrict_to_workspace: bool = False,
        bus: MessageBus | None = None,
        consolidator: Consolidator | None = None,
        exec_config: ExecToolConfig | None = None,
    ) -> None:
        self.workspace = workspace
        # ``config.tools.exec`` for the children's tools (nanobot
        # ``_subagent_tools_config``, subagent.py:511).
        self.exec_config = exec_config
        # Summarizes a child's history when it outgrows the context window
        # (nanobot requires one, subagent.py:223). Without it, children are
        # never compacted. The agent loop sets its own.
        self.consolidator = consolidator
        # Where reports go (nanobot ``SubagentManager.bus``); the agent loop
        # passes its own bus.
        self.bus = bus if bus is not None else MessageBus()
        self.restrict_to_workspace = restrict_to_workspace
        self.max_tool_result_chars = max_tool_result_chars
        self.max_iterations = max_iterations
        self.max_concurrent_subagents = max_concurrent_subagents
        self._run_slots = asyncio.Semaphore(max_concurrent_subagents)
        self.runner = AgentRunner()
        self._running_tasks: dict[str, asyncio.Task[str]] = {}
        self._task_statuses: dict[str, SubagentStatus] = {}
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}
        # The manager's own exec sessions, for tools built without a child's
        # manager (nanobot subagent.py:274).
        self._exec_session_manager = ExecSessionManager()
        # Each child's exec sessions, kept until their cleanup succeeded
        # (nanobot ``_SubagentTask.exec_manager`` / ``cleanup_task``).
        self._exec_managers: dict[str, ExecSessionManager] = {}
        self._exec_cleanups: dict[str, asyncio.Task[int]] = {}
        self._close_task: asyncio.Task[int] | None = None

    def _build_tools(self, exec_manager: ExecSessionManager | None = None) -> ToolRegistry:
        """The child's own tool set: the standard tools, without ``spawn``, with
        the main agent's workspace restriction and exec settings (nanobot
        ``_build_tools`` / ``_subagent_tools_config``, subagent.py:511-541)."""
        return build_default_tools(
            self.workspace,
            restrict_to_workspace=self.restrict_to_workspace,
            exec_session_manager=exec_manager or self._exec_session_manager,
            exec_config=self.exec_config,
        )

    def _new_exec_manager(self, task_id: str) -> ExecSessionManager:
        manager = ExecSessionManager()
        self._exec_managers[task_id] = manager
        return manager

    def _start_cleanup(self, task_id: str, *, retry: bool = False) -> asyncio.Task[int] | None:
        """End a child's exec sessions once; ``retry`` restarts a failed attempt.
        (nanobot subagent.py:313)"""
        manager = self._exec_managers.get(task_id)
        if manager is None:
            return None
        cleanup = self._exec_cleanups.get(task_id)
        if retry and cleanup is not None and cleanup.done():
            cleanup = None
        if cleanup is None:
            cleanup = asyncio.create_task(manager.close_all())
            cleanup.add_done_callback(partial(self._cleanup_done, task_id))
            self._exec_cleanups[task_id] = cleanup
        return cleanup

    def _cleanup_done(self, task_id: str, cleanup: asyncio.Task[int]) -> None:
        """Release the child's manager once its cleanup succeeded (nanobot ``_cleanup_done``)."""
        if not cleanup.cancelled() and cleanup.exception() is None:
            if self._exec_cleanups.get(task_id) is cleanup:
                self._exec_cleanups.pop(task_id, None)
                self._exec_managers.pop(task_id, None)

    async def _cleanup_exec(self, task_id: str) -> str | None:
        """Wait for the child's exec cleanup; the error text if it failed.
        (nanobot subagent.py:736-743)"""
        cleanup = self._start_cleanup(task_id)
        if cleanup is None:
            return None
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.opt(exception=True).error("Subagent [{}] exec cleanup failed", task_id)
            return f"Error cleaning up task processes: {exc}"
        return None

    @staticmethod
    def _display_label(task: str, label: str | None) -> str:
        return label or task[:30] + ("..." if len(task) > 30 else "")

    def _register(self, task_id: str, session_key: str | None, task: asyncio.Task[str]) -> None:
        self._running_tasks[task_id] = task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

    def _forget(self, task_id: str, session_key: str | None) -> None:
        # A child cancelled before it started still releases its exec manager.
        self._start_cleanup(task_id)
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
        origin_chat_id: str = "direct",
    ) -> str:
        """Start a subagent in the background; its report goes on the bus."""
        if temperature is not None:
            runtime = dataclasses.replace(runtime, temperature=temperature)
        task_id = str(uuid.uuid4())[:8]
        display_label = self._display_label(task, label)
        status = SubagentStatus(task_id, display_label, task, time.monotonic())
        self._task_statuses[task_id] = status
        self._new_exec_manager(task_id)
        bg_task = asyncio.create_task(
            self._run_subagent(
                task_id, task, display_label, origin_channel, session_key, status, runtime, chat_id=origin_chat_id
            )
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
        origin_chat_id: str = "direct",
    ) -> str:
        """Run a subagent and return its final answer (an error result if it failed)."""
        if temperature is not None:
            runtime = dataclasses.replace(runtime, temperature=temperature)
        task_id = str(uuid.uuid4())[:8]
        display_label = self._display_label(task, label)
        status = SubagentStatus(task_id, display_label, task, time.monotonic())
        self._task_statuses[task_id] = status
        self._new_exec_manager(task_id)
        inline_task = asyncio.create_task(
            self._run_subagent(
                task_id,
                task,
                display_label,
                origin_channel,
                session_key,
                status,
                runtime,
                announce=False,
                chat_id=origin_chat_id,
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
        chat_id: str = "direct",
    ) -> str:
        """Wait for a free slot, then run the task.

        A cancelled child still ends its exec sessions before it stops
        (nanobot ``_cancel_task`` / ``_task_done`` start the cleanup).
        """
        status.phase = "queued"
        try:
            async with self._run_slots:
                status.phase = "initializing"
                return await self._run_admitted_subagent(
                    task_id, task, label, channel, session_key, status, runtime, announce=announce, chat_id=chat_id
                )
        except asyncio.CancelledError:
            cleanup = self._start_cleanup(task_id)
            if cleanup is not None:
                with suppress(Exception):
                    await asyncio.shield(cleanup)
            raise

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
        chat_id: str = "direct",
    ) -> str:
        """Run the child agent and report the result."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)
        result = None
        messages: list[dict[str, Any]] = []
        try:
            tools = self._build_tools(exec_manager=self._exec_managers.get(task_id))
            messages = [
                {"role": "system", "content": self._build_subagent_prompt()},
                {"role": "user", "content": task},
            ]
            token = bind_request_context(
                RequestContext(channel=channel, session_key=session_key, runtime=runtime, chat_id=chat_id)
            )
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
                        **self._governance(task_id, tools, runtime),
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
        # The child's processes end before its result is reported; a failed
        # cleanup makes the result an error (nanobot subagent.py:736-744,
        # ``_SubagentTask.finish``).
        cleanup_error = await self._cleanup_exec(task_id)
        if cleanup_error is not None:
            partial_text = self._last_progress_text(result.messages[len(messages):]) if result is not None else None
            status.phase = "error"
            status.stop_reason = "error"
            status.error = cleanup_error[: self.max_tool_result_chars]
            status.partial = partial_text is not None
            final_result = partial_text if partial_text is not None else cleanup_error
        if announce:
            await self._announce_result(task_id, label, task, final_result, channel, session_key, status, chat_id)
        return final_result

    def _governance(self, task_id: str, tools: ToolRegistry, runtime: LLMRuntime) -> dict[str, Any]:
        """Context-governance settings for a child's run (nanobot subagent.py:826-860).

        The summary journals nothing (``persist=False``) and is logged under
        the child's own key, ``subagent:<task id>``.
        """
        if self.consolidator is None:
            return {}
        return {
            "context_window_tokens": runtime.context_window_tokens,
            "consolidate_history": partial(
                self.consolidator.summarize_transcript,
                runtime=runtime,
                session_key=f"subagent:{task_id}",
                tools=tools.get_definitions(),
                persist=False,
            ),
        }

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
        chat_id: str = "direct",
    ) -> None:
        """Publish the report on the bus as a system message for the main agent.

        Mirrors nanobot subagent.py:885 ``_announce_result``: ``session_key_override``
        points at the parent's session, so the loop injects the report into a
        running turn of that session instead of starting a competing one.
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
        report = SubagentResult(
            task_id=task_id,
            label=label,
            content=content,
            status=report_status,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            stop_reason=status.stop_reason,
            partial=status.partial,
        )
        await self.bus.publish_inbound(report.to_inbound())
        logger.debug("Subagent [{}] announced result to {}:{}", task_id, channel, chat_id)

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
        """Cancel all running subagents, then end every exec session they and
        this manager own, waiting at most ``CANCEL_WAIT_SECONDS``.
        (nanobot subagent.py:985)"""
        tasks = self.running_tasks()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        cleanups: set[asyncio.Task[int]] = set()
        for task_id in list(self._exec_managers):
            cleanup = self._exec_cleanups.get(task_id)
            retry = cleanup is not None and cleanup.done() and (cleanup.cancelled() or cleanup.exception() is not None)
            started = self._start_cleanup(task_id, retry=retry)
            if started is not None:
                cleanups.add(started)
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._exec_session_manager.close_all())
        cleanups.add(self._close_task)
        done, _ = await asyncio.wait(cleanups, timeout=self.CANCEL_WAIT_SECONDS)
        for task in done:
            task.result()
        if self._exec_managers:
            logger.warning("Subagent shutdown returned with exec cleanup still pending")

    @property
    def has_pending_work(self) -> bool:
        """Running children, or exec cleanup not yet done (nanobot subagent.py:1012)."""
        return bool(self._running_tasks) or bool(self._exec_managers) or self._exec_session_manager.has_pending_work

    def get_running_count(self) -> int:
        return len(self._running_tasks)

    def get_running_count_by_session(self, session_key: str) -> int:
        tids = self._session_tasks.get(session_key, set())
        return sum(1 for tid in tids if tid in self._running_tasks and not self._running_tasks[tid].done())
