"""Subagents: session-owned child tasks the main agent can create and control.

Mirrors nanobot/agent/subagent.py at c4dac64 ("subagents v2", 6ff96f2+). The
main agent's ``subagent`` tool (tools/subagent.py) creates a task, sends it
messages, cancels it or checks on it. Each task:

- belongs to the session that created it (``owner``); only that session can
  control or read it;
- has a private child session ``subagent:<task id>`` (subagent_sessions.py)
  holding the child's transcript and its status, saved on every observation
  (when a session manager is given; the agent loop gives its own);
- starts fresh: a short system prompt, the task as its only message, and the
  normal tools without ``subagent`` (no subagents inside subagents);
- has an inbox: ``send`` queues up to 16 messages (8 KiB each), which the
  runner injects into the child's transcript before its next model call
  (layer 14's ``injection_callback``), with a receipt per message
  (``accepted`` -> ``delivered`` or ``undelivered``);
- owns its exec sessions (layer 16), ended when it finishes, fails or is
  cancelled, before its report goes out;
- ends with one outcome: ``done``, ``incomplete`` (out of iterations),
  ``error`` or ``cancelled``. An unfinished task reports the last text the
  child wrote (``_SubagentHook``), marked as partial.

At most ``max_concurrent_subagents`` run at once; the rest wait (``queued``).
``cancel`` is idempotent and waits at most 5 s (``CANCEL_WAIT_SECONDS``) for
the child to stop; a child that resists keeps its slot until it really exits.

Two ways to run one: ``run_inline`` (the caller waits and gets the result)
and ``spawn`` (background: when it finishes, a report rendered from
``subagent_announce.md`` is published on the bus as a system
:class:`InboundMessage` from sender ``subagent``; the agent loop injects it
into the session's running turn or runs it as a follow-up turn).

A child's requests go through the same context governance as the main
agent's (layer 15): compacted with the loop's ``Consolidator``, journaling
nothing (``persist=False``), logged as ``subagent:<id>``.

Status changes are published as ``SubagentTaskChanged`` on the bus (nanobot's
WebUI task cards read them; nothing in NanoClaw consumes them yet).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from loguru import logger

from nanoclaw.agent.hook import AgentHook, AgentHookContext
from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.session_activity import SessionActivity
from nanoclaw.agent.skills import SkillsLoader
from nanoclaw.agent.subagent_sessions import SubagentSessions
from nanoclaw.agent.subagent_status import SubagentState as SubagentState
from nanoclaw.agent.subagent_status import SubagentStatus
from nanoclaw.agent.tools import ToolRegistry, ToolResult, build_default_tools
from nanoclaw.agent.tools.base import Tool
from nanoclaw.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanoclaw.agent.tools.exec_session import ExecSessionManager
from nanoclaw.agent.tools.file_state import FileStates
from nanoclaw.agent.turn_hooks import AgentTurnHookSpec, build_agent_turn_hook
from nanoclaw.bus.events import InboundMessage
from nanoclaw.bus.queue import MessageBus
from nanoclaw.bus.runtime_events import RuntimeEventContext, SubagentTaskChanged
from nanoclaw.events import AgentEvent, EventSink
from nanoclaw.providers import ToolCallRequest
from nanoclaw.security.workspace_access import WorkspaceScope, bind_workspace_scope, reset_workspace_scope
from nanoclaw.session.manager import Session, SessionManager, SessionPolicy
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

# The report's first line (subagent_announce.md), for ``SubagentResult.from_inbound``.
_REPORT_HEADER_RE = re.compile(
    r"^\[Subagent '(?P<label>.*)' (?:completed successfully|was cancelled|stopped before completing the task|failed)\]"
)


class _SubagentOrigin(TypedDict):
    channel: str
    chat_id: str
    session_key: str | None


class SubagentControlError(ValueError):
    """A task control request violates ownership or admission rules."""


class SubagentMessageReceipt(TypedDict):
    task_id: str
    message_id: str
    receipt: Literal["accepted"]
    delivered: Literal[False]


@dataclass(frozen=True, slots=True)
class _SubagentOutcome:
    """Execution outcome, independent of process cleanup errors."""

    state: Literal["done", "incomplete", "error", "cancelled"]
    result: str
    stop_reason: str | None = None
    error: str | None = None


@dataclass(slots=True)
class _SubagentTask:
    """Own one execution, its inbox, and resources until cleanup succeeds.

    ``policy`` is the child session's policy (nanobot keeps it on the
    session; NanoClaw's sessions don't carry one).
    """

    status: SubagentStatus
    session: Session
    runtime: LLMRuntime
    origin: _SubagentOrigin
    origin_message_id: str | None = None
    workspace_scope: WorkspaceScope | None = None
    announce: bool = True
    begun: bool = False
    suppress_notice: bool = False
    inbox: list[tuple[str, str]] = field(default_factory=list)
    outcome: _SubagentOutcome | None = None
    exec_manager: ExecSessionManager = field(default_factory=ExecSessionManager)
    cleanup_task: asyncio.Task[int] | None = None
    policy: SessionPolicy = field(default_factory=SessionPolicy)

    def raise_if_stopping(self) -> None:
        if self.outcome is not None:
            raise asyncio.CancelledError

    def decide(self, outcome: _SubagentOutcome) -> bool:
        """Seal execution once, before cancellation or cleanup can yield."""
        if self.outcome is not None:
            return False
        self.outcome = outcome
        self.status.state = self.status.phase = "stopping"
        return True

    def finish(self, max_chars: int, cleanup_error: str | None = None) -> _SubagentOutcome:
        execution_outcome = self.outcome
        if execution_outcome is None:
            raise RuntimeError("cannot finish a task before execution has an outcome")
        outcome = execution_outcome
        if cleanup_error is not None:
            outcome = _SubagentOutcome("error", cleanup_error, "error", cleanup_error)
        status = self.status
        has_partial_result = outcome.state != "done" and bool(status.result)
        if outcome.state != "done" and status.result:
            outcome = replace(outcome, result=status.result)
        status.state = status.phase = outcome.state
        status.finished_at = time.monotonic()
        status.completed_at = time.time()
        status.result = outcome.result[:max_chars]
        status.partial = has_partial_result
        status.stop_reason = outcome.stop_reason
        status.error = outcome.error[:max_chars] if outcome.error else None
        status.task_description = status.task_description[:max_chars]
        status.label = status.label[:256]
        status.tool_events = []
        for message_id, _ in self.inbox:
            status.receipts[message_id] = "undelivered"
        self.inbox.clear()
        # Failed cleanup can retain this task after its public status is evicted.
        self.outcome = replace(
            execution_outcome,
            result=execution_outcome.result[:max_chars],
            error=execution_outcome.error[:max_chars] if execution_outcome.error else None,
        )
        return outcome

    def snapshot(self) -> SubagentStatus:
        return deepcopy(self.status)


class _SubagentHook(AgentHook):
    """Update task status at runner boundaries."""

    def __init__(
        self,
        status: SubagentStatus | None = None,
        *,
        check_active: Callable[[], None] | None = None,
        max_result_chars: int = 16000,
        on_status: Callable[[SubagentStatus], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        self._status = status
        self._check_active = check_active
        self._max_result_chars = max_result_chars
        self._on_status = on_status

    def _remember_output(self, context: AgentHookContext) -> None:
        if self._status is None or context.error or context.response is None:
            return
        content = context.response.content
        if content and content.strip():
            self._status.result = content[: self._max_result_chars]
            self._status.partial = True

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        self._remember_output(context)
        if self._status is not None:
            self._status.tool_events = [{"name": call.name, "status": "running"} for call in context.tool_calls]
        if self._status is not None and self._on_status is not None:
            await self._on_status(self._status)

    async def before_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Tool | None,
        params: dict[str, Any],
    ) -> None:
        if self._check_active is not None:
            self._check_active()

    async def after_iteration(self, context: AgentHookContext) -> None:
        if self._status is None:
            return
        self._status.iteration = context.iteration
        self._status.tool_events = list(context.tool_events)
        self._status.usage = context.usage
        self._remember_output(context)
        if context.error:
            self._status.error = str(context.error)
        if self._on_status is not None:
            await self._on_status(self._status)


@dataclass(frozen=True)
class SubagentResult:
    """A finished background task's report, as NanoClaw's ``ask`` reads it.

    NanoClaw's own helper: on the bus the report is nanobot's
    ``InboundMessage`` (channel ``system``, sender ``subagent``,
    ``injected_event`` metadata; ``_announce_result``). ``to_inbound`` /
    ``from_inbound`` convert; the label is read back from the report's
    first line.
    """

    task_id: str
    label: str
    content: str
    status: str  # "ok" | "incomplete" | "error" | "cancelled"
    session_key: str | None
    channel: str
    chat_id: str = "direct"
    stop_reason: str | None = None
    partial: bool = False

    def to_inbound(self) -> InboundMessage:
        """The report as nanobot publishes it (subagent.py:927-947)."""
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
                "subagent_message_receipts": {},
            },
        )

    @classmethod
    def from_inbound(cls, msg: InboundMessage) -> SubagentResult:
        meta = msg.metadata
        channel, _, chat_id = msg.chat_id.partition(":")
        state = meta.get("subagent_state")
        header = _REPORT_HEADER_RE.match(msg.content)
        return cls(
            task_id=str(meta.get("subagent_task_id") or ""),
            label=header.group("label") if header else "",
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
    """Manages session-owned subagent tasks. (nanobot ``SubagentManager``)

    Differences in construction: ``workspace`` comes first and the rest are
    keywords; ``consolidator`` is optional (without one, children are never
    compacted); ``exec_config`` is ``config.tools.exec`` (nanobot passes the
    whole ``ToolsConfig``); there are no deprecated ``provider`` / ``model``
    arguments: ``spawn`` and ``run_inline`` always take ``runtime``.
    """

    MAX_ACTIVE = 128
    MAX_TERMINAL = 128
    MAX_INBOX = 16
    MAX_MESSAGES = 128
    MAX_MESSAGE_BYTES = 8192

    CANCEL_WAIT_SECONDS = 5.0

    def __init__(
        self,
        workspace: Path,
        *,
        bus: MessageBus | None = None,
        max_tool_result_chars: int = 16_000,
        max_iterations: int = 200,  # nanobot max_tool_iterations (config/schema.py:180)
        max_concurrent_subagents: int = DEFAULT_MAX_CONCURRENT_SUBAGENTS,
        restrict_to_workspace: bool = False,
        consolidator: Consolidator | None = None,
        exec_config: ExecToolConfig | None = None,
        session_manager: SessionManager | None = None,
        disabled_skills: list[str] | None = None,
    ) -> None:
        self.workspace = workspace
        self.disabled_skills = set(disabled_skills or [])
        self.bus = bus if bus is not None else MessageBus()
        self.exec_config = exec_config
        self.max_tool_result_chars = max_tool_result_chars
        self.restrict_to_workspace = restrict_to_workspace
        self.max_iterations = max_iterations
        self.max_concurrent_subagents = max_concurrent_subagents
        self.consolidator = consolidator
        self._run_slots = asyncio.Semaphore(max_concurrent_subagents)
        self.runner = AgentRunner()
        # The manager's own exec sessions, for tools built without a task's
        # manager (nanobot subagent.py:274).
        self._exec_session_manager = ExecSessionManager()
        self._running_tasks: dict[str, asyncio.Task[str]] = {}
        self._tasks: dict[str, _SubagentTask] = {}
        self._terminal_statuses: dict[str, SubagentStatus] = {}
        self._closed = False
        self._close_task: asyncio.Task[int] | None = None
        self._admitting = 0
        self.sessions = SubagentSessions(session_manager) if session_manager is not None else None

    # -- status persistence and publication ----------------------------------

    async def _save_status(self, status: SubagentStatus) -> None:
        status.revision += 1
        if self.sessions is not None and not self.sessions.save(status):
            return
        self._publish_status(status)

    def _publish_status(self, status: SubagentStatus) -> None:
        origin = self._tasks[status.task_id].origin
        self.bus.publish_nowait(
            SubagentTaskChanged(
                context=RuntimeEventContext(
                    channel=origin["channel"], chat_id=origin["chat_id"], session_key=status.owner
                ),
                task_id=status.task_id,
            )
        )

    async def recover_interrupted(self) -> None:
        """Mark tasks a previous process left unfinished as ``interrupted``."""
        if self.sessions is not None:
            await self.sessions.interrupt_pending()

    # -- cleanup ---------------------------------------------------------------

    def _start_cleanup(self, record: _SubagentTask, *, retry: bool = False) -> asyncio.Task[int]:
        cleanup = record.cleanup_task
        if retry and cleanup is not None and cleanup.done():
            record.cleanup_task = None
        if record.cleanup_task is None:
            record.cleanup_task = asyncio.create_task(record.exec_manager.close_all())
            record.cleanup_task.add_done_callback(partial(self._cleanup_done, record))
        return record.cleanup_task

    def _cleanup_done(self, record: _SubagentTask, task: asyncio.Task[int]) -> None:
        if not task.cancelled() and task.exception() is None:
            self._release_task(record)

    def _release_task(self, record: _SubagentTask) -> None:
        cleanup = record.cleanup_task
        if (
            record.status.finished_at is not None
            and record.status.task_id not in self._running_tasks
            and cleanup is not None
            and cleanup.done()
            and not cleanup.cancelled()
            and cleanup.exception() is None
        ):
            self._tasks.pop(record.status.task_id, None)

    async def _finish(self, record: _SubagentTask, cleanup_error: str | None = None) -> _SubagentOutcome:
        outcome = record.finish(self.max_tool_result_chars, cleanup_error)
        await self._save_transcript(record)
        await self._save_status(record.status)
        if self.sessions is not None:
            self.sessions.release(record.status.task_id)
        self._terminal_statuses[record.status.task_id] = record.snapshot()
        while len(self._terminal_statuses) > self.MAX_TERMINAL:
            del self._terminal_statuses[next(iter(self._terminal_statuses))]
        return outcome

    async def _drain_inbox(self, record: _SubagentTask) -> list[dict[str, str]]:
        record.raise_if_stopping()
        snapshot, record.inbox = record.inbox, []
        for message_id, _ in snapshot:
            record.status.receipts[message_id] = "delivered"
        if snapshot:
            await self._save_status(record.status)
        return [{"role": "user", "content": content} for _, content in snapshot]

    # -- control -----------------------------------------------------------------

    def _owned_status(self, task_id: str, owner: str | None) -> SubagentStatus:
        status = self._terminal_statuses.get(task_id)
        record = self._tasks.get(task_id)
        if status is None and record is not None and record.status.finished_at is None:
            status = record.status
        if self.sessions is not None and owner:
            if not self.sessions.exists(owner) or not self.sessions.contains(task_id, owner):
                raise SubagentControlError("task unavailable")
            if status is None:
                status = self.sessions.status(task_id, owner)
        if not owner or status is None or status.owner != owner:
            raise SubagentControlError("task unavailable")
        return status

    def check(self, task_id: str, owner: str | None) -> SubagentStatus:
        """Return detached status for one task owned by the caller's session."""
        return deepcopy(self._owned_status(task_id, owner))

    def accepts_result(self, msg: InboundMessage, owner: str) -> bool:
        """Reject a completion whose owning child was removed after publication."""
        task_id = msg.metadata.get("subagent_task_id")
        if msg.sender_id != "subagent" or not isinstance(task_id, str) or self.sessions is None:
            return True
        return self.sessions.contains(task_id, owner)

    def read_session(self, task_id: str, owner: str) -> tuple[SubagentStatus, Session]:
        """Return task status and detached history through the ownership boundary."""
        status = self.check(task_id, owner)
        child = self.sessions.snapshot(task_id, owner) if self.sessions is not None else None
        if child is None:
            raise SubagentControlError("task unavailable")
        return status, child

    async def send(self, task_id: str, owner: str | None, message: str | None) -> SubagentMessageReceipt:
        """Accept a bounded follow-up only while execution is open."""
        status = self._owned_status(task_id, owner)
        if status.state not in {"queued", "running"}:
            raise SubagentControlError("task is not accepting messages")
        if not message or not message.strip() or len(message.encode("utf-8")) > self.MAX_MESSAGE_BYTES:
            raise SubagentControlError("message must contain text and be at most 8192 UTF-8 bytes")
        record = self._tasks[task_id]
        if len(record.inbox) >= self.MAX_INBOX or len(status.receipts) >= self.MAX_MESSAGES:
            raise SubagentControlError("task message capacity reached")
        message_id = str(uuid.uuid4())
        record.inbox.append((message_id, message))
        status.receipts[message_id] = "accepted"
        await self._save_status(status)
        return {"task_id": task_id, "message_id": message_id, "receipt": "accepted", "delivered": False}

    async def cancel(self, task_id: str, owner: str | None) -> SubagentStatus:
        """Stop an owned task and return its detached state after the bounded wait."""
        status = self._owned_status(task_id, owner)
        record = self._tasks.get(task_id)
        if record is not None and record.status.finished_at is None:
            await self._cancel_task(record)
            status = record.status
        return deepcopy(status)

    async def _cancel_task(self, record: _SubagentTask, *, suppress_notice: bool = False) -> None:
        record.suppress_notice |= suppress_notice
        task_id = record.status.task_id
        task = self._running_tasks.get(task_id)
        if record.decide(_SubagentOutcome("cancelled", "Task cancelled.", "cancelled")):
            await self._save_status(record.status)
            if task is not None and record.begun and not task.done():
                task.cancel()
        if record.status.finished_at is None:
            self._start_cleanup(record)
        if task is not None:
            # wait_for waits for cancellation acknowledgement and can hang here.
            # Keep resistant work tracked, with its slot held, until it really exits.
            await asyncio.wait({task}, timeout=self.CANCEL_WAIT_SECONDS)

    # -- status views ----------------------------------------------------------------

    def runtime_statuses(self) -> Mapping[str, SubagentStatus]:
        """Return detached active and retained terminal data, never task resources."""
        return {
            **{tid: record.snapshot() for tid, record in self._tasks.items() if record.status.finished_at is None},
            **deepcopy(self._terminal_statuses),
        }

    def _runtime_statuses_for_session(self, session_key: str) -> dict[str, SubagentStatus]:
        return {
            **{
                tid: record.snapshot()
                for tid, record in self._tasks.items()
                if record.status.owner == session_key and record.status.finished_at is None
            },
            **{
                tid: deepcopy(status)
                for tid, status in self._terminal_statuses.items()
                if status.owner == session_key
            },
        }

    def _merge_session_statuses(
        self, session_key: str, runtime: Mapping[str, SubagentStatus], *, include_history: bool
    ) -> Mapping[str, SubagentStatus]:
        if self.sessions is not None and not self.sessions.exists(session_key):
            return {}
        statuses = self.sessions.load(session_key) if include_history and self.sessions is not None else {}
        for task_id, status in runtime.items():
            if self.sessions is not None and not self.sessions.contains(task_id, session_key):
                continue
            persisted = statuses.get(task_id)
            if persisted is None or status.revision >= persisted.revision:
                statuses[task_id] = status
        return statuses

    def statuses_for_session(
        self, session_key: str | None, *, include_history: bool = True
    ) -> Mapping[str, SubagentStatus]:
        """Return only tasks owned by the given session, never a global fallback."""
        if not session_key:
            return {}
        return self._merge_session_statuses(
            session_key, self._runtime_statuses_for_session(session_key), include_history=include_history
        )

    async def read_statuses_for_session(self, session_key: str) -> Mapping[str, SubagentStatus]:
        """Capture runtime observations on the owner loop; read stored metadata on a worker."""
        runtime = self._runtime_statuses_for_session(session_key)
        return await asyncio.to_thread(self._merge_session_statuses, session_key, runtime, include_history=True)

    # -- tools and admission -------------------------------------------------------------

    def _build_tools(
        self,
        exec_manager: ExecSessionManager | None = None,
        restrict_to_workspace: bool | None = None,
    ) -> ToolRegistry:
        """The child's own tool set: the standard tools, without ``subagent``,
        with the workspace restriction and exec settings (nanobot
        ``_build_tools`` / ``_subagent_tools_config``, subagent.py:511-541).

        Built from the agent workspace; a bound workspace scope supplies the
        project. Each child gets a fresh file-read state (nanobot
        subagent.py:535), so it never inherits the parent session's reads."""
        return build_default_tools(
            self.workspace,
            restrict_to_workspace=(
                self.restrict_to_workspace if restrict_to_workspace is None else restrict_to_workspace
            ),
            exec_session_manager=exec_manager or self._exec_session_manager,
            exec_config=self.exec_config,
            file_states=FileStates(),
        )

    async def _create_task(
        self,
        task: str,
        label: str | None,
        origin_channel: str,
        origin_chat_id: str,
        session_key: str | None,
        origin_message_id: str | None,
        temperature: float | None,
        workspace_scope: WorkspaceScope | None,
        runtime: LLMRuntime,
        *,
        announce: bool,
        origin_turn_id: str | None,
        session_policy: SessionPolicy | None,
    ) -> _SubagentTask:
        if temperature is not None:
            runtime = dataclasses.replace(runtime, temperature=temperature)
        # A failed cleanup owns admission even after its status has been evicted.
        if self._closed or len(self._tasks) + self._admitting >= self.MAX_ACTIVE:
            raise SubagentControlError("subagent manager is closed or at task capacity")
        owner = session_key or f"{origin_channel}:{origin_chat_id}"
        task_id = str(uuid.uuid4())
        status = SubagentStatus(
            task_id=task_id,
            revision=1,
            label=label or task[:30] + ("..." if len(task) > 30 else ""),
            task_description=task,
            started_at=time.monotonic(),
            owner=owner,
            origin_message_id=origin_message_id,
            origin_turn_id=origin_turn_id,
        )
        policy = session_policy or SessionPolicy()
        if self.sessions is not None:
            child = self.sessions.create(status, policy=policy)
        else:
            child = Session(key=SubagentSessions.key(task_id))
            child.add_message("user", task)
        record = _SubagentTask(
            status=status,
            session=child,
            runtime=runtime,
            origin={"channel": origin_channel, "chat_id": origin_chat_id, "session_key": owner},
            origin_message_id=origin_message_id,
            workspace_scope=workspace_scope,
            announce=announce,
            policy=policy,
        )
        self._tasks[task_id] = record
        self._publish_status(status)
        execution = asyncio.create_task(self._run_subagent(record))
        self._running_tasks[task_id] = execution
        execution.add_done_callback(partial(self._task_done, record))
        logger.info(
            "Started subagent [{}]: {}", task_id, record.status.label if policy.log_content else "[content hidden]"
        )
        return record

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        origin_message_id: str | None = None,
        temperature: float | None = None,
        workspace_scope: WorkspaceScope | None = None,
        *,
        runtime: LLMRuntime,
        origin_turn_id: str | None = None,
        session_policy: SessionPolicy | None = None,
    ) -> str:
        """Start background work and route its terminal result to the parent.

        ``workspace_scope`` is the scope of the turn that started it; the
        child works on the same project (nanobot subagent.py:772, :824).
        """
        try:
            record = await self._create_task(
                task,
                label,
                origin_channel,
                origin_chat_id,
                session_key,
                origin_message_id,
                temperature,
                workspace_scope,
                runtime,
                announce=True,
                origin_turn_id=origin_turn_id,
                session_policy=session_policy,
            )
        except SubagentControlError as exc:
            return ToolResult.error(f"Error: {exc}")
        status = record.status
        return f"Subagent [{status.label}] started (id: {status.task_id}). I'll notify you when it completes."

    async def run_inline(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        origin_message_id: str | None = None,
        temperature: float | None = None,
        workspace_scope: WorkspaceScope | None = None,
        *,
        runtime: LLMRuntime,
        origin_turn_id: str | None = None,
        session_policy: SessionPolicy | None = None,
    ) -> str:
        """Wait for the same task lifecycle without a background notice."""
        try:
            record = await self._create_task(
                task,
                label,
                origin_channel,
                origin_chat_id,
                session_key,
                origin_message_id,
                temperature,
                workspace_scope,
                runtime,
                announce=False,
                origin_turn_id=origin_turn_id,
                session_policy=session_policy,
            )
        except SubagentControlError as exc:
            return ToolResult.error(f"Error: {exc}")
        execution = self._running_tasks[record.status.task_id]
        try:
            result = await asyncio.shield(execution)
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if execution.cancelled() and caller is not None and not caller.cancelling():
                partial_result = record.status.result if record.status.partial else None
                return ToolResult.error(
                    "Task cancelled." + (f"\nPartial result:\n{partial_result}" if partial_result else "")
                )
            await self._cancel_task(record, suppress_notice=True)
            raise
        if record.status.state == "error":
            error = record.status.error or result
            return ToolResult.error(error + (f"\nPartial result:\n{result}" if error != result else ""))
        if record.status.state == "incomplete":
            return ToolResult.error(f"Task incomplete ({record.status.stop_reason}).\n{result}")
        return result

    def _task_done(self, record: _SubagentTask, task: asyncio.Task[str]) -> None:
        self._running_tasks.pop(record.status.task_id, None)
        error = None if task.cancelled() else task.exception()
        if error is not None:
            logger.error(
                "Subagent [{}] failed: {}",
                record.status.task_id,
                error if record.policy.log_content else type(error).__name__,
            )
        if record.status.finished_at is None:
            # An externally cancelled execution still owns cleanup and a terminal result.
            record.decide(
                _SubagentOutcome("error", f"Error: {error}", "error", str(error))
                if error is not None
                else _SubagentOutcome("cancelled", "Task cancelled.", "cancelled")
            )
            self._start_cleanup(record)
            settlement = asyncio.create_task(self._settle_unfinished_task(record))
            self._running_tasks[record.status.task_id] = settlement
            settlement.add_done_callback(partial(self._task_done, record))
        self._release_task(record)

    async def _save_transcript(self, record: _SubagentTask) -> None:
        if self.sessions is not None:
            self.sessions.save_transcript(record.session)

    async def _settle_unfinished_task(self, record: _SubagentTask) -> str:
        outcome = await self._finish(record)
        return outcome.result

    # -- execution -------------------------------------------------------------------------

    async def _run_subagent(self, record: _SubagentTask) -> str:
        """Execute once, seal the outcome, then clean up and publish once."""
        status = record.status
        task_text, label = status.task_description, status.label
        record.begun = True
        try:
            record.raise_if_stopping()
            async with self._run_slots:
                record.raise_if_stopping()
                status.state = "running"
                status.phase = "initializing"
                await self._save_status(status)
                outcome = await self._run_admitted_subagent(record)
                record.decide(outcome)
        except asyncio.CancelledError:
            record.decide(_SubagentOutcome("cancelled", "Task cancelled.", "cancelled"))
        except Exception as exc:
            logger.opt(exception=record.policy.log_content).error("Subagent [{}] failed", status.task_id)
            record.decide(_SubagentOutcome("error", f"Error: {exc}", "error", str(exc)))
        cleanup_error = None
        try:
            await asyncio.shield(self._start_cleanup(record))
        except Exception as exc:
            logger.opt(exception=record.policy.log_content).error("Subagent [{}] exec cleanup failed", status.task_id)
            cleanup_error = f"Error cleaning up task processes: {exc}"
        outcome = await self._finish(record, cleanup_error)
        if record.announce and not record.suppress_notice and not self._closed:
            await self._announce_result(
                status.task_id,
                label,
                task_text,
                outcome.result,
                record.origin,
                "ok" if outcome.state == "done" else outcome.state,
                record.origin_message_id,
                receipts=dict(status.receipts),
                stop_reason=status.stop_reason,
                error=status.error,
                partial_result=status.partial,
            )
        if outcome.state == "cancelled":
            raise asyncio.CancelledError
        return outcome.result

    async def _run_admitted_subagent(self, record: _SubagentTask) -> _SubagentOutcome:
        """Execute the admitted task with task-owned shell resources."""
        status, origin, runtime = record.status, record.origin, record.runtime
        task_id, label = status.task_id, status.label
        origin_message_id, workspace_scope = record.origin_message_id, record.workspace_scope
        policy = record.policy
        logger.info("Subagent [{}] starting task: {}", task_id, label if policy.log_content else "[content hidden]")

        async def _on_checkpoint(payload: dict[str, Any]) -> None:
            record.raise_if_stopping()
            phase = payload.get("phase", status.phase)
            iteration = payload.get("iteration", status.iteration)
            if (phase, iteration) != (status.phase, status.iteration):
                status.phase, status.iteration = phase, iteration
                await self._save_status(status)

        root = workspace_scope.project_path if workspace_scope is not None else self.workspace
        restrict = workspace_scope.restrict_to_workspace if workspace_scope is not None else None
        # Construct from the agent workspace; the bound scope below supplies the project cwd.
        tools = self._build_tools(exec_manager=record.exec_manager, restrict_to_workspace=restrict)
        system_prompt = self._build_subagent_prompt(workspace=root)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *record.session.get_history(),
        ]
        activity = SessionActivity()

        async def save_observation(status: SubagentStatus) -> None:
            record.session.messages = activity.transcript()
            await self._save_transcript(record)
            await self._save_status(status)

        async def publish_activity(event: AgentEvent) -> None:
            if activity.remember(event):
                await save_observation(status)

        events = EventSink(publish_activity)
        hook = build_agent_turn_hook(
            AgentTurnHookSpec(
                events=events,
                streaming=True,
                channel=origin["channel"],
                chat_id=origin["chat_id"],
                message_id=origin_message_id,
                session_key=record.session.key,
                workspace=root,
                turn_hooks=[
                    activity,
                    _SubagentHook(
                        status,
                        check_active=record.raise_if_stopping,
                        max_result_chars=self.max_tool_result_chars,
                        on_status=save_observation,
                    ),
                ],
                ephemeral=not policy.persist,
                run_extra_hooks_for_ephemeral=True,
                log_content=policy.log_content,
            )
        )

        request_token = bind_request_context(
            RequestContext(
                channel=origin["channel"],
                chat_id=origin["chat_id"],
                message_id=origin_message_id,
                session_key=record.session.key,
                runtime=runtime,
                log_content=policy.log_content,
                persist_session=policy.persist,
            )
        )
        token = bind_workspace_scope(workspace_scope) if workspace_scope is not None else None
        try:
            governance: dict[str, Any] = {}
            if self.consolidator is not None:
                governance = {
                    "context_window_tokens": runtime.context_window_tokens,
                    "consolidate_history": partial(
                        self.consolidator.summarize_transcript,
                        runtime=runtime,
                        session_key=record.session.key,
                        tools=tools.get_definitions(),
                        persist=False,
                    ),
                }
            result = await self.runner.run(
                AgentRunSpec(
                    initial_messages=messages,
                    tools=tools,
                    provider=runtime.provider,
                    model=runtime.model,
                    max_iterations=self.max_iterations,
                    max_tool_result_chars=self.max_tool_result_chars,
                    max_tokens=runtime.max_tokens,
                    temperature=runtime.temperature,
                    hook=hook,
                    max_iterations_message=ITERATION_LIMIT_MESSAGE,
                    finalize_on_max_iterations=False,
                    checkpoint_callback=_on_checkpoint,
                    injection_callback=partial(self._drain_inbox, record),
                    session_key=record.session.key,
                    workspace=root if policy.persist else None,
                    events=events,
                    **governance,
                )
            )
        finally:
            if token is not None:
                reset_workspace_scope(token)
            reset_request_context(request_token)
        status.usage = result.usage
        if result.messages:
            record.session.messages = activity.transcript(result.messages)
        checkpoint = result.summary_checkpoint
        if checkpoint is not None:
            boundary = checkpoint.transcript_boundary
            if 0 <= boundary <= len(result.messages):
                insert_at = sum(message.get("role") != "system" for message in result.messages[:boundary])
                record.session.commit_summary_checkpoint(checkpoint.summary, insert_at=insert_at)
        if result.stop_reason in {"error", "empty_final_response"}:
            final_result = result.error or EXECUTION_FAILED_MESSAGE
            return _SubagentOutcome("error", final_result, result.stop_reason, result.error)
        if result.stop_reason == "max_iterations":
            return _SubagentOutcome("incomplete", result.final_content or ITERATION_LIMIT_RESULT, result.stop_reason)
        final_result = result.final_content or NO_FINAL_RESPONSE_MESSAGE
        return _SubagentOutcome("done", final_result, result.stop_reason, result.error)

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: _SubagentOrigin,
        status: str,
        origin_message_id: str | None = None,
        *,
        receipts: dict[str, str] | None = None,
        stop_reason: str | None = None,
        error: str | None = None,
        partial_result: bool = False,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus.

        ``session_key_override`` points at the parent's session, so the loop
        injects the report into a running turn of that session instead of
        starting a competing one (nanobot subagent.py:917-921).
        """
        status_text = {
            "ok": "completed successfully",
            "cancelled": "was cancelled",
            "incomplete": "stopped before completing the task",
        }.get(status, "failed")

        announce_content = render_template(
            "agent/subagent_announce.md",
            label=label,
            status_text=status_text,
            task=task,
            result=result,
            stop_reason=stop_reason,
            error=error,
            partial_result=partial_result,
        )

        override = origin.get("session_key") or f"{origin['channel']}:{origin['chat_id']}"
        if self.sessions is not None and (
            not self.sessions.exists(override) or not self.sessions.contains(task_id, override)
        ):
            return
        metadata: dict[str, Any] = {
            "injected_event": SUBAGENT_RESULT_EVENT,
            "subagent_task_id": task_id,
            "subagent_state": "done" if status == "ok" else status,
            "subagent_stop_reason": stop_reason,
            "subagent_partial": partial_result,
        }
        metadata["subagent_message_receipts"] = receipts or {}
        if receipts:
            announce_content += "\nMessage receipts: " + json.dumps(receipts)
        if origin_message_id:
            metadata["origin_message_id"] = origin_message_id
        msg = InboundMessage(
            channel="system",
            sender_id=SUBAGENT_SENDER_ID,
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
            session_key_override=override,
            metadata=metadata,
        )

        await self.bus.publish_inbound(msg)
        logger.debug("Subagent [{}] announced result to {}:{}", task_id, origin["channel"], origin["chat_id"])

    def _build_subagent_prompt(self, workspace: Path | None = None) -> str:
        """The child's short system prompt (nanobot subagent.py:952).

        With a project other than the agent workspace, it names the agent
        workspace and gives the journal's absolute path.
        """
        agent_workspace = self.workspace.expanduser().resolve()
        project_workspace = workspace.expanduser().resolve() if workspace else agent_workspace
        skills_summary = SkillsLoader(
            self.workspace,
            disabled_skills=self.disabled_skills,
        ).build_skills_summary(workspace=project_workspace)
        history_log = (
            str(agent_workspace / "memory" / "history.jsonl")
            if agent_workspace != project_workspace
            else "memory/history.jsonl"
        )
        return render_template(
            "agent/subagent_system.md",
            workspace=str(project_workspace),
            agent_workspace=str(agent_workspace),
            history_log=history_log,
            skills_summary=skills_summary or "",
        )

    # -- stopping ----------------------------------------------------------------------------

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        records = [
            record
            for tid, record in self._tasks.items()
            if record.status.owner == session_key and tid in self._running_tasks and not self._running_tasks[tid].done()
        ]
        # Suppress every notice before cancellation can yield to a sibling.
        for record in records:
            record.suppress_notice = True
        await asyncio.gather(*(self._cancel_task(record, suppress_notice=True) for record in records))
        return len(records)

    async def close(self) -> None:
        """Request cancellation and bounded cleanup, retaining unfinished work."""
        self._closed = True
        await asyncio.gather(
            *(self._cancel_task(self._tasks[tid], suppress_notice=True) for tid in list(self._running_tasks))
        )
        cleanups: set[asyncio.Task[int]] = set()
        for record in list(self._tasks.values()):
            cleanup = record.cleanup_task
            retry = cleanup is not None and cleanup.done() and (cleanup.cancelled() or cleanup.exception() is not None)
            cleanups.add(self._start_cleanup(record, retry=retry))
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._exec_session_manager.close_all())
        cleanups.add(self._close_task)
        done, _ = await asyncio.wait(cleanups, timeout=self.CANCEL_WAIT_SECONDS)
        for task in done:
            task.result()
        if self._tasks:
            logger.warning("Subagent shutdown returned with tasks or cleanup still pending")

    def running_tasks(self) -> list[asyncio.Task[str]]:
        """The subagent tasks that haven't finished (NanoClaw's ``ask`` waits on them)."""
        return [task for task in self._running_tasks.values() if not task.done()]

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)

    @property
    def has_pending_work(self) -> bool:
        """Completed turns may still own exec cleanup, including failed cleanup."""
        return bool(self._tasks) or self._exec_session_manager.has_pending_work

    def get_running_count_by_session(self, session_key: str) -> int:
        """Return the number of currently running subagents for a session."""
        return sum(
            1
            for tid, task in self._running_tasks.items()
            if self._tasks[tid].status.owner == session_key and not task.done()
        )
