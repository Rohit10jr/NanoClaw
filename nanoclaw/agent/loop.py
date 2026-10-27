"""The agent loop: turns against saved sessions, served from a message bus.

Mirrors nanobot/agent/loop.py. A turn is ``_process_message``: load the
session, dispatch a slash command or build the request, run the agent, save
the result (nanobot's turn stages ``_restore_turn``, ``_dispatch_command``,
``_build_turn``, ``_run_turn``, ``_persist_turn``, ``_prepare_outbound``).
Two ways in:

- ``process_direct`` (``nanoclaw ask``, compaction, Dream, tests): one turn,
  text in, :class:`AgentRunResult` out.
- ``run`` (since layer 14, ``nanoclaw chat``): consume :class:`InboundMessage`
  from the bus, as nanobot does. Priority commands (``/stop``, ``/status``)
  run at once; everything else goes to the session's queue, served by one
  worker task per busy session. A message that arrives while the session's
  turn runs is injected into that turn (``_drain_pending``), and a turn about
  to end waits for the session's running subagents so their reports fold in
  (``_wait_for_pending``). Output, stream events and the final reply go back
  on the bus through a :class:`TurnDelivery`. ``cancel_active_tasks`` (``/stop``)
  cancels a session's worker and subagents; the cancelled turn keeps its
  latest checkpoint (finished tool results) in the history.

Since layer 6 a turn also compacts the conversation when it no longer fits the
model's context window, and the loop runs nanobot's ``/compact``, ``/new``
(with archiving) and ``/dream`` commands.

Since layer 8 the main agent has the ``spawn`` tool. A background subagent's
report arrives on the bus (since layer 14): injected into a running turn of
its session, or run as a follow-up turn (``process_subagent_result``): saved
as an assistant message and shown to the model as new input.

Since layer 13 each turn builds nanobot's hook chain (``build_agent_turn_hook``)
over an :class:`EventSink`: progress, tool hints, reasoning, stream deltas and
ends, retry waits and context compaction all arrive there as events.

Left out of nanobot's loop: unified sessions, goals and turn continuations,
automation (cron) turns, WebUI recovery admission and follow-up journal,
model presets, workspace scopes, the global concurrency gate and idle
auto-compaction (the 1 s idle tick of ``run`` is kept).
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger

from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.context_governance import merge_adjacent_user_messages_for_model
from nanoclaw.agent.hook import AgentHook
from nanoclaw.agent.memory import COMPACTION_NANOBOT, Consolidator, MemoryStore
from nanoclaw.agent.messages import EMPTY_FINAL_RESPONSE_MESSAGE, is_blank
from nanoclaw.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec, ProgressCallback
from nanoclaw.agent.subagent import (
    DEFAULT_MAX_CONCURRENT_SUBAGENTS,
    SUBAGENT_RESULT_EVENT,
    SubagentManager,
    SubagentResult,
    is_subagent_report,
)
from nanoclaw.agent.tools import ToolRegistry, build_default_tools
from nanoclaw.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanoclaw.agent.tools.spawn import SpawnTool
from nanoclaw.agent.turn_delivery import TurnDelivery, TurnDeliveryFactory
from nanoclaw.agent.turn_hooks import AgentTurnHookSpec, build_agent_turn_hook
from nanoclaw.bus.events import InboundMessage, OutboundMessage
from nanoclaw.bus.outbound_events import StreamDeltaEvent, StreamedResponseEvent, StreamEndEvent
from nanoclaw.bus.queue import MessageBus
from nanoclaw.command import CommandContext, CommandRouter, command_text, register_builtin_commands
from nanoclaw.events import NO_EVENTS, AgentEvent, ContextCompactionEvent, EventSink
from nanoclaw.providers import LLMProvider
from nanoclaw.runtime_context import (
    RUNTIME_CONTEXT_HISTORY_META,
    RUNTIME_CONTEXT_MESSAGE_META,
    RuntimeContextBlock,
    append_runtime_context,
    current_time_block,
    iteration_limit_block,
    record_iteration_limit,
)
from nanoclaw.session import Session, SessionManager
from nanoclaw.session.recovery import (
    PENDING_USER_TURN_KEY,
    RUNTIME_CHECKPOINT_KEY,
    finish_turn,
    restore_pending_interruption,
    restore_runtime_checkpoint,
)
from nanoclaw.session.summary import HIDDEN_HISTORY_META, SessionSummary, session_summary_from_metadata
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.utils.progress_events import output_events
from nanoclaw.utils.tokens import estimate_prompt_tokens

DREAM_NO_INPUT_MESSAGE = "\n".join([
    "Dream has no conversation history to process yet.",
    "",
    "Dream reads new entries from `memory/history.jsonl` after the current Dream cursor.",
    "Conversations only reach that file when they are compacted (`nanoclaw compact`, or",
    "automatically when they grow too long) or cleared with `nanoclaw ask --new`.",
])
# Headroom for token-estimate drift (nanobot providers/base.py).
CONTEXT_SAFETY_BUFFER = 1024
# nanobot's default is 200,000; NanoClaw's default model, gpt-4o-mini, has 128,000.
DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
DEFAULT_SESSION_KEY = "cli:direct"
LAST_USAGE_KEY = "_last_usage"
# How long a turn about to end waits for its session's running subagents
# (nanobot loop.py:135).
_SUBAGENT_TERMINAL_WAIT_SECONDS = 300.0


class ContextWindowExceededError(RuntimeError):
    """The request is still too big for the model after compacting."""

    def __init__(self, *, session_key: str | None, estimated_tokens: int, input_budget: int, source: str) -> None:
        self.session_key = session_key
        self.estimated_tokens = estimated_tokens
        self.input_budget = input_budget
        self.source = source
        super().__init__(
            "Model input exceeds the local context budget "
            f"for {session_key or 'default'}: {estimated_tokens}/{input_budget} via {source}"
        )


@dataclass
class TurnContext:
    """One turn's state while it moves through the stages. (nanobot ``TurnContext``)

    ``kind`` is ``"user"`` (user input) or ``"system"`` (a subagent report).
    ``result`` is the run's result; it stays None when a slash command handled
    the input (``command_handled``), whose reply is ``outbound``.
    """

    msg: InboundMessage
    session_key: str
    kind: str
    events: EventSink
    streaming: bool
    pending_queue: asyncio.Queue[InboundMessage] | None = None
    ephemeral: bool = False
    tools: ToolRegistry | None = None
    hooks: list[AgentHook] = field(default_factory=list)
    delivery: TurnDelivery | None = None
    session: Session | None = None
    result: AgentRunResult | None = None
    outbound: OutboundMessage | None = None
    command_handled: bool = False
    streamed_content: bool = False
    started_at: float = field(default_factory=time.time)
    latency_ms: int | None = None


class AgentLoop:
    """Runs turns against saved sessions, directly or from a message bus."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        *,
        model: str | None = None,
        tools: ToolRegistry | None = None,
        sessions: SessionManager | None = None,
        bus: MessageBus | None = None,
        # nanobot config max_tool_iterations (schema.py:180); phase 2 decision 7.
        max_iterations: int = 200,
        max_tool_result_chars: int = 16_000,
        max_concurrent_subagents: int = DEFAULT_MAX_CONCURRENT_SUBAGENTS,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
        restrict_to_workspace: bool = False,
        compaction: str = COMPACTION_NANOBOT,
        provider_retry_mode: str = "standard",
    ) -> None:
        self.provider = provider
        # For the main agent's model calls only; subagents and memory archiving
        # retry in standard mode, as in nanobot (loop.py:367, :1346).
        self.provider_retry_mode = provider_retry_mode
        self.restrict_to_workspace = restrict_to_workspace
        self.workspace = workspace
        self.model = model
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_window_tokens = context_window_tokens
        self.bus = bus if bus is not None else MessageBus()
        self.context = ContextBuilder(workspace)
        self.memory: MemoryStore = self.context.memory
        self.subagents = SubagentManager(
            workspace,
            max_tool_result_chars=max_tool_result_chars,
            max_iterations=max_iterations,
            max_concurrent_subagents=max_concurrent_subagents,
            restrict_to_workspace=restrict_to_workspace,
            bus=self.bus,
        )
        if tools is None:
            # The main agent's tools: the standard set plus spawn. Subagents and
            # Dream build their own sets without it.
            tools = build_default_tools(workspace, restrict_to_workspace=restrict_to_workspace)
            tools.register(SpawnTool(self.subagents))
            logger.info("Registered {} tools: {}", len(tools.tool_names), tools.tool_names)
        self.tools = tools
        self.sessions = sessions if sessions is not None else SessionManager(workspace)
        self.consolidator = Consolidator(
            self.memory, self.sessions, self.context.build_messages, self.tools.get_definitions, compaction=compaction
        )
        self.runner = AgentRunner()
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)
        self.turn_delivery_factory = TurnDeliveryFactory(self.bus)
        self.start_time = time.time()
        self._running = False
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Per session: the worker task(s) and the inbox of a busy session
        # (nanobot ``_active_tasks`` / ``_pending_queues``).
        self._active_tasks: dict[str, set[asyncio.Task[Any]]] = {}
        self._pending_queues: dict[str, asyncio.Queue[InboundMessage]] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._close_lock: asyncio.Lock | None = None

    @property
    def runtime(self) -> LLMRuntime:
        return LLMRuntime(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            context_window_tokens=self.context_window_tokens,
        )

    # -- direct entry points ----------------------------------------------

    async def process_direct(
        self,
        content: str,
        session_key: str = DEFAULT_SESSION_KEY,
        channel: str = "cli",
        *,
        chat_id: str = "direct",
        ephemeral: bool = False,
        tools: ToolRegistry | None = None,
        on_progress: ProgressCallback | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        events: EventSink = NO_EVENTS,
        streaming: bool = False,
        hooks: list[AgentHook] | None = None,
    ) -> AgentRunResult:
        """Run one turn in the session and return the result. (nanobot loop.py:2667)

        Shares the session lock with bus turns, so they run one at a time. An
        ``ephemeral`` turn starts from an empty session and saves nothing
        (used by Dream); ``tools`` replaces the default tool set for this turn.
        A registered slash command is dispatched as on the bus; its reply is
        the result's ``final_content``.

        Output goes to ``events``, not to the bus (nanobot publishes through
        a turn delivery). ``on_progress``, ``on_stream`` and ``on_stream_end``
        are nanobot's callbacks, adapted with ``output_events``. The answer
        streams when ``on_stream`` is given or ``streaming`` is set (nanobot
        loop.py:1931). ``hooks`` are extra hooks for this turn only.
        """
        if channel == "system":
            raise ValueError("channel 'system' is reserved for internal messages")
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        async with self._get_session_lock(session_key):
            ctx = await self._process_message(
                msg,
                session_key=session_key,
                ephemeral=ephemeral,
                tools=tools,
                events=output_events(
                    default=events, on_progress=on_progress, on_stream=on_stream, on_stream_end=on_stream_end
                ),
                streaming=streaming or on_stream is not None,
                hooks=hooks,
            )
        if ctx.result is None:
            reply = ctx.outbound.content if ctx.outbound is not None else ""
            return AgentRunResult(final_content=reply, messages=[])
        return ctx.result

    async def process_subagent_result(
        self,
        report: SubagentResult | InboundMessage,
        *,
        on_progress: ProgressCallback | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        events: EventSink = NO_EVENTS,
        streaming: bool = False,
    ) -> AgentRunResult:
        """Let the main agent react to a finished background subagent.

        The direct form of the system turn ``run`` makes from a report (nanobot
        ``_build_turn`` with ``is_subagent``): the report is saved as an
        assistant message (once per task) and given to the model as new input,
        without the time block. The model's reply is what the user sees.
        Output arguments as in :meth:`process_direct`.
        """
        msg = report.to_inbound() if isinstance(report, SubagentResult) else report
        key = self._system_session_key(msg)
        async with self._get_session_lock(key):
            ctx = await self._process_message(
                msg,
                session_key=key,
                events=output_events(
                    default=events, on_progress=on_progress, on_stream=on_stream, on_stream_end=on_stream_end
                ),
                streaming=streaming or on_stream is not None,
            )
        assert ctx.result is not None
        return ctx.result

    async def next_subagent_result(self) -> SubagentResult | None:
        """The next background subagent report, waiting for running subagents.

        Reads the bus directly, so only for callers that don't ``run`` the
        loop (``nanoclaw ask``). Returns ``None`` when no report is waiting and
        no subagent is running.
        """
        while True:
            while not self.bus.inbound.empty():
                msg = self.bus.inbound.get_nowait()
                if is_subagent_report(msg):
                    return SubagentResult.from_inbound(msg)
                logger.warning("Dropping non-subagent inbound message for {}", msg.session_key)
            running = self.subagents.running_tasks()
            if not running:
                return None
            # A subagent publishes its report before its task ends.
            await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)

    # -- the bus ------------------------------------------------------------

    async def run(self) -> None:
        """Consume inbound messages until stopped. (nanobot loop.py:1428)

        Turns run as session worker tasks, so the loop stays free for
        ``/stop``. The 1 s timeout is nanobot's idle tick (it drives idle
        auto-compaction there; NanoClaw has none yet).
        """
        self._running = True
        try:
            logger.info("Agent loop started")
            while self._running:
                try:
                    async with asyncio.timeout(1.0):
                        msg = await self.bus.consume_inbound()
                except TimeoutError:
                    continue
                except asyncio.CancelledError:
                    # Real cancellation ends the loop; a stray CancelledError
                    # leaked by an integration does not (nanobot loop.py:1446).
                    task = asyncio.current_task()
                    if not self._running or (task is not None and task.cancelling()):
                        raise
                    logger.warning("Ignoring leaked CancelledError while consuming inbound messages")
                    continue
                except Exception as e:
                    logger.warning("Error consuming inbound message: {}, continuing...", e)
                    continue
                try:
                    await self._dispatch_inbound(msg)
                except Exception:
                    logger.exception("Failed to dispatch inbound message for {}", msg.session_key)
        finally:
            await self.aclose()

    def stop(self) -> None:
        """Stop ``run`` at its next idle tick. (nanobot loop.py:1855)"""
        self._running = False
        logger.info("Agent loop stopping")

    async def _dispatch_inbound(self, msg: InboundMessage) -> None:
        """Route one input through command and session admission. (nanobot loop.py:1468)"""
        raw = command_text(msg)
        key = msg.session_key
        if msg.channel != "system" and self.commands.is_priority(raw):
            await self._dispatch_command_inline(msg, key, raw, self.commands.dispatch_priority)
            return
        # A busy session answers its other commands at once instead of
        # queueing them behind the running turn.
        if key in self._pending_queues and msg.channel != "system" and self.commands.is_dispatchable_command(raw):
            await self._dispatch_command_inline(msg, key, raw, self.commands.dispatch)
            return
        await self._enqueue_session_message(msg)

    async def _dispatch_command_inline(
        self,
        msg: InboundMessage,
        key: str,
        raw: str,
        dispatch_fn: Callable[[CommandContext], Awaitable[InboundMessage | OutboundMessage | None]],
    ) -> None:
        """Run a command from the run() loop and publish its reply. (nanobot loop.py:849)"""
        if self.commands.requires_session_queue(raw):
            await self._enqueue_session_message(msg)
            return
        ctx = CommandContext(
            msg=msg,
            session=None,
            key=key,
            raw=raw,
            loop=self,
            is_user_turn=msg.is_user_input and msg.channel != "system" and msg.sender_id != "subagent",
            events=self.turn_delivery_factory.create(msg, key).events,
        )
        result = await dispatch_fn(ctx)
        self._track_command_tasks(ctx)
        if isinstance(result, InboundMessage):
            await self._enqueue_session_message(result)
        elif result is not None:
            await self.bus.publish_outbound(result)

    async def _enqueue_session_message(self, msg: InboundMessage) -> None:
        """Queue session work; start the session's worker if it has none. (nanobot loop.py:1564)"""
        key = msg.session_key
        pending = self._pending_queues.get(key)
        if pending is not None:
            pending.put_nowait(msg)
            return
        new_pending: asyncio.Queue[InboundMessage] = asyncio.Queue()
        new_pending.put_nowait(msg)
        self._pending_queues[key] = new_pending
        task = asyncio.create_task(self._run_session_queue(key, new_pending))
        self._track_active_task(key, task)

    async def _run_session_queue(self, session_key: str, pending: asyncio.Queue[InboundMessage]) -> None:
        """The session's only worker: process its inbox in order until empty.
        (nanobot loop.py:1594)"""
        try:
            while self._pending_queues.get(session_key) is pending:
                try:
                    msg = pending.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    await self._dispatch_one(msg, pending)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Session worker failed one message for {}; continuing FIFO", session_key)
        except asyncio.CancelledError:
            await self._cancel_pending_messages(session_key, pending)
            raise
        finally:
            if self._pending_queues.get(session_key) is pending:
                self._pending_queues.pop(session_key, None)

    async def _cancel_pending_messages(self, session_key: str, pending: asyncio.Queue[InboundMessage]) -> None:
        """Drop queued input after its worker stopped; queued session commands
        still get their (empty) completion. (nanobot loop.py:1636)"""
        while not pending.empty():
            msg = pending.get_nowait()
            if msg.channel != "system" and self.commands.requires_session_queue(command_text(msg)):
                delivery = self.turn_delivery_factory.unrouted(msg, session_key)
                await delivery.complete(None, publish_completion=True)

    async def _dispatch_one(self, msg: InboundMessage, pending: asyncio.Queue[InboundMessage]) -> None:
        """Process one message under the session lock while later input stays in
        the session's inbox. (nanobot loop.py:1656)"""
        session_key = msg.session_key
        delivery = self.turn_delivery_factory.unrouted(msg, session_key)
        completion_published = False
        try:
            async with self._get_session_lock(session_key):
                try:
                    delivery = self.turn_delivery_factory.create(msg, session_key, enable_stream=True)
                    ctx = await self._process_message(
                        msg,
                        session_key=session_key,
                        pending_queue=pending,
                        delivery=delivery,
                        events=delivery.events,
                        streaming=delivery.streaming,
                    )
                    await delivery.complete(ctx.outbound, publish_completion=True)
                    completion_published = True
                except asyncio.CancelledError:
                    logger.info("Task cancelled for session {}", session_key)
                    try:
                        await delivery.abort_stream()
                    except Exception:
                        logger.opt(exception=True).debug("Could not close stream for cancelled session {}", session_key)
                    # Keep what the turn finished (the latest checkpoint), and
                    # always save, even without a checkpoint (nanobot 421bd6b).
                    try:
                        session = self.sessions.get_or_create(session_key)
                        restore_runtime_checkpoint(session)
                        finish_turn(session)
                        self.sessions.save(session)
                        logger.info("Saved cancelled turn for session {}", session_key)
                    except Exception:
                        logger.opt(exception=True).debug("Could not finalize cancelled turn for session {}", session_key)
                    raise
                except Exception:
                    logger.exception("Error processing message for session {}", session_key)
                    await delivery.fail(publish_completion=True)
                finally:
                    await delivery.idle()
        except asyncio.CancelledError:
            if not completion_published and self.commands.requires_session_queue(command_text(msg)):
                await delivery.complete(None, publish_completion=True)
            raise

    def _track_active_task(self, key: str, task: asyncio.Task[Any]) -> None:
        """(nanobot loop.py:932)"""
        tasks = self._active_tasks.setdefault(key, set())
        tasks.add(task)
        task.add_done_callback(partial(self._active_task_done, key, tasks))

    def _active_task_done(self, key: str, tasks: set[asyncio.Task[Any]], task: asyncio.Task[Any]) -> None:
        tasks.discard(task)
        if not tasks and self._active_tasks.get(key) is tasks:
            self._active_tasks.pop(key, None)

    def _track_command_tasks(self, ctx: CommandContext) -> None:
        """Keep background work a command started (/dream) until it ends, so
        ``aclose`` can wait for it (nanobot ``schedule_background``, loop.py:1844)."""
        for task in ctx.tasks:
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def wait_for_background_tasks(self) -> None:
        """Wait for work commands started in the background (/dream)."""
        while self._background_tasks:
            await asyncio.gather(*tuple(self._background_tasks), return_exceptions=True)

    async def cancel_active_tasks(self, key: str) -> int:
        """Cancel and await the session's worker and subagents; returns how many.

        nanobot ``_cancel_active_tasks`` (loop.py:948). One difference: a
        command running on the session's own worker (/new on an idle session)
        never cancels that worker; nanobot cancels it and the cancellation is
        swallowed, leaving the worker marked as cancelling for its next turn.
        """
        current = asyncio.current_task()
        group = self._active_tasks.get(key, set())
        tasks = tuple(t for t in group if t is not current)
        if current in group:
            group.difference_update(tasks)
        else:
            self._active_tasks.pop(key, None)
        pending = self._pending_queues.get(key)
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await t
        if tasks and pending is not None and self._pending_queues.get(key) is pending:
            # A worker cancelled before its first step never ran its cleanup.
            self._pending_queues.pop(key, None)
            await self._cancel_pending_messages(key, pending)
        return cancelled + await self.subagents.cancel_by_session(key)

    async def aclose(self) -> None:
        """Cancel session workers, background work and subagents. (nanobot loop.py:1771)"""
        if self._close_lock is None:
            self._close_lock = asyncio.Lock()
        async with self._close_lock:
            current = asyncio.current_task()
            groups = self._active_tasks
            pending_queues = {key: self._pending_queues[key] for key in groups if key in self._pending_queues}
            tasks = tuple({t for group in groups.values() for t in group if t is not current})
            groups.clear()
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                for key, pending in pending_queues.items():
                    if self._pending_queues.get(key) is pending:
                        self._pending_queues.pop(key)
                        await self._cancel_pending_messages(key, pending)
                if self._background_tasks:
                    for task in tuple(self._background_tasks):
                        task.cancel()
                    await asyncio.gather(*tuple(self._background_tasks), return_exceptions=True)
            finally:
                self._background_tasks.clear()
                await self.subagents.close()

    async def close(self) -> None:
        """Same as :meth:`aclose` (NanoClaw's older name)."""
        await self.aclose()

    # -- one turn -----------------------------------------------------------

    @staticmethod
    def _system_session_key(msg: InboundMessage) -> str:
        """A system message's session: its override, or the chat it names.
        (nanobot loop.py:1881)"""
        if msg.session_key_override:
            return msg.session_key_override
        channel, chat_id = msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
        return f"{channel}:{chat_id}"

    async def _process_message(
        self,
        msg: InboundMessage,
        *,
        session_key: str | None = None,
        pending_queue: asyncio.Queue[InboundMessage] | None = None,
        delivery: TurnDelivery | None = None,
        events: EventSink = NO_EVENTS,
        streaming: bool = False,
        ephemeral: bool = False,
        tools: ToolRegistry | None = None,
        hooks: list[AgentHook] | None = None,
    ) -> TurnContext:
        """One message through the turn stages; the caller holds the session lock.
        (nanobot loop.py:1860)"""
        kind = "user" if msg.is_user_input else "system"
        if kind == "system":
            key = session_key or self._system_session_key(msg)
        else:
            key = session_key or msg.session_key
        ctx = TurnContext(
            msg=msg,
            session_key=key,
            kind=kind,
            events=events,
            streaming=streaming,
            pending_queue=pending_queue,
            ephemeral=ephemeral,
            tools=tools,
            hooks=list(hooks or []),
            delivery=delivery,
        )
        if streaming:
            # Whether the last streamed segment had text, so the final answer
            # is already on screen (nanobot loop.py:1945-1961).
            publish = events.publish
            segment_streamed = False

            async def track_output(event: AgentEvent) -> None:
                nonlocal segment_streamed
                if isinstance(event, StreamDeltaEvent) and event.content:
                    segment_streamed = True
                elif isinstance(event, StreamEndEvent):
                    ctx.streamed_content, segment_streamed = segment_streamed, False
                if publish is not None:
                    await publish(event)

            ctx.events = EventSink(track_output, events.accepts_type)

        await self._restore_turn(ctx)
        if ctx.kind == "user" and await self._dispatch_command(ctx):
            return ctx
        await self._run_and_save_turn(ctx)
        self._prepare_outbound(ctx)
        return ctx

    async def _restore_turn(self, ctx: TurnContext) -> None:
        """Load the session and close an earlier interrupted turn. (nanobot loop.py:2079)"""
        self.subagents.max_iterations = self.max_iterations  # nanobot _sync_subagent_runtime_limits
        msg = ctx.msg
        if ctx.ephemeral:
            ctx.session = Session(key=ctx.session_key)
        else:
            ctx.session = self.sessions.get_or_create(ctx.session_key)
        if ctx.kind == "system":
            logger.info("Processing system message from {}", msg.sender_id)
        elif not ctx.ephemeral:
            preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)
        else:
            logger.info("Processing message from {}:{}: [content hidden]", msg.channel, msg.sender_id)
        if ctx.delivery is not None:
            await ctx.delivery.started()
        if ctx.ephemeral:
            return
        # A turn that died mid-way: its checkpoint first, then an error reply
        # for a user message left unanswered.
        changed = restore_runtime_checkpoint(ctx.session)
        changed = restore_pending_interruption(ctx.session) or changed
        if changed:
            self.sessions.save(ctx.session)

    async def _dispatch_command(self, ctx: TurnContext) -> bool:
        """Run a registered slash command; True when it handled the input.
        (nanobot loop.py:2153)

        A command's reply is saved as a hidden pair of ``_command`` messages
        (not shown to the model), except for /new which clears the session.
        """
        session = ctx.session
        assert session is not None
        raw = command_text(ctx.msg)
        cmd_ctx = CommandContext(
            msg=ctx.msg,
            session=session,
            key=ctx.session_key,
            raw=raw,
            loop=self,
            is_user_turn=ctx.msg.channel != "system" and ctx.msg.sender_id != "subagent",
            events=ctx.events,
        )
        command_matched = self.commands.command_name(raw) is not None
        result = await self.commands.dispatch(cmd_ctx)
        self._track_command_tasks(cmd_ctx)
        if isinstance(result, InboundMessage):
            ctx.msg = result
            return False
        if result is not None:
            ctx.outbound = result
            ctx.command_handled = True
            if cmd_ctx.raw.lower() != "/new" and not ctx.ephemeral:
                self._persist_user_message_early(ctx.msg.content, session, [], _command=True)
                session.add_message("assistant", result.content, _command=True)
                session.metadata.pop(PENDING_USER_TURN_KEY, None)
                self.sessions.save(session)
            return True
        ctx.command_handled = command_matched
        return command_matched

    def _runtime_blocks(self, content: str, session: Session, *, ephemeral: bool) -> list[RuntimeContextBlock]:
        """The runtime context for user input: the time, the iteration-limit
        notice (user turns only, nanobot loop.py:663) and $skill-name skills
        (nanobot ``_resolve_runtime_context_for_request``, loop.py:829)."""
        blocks: list[RuntimeContextBlock] = [current_time_block()]
        if not ephemeral and (limit_block := iteration_limit_block(session.metadata)) is not None:
            blocks.append(limit_block)
        skill_context = self.context.skills.build_explicit_skill_runtime_context(content)
        if skill_context is not None and skill_context not in blocks:
            blocks.append(skill_context)
        return blocks

    async def _run_and_save_turn(self, ctx: TurnContext) -> None:
        """Build the request, run the agent, save the turn.
        (nanobot ``_build_turn``, ``_run_turn``, ``_persist_turn``)"""
        session = ctx.session
        assert session is not None
        msg, ephemeral = ctx.msg, ctx.ephemeral
        content = msg.content
        tools = ctx.tools if ctx.tools is not None else self.tools
        events = ctx.events

        # Build: history first, then save the user's message before running,
        # so it survives a crash or an error mid-turn.
        summary = session_summary_from_metadata(session.metadata, fallback_last_active=session.updated_at)
        history = session.get_history()
        boundary = len(session.messages)
        blocks: list[RuntimeContextBlock] = []
        if ctx.kind == "system":
            # Saved as the agent's own record; it reaches the model as input.
            if self._persist_subagent_followup(session, msg):
                logger.debug("Subagent result persisted for session {}", session.key)
                self.sessions.save(session)
            persisted_early = True
        else:
            blocks = self._runtime_blocks(content, session, ephemeral=ephemeral)
            persisted_early = not ephemeral and self._persist_user_message_early(content, session, blocks)
        messages = self.context.build_messages(
            history, content, channel=msg.channel, runtime_context_blocks=blocks, session_summary=summary
        )
        # Temporary turns are checked too, but never journal or save a summary.
        messages, checkpoint = await self._compact_if_needed(
            session, messages, history, summary, content, msg.channel, blocks, tools,
            persist=not ephemeral, events=events,
        )

        if ctx.delivery is not None:
            await ctx.delivery.running(started_at=time.time())
        result = await self._run_agent_loop(ctx, messages, tools)
        if ctx.kind == "user" and is_blank(result.final_content):
            result = dataclasses.replace(result, final_content=EMPTY_FINAL_RESPONSE_MESSAGE)
        ctx.result = result
        ctx.latency_ms = max(0, int((time.time() - ctx.started_at) * 1000))
        if ctx.delivery is not None:
            ctx.delivery.record_latency(ctx.latency_ms)

        # Persist: the summary boundary (if compacted), then everything the
        # runner added after the user's message.
        if not ephemeral:
            if checkpoint is not None:
                session.commit_summary_checkpoint(checkpoint, insert_at=boundary)
            skip = len(messages) if persisted_early else len(messages) - 1
            # Token use of the last turn, for /status (nanobot ``_last_usage``).
            if result.usage is not None:
                session.metadata[LAST_USAGE_KEY] = result.usage.to_dict()  # nanobot loop.py:2384
            # Only user turns set or clear the iteration-limit flag; a
            # subagent follow-up leaves it alone (nanobot loop.py:2385).
            if ctx.kind == "user":
                record_iteration_limit(session.metadata, result.stop_reason)
            self._save_turn(session, result.messages, skip)
            finish_turn(session)
            self.sessions.save(session)

    async def _run_agent_loop(
        self, ctx: TurnContext, messages: list[dict[str, Any]], tools: ToolRegistry
    ) -> AgentRunResult:
        """Run the runner with the turn's hooks, checkpoints and injections.
        (nanobot loop.py:1051)"""
        session = ctx.session
        assert session is not None
        ephemeral, pending_queue, events = ctx.ephemeral, ctx.pending_queue, ctx.events

        async def _checkpoint(payload: dict[str, Any]) -> None:
            # The latest finished step, kept so a cancelled turn can be saved
            # (nanobot loop.py:1082 stores it in the session).
            if ephemeral:
                return
            session.metadata[RUNTIME_CHECKPOINT_KEY] = copy.deepcopy(payload)
            self.sessions.save(session)

        async def _drain_pending(*, first_msg: InboundMessage | None = None) -> list[dict[str, Any]]:
            """One atomic snapshot of the messages already waiting. (nanobot loop.py:1098)"""
            if pending_queue is None:
                return []
            # Fix the snapshot before converting: later arrivals stay queued.
            pending_messages: list[InboundMessage] = [first_msg] if first_msg is not None else []
            for _ in range(pending_queue.qsize()):
                try:
                    pending_messages.append(pending_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            consumed = 0
            try:
                converted: list[dict[str, Any]] = []
                for pending_msg in pending_messages:
                    # Commands are barriers: their worker turn handles them.
                    if not self._can_inject_message(pending_msg):
                        break
                    converted.append(self._injected_user_message(pending_msg, session, ephemeral=ephemeral))
                consumed = len(converted)
                return converted
            finally:
                # Commit only the converted prefix; the rest goes back first,
                # ahead of anything that arrived meanwhile.
                unconsumed = pending_messages[consumed:]
                if unconsumed:
                    later = [pending_queue.get_nowait() for _ in range(pending_queue.qsize())]
                    for pending_msg in [*unconsumed, *later]:
                        pending_queue.put_nowait(pending_msg)

        terminal_wait_deadline: float | None = None

        async def _wait_for_pending() -> list[dict[str, Any]]:
            """When the turn would end: wait for a running subagent of this
            session to report, up to the terminal wait. (nanobot loop.py:1217)"""
            nonlocal terminal_wait_deadline
            items = await _drain_pending()
            if items or pending_queue is None or self.subagents.get_running_count_by_session(session.key) == 0:
                return items
            now = asyncio.get_running_loop().time()
            if terminal_wait_deadline is None:
                terminal_wait_deadline = now + _SUBAGENT_TERMINAL_WAIT_SECONDS
            remaining = terminal_wait_deadline - now
            if remaining <= 0:
                return []
            try:
                msg = await asyncio.wait_for(pending_queue.get(), timeout=remaining)
            except TimeoutError:
                logger.warning("Timeout waiting for sub-agent completion before session {} exits", session.key)
                return []
            return await _drain_pending(first_msg=msg)

        # Tools such as spawn read the turn's session, chat and model settings.
        token = bind_request_context(
            RequestContext(
                channel=ctx.msg.channel, session_key=session.key, runtime=self.runtime, chat_id=ctx.msg.chat_id
            )
        )
        # nanobot loop.py:1310-1328: the progress hook first, then this
        # turn's hooks; temporary turns get the progress hook only.
        hook = build_agent_turn_hook(
            AgentTurnHookSpec(
                events=events,
                streaming=ctx.streaming,
                channel=ctx.msg.channel,
                session_key=session.key,
                workspace=self.workspace,
                turn_hooks=ctx.hooks,
                ephemeral=ephemeral,
            )
        )
        try:
            result = await self.runner.run(
                AgentRunSpec(
                    initial_messages=messages,
                    tools=tools,
                    provider=self.provider,
                    model=self.model,
                    max_iterations=self.max_iterations,
                    max_tool_result_chars=self.max_tool_result_chars,
                    # Temporary turns never write spill files.
                    workspace=None if ephemeral else self.workspace,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    session_key=session.key,
                    provider_retry_mode=self.provider_retry_mode,
                    hook=hook,
                    events=events,
                    checkpoint_callback=_checkpoint,
                    injection_callback=_drain_pending,
                    terminal_injection_callback=_wait_for_pending,
                )
            )
        finally:
            reset_request_context(token)
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            # Push the final text through the stream so streaming outputs
            # show it (nanobot loop.py:1386-1402; without goals it always streams).
            if events.publish is not None and ctx.streaming:
                await events.publish(StreamDeltaEvent(content=result.final_content or ""))
                await events.publish(StreamEndEvent())
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result

    def _can_inject_message(self, msg: InboundMessage) -> bool:
        """Commands stay out of injected batches. (nanobot loop.py:1007)"""
        return msg.channel == "system" or not self.commands.is_dispatchable_command(command_text(msg))

    def _injected_user_message(self, msg: InboundMessage, session: Session, *, ephemeral: bool) -> dict[str, Any]:
        """A queued message as a user message for the running turn.
        (nanobot ``_to_user_message``, loop.py:1119)

        User input gets its runtime context, recorded in ``_meta`` as for the
        turn's own message; a subagent report is marked hidden with its task
        id, so it is saved once and not shown as a chat turn.
        """
        row: dict[str, Any] = {
            "role": "user",
            "content": msg.content,
            "_meta": {"timestamp": msg.timestamp.isoformat()},
        }
        if HIDDEN_HISTORY_META in msg.metadata:
            row[HIDDEN_HISTORY_META] = msg.metadata[HIDDEN_HISTORY_META]
        if msg.is_user_input:
            blocks = self._runtime_blocks(msg.content, session, ephemeral=ephemeral)
            row["content"], marker = append_runtime_context(msg.content, blocks)
            if marker is not None:
                row["_meta"][RUNTIME_CONTEXT_MESSAGE_META] = marker
        if is_subagent_report(msg):
            subagent_marker: dict[str, Any] = {"kind": SUBAGENT_RESULT_EVENT}
            task_id = msg.metadata.get("subagent_task_id")
            if isinstance(task_id, str) and task_id:
                subagent_marker["subagent_task_id"] = task_id
                row["subagent_task_id"] = task_id
            row[HIDDEN_HISTORY_META] = subagent_marker
            row["injected_event"] = SUBAGENT_RESULT_EVENT
        return row

    def _prepare_outbound(self, ctx: TurnContext) -> None:
        """The turn's final reply for the bus. (nanobot loop.py:2416, :2054)"""
        delivery, result = ctx.delivery, ctx.result
        if delivery is None or result is None:
            return
        delivery.record_stop_reason(result.stop_reason)
        if ctx.kind == "system":
            ctx.outbound = delivery.background_response(
                result.final_content,
                stop_reason=result.stop_reason,
                streamed=ctx.streamed_content,
                latency_ms=ctx.latency_ms,
            )
            return
        message = delivery.delivery_message
        metadata = dict(message.metadata or {})
        if ctx.latency_ms is not None:
            metadata["latency_ms"] = ctx.latency_ms
        streamed = ctx.streamed_content and result.stop_reason not in {"error", "tool_error"}
        ctx.outbound = OutboundMessage(
            channel=message.channel,
            chat_id=message.chat_id,
            content=result.final_content or "",
            event=StreamedResponseEvent() if streamed else None,
            metadata=metadata,
        )

    def _input_budget(self) -> int:
        """Tokens the request may use: window minus the reply and a safety margin."""
        if not self.context_window_tokens:
            return 0
        budget = self.context_window_tokens - self.max_tokens - CONTEXT_SAFETY_BUFFER
        return budget if budget > 0 else 0

    async def _compact_if_needed(
        self,
        session: Session,
        messages: list[dict[str, Any]],
        history: list[dict[str, Any]],
        summary: SessionSummary | None,
        content: str,
        channel: str,
        blocks: list[RuntimeContextBlock],
        tools: ToolRegistry,
        *,
        persist: bool = True,
        events: EventSink = NO_EVENTS,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Summarize the history if the request doesn't fit the context window.

        Returns the request to send and the new summary (``None`` if nothing
        was compacted). Mirrors nanobot's compaction before a model request
        (``ContextGovernor.prepare_request``) at the first request of a turn:
        the history (everything before the new message) is replaced by a
        summary in the system prompt, and the new message is kept. nanobot
        checks before every request of a turn; NanoClaw only before the first
        (see docs/notes/06-memory.md).

        Publishes :class:`ContextCompactionEvent` (reason ``context_pressure``)
        started, then succeeded, failed or cancelled (nanobot
        context_governance.py:634-714).
        """
        if not self.context_window_tokens:
            return messages, None
        definitions = tools.get_definitions()
        budget = self._input_budget()
        measured, source = estimate_prompt_tokens(merge_adjacent_user_messages_for_model(messages), definitions)
        if budget > 0 and measured < budget:
            return messages, None

        compaction_id = uuid.uuid4().hex
        await events.emit(ContextCompactionEvent(compaction_id, "started", reason="context_pressure"))
        try:
            accepted = merge_adjacent_user_messages_for_model(messages[: 1 + len(history)])
            new_summary = await self.consolidator.summarize_transcript(
                accepted,
                summary["text"] if summary else None,
                runtime=self.runtime,
                session_key=session.key,
                tools=definitions,
                persist=persist,
            )
            if not new_summary:
                raise ContextWindowExceededError(
                    session_key=session.key, estimated_tokens=measured, input_budget=budget, source=source
                )
            compacted = self.context.build_messages(
                [],
                content,
                channel=channel,
                runtime_context_blocks=blocks,
                session_summary={"text": new_summary, "last_active": datetime.now().astimezone().isoformat()},
            )
        except (Exception, asyncio.CancelledError) as exc:
            phase = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            await events.emit(ContextCompactionEvent(compaction_id, phase, reason="context_pressure"))
            raise
        await events.emit(ContextCompactionEvent(compaction_id, "succeeded", reason="context_pressure"))
        measured, source = estimate_prompt_tokens(merge_adjacent_user_messages_for_model(compacted), definitions)
        if budget <= 0 or measured >= budget:
            raise ContextWindowExceededError(
                session_key=session.key, estimated_tokens=measured, input_budget=budget, source=source
            )
        return compacted, new_summary

    # -- session commands -----------------------------------------------------

    async def new_session(self, session_key: str = DEFAULT_SESSION_KEY, *, lock_held: bool = False) -> None:
        """Start the conversation over (nanobot's ``/new`` command).

        Holds the session lock (nanobot 9650f4f / research D6), unless the
        caller already does (``lock_held``: /new on the session's worker).
        Messages not yet archived are first summarized into the history
        journal, so Dream can still learn from them. nanobot does that in the
        background and ignores failures; NanoClaw waits for it and ignores
        failures.
        """
        if not lock_held:
            async with self._get_session_lock(session_key):
                await self._new_session(session_key)
            return
        await self._new_session(session_key)

    async def _new_session(self, session_key: str) -> None:
        session = self.sessions.get_or_create(session_key)
        snapshot = list(session.messages)
        if session.last_archived < len(snapshot):
            archived = Session(
                key=session.key,
                messages=snapshot,
                created_at=session.created_at,
                updated_at=session.updated_at,
                metadata=dict(session.metadata),
                last_archived=session.last_archived,
            )
            try:
                await self.consolidator.archive_session(archived, archive_end=len(snapshot), runtime=self.runtime)
            except Exception:
                pass
        session.clear()
        self.sessions.save(session)
        self.sessions.invalidate(session_key)

    async def compact_session(
        self, session_key: str = DEFAULT_SESSION_KEY, *, events: EventSink = NO_EVENTS, lock_held: bool = False
    ) -> str | None:
        """Summarize the whole conversation now (nanobot's ``/compact`` command).

        Returns the summary, ``""`` if there was nothing new, ``None`` on failure.
        Compaction events go to ``events`` with ``notify=True`` and reason
        ``manual`` (nanobot command/builtin.py:358-365). Holds the session
        lock unless the caller does (``lock_held``: /compact on the worker).
        """
        if not lock_held:
            async with self._get_session_lock(session_key):
                return await self.compact_session(session_key, events=events, lock_held=True)
        return await self.consolidator.compact_idle_session(
            session_key, runtime=self.runtime, events=events, notify=True, reason="manual"
        )

    async def run_dream(self) -> str:
        """Let Dream fold new journal entries into the memory files (nanobot ``/dream``).

        Runs one temporary turn with Dream's instructions and restricted file
        tools. The Dream cursor only moves forward if the run completed, so an
        interrupted run sees the same entries next time.
        """
        start = time.monotonic()
        result = self.memory.build_dream_prompt()
        if result is None:
            logger.info("Dream: nothing to process")
            return DREAM_NO_INPUT_MESSAGE
        prompt, last_cursor = result
        # nanobot reads the change from its git store; NanoClaw compares the files.
        files = (self.memory.memory_file, self.memory.soul_file, self.memory.user_file)
        before = [MemoryStore.read_file(f) for f in files]
        try:
            response = await self.process_direct(
                prompt, session_key=MemoryStore.dream_session_key(), ephemeral=True, tools=self.memory.build_dream_tools()
            )
            elapsed = time.monotonic() - start
            if response.stop_reason == "completed":
                self.memory.set_last_dream_cursor(last_cursor)
                if [MemoryStore.read_file(f) for f in files] != before:
                    logger.info("Dream completed, cursor advanced to {}", last_cursor)
                    return f"Dream completed in {elapsed:.1f}s."
                logger.info("Dream completed with no memory changes; cursor advanced to {}", last_cursor)
                return f"Dream completed in {elapsed:.1f}s; no memory changes."
            logger.warning(
                "Dream did not complete ({}); cursor remains at {}",
                f"stop_reason: {response.stop_reason}",
                self.memory.get_last_dream_cursor(),
            )
            return (
                f"Dream did not complete after {elapsed:.1f}s (stop_reason: {response.stop_reason}); "
                "memory cursor was not advanced."
            )
        except Exception as exc:
            logger.exception("Dream failed")
            return f"Dream failed after {time.monotonic() - start:.1f}s: {exc}"
        finally:
            self.memory.compact_history()

    def _get_session_lock(self, session_key: str) -> asyncio.Lock:
        return self._session_locks.setdefault(session_key, asyncio.Lock())

    # -- saving -----------------------------------------------------------------

    def _persist_user_message_early(
        self, content: str, session: Session, blocks: list[RuntimeContextBlock], **extra: Any
    ) -> bool:
        """Save the user's message, with its runtime context, before the run.

        The message is saved exactly as the model sees it (time included), with a
        marker recording the appended part. Replaying it unchanged keeps earlier
        turns byte-identical between requests, which helps provider caching.
        ``extra`` keys go on the saved message (``_command`` for commands).
        """
        if not (content.strip() or blocks):
            return False
        text, marker = append_runtime_context(content, blocks)
        if marker is not None:
            extra[RUNTIME_CONTEXT_HISTORY_META] = marker
        session.add_message("user", text, **extra)
        # Cleared once the turn is saved; still set next time means it crashed.
        session.metadata[PENDING_USER_TURN_KEY] = True
        self.sessions.save(session)
        return True

    @staticmethod
    def _persist_subagent_followup(session: Session, msg: InboundMessage) -> bool:
        """Save a subagent report as an assistant message, once per task.

        Returns False if the report is empty or this task's report is already
        saved (also when it was injected into a turn). (nanobot loop.py:2619)
        """
        if not msg.content:
            return False
        task_id = msg.metadata.get("subagent_task_id")
        if task_id and any(
            m.get("injected_event") == SUBAGENT_RESULT_EVENT and m.get("subagent_task_id") == task_id
            for m in session.messages
        ):
            return False
        session.add_message(
            "assistant",
            msg.content,
            sender_id=msg.sender_id,
            injected_event=SUBAGENT_RESULT_EVENT,
            subagent_task_id=task_id,
        )
        return True

    @staticmethod
    def _save_turn(session: Session, messages: list[dict[str, Any]], skip: int) -> None:
        """Append the turn's new messages to the session.

        Skips empty assistant messages (they confuse later turns) and tool
        results whose call is unknown or already answered (providers reject
        those). Every message gets a timestamp; an injected user message keeps
        the time it arrived (nanobot loop.py:2492).
        """
        declared = {
            str(call["id"])
            for m in session.messages
            if m.get("role") == "assistant"
            for call in m.get("tool_calls") or []
            if isinstance(call, dict) and call.get("id")
        }
        fulfilled = {
            str(m["tool_call_id"]) for m in session.messages if m.get("role") == "tool" and m.get("tool_call_id")
        }
        for message in messages[skip:]:
            entry = dict(message)
            meta = entry.pop("_meta", None)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue
            if role == "tool":
                call_id = str(entry.get("tool_call_id") or "")
                if not call_id or call_id not in declared or call_id in fulfilled:
                    logger.warning(
                        "Dropping invalid tool result {} from session {} during persistence",
                        call_id or "(missing id)",
                        session.key,
                    )
                    continue
                fulfilled.add(call_id)
            elif role == "user" and isinstance(meta, dict):
                marker = meta.get(RUNTIME_CONTEXT_MESSAGE_META)
                if isinstance(marker, dict):
                    entry[RUNTIME_CONTEXT_HISTORY_META] = marker
                timestamp = meta.get("timestamp")
                if isinstance(timestamp, str) and timestamp:
                    entry["timestamp"] = timestamp
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
            if role == "assistant":
                declared.update(
                    str(call["id"])
                    for call in entry.get("tool_calls") or []
                    if isinstance(call, dict) and call.get("id")
                )
        session.updated_at = datetime.now()
