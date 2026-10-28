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
  cancels a session's worker and subagents and ends its exec sessions; the
  cancelled turn keeps its latest checkpoint (finished tool results) in the
  history.

Since layer 6 a turn also compacts the conversation when it no longer fits the
model's context window, and the loop runs nanobot's ``/compact``, ``/new``
(with archiving) and ``/dream`` commands. Since layer 15 that check runs before
every model request of the turn, inside the runner (context_governance.py);
the loop passes the summarizer and commits the summary when it saves the turn.

Since layer 8 the main agent can start subagents; since layer 19 with
nanobot's ``subagent`` tool (create / send / cancel / check session-owned
tasks, each with a private child session). A child created while no bus loop
runs (``process_direct``, e.g. ``nanoclaw ask``) or from a temporary turn
always runs inline, as in nanobot. A background subagent's report arrives on
the bus (since layer 14): injected into a running turn of its session, or run
as a follow-up turn (``process_subagent_result``): saved as an assistant
message and shown to the model as new input. A report whose child session no
longer belongs to the session is dropped (``accepts_result``).

Since layer 13 each turn builds nanobot's hook chain (``build_agent_turn_hook``)
over an :class:`EventSink`: progress, tool hints, reasoning, stream deltas and
ends, retry waits and context compaction all arrive there as events.

Since layer 20 the provider and model settings live in a
``ModelRuntimeResolver`` (``runtime_resolver``): ``provider``, ``model``,
``max_tokens``, ... read and set its default runtime, and each turn runs with
its session's runtime: the preset picked with ``/model`` (saved in the
session metadata), else the default (nanobot ``runtime_for_session``).

Since layer 21 the 1 s idle tick of ``run`` drives idle auto-compaction
(``_check_expired_sessions_if_due``, agent/autocompact.py): sessions idle for
``session_ttl_minutes`` are summarized in the background.

Left out of nanobot's loop: unified sessions, goals and turn continuations,
automation (cron) turns, WebUI recovery admission and follow-up journal,
config hot reload and the global concurrency gate.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import time
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger

from nanoclaw.agent.autocompact import AutoCompact
from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.context_governance import (
    CONTEXT_SAFETY_BUFFER,
    ContextWindowExceededError,
)
from nanoclaw.agent.hook import AgentHook
from nanoclaw.agent.memory import COMPACTION_NANOBOT, Consolidator, MemoryStore
from nanoclaw.agent.messages import EMPTY_FINAL_RESPONSE_MESSAGE, is_blank
from nanoclaw.agent.model_presets import PresetRuntimeLoader
from nanoclaw.agent.model_runtime import ModelRuntimeResolver
from nanoclaw.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec, ProgressCallback
from nanoclaw.agent.subagent import (
    DEFAULT_MAX_CONCURRENT_SUBAGENTS,
    SUBAGENT_RESULT_EVENT,
    SubagentManager,
    SubagentResult,
    is_subagent_report,
)
from nanoclaw.agent.tools import ToolRegistry, build_default_tools
from nanoclaw.agent.tools.exec_session import ExecSessionManager
from nanoclaw.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanoclaw.agent.tools.file_state import FileStateStore, bind_file_states, reset_file_states
from nanoclaw.agent.tools.subagent import SubagentTool
from nanoclaw.agent.turn_delivery import TurnDelivery, TurnDeliveryFactory
from nanoclaw.agent.turn_hooks import AgentTurnHookSpec, build_agent_turn_hook
from nanoclaw.bus.events import INBOUND_META_TURN_ID, InboundMessage, OutboundMessage
from nanoclaw.bus.outbound_events import StreamDeltaEvent, StreamedResponseEvent, StreamEndEvent
from nanoclaw.bus.queue import MessageBus
from nanoclaw.command import CommandContext, CommandRouter, command_text, register_builtin_commands
from nanoclaw.config.schema import ExecToolConfig, ModelPresetConfig
from nanoclaw.events import NO_EVENTS, AgentEvent, EventSink
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
from nanoclaw.security.workspace_access import (
    WorkspaceScope,
    bind_workspace_scope,
    default_workspace_scope,
    project_workspace_scope,
    reset_workspace_scope,
    workspace_access_block,
)
from nanoclaw.session import Session, SessionManager
from nanoclaw.session.manager import SESSION_CACHE_MAX_SIZE
from nanoclaw.session.model_selection import SESSION_MODEL_PRESET_METADATA_KEY, model_preset_from_metadata
from nanoclaw.session.recovery import (
    PENDING_USER_TURN_KEY,
    RUNTIME_CHECKPOINT_KEY,
    finish_turn,
    restore_pending_interruption,
    restore_runtime_checkpoint,
)
from nanoclaw.session.summary import (
    HIDDEN_HISTORY_META,
    SessionSummary,
    SessionSummaryCheckpoint,
    session_summary_from_metadata,
)
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.utils.progress_events import output_events

DREAM_NO_INPUT_MESSAGE = "\n".join([
    "Dream has no conversation history to process yet.",
    "",
    "Dream reads new entries from `memory/history.jsonl` after the current Dream cursor.",
    "Conversations only reach that file when they are compacted (`nanoclaw compact`, or",
    "automatically when they grow too long) or cleared with `nanoclaw ask --new`.",
])
# nanobot's default is 200,000; NanoClaw's default model, gpt-4o-mini, has 128,000.
DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
DEFAULT_SESSION_KEY = "cli:direct"
LAST_USAGE_KEY = "_last_usage"
# How long a turn about to end waits for its session's running subagents
# (nanobot loop.py:135).
_SUBAGENT_TERMINAL_WAIT_SECONDS = 300.0


# ContextWindowExceededError now comes from the runner; callers keep importing it here.
__all__ = ["AgentLoop", "ContextWindowExceededError", "TurnContext"]


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
    # The turn's project folder and access mode (nanobot ``RequestContext.workspace_scope``).
    workspace_scope: WorkspaceScope | None = None
    session: Session | None = None
    # The provider and model settings the turn runs with (nanobot
    # ``TurnContext.runtime``): the session's preset, else the loop default.
    runtime: LLMRuntime | None = None
    # The summary the turn starts from (nanobot ``TurnContext.pending_summary``).
    pending_summary: SessionSummary | None = None
    result: AgentRunResult | None = None
    outbound: OutboundMessage | None = None
    command_handled: bool = False
    streamed_content: bool = False
    started_at: float = field(default_factory=time.time)
    latency_ms: int | None = None
    # ``<session key>:<ns>``; a subagent task records it as its origin turn
    # (nanobot ``TurnContext.turn_id``, loop.py:1910).
    turn_id: str = ""


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
        exec_config: ExecToolConfig | None = None,
        project: Path | None = None,
        model_presets: Mapping[str, ModelPresetConfig] | None = None,
        model_preset: str | None = None,
        dream_model_preset: str | None = None,
        preset_runtime_loader: PresetRuntimeLoader | None = None,
        # Idle auto-compaction (nanobot loop.py:300, :318): 0 = off here, the
        # config default is 15 minutes / 60 s (agents.defaults).
        session_ttl_minutes: int = 0,
        idle_compact_check_interval_seconds: int = 0,
    ) -> None:
        # ``model_presets`` include "default" (nanobot ``configured_model_presets``);
        # without one, "default" is these constructor settings.
        # ``preset_runtime_loader`` builds a provider per preset from the
        # config; without it a preset reuses ``provider``.
        presets = dict(model_presets or {})
        presets.setdefault(
            "default",
            ModelPresetConfig(
                model=model or provider.default_model or "",
                max_tokens=max_tokens,
                context_window_tokens=context_window_tokens,
                temperature=temperature,
            ),
        )
        self.runtime_resolver = ModelRuntimeResolver(
            LLMRuntime(
                provider=provider,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                context_window_tokens=context_window_tokens,
                model_preset=model_preset,
            ),
            model_presets=presets,
            preset_runtime_loader=preset_runtime_loader,
        )
        # The preset Dream runs with (``agents.defaults.dream.modelOverride``).
        self.dream_model_preset = dream_model_preset
        # For the main agent's model calls only; subagents and memory archiving
        # retry in standard mode, as in nanobot (loop.py:367, :1346).
        self.provider_retry_mode = provider_retry_mode
        self.restrict_to_workspace = restrict_to_workspace
        self.workspace = workspace
        # The folder turns work on (``--project``; nanobot's WebUI scope) and
        # the agent workspace as a scope, for Dream (nanobot: every channel
        # but the WebUI). Raises WorkspaceScopeError for a missing folder.
        self.agent_workspace_scope = default_workspace_scope(workspace, restrict_to_workspace)
        self.workspace_scope = project_workspace_scope(
            project, workspace=workspace, restrict_to_workspace=restrict_to_workspace
        )
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.bus = bus if bus is not None else MessageBus()
        self.context = ContextBuilder(workspace)
        self.memory: MemoryStore = self.context.memory
        # ``config.tools.exec`` (nanobot ``self.exec_config``, loop.py:375).
        self.exec_config = exec_config if exec_config is not None else ExecToolConfig()
        # This loop's background shell sessions, shared by its exec tools
        # (nanobot loop.py:397): /stop and /new end the session's ones,
        # ``aclose`` ends all.
        self._exec_session_manager = ExecSessionManager()
        self.sessions = sessions if sessions is not None else SessionManager(workspace)
        # Children keep private sessions (``subagent:<id>``) in the same store
        # (nanobot loop.py:409, ``session_manager=self.sessions``).
        self.subagents = SubagentManager(
            workspace,
            max_tool_result_chars=max_tool_result_chars,
            max_iterations=max_iterations,
            max_concurrent_subagents=max_concurrent_subagents,
            restrict_to_workspace=restrict_to_workspace,
            bus=self.bus,
            exec_config=self.exec_config,
            session_manager=self.sessions,
        )
        if tools is None:
            # The main agent's tools: the standard set plus ``subagent``.
            # Subagents and Dream build their own sets without it.
            tools = build_default_tools(
                workspace,
                restrict_to_workspace=restrict_to_workspace,
                exec_session_manager=self._exec_session_manager,
                exec_config=self.exec_config,
            )
            tools.register(SubagentTool(self.subagents))
            logger.info("Registered {} tools: {}", len(tools.tool_names), tools.tool_names)
        self.tools = tools
        # One file-read/write tracker per session (nanobot loop.py:389). The
        # tool registry is shared by this loop, so tools resolve the active
        # state via contextvars; ``read_file`` uses it to answer a repeat read
        # with "[File unchanged since last read: ...]".
        self._file_state_store = FileStateStore(max_sessions=SESSION_CACHE_MAX_SIZE)
        self.consolidator = Consolidator(
            self.memory,
            self.sessions,
            partial(self.context.build_messages, workspace=self.workspace_scope.project_path),
            self.tools.get_definitions,
            compaction=compaction,
        )
        self.subagents.consolidator = self.consolidator
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
        # nanobot loop.py:455-463.
        self.auto_compact = AutoCompact(
            sessions=self.sessions,
            consolidator=self.consolidator,
            session_ttl_minutes=session_ttl_minutes,
            bind_events=self._idle_events,
            session_lock=self._get_session_lock,
        )
        self._idle_compact_check_interval_s = idle_compact_check_interval_seconds
        self._next_idle_compact_check_at = time.monotonic()

    # -- model selection (nanobot loop.py:230-290, 560-640) ------------------

    @property
    def runtime(self) -> LLMRuntime:
        """The default runtime: turns of sessions without a ``/model`` preset."""
        return self.runtime_resolver.runtime

    @property
    def provider(self) -> LLMProvider:
        return self.runtime_resolver.runtime.provider

    @provider.setter
    def provider(self, value: LLMProvider) -> None:
        self.runtime_resolver.update_default(provider=value)

    @property
    def model(self) -> str | None:
        """Model selected for future turns (None: the provider's default model)."""
        return self.runtime_resolver.runtime.model

    @model.setter
    def model(self, value: str | None) -> None:
        self.runtime_resolver.update_default(model=value)

    @property
    def max_tokens(self) -> int:
        return self.runtime_resolver.runtime.max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self.runtime_resolver.update_default(max_tokens=value)

    @property
    def temperature(self) -> float:
        return self.runtime_resolver.runtime.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.runtime_resolver.update_default(temperature=value)

    @property
    def context_window_tokens(self) -> int:
        """Context limit selected for future turns."""
        return self.runtime_resolver.runtime.context_window_tokens

    @context_window_tokens.setter
    def context_window_tokens(self, value: int) -> None:
        self.runtime_resolver.update_default(context_window_tokens=value)

    @property
    def model_presets(self) -> Mapping[str, ModelPresetConfig]:
        """Configured model presets exposed for selection and display."""
        return self.runtime_resolver.model_presets

    @property
    def model_preset(self) -> str | None:
        return self.runtime_resolver.model_preset

    def set_model_preset(self, name: str | None) -> LLMRuntime:
        """Select a named default runtime for future turns (nanobot loop.py:621)."""
        old_model = self.model
        runtime = self.runtime_resolver.select_preset(name)
        logger.info("Runtime model switched for next turn: {} -> {}", old_model, runtime.model)
        return runtime

    def set_runtime_model(self, model: str) -> LLMRuntime:
        """Select a model on the current provider for future turns (nanobot loop.py:636)."""
        return self.runtime_resolver.select_model(model)

    def dream_runtime(self) -> LLMRuntime | None:
        """The preset Dream runs with, without changing defaults (nanobot loop.py:271)."""
        if not self.dream_model_preset:
            return None
        return self.runtime_resolver.resolve_preset(self.dream_model_preset)

    def _resolve_session_runtime(self, session: Session, *, recover_removed: bool) -> LLMRuntime | None:
        """The session's runtime, or None after dropping a removed preset (nanobot loop.py:560)."""
        name = model_preset_from_metadata(session.metadata)
        if name is None:
            return self.runtime
        try:
            return self.runtime_resolver.resolve_preset(name)
        except KeyError:
            if not recover_removed or name in self.runtime_resolver.model_presets:
                raise
            logger.warning(
                "Session '{}' references removed model preset '{}'; falling back to default", session.key, name
            )
            session.metadata.pop(SESSION_MODEL_PRESET_METADATA_KEY, None)
            return None

    async def runtime_for_session(self, session: Session, *, recover_removed: bool = True) -> LLMRuntime:
        """The runtime a session's turns use (nanobot loop.py:582).

        Raises ``KeyError`` for a removed preset when ``recover_removed`` is
        False, and ``ValueError`` when the preset's provider is misconfigured.
        """
        runtime = self._resolve_session_runtime(session, recover_removed=recover_removed)
        if runtime is not None:
            return runtime
        # Only saved sessions carry a preset, so this never saves a temporary one.
        self.sessions.save(session)
        return self.runtime

    async def set_session_model_preset(self, session_key: str, name: str, *, session: Session | None = None) -> LLMRuntime:
        """Validate and save one session's preset (nanobot loop.py:598)."""
        runtime = self.runtime_resolver.resolve_preset(name)
        session = session or self.sessions.get_or_create(session_key)
        session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = runtime.model_preset
        self.sessions.save(session)
        return runtime

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
        workspace_scope: WorkspaceScope | None = None,
        runtime: LLMRuntime | None = None,
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
        ``workspace_scope`` replaces the loop's project for this turn (Dream
        uses the agent workspace). ``runtime`` replaces the session's model
        settings for this turn (Dream's preset).
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
                workspace_scope=workspace_scope,
                runtime=runtime,
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
        if ctx.result is None:  # dropped: the child no longer belongs to the session
            return AgentRunResult(final_content=None, messages=[])
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
        ``/stop``. The 1 s timeout is nanobot's idle tick: it drives idle
        auto-compaction (``_check_expired_sessions_if_due``).
        """
        self._running = True
        try:
            logger.info("Agent loop started")
            # Tasks a previous process left running can't finish any more
            # (nanobot runs this when the gateway starts, gateway_runtime.py:1029).
            await self.subagents.recover_interrupted()
            while self._running:
                try:
                    async with asyncio.timeout(1.0):
                        msg = await self.bus.consume_inbound()
                except TimeoutError:
                    await self._check_expired_sessions_if_due()
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

    async def _check_expired_sessions_if_due(self) -> None:
        """Scan idle sessions without blocking the event loop. (nanobot loop.py:1410)

        At most once per ``idle_compact_check_interval_seconds``; sessions
        with a busy worker are left alone (D13).
        """
        now = time.monotonic()
        if now < self._next_idle_compact_check_at:
            return
        self._next_idle_compact_check_at = now + self._idle_compact_check_interval_s
        await self.auto_compact.check_expired(
            self.schedule_background,
            self.runtime_for_session,
            active_session_keys=self._pending_queues.keys(),
        )

    def _idle_events(self, session_key: str) -> EventSink:
        """Bind one idle compaction to its session's chat. (nanobot loop.py:1017)"""
        return self.turn_delivery_factory.session_events(session_key)

    def schedule_background(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Run a coroutine as a tracked background task, cancelled by ``aclose``.
        (nanobot loop.py:1844)"""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

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
        if not self.subagents.accepts_result(msg, key):
            return
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
        """Cancel and await the session's worker, subagents and exec sessions;
        returns how many.

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
        sub_cancelled = await self.subagents.cancel_by_session(key)
        exec_cancelled = await self._exec_session_manager.terminate_by_owner(key)
        return cancelled + sub_cancelled + exec_cancelled

    def _has_pending_work(self) -> bool:
        """Whether the loop still owns work. (nanobot loop.py:1526)

        nanobot reads it only to pause intake for an update
        (``maintenance``, used by update_stop.py); NanoClaw has no updater,
        so nothing reads it outside tests. A finished exec session whose
        output was not read yet counts as work, as in nanobot.
        """
        return bool(
            self.bus.inbound_size
            or self._pending_queues
            or self.bus.has_pending_events
            or self._active_tasks
            or self._background_tasks
            or (self._close_lock is not None and self._close_lock.locked())
            or self.subagents.has_pending_work
            or self._exec_session_manager.has_pending_work
        )

    async def aclose(self) -> None:
        """Cancel session workers, background work and subagents, then end
        every exec session. (nanobot loop.py:1771)"""
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
                # Each step runs even when one before it fails (nanobot loop.py:1817).
                errors: list[BaseException] = []
                for cleanup in (
                    self.subagents.close,
                    self._exec_session_manager.close_all,
                    # Every provider client handed out (nanobot ``_close_runtime``).
                    self.runtime_resolver.aclose,
                ):
                    try:
                        await cleanup()
                    except BaseException as exc:
                        errors.append(exc)
                if len(errors) == 1:
                    raise errors[0]
                if errors:
                    raise BaseExceptionGroup("failed to close agent resources", errors)

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
        workspace_scope: WorkspaceScope | None = None,
        runtime: LLMRuntime | None = None,
    ) -> TurnContext:
        """One message through the turn stages; the caller holds the session lock.
        (nanobot loop.py:1860)"""
        kind = "user" if msg.is_user_input else "system"
        if kind == "system":
            key = session_key or self._system_session_key(msg)
        else:
            key = session_key or msg.session_key
        if not self.subagents.accepts_result(msg, key):
            # The report's child session no longer belongs to this session
            # (nanobot loop.py:1888). No turn runs.
            return TurnContext(msg=msg, session_key=key, kind=kind, events=events, streaming=streaming)
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
            workspace_scope=workspace_scope or self.workspace_scope,
            runtime=runtime,
            turn_id=f"{key}:{time.time_ns()}",
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
        await self._compact_session(ctx)
        if ctx.kind == "user" and await self._dispatch_command(ctx):
            return ctx
        if ctx.runtime is None:
            # Resolved after commands, so /model can still repair a session
            # whose preset no longer works.
            assert ctx.session is not None
            ctx.runtime = await self.runtime_for_session(ctx.session)
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
        # A turn that died mid-way (crash, kill, Ctrl-C): its checkpoint first,
        # then an error reply for a user message left unanswered, saved before
        # this turn runs or a command answers (nanobot loop.py:2134-2141).
        changed = restore_runtime_checkpoint(ctx.session)
        changed = restore_pending_interruption(ctx.session) or changed
        if changed:
            self.sessions.save(ctx.session)

    async def _compact_session(self, ctx: TurnContext) -> None:
        """Pick up the summary an idle compaction left. (nanobot loop.py:2143)

        Runs before commands, as in nanobot, so ``/new`` also drops a summary
        kept in memory.
        """
        if ctx.ephemeral:
            return
        assert ctx.session is not None
        ctx.session, ctx.pending_summary = await self.auto_compact.prepare_session(ctx.session, ctx.session_key)

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

    def _runtime_blocks(
        self, content: str, session: Session, *, ephemeral: bool, scope: WorkspaceScope | None = None
    ) -> list[RuntimeContextBlock]:
        """The runtime context for user input: the time, the project and
        file-access mode (every turn, nanobot loop.py:662), the iteration-limit
        notice (user turns only, nanobot loop.py:663) and $skill-name skills
        (nanobot ``_resolve_runtime_context_for_request``, loop.py:829)."""
        blocks: list[RuntimeContextBlock] = [
            current_time_block(),
            workspace_access_block(scope or self.workspace_scope, sandbox=self.exec_config.sandbox),
        ]
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

        # Build: history first, then save the user's message before running,
        # so it survives a crash or an error mid-turn.
        # A summary an idle compaction just wrote, else the saved one
        # (``_compact_session``). A temporary turn reads its own session's.
        if ephemeral:
            summary = session_summary_from_metadata(session.metadata, fallback_last_active=session.updated_at)
        else:
            summary = ctx.pending_summary
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
            blocks = self._runtime_blocks(content, session, ephemeral=ephemeral, scope=ctx.workspace_scope)
            persisted_early = not ephemeral and self._persist_user_message_early(content, session, blocks)
        # [system, *history, new message], the new message never merged into
        # the history (nanobot ``ContextBuilder.build_transcript``): system
        # prompt + history are what the model has already accepted, so they
        # are what compaction summarizes; the model copy merges adjacent users.
        messages = [
            *self.context.build_messages(
                history, None, channel=msg.channel, session_summary=summary, workspace=self._project(ctx)
            ),
            self.context.build_current_message(content, runtime_context_blocks=blocks),
        ]

        if ctx.delivery is not None:
            await ctx.delivery.running(started_at=time.time())
        result = await self._run_agent_loop(
            ctx, messages, tools, accepted_boundary=1 + len(history), active_summary=summary["text"] if summary else None
        )
        if ctx.kind == "user" and is_blank(result.final_content):
            result = dataclasses.replace(result, final_content=EMPTY_FINAL_RESPONSE_MESSAGE)
        ctx.result = result
        ctx.latency_ms = max(0, int((time.time() - ctx.started_at) * 1000))
        if ctx.delivery is not None:
            ctx.delivery.record_latency(ctx.latency_ms)

        # Persist everything the runner added after the user's message, with
        # the summary marker where the compacted history ends (if compacted).
        # Temporary turns never save a summary (nanobot loop.py:2393).
        if not ephemeral:
            skip = len(messages) if persisted_early else len(messages) - 1
            # Token use of the last turn, for /status (nanobot ``_last_usage``).
            if result.usage is not None:
                session.metadata[LAST_USAGE_KEY] = result.usage.to_dict()  # nanobot loop.py:2384
            # Only user turns set or clear the iteration-limit flag; a
            # subagent follow-up leaves it alone (nanobot loop.py:2385).
            if ctx.kind == "user":
                record_iteration_limit(session.metadata, result.stop_reason)
            self._save_turn(
                session, result.messages, skip, summary_checkpoint=result.summary_checkpoint, input_boundary=boundary
            )
            finish_turn(session)
            self.sessions.save(session)

    async def _run_agent_loop(
        self,
        ctx: TurnContext,
        messages: list[dict[str, Any]],
        tools: ToolRegistry,
        *,
        accepted_boundary: int | None = None,
        active_summary: str | None = None,
    ) -> AgentRunResult:
        """Run the runner with the turn's hooks, checkpoints, injections and
        context governance. (nanobot loop.py:1051)

        ``accepted_boundary``: the system prompt and history at the start of
        ``messages``, which compaction may replace with a summary.
        """
        session = ctx.session
        assert session is not None
        ephemeral, pending_queue, events = ctx.ephemeral, ctx.pending_queue, ctx.events
        scope = ctx.workspace_scope or self.workspace_scope
        runtime = ctx.runtime or self.runtime

        async def _checkpoint(payload: dict[str, Any]) -> None:
            # The latest finished step, saved to disk so a cancelled turn, or
            # the next turn after a crash, can restore it (nanobot
            # ``_set_runtime_checkpoint``, loop.py:2648). ``finish_turn``
            # clears it once the turn is saved (``_clear_runtime_checkpoint``, :2663).
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
                    converted.append(
                        self._injected_user_message(pending_msg, session, ephemeral=ephemeral, scope=scope)
                    )
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

        # Tools such as subagent read the turn's session, chat and model
        # settings (nanobot ``_request_context_for_turn``, loop.py:777).
        token = bind_request_context(
            RequestContext(
                channel=ctx.msg.channel,
                session_key=session.key,
                runtime=runtime,
                chat_id=ctx.msg.chat_id,
                message_id=ctx.msg.metadata.get("message_id"),
                # A transport's turn id wins (nanobot loop.py:781, ``route.turn_id``).
                turn_id=ctx.msg.metadata.get(INBOUND_META_TURN_ID) or ctx.turn_id,
                log_content=not ephemeral,
                persist_session=not ephemeral,
                can_receive_background_results=self._running,
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
                workspace=scope.project_path,
                turn_hooks=ctx.hooks,
                ephemeral=ephemeral,
            )
        )
        file_state_token = bind_file_states(self._file_state_store.for_session(session.key))
        # Tools resolve paths against the turn's project (nanobot loop.py:1293).
        workspace_token = bind_workspace_scope(scope)
        try:
            result = await self.runner.run(
                AgentRunSpec(
                    initial_messages=messages,
                    tools=tools,
                    provider=runtime.provider,
                    model=runtime.model,
                    max_iterations=self.max_iterations,
                    max_tool_result_chars=self.max_tool_result_chars,
                    # Temporary turns never write spill files. Others write them
                    # in the project, as nanobot does (loop.py:1344).
                    workspace=None if ephemeral else scope.project_path,
                    max_tokens=runtime.max_tokens,
                    temperature=runtime.temperature,
                    session_key=session.key,
                    provider_retry_mode=self.provider_retry_mode,
                    hook=hook,
                    events=events,
                    checkpoint_callback=_checkpoint,
                    injection_callback=_drain_pending,
                    terminal_injection_callback=_wait_for_pending,
                    # Checked before every model request (nanobot loop.py:1335-1361).
                    # Temporary turns are checked too, but never journal.
                    context_window_tokens=runtime.context_window_tokens,
                    consolidate_history=partial(
                        self.consolidator.summarize_transcript,
                        runtime=runtime,
                        session_key=session.key,
                        tools=tools.get_definitions(),
                        persist=not ephemeral,
                    ),
                    accepted_boundary=accepted_boundary,
                    active_summary=active_summary,
                    summary_transcript_builder=partial(
                        self._summary_transcript, ctx.msg.channel, workspace=scope.project_path
                    ),
                )
            )
        finally:
            reset_workspace_scope(workspace_token)
            reset_request_context(token)
            reset_file_states(file_state_token)
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

    def _injected_user_message(
        self, msg: InboundMessage, session: Session, *, ephemeral: bool, scope: WorkspaceScope | None = None
    ) -> dict[str, Any]:
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
            blocks = self._runtime_blocks(msg.content, session, ephemeral=ephemeral, scope=scope)
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
        """Tokens a request may use: window minus the reply and a safety margin.
        (nanobot ``ContextGovernor.input_budget``)"""
        if not self.context_window_tokens:
            return 0
        budget = self.context_window_tokens - self.max_tokens - CONTEXT_SAFETY_BUFFER
        return budget if budget > 0 else 0

    def _project(self, ctx: TurnContext) -> Path:
        """The folder the turn works on."""
        return (ctx.workspace_scope or self.workspace_scope).project_path

    def _summary_transcript(
        self, channel: str | None, summary: str, *, workspace: Path | None = None
    ) -> list[dict[str, Any]]:
        """The system prompt around a new summary, with no history or message.
        (nanobot ``ContextCompactionState.from_transcript``, context_governance.py:162)"""
        return self.context.build_messages(
            [],
            None,
            channel=channel,
            session_summary={"text": summary, "last_active": datetime.now().astimezone().isoformat()},
            workspace=workspace or self.workspace_scope.project_path,
        )

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

    def discard_session_file_state(self, key: str) -> None:
        """Forget ephemeral file-read state for a reset session (nanobot loop.py:997)."""
        self._file_state_store.discard(key)

    async def _new_session(self, session_key: str) -> None:
        # nanobot cmd_new (builtin.py:319): the new conversation has read nothing.
        self.discard_session_file_state(session_key)
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
                runtime = await self.runtime_for_session(session)
                await self.consolidator.archive_session(archived, archive_end=len(snapshot), runtime=runtime)
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
        runtime = await self.runtime_for_session(self.sessions.get_or_create(session_key))
        return await self.consolidator.compact_idle_session(
            session_key, runtime=runtime, events=events, notify=True, reason="manual"
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
                prompt,
                session_key=MemoryStore.dream_session_key(),
                ephemeral=True,
                tools=self.memory.build_dream_tools(),
                # Dream edits the agent's own files, never the project.
                workspace_scope=self.agent_workspace_scope,
                runtime=self.dream_runtime(),
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
    def _validated_checkpoint_boundary(
        checkpoint: SessionSummaryCheckpoint | None, *, skip: int, message_count: int, session_key: str
    ) -> int | None:
        """The checkpoint's boundary, if it lies in this turn. (nanobot loop.py:2470)"""
        if checkpoint is None:
            return None
        boundary = checkpoint.transcript_boundary
        if skip - 1 <= boundary <= message_count:
            return boundary
        logger.warning(
            "Ignoring invalid summary boundary {} outside [{}, {}] for {}", boundary, skip - 1, message_count, session_key
        )
        return None

    @staticmethod
    def _save_turn(
        session: Session,
        messages: list[dict[str, Any]],
        skip: int,
        *,
        summary_checkpoint: SessionSummaryCheckpoint | None = None,
        input_boundary: int | None = None,
    ) -> None:
        """Append the turn's new messages to the session.

        Skips empty assistant messages (they confuse later turns) and tool
        results whose call is unknown or already answered (providers reject
        those). Every message gets a timestamp; an injected user message keeps
        the time it arrived (nanobot loop.py:2492).

        A ``summary_checkpoint`` (the turn compacted) is committed where its
        transcript boundary falls: before the turn's input, or between two of
        the turn's messages (nanobot loop.py:2526-2613). ``input_boundary`` is
        the session length before the input was saved early; nanobot assumes
        the last saved row is the input, which is wrong for a repeated
        subagent report (docs/UPSTREAM_BUGS.md, bug 5).
        """
        checkpoint_boundary = AgentLoop._validated_checkpoint_boundary(
            summary_checkpoint, skip=skip, message_count=len(messages), session_key=session.key
        )
        if summary_checkpoint is not None and checkpoint_boundary == skip - 1:
            insert_at = len(session.messages) if input_boundary is None else input_boundary
            session.commit_summary_checkpoint(summary_checkpoint.summary, insert_at=insert_at)
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
        for message_index, message in enumerate(messages[skip:], start=skip):
            # Before the message is filtered, so dropping one can't move the boundary.
            if summary_checkpoint is not None and checkpoint_boundary == message_index:
                session.commit_summary_checkpoint(summary_checkpoint.summary)
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
        if summary_checkpoint is not None and checkpoint_boundary == len(messages):
            session.commit_summary_checkpoint(summary_checkpoint.summary)
        session.updated_at = datetime.now()
