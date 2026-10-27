"""Lifecycle hooks: let code watch or adjust an agent run without editing the runner.

Copied from nanobot/agent/hook.py. The runner calls an :class:`AgentHook` at
fixed points (before/after the run, each iteration, each tool call, while
streaming). :class:`CompositeHook` fans one call out to many hooks, so a
broken extra hook can't crash the run.

Left out versus nanobot: ``AgentRunHookContext.had_injections`` (NanoClaw
has no mid-turn message injection).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nanoclaw.events import NO_EVENTS, EventSink
from nanoclaw.providers import LLMResponse, LLMUsage, ToolCallRequest


@dataclass(slots=True)
class AgentHookContext:
    """Mutable per-iteration state the runner shares with hooks. (nanobot hook.py:18)

    ``usage`` is this iteration's request usage. ``tool_results`` are the raw
    tool outputs and ``tool_events`` one ``{"name", "status": "ok"|"error",
    "detail"}`` dict per call, both in ``tool_calls`` order (nanobot
    tools/execution.py). ``streamed_content`` / ``streamed_reasoning`` say the
    answer / reasoning already reached the user by streaming.
    ``stream_continues_current_message`` marks a segment that continues the
    previous one (nanobot sets it during length recovery; NanoClaw has none
    yet, so it stays False).
    """

    iteration: int
    messages: list[dict[str, Any]]
    response: LLMResponse | None = None
    usage: LLMUsage | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    tool_events: list[dict[str, str]] = field(default_factory=list)
    streamed_content: bool = False
    streamed_reasoning: bool = False
    stream_continues_current_message: bool = False
    final_content: str | None = None
    stop_reason: str | None = None
    error: str | None = None
    session_key: str | None = None


@dataclass(slots=True)
class AgentRunHookContext:
    """Run-level snapshot for before_run / after_run / on_error / on_finally.

    Fields after ``messages`` mirror AgentRunResult; ``exception`` is set when
    the run raised. (nanobot hook.py:38)
    """

    messages: list[dict[str, Any]]
    final_content: str | None = None
    tools_used: list[str] = field(default_factory=list)
    usage: LLMUsage | None = None
    stop_reason: str | None = None
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    exception: BaseException | None = None


@dataclass(slots=True)
class AgentTurnHookContext:
    """What a hook factory gets when building hooks for one turn. (nanobot hook.py:53)"""

    events: EventSink = NO_EVENTS
    workspace: Path | None = None
    channel: str = "cli"
    chat_id: str = "direct"
    message_id: str | None = None
    session_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    ephemeral: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)


_RUN_HOOK_NAMES = ("before_run", "after_run", "on_error", "on_finally")


class AgentHook:
    """Base hook: every callback is a no-op. Override the ones you need.

    ``reraise=True`` makes :class:`CompositeHook` let this hook's errors
    propagate instead of logging them (the progress hook uses it, so a failed
    output channel fails the run). (nanobot hook.py:67)
    """

    def __init__(self, reraise: bool = False) -> None:
        self._reraise = reraise

    def wants_streaming(self) -> bool:
        return False

    def has_run_hooks(self) -> bool:
        """Whether any run-level callback is overridden.

        The runner only builds the (copied) message snapshot for
        :class:`AgentRunHookContext` when this is True.
        """
        return any(
            getattr(getattr(self, name), "__func__", None) is not getattr(AgentHook, name)
            for name in _RUN_HOOK_NAMES
        )

    async def before_run(self, context: AgentRunHookContext) -> None:
        pass

    async def after_run(self, context: AgentRunHookContext) -> None:
        pass

    async def on_error(self, context: AgentRunHookContext) -> None:
        pass

    async def on_finally(self, context: AgentRunHookContext) -> None:
        pass

    async def before_iteration(self, context: AgentHookContext) -> None:
        pass

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        pass

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        pass

    async def on_provider_tool_event(
        self,
        context: AgentHookContext,
        event: dict[str, Any],
    ) -> None:
        """A provider-hosted tool (e.g. server-side web search) started or ended."""
        pass

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        pass

    async def before_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
    ) -> None:
        pass

    async def after_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
        result: Any,
    ) -> None:
        pass

    async def on_execute_tool_error(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
        error: Any,
    ) -> None:
        pass

    async def emit_reasoning(self, reasoning_content: str | None) -> None:
        pass

    async def emit_reasoning_end(self) -> None:
        """The current reasoning stream ended; hooks that buffer chunks flush here."""
        pass

    async def after_iteration(self, context: AgentHookContext) -> None:
        pass

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return content


AgentTurnHookFactory = Callable[[AgentTurnHookContext], AgentHook | None]


class CompositeHook(AgentHook):
    """Calls an ordered list of hooks. (nanobot hook.py:165)

    Async callbacks are isolated: a hook that raises is logged and the next
    one still runs (unless it set ``reraise``). ``finalize_content`` is a
    pipeline with no isolation, so its bugs surface.
    """

    __slots__ = ("_hooks",)

    def __init__(self, hooks: list[AgentHook]) -> None:
        super().__init__()
        self._hooks = list(hooks)

    def wants_streaming(self) -> bool:
        return any(h.wants_streaming() for h in self._hooks)

    def has_run_hooks(self) -> bool:
        if any(
            getattr(getattr(self, name), "__func__", None) is not getattr(CompositeHook, name)
            for name in _RUN_HOOK_NAMES
        ):
            return True
        return any(h.has_run_hooks() for h in self._hooks)

    async def _for_each_hook_safe(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        for h in self._hooks:
            # getattr: subclasses that skip super().__init__() have no _reraise.
            if getattr(h, "_reraise", False):
                await getattr(h, method_name)(*args, **kwargs)
                continue
            try:
                await getattr(h, method_name)(*args, **kwargs)
            except Exception:
                logger.opt(exception=True).error(
                    "AgentHook.{} error in {}", method_name, type(h).__name__,
                )

    async def before_iteration(self, context: AgentHookContext) -> None:
        await self._for_each_hook_safe("before_iteration", context)

    async def before_run(self, context: AgentRunHookContext) -> None:
        await self._for_each_hook_safe("before_run", context)

    async def after_run(self, context: AgentRunHookContext) -> None:
        await self._for_each_hook_safe("after_run", context)

    async def on_error(self, context: AgentRunHookContext) -> None:
        await self._for_each_hook_safe("on_error", context)

    async def on_finally(self, context: AgentRunHookContext) -> None:
        await self._for_each_hook_safe("on_finally", context)

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        await self._for_each_hook_safe("on_stream", context, delta)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        await self._for_each_hook_safe("on_stream_end", context, resuming=resuming)

    async def on_provider_tool_event(
        self,
        context: AgentHookContext,
        event: dict[str, Any],
    ) -> None:
        await self._for_each_hook_safe("on_provider_tool_event", context, event)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        await self._for_each_hook_safe("before_execute_tools", context)

    async def before_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
    ) -> None:
        await self._for_each_hook_safe("before_execute_tool", context, tool_call, tool, params)

    async def after_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
        result: Any,
    ) -> None:
        await self._for_each_hook_safe(
            "after_execute_tool", context, tool_call, tool, params, result,
        )

    async def on_execute_tool_error(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
        error: Any,
    ) -> None:
        await self._for_each_hook_safe(
            "on_execute_tool_error", context, tool_call, tool, params, error,
        )

    async def emit_reasoning(self, reasoning_content: str | None) -> None:
        await self._for_each_hook_safe("emit_reasoning", reasoning_content)

    async def emit_reasoning_end(self) -> None:
        await self._for_each_hook_safe("emit_reasoning_end")

    async def after_iteration(self, context: AgentHookContext) -> None:
        await self._for_each_hook_safe("after_iteration", context)

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        for h in self._hooks:
            content = h.finalize_content(context, content)
        return content


class SDKCaptureHook(AgentHook):
    """Records tool names, the final messages and usage of a run. (nanobot hook.py:292)

    ``after_iteration`` refreshes the snapshot each round (the runner mutates
    ``context.messages`` in place); ``after_run`` is authoritative.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tools_used: list[str] = []
        self.messages: list[dict[str, Any]] = []
        self.usage: LLMUsage | None = None
        self.stop_reason: str | None = None
        self.error: str | None = None
        self.tool_events: list[dict[str, str]] = []

    async def after_iteration(self, context: AgentHookContext) -> None:
        for call in context.tool_calls:
            self.tools_used.append(call.name)
        self.messages = list(context.messages)
        self.usage = context.usage
        self.stop_reason = context.stop_reason
        self.error = context.error
        self.tool_events = list(context.tool_events)

    async def after_run(self, context: AgentRunHookContext) -> None:
        self.tools_used = list(context.tools_used)
        self.messages = list(context.messages)
        self.usage = context.usage
        self.stop_reason = context.stop_reason
        self.error = context.error
        self.tool_events = list(context.tool_events)
