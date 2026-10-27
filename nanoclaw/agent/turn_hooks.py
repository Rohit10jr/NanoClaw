"""Build the hook chain for one agent turn. Copied from nanobot/agent/turn_hooks.py.

The progress hook always comes first, then registered hook factories,
registered hooks, per-turn factories and per-turn hooks, in that order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nanoclaw.agent.hook import (
    AgentHook,
    AgentTurnHookContext,
    AgentTurnHookFactory,
    CompositeHook,
)
from nanoclaw.agent.progress_hook import AgentProgressHook
from nanoclaw.events import NO_EVENTS, EventSink


@dataclass(slots=True)
class AgentTurnHookSpec:
    """Inputs for one turn's hook chain. (nanobot turn_hooks.py:22)

    ``registered_*`` are the loop's long-lived hooks; ``turn_*`` are passed
    for this turn only. Ephemeral turns get just the progress hook unless
    ``run_extra_hooks_for_ephemeral``.
    """

    events: EventSink = NO_EVENTS
    streaming: bool = False
    channel: str = "cli"
    chat_id: str = "direct"
    message_id: str | None = None
    metadata: dict[str, Any] | None = None
    session_key: str | None = None
    workspace: Path | None = None
    tool_hint_max_length: int = 40
    registered_hook_factories: list[AgentTurnHookFactory] = field(default_factory=list)
    turn_hook_factories: list[AgentTurnHookFactory] = field(default_factory=list)
    registered_hooks: list[AgentHook] = field(default_factory=list)
    turn_hooks: list[AgentHook] = field(default_factory=list)
    ephemeral: bool = False
    run_extra_hooks_for_ephemeral: bool = False
    attributes: dict[str, Any] | None = None
    log_content: bool = True


def _create_from_factories(
    factories: list[AgentTurnHookFactory],
    turn_context: AgentTurnHookContext,
    log_content: bool,
) -> list[AgentHook]:
    """Call each factory; one that raises is logged and skipped."""
    hooks: list[AgentHook] = []
    for factory in factories:
        try:
            created_hook = factory(turn_context)
        except Exception:
            logger.opt(exception=log_content).error(
                "Agent turn hook factory failed: {}",
                factory if log_content else type(factory).__name__,
            )
            continue
        if created_hook is not None:
            hooks.append(created_hook)
    return hooks


def build_agent_turn_hook(spec: AgentTurnHookSpec) -> AgentHook:
    """The hook ``AgentRunner`` uses for one turn. (nanobot turn_hooks.py:44)"""
    log_content = spec.log_content and not spec.ephemeral
    progress_hook = AgentProgressHook(
        events=spec.events,
        streaming=spec.streaming,
        session_key=spec.session_key,
        tool_hint_max_length=spec.tool_hint_max_length,
        log_content=log_content,
    )
    if spec.ephemeral and not spec.run_extra_hooks_for_ephemeral:
        return progress_hook

    turn_context = AgentTurnHookContext(
        events=spec.events,
        workspace=spec.workspace,
        channel=spec.channel,
        chat_id=spec.chat_id,
        message_id=spec.message_id,
        session_key=spec.session_key,
        metadata=dict(spec.metadata or {}),
        attributes=dict(spec.attributes or {}),
        ephemeral=spec.ephemeral,
    )
    hook_chain: list[AgentHook] = [progress_hook]
    hook_chain.extend(_create_from_factories(spec.registered_hook_factories, turn_context, log_content))
    hook_chain.extend(spec.registered_hooks)
    hook_chain.extend(_create_from_factories(spec.turn_hook_factories, turn_context, log_content))
    hook_chain.extend(spec.turn_hooks)
    return CompositeHook(hook_chain) if len(hook_chain) > 1 else progress_hook
