"""Built-in slash commands: their specs and handlers.

Mirrors nanobot/command/builtin.py for the commands NanoClaw has (/new,
/compact, /stop, /status, /model, /history, /dream, /skill, /help), with
nanobot's spec fields and texts. Handlers return an :class:`OutboundMessage` instead of
printing, so any channel can show them. nanobot's other commands (/restart,
/goal, /trigger, /dream-log, /dream-restore, /dream-prompt,
/evaluator-prompt, /pairing, the ``!`` shell) need features NanoClaw does not
have yet.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nanoclaw import __version__
from nanoclaw.bus.events import META_RENDER_AS, OutboundMessage
from nanoclaw.command.router import CommandContext, CommandRouter, normalize_command_text
from nanoclaw.session.summary import is_hidden_history_message

if TYPE_CHECKING:
    from nanoclaw.agent.loop import AgentLoop
    from nanoclaw.session import Session

# How a slash command takes part in turn state (nanobot builtin.py:31-43):
# - side_channel: control text, no agent turn started or ended.
# - finalize_active_turn: side channel that also closes the active UI turn.
# - stop_active_turn: cancels the active turn.
# - agent_turn: always goes down the normal agent path.
# - agent_turn_with_args: without args a side channel; with args an agent turn.
CommandLifecycle = Literal[
    "side_channel",
    "finalize_active_turn",
    "stop_active_turn",
    "agent_turn",
    "agent_turn_with_args",
]

# nanobot writes "🐈 nanobot" in its help and status texts; NanoClaw's
# terminal name is plain "nanoclaw" (cli/stream.py BOT_NAME).
BOT_LABEL = "nanoclaw"

_HISTORY_DEFAULT_COUNT = 10
_HISTORY_MAX_COUNT = 50
_HISTORY_MAX_CONTENT_CHARS = 200
_CONTEXT_SAFETY_BUFFER = 1024


@dataclass(frozen=True)
class BuiltinCommandSpec:
    """(nanobot builtin.py:48)"""

    command: str
    title: str
    description: str
    icon: str
    arg_hint: str = ""
    lifecycle: CommandLifecycle = "side_channel"
    accepts_args: bool = False

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "command": self.command,
            "title": self.title,
            "description": self.description,
            "icon": self.icon,
            "arg_hint": self.arg_hint,
            "lifecycle": self.lifecycle,
            "accepts_args": self.accepts_args,
        }


# nanobot builtin.py:70, the commands NanoClaw has, in nanobot's order.
BUILTIN_COMMAND_SPECS: tuple[BuiltinCommandSpec, ...] = (
    BuiltinCommandSpec(
        "/new",
        "New chat",
        "Reset this chat and start a fresh conversation.",
        "square-pen",
        lifecycle="finalize_active_turn",
    ),
    BuiltinCommandSpec(
        "/compact",
        "Compact context",
        "Compact this chat's context and continue the conversation.",
        "archive",
    ),
    BuiltinCommandSpec(
        "/stop",
        "Stop current task",
        "Cancel the active agent turn for this chat.",
        "square",
        lifecycle="stop_active_turn",
    ),
    BuiltinCommandSpec(
        "/status",
        "Show status",
        "Display runtime, provider, and channel status.",
        "activity",
    ),
    BuiltinCommandSpec(
        "/model",
        "Switch model preset",
        "Show or switch the active model preset.",
        "brain",
        "[preset]",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/history",
        "Show conversation history",
        "Print the last N persisted conversation messages.",
        "history",
        "[n]",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/dream",
        "Run Dream",
        "Manually trigger memory consolidation.",
        "sparkles",
    ),
    BuiltinCommandSpec(
        "/skill",
        "List skills",
        "List all enabled skills available to the agent.",
        "wrench",
    ),
    BuiltinCommandSpec(
        "/help",
        "Show help",
        "List available slash commands.",
        "circle-help",
    ),
)


def builtin_command_palette() -> list[dict[str, str | bool]]:
    """Command metadata for UI command palettes. (nanobot builtin.py:195)"""
    return [spec.as_dict() for spec in BUILTIN_COMMAND_SPECS]


def builtin_command_starts_agent_turn(text: str) -> bool:
    """Whether *text* goes through a normal agent turn. (nanobot builtin.py:200)"""
    command, separator, args = normalize_command_text(text).partition(" ")
    spec = next((item for item in BUILTIN_COMMAND_SPECS if item.command == command.lower()), None)
    if spec is None or (separator and not spec.accepts_args):
        return True
    if spec.lifecycle == "agent_turn":
        return True
    return spec.lifecycle == "agent_turn_with_args" and bool(args.strip())


def _reply(ctx: CommandContext, content: str, *, as_text: bool = False) -> OutboundMessage:
    """A reply to the command's chat, keeping its routing metadata."""
    metadata = dict(ctx.msg.metadata or {})
    if as_text:
        metadata[META_RENDER_AS] = "text"
    return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content, metadata=metadata)


def _session(ctx: CommandContext) -> Session:
    return ctx.session or ctx.loop.sessions.get_or_create(ctx.key)


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel the session's active turn and subagents. (nanobot builtin.py:215)

    ``cancel_active_tasks`` is nanobot's ``loop._cancel_active_tasks``.
    """
    total = await ctx.loop.cancel_active_tasks(ctx.key)
    # Also drop messages waiting to be injected into the turn, so a stopped
    # turn cannot deadlock on them (nanobot builtin.py:221).
    pending = ctx.loop._pending_queues.pop(ctx.key, None)
    if pending is not None:
        while not pending.empty():
            try:
                pending.get_nowait()
                total += 1
            except Exception:
                break
    return _reply(ctx, f"Stopped {total} task(s)." if total else "No active task to stop.")


def build_status_content(
    *,
    model: str,
    start_time: float,
    last_usage: dict[str, Any] | None,
    context_window_tokens: int,
    session_msg_count: int,
    context_tokens_estimate: int,
    active_task_count: int,
    max_completion_tokens: int,
) -> str:
    """Runtime status for /status. (nanobot ``build_status_content``, utils/helpers.py:900)

    nanobot also shows web-search usage; NanoClaw has no usage endpoint.
    """
    uptime_s = int(time.time() - start_time)
    uptime = (
        f"{uptime_s // 3600}h {(uptime_s % 3600) // 60}m" if uptime_s >= 3600 else f"{uptime_s // 60}m {uptime_s % 60}s"
    )
    last_in = int((last_usage or {}).get("input_tokens") or 0)
    last_out = int((last_usage or {}).get("output_tokens") or 0)
    cached = int((last_usage or {}).get("cache_read_tokens") or 0)
    token_line = f"📊 Tokens: {last_in} in / {last_out} out"
    if cached and last_in:
        token_line += f" ({cached * 100 // last_in}% cached)"
    ctx_total = max(context_window_tokens, 0)
    ctx_budget = max(ctx_total - int(max_completion_tokens) - _CONTEXT_SAFETY_BUFFER, 1)
    ctx_pct = min(int((context_tokens_estimate / ctx_budget) * 100), 999) if ctx_budget > 0 else 0
    ctx_used_str = f"{context_tokens_estimate // 1000}k" if context_tokens_estimate >= 1000 else str(context_tokens_estimate)
    ctx_total_str = f"{ctx_total // 1000}k" if ctx_total > 0 else "n/a"
    return "\n".join(
        [
            f"{BOT_LABEL} v{__version__}",
            f"🧠 Model: {model}",
            token_line,
            f"📚 Context: {ctx_used_str}/{ctx_total_str} ({ctx_pct}% of input budget)",
            f"💬 Session: {session_msg_count} messages",
            f"⏱ Uptime: {uptime}",
            f"⚡ Tasks: {active_task_count} active",
        ]
    )


def _active_task_count(loop: AgentLoop, key: str) -> int:
    """The session's running worker plus its subagents (nanobot builtin.py:295-298)."""
    count = sum(1 for task in loop._active_tasks.get(key, ()) if not task.done())
    with suppress(Exception):
        count += loop.subagents.get_running_count_by_session(key)
    return count


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Runtime status for the session. (nanobot builtin.py:268)"""
    # Imported here: the agent loop will import this module (session workers).
    from nanoclaw.agent.loop import LAST_USAGE_KEY

    loop = ctx.loop
    session = _session(ctx)
    runtime = ctx.runtime or await loop.runtime_for_session(session)
    estimate = 0
    with suppress(Exception):
        estimate, _ = loop.consolidator.estimate_session_prompt_tokens(session, runtime=runtime)
    last_usage = session.metadata.get(LAST_USAGE_KEY)
    if not isinstance(last_usage, dict):
        last_usage = None
    if estimate <= 0 and last_usage is not None:
        estimate = int(last_usage.get("input_tokens") or 0)
    content = build_status_content(
        model=runtime.model or runtime.provider.default_model,
        start_time=loop.start_time,
        last_usage=last_usage,
        context_window_tokens=runtime.context_window_tokens,
        session_msg_count=len(session.get_history()),
        context_tokens_estimate=estimate,
        active_task_count=_active_task_count(loop, ctx.key),
        max_completion_tokens=runtime.max_tokens,
    )
    return _reply(ctx, content, as_text=True)


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Stop the session's work and start a fresh conversation. (nanobot builtin.py:315)

    nanobot archives the old messages in the background; NanoClaw's
    ``new_session`` waits for the archive (agent/loop.py). It takes the
    session lock, unless the command runs on the session's worker, which
    holds it already (``ctx.session`` is set there).
    """
    await ctx.loop.cancel_active_tasks(ctx.key)
    await ctx.loop.new_session(ctx.key, lock_held=ctx.session is not None)
    return _reply(ctx, "New session started.")


async def cmd_compact(ctx: CommandContext) -> None:
    """Compact the session now, keeping the conversation. (nanobot builtin.py:351)

    The compaction events (started / succeeded / failed) go to ``ctx.events``;
    there is no reply. On the session's worker (``ctx.session`` set) the
    session lock is already held.
    """
    try:
        await ctx.loop.compact_session(ctx.key, events=ctx.events, lock_held=ctx.session is not None)
    except Exception:
        logger.exception("Manual context compaction failed for {}", ctx.key)
    return None


async def _publish_followup(ctx: CommandContext, msg: OutboundMessage) -> None:
    """Send a reply that comes after the command's own (nanobot: ``loop.bus.publish_outbound``)."""
    bus = ctx.bus or getattr(ctx.loop, "bus", None)
    if bus is None:
        logger.warning("No bus for the follow-up to {}; dropped: {}", ctx.raw, msg.content)
        return
    await bus.publish_outbound(msg)


async def cmd_dream(ctx: CommandContext) -> OutboundMessage:
    """Start a Dream run in the background; its result follows on the bus.
    (nanobot builtin.py:452)

    The run is added to ``ctx.tasks`` so a caller can wait for it. nanobot's
    git commit of the memory change is not here (no git store yet).
    """
    from nanoclaw.agent.loop import DREAM_NO_INPUT_MESSAGE  # see cmd_status

    async def _run_dream() -> None:
        content = await ctx.loop.run_dream()
        # nanobot sends the nothing-to-do message as plain text (keeps its line breaks).
        metadata = {META_RENDER_AS: "text"} if content == DREAM_NO_INPUT_MESSAGE else {}
        await _publish_followup(
            ctx, OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content, metadata=metadata)
        )

    ctx.tasks.append(asyncio.create_task(_run_dream()))
    return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content="Dreaming...")


def _format_history_message(msg: dict[str, Any]) -> str | None:
    """One message for /history, or None to skip it. (nanobot builtin.py:843)

    Hidden rows (a subagent report injected into a turn) are skipped, as
    nanobot's display history does (session/history.py:59).
    """
    role = msg.get("role")
    if role not in ("user", "assistant") or is_hidden_history_message(msg):
        return None
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
        )
    content = str(content).strip()
    if not content:
        return None
    if len(content) > _HISTORY_MAX_CONTENT_CHARS:
        content = content[:_HISTORY_MAX_CONTENT_CHARS] + "…"
    label = "👤 You" if role == "user" else "🤖 Bot"
    return f"{label}: {content}"


async def cmd_history(ctx: CommandContext) -> OutboundMessage:
    """The last N messages of the session (default 10, max 50). (nanobot builtin.py:867)

    Usage: /history [count]
    """
    count = _HISTORY_DEFAULT_COUNT
    if ctx.args.strip():
        try:
            count = max(1, min(int(ctx.args.strip()), _HISTORY_MAX_COUNT))
        except ValueError:
            return _reply(ctx, "Usage: /history [count] — e.g. /history 5 (default: 10, max: 50)")
    history = _session(ctx).get_history(include_runtime_context=False)
    visible = [m for m in (_format_history_message(m) for m in history) if m is not None]
    recent = visible[-count:]
    if not recent:
        return _reply(ctx, "No conversation history yet.")
    return _reply(ctx, f"Last {len(recent)} message(s):\n" + "\n".join(recent), as_text=True)


async def cmd_skill(ctx: CommandContext) -> OutboundMessage:
    """All enabled skills, name and description. (nanobot builtin.py:956)"""
    loader = ctx.loop.context.skills
    skills = loader.list_skills(filter_unavailable=False)
    if not skills:
        content = "No skills available."
    else:
        lines = [f"Available skills ({len(skills)}):", ""]
        for entry in skills:
            lines.append(f"- **{entry['name']}** — {loader.get_skill_description(entry['name'])}")
        content = "\n".join(lines)
    return _reply(ctx, content)


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """(nanobot builtin.py:1024)"""
    return _reply(ctx, build_help_text(), as_text=True)


def build_help_text() -> str:
    """Help text shared by all channels. (nanobot builtin.py:1058)"""
    lines = [f"{BOT_LABEL} commands:"]
    for spec in BUILTIN_COMMAND_SPECS:
        command = f"{spec.command} {spec.arg_hint}" if spec.arg_hint else spec.command
        lines.append(f"{command} — {spec.description}")
    return "\n".join(lines)


def _format_preset_names(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) if names else "(none configured)"


def _model_preset_names(loop: AgentLoop) -> list[str]:
    names = set(loop.model_presets)
    names.add("default")
    return ["default", *sorted(name for name in names if name != "default")]


def _command_error_message(exc: Exception) -> str:
    return str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)


async def _model_command_status(loop: AgentLoop, session: Session) -> str:
    """nanobot builtin.py:386."""
    names = _model_preset_names(loop)
    try:
        runtime = await loop.runtime_for_session(session, recover_removed=False)
    except (KeyError, ValueError) as exc:
        return "\n".join(
            [
                "## Model",
                f"- Current selection error: {_command_error_message(exc)}",
                f"- Available presets: {_format_preset_names(names)}",
                "- Switch with `/model <preset>`.",
            ]
        )
    active = runtime.model_preset or "default"
    return "\n".join(
        [
            "## Model",
            f"- Current model: `{runtime.model or runtime.provider.default_model}`",
            f"- Current preset: `{active}`",
            f"- Available presets: {_format_preset_names(names)}",
        ]
    )


async def cmd_model(ctx: CommandContext) -> OutboundMessage:
    """Show or switch the session's model preset. (nanobot builtin.py:406)

    The choice is saved in the session (it survives /new and restarts) and
    applies from the next turn; other sessions keep theirs. A preset builds
    its own provider once; the loop closes them all when it closes.
    """
    loop = ctx.loop
    args = ctx.args.strip()
    if not args:
        return _reply(ctx, await _model_command_status(loop, _session(ctx)), as_text=True)
    try:
        runtime = await loop.set_session_model_preset(ctx.key, args, session=ctx.session)
    except (KeyError, ValueError) as exc:
        names = _model_preset_names(loop)
        return _reply(
            ctx,
            f"Could not switch model preset: {_command_error_message(exc)}\n\n"
            f"Available presets: {_format_preset_names(names)}",
            as_text=True,
        )
    lines = [
        f"Switched model preset to `{runtime.model_preset}`.",
        "- Scope: current session",
        f"- Model: `{runtime.model or runtime.provider.default_model}`",
        f"- Context window: {runtime.context_window_tokens}",
        f"- Max output tokens: {runtime.max_tokens}",
    ]
    return _reply(ctx, "\n".join(lines), as_text=True)


def register_builtin_commands(router: CommandRouter) -> None:
    """Register NanoClaw's slash commands, as nanobot does (builtin.py:1069)."""
    router.priority("/stop", cmd_stop)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/compact", cmd_compact, session_queue=True)
    router.exact("/status", cmd_status)
    router.exact("/model", cmd_model)
    router.prefix("/model ", cmd_model)
    router.exact("/history", cmd_history)
    router.prefix("/history ", cmd_history)
    router.exact("/dream", cmd_dream)
    router.exact("/skill", cmd_skill)
    router.exact("/help", cmd_help)
