"""Slash-command routing, independent of any terminal or channel.

Copied from nanobot/command/router.py. A handler gets a
:class:`CommandContext` and returns an :class:`OutboundMessage` (the reply), an
:class:`InboundMessage` (input for the agent instead), or None (nothing to say).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nanoclaw.bus.events import META_ORIGINAL_COMMAND, META_RENDER_AS, InboundMessage, OutboundMessage
from nanoclaw.events import NO_EVENTS, EventSink

if TYPE_CHECKING:
    from nanoclaw.agent.loop import AgentLoop
    from nanoclaw.bus.queue import MessageBus
    from nanoclaw.session import Session
    from nanoclaw.utils.llm_runtime import LLMRuntime

Handler = Callable[["CommandContext"], Awaitable["InboundMessage | OutboundMessage | None"]]
_BOT_SUFFIX_RE = re.compile(r"^[A-Za-z0-9_]+$")


def command_text(message: InboundMessage) -> str:
    """The routable text of *message*; agent input a command produced is never
    routed again. (nanobot router.py:21)"""
    return "" if message.metadata.get(META_ORIGINAL_COMMAND) else message.content.strip()


def normalize_command_text(text: str) -> str:
    """Strip a transport ``@bot`` suffix from the command word. (nanobot router.py:26)

    Telegram- and Discord-style dispatch can produce ``/cmd@bot args``; the
    suffix belongs to the transport, so ``/cmd@bot args`` becomes ``/cmd args``.
    Arguments stay verbatim.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return stripped
    first, sep, rest = stripped.partition(" ")
    if "@" not in first:
        return stripped
    command, suffix = first.rsplit("@", 1)
    if command and suffix and _BOT_SUFFIX_RE.fullmatch(suffix):
        return f"{command}{sep}{rest}" if sep else command
    return stripped


@dataclass
class CommandContext:
    """Everything a command handler needs. (nanobot router.py:45)

    nanobot's ``turn_scopes`` is left out (no turn scopes in NanoClaw). Three
    fields are NanoClaw's own, standing in for nanobot loop plumbing it does
    not have yet:

    - ``events``: where a command's progress events go (/compact). nanobot
      builds a turn delivery for this (``loop.turn_delivery_factory``).
    - ``bus``: where follow-up replies go (/dream's result). nanobot uses
      ``loop.bus``; None falls back to ``loop.bus`` if the loop has one.
    - ``tasks``: background work a handler started (/dream), so a caller that
      wants the follow-up (the terminal chat) can wait for it.
    """

    msg: InboundMessage
    session: Session | None
    key: str
    raw: str
    args: str = ""
    loop: AgentLoop = field(kw_only=True)
    runtime: LLMRuntime | None = None
    is_user_turn: bool = False
    events: EventSink = field(default=NO_EVENTS, kw_only=True)
    bus: MessageBus | None = field(default=None, kw_only=True)
    tasks: list[asyncio.Task[None]] = field(default_factory=list, kw_only=True)


class CommandRouter:
    """Dict-based command dispatch. (nanobot router.py:60)

    Three tiers, checked in order:
      1. *priority*: exact-match commands handled before the session lock
         (e.g. /stop).
      2. *exact*: exact-match commands handled inside it.
      3. *prefix*: longest prefix first (e.g. "/history ").
    """

    def __init__(self) -> None:
        self._priority: dict[str, Handler] = {}
        self._exact: dict[str, Handler] = {}
        self._prefix: list[tuple[str, Handler]] = []
        self._session_commands: set[str] = set()

    def priority(self, cmd: str, handler: Handler) -> None:
        self._priority[cmd] = handler

    def exact(self, cmd: str, handler: Handler, *, session_queue: bool = False) -> None:
        """``session_queue=True``: the command must run on the session's worker,
        in order with its turns (nanobot router.py:79)."""
        self._exact[cmd] = handler
        if session_queue:
            self._session_commands.add(cmd)
        else:
            self._session_commands.discard(cmd)

    def prefix(self, pfx: str, handler: Handler) -> None:
        self._prefix.append((pfx, handler))
        self._prefix.sort(key=lambda p: len(p[0]), reverse=True)

    def is_priority(self, text: str) -> bool:
        return normalize_command_text(text).lower() in self._priority

    def requires_session_queue(self, text: str) -> bool:
        """Whether an exact command must run on the session worker. (nanobot router.py:93)"""
        cmd = normalize_command_text(text).lower()
        return cmd in self._session_commands and cmd not in self._priority

    def command_name(self, text: str) -> str | None:
        """The registered command *text* starts with, or None for ordinary chat
        text. (nanobot router.py:98)

        A leading slash alone is not a command: absolute paths such as
        ``/tmp/foo.py explain`` and unknown names go to the model.
        """
        parts = normalize_command_text(text).split(maxsplit=1)
        return self._registered_commands().get(parts[0].lower()) if parts else None

    def is_dispatchable_command(self, text: str) -> bool:
        """Whether *text* goes to non-priority ``dispatch``. (nanobot router.py:107)

        Exact priority commands are handled separately. Known non-priority
        commands and wrong uses of registered commands (so argument errors are
        answered instead of reaching the model) are dispatched here.
        """
        cmd = normalize_command_text(text).lower()
        if cmd in self._priority:
            return False
        if cmd in self._exact:
            return True
        for pfx, _ in self._prefix:
            if cmd.startswith(pfx):
                return True
        return self.command_name(text) is not None

    async def dispatch_priority(self, ctx: CommandContext) -> InboundMessage | OutboundMessage | None:
        """Run a priority command, without the session lock. (nanobot router.py:124)"""
        ctx.raw = normalize_command_text(ctx.raw)
        handler = self._priority.get(ctx.raw.lower())
        if handler:
            return await handler(ctx)
        return None

    async def dispatch(self, ctx: CommandContext) -> InboundMessage | OutboundMessage | None:
        """Try exact and prefix handlers, then answer wrong uses of registered
        commands. (nanobot router.py:132)"""
        if ctx.msg.metadata.get(META_ORIGINAL_COMMAND):
            return None
        ctx.raw = normalize_command_text(ctx.raw)
        cmd = ctx.raw.lower()

        if handler := self._exact.get(cmd):
            return await handler(ctx)

        for pfx, handler in self._prefix:
            if cmd.startswith(pfx):
                ctx.args = ctx.raw[len(pfx):]
                return await handler(ctx)

        return self._invalid_command_response(ctx)

    def invalid_command_text(self, raw: str) -> str | None:
        """What a registered command used with wrong arguments gets, or None
        when *raw* isn't a registered command. (nanobot router.py:149-168,
        split out of ``_invalid_command_response``)"""
        raw = normalize_command_text(raw)
        canonical = self.command_name(raw)
        if canonical is None:
            return None
        entered = raw.split(maxsplit=1)[0]
        accepts_args = any(pfx.rstrip().lower() == entered.lower() for pfx, _ in self._prefix)
        if accepts_args:
            return f'Invalid command "{entered}". Use "/help" to list available commands.'
        return f'Command "{canonical}" does not accept arguments. Did you mean "{canonical}"?'

    def _invalid_command_response(self, ctx: CommandContext) -> OutboundMessage | None:
        """(nanobot router.py:149)"""
        content = self.invalid_command_text(ctx.raw)
        if content is None:
            return None
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=content,
            metadata={**dict(ctx.msg.metadata or {}), META_RENDER_AS: "text"},
        )

    def _registered_commands(self) -> dict[str, str]:
        """(nanobot router.py:176)"""
        commands = [*self._priority, *self._exact]
        commands.extend(pfx.rstrip() for pfx, _ in self._prefix)
        return {command.lower(): command for command in commands if command}
