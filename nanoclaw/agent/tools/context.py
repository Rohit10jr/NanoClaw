"""Who a tool call belongs to: the turn's session and model settings.

Mirrors the request part of nanobot/agent/tools/context.py (``RequestContext``,
``bind_request_context``, ``current_request_context``). The loop binds it
around a run, so a tool such as ``subagent`` can start work for the same
conversation with the same model, without that being passed through every
call. Since layer 14 it also carries the chat id, so a background subagent's
report can be addressed to the chat on the message bus. nanobot's context
also carries message IDs, the workspace scope and other routing data, which
NanoClaw doesn't have.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

from nanoclaw.utils.llm_runtime import LLMRuntime


@dataclass(frozen=True)
class RequestContext:
    """Per-request context for tools. (nanobot context.py:30)

    Field order is NanoClaw's (``channel, session_key, runtime, chat_id``).
    Since layer 19 it also carries what the ``subagent`` tool reads:
    ``message_id`` / ``turn_id`` (the task's origin), ``log_content``,
    ``persist_session`` (False in a temporary turn: a child then runs inline
    and saves nothing) and ``can_receive_background_results`` (False when no
    bus loop is running, e.g. ``nanoclaw ask``: a child then always runs
    inline, as in nanobot, whose loop sets it to ``self._running``).
    """

    channel: str
    session_key: str | None = None
    runtime: LLMRuntime | None = None
    chat_id: str = "direct"
    message_id: str | None = None
    turn_id: str | None = None
    log_content: bool = True
    # The host can consume completion messages after this request returns.
    can_receive_background_results: bool = True
    persist_session: bool = True


_current: ContextVar[RequestContext | None] = ContextVar("nanoclaw_request_context", default=None)


def bind_request_context(ctx: RequestContext) -> Token[RequestContext | None]:
    return _current.set(ctx)


def reset_request_context(token: Token[RequestContext | None]) -> None:
    _current.reset(token)


@contextmanager
def request_context(ctx: RequestContext) -> Iterator[RequestContext]:
    """Bind ``ctx`` for a block and restore the previous value (nanobot context.py:69)."""
    token = bind_request_context(ctx)
    try:
        yield ctx
    finally:
        reset_request_context(token)


def current_request_context() -> RequestContext | None:
    return _current.get()


def current_request_session_key() -> str | None:
    """The session key of the turn running this tool call (nanobot context.py:88).

    ``exec`` stamps it on the exec sessions it starts, and ``exec_session`` /
    ``list_exec_sessions`` only see sessions with the same key, so one chat
    can't read or drive another chat's processes.
    """
    ctx = current_request_context()
    return ctx.session_key if ctx else None
