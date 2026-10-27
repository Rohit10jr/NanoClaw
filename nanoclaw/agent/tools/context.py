"""Who a tool call belongs to: the turn's session and model settings.

Mirrors the request part of nanobot/agent/tools/context.py (``RequestContext``,
``bind_request_context``, ``current_request_context``). The loop binds it
around a run, so a tool such as ``spawn`` can start work for the same
conversation with the same model, without that being passed through every
call. Since layer 14 it also carries the chat id, so a background subagent's
report can be addressed to the chat on the message bus. nanobot's context
also carries message IDs, the workspace scope and other routing data, which
NanoClaw doesn't have.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from nanoclaw.utils.llm_runtime import LLMRuntime


@dataclass(frozen=True)
class RequestContext:
    channel: str
    session_key: str | None
    runtime: LLMRuntime | None
    chat_id: str = "direct"


_current: ContextVar[RequestContext | None] = ContextVar("nanoclaw_request_context", default=None)


def bind_request_context(ctx: RequestContext) -> Token[RequestContext | None]:
    return _current.set(ctx)


def reset_request_context(token: Token[RequestContext | None]) -> None:
    _current.reset(token)


def current_request_context() -> RequestContext | None:
    return _current.get()
