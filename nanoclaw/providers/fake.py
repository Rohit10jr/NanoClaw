"""A scripted provider for tests: no network, no API key."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from nanoclaw.providers.base import (
    ContentDeltaCallback,
    LLMProvider,
    LLMResponse,
    RetryWaitCallback,
    StreamRecoverCallback,
    ToolCallDeltaCallback,
)


@dataclass
class RecordedCall:
    """What the agent sent on one ``chat`` or ``chat_stream`` call."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    model: str | None
    max_tokens: int = 4096
    temperature: float = 0.7
    streamed: bool = False


@dataclass
class ScriptedStream:
    """A script entry with explicit stream deltas.

    ``chat_stream`` sends each delta to its callback in order, then returns
    ``response``. ``response`` defaults to a reply whose content is the joined
    ``content_deltas``; give an error response to script a stream that fails
    after showing text. ``chat`` returns ``response`` without the deltas.
    """

    content_deltas: list[str] = field(default_factory=list)
    thinking_deltas: list[str] = field(default_factory=list)
    tool_call_deltas: list[dict[str, Any]] = field(default_factory=list)
    response: LLMResponse | None = None

    def final_response(self) -> LLMResponse:
        if self.response is not None:
            return self.response
        return LLMResponse(content="".join(self.content_deltas) or None)


ScriptItem = LLMResponse | ScriptedStream | str


def _split_deltas(text: str) -> list[str]:
    """``"Hello big world"`` -> ``["Hello ", "big ", "world"]``: one delta per word."""
    return re.findall(r"\s*\S+\s*|\s+", text) if text else []


def _as_stream(item: LLMResponse | ScriptedStream) -> ScriptedStream:
    """How a plain scripted response streams: content word by word, the
    reasoning as one thinking delta, each tool call as one tool-call delta.
    Error responses stream nothing."""
    if isinstance(item, ScriptedStream):
        return item
    if item.is_error:
        return ScriptedStream(response=item)
    return ScriptedStream(
        content_deltas=_split_deltas(item.content or ""),
        thinking_deltas=[item.reasoning_content] if item.reasoning_content else [],
        tool_call_deltas=[
            {
                "index": index,
                "call_id": call.id,
                "name": call.name,
                "arguments_delta": call.to_openai_tool_call()["function"]["arguments"],
            }
            for index, call in enumerate(item.tool_calls)
        ],
        response=item,
    )


class FakeProvider(LLMProvider):
    """Returns queued responses in order and records every request.

    A plain string in the script is shorthand for a text-only reply; a
    :class:`ScriptedStream` scripts the exact deltas ``chat_stream`` sends.
    Every call, streamed or not, consumes one entry; ``calls[i].streamed``
    tells which kind it was.

    By default each script entry answers one logical request:
    :meth:`chat_with_retry` and :meth:`chat_stream_with_retry` make a single
    call, so a scripted error reply is final and no test waits on backoff.
    ``retries=True`` runs the real retry policy, where a retried error
    consumes the next entry.
    """

    def __init__(self, script: list[ScriptItem] | None = None, *, retries: bool = False) -> None:
        super().__init__(default_model="fake-model")
        self.retries = retries
        # The retry_mode of the latest chat_with_retry / chat_stream_with_retry call.
        self.last_retry_mode: str | None = None
        self._script: list[LLMResponse | ScriptedStream] = [self._entry(item) for item in (script or [])]
        self.calls: list[RecordedCall] = []

    @staticmethod
    def _entry(item: ScriptItem) -> LLMResponse | ScriptedStream:
        return LLMResponse(content=item) if isinstance(item, str) else item

    def queue(self, *items: ScriptItem) -> None:
        for item in items:
            self._script.append(self._entry(item))

    @property
    def stream_calls(self) -> int:
        """How many calls went through ``chat_stream``."""
        return sum(call.streamed for call in self.calls)

    def _next(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        *,
        streamed: bool,
    ) -> LLMResponse | ScriptedStream:
        # Deep-copy so later mutation of the caller's history can't rewrite
        # what the test sees as "sent".
        self.calls.append(
            RecordedCall(
                copy.deepcopy(messages),
                copy.deepcopy(tools),
                model or self.default_model,
                max_tokens,
                temperature,
                streamed,
            )
        )
        if not self._script:
            return LLMResponse(content="FakeProvider: script exhausted", finish_reason="error")
        return self._script.pop(0)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        item = self._next(messages, tools, model, max_tokens, temperature, streamed=False)
        return item.final_response() if isinstance(item, ScriptedStream) else item

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: ContentDeltaCallback | None = None,
        on_thinking_delta: ContentDeltaCallback | None = None,
        on_tool_call_delta: ToolCallDeltaCallback | None = None,
    ) -> LLMResponse:
        stream = _as_stream(self._next(messages, tools, model, max_tokens, temperature, streamed=True))
        for text in stream.thinking_deltas:
            if on_thinking_delta:
                await on_thinking_delta(text)
        for text in stream.content_deltas:
            if on_content_delta:
                await on_content_delta(text)
        for tool_delta in stream.tool_call_deltas:
            if on_tool_call_delta:
                await on_tool_call_delta(dict(tool_delta))
        return stream.final_response()

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
        retry_mode: str = "standard",
        on_retry_wait: RetryWaitCallback | None = None,
    ) -> LLMResponse:
        self.last_retry_mode = retry_mode
        if self.retries:
            return await super().chat_with_retry(
                messages, tools, model, max_tokens, temperature, tool_choice, retry_mode, on_retry_wait
            )
        return await self.safe_chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tool_choice=tool_choice,
        )

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: ContentDeltaCallback | None = None,
        on_thinking_delta: ContentDeltaCallback | None = None,
        on_tool_call_delta: ToolCallDeltaCallback | None = None,
        on_stream_recover: StreamRecoverCallback | None = None,
        retry_mode: str = "standard",
        on_retry_wait: RetryWaitCallback | None = None,
    ) -> LLMResponse:
        self.last_retry_mode = retry_mode
        if self.retries:
            return await super().chat_stream_with_retry(
                messages,
                tools,
                model,
                max_tokens,
                temperature,
                tool_choice,
                on_content_delta,
                on_thinking_delta,
                on_tool_call_delta,
                on_stream_recover,
                retry_mode,
                on_retry_wait,
            )
        return await self.safe_chat_stream(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tool_choice=tool_choice,
            on_content_delta=on_content_delta,
            on_thinking_delta=on_thinking_delta,
            on_tool_call_delta=on_tool_call_delta,
        )
