"""A running AgentLoop on its bus, for the layer 14 tests (test_loop_bus.py, test_cancellation.py)."""

from __future__ import annotations

import asyncio
from typing import Any

from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.tools import Tool, ToolRegistry
from nanoclaw.agent.tools.spawn import SpawnTool
from nanoclaw.bus import InboundMessage, OutboundMessage
from nanoclaw.bus.outbound_events import StreamedResponseEvent
from nanoclaw.bus.runtime_events import SessionTurnStarted, TurnCompleted, TurnRunStatusChanged
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.session import SessionManager
from nanoclaw.utils.templates import sync_workspace_templates
from toy_tools import EchoTool


def call(name: str, args: dict[str, Any], call_id: str = "c1", content: str | None = None) -> LLMResponse:
    return LLMResponse(
        content=content, tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=args)], finish_reason="tool_calls"
    )


class SlowTool(Tool):
    """``slow(text)`` sets ``started[text]`` and waits for ``gates[text]``."""

    def __init__(self) -> None:
        self.started: dict[str, asyncio.Event] = {}
        self.gates: dict[str, asyncio.Event] = {}
        self.cancelled: list[str] = []

    def gate(self, text: str) -> asyncio.Event:
        return self.gates.setdefault(text, asyncio.Event())

    def running(self, text: str) -> asyncio.Event:
        return self.started.setdefault(text, asyncio.Event())

    @property
    def name(self) -> str:
        return "slow"

    @property
    def description(self) -> str:
        return "Wait."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, text: str) -> str:
        self.running(text).set()
        try:
            await self.gate(text).wait()
        except asyncio.CancelledError:
            self.cancelled.append(text)
            raise
        return f"slow:{text}"


class Rig:
    """``async with Rig(...) as rig``: the loop runs, outbound and lifecycle events are recorded."""

    def __init__(self, ws, tmp_path, script=(), *, spawn: bool = False, sessions_root=None) -> None:
        sync_workspace_templates(ws)
        self.provider = FakeProvider(list(script))
        self.slow = SlowTool()
        tools = ToolRegistry()
        tools.register(self.slow)
        tools.register(EchoTool())
        self.loop = AgentLoop(
            self.provider,
            ws,
            tools=tools,
            sessions=SessionManager(ws, sessions_root=sessions_root or tmp_path / "sessions"),
        )
        if spawn:
            tools.register(SpawnTool(self.loop.subagents))
        self.outbound: list[OutboundMessage] = []
        self.lifecycle: list[Any] = []
        self._tasks: list[asyncio.Task[Any]] = []

    async def __aenter__(self) -> Rig:
        async def consume() -> None:
            while True:
                self.outbound.append(await self.loop.bus.consume_outbound())

        async def record(event: Any) -> None:
            self.lifecycle.append(event)

        for event_type in (SessionTurnStarted, TurnRunStatusChanged, TurnCompleted):
            self.loop.bus.subscribe(record, event_type)
        self._tasks = [asyncio.create_task(self.loop.run()), asyncio.create_task(consume())]
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.loop.stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def send(self, content: str, chat: str = "a", *, stream: bool = True, **metadata: Any) -> InboundMessage:
        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id=chat,
            content=content,
            metadata={"_wants_stream": True, **metadata} if stream else dict(metadata),
        )
        await self.loop.bus.publish_inbound(msg)
        return msg

    @staticmethod
    async def until(predicate, timeout: float = 5) -> None:
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.01)

    def replies(self, chat: str | None = None) -> list[OutboundMessage]:
        """Final answers and command replies (no stream or progress events)."""
        return [
            m
            for m in self.outbound
            if (m.event is None or isinstance(m.event, StreamedResponseEvent)) and (chat is None or m.chat_id == chat)
        ]

    def texts(self, chat: str | None = None) -> list[str]:
        return [m.content for m in self.replies(chat)]

    async def settle(self, n_replies: int, chat: str | None = None) -> None:
        """Wait for *n_replies* replies and for every session worker to finish."""
        await self.until(lambda: len(self.replies(chat)) >= n_replies and not self.loop._active_tasks)

    def saved(self, key: str = "cli:a") -> list[dict[str, Any]]:
        return self.loop.sessions.get_or_create(key).messages

    def session(self, key: str = "cli:a"):
        return self.loop.sessions.get_or_create(key)
