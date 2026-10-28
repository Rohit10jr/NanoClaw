"""TurnDelivery bookkeeping the web UI reads (layer 22).

Ported from nanobot tests/agent/test_turn_delivery.py where NanoClaw has the
feature: retry status audiences and terminal retry state, per-turn runtime
and usage on ``TurnCompleted``, ``TurnRuntimeAdmitted``, and the model-error
reply a ``websocket`` chat does not get.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nanoclaw.agent.runner import AgentRunner
from nanoclaw.agent.turn_delivery import TurnDeliveryFactory
from nanoclaw.bus.events import InboundMessage, OutboundMessage
from nanoclaw.bus.queue import MessageBus
from nanoclaw.bus.runtime_events import TurnCompleted, TurnRuntimeAdmitted
from nanoclaw.events import RetryStatusEvent
from nanoclaw.providers import FakeProvider, LLMProvider, LLMResponse, LLMUsage
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.webui.metadata import WEBUI_TURN_METADATA_KEY


def _runtime() -> LLMRuntime:
    return LLMRuntime(
        provider=FakeProvider(),
        model="m",
        max_tokens=10,
        temperature=0.1,
        context_window_tokens=64_000,
        model_preset="fast",
    )


def _user(channel: str = "websocket", **metadata) -> InboundMessage:
    return InboundMessage(
        channel=channel,
        sender_id="user",
        chat_id="chat",
        content="hi",
        metadata=metadata,
    )


async def test_user_websocket_turn_publishes_retry_status() -> None:
    bus = MessageBus()
    delivery = TurnDeliveryFactory(bus).create(
        _user(**{WEBUI_TURN_METADATA_KEY: "t"}), "websocket:chat"
    )
    assert delivery.events.accepts(RetryStatusEvent)
    await delivery.events.emit(
        RetryStatusEvent("waiting", 1, 4, "connection", next_retry_at=5.0)
    )
    outbound = await bus.consume_outbound()
    assert isinstance(outbound.event, RetryStatusEvent)
    assert outbound.metadata[WEBUI_TURN_METADATA_KEY] == "t"


@pytest.mark.parametrize("channel", ["telegram", "cli", "websocket"])
async def test_background_retry_status_is_quiet(channel) -> None:
    factory = TurnDeliveryFactory(MessageBus())
    delivery = factory.create(
        InboundMessage(
            channel="system", sender_id="job", chat_id=f"{channel}:chat", content=""
        ),
        f"{channel}:chat",
    )
    assert not delivery.events.accepts(RetryStatusEvent)
    await delivery.events.emit(RetryStatusEvent("waiting", 1, 4, "connection"))
    assert factory.bus.outbound.empty()


async def test_cli_turn_does_not_take_retry_status() -> None:
    delivery = TurnDeliveryFactory(MessageBus()).create(_user("cli"), "cli:chat")
    assert not delivery.events.accepts(RetryStatusEvent)


async def test_retry_completion_is_isolated_between_turns_in_one_session() -> None:
    bus = MessageBus()
    seen: list[TurnCompleted] = []
    bus.subscribe(seen.append, TurnCompleted)
    factory = TurnDeliveryFactory(bus)
    deliveries = [
        factory.create(_user(**{WEBUI_TURN_METADATA_KEY: turn}), "websocket:chat")
        for turn in ("first", "second")
    ]
    await deliveries[0].events.emit(RetryStatusEvent("exhausted", 4, 4, "connection"))
    for delivery in reversed(deliveries):
        delivery.record_stop_reason("error")
        await delivery.complete(None, publish_completion=True)
    assert [
        (
            e.context.metadata[WEBUI_TURN_METADATA_KEY],
            e.failure_attempts,
            e.failure_error_kind,
        )
        for e in seen
    ] == [
        ("second", None, None),
        ("first", 4, "connection"),
    ]


async def test_next_model_request_clears_exhaustion_within_the_same_turn() -> None:
    class Provider(LLMProvider):
        async def chat(self, **kwargs):
            return LLMResponse(
                content="payment required", finish_reason="error", error_status_code=402
            )

    bus = MessageBus()
    completed: list[TurnCompleted] = []
    bus.subscribe(completed.append, TurnCompleted)
    delivery = TurnDeliveryFactory(bus).create(_user(), "websocket:chat")
    await delivery.events.emit(RetryStatusEvent("exhausted", 4, 4, "connection"))
    # The runner opens each request chain with a "cleared" status (nanobot
    # does it in the provider's _retry_notifications).
    on_status = await AgentRunner._retry_status_callback(
        SimpleNamespace(events=delivery.events)
    )  # type: ignore[arg-type]
    assert on_status is not None
    with patch("nanoclaw.providers.base.asyncio.sleep", new_callable=AsyncMock):
        response = await Provider().chat_stream_with_retry(
            [{"role": "user", "content": "continue"}], on_retry_status=on_status
        )
    delivery.record_stop_reason(
        "error", failure_error_kind=LLMProvider.public_error_kind(response)
    )
    await delivery.complete(None, publish_completion=True)
    assert completed[0].failure_error_kind == "billing"
    assert completed[0].failure_attempts is None


async def test_exhaustion_is_not_reported_for_a_completed_turn() -> None:
    bus = MessageBus()
    completed: list[TurnCompleted] = []
    bus.subscribe(completed.append, TurnCompleted)
    delivery = TurnDeliveryFactory(bus).create(_user(), "websocket:chat")
    await delivery.events.emit(RetryStatusEvent("exhausted", 4, 4, "connection"))
    delivery.record_stop_reason("completed")
    await delivery.complete(None, publish_completion=True)
    assert (
        completed[0].outcome,
        completed[0].failure_attempts,
        completed[0].failure_error_kind,
    ) == (
        "completed",
        None,
        None,
    )


async def test_turn_completed_carries_runtime_and_usage() -> None:
    bus = MessageBus()
    admitted: list[TurnRuntimeAdmitted] = []
    completed: list[TurnCompleted] = []
    bus.subscribe(admitted.append, TurnRuntimeAdmitted)
    bus.subscribe(completed.append, TurnCompleted)
    delivery = TurnDeliveryFactory(bus).create(_user(), "websocket:chat")
    runtime = _runtime()
    await delivery.runtime_admitted(runtime)
    first = LLMUsage.reported(input_tokens=10, output_tokens=2)
    second = LLMUsage.reported(input_tokens=20, output_tokens=3)
    delivery.record_usage([first])
    delivery.record_usage([second])
    delivery.record_usage([])
    delivery.record_latency(12)
    await delivery.complete(None, publish_completion=True)
    assert [event.runtime for event in admitted] == [runtime]
    [event] = completed
    assert event.runtime is runtime
    assert event.usage == first + second
    assert event.round_usages == (first, second)
    assert event.latency_ms == 12
    # The next turn starts clean.
    await delivery.complete(None, publish_completion=True)
    assert (completed[1].runtime, completed[1].usage, completed[1].round_usages) == (
        None,
        None,
        (),
    )


async def test_background_turn_records_runtime_without_publishing_it() -> None:
    bus = MessageBus()
    admitted: list[TurnRuntimeAdmitted] = []
    completed: list[TurnCompleted] = []
    bus.subscribe(admitted.append, TurnRuntimeAdmitted)
    bus.subscribe(completed.append, TurnCompleted)
    delivery = TurnDeliveryFactory(bus).create(
        InboundMessage(
            channel="system", sender_id="subagent", chat_id="websocket:chat", content=""
        ),
        "websocket:chat",
    )
    runtime = _runtime()
    await delivery.runtime_admitted(runtime)
    await delivery.complete(None, publish_completion=True)
    assert admitted == []
    assert completed[0].runtime is runtime


@pytest.mark.parametrize(
    ("channel", "delivered"), [("websocket", False), ("cli", True), ("telegram", True)]
)
async def test_model_error_reply_is_not_sent_to_the_web_ui(channel, delivered) -> None:
    bus = MessageBus()
    delivery = TurnDeliveryFactory(bus).create(_user(channel), f"{channel}:chat")
    delivery.record_stop_reason("error")
    response = OutboundMessage(
        channel=channel,
        chat_id="chat",
        content="Sorry, I encountered an error calling the AI model.",
    )
    await delivery.complete(response, publish_completion=True)
    assert (not bus.outbound.empty()) is delivered


async def test_websocket_tool_error_reply_is_still_sent() -> None:
    bus = MessageBus()
    delivery = TurnDeliveryFactory(bus).create(_user(), "websocket:chat")
    delivery.record_stop_reason("tool_error")
    response = OutboundMessage(
        channel="websocket", chat_id="chat", content="tool failed"
    )
    await delivery.complete(response, publish_completion=True)
    assert await bus.consume_outbound() is response


# --- what the runner reports for it -----------------------------------------------


def _spec(provider, **kwargs):
    from nanoclaw.agent.hook import AgentHook
    from nanoclaw.agent.runner import AgentRunSpec
    from nanoclaw.agent.tools import ToolRegistry
    from toy_tools import EchoTool

    tools = ToolRegistry()
    tools.register(EchoTool())
    return AgentRunSpec(
        initial_messages=[{"role": "user", "content": "go"}],
        tools=tools,
        provider=provider,
        model="m",
        hook=AgentHook(),
        **kwargs,
    )


async def test_run_result_lists_each_requests_usage() -> None:
    from nanoclaw.providers import ToolCallRequest

    script = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="c1", name="echo", arguments={"text": "x"})],
            finish_reason="tool_calls",
            usage=LLMUsage.reported(input_tokens=10, output_tokens=1),
        ),
        LLMResponse(
            content="done", usage=LLMUsage.reported(input_tokens=20, output_tokens=2)
        ),
    ]
    result = await AgentRunner().run(_spec(FakeProvider(script)))
    assert [u.input_tokens for u in result.round_usages] == [10, 20]
    assert result.usage == result.round_usages[0] + result.round_usages[1]
    assert result.failure_error_kind is None


async def test_run_result_names_the_public_error_kind_of_a_model_failure() -> None:
    script = [
        LLMResponse(content="Error: nope", finish_reason="error", error_status_code=401)
    ]
    result = await AgentRunner().run(_spec(FakeProvider(script)))
    assert result.stop_reason == "error"
    assert result.failure_error_kind == "authentication"
    assert len(result.round_usages) == 1


async def test_runner_sends_retry_status_only_to_a_sink_that_takes_it() -> None:
    from nanoclaw.events import EventSink

    seen: list = []

    async def publish(event):
        seen.append(event)

    quiet = EventSink(publish, lambda event_type: event_type is not RetryStatusEvent)
    await AgentRunner().run(_spec(FakeProvider(["ok"]), events=quiet))
    assert not any(isinstance(e, RetryStatusEvent) for e in seen)
    await AgentRunner().run(_spec(FakeProvider(["ok"]), events=EventSink(publish)))
    assert [e for e in seen if isinstance(e, RetryStatusEvent)] == [
        RetryStatusEvent("cleared", 1, None, "unknown")
    ]
