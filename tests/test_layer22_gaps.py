"""Gaps found by the layer 22 mutation checks (web UI session history).

Each test pins one behaviour of the layer 22 code that the rest of the suite
left open: wire encoders and the outbound projector, retry status and turn
lifecycle events, the per-chat display history and webui-thread projection,
and the session list / titles / deletion helpers.
"""

from __future__ import annotations

import pytest

from nanoclaw.webui.outbound_wire import project_tool_events

# --- area 1: wire encoders and projector ---------------------------------------


def test_tool_event_scrub_recognises_data_urls_loosely_but_only_base64() -> None:
    long_mime = "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,AAAA"
    [projected] = project_tool_events([
        {
            "padded": "  \n data:image/png;base64,AAAA",
            "upper": "DATA:IMAGE/PNG;BASE64,AAAA",
            "long_mime": long_mime,
            "plain_data_url": "data:text/plain,hello",
            "late_marker": "data:" + "x" * 300 + ";base64,AAAA",
        }
    ])
    omitted = "[binary content omitted from WebUI]"
    assert projected["padded"] == omitted
    assert projected["upper"] == omitted
    assert projected["long_mime"] == omitted
    # Not base64 (or the marker is beyond the 256-char window): kept as is.
    assert projected["plain_data_url"] == "data:text/plain,hello"
    assert projected["late_marker"].startswith("data:xxx")


def test_tool_event_projection_scrubs_tuples_stringifies_others_and_skips_non_objects() -> None:
    from pathlib import PurePosixPath

    projected = project_tool_events([
        "not an event",  # type: ignore[list-item]
        {"args": ("data:image/png;base64,AAAA", 1), "path": PurePosixPath("/w/a.txt")},
    ])
    assert projected == [
        {"args": ["[binary content omitted from WebUI]", 1], "path": "/w/a.txt"},
    ]


def test_notification_offline_delivery_is_only_for_compaction() -> None:
    from nanoclaw.events import ContextCompactionEvent, RetryStatusEvent
    from nanoclaw.webui.outbound_wire import project_notification

    compaction = project_notification("c", ContextCompactionEvent("k1", "succeeded"))
    retry = project_notification("c", RetryStatusEvent("waiting", 1, 3, "server"))
    assert compaction is not None and compaction.deliver_offline is True
    assert retry is not None and retry.deliver_offline is False


def test_encoders_keep_present_but_falsy_optional_fields() -> None:
    from nanoclaw.bus.outbound_events import TurnEndEvent
    from nanoclaw.events import RetryStatusEvent
    from nanoclaw.webui.outbound_wire import (
        encode_delta,
        encode_retry_status,
        encode_stream_end,
        encode_turn_end,
    )

    assert encode_retry_status("c", RetryStatusEvent("exhausted", 1, 0, "server"))["max_attempts"] == 0
    assert encode_turn_end("c", TurnEndEvent(goal_state={}))["goal_state"] == {}
    assert encode_delta("c", "x", stream_id="")["stream_id"] == ""
    assert encode_stream_end("c", text="")["text"] == ""


def test_encode_message_shell_flag_turn_id_and_empty_agent_ui() -> None:
    from nanoclaw.bus.events import OUTBOUND_META_AGENT_UI, OutboundMessage
    from nanoclaw.webui.outbound_wire import encode_message

    msg = OutboundMessage(
        channel="websocket", chat_id="c", content="raw",
        metadata={"_user_shell": "yes", "webui_turn_id": "t-1", OUTBOUND_META_AGENT_UI: {}},
    )
    payload = encode_message(msg, None, wire_text="rewritten")
    # Only ``_user_shell is True`` marks a shell reply.
    assert payload["text"] == "rewritten" and "render_as" not in payload
    assert payload["turn_id"] == "t-1"
    assert payload["agent_ui"] == {}


def test_model_update_encoders_normalise_names_and_optional_fields() -> None:
    from nanoclaw.webui.outbound_wire import encode_runtime_model_updated, encode_turn_model_updated

    assert encode_runtime_model_updated(model_name="  m1  ") == {
        "event": "runtime_model_updated", "model_name": "m1",
    }
    assert encode_runtime_model_updated(model_name="   ") is None
    body = encode_turn_model_updated(
        "c", model_name="m", context_window_tokens=0, reauth_provider="openai",
    )
    assert body == {"event": "turn_model_updated", "chat_id": "c", "model_name": "m"}


def test_message_source_label_is_stripped_and_one_bad_response_source_drops_all() -> None:
    from nanoclaw.webui.outbound_wire import response_sources, webui_message_source

    assert webui_message_source(
        {"_webui_message_source": {"kind": "cron", "label": "  nightly  "}}
    ) == {"kind": "cron", "label": "nightly"}
    good = {"provider": "p", "model": "m", "preset": "s"}
    assert response_sources([good, {"provider": "p", "model": "", "preset": "s"}]) == []
    assert response_sources([good, "not a source"]) == []
    assert response_sources([{**good, "fallback": "yes"}]) == [{**good, "fallback": False}]


def _out(event=None, *, content=None, metadata=None):
    from nanoclaw.bus.events import OutboundMessage
    from nanoclaw.bus.outbound_events import outbound_message_for_event

    if event is None:
        return OutboundMessage(
            channel="websocket", chat_id="chat-1", content=content or "", metadata=dict(metadata or {}),
        )
    return outbound_message_for_event(
        channel="websocket", chat_id="chat-1", event=event, content=content, metadata=metadata,
    )


def test_system_command_prefix_is_the_wire_literal() -> None:
    from nanoclaw.bus.outbound_events import TurnEndEvent
    from nanoclaw.webui.outbound_projection import WebUIOutboundProjector

    _, updated = WebUIOutboundProjector().project(
        _out(TurnEndEvent(), metadata={"webui_turn_id": "webui-system:restart"})
    )
    assert updated.payload["scope"] == "metadata"


def test_goal_status_ignores_a_non_string_turn_owner() -> None:
    from nanoclaw.bus.outbound_events import GoalStatusEvent
    from nanoclaw.webui.outbound_projection import WebUIOutboundProjector

    [frame] = WebUIOutboundProjector().project(
        _out(GoalStatusEvent(status="idle"), metadata={"_websocket_turn_owner": 123})
    )
    assert frame.release_turn_owner is True and frame.turn_owner is None


def test_open_streams_reasoning_buffers_and_forget_chat() -> None:
    from nanoclaw.bus.outbound_events import ProgressEvent, StreamDeltaEvent
    from nanoclaw.webui.outbound_projection import WebUIOutboundProjector

    projector = WebUIOutboundProjector()
    projector.project(_out(ProgressEvent(content="think", reasoning_delta=True, stream_id="r")))
    assert projector.has_open_stream("chat-1")
    [end] = projector.project(_out(ProgressEvent(reasoning_end=True, stream_id="r")))
    assert end.record is not None and end.record["text"] == "think"
    assert not projector.has_open_stream("chat-1")
    # The buffer was consumed: a second end carries no text.
    [again] = projector.project(_out(ProgressEvent(reasoning_end=True, stream_id="r")))
    assert again.record is not None and "text" not in again.record

    projector.project(_out(StreamDeltaEvent(stream_id="s"), content="partial"))
    assert projector.has_open_stream("chat-1")
    projector.forget_chat("chat-1")
    assert not projector.has_open_stream("chat-1")


def test_reasoning_and_file_edits_carry_no_message_source() -> None:
    from nanoclaw.bus.outbound_events import ProgressEvent
    from nanoclaw.webui.outbound_projection import WebUIOutboundProjector

    meta = {"_webui_message_source": {"kind": "cron"}, "response_sources": []}
    projector = WebUIOutboundProjector()
    [delta] = projector.project(_out(ProgressEvent(content="t", reasoning_delta=True), metadata=meta))
    [end] = projector.project(_out(ProgressEvent(reasoning_end=True), metadata=meta))
    [edit] = projector.project(
        _out(ProgressEvent(file_edit_events=[{"path": "a"}]), metadata=meta)
    )
    for frame in (delta, end, edit):
        assert "source" not in frame.payload and "response_sources" not in frame.payload


def test_annotate_needs_a_phase_and_a_non_empty_turn_id() -> None:
    from nanoclaw.webui.outbound_projection import WebUIOutboundProjector

    projector = WebUIOutboundProjector()
    no_phase: dict = {"event": "message"}
    projector.annotate("c", no_phase, metadata={"webui_turn_id": "t"})
    empty_turn: dict = {"event": "message"}
    projector.annotate("c", empty_turn, metadata={"webui_turn_id": ""}, phase="answer")
    assert no_phase == {"event": "message"} and empty_turn == {"event": "message"}
    first: dict = {}
    projector.annotate("c", first, metadata={"webui_turn_id": "t"}, phase="user")
    assert first["turn_seq"] == 1


def test_transient_notifications_do_not_carry_turn_metadata() -> None:
    from nanoclaw.events import RetryStatusEvent
    from nanoclaw.webui.outbound_projection import WebUIOutboundProjector

    [frame] = WebUIOutboundProjector().project(
        _out(RetryStatusEvent("waiting", 1, 3, "server"), metadata={"webui_turn_id": "t"})
    )
    assert frame.payload["turn_id"] == "t"
    assert frame.metadata is None


def test_lifecycle_running_without_a_positive_start_starts_now() -> None:
    from nanoclaw.bus.runtime_events import RuntimeEventContext, TurnRunStatusChanged
    from nanoclaw.webui.outbound_projection import lifecycle_outbound

    ctx = RuntimeEventContext(channel="websocket", chat_id="c", session_key="websocket:c", metadata={})
    msg = lifecycle_outbound(
        TurnRunStatusChanged(context=ctx, status="running", started_at=0), now=lambda: 42.0
    )
    assert msg is not None and msg.event.started_at == 42.0  # type: ignore[union-attr]


# --- area 2: retry status and turn lifecycle -----------------------------------


async def test_fake_provider_chat_with_retry_reports_retry_status() -> None:
    from unittest.mock import AsyncMock, patch

    from nanoclaw.providers import FakeProvider, LLMResponse

    provider = FakeProvider(
        [LLMResponse(content="503", finish_reason="error", error_status_code=503), "ok"], retries=True,
    )
    states: list[str] = []

    async def record(status) -> None:
        states.append(status.state)

    with patch("nanoclaw.providers.base.asyncio.sleep", new_callable=AsyncMock):
        response = await provider.chat_with_retry([{"role": "user", "content": "hi"}], on_retry_status=record)
    assert response.content == "ok"
    assert states == ["waiting", "recovered"]


async def test_fallback_that_fails_without_exhausting_does_not_clear_twice() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    from nanoclaw.providers.fallback_provider import FallbackProvider
    from test_fallback_provider import HI, SLEEP, _FakeProvider, _fallback, _make_response, _retryable_error

    primary = _FakeProvider("primary", responses=[_retryable_error(f"attempt {n}") for n in range(4)])
    auth_failure = _FakeProvider("a", _make_response("auth", finish_reason="error", error_status_code=401))
    rescue = _FakeProvider("b", _make_response("fallback ok"))
    provider = FallbackProvider(
        primary, [_fallback("fallback-a"), _fallback("fallback-b")], MagicMock(side_effect=[auth_failure, rescue]),
    )
    states: list[str] = []

    async def record(status) -> None:
        states.append(status.state)

    with patch(SLEEP, new_callable=AsyncMock):
        result = await provider.chat_with_retry(HI, on_retry_status=record)
    assert result.content == "fallback ok"
    assert states == ["waiting", "waiting", "waiting", "cleared"]


def _usage_spec(script, **kwargs):
    from nanoclaw.agent.hook import AgentHook
    from nanoclaw.agent.runner import AgentRunSpec
    from nanoclaw.agent.tools import ToolRegistry
    from nanoclaw.providers import FakeProvider
    from toy_tools import EchoTool

    tools = ToolRegistry()
    tools.register(EchoTool())
    return AgentRunSpec(
        initial_messages=[{"role": "user", "content": "go"}], tools=tools,
        provider=FakeProvider(script), model="m", hook=AgentHook(), **kwargs,
    )


async def test_run_result_round_usages_include_the_empty_reply_finalization() -> None:
    from nanoclaw.agent.runner import AgentRunner
    from nanoclaw.providers import LLMResponse, LLMUsage

    script = [
        LLMResponse(content="", usage=LLMUsage.reported(input_tokens=1, output_tokens=0)),
        LLMResponse(content="", usage=LLMUsage.reported(input_tokens=2, output_tokens=0)),
        LLMResponse(content="final", usage=LLMUsage.reported(input_tokens=3, output_tokens=1)),
    ]
    result = await AgentRunner().run(_usage_spec(script))
    assert result.final_content == "final"
    assert [u.input_tokens for u in result.round_usages] == [1, 2, 3]


async def test_run_result_round_usages_include_the_max_iterations_summary() -> None:
    from nanoclaw.agent.runner import AgentRunner
    from nanoclaw.providers import LLMResponse, LLMUsage, ToolCallRequest

    call = LLMResponse(
        content=None, tool_calls=[ToolCallRequest(id="c1", name="echo", arguments={"text": "x"})],
        finish_reason="tool_calls", usage=LLMUsage.reported(input_tokens=1, output_tokens=0),
    )
    summary = LLMResponse(content="summary", usage=LLMUsage.reported(input_tokens=9, output_tokens=1))
    result = await AgentRunner().run(_usage_spec([call, summary], max_iterations=1))
    assert result.stop_reason == "max_iterations"
    assert [u.input_tokens for u in result.round_usages] == [1, 9]


async def test_runner_failure_kind_wins_over_the_retry_exhaustion_kind() -> None:
    from nanoclaw.agent.turn_delivery import TurnDeliveryFactory
    from nanoclaw.bus.events import InboundMessage
    from nanoclaw.bus.queue import MessageBus
    from nanoclaw.bus.runtime_events import TurnCompleted
    from nanoclaw.events import RetryStatusEvent

    bus = MessageBus()
    completed: list[TurnCompleted] = []
    bus.subscribe(completed.append, TurnCompleted)
    delivery = TurnDeliveryFactory(bus).create(
        InboundMessage(channel="websocket", sender_id="u", chat_id="chat", content="hi"), "websocket:chat",
    )
    await delivery.events.emit(RetryStatusEvent("exhausted", 4, 4, "server"))
    delivery.record_stop_reason("error", failure_error_kind="authentication")
    await delivery.complete(None, publish_completion=True)
    assert (completed[0].failure_error_kind, completed[0].failure_attempts) == ("authentication", 4)


async def test_idle_drops_the_runtime_and_usage_of_an_uncompleted_turn() -> None:
    from nanoclaw.agent.turn_delivery import TurnDeliveryFactory
    from nanoclaw.bus.events import InboundMessage
    from nanoclaw.bus.queue import MessageBus
    from nanoclaw.bus.runtime_events import TurnCompleted
    from nanoclaw.providers import FakeProvider, LLMUsage
    from nanoclaw.utils.llm_runtime import LLMRuntime

    bus = MessageBus()
    completed: list[TurnCompleted] = []
    bus.subscribe(completed.append, TurnCompleted)
    factory = TurnDeliveryFactory(bus)
    inbound = InboundMessage(channel="websocket", sender_id="u", chat_id="chat", content="hi")
    first = factory.create(inbound, "websocket:chat")
    await first.runtime_admitted(LLMRuntime(
        provider=FakeProvider(), model="m", max_tokens=10, temperature=0.1,
        context_window_tokens=1000, model_preset="fast",
    ))
    first.record_usage([LLMUsage.reported(input_tokens=5, output_tokens=1)])
    await first.idle()
    second = factory.create(inbound, "websocket:chat")
    await second.complete(None, publish_completion=True)
    assert (completed[0].runtime, completed[0].usage, completed[0].round_usages) == (None, None, ())


# --- area 3: display history store and webui-thread projection ------------------


def test_legacy_role_records_become_input_and_answer_facts() -> None:
    from nanoclaw.session.display_history import HistoryKind
    from nanoclaw.webui.history import fact_from_record

    user = fact_from_record({"role": "user", "content": "question"})
    answer = fact_from_record({"role": "assistant", "content": "reply"})
    assert (user.kind, user.content["text"]) == (HistoryKind.INPUT, "question")
    assert (answer.kind, answer.content["text"]) == (HistoryKind.ANSWER, "reply")



@pytest.mark.parametrize("bad", [
    {"created_at_ms": True},
    {"created_at_ms": -1},
    {"created_at_ms": 10**16},
    {"turn_seq": True},
    {"turn_seq": -1},
    {"turn_id": "t" * 129},
    {"turn_phase": 3},
])
def test_history_record_validation_rejects_bad_identity_fields(bad) -> None:
    from nanoclaw.webui.history import fact_from_record

    with pytest.raises(ValueError):
        fact_from_record({"event": "user", "text": "x", **bad})
    # The boundary values themselves are fine.
    fact_from_record({"event": "user", "text": "x", "created_at_ms": 10**16 - 1, "turn_seq": 0, "turn_id": "t" * 128})


def test_history_directory_must_not_be_a_symlink_and_is_private(tmp_path) -> None:
    from nanoclaw.webui.history import WebUIHistoryStore

    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError):
        WebUIHistoryStore(tmp_path / "link").append_records("websocket:c", [{"event": "user", "text": "x"}])
    assert list(real.iterdir()) == []
    store = WebUIHistoryStore(tmp_path / "webui")
    store.append_records("websocket:c", [{"event": "user", "text": "x"}])
    assert store.directory.stat().st_mode & 0o777 == 0o700
    assert store.path("websocket:c").stat().st_mode & 0o777 == 0o600


def test_symlinked_history_file_is_refused_with_value_error(tmp_path) -> None:
    from nanoclaw.webui.history import WebUIHistoryStore

    store = WebUIHistoryStore(tmp_path / "webui")
    store.directory.mkdir(parents=True)
    target = tmp_path / "elsewhere.jsonl"
    target.write_text("")
    store.path("websocket:c").symlink_to(target)
    with pytest.raises(ValueError):
        store.append_records("websocket:c", [{"event": "user", "text": "x"}])


def test_history_open_does_not_follow_a_symlink_swapped_in_after_the_check(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from nanoclaw.webui.history import WebUIHistoryStore

    store = WebUIHistoryStore(tmp_path / "webui")
    store.directory.mkdir(parents=True)
    target = tmp_path / "elsewhere.jsonl"
    target.write_text("")
    store.path("websocket:c").symlink_to(target)
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)  # the race: the check saw a file
    with pytest.raises(OSError):
        store.append_records("websocket:c", [{"event": "user", "text": "x"}])
    assert target.read_text() == ""


def test_oversized_fact_is_rejected_before_writing(tmp_path) -> None:
    from nanoclaw.session.display_history import MAX_HISTORY_FACT_BYTES
    from nanoclaw.webui.history import WebUIHistoryStore

    store = WebUIHistoryStore(tmp_path / "webui")
    with pytest.raises(ValueError):
        store.append_records("websocket:c", [{"event": "user", "text": "x" * (MAX_HISTORY_FACT_BYTES + 1)}])
    assert not store.exists("websocket:c")


def test_appending_nothing_creates_nothing(tmp_path) -> None:
    from nanoclaw.webui.history import WebUIHistoryStore

    store = WebUIHistoryStore(tmp_path / "webui")
    store.append("websocket:c", [])
    assert not store.directory.exists()


def test_history_cache_sees_size_only_and_mtime_only_changes(tmp_path) -> None:
    import os

    from nanoclaw.webui.history import WebUIHistoryStore

    store = WebUIHistoryStore(tmp_path / "webui")
    key = "websocket:c"
    store.append_records(key, [{"event": "user", "text": "aaaa", "created_at_ms": 1}])
    path = store.path(key)
    stat = path.stat()
    assert [f.content["text"] for f in store.read_facts(key)[0]] == ["aaaa"]
    # A size change with the old mtime.
    store.append_records(key, [{"event": "user", "text": "bbbb", "created_at_ms": 1}])
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert [f.content["text"] for f in store.read_facts(key)[0]] == ["aaaa", "bbbb"]
    # Same size, new mtime.
    stat = path.stat()
    path.write_text(path.read_text().replace("bbbb", "cccc"))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert path.stat().st_size == stat.st_size
    assert [f.content["text"] for f in store.read_facts(key)[0]] == ["aaaa", "cccc"]


def test_history_cache_is_bounded_and_returns_copies(tmp_path) -> None:
    from nanoclaw.webui.history import WebUIHistoryStore

    store = WebUIHistoryStore(tmp_path / "webui")
    for n in range(40):
        store.append_records(f"websocket:c{n}", [{"event": "user", "text": str(n)}])
        store.read_facts(f"websocket:c{n}")
    assert len(store._cache) <= 32
    facts, _ = store.read_facts("websocket:c39")
    facts.clear()
    assert len(store.read_facts("websocket:c39")[0]) == 1


def test_blank_history_lines_are_not_reported_as_unreadable(tmp_path) -> None:
    from loguru import logger

    from nanoclaw.webui.history import WebUIHistoryStore

    store = WebUIHistoryStore(tmp_path / "webui")
    store.append_records("websocket:c", [{"event": "user", "text": "x"}])
    with store.path("websocket:c").open("a") as handle:
        handle.write("\n   \n")
    warnings: list[str] = []
    logger.enable("nanoclaw")
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        assert len(store.read_facts("websocket:c")[0]) == 1
        assert warnings == []
        with store.path("websocket:c").open("a") as handle:
            handle.write("{broken\n")
        assert len(store.read_facts("websocket:c")[0]) == 1
    finally:
        logger.remove(sink)
        logger.disable("nanoclaw")
    assert len(warnings) == 1 and "Skipped 1 unreadable" in warnings[0]


def _facts(*kinds):
    from nanoclaw.session.display_history import HistoryFact

    return [HistoryFact(kind, {"text": f"{n}"}) for n, kind in enumerate(kinds)]


def test_history_reader_bounded_reads_and_navigation() -> None:
    from nanoclaw.session.display_history import HistoryKind as K
    from nanoclaw.session.display_history import HistoryReader

    facts = _facts(K.INPUT, K.PROGRESS, K.FILE_ACTIVITY, K.ANSWER, K.COMPLETION)
    reader = HistoryReader("websocket:c", facts)
    assert reader.read_entries(-1) == []
    size = len(facts[0].encode().encode("utf-8"))  # exactly the input fact
    # Two records at most: the first input and the completion, in file order.
    picked = reader.read_entries(0, max_records=2, max_bytes=10**6)
    assert [position for position, _ in picked] == [0, 4]
    # A byte budget for about one fact keeps only the input.
    assert [p for p, _ in reader.read_entries(0, max_records=5, max_bytes=size)] == [0]
    assert [fact.kind for _, _, fact in reader.navigation()] == [K.INPUT, K.ANSWER, K.COMPLETION]
    # Selected rows always come back in file order.
    long_turn = HistoryReader("websocket:c", _facts(K.INPUT, *[K.PROGRESS] * 7, K.COMPLETION))
    assert [p for p, _ in long_turn.read_entries(0, max_records=3, max_bytes=10**6)] == [0, 7, 8]


def test_turn_end_usage_drops_negative_and_boolean_counts(history) -> None:
    key = "websocket:u"
    history.append(key, {"event": "user", "chat_id": "u", "text": "q"})
    history.append(key, {"event": "turn_end", "chat_id": "u", "usage": {"a": -1, "b": True, "c": 3}})
    out = history.build(key)
    [end] = [e for e in out["events"] if e["event"] == "turn_end"]
    assert end["usage"] == {"c": 3}


def test_delta_compaction_keys_streams_by_turn_phase() -> None:
    from nanoclaw.webui.transcript import _compact_completed_stream_deltas

    turn = [
        {"event": "delta", "text": "side", "turn_id": "t", "turn_phase": "activity", "stream_id": "s"},
        {"event": "stream_end", "text": "", "turn_id": "t", "turn_phase": "answer", "stream_id": "s"},
        {"event": "turn_end", "turn_id": "t"},
    ]
    compacted, dropped = _compact_completed_stream_deltas(turn)
    assert dropped == 0 and compacted == turn


def test_page_cursor_rejects_negative_and_boolean_turns() -> None:
    import base64
    import json

    from nanoclaw.webui.transcript import decode_transcript_cursor

    def encode(value) -> str:
        return base64.urlsafe_b64encode(json.dumps({"before_turn": value}).encode()).decode().rstrip("=")

    assert decode_transcript_cursor(encode(-1)) is None
    assert decode_transcript_cursor(encode(True)) is None
    assert decode_transcript_cursor(encode(0)) == 0


def test_page_limit_is_clamped_and_defaults_to_160(history) -> None:
    from nanoclaw.webui.transcript import TranscriptReplayStats

    key = "websocket:lim"
    history.append(key, {"event": "user", "chat_id": "lim", "text": "q"})
    for limit, expected in ((None, 160), (0, 1), (5000, 1000)):
        stats = TranscriptReplayStats()
        history.build(key, limit=limit, stats=stats)
        assert stats.effective_limit == expected


def test_page_stops_once_the_limit_is_reached_and_first_turn_has_no_before_cursor(history) -> None:
    key = "websocket:pg"
    for n in range(3):
        history.append(key, {"event": "user", "chat_id": "pg", "text": f"q{n}"})
        history.append(key, {"event": "turn_end", "chat_id": "pg"})
    latest = history.build(key, limit=2)
    assert [e.get("text") for e in latest["events"] if e["event"] == "user_message"] == ["q2"]
    oldest = history.build(key, limit=2, before=history.build(key, limit=4)["page"]["before_cursor"])
    assert oldest["page"]["before_cursor"] is None and oldest["page"]["has_more_before"] is False


def test_normalize_turn_id_replaces_invalid_ids() -> None:
    from nanoclaw.webui.transcript import normalize_webui_turn_id

    assert normalize_webui_turn_id(" ok-1 ") == "ok-1"
    replaced = normalize_webui_turn_id("bad id!")
    assert replaced != "bad id!" and len(replaced) == 36


async def test_recorder_reports_append_failures_applies_overrides_and_restarts_sequences(tmp_path) -> None:
    from nanoclaw.webui.history import WebUIHistoryStore
    from nanoclaw.webui.transcript import WebUITranscriptRecorder

    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    broken = WebUITranscriptRecorder(WebUIHistoryStore(tmp_path / "link"))
    assert await broken.append("c", {"event": "user", "text": "x"}, metadata=None) is False

    store = WebUIHistoryStore(tmp_path / "webui")
    recorder = WebUITranscriptRecorder(store)
    meta = {"webui_turn_id": "t"}
    assert await recorder.prepare_and_append(
        "c", {"event": "message", "chat_id": "c", "text": "wire"}, metadata=meta, phase="answer",
        transcript_overrides={"text": "full"},
    )
    end: dict = {"event": "turn_end", "chat_id": "c"}
    recorder.prepare_event("c", end, metadata=meta, phase="complete")
    nxt: dict = {"event": "user", "chat_id": "c"}
    recorder.prepare_event("c", nxt, metadata=meta, phase="user")
    assert (end["turn_seq"], nxt["turn_seq"]) == (2, 1)
    [fact] = store.read_facts("websocket:c")[0]
    assert fact.content["text"] == "full"


def test_user_event_keeps_only_local_paths_that_are_attached() -> None:
    from nanoclaw.webui.transcript import build_user_transcript_event

    event = build_user_transcript_event("c", "x", media_paths=["/a.png"], local_media_paths=["/a.png", "/b.png"])
    assert event is not None and event["local_media_paths"] == ["/a.png"]


def test_session_display_facts_skip_hidden_and_blank_assistant_messages() -> None:
    from types import SimpleNamespace

    from nanoclaw.session.history_visibility import HIDDEN_HISTORY_META
    from nanoclaw.webui.transcript import session_display_facts

    session = SimpleNamespace(key="websocket:c", messages=[
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "secret", HIDDEN_HISTORY_META: True},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": "a"},
    ])
    assert [f.content["text"] for f in session_display_facts(session)] == ["q", "a"]


def _incomplete_turn(history, key, chat, *events, created_at_ms=1000):
    history.append(key, {"event": "user", "chat_id": chat, "text": "repeat", "turn_id": "t1"})
    for event in events:
        history.append(key, {"chat_id": chat, "turn_id": "t1", **event})
    history.append(key, {
        "event": "turn_end", "chat_id": chat, "turn_id": "t1",
        "transcript_incomplete": True, "created_at_ms": created_at_ms,
    })


def test_recovery_ignores_unanswered_session_turns_and_spaces_answer_times(history) -> None:
    key = "websocket:rec"
    _incomplete_turn(history, key, "rec")
    out = history.build(key, session_messages=[
        {"role": "user", "content": "repeat"},  # never answered: not a candidate
        {"role": "user", "content": "repeat"},
        {"role": "assistant", "content": "one"},
        {"role": "assistant", "content": "two"},
    ])
    answers = [e for e in out["events"] if e["event"] == "message"]
    assert [e["text"] for e in answers] == ["one", "two"]
    assert [e["created_at_ms"] for e in answers] == [998, 999]
    assert out["completed_turn_ids"] == ["t1"]


def test_recovery_needs_exactly_one_user_row(history) -> None:
    key = "websocket:rec2"
    _incomplete_turn(history, key, "rec2", {"event": "user", "text": "again"})
    out = history.build(key, session_messages=[
        {"role": "user", "content": "repeat"}, {"role": "assistant", "content": "one"},
    ])
    assert not [e for e in out["events"] if e["event"] == "message"]
    assert out["has_pending_tool_calls"] is True


def test_pending_tool_calls_rules() -> None:
    from nanoclaw.webui.transcript import has_pending_tool_calls

    done = [{"event": "user", "turn_id": "t"}, {"event": "turn_end", "turn_id": "t"}]
    running = [{"event": "user", "turn_id": "t"}, {"event": "message", "turn_id": "t"}]
    assert has_pending_tool_calls(done, active_turn_started_at=1.0, active_turn_id="t") is False
    assert has_pending_tool_calls(running, active_turn_started_at=1.0, active_turn_id="t") is True
    assert has_pending_tool_calls(
        done, active_turn_started_at=1.0, active_turn_id="t", active_turn_transcript_persistence_failed=True,
    ) is True
    assert has_pending_tool_calls(done, active_turn_started_at=1.0) is True
    # Without an active turn: the tail decides.
    assert has_pending_tool_calls([{"event": "user"}]) is False
    assert has_pending_tool_calls([{"event": "file_edit"}]) is True
    assert has_pending_tool_calls([{"event": "message", "kind": "reasoning"}]) is True
    assert has_pending_tool_calls([{"event": "message"}]) is False


def test_completed_turn_ids_are_unique() -> None:
    from nanoclaw.webui.transcript import completed_turn_ids

    assert completed_turn_ids([{"event": "turn_end", "turn_id": "t"}] * 2) == ["t"]


def test_first_fork_marker_is_the_boundary(history) -> None:
    key = "websocket:forks"
    history.append(key, {"event": "user", "chat_id": "forks", "text": "one"})
    history.append(key, {"event": "fork_marker"})
    history.append(key, {"event": "user", "chat_id": "forks", "text": "two"})
    history.append(key, {"event": "fork_marker"})
    history.append(key, {"event": "user", "chat_id": "forks", "text": "three"})
    assert history.build(key)["fork_boundary_event_index"] == 1


def test_replay_validates_compaction_phase_and_turn_end_failure_fields(history) -> None:
    key = "websocket:replay"
    history.append(key, {"event": "user", "chat_id": "replay", "text": "q"})
    history.append(key, {"event": "context_compaction", "chat_id": "replay", "compaction_id": "k", "phase": "bogus"})
    history.append(key, {
        "event": "turn_end", "chat_id": "replay", "outcome": "failed", "failure_kind": "model",
        "failure_error_kind": "server", "failure_attempts": True,
    })
    events = history.build(key)["events"]
    assert not [e for e in events if e["event"] == "context_compaction"]
    [end] = [e for e in events if e["event"] == "turn_end"]
    assert end["failure_error_kind"] == "server"
    assert "failure_attempts" not in end


def _session(history, key, messages):
    session = history.sessions.get_or_create(key)
    session.messages = messages
    history.sessions.save(session)
    return session


def test_session_thread_skips_hidden_rows_entirely(history) -> None:
    from nanoclaw.session.history_visibility import HIDDEN_HISTORY_META
    from nanoclaw.webui.transcript import build_session_thread_response

    session = _session(history, "websocket:hid", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "reasoning_content": "private", HIDDEN_HISTORY_META: True},
        {"role": "assistant", "content": "a"},
    ])
    events = build_session_thread_response(session)["events"]
    assert [e.get("text") for e in events if e["event"] != "turn_end"] == ["q", "a"]


def test_backfill_closes_each_turn_once(history) -> None:
    from nanoclaw.session.display_history import HistoryKind
    from nanoclaw.webui.transcript import session_backfill_history

    session = _session(history, "websocket:ends", [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": ""},  # no row of its own
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ])
    facts = session_backfill_history(session).all_facts()
    assert [f.kind for f in facts].count(HistoryKind.COMPLETION) == 2


def test_backfilled_payload_of_an_active_chat_leaves_the_last_turn_open_and_has_a_revision(history) -> None:
    from nanoclaw.webui.transcript import build_webui_thread_payload

    key = "websocket:bf"
    _session(history, key, [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ])
    idle = build_webui_thread_payload(key, store=history.store, sessions=history.sessions, revision_variant={})
    assert idle["completed_turn_ids"] == [f"session:{key}:0", f"session:{key}:2"]
    assert idle["revision"]
    active = build_webui_thread_payload(
        key, store=history.store, sessions=history.sessions, active_turn_started_at=1.0, active_turn_id="live",
    )
    assert active["completed_turn_ids"] == [f"session:{key}:0"]


def test_recorded_log_revision_follows_the_session_file(history) -> None:
    import os

    from nanoclaw.webui.transcript import build_webui_thread_payload

    key = "websocket:rv"
    session = _session(history, key, [{"role": "user", "content": "q"}])
    history.append(key, {"event": "user", "chat_id": "rv", "text": "q"})
    first = build_webui_thread_payload(key, store=history.store, sessions=history.sessions, revision_variant={})
    session.messages.append({"role": "assistant", "content": "longer answer"})
    history.sessions.save(session)
    path = history.sessions.get_session_path(key)
    os.utime(path, ns=(0, path.stat().st_mtime_ns + 1_000_000))
    second = build_webui_thread_payload(key, store=history.store, sessions=history.sessions, revision_variant={})
    assert first["revision"] != second["revision"]


_HANDLE = "handle_" + "0" * 32


def test_prompt_index_dedups_users_by_turn_and_message_and_trims_previews(history) -> None:
    key = "websocket:pi"
    long = "  " + "x" * 400
    history.append(key, {"event": "user", "chat_id": "pi", "text": long, "turn_id": "t1"})
    history.append(key, {"event": "user", "chat_id": "pi", "text": "same turn again", "turn_id": "t1"})
    provenance = {"message_id": "m1", "session": {"id": _HANDLE, "name": "other"}}
    history.append(key, {"event": "user", "chat_id": "pi", "text": "relayed", "session_message": provenance})
    history.append(key, {"event": "user", "chat_id": "pi", "text": "relayed twice", "session_message": provenance})
    history.append(key, {"event": "turn_end", "chat_id": "pi"})
    prompts = history.prompts(key)
    assert [p["preview"] for p in prompts] == ["x" * 320, "relayed"]


def test_prompt_index_response_for_an_unchanged_revision_sends_nothing(history) -> None:
    key = "websocket:pr"
    history.append(key, {"event": "user", "chat_id": "pr", "text": "q"})
    first = history.prompt_response(key, None)
    again = history.prompt_response(key, first["revision"])
    assert again == {"prompts": (), "replaceFrom": 1, "revision": first["revision"]}


def test_prompt_snapshot_cache_is_bounded(history) -> None:
    from nanoclaw.webui import transcript

    transcript._PROMPT_SNAPSHOTS.clear()
    for n in range(12):
        key = f"websocket:cache{n}"
        history.append(key, {"event": "user", "chat_id": f"cache{n}", "text": "q"})
        history.prompts(key)
    assert len(transcript._PROMPT_SNAPSHOTS) <= 8


# --- area 4: session list, titles, visibility, deletion -------------------------


def _saved(history, key, messages=(), *, metadata=None, updated_at=None):
    session = history.sessions.get_or_create(key)
    session.messages = list(messages)
    session.metadata.update(metadata or {})
    if updated_at is not None:
        session.updated_at = updated_at
    history.sessions.save(session)
    return session


def _rows(history):
    from nanoclaw.webui.session_list_index import list_webui_sessions

    return {row["key"]: row for row in list_webui_sessions(history.sessions, history.store)}


def test_preview_text_sanitizes_skips_non_text_blocks_and_truncates() -> None:
    from nanoclaw.webui.session_list_index import message_preview_text, scrub_subagent_announce_body

    assert message_preview_text({"content": "[Message Time: 2026-01-01 10:00]\nhello"}) == "hello"
    assert message_preview_text({"content": [
        {"type": "reasoning", "text": "secret"}, {"type": "text", "text": "shown"},
    ]}) == "shown"
    long = message_preview_text({"content": "word " * 60})
    assert len(long) == 120 and long.endswith("…")
    announce = "[Subagent 'x' done]\n\nTask: t\n\nResult:\n" + "y" * 1000 + "\n\nSummarize this naturally"
    scrubbed = scrub_subagent_announce_body(announce)
    body = scrubbed.split("\n\n", 1)[1]
    assert len(body) == 800 and body.endswith("…")


def test_recovery_state_is_sanitized() -> None:
    from nanoclaw.webui.session_list_index import recovery_state_from_metadata

    assert recovery_state_from_metadata({"webui_recovery": {"status": "bogus", "recovery_id": "r"}}) is None
    assert recovery_state_from_metadata({"webui_recovery": {
        "status": "failed", "recovery_id": "r", "attempts": -1, "can_continue": False,
        "failure_error_kind": "secret-provider-detail",
    }}) == {"status": "failed", "recovery_id": "r", "can_continue": False}


def test_session_preview_and_activity_rules(history) -> None:
    _saved(history, "websocket:users", [
        {"role": "assistant", "content": "greeting"},
        {"role": "user", "content": "first question", "timestamp": "2026-01-01T10:00:00"},
        {"role": "user", "content": "second question", "timestamp": "2026-01-01T11:00:00"},
        {"role": "assistant", "content": "answer", "timestamp": "2026-01-01T12:00:00"},
        {"role": "tool", "content": "out", "timestamp": "2026-01-01T13:00:00"},
        {"role": "assistant", "content": "hidden", "timestamp": "2026-01-01T14:00:00", "_hidden_history": True},
    ])
    _saved(history, "websocket:answers", [
        {"role": "assistant", "content": "first answer"},
        {"role": "assistant", "content": "second answer"},
    ])
    rows = _rows(history)
    assert rows["websocket:users"]["preview"] == "first question"
    assert rows["websocket:users"]["updated_at"] == "2026-01-01T12:00:00"
    assert rows["websocket:answers"]["preview"] == "first answer"


def test_stored_message_time_wins_over_older_event_log_activity(history) -> None:
    from datetime import datetime

    key = "websocket:act"
    _saved(history, key, [{"role": "user", "content": "q", "timestamp": "2026-01-02T00:00:00"}])
    old = int(datetime(2026, 1, 1).timestamp() * 1000)
    history.append(key, {"event": "user", "chat_id": "act", "text": "q", "created_at_ms": old})
    assert _rows(history)[key]["updated_at"] == "2026-01-02T00:00:00"


def test_history_head_uses_latest_time_first_input_and_answer_segments(tmp_path) -> None:
    from nanoclaw.webui.history import WebUIHistoryStore
    from nanoclaw.webui.session_list_index import history_head

    store = WebUIHistoryStore(tmp_path / "webui")
    store.append_records("websocket:a", [
        {"event": "stream_end", "text": "streamed answer", "created_at_ms": 5},
        {"event": "turn_end", "created_at_ms": 3},
    ])
    assert history_head(store, "websocket:a") == (5, "streamed answer")
    store.append_records("websocket:b", [
        {"event": "user", "text": "one"}, {"event": "user", "text": "two"},
    ])
    assert history_head(store, "websocket:b")[1] == "one"


def test_session_metadata_rows_are_newest_first(history) -> None:
    from datetime import datetime

    from nanoclaw.webui.session_list_index import list_session_metadata

    _saved(history, "websocket:a-old", updated_at=datetime(2026, 1, 1))
    _saved(history, "websocket:b-new", updated_at=datetime(2026, 3, 1))
    _saved(history, "websocket:c-mid", updated_at=datetime(2026, 2, 1))
    keys = [row["key"] for row in list_session_metadata(history.sessions, history.store)]
    assert keys == ["websocket:b-new", "websocket:c-mid", "websocket:a-old"]


def test_oversized_workspace_scope_is_not_indexed(history) -> None:
    _saved(history, "websocket:scope", metadata={"workspace_scope": {"path": "x" * 5000}})
    row = _rows(history)["websocket:scope"]
    assert (row["_workspace_scope_present"], row["_workspace_scope_value"]) == (True, False)


def test_delete_removes_descendant_event_logs(history) -> None:
    from nanoclaw.webui.session_list_index import delete_webui_session

    _saved(history, "websocket:p")
    _saved(history, "subagent:p:1", metadata={"session_type": "subagent", "parent_session_key": "websocket:p"})
    history.store.append_records("subagent:p:1", [{"event": "user", "text": "x"}])
    assert delete_webui_session(history.sessions, history.store, "websocket:p") is True
    assert not history.store.exists("subagent:p:1")


def test_delete_session_family_survives_cycles_and_a_missing_root(history) -> None:
    manager = history.sessions
    # a -> {b, c, d}; a's own parent is the child walked first (the last one
    # listed), so the walk meets a again while siblings are still pending.
    for child in ("b", "c", "d"):
        _saved(history, f"websocket:{child}", metadata={"parent_session_key": "websocket:a"})
    children = [h["key"] for h in manager.list_session_headers() if h["metadata"].get("parent_session_key")]
    _saved(history, "websocket:a", metadata={"parent_session_key": children[-1]})
    assert sorted(manager.delete_session_family("websocket:a")) == [
        "websocket:a", "websocket:b", "websocket:c", "websocket:d",
    ]
    # A root without a file still takes its children with it.
    _saved(history, "websocket:child", metadata={"parent_session_key": "websocket:gone"})
    assert manager.delete_session_family("websocket:gone") == ["websocket:child"]
    assert manager.delete_session("websocket:gone") is False


def test_automation_kind_is_an_allowlist() -> None:
    from nanoclaw.session.history_visibility import is_automation_kind

    assert is_automation_kind("cron") and not is_automation_kind("other")


class _Titles:
    def __init__(self, content, on_call=None) -> None:
        self.content, self.on_call, self.calls = content, on_call, []

    async def chat_stream_with_retry(self, messages, **kwargs):
        from types import SimpleNamespace

        self.calls.append({"messages": messages, **kwargs})
        if self.on_call:
            self.on_call()
        return SimpleNamespace(content=self.content, finish_reason="stop")


def _webui(history, key, messages, **metadata):
    return _saved(history, key, messages, metadata={"webui": True, **metadata})


async def _title(history, key, provider):
    from nanoclaw.session.webui_turns import maybe_generate_webui_title

    return await maybe_generate_webui_title(sessions=history.sessions, session_key=key, provider=provider, model="m")


def test_mark_webui_session_needs_literal_true() -> None:
    from types import SimpleNamespace

    from nanoclaw.session.webui_turns import mark_webui_session

    session = SimpleNamespace(metadata={})
    assert mark_webui_session(session, {"webui": "yes"}) is False and session.metadata == {}


async def test_title_prompt_uses_first_visible_user_and_assistant_and_bounded_tokens(history) -> None:
    _webui(history, "websocket:t", [
        {"role": "user", "content": "/new", "_command": True},
        {"role": "user", "content": "secret", "_hidden_history": True},
        {"role": "user", "content": "first question"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "the answer"},
    ])
    provider = _Titles("A title")
    assert await _title(history, "websocket:t", provider) is True
    [call] = provider.calls
    prompt = call["messages"][1]["content"]
    assert "User: first question\nAssistant: the answer" in prompt
    assert "/new" not in prompt and "secret" not in prompt
    assert call["max_tokens"] == 96


async def test_title_needs_a_user_message_and_a_usable_reply(history) -> None:
    _webui(history, "websocket:none", [{"role": "assistant", "content": "hello"}])
    provider = _Titles("A title")
    assert await _title(history, "websocket:none", provider) is False and provider.calls == []
    _webui(history, "websocket:err", [{"role": "user", "content": "q"}])
    assert await _title(history, "websocket:err", _Titles("Error: provider down")) is False
    assert "title" not in history.sessions.get_existing("websocket:err").metadata


async def test_title_existing_and_user_edited_titles(history) -> None:
    _webui(history, "websocket:old", [{"role": "user", "content": "q"}], title="Title: Old one.")
    provider = _Titles("New")
    assert await _title(history, "websocket:old", provider) is False
    assert history.sessions.read_session_metadata("websocket:old")["metadata"]["title"] == "Old one"
    _webui(history, "websocket:ed", [{"role": "user", "content": "q"}], title="", title_user_edited=True)
    assert await _title(history, "websocket:ed", provider) is False
    assert provider.calls == []


async def test_title_is_not_written_over_a_change_made_while_generating(history) -> None:
    for change in ({"title_user_edited": True}, {"title": "Typed meanwhile"}):
        key = f"websocket:race{len(change)}{next(iter(change))}"
        session = _webui(history, key, [{"role": "user", "content": "q"}])
        provider = _Titles("Generated", on_call=lambda s=session, c=change: s.metadata.update(c))
        assert await _title(history, key, provider) is False
        assert session.metadata.get("title") != "Generated"


async def test_title_after_turn_targets_the_chat_session_with_the_latest_exchange(history) -> None:
    from nanoclaw.session.webui_turns import maybe_generate_webui_title_after_turn

    _webui(history, "unified:main", [
        {"role": "user", "content": "old question"}, {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"}, {"role": "assistant", "content": "new answer"},
    ])
    provider = _Titles("Chat title")
    kwargs = dict(chat_id="c", metadata={"webui": True}, sessions=history.sessions,
                  session_key="unified:main", provider=provider, model="m")
    assert await maybe_generate_webui_title_after_turn(channel="telegram", **kwargs) is False
    assert await maybe_generate_webui_title_after_turn(channel="websocket", **kwargs) is True
    assert "User: new question\nAssistant: new answer" in provider.calls[0]["messages"][1]["content"]
    assert history.sessions.get_existing("websocket:c").metadata["title"] == "Chat title"
    assert "title" not in history.sessions.get_existing("unified:main").metadata
