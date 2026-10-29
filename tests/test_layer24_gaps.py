"""Focused tests closing gaps found by the layer 24 mutation checks
(Telegram channel, channel dispatch, pairing)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanoclaw.bus.events import OutboundMessage
from nanoclaw.bus.outbound_events import ProgressEvent
from nanoclaw.bus.queue import MessageBus
from nanoclaw.channels.telegram.runtime import (
    TelegramChannel,
    TelegramConfig,
    _markdown_to_telegram_html,
    _render_table_box,
    _split_telegram_markdown,
    _split_telegram_markdown_html,
    _split_telegram_markdown_html_chunks,
    _strip_md_block,
)


class _Bot:
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []
        self.edits: list[dict] = []
        self.reactions: list[dict] = []
        self.api_requests: list[tuple[str, dict]] = []

    async def send_message(self, **kwargs):
        self.sent_messages.append(kwargs)
        return SimpleNamespace(message_id=len(self.sent_messages))

    async def edit_message_text(self, **kwargs):
        self.edits.append(kwargs)

    async def set_message_reaction(self, **kwargs):
        self.reactions.append(kwargs)

    async def send_chat_action(self, **kwargs):
        pass

    async def do_api_request(self, method, api_kwargs=None):
        self.api_requests.append((method, api_kwargs or {}))
        return True


def _channel(**cfg) -> tuple[TelegramChannel, _Bot]:
    channel = TelegramChannel(TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], **cfg), MessageBus())
    bot = _Bot()
    channel._app = SimpleNamespace(bot=bot)  # type: ignore[assignment]
    channel._app_ready.set()
    return channel, bot


# -- Markdown rendering / splitting ---------------------------------------------------


def test_backtick_info_string_line_is_not_a_fence_opener() -> None:
    # CommonMark: a ``` opener's info string cannot hold a backtick, so the first
    # line is inline code and the later ``` pair is the block.
    text = "```a`b```\n**x**\n```\ny\n```"
    assert _markdown_to_telegram_html(text) == (
        "<pre><code>a`b</code></pre>\n<b>x</b>\n<pre><code>y\n</code></pre>"
    )


def test_split_prefers_word_boundary_without_newlines() -> None:
    assert _split_telegram_markdown("aaa bbb ccc", 6) == ["aaa", "bbb", "ccc"]


def test_split_reopened_code_block_has_no_blank_first_line() -> None:
    content = "```python\n" + ("print(1)\n" * 6) + "```\nDone"
    chunks = _split_telegram_markdown(content, 40)
    assert chunks[1] == "```python\nprint(1)\nprint(1)\nprint(1)\n```"


def test_unclosed_fence_line_is_kept() -> None:
    assert _markdown_to_telegram_html("```python\nhello") == "```python\nhello"


def test_strip_md_block_plain_preview() -> None:
    assert _strip_md_block("> quoted\n# Head\n- item\n[l](http://x)\n_it_") == "quoted\nHead\n• item\nl\nit"


def test_table_box_uses_display_width_and_strips_cell_markdown() -> None:
    box = _render_table_box(["| 中文 | x |", "|---|---|", "| ab | **y** |"])
    assert box == "中文  x\n────  ─\nab    y"


# -- send ---------------------------------------------------------------------------------


async def test_tool_hint_is_escaped_expandable_blockquote() -> None:
    channel, bot = _channel()
    await channel.send(
        OutboundMessage(channel="telegram", chat_id="5", content="read a<b>", event=ProgressEvent(tool_hint=True))
    )
    assert bot.sent_messages[-1]["text"] == "<blockquote expandable>read a&lt;b&gt;</blockquote>"
    assert bot.sent_messages[-1]["parse_mode"] == "HTML"


def test_table_without_separator_stays_text() -> None:
    assert _markdown_to_telegram_html("| a | b |\n| c | d |") == "| a | b |\n| c | d |"


def test_indented_table_renders_as_box() -> None:
    html = _markdown_to_telegram_html("  | a | b |\n  |---|---|\n  | c | d |")
    assert html == "<pre><code>a  b\n─  ─\nc  d</code></pre>"


def test_table_short_row_is_padded() -> None:
    assert _render_table_box(["| a | b |", "|---|---|", "| c |"]).split("\n")[-1] == "c   "


def test_inline_markdown_rendering() -> None:
    html = _markdown_to_telegram_html("`a<b` > q __u__ a_b_c ~~s~~")
    assert html == "<code>a&lt;b</code> &gt; q <b>u</b> a_b_c <s>s</s>"
    assert _markdown_to_telegram_html("> q") == "q"


def test_html_chunks_exact_limit_is_one_chunk() -> None:
    assert _split_telegram_markdown_html_chunks("abcdefghij", 10) == [("abcdefghij", "abcdefghij")]


def test_html_chunks_resplit_keeps_safety_margin() -> None:
    content = "word a <b> a **x** **x** **x** ab&cd **x** & a **x** a"
    assert _split_telegram_markdown_html_chunks(content, 77) == [
        ("word a <b> a **x** **x** **x** ab&cd", "word a &lt;b&gt; a <b>x</b> <b>x</b> <b>x</b> ab&amp;cd"),
        ("**x** & a **x** a", "<b>x</b> &amp; a <b>x</b> a"),
    ]


def test_html_chunks_resplit_keeps_order() -> None:
    content = ("& " * 1000).strip() + "\n" + "b" * 2500
    chunks = _split_telegram_markdown_html_chunks(content, 4096)
    assert [len(html) for _, html in chunks] == [4067, 1931, 2500]
    assert chunks[-1][0] == "b" * 2500


def test_html_split_uses_raw_4000_margin() -> None:
    content = " ".join(["word"] * 810)  # 4049 chars: under 4096, over 4000
    assert len(_split_telegram_markdown_html(content, 4096)) == 2


# -- allowFrom / start / supervisor ----------------------------------------------------


def test_legacy_sender_with_extra_pipes_is_rejected() -> None:
    channel = TelegramChannel(TelegramConfig(allow_from=["1"]), MessageBus())
    assert channel.is_allowed("1|alice") is True
    assert channel.is_allowed("1|a|b") is False


async def test_start_without_token_returns_without_running() -> None:
    channel = TelegramChannel(TelegramConfig(enabled=True, token=""), MessageBus())
    channel._start_app = AsyncMock(side_effect=AssertionError("must not start"))  # type: ignore[method-assign]
    await channel.start()
    assert channel._running is False
    channel._start_app.assert_not_called()


async def test_startup_backoff_doubles_and_resets_after_success() -> None:
    from telegram.error import NetworkError

    channel = TelegramChannel(TelegramConfig(enabled=True, token="1:a"), MessageBus())
    script = ["fail", "fail", "ok", "fail", "stop"]
    idles: list[float] = []

    async def start_app() -> None:
        step = script.pop(0)
        if step == "fail":
            raise NetworkError("connect failed")
        if step == "stop":
            channel._running = False

    async def idle(seconds: float) -> None:
        idles.append(seconds)

    async def stalled() -> bool:
        return True

    channel._start_app = start_app  # type: ignore[method-assign]
    channel._idle = idle  # type: ignore[method-assign]
    channel._watch_polling = stalled  # type: ignore[method-assign]
    channel._teardown_app = AsyncMock()  # type: ignore[method-assign]
    await channel.start()
    assert idles == [5.0, 10.0, 5.0]
    assert script == []


def test_asyncio_timeout_is_a_transient_startup_error() -> None:
    assert TelegramChannel._is_transient_startup_error(asyncio.TimeoutError()) is True
    assert TelegramChannel._is_transient_startup_error(ValueError("bad proxy")) is False


async def _started_app(monkeypatch, **cfg):
    from test_telegram_channel import _FakeApp, _FakeBuilder, _FakeHTTPXRequest

    channel = TelegramChannel(TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], **cfg), MessageBus())
    app = _FakeApp(lambda: setattr(channel, "_running", False))
    monkeypatch.setattr("nanoclaw.channels.telegram.runtime.HTTPXRequest", _FakeHTTPXRequest)
    monkeypatch.setattr(
        "nanoclaw.channels.telegram.runtime.Application", SimpleNamespace(builder=lambda: _FakeBuilder(app))
    )
    await channel.start()
    return channel, app


async def test_start_handler_accepts_bot_suffix_and_inline_keyboards(monkeypatch) -> None:
    from telegram.ext import CallbackQueryHandler

    channel, app = await _started_app(monkeypatch, inline_keyboards=True)
    start = next(h for h in app.handlers if getattr(h, "callback", None) == channel._on_start)
    assert start.filters.pattern.match("/start@nanobot_test")
    assert any(isinstance(h, CallbackQueryHandler) for h in app.handlers)
    assert app.updater.start_polling_kwargs["allowed_updates"] == ["message", "callback_query"]


async def test_start_records_bot_identity(monkeypatch) -> None:
    channel, app = await _started_app(monkeypatch)
    assert (channel._bot_user_id, channel._bot_username) == (999, "nanobot_test")
    assert await channel._ensure_bot_identity() == (999, "nanobot_test")
    assert app.bot.get_me_calls == 1


async def test_command_menu_failure_does_not_stop_startup(monkeypatch) -> None:
    from test_telegram_channel import _FakeBot

    async def fail(self, commands) -> None:
        raise RuntimeError("menu rejected")

    monkeypatch.setattr(_FakeBot, "set_my_commands", fail)
    _, app = await _started_app(monkeypatch)
    assert app.updater.start_polling_kwargs is not None


async def test_webhook_url_and_secret_are_stripped(monkeypatch) -> None:
    _, app = await _started_app(
        monkeypatch,
        mode="webhook",
        webhook_url=" https://example.com/telegram ",
        webhook_secret_token=" secret-token ",
    )
    assert app.updater.start_webhook_kwargs["webhook_url"] == "https://example.com/telegram"
    assert app.updater.start_webhook_kwargs["secret_token"] == "secret-token"


async def test_polling_start_stamps_liveness(monkeypatch) -> None:
    import time

    channel, _ = await _started_app(monkeypatch)
    assert time.monotonic() - channel._last_poll_ok < 5


async def test_send_waits_out_a_short_rebuild() -> None:
    channel, _ = _channel()
    app = channel._app
    channel._running = True
    channel._app_ready.clear()
    asyncio.get_running_loop().call_later(0.05, channel._app_ready.set)
    assert await channel._wait_for_app() is app


async def _watch_for(channel: TelegramChannel, seconds: float) -> bool:
    channel._running = True
    asyncio.get_running_loop().call_later(seconds, setattr, channel, "_running", False)
    return await channel._watch_polling()


async def test_watchdog_ignores_webhook_mode_and_detects_stale_polling(monkeypatch) -> None:
    import time

    monkeypatch.setattr("nanoclaw.channels.telegram.runtime.POLL_WATCH_INTERVAL", 0.01)
    webhook = TelegramChannel(
        TelegramConfig(mode="webhook", webhook_url="https://x.example/t", webhook_secret_token="s"), MessageBus()
    )
    webhook._last_poll_ok = 0.0
    assert await _watch_for(webhook, 0.1) is False

    polling = TelegramChannel(TelegramConfig(), MessageBus())
    polling._last_poll_ok = time.monotonic() - 121
    started = time.monotonic()
    assert await _watch_for(polling, 2.0) is True
    assert time.monotonic() - started < 1.0


async def test_idle_waits_for_the_deadline(monkeypatch) -> None:
    import time

    monkeypatch.setattr("nanoclaw.channels.telegram.runtime.POLL_WATCH_INTERVAL", 0.01)
    channel = TelegramChannel(TelegramConfig(), MessageBus())
    channel._running = True
    started = time.monotonic()
    await channel._idle(0.1)
    assert time.monotonic() - started >= 0.09


async def test_teardown_clears_ready_and_survives_failing_steps() -> None:
    calls: list[str] = []

    async def failing() -> None:
        calls.append("updater.stop")
        raise RuntimeError("updater broken")

    def step(name: str):
        async def run() -> None:
            calls.append(name)

        return run

    app = SimpleNamespace(
        updater=SimpleNamespace(stop=failing),
        stop=step("stop"),
        shutdown=step("shutdown"),
        bot=SimpleNamespace(shutdown=step("bot.shutdown")),
    )
    channel = TelegramChannel(TelegramConfig(), MessageBus())
    channel._app = app  # type: ignore[assignment]
    channel._app_ready.set()
    await channel._teardown_app()
    assert calls == ["updater.stop", "stop", "shutdown", "bot.shutdown"]
    assert channel._app is None and not channel._app_ready.is_set()


async def test_stop_cancels_typing_media_group_and_inbound_tasks() -> None:
    channel = TelegramChannel(TelegramConfig(), MessageBus())
    tasks = [asyncio.create_task(asyncio.sleep(30)) for _ in range(3)]
    channel._typing_tasks["1"] = tasks[0]
    channel._media_group_tasks["1:g"] = tasks[1]
    channel._inbound_workers["telegram:1"] = tasks[2]
    await channel.stop()
    await asyncio.sleep(0)
    assert all(t.cancelled() for t in tasks)
    assert not channel._typing_tasks and not channel._media_group_tasks and not channel._inbound_workers


@pytest.mark.parametrize(
    ("path", "kind"),
    [("a.JPG", "photo"), ("a.webp", "photo"), ("a.mp4", "video"), ("a.ogg", "voice"), ("a.mp3", "audio"),
     ("https://x.example/p/a.png?x=1", "photo"), ("a.pdf", "document")],
)
def test_media_type_from_extension(path: str, kind: str) -> None:
    assert TelegramChannel._get_media_type(path) == kind


# -- rich messages ----------------------------------------------------------------------


def test_rich_streaming_respects_latch() -> None:
    channel = TelegramChannel(TelegramConfig(rich_messages=True), MessageBus())
    assert channel._rich_streaming_enabled() is True
    channel._rich_send_disabled = True
    assert channel._rich_streaming_enabled() is False


async def test_try_send_rich_payload_has_reply_thread_and_markup() -> None:
    from telegram import InlineKeyboardMarkup, ReplyParameters

    channel, bot = _channel(rich_messages=True)
    markup = InlineKeyboardMarkup([])
    assert await channel._try_send_rich(5, "**hi**", ReplyParameters(message_id=7), {"message_thread_id": 9}, markup)
    method, payload = bot.api_requests[-1]
    assert method == "sendRichMessage"
    assert payload == {
        "chat_id": 5,
        "rich_message": {"markdown": "**hi**"},
        "reply_parameters": {"message_id": 7, "allow_sending_without_reply": True},
        "message_thread_id": 9,
        "reply_markup": markup,
    }


async def test_rich_draft_unexpected_error_propagates() -> None:
    channel, bot = _channel(rich_messages=True)
    bot.do_api_request = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="boom"):
        await channel._try_send_rich_draft(5, 1, "x", {})


async def test_legacy_stream_start_sends_leading_chunks() -> None:
    from nanoclaw.channels.telegram.runtime import _StreamBuf

    channel, bot = _channel()
    text = "\n".join(f"line {i} " + "x" * 40 for i in range(120))  # ~5.8k chars
    buf = _StreamBuf(text=text, draft_id=3)
    await channel._start_legacy_stream(5, buf, {})
    assert len(bot.sent_messages) == 2
    assert bot.sent_messages[0]["parse_mode"] == "HTML"
    assert bot.sent_messages[0]["text"].startswith("line 0 ")
    assert "parse_mode" not in bot.sent_messages[1]
    assert bot.sent_messages[1]["text"] == buf.text
    assert buf.message_id == 2 and buf.draft_id is None


# -- send ---------------------------------------------------------------------------------


def _out(content: str = "hi", chat_id: str = "5", **kw) -> OutboundMessage:
    return OutboundMessage(channel="telegram", chat_id=chat_id, content=content, **kw)


async def test_progress_keeps_typing_final_reply_stops_it() -> None:
    channel, _ = _channel()
    task = asyncio.create_task(asyncio.sleep(30))
    channel._typing_tasks["5"] = task
    await channel.send(_out("working", event=ProgressEvent()))
    assert channel._typing_tasks.get("5") is task
    await channel.send(_out("done"))
    await asyncio.sleep(0)
    assert "5" not in channel._typing_tasks and task.cancelled()


async def test_invalid_chat_id_is_rejected_not_retried() -> None:
    from nanoclaw.bus.delivery import DeliveryRejectedError

    channel, _ = _channel()
    with pytest.raises(DeliveryRejectedError):
        await channel.send(_out(chat_id="not-a-number"))


async def test_reply_parameters_only_when_configured() -> None:
    channel, bot = _channel()
    await channel.send(_out(metadata={"message_id": 3}))
    assert bot.sent_messages[-1]["reply_parameters"] is None
    channel.config.reply_to_message = True
    await channel.send(_out(metadata={"message_id": 3}))
    assert bot.sent_messages[-1]["reply_parameters"].message_id == 3


class _MediaBot(_Bot):
    def __init__(self) -> None:
        super().__init__()
        self.media: list[tuple[str, dict]] = []

    def __getattr__(self, name: str):
        if name.startswith("send_"):
            async def send(**kwargs):
                self.media.append((name, kwargs))

            return send
        raise AttributeError(name)


async def test_local_media_methods(tmp_path) -> None:
    channel, _ = _channel()
    bot = _MediaBot()
    channel._app = SimpleNamespace(bot=bot)  # type: ignore[assignment]
    for name in ("v.mp4", "n.ogg"):
        (tmp_path / name).write_bytes(b"x")
    await channel.send(_out("", media=[str(tmp_path / "v.mp4"), str(tmp_path / "n.ogg")]))
    (video, vkw), (voice, nkw) = bot.media
    assert video == "send_video" and vkw["supports_streaming"] is True and vkw["filename"] == "v.mp4"
    assert voice == "send_voice" and nkw["voice"] == b"x" and "supports_streaming" not in nkw


async def test_empty_message_marker_is_not_sent() -> None:
    channel, bot = _channel()
    await channel.send(_out("[empty message]"))
    assert bot.sent_messages == []


async def test_plain_progress_is_markdown_not_blockquote() -> None:
    channel, bot = _channel()
    await channel.send(_out("**x**", event=ProgressEvent()))
    assert bot.sent_messages[-1]["text"] == "<b>x</b>"


async def test_rich_send_success_is_final_and_tool_hints_skip_rich() -> None:
    channel, bot = _channel(rich_messages=True)
    await channel.send(_out("**x**"))
    assert len(bot.api_requests) == 1 and bot.sent_messages == []
    await channel.send(_out("read f", event=ProgressEvent(tool_hint=True)))
    assert len(bot.api_requests) == 1
    assert bot.sent_messages[-1]["text"] == "<blockquote expandable>read f</blockquote>"


async def test_keyboard_only_on_last_chunk() -> None:
    channel, bot = _channel(inline_keyboards=True)
    await channel.send(_out("\n".join("y" * 50 for _ in range(100)), buttons=[["Yes", "No"]]))
    assert len(bot.sent_messages) == 2
    assert bot.sent_messages[0]["reply_markup"] is None
    assert bot.sent_messages[1]["reply_markup"] is not None


async def test_retry_backoff_doubles_on_timeouts(monkeypatch) -> None:
    from telegram.error import TimedOut

    import nanoclaw.channels.telegram.runtime as runtime

    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(runtime.asyncio, "sleep", fake_sleep)
    channel, _ = _channel()
    fn = AsyncMock(side_effect=[TimedOut(), TimedOut(), "ok"])
    assert await channel._call_with_retry(fn) == "ok"
    assert delays == [0.5, 1.0]


async def test_html_fallback_sends_raw_text_with_markup() -> None:
    from telegram import InlineKeyboardMarkup
    from telegram.error import BadRequest

    channel, bot = _channel()
    calls: list[dict] = []

    async def send_message(**kwargs):
        calls.append(kwargs)
        if kwargs.get("parse_mode") == "HTML":
            raise BadRequest("can't parse entities")
        return SimpleNamespace(message_id=1)

    bot.send_message = send_message  # type: ignore[method-assign]
    markup = InlineKeyboardMarkup([])
    await channel._send_text(5, "**x** <y>", reply_markup=markup)
    assert calls[-1]["text"] == "**x** <y>" and calls[-1]["reply_markup"] is markup


async def test_compaction_notices_are_bounded_oldest_first() -> None:
    from nanoclaw.channels.telegram.runtime import COMPACTION_NOTICES_MAX
    from nanoclaw.events import ContextCompactionEvent

    channel, _ = _channel()
    for i in range(COMPACTION_NOTICES_MAX):
        channel._compaction_notices[("5", f"old{i}")] = i
    event = ContextCompactionEvent(compaction_id="new", phase="started", notify=True)
    await channel.send(_out("Compacting", event=event))
    assert len(channel._compaction_notices) == COMPACTION_NOTICES_MAX
    assert ("5", "old0") not in channel._compaction_notices
    assert ("5", "old1") in channel._compaction_notices and ("5", "new") in channel._compaction_notices


async def test_compaction_started_replies_and_not_modified_edit_is_quiet() -> None:
    from telegram.error import BadRequest

    from nanoclaw.events import ContextCompactionEvent

    channel, bot = _channel(reply_to_message=True)
    started = ContextCompactionEvent(compaction_id="c", phase="started", notify=True)
    await channel.send(_out("Compacting", metadata={"message_id": 4}, event=started))
    assert bot.sent_messages[-1]["reply_parameters"].message_id == 4
    bot.edit_message_text = AsyncMock(side_effect=BadRequest("Message is not modified"))  # type: ignore[method-assign]
    done = ContextCompactionEvent(compaction_id="c", phase="succeeded", notify=True)
    await channel.send(_out("Compacting", event=done))
    assert len(bot.sent_messages) == 1


# -- streaming ----------------------------------------------------------------------------


def _legacy_buf(channel: TelegramChannel, text: str = "abc", chat_id: str = "5", **kw):
    from nanoclaw.channels.telegram.runtime import _StreamBuf

    buf = _StreamBuf(text=text, message_id=1, **kw)
    channel._stream_bufs[chat_id] = buf
    return buf


async def test_merge_next_end_without_delta_does_nothing() -> None:
    channel, bot = _channel()
    _legacy_buf(channel)
    await channel.send_delta("5", "", stream_end=True, merge_next=True)
    assert bot.edits == [] and channel._stream_bufs["5"].text == "abc"


async def test_stream_end_with_empty_buffer_is_a_noop() -> None:
    channel, bot = _channel()
    _legacy_buf(channel, text="")
    await channel.send_delta("5", "", stream_end=True)
    assert bot.edits == [] and bot.sent_messages == []


async def test_stream_end_for_other_stream_is_ignored() -> None:
    channel, bot = _channel()
    _legacy_buf(channel, stream_id="a")
    await channel.send_delta("5", "", stream_id="b", stream_end=True)
    assert bot.edits == [] and "5" in channel._stream_bufs


async def test_stream_end_stops_typing_and_removes_reaction() -> None:
    channel, bot = _channel()
    _legacy_buf(channel)
    task = asyncio.create_task(asyncio.sleep(30))
    channel._typing_tasks["5"] = task
    await channel.send_delta("5", "", {"message_id": 8}, stream_end=True)
    await asyncio.sleep(0)
    assert task.cancelled() and "5" not in channel._typing_tasks
    assert bot.reactions == [{"chat_id": 5, "message_id": 8, "reaction": []}]


async def test_rich_stream_end_uses_rich_limit() -> None:
    channel, bot = _channel(rich_messages=True)
    _legacy_buf(channel, text="word " * 1000, draft_id=3).message_id = None
    await channel.send_delta("5", "", stream_end=True)
    assert [m for m, _ in bot.api_requests] == ["sendRichMessage"]
    assert bot.sent_messages == []


async def test_rich_stream_end_after_latch_uses_legacy_messages() -> None:
    channel, bot = _channel(rich_messages=True)
    channel._rich_send_disabled = True
    _legacy_buf(channel, text="**x**", draft_id=3).message_id = None
    await channel.send_delta("5", "", stream_end=True)
    assert bot.api_requests == []
    assert bot.sent_messages[-1]["text"] == "<b>x</b>"


async def test_rich_stream_end_failure_sends_only_unsent_chunks() -> None:
    from telegram.error import BadRequest

    channel, bot = _channel(rich_messages=True)
    first = "A" * 30000
    second = "B" * 3000
    _legacy_buf(channel, text=first + "\n" + second, draft_id=3).message_id = None
    results = [True, BadRequest("Method not found")]

    async def api(method, api_kwargs=None):
        bot.api_requests.append((method, api_kwargs))
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    bot.do_api_request = api  # type: ignore[method-assign]
    await channel.send_delta("5", "", stream_end=True)
    assert len(bot.api_requests) == 2
    assert [m["text"] for m in bot.sent_messages] == [second]


async def test_final_edit_not_modified_is_not_retried_plain() -> None:
    from telegram.error import BadRequest

    channel, bot = _channel()
    _legacy_buf(channel)
    bot.edit_message_text = AsyncMock(side_effect=BadRequest("Message is not modified"))  # type: ignore[method-assign]
    await channel.send_delta("5", "", stream_end=True)
    assert bot.edit_message_text.await_count == 1
    assert "5" not in channel._stream_bufs


async def test_final_plain_fallback_uses_raw_first_chunk() -> None:
    from telegram.error import BadRequest

    channel, bot = _channel()
    text = "\n".join("**w** " + "z" * 60 for _ in range(80))  # > 4000 raw
    _legacy_buf(channel, text=text)
    edits: list[dict] = []

    async def edit(**kwargs):
        edits.append(kwargs)
        if kwargs.get("parse_mode") == "HTML":
            raise BadRequest("can't parse entities")

    bot.edit_message_text = edit  # type: ignore[method-assign]
    await channel.send_delta("5", "", stream_end=True)
    plain = edits[1]
    assert "parse_mode" not in plain
    assert plain["text"].startswith("**w** ") and len(plain["text"]) <= 4000


async def test_final_plain_fallback_errors() -> None:
    from telegram.error import BadRequest

    channel, bot = _channel()
    _legacy_buf(channel)
    bot.edit_message_text = AsyncMock(  # type: ignore[method-assign]
        side_effect=[BadRequest("can't parse"), BadRequest("Message is not modified")]
    )
    await channel.send_delta("5", "", stream_end=True)  # not modified: success
    assert "5" not in channel._stream_bufs

    _legacy_buf(channel)
    bot.edit_message_text = AsyncMock(  # type: ignore[method-assign]
        side_effect=[BadRequest("can't parse"), BadRequest("chat not found")]
    )
    with pytest.raises(BadRequest):
        await channel.send_delta("5", "", stream_end=True)


async def test_final_extra_chunks_keep_thread_and_fall_back() -> None:
    channel, bot = _channel()
    text = "\n".join("y" * 50 for _ in range(100))
    _legacy_buf(channel, text=text)
    calls: list[dict] = []

    async def send_message(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("flaky")
        return SimpleNamespace(message_id=len(calls))

    bot.send_message = send_message  # type: ignore[method-assign]
    await channel.send_delta("5", "", {"message_thread_id": 7}, stream_end=True)
    assert len(calls) == 2
    assert calls[0]["message_thread_id"] == 7 and calls[1]["message_thread_id"] == 7


async def test_delta_adopts_stream_id_and_ignores_blank_text() -> None:
    channel, bot = _channel()
    buf = _legacy_buf(channel, text="")
    buf.message_id = None
    await channel.send_delta("5", "   ", stream_id="s1")
    assert buf.stream_id == "s1"
    assert bot.sent_messages == [] and bot.edits == []


async def test_oversized_first_delta_skips_rich_draft() -> None:
    channel, bot = _channel(rich_messages=True)
    await channel.send_delta("5", "x " * 17000, {"is_group": False})
    assert bot.api_requests == []
    assert bot.sent_messages and channel._stream_bufs["5"].draft_id is None


async def test_initial_stream_send_failure_raises() -> None:
    channel, bot = _channel()
    bot.send_message = AsyncMock(side_effect=RuntimeError("down"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="down"):
        await channel.send_delta("5", "hello")


async def test_draft_updates_respect_minimum_interval() -> None:
    import time

    channel, bot = _channel(rich_messages=True, stream_edit_interval=0.1)
    buf = _legacy_buf(channel, text="a", draft_id=3)
    buf.message_id = None
    buf.last_edit = time.monotonic() - 0.5
    await channel.send_delta("5", "b", {"is_group": False})
    assert bot.api_requests == []


async def test_preview_edit_updates_last_edit_and_failures_raise() -> None:
    channel, bot = _channel()
    buf = _legacy_buf(channel)
    await channel.send_delta("5", "d")
    assert bot.edits[-1]["text"] == "abcd" and buf.last_edit > 0
    await channel.send_delta("5", "e")
    assert len(bot.edits) == 1  # within the edit interval
    buf.last_edit = 0.0
    bot.edit_message_text = AsyncMock(side_effect=RuntimeError("edit failed"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="edit failed"):
        await channel.send_delta("5", "f")


async def test_overflow_with_latched_rich_switches_to_legacy() -> None:
    channel, bot = _channel(rich_messages=True)
    channel._rich_send_disabled = True
    buf = _legacy_buf(channel, text="hello", draft_id=3)
    buf.message_id = None
    await channel._flush_stream_overflow(5, buf, {})
    assert buf.draft_id is None and buf.message_id == 1
    assert bot.api_requests == []


async def test_rich_overflow_failure_resends_only_unsent_chunks() -> None:
    from telegram.error import BadRequest

    channel, bot = _channel(rich_messages=True)
    text = "A" * 30000 + "\n" + "B" * 30000 + "\n" + "C" * 100
    buf = _legacy_buf(channel, text=text, draft_id=3)
    buf.message_id = None
    results = [True, BadRequest("Method not found")]

    async def api(method, api_kwargs=None):
        bot.api_requests.append((method, api_kwargs))
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    bot.do_api_request = api  # type: ignore[method-assign]
    await channel._flush_stream_overflow(5, buf, {})
    assert not any("A" in m["text"] for m in bot.sent_messages)


async def test_legacy_overflow_single_chunk_is_noop() -> None:
    channel, bot = _channel()
    buf = _legacy_buf(channel, text="short")
    await channel._flush_stream_overflow_legacy(5, buf, {})
    assert bot.edits == [] and bot.sent_messages == []


async def test_legacy_overflow_sends_middle_chunks() -> None:
    channel, bot = _channel()
    text = "\n".join("y" * 50 for _ in range(200))  # ~10k: three chunks
    buf = _legacy_buf(channel, text=text)
    await channel._flush_stream_overflow_legacy(5, buf, {})
    assert len(bot.edits) == 1 and len(bot.sent_messages) == 2
    assert buf.message_id == 2


async def test_legacy_overflow_edit_failures_raise() -> None:
    from telegram.error import BadRequest

    channel, bot = _channel()
    text = "\n".join("y" * 50 for _ in range(100))
    buf = _legacy_buf(channel, text=text)
    bot.edit_message_text = AsyncMock(side_effect=RuntimeError("down"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await channel._flush_stream_overflow_legacy(5, buf, {})
    bot.edit_message_text = AsyncMock(  # type: ignore[method-assign]
        side_effect=[BadRequest("can't parse"), BadRequest("chat not found")]
    )
    with pytest.raises(BadRequest):
        await channel._flush_stream_overflow_legacy(5, buf, {})


# -- inbound ------------------------------------------------------------------------------

_USER = SimpleNamespace(id=12345, username="alice", first_name="Alice")


def _msg(**kw):
    fields = dict(
        chat=SimpleNamespace(type="private", is_forum=False),
        chat_id=777,
        text=None,
        caption=None,
        entities=[],
        caption_entities=[],
        reply_to_message=None,
        photo=None,
        voice=None,
        audio=None,
        document=None,
        video=None,
        video_note=None,
        animation=None,
        location=None,
        media_group_id=None,
        message_thread_id=None,
        message_id=1,
        from_user=None,
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


def _update(message, user=_USER, update_id: int = 1):
    return SimpleNamespace(message=message, effective_user=user, update_id=update_id)


async def test_start_greets_by_first_name() -> None:
    channel, _ = _channel()
    reply = AsyncMock()
    await channel._on_start(_update(_msg(reply_text=reply)), None)
    assert "Hi Alice!" in reply.await_args.args[0]


def test_sender_id_and_metadata() -> None:
    assert TelegramChannel._sender_id(_USER) == "12345|alice"
    msg = _msg(
        chat=SimpleNamespace(type="supergroup", is_forum=True),
        message_thread_id=9,
        message_id=4,
        reply_to_message=SimpleNamespace(message_id=3),
    )
    assert TelegramChannel._build_message_metadata(msg, _USER) == {
        "message_id": 4,
        "user_id": 12345,
        "username": "alice",
        "first_name": "Alice",
        "is_group": True,
        "message_thread_id": 9,
        "is_forum": True,
        "reply_to_message_id": 3,
    }


async def test_reply_context_names_the_bot() -> None:
    channel, _ = _channel()
    channel._bot_user_id, channel._bot_username = 999, "nb"
    reply = SimpleNamespace(text="earlier", caption=None, from_user=SimpleNamespace(id=999, username="nb"))
    assert await channel._extract_reply_context(_msg(reply_to_message=reply)) == "[Reply to bot: earlier]"


class _File:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def download_to_drive(self, path: str) -> None:
        if self.fail:
            raise OSError("download failed")


def _media_channel(monkeypatch, tmp_path, fail: bool = False):
    channel, bot = _channel()
    monkeypatch.setattr("nanoclaw.channels.telegram.runtime.get_data_dir", lambda: tmp_path)
    requested: list[str] = []

    async def get_file(file_id: str):
        requested.append(file_id)
        return _File(fail)

    bot.get_file = get_file  # type: ignore[attr-defined]
    return channel, requested


def _media(file_id: str, **kw):
    return SimpleNamespace(file_id=file_id, file_unique_id=f"u{file_id}", **kw)


async def test_download_media_kinds(monkeypatch, tmp_path) -> None:
    channel, requested = _media_channel(monkeypatch, tmp_path)
    media_dir = tmp_path / "media" / "telegram"
    cases = [
        (_msg(photo=[_media("small"), _media("large")]), f"[image: {media_dir / 'ularge.jpg'}]"),
        (_msg(voice=_media("v")), f"[voice: {media_dir / 'uv.ogg'}]"),
        (_msg(document=_media("d", file_name="r.tar.gz")), f"[file: {media_dir / 'ud.tar.gz'}]"),
        (_msg(video_note=_media("n")), f"[video: {media_dir / 'un.mp4'}]"),
        (_msg(animation=_media("a")), f"[animation: {media_dir / 'ua'}]"),
    ]
    for message, part in cases:
        paths, parts = await channel._download_message_media(message)
        assert parts == [part] and paths == [part.split(": ", 1)[1][:-1]]
    assert requested == ["large", "v", "d", "n", "a"]


async def test_download_failure_content_only_when_asked(monkeypatch, tmp_path) -> None:
    channel, _ = _media_channel(monkeypatch, tmp_path, fail=True)
    message = _msg(photo=[_media("p")])
    assert await channel._download_message_media(message, add_failure_content=True) == ([], ["[image: download failed]"])
    assert await channel._download_message_media(message) == ([], [])


def test_extension_prefers_mime_type() -> None:
    channel, _ = _channel()
    assert channel._get_extension("voice", "audio/mpeg") == ".mp3"
    assert channel._get_extension("audio", "audio/ogg") == ".ogg"


async def _processed(channel: TelegramChannel, message, user=_USER) -> dict:
    channel._handle_message = AsyncMock(return_value=True)  # type: ignore[method-assign]
    await channel._process_message_update(_update(message, user))
    return channel._handle_message.await_args.kwargs


async def test_message_content_media_reply_and_topic(monkeypatch, tmp_path) -> None:
    channel, _ = _media_channel(monkeypatch, tmp_path)
    media_dir = tmp_path / "media" / "telegram"
    reply = _msg(text=None, caption=None, photo=[_media("r")], from_user=None)
    kwargs = await _processed(
        channel,
        _msg(text="look", location=SimpleNamespace(latitude=1, longitude=2), photo=[_media("c")],
             reply_to_message=reply, message_thread_id=5),
    )
    assert kwargs["content"] == (
        f"[Reply to: [image: {media_dir / 'ur.jpg'}]]\nlook\n[location: 1, 2]\n[image: {media_dir / 'uc.jpg'}]"
    )
    assert kwargs["media"] == [str(media_dir / "ur.jpg"), str(media_dir / "uc.jpg")]
    assert kwargs["session_key"] == "telegram:777:topic:5"


async def test_failed_downloads_in_message_and_reply(monkeypatch, tmp_path) -> None:
    channel, _ = _media_channel(monkeypatch, tmp_path, fail=True)
    reply = _msg(text=None, caption=None, photo=[_media("r")])
    kwargs = await _processed(channel, _msg(photo=[_media("c")], reply_to_message=reply))
    assert kwargs["content"] == "[image: download failed]"


async def test_message_without_content_is_marked_empty() -> None:
    channel, _ = _channel()
    kwargs = await _processed(channel, _msg())
    assert kwargs["content"] == "[empty message]"


def test_text_mention_entity_and_reply_to_others() -> None:
    mention = SimpleNamespace(type="text_mention", user=SimpleNamespace(id=999))
    other = SimpleNamespace(type="text_mention", user=SimpleNamespace(id=1))
    assert TelegramChannel._has_mention_entity("hey", [mention], "nb", 999) is True
    assert TelegramChannel._has_mention_entity("hey", [other], "nb", 999) is False
    at_other = SimpleNamespace(type="mention", offset=0, length=6)
    assert TelegramChannel._has_mention_entity("@other hi", [at_other], "nb", 999) is False
    assert TelegramChannel._has_mention_entity("hi @NB there", [], "nb", 999) is True


async def test_group_reply_to_another_user_is_not_for_bot() -> None:
    channel, _ = _channel()
    channel._bot_user_id, channel._bot_username = 999, "nb"
    group = SimpleNamespace(type="group", is_forum=False)
    to_other = _msg(chat=group, text="ok", reply_to_message=SimpleNamespace(from_user=SimpleNamespace(id=1)))
    to_bot = _msg(chat=group, text="ok", reply_to_message=SimpleNamespace(from_user=SimpleNamespace(id=999)))
    assert await channel._is_group_message_for_bot(to_other) is False
    assert await channel._is_group_message_for_bot(to_bot) is True


async def test_thread_context_is_remembered_and_bounded() -> None:
    channel, _ = _channel()
    channel._handle_message = AsyncMock(return_value=True)  # type: ignore[method-assign]
    await channel._process_message_update(_update(_msg(text="hi", message_thread_id=5, message_id=3)))
    assert channel._message_threads == {("777", 3): 5}
    for i in range(1000):
        channel._remember_thread_context(_msg(message_thread_id=1, message_id=100 + i))
    assert len(channel._message_threads) == 1000 and ("777", 3) not in channel._message_threads


async def test_unauthorized_command_does_not_touch_thread_cache() -> None:
    channel, _ = _channel()
    channel.config.allow_from = ["1"]
    group = SimpleNamespace(type="group", is_forum=False)
    await channel._process_forward_command(_update(_msg(chat=group, text="/new", message_thread_id=5)))
    assert channel._message_threads == {}


async def test_forwarded_command_routing() -> None:
    channel, _ = _channel()
    channel._handle_message = AsyncMock(return_value=True)  # type: ignore[method-assign]
    group = SimpleNamespace(type="group", is_forum=True)
    await channel._process_forward_command(_update(_msg(chat=group, text="/new@nb", message_thread_id=5)))
    kwargs = channel._handle_message.await_args.kwargs
    assert kwargs["content"] == "/new"
    assert kwargs["session_key"] == "telegram:777:topic:5"
    assert kwargs["is_dm"] is False


def test_queue_and_sort_keys() -> None:
    assert TelegramChannel._queue_key_for_message(_msg(message_thread_id=5)) == "telegram:777:topic:5"
    assert TelegramChannel._sort_key_for_update(_update(_msg(message_id=4), update_id=9)) == (4, 9)


async def test_ordered_queue_window_errors_and_cleanup() -> None:
    channel, _ = _channel()
    channel._running = True
    seen: list[int] = []

    async def process(update) -> None:
        seen.append(update.message.message_id)
        if update.message.message_id == 1:
            raise RuntimeError("handler failed")

    channel._process_message_update = process  # type: ignore[method-assign]
    channel._enqueue_ordered_update(kind="message", update=_update(_msg(message_id=2), update_id=2))
    worker = channel._inbound_workers["telegram:777"]
    await asyncio.sleep(0.05)
    channel._enqueue_ordered_update(kind="message", update=_update(_msg(message_id=1), update_id=1))
    assert channel._inbound_workers["telegram:777"] is worker
    await asyncio.sleep(0.6)
    assert seen == [1, 2]
    assert "telegram:777" not in channel._inbound_workers
    channel._running = False


async def test_media_group_is_one_turn_and_stop_cancels_pending(monkeypatch) -> None:
    channel, _ = _channel()
    channel._app = None  # no downloads, no reactions
    channel._handle_message = AsyncMock(return_value=True)  # type: ignore[method-assign]
    for chat_id, msg_id in ((777, 1), (777, 2), (888, 3)):
        await channel._process_message_update(
            _update(_msg(chat_id=chat_id, message_id=msg_id, media_group_id="g", message_thread_id=4))
        )
    await asyncio.sleep(0.75)
    calls = [c.kwargs for c in channel._handle_message.await_args_list]
    assert sorted(c["chat_id"] for c in calls) == ["777", "888"]
    assert all(c["content"] == "[empty message]" for c in calls)
    assert all(c["session_key"].endswith(":topic:4") for c in calls)
    assert channel._media_group_tasks == {}

    channel._handle_message.reset_mock()
    tasks = []
    for msg_id in (5, 6):
        await channel._process_message_update(_update(_msg(message_id=msg_id, media_group_id="h")))
        tasks.append(channel._media_group_tasks["777:h"])
    assert tasks[0] is tasks[1]  # one flush task per album, so stop() cancels it
    await channel.stop()
    await asyncio.sleep(0.75)
    channel._handle_message.assert_not_called()


async def test_media_group_deduplicates_media(monkeypatch) -> None:
    channel, _ = _channel()
    channel._handle_message = AsyncMock(return_value=True)  # type: ignore[method-assign]

    async def download(message, *, add_failure_content=False):
        return ["/m/a.jpg"], ["[image: /m/a.jpg]"]

    channel._download_message_media = download  # type: ignore[method-assign]
    for msg_id in (1, 2):
        await channel._process_message_update(_update(_msg(message_id=msg_id, media_group_id="g")))
    await asyncio.sleep(0.75)
    assert channel._handle_message.await_args.kwargs["media"] == ["/m/a.jpg"]


async def test_typing_tasks_are_replaced_and_cancelled() -> None:
    channel, _ = _channel()
    channel._typing_loop = lambda chat_id: asyncio.sleep(30)  # type: ignore[method-assign]
    channel._start_typing("5")
    first = channel._typing_tasks["5"]
    channel._start_typing("5")
    await asyncio.sleep(0)
    assert first.cancelled()
    second = channel._typing_tasks["5"]
    channel._stop_typing("5")
    await asyncio.sleep(0)
    assert second.cancelled()


async def test_typing_loop_interval_and_empty_reaction() -> None:
    channel, bot = _channel()
    calls: list[dict] = []

    async def action(**kwargs):
        calls.append(kwargs)

    bot.send_chat_action = action  # type: ignore[method-assign]
    task = asyncio.create_task(channel._typing_loop("5"))
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert calls == [{"chat_id": 5, "action": "typing"}]
    await channel._add_reaction("5", 1, "")
    assert bot.reactions == []


def test_format_error_uses_cause() -> None:
    from telegram.error import NetworkError

    err = NetworkError("")
    err.__cause__ = OSError("connection refused")
    assert TelegramChannel._format_telegram_error(err) == "NetworkError (connection refused)"


def test_polling_error_levels(monkeypatch) -> None:
    from telegram.error import NetworkError

    channel, _ = _channel()
    recorded: list[str] = []
    monkeypatch.setattr(channel.logger, "warning", lambda message, summary: recorded.append("warning"))
    monkeypatch.setattr(channel.logger, "error", lambda message, summary: recorded.append("error"))
    channel._on_polling_error(NetworkError("blip"))
    channel._on_polling_error(ValueError("bug"))
    assert recorded == ["warning", "error"]


async def test_callback_query_forwards_label_and_clears_keyboard() -> None:
    from datetime import datetime

    from telegram import Chat, Message

    cleared: list[object] = []

    class _Msg(Message):
        async def edit_reply_markup(self, reply_markup=None, **kwargs):  # type: ignore[override]
            cleared.append(reply_markup)

    channel, _ = _channel()
    channel._typing_loop = lambda chat_id: asyncio.sleep(0)  # type: ignore[method-assign]
    channel._handle_message = AsyncMock(return_value=True)  # type: ignore[method-assign]
    message = _Msg(message_id=1, date=datetime.now(), chat=Chat(id=5, type="private"))
    query = SimpleNamespace(message=message, data="Yes", id="q1", answer=AsyncMock())
    await channel._on_callback_query(SimpleNamespace(callback_query=query, effective_user=_USER), None)
    kwargs = channel._handle_message.await_args.kwargs
    assert kwargs["content"] == "Yes" and kwargs["metadata"]["is_callback"] is True
    assert cleared == [None]


# -- validation / CLI ---------------------------------------------------------------------

_GOOD_TOKEN = "123456:abcdefghijklmnopqrstuvwxyz"


def test_validation_payload_helpers(monkeypatch) -> None:
    from nanoclaw.channels.telegram import validation as v

    assert v.payload("t", "connected", [], identity={"name": "b", "account": ""})["identity"] == {"name": "b"}
    assert v.payload("t", "configured", [], missing_fields=["x"])["can_enable"] is False
    assert v.payload("t", "invalid", [])["can_enable"] is False
    assert v.payload("t", "configured", [])["can_enable"] is True
    assert v.payload("t", "connected", [], can_enable=False)["can_enable"] is False
    skipped = [v.check("a", "A", "pass"), v.check("b", "B", "skipped")]
    assert v.status_from_checks("t", skipped, [])["status"] == "configured"
    assert v.string_value("  x ") == "x" and v.string_value(None) == "" and v.string_value(5) == "5"
    assert v.message_from_response({"description": "Unauthorized"}, "fallback") == "Unauthorized"
    monkeypatch.delenv("NC_UNSET_TOKEN_X", raising=False)
    assert v.resolve_env_refs("${NC_UNSET_TOKEN_X}") == ""


def test_get_me_raises_http_status(monkeypatch) -> None:
    import httpx

    from nanoclaw.channels.telegram import validation as v

    request = httpx.Request("GET", "https://api.telegram.org/x")

    class Client:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def get(self, url: str):
            return httpx.Response(401, request=request, json={"ok": False})

    monkeypatch.setattr(v.httpx, "Client", Client)
    with pytest.raises(httpx.HTTPStatusError):
        v._get_me(_GOOD_TOKEN, None)


def _validate_with(monkeypatch, token: str, get_me) -> dict:
    from nanoclaw.channels.telegram import validation as v

    monkeypatch.setattr(v, "_get_me", get_me)
    return v.validate({"token": token})


def _check(result: dict, check_id: str) -> dict:
    return next(c for c in result["checks"] if c["id"] == check_id)


def test_validate_token_checks(monkeypatch) -> None:
    import httpx

    def unexpected(token, proxy):
        raise AssertionError("getMe must not run")

    monkeypatch.delenv("NC_UNSET_TOKEN_X", raising=False)
    result = _validate_with(monkeypatch, "${NC_UNSET_TOKEN_X}", unexpected)
    assert result["status"] == "invalid" and _check(result, "token_env")["status"] == "fail"

    result = _validate_with(monkeypatch, _GOOD_TOKEN + "!!", unexpected)
    assert _check(result, "token_format")["status"] == "fail"

    result = _validate_with(monkeypatch, _GOOD_TOKEN, lambda t, p: {"ok": False, "result": {"id": 1}})
    assert result["status"] == "invalid"

    result = _validate_with(monkeypatch, _GOOD_TOKEN, lambda t, p: {"ok": True, "result": {"id": 42, "first_name": "Bo"}})
    assert result["status"] == "connected" and result["identity"] == {"name": "Bo", "account": "42"}

    def transport(token, proxy):
        raise httpx.ConnectError("offline")

    result = _validate_with(monkeypatch, _GOOD_TOKEN, transport)
    assert result["status"] == "configured" and _check(result, "get_me")["status"] == "warn"

    def broken(token, proxy):
        raise ValueError("odd json")

    result = _validate_with(monkeypatch, _GOOD_TOKEN, broken)
    assert result["status"] == "configured" and _check(result, "get_me")["status"] == "warn"


def test_validate_unsupported_channel() -> None:
    from nanoclaw.channels.telegram.validation import validate_channel_config

    result = validate_channel_config("discord")
    assert result["status"] == "unsupported" and result["name"] == "discord"


def test_cli_channels_output(tmp_path, capsys, monkeypatch) -> None:
    import httpx

    from nanoclaw.__main__ import main
    from nanoclaw.channels.telegram import validation
    from nanoclaw.config.loader import save_config
    from nanoclaw.config.schema import Config

    monkeypatch.setattr("nanoclaw.config.loader._current_config_path", None)  # main() sets it; restore after
    path = tmp_path / "config.json"
    save_config(Config.model_validate({"channels": {"telegram": {"enabled": True, "token": _GOOD_TOKEN}}}), path)
    assert main(["channels", "status", "-c", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Telegram  yes" in out and "WebSocket no" in out

    monkeypatch.setattr(validation, "_get_me", lambda token, proxy: {"ok": True, "result": {"id": 42, "username": "b"}})
    assert main(["channels", "validate", "-c", str(path)]) == 0
    out = capsys.readouterr().out
    assert "  [pass] Token format: Looks like a BotFather token." in out
    assert "  bot: @b (id 42)" in out

    def offline(token, proxy):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(validation, "_get_me", offline)
    assert main(["channels", "validate", "-c", str(path)]) == 0  # configured: could not verify

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["channels", "status", "-c", str(path)])


# -- pairing store ------------------------------------------------------------------------


@pytest.fixture
def pairing_store(tmp_path, monkeypatch):
    from nanoclaw.pairing import store

    path = tmp_path / "pairing.json"
    monkeypatch.setattr(store, "_store_path", lambda: path)
    return store, path


def test_pairing_store_sorts_approved_ids(pairing_store) -> None:
    import json

    store, path = pairing_store
    ids = [f"user{i:02d}" for i in range(25, 0, -1)]
    for sender in ids:
        assert store.approve_code(store.generate_code("telegram", sender)) is not None
    assert json.loads(path.read_text())["approved"]["telegram"] == sorted(ids)
    assert store.get_approved("telegram") == sorted(ids)


def test_pending_entries_with_bad_channel_are_dropped(pairing_store) -> None:
    import json
    import time

    store, path = pairing_store
    later = time.time() + 600
    path.write_text(json.dumps({"approved": {}, "pending": {
        "AAAA-AAAA": {"channel": 5, "sender_id": "1", "expires_at": later},
        "BBBB-BBBB": {"channel": "", "sender_id": "1", "expires_at": later},
        "CCCC-CCCC": {"channel": "telegram", "sender_id": "1", "expires_at": later},
    }}))
    assert [p["code"] for p in store.list_pending()] == ["CCCC-CCCC"]


def test_generate_code_ignores_expired_code_and_stores_str_sender(pairing_store, monkeypatch) -> None:
    store, _ = pairing_store
    clock = {"now": 1000.0}
    monkeypatch.setattr(store.time, "time", lambda: clock["now"])
    old = store.generate_code("telegram", 123, ttl=1)  # type: ignore[arg-type]
    clock["now"] += 5
    changes: list[int] = []
    new = store.generate_code("telegram", 123, on_change=lambda: changes.append(1))  # type: ignore[arg-type]
    assert new != old and changes == [1]
    (entry,) = store.list_pending()
    assert entry["sender_id"] == "123"
    assert entry["expires_at"] - entry["created_at"] == 600
    assert store.generate_code("telegram", "123", on_change=lambda: changes.append(2)) == new
    assert changes == [1]  # reusing the active code changes nothing


def test_expired_code_is_not_revived_or_denied(pairing_store, monkeypatch) -> None:
    store, _ = pairing_store
    clock = {"now": 1000.0}
    monkeypatch.setattr(store.time, "time", lambda: clock["now"])
    code = store.generate_code("telegram", "1", ttl=1)
    clock["now"] += 5
    changes: list[int] = []
    store.record_delivery(code, "sent", on_change=lambda: changes.append(1))
    assert changes == []
    assert store.deny_code(code) is False


def test_approval_is_per_channel_and_revoke_channel_persists(pairing_store) -> None:
    store, _ = pairing_store
    for sender in ("1", "2"):
        store.approve_code(store.generate_code("discord", sender))
    assert store.is_approved("discord", "1") and not store.is_approved("telegram", "1")
    assert store.revoke_channel("discord") == 2
    assert store.get_approved("discord") == []


def test_format_expiry_past_is_expired() -> None:
    import time

    from nanoclaw.pairing.store import format_expiry

    assert format_expiry(time.time() - 10) == "expired"
    assert format_expiry(time.time() + 100) in {"99s", "100s"}


def test_pairing_command_texts(pairing_store) -> None:
    store, _ = pairing_store
    code = store.generate_code("telegram", "7")
    listing = store.handle_pairing_command("telegram", "list")
    assert f"- `{code}` | telegram | 7 | " in listing and listing.rstrip().endswith("s")
    assert store.handle_pairing_command("telegram", "deny") == "Usage: `/pairing deny <code>`"
    assert store.handle_pairing_command("telegram", "revoke a b c").startswith("Usage: `/pairing revoke")

    store.approve_code(store.generate_code("discord", "9"))
    changes: list[int] = []
    reply = store.handle_pairing_command("discord", "revoke 9", on_change=lambda: changes.append(1))
    assert reply == "Revoked 9 from discord" and changes == [1]


class _PairingChannel:
    @staticmethod
    def make(send_error: Exception | None = None):
        from nanoclaw.channels.base import BaseChannel

        class Channel(BaseChannel):
            name = "dummy"

            def __init__(self) -> None:
                super().__init__(SimpleNamespace(allow_from=[]), MessageBus())
                self.statuses: list[object] = []
                self.events: list[object] = []
                self.bus.publish_nowait = self.events.append  # type: ignore[method-assign]

            async def start(self) -> None:
                return None

            async def stop(self) -> None:
                return None

            async def send(self, msg: OutboundMessage) -> None:
                from nanoclaw.pairing import list_pending

                self.statuses.append(list_pending()[0].get("delivery_status"))
                if send_error is not None:
                    raise send_error

        return Channel()


async def test_pairing_reply_records_delivery_and_publishes_changes(pairing_store) -> None:
    store, _ = pairing_store
    channel = _PairingChannel.make()
    assert await channel._handle_message(sender_id="u1", chat_id="c1", content="hi", is_dm=True) is False
    assert channel.statuses == ["sending"]
    assert store.list_pending()[0]["delivery_status"] == "sent"
    assert len(channel.events) == 3  # code created, sending, sent


async def test_pairing_reply_failure_is_recorded(pairing_store) -> None:
    store, _ = pairing_store
    channel = _PairingChannel.make(RuntimeError("send failed"))
    with pytest.raises(RuntimeError):
        await channel._handle_message(sender_id="u1", chat_id="c1", content="hi", is_dm=True)
    assert store.list_pending()[0]["delivery_status"] == "failed"


# -- split_message ------------------------------------------------------------------------


def _finishes(fn, *args, timeout: float = 5.0):
    """Run ``fn`` in a daemon thread: a split that never advances fails the test
    instead of hanging the run."""
    import threading

    result: list[object] = []
    thread = threading.Thread(target=lambda: result.append(fn(*args)), daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "split_message did not terminate"
    return result[0]


@pytest.mark.parametrize(
    ("content", "max_len", "expected"),
    [
        ("", 5, []),
        ("abc", 0, ["abc"]),
        ("  \taxyz\t bxyza\n\r\n", 2, ["\ta", "xy", "z\t", "bx", "yz", "a"]),
        ("xyz\t\r\n\n \nb\r\nxyz", 3, ["xyz", "b", "xyz"]),
        (" \r\n   b\r\n b\n\tbba", 5, ["   b", " b", "\tbba"]),
        ("a\nab    b\nb", 5, ["a", "ab  ", "b\nb"]),
        ("   abb", 3, ["abb"]),
        (" ab\r\n   ab  \r\n\n\r\na", 4, [" ab", "   a", "b ", "\n\r\na"]),
        ("abxyz\t\n\r\nxyz", 5, ["abxyz", "\r\nxyz"]),
        ("  \t  \n", 2, ["  "]),
    ],
)
def test_split_message_boundaries(content: str, max_len: int, expected: list[str]) -> None:
    from nanoclaw.utils.helpers import split_message

    assert _finishes(split_message, content, max_len) == expected


# -- pairing: channel, command, web UI ------------------------------------------------------


async def test_handle_message_passes_session_generation() -> None:
    from nanoclaw.channels.base import BaseChannel

    class Channel(BaseChannel):
        name = "dummy"

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def send(self, msg: OutboundMessage) -> None:
            return None

    bus = MessageBus()
    channel = Channel({"allowFrom": ["*"]}, bus)
    assert await channel._handle_message(sender_id="u", chat_id="c", content="hi", session_generation="g1")
    assert (await bus.consume_inbound()).session_generation == "g1"


def test_pairing_command_with_args_is_not_an_agent_turn() -> None:
    from nanoclaw.command.builtin import builtin_command_palette, builtin_command_starts_agent_turn

    assert builtin_command_starts_agent_turn("/pairing approve ABCD-EFGH") is False
    spec = next(item for item in builtin_command_palette() if item["command"] == "/pairing")
    assert spec["accepts_args"] is True


def test_pairing_payload_clamps_expiry_and_action_strips_code() -> None:
    from nanoclaw.webui.settings_system import pairing_action, pairing_payload

    payload = pairing_payload(lambda: [{"code": "X", "channel": "t", "sender_id": "1", "expires_at": 90.0}], now=100.0)
    assert payload["requests"][0]["expires_in_seconds"] == 0
    approved: list[str] = []
    status, body = pairing_action(
        "  ABCD-EFGH ",
        "approve",
        list_pending=lambda: [],
        approve_code=lambda code: approved.append(code) or ("t", "1"),
        deny_code=lambda code: False,
    )
    assert status == 200 and approved == ["ABCD-EFGH"]
    assert isinstance(body, dict) and body["last_action"]["code"] == "ABCD-EFGH"


def test_pairing_bus_falls_back_to_the_agent_loop() -> None:
    from nanoclaw.webui.rest_api import _pairing_bus

    bus = MessageBus()
    assert _pairing_bus(SimpleNamespace(channel=None, loop=SimpleNamespace(bus=bus))) is bus  # type: ignore[arg-type]
    assert _pairing_bus(SimpleNamespace(channel=None, loop=None)) is None  # type: ignore[arg-type]


async def test_websocket_send_pairing_changed_targets_webui_clients() -> None:
    from nanoclaw.channels.websocket.runtime import WebSocketChannel

    fake = SimpleNamespace(broadcast=AsyncMock())
    await WebSocketChannel.send_pairing_changed(fake)  # type: ignore[arg-type]
    fake.broadcast.assert_awaited_once_with("*", {"event": "pairing_changed"}, audience="webui")


# -- config / manager / base ----------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"webhook_path": "telegram"},
        {"mode": "webhook", "webhook_url": "https://", "webhook_secret_token": "s"},
        {"mode": "webhook", "webhook_url": "https://x.example/t", "webhook_secret_token": "a" * 257},
        {"mode": "webhook", "webhook_url": "https://x.example/t", "webhook_secret_token": "bad secret!"},
        {"stream_edit_interval": 0.05},
        {"webhook_listen_port": 0},
        {"webhook_listen_port": 65536},
        {"webhook_max_connections": 0},
        {"webhook_max_connections": 101},
    ],
)
def test_telegram_config_rejects(fields: dict) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TelegramConfig(**fields)


def test_telegram_config_defaults() -> None:
    from nanoclaw.config.schema import ChannelsConfig

    cfg = TelegramConfig(webhook_path="  ")
    assert cfg.webhook_path == "/telegram"
    assert (cfg.enabled, cfg.reply_to_message, cfg.webhook_listen_host, cfg.allow_from) == (False, False, "127.0.0.1", [])
    assert (cfg.send_progress, cfg.send_tool_hints, cfg.show_reasoning, cfg.show_compaction_notices) == (None,) * 4
    assert ChannelsConfig().show_compaction_notices is False
    assert TelegramConfig(webhook_secret_token="a" * 256, mode="webhook", webhook_url="https://x.example/t")


def test_mask_secrets_hides_channel_tokens() -> None:
    from nanoclaw.config.loader import mask_secrets

    secret_token = "123456:abcdefghijklmnopqrstuvwxyz"
    secret_hook = "hook-secret-0123456789"
    masked = mask_secrets({"channels": {"telegram": {"token": secret_token, "webhookSecretToken": secret_hook}}})
    assert secret_token not in str(masked) and secret_hook not in str(masked)


async def test_send_reasoning_skips_empty_and_keeps_stream_id() -> None:
    from nanoclaw.channels.base import BaseChannel

    assert BaseChannel.projects_outbound is True
    fake = SimpleNamespace(send_reasoning_delta=AsyncMock(), send_reasoning_end=AsyncMock())
    await BaseChannel.send_reasoning(fake, _out("", event=ProgressEvent(reasoning=True, stream_id="r1")))  # type: ignore[arg-type]
    fake.send_reasoning_delta.assert_not_called()
    await BaseChannel.send_reasoning(fake, _out("why", event=ProgressEvent(reasoning=True, stream_id="r1")))  # type: ignore[arg-type]
    fake.send_reasoning_delta.assert_awaited_once_with("5", "why", {}, stream_id="r1")
    fake.send_reasoning_end.assert_awaited_once_with("5", {}, stream_id="r1")


def _manager(channels: dict):
    from nanoclaw.channels.manager import ChannelManager
    from nanoclaw.config.schema import Config

    return ChannelManager(Config.model_validate({"channels": channels}), MessageBus())


def test_manager_builds_telegram_with_flags_and_logs_missing_allow_from() -> None:
    from loguru import logger

    lines: list[str] = []
    logger.enable("nanoclaw")  # NanoClaw's logger is off outside the CLI
    sink = logger.add(lambda m: lines.append(m.record["message"]), level="INFO")
    try:
        assert "telegram" not in _manager({"telegram": {"token": "1:a"}}).channels
        manager = _manager({
            "showCompactionNotices": False,
            "sendProgress": True,
            "telegram": {"enabled": True, "token": "1:a", "showCompactionNotices": True, "sendProgress": False},
        })
        channel = manager.channels["telegram"]
        assert channel.show_compaction_notices is True and channel.send_progress is False
        assert any('"telegram" has no allowFrom' in line for line in lines)
        lines.clear()
        _manager({"telegram": {"enabled": True, "token": "1:a", "allowFrom": ["1"]}})
        assert not any("has no allowFrom" in line for line in lines)
    finally:
        logger.remove(sink)
        logger.disable("nanoclaw")


def test_manager_reports_telegram_load_failure(monkeypatch) -> None:
    import nanoclaw.channels.telegram as telegram_pkg

    class Broken:
        def __init__(self, *args) -> None:
            raise RuntimeError("no telegram")

    monkeypatch.setattr(telegram_pkg, "TelegramChannel", Broken)
    manager = _manager({"telegram": {"enabled": True, "token": "1:a"}})
    assert "telegram" not in manager.channels
    assert manager.get_status()["telegram"]["error"] == "Channel runtime could not be loaded. Check gateway logs."


def test_drop_filters_reasoning_variants() -> None:
    from nanoclaw.channels.manager import ChannelManager

    channel = SimpleNamespace(show_reasoning=False, send_progress=True, send_tool_hints=True)
    for event in (ProgressEvent(reasoning_end=True), ProgressEvent(reasoning=True, content="x")):
        assert ChannelManager._drop_for_channel(channel, _out(event=event)) is True  # type: ignore[arg-type]
    channel.show_reasoning = True
    assert ChannelManager._drop_for_channel(channel, _out(event=ProgressEvent(reasoning_end=True))) is False  # type: ignore[arg-type]


async def test_projecting_channel_gets_every_event_through_send() -> None:
    from nanoclaw.bus.outbound_events import StreamDeltaEvent, StreamEndEvent
    from nanoclaw.channels.manager import ChannelManager

    projecting = SimpleNamespace(projects_outbound=True, send=AsyncMock(), send_delta=AsyncMock())
    await ChannelManager._send_once(projecting, _out(event=StreamDeltaEvent(content="x", stream_id="s")))  # type: ignore[arg-type]
    projecting.send.assert_awaited_once()
    projecting.send_delta.assert_not_called()

    per_event = SimpleNamespace(
        projects_outbound=False, send_delta=AsyncMock(), send_reasoning_delta=AsyncMock(), send_reasoning_end=AsyncMock()
    )
    await ChannelManager._send_once(per_event, _out("", event=StreamEndEvent(stream_id="s", resuming=True)))  # type: ignore[arg-type]
    assert per_event.send_delta.await_args.kwargs["resuming"] is True
    await ChannelManager._send_once(per_event, _out("r", event=ProgressEvent(reasoning_delta=True, stream_id="r1")))  # type: ignore[arg-type]
    assert per_event.send_reasoning_delta.await_args.kwargs == {"stream_id": "r1"}
    await ChannelManager._send_once(per_event, _out("", event=ProgressEvent(reasoning_end=True, stream_id="r2")))  # type: ignore[arg-type]
    assert per_event.send_reasoning_end.await_args.kwargs == {"stream_id": "r2"}


async def test_projecting_channel_is_not_filtered_by_dispatcher() -> None:
    from nanoclaw.channels.base import BaseChannel
    from nanoclaw.channels.manager import ChannelManager
    from nanoclaw.config.schema import Config

    class Projecting(BaseChannel):
        name = "proj"

        def __init__(self, bus: MessageBus) -> None:
            super().__init__({"allow_from": ["*"]}, bus)
            self.sent: list[str] = []

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def send(self, msg: OutboundMessage) -> None:
            self.sent.append(msg.content)

    bus = MessageBus()
    channel = Projecting(bus)
    channel.send_progress = False
    manager = ChannelManager(Config(), bus, channels={"proj": channel})
    await bus.publish_outbound(OutboundMessage(channel="proj", chat_id="1", content="step", event=ProgressEvent()))
    task = asyncio.create_task(manager._dispatch_outbound_loop())
    try:
        async with asyncio.timeout(5):
            while not channel.sent:
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert channel.sent == ["step"]


async def test_websocket_inbound_is_never_a_dm() -> None:
    from nanoclaw.channels.websocket.runtime import WebSocketChannel

    fake = SimpleNamespace(_handle_message=AsyncMock(return_value=True))
    await WebSocketChannel.dispatch_inbound(fake, sender_id="s", chat_id="c", content="x", metadata={})  # type: ignore[arg-type]
    assert fake._handle_message.await_args.kwargs["is_dm"] is False
