"""The Telegram channel inside a running gateway (task 24.1).

``nanoclaw gateway`` with ``channels.telegram`` enabled and a fake
python-telegram-bot ``Application`` (no token, no network): an update from an
allowed user runs an agent turn on the FakeProvider and the reply comes back
through the fake bot (split, rendered as HTML); an unknown DM sender gets a
pairing code and no turn; the heartbeat then targets the Telegram chat and
delivers there.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from nanoclaw.__main__ import main
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.tools.mcp import MCPProvider
from nanoclaw.channels.telegram import TelegramChannel
from nanoclaw.channels.telegram.runtime import (
    TELEGRAM_HTML_MAX_LEN,
    _markdown_to_telegram_html,
    _split_telegram_markdown,
)
from nanoclaw.cli.gateway import Gateway
from nanoclaw.config import Config
from nanoclaw.config.loader import save_config, set_config_path
from nanoclaw.cron.service import CronService
from nanoclaw.cron.types import CronJob
from nanoclaw.providers import FakeProvider
from nanoclaw.providers.base import LLMResponse, ToolCallRequest
from nanoclaw.session import SessionManager
from nanoclaw.utils.templates import sync_workspace_templates

USER_ID = 12345
CHAT_ID = 12345


class FakeBot:
    """Records every Bot API call in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._next_id = 100

    def _record(self, method: str, kwargs: dict[str, Any]) -> SimpleNamespace:
        self.calls.append((method, kwargs))
        self._next_id += 1
        return SimpleNamespace(message_id=self._next_id)

    def of(self, method: str) -> list[dict[str, Any]]:
        return [kwargs for name, kwargs in self.calls if name == method]

    async def get_me(self):
        return SimpleNamespace(id=999, username="nanoclaw_test_bot")

    async def set_my_commands(self, commands) -> None:
        self._record("set_my_commands", {"commands": commands})

    async def send_message(self, **kwargs):
        return self._record("send_message", kwargs)

    async def edit_message_text(self, **kwargs):
        return self._record("edit_message_text", kwargs)

    async def set_message_reaction(self, **kwargs):
        return self._record("set_message_reaction", kwargs)

    async def send_chat_action(self, **kwargs):
        return self._record("send_chat_action", kwargs)

    async def shutdown(self) -> None:
        pass


class FakeUpdater:
    def __init__(self) -> None:
        self.polling: dict[str, Any] | None = None

    async def start_polling(self, **kwargs) -> None:
        self.polling = kwargs

    async def stop(self) -> None:
        pass


class FakeApp:
    def __init__(self) -> None:
        self.bot = FakeBot()
        self.updater = FakeUpdater()
        self.handlers: list[Any] = []

    def add_error_handler(self, handler) -> None:
        pass

    def add_handler(self, handler) -> None:
        self.handlers.append(handler)

    async def initialize(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class FakeBuilder:
    def __init__(self, app: FakeApp) -> None:
        self.app = app

    def token(self, _token):
        return self

    def request(self, _request):
        return self

    def get_updates_request(self, _request):
        return self

    def build(self):
        return self.app


@pytest.fixture
def fake_app(monkeypatch) -> FakeApp:
    app = FakeApp()
    monkeypatch.setattr("nanoclaw.channels.telegram.runtime.HTTPXRequest", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(
        "nanoclaw.channels.telegram.runtime.Application", SimpleNamespace(builder=lambda: FakeBuilder(app))
    )
    return app


@pytest.fixture(autouse=True)
def config_dir(tmp_path):
    """Pairing codes and media live next to the config file."""
    set_config_path(tmp_path / "config.json")
    yield tmp_path
    set_config_path(None)


class _NoMCP(MCPProvider):
    def __init__(self) -> None:
        super().__init__({}, MagicMock())

    async def connect(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


def make_agent(tmp_path: Path, *script) -> AgentLoop:
    ws = tmp_path / "ws"
    sync_workspace_templates(ws)
    return AgentLoop(
        FakeProvider(list(script)),
        ws,
        sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"),
        cron_service=CronService(ws / "cron" / "jobs.json"),
    )


def telegram_config(**overrides: Any) -> Config:
    section = {"enabled": True, "token": "123456:test-token", "allowFrom": [str(USER_ID)], **overrides}
    return Config.model_validate({"channels": {"telegram": section}})


def private_update(text: str, *, user_id: int = USER_ID, message_id: int = 1, update_id: int = 1):
    user = SimpleNamespace(id=user_id, username=None, first_name="Alice")
    message = SimpleNamespace(
        chat=SimpleNamespace(type="private", is_forum=False),
        chat_id=user_id,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        reply_to_message=None,
        location=None,
        media_group_id=None,
        message_thread_id=None,
        message_id=message_id,
    )
    return SimpleNamespace(message=message, effective_user=user, update_id=update_id)


async def _until(predicate, timeout: float = 10) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


async def _run_gateway(gateway: Gateway, body) -> None:
    run = asyncio.create_task(gateway.run())
    try:
        channel = gateway.channels.get_channel("telegram")
        await _until(lambda: channel is not None and channel._app_ready.is_set())
        await body(channel)
    finally:
        gateway.shutdown_event.set()
        await asyncio.wait_for(run, 15)


LONG_REPLY = (
    "# Report\n\n**Status**: all good & <fine>\n\n"
    "| Name | Count |\n|---|---|\n| apples | 3 |\n| pears | 12 |\n\n"
    "```python\nprint('x < y')\n```\n\n" + "\n".join(f"- item {i}: " + " ".join(["word"] * 12) for i in range(120))
)


async def test_allowed_user_gets_an_agent_reply_split_and_rendered(tmp_path, fake_app) -> None:
    assert len(LONG_REPLY) > 4000
    agent = make_agent(tmp_path, LONG_REPLY)
    gateway = Gateway(
        telegram_config(streaming=False), agent, _NoMCP(), console=Console(record=True, width=200),
        health_server_enabled=False,
    )
    assert gateway.enabled_channels == ["telegram"]
    bot = fake_app.bot

    async def body(channel: TelegramChannel) -> None:
        await channel._on_message(private_update("hello bot"), None)
        await _until(lambda: len(bot.of("send_message")) >= 3 and bot.of("set_message_reaction")[-1]["reaction"] == [])

    await _run_gateway(gateway, body)

    assert fake_app.updater.polling == {
        "allowed_updates": ["message"],
        "drop_pending_updates": False,
        "error_callback": gateway.channels.get_channel("telegram")._on_polling_error,
    }
    # The user's text reached the model.
    assert agent.provider.calls[0].messages[-1]["content"].startswith("hello bot")
    sends = bot.of("send_message")
    chunks = _split_telegram_markdown(LONG_REPLY, 4000)
    assert len(chunks) == 3
    assert [s["text"] for s in sends] == [_markdown_to_telegram_html(c) for c in chunks]
    assert all(s["parse_mode"] == "HTML" and s["chat_id"] == CHAT_ID for s in sends)
    assert all(len(s["text"]) <= TELEGRAM_HTML_MAX_LEN for s in sends)
    first = sends[0]["text"]
    assert first.startswith("<b>Report</b>")
    assert "<b>Status</b>: all good &amp; &lt;fine&gt;" in first
    assert "<pre><code>Name    Count\n──────  ─────\napples  3    \npears   12   </code></pre>" in first
    assert "<pre><code>print('x &lt; y')\n</code></pre>" in first
    # 👀 on the user's message while working, removed with the reply; typing shown.
    reactions = bot.of("set_message_reaction")
    assert reactions[0]["message_id"] == 1 and reactions[0]["reaction"][0].emoji == "👀"
    assert reactions[-1] == {"chat_id": CHAT_ID, "message_id": 1, "reaction": []}
    assert bot.of("send_chat_action")[0] == {"chat_id": CHAT_ID, "action": "typing"}
    # The session is the Telegram chat's.
    assert agent.sessions.get_existing(f"telegram:{CHAT_ID}") is not None


async def test_streamed_reply_is_sent_then_edited_into_html(tmp_path, fake_app) -> None:
    agent = make_agent(tmp_path, "Here is **bold** and `code`.")
    gateway = Gateway(
        telegram_config(), agent, _NoMCP(), console=Console(record=True, width=200), health_server_enabled=False
    )
    bot = fake_app.bot

    async def body(channel: TelegramChannel) -> None:
        await channel._on_message(private_update("stream please"), None)
        await _until(lambda: bot.of("edit_message_text"))

    await _run_gateway(gateway, body)

    (preview,) = bot.of("send_message")
    assert preview == {"chat_id": CHAT_ID, "text": "Here is bold and code."}
    final = bot.of("edit_message_text")[-1]
    assert final == {
        "chat_id": CHAT_ID,
        "message_id": 101 + [m for m, _ in bot.calls].index("send_message"),
        "text": "Here is <b>bold</b> and <code>code</code>.",
        "parse_mode": "HTML",
    }


async def test_unknown_dm_sender_gets_a_pairing_code_and_no_turn(tmp_path, fake_app, config_dir) -> None:
    agent = make_agent(tmp_path)
    gateway = Gateway(
        telegram_config(), agent, _NoMCP(), console=Console(record=True, width=200), health_server_enabled=False
    )
    bot = fake_app.bot

    async def body(channel: TelegramChannel) -> None:
        await channel._on_message(private_update("hi?", user_id=777), None)
        await _until(lambda: bot.of("send_message"))

    await _run_gateway(gateway, body)

    (reply,) = bot.of("send_message")
    store = json.loads((config_dir / "pairing.json").read_text(encoding="utf-8"))
    ((code, entry),) = store["pending"].items()
    assert entry["channel"] == "telegram" and entry["sender_id"] == "777"
    assert entry["delivery_status"] == "sent"
    assert reply["chat_id"] == 777 and reply["parse_mode"] == "HTML"
    assert f"Your pairing code is: <code>{code}</code>" in reply["text"]
    assert agent.provider.calls == []
    assert agent.sessions.get_existing("telegram:777") is None


async def test_unknown_sender_in_a_group_is_ignored(tmp_path, fake_app) -> None:
    agent = make_agent(tmp_path)
    gateway = Gateway(
        telegram_config(), agent, _NoMCP(), console=Console(record=True, width=200), health_server_enabled=False
    )
    bot = fake_app.bot

    async def body(channel: TelegramChannel) -> None:
        update = private_update("@nanoclaw_test_bot hi", user_id=777)
        update.message.chat.type = "group"
        update.message.chat_id = -100
        await channel._on_message(update, None)
        await asyncio.sleep(0.5)

    await _run_gateway(gateway, body)
    assert bot.of("send_message") == [] and agent.provider.calls == []


def _eval_tool_call(should_notify: bool) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCallRequest(id="eval_1", name="evaluate_notification", arguments={"should_notify": should_notify})
        ],
    )


async def test_heartbeat_targets_the_telegram_chat_and_delivers_there(tmp_path, fake_app) -> None:
    agent = make_agent(tmp_path, "Hi Alice.", "The build is red.", _eval_tool_call(True))
    (agent.workspace / "HEARTBEAT.md").write_text("## Active Tasks\n\n- Check the build\n", encoding="utf-8")
    gateway = Gateway(
        telegram_config(streaming=False), agent, _NoMCP(), console=Console(record=True, width=200),
        health_server_enabled=False,
    )
    bot = fake_app.bot

    async def body(channel: TelegramChannel) -> None:
        await channel._on_message(private_update("hello"), None)
        await _until(lambda: bot.of("send_message"))
        assert gateway.pick_heartbeat_target() == ("telegram", str(CHAT_ID))
        result = await gateway.on_cron_job(CronJob(id="heartbeat", name="heartbeat"))
        assert result == "The build is red."
        await _until(lambda: len(bot.of("send_message")) == 2)

    await _run_gateway(gateway, body)
    assert [s["text"] for s in bot.of("send_message")] == ["Hi Alice.", "The build is red."]
    assert bot.of("send_message")[1]["chat_id"] == CHAT_ID


def test_banner_lists_telegram(tmp_path, fake_app) -> None:
    console = Console(record=True, width=200)
    gateway = Gateway(telegram_config(), make_agent(tmp_path), _NoMCP(), console=console)
    gateway.print_banner()
    assert "Channels enabled: telegram" in console.export_text()
    assert gateway.enabled_channels == ["telegram"]
    assert gateway.channels.get_status()["telegram"]["state"] == "stopped"


def test_cli_channels_status_and_validate(tmp_path, capsys, monkeypatch) -> None:
    path = tmp_path / "config.json"
    save_config(telegram_config(), path)
    assert main(["channels", "status", "-c", str(path)]) == 0
    assert "Telegram  yes" in capsys.readouterr().out

    from nanoclaw.channels.telegram import validation

    monkeypatch.setattr(validation, "_get_me", lambda token, proxy: {"ok": True, "result": {"id": 42, "username": "b"}})
    assert main(["channels", "validate", "telegram", "-c", str(path)]) == 1  # "test-token" is too short
    out = capsys.readouterr().out
    assert "telegram: invalid" in out and "test-token" not in out
