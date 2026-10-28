"""The ``/pairing`` command.

Ported from nanobot tests/command/test_pairing_command.py and the pairing
parts of tests/command/test_router_dispatchable.py (c4dac64).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nanoclaw.bus.events import InboundMessage
from nanoclaw.bus.queue import MessageBus
from nanoclaw.command.builtin import build_help_text, cmd_pairing, register_builtin_commands
from nanoclaw.command.router import CommandContext, CommandRouter
from nanoclaw.pairing import store
from nanoclaw.pairing.events import PairingChanged


@pytest.fixture()
def router() -> CommandRouter:
    r = CommandRouter()
    register_builtin_commands(r)
    return r


async def test_pairing_command_notifies_after_persistence_without_a_file_watcher(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_store_path", lambda: tmp_path / "pairing.json")
    code = store.generate_code("telegram", "alice")
    bus = MessageBus()
    approved = []
    bus.subscribe(lambda _event: approved.append(store.is_approved("telegram", "alice")), PairingChanged)
    loop = MagicMock(bus=bus)
    msg = InboundMessage(channel="cli", sender_id="owner", chat_id="direct", content=f"/pairing approve {code}")
    ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=msg.content, args=f"approve {code}", loop=loop)
    result = await cmd_pairing(ctx)
    await bus.drain()
    assert "Approved" in result.content and approved == [True]
    ctx.args = "approve invalid"
    await cmd_pairing(ctx)
    ctx.args = "list"
    await cmd_pairing(ctx)
    await bus.drain()
    assert approved == [True]


def test_pairing_is_dispatchable(router: CommandRouter) -> None:
    assert router.is_dispatchable_command("/pairing")
    assert router.is_dispatchable_command("/pairing list")
    assert router.is_dispatchable_command("/pairing approve CODE")
    assert router.is_dispatchable_command("/PAIRING")
    assert router.is_dispatchable_command("  /pairing list  ")


def test_help_lists_pairing() -> None:
    assert "/pairing [list|approve <code>|deny <code>|revoke <user_id>]" in build_help_text()


class TestPairingCommandDispatch:
    """Verify /pairing works via CommandRouter."""

    @pytest.fixture()
    def fake_msg(self) -> MagicMock:
        msg = MagicMock()
        msg.channel = "telegram"
        msg.chat_id = "chat1"
        msg.content = "/pairing list"
        msg.metadata = {}
        return msg

    async def test_pairing_list_dispatched(self, router: CommandRouter, fake_msg: MagicMock, monkeypatch) -> None:
        monkeypatch.setattr(
            "nanoclaw.pairing.store.list_pending",
            lambda: [
                {
                    "code": "ABCD-EFGH",
                    "channel": "telegram",
                    "sender_id": "123",
                    "expires_at": 9999999999,
                }
            ],
        )
        ctx = CommandContext(
            msg=fake_msg, session=None,
            key="telegram:chat1", raw="/pairing list", args="list", loop=MagicMock(),
        )
        result = await router.dispatch(ctx)
        assert result is not None
        assert "ABCD-EFGH" in result.content
        assert result.metadata.get("_pairing_command") is True

    async def test_pairing_approve_dispatched(self, router: CommandRouter, fake_msg: MagicMock, monkeypatch) -> None:
        monkeypatch.setattr(
            "nanoclaw.pairing.store.approve_code",
            lambda code, *, on_change=None: ("telegram", "123") if code == "ABCD-EFGH" else None,
        )
        fake_msg.content = "/pairing approve ABCD-EFGH"
        ctx = CommandContext(
            msg=fake_msg, session=None,
            key="telegram:chat1", raw="/pairing approve ABCD-EFGH",
            args="approve ABCD-EFGH", loop=MagicMock(),
        )
        result = await router.dispatch(ctx)
        assert result is not None
        assert result.content == "Approved pairing code `ABCD-EFGH` — 123 can now access telegram"
        assert result.metadata.get("_pairing_command") is True

    async def test_pairing_revoke_dispatched(self, router: CommandRouter, fake_msg: MagicMock, monkeypatch) -> None:
        monkeypatch.setattr(
            "nanoclaw.pairing.store.revoke",
            lambda ch, sid, *, on_change=None: sid == "123",
        )
        fake_msg.content = "/pairing revoke 123"
        ctx = CommandContext(
            msg=fake_msg, session=None,
            key="telegram:chat1", raw="/pairing revoke 123",
            args="revoke 123", loop=MagicMock(),
        )
        result = await router.dispatch(ctx)
        assert result is not None
        assert result.content == "Revoked 123 from telegram"
        assert result.metadata.get("_pairing_command") is True
