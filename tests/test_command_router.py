"""The command router and built-in commands (nanobot command/router.py, command/builtin.py)."""

import asyncio

import pytest

from nanoclaw.agent.loop import DREAM_NO_INPUT_MESSAGE, AgentLoop
from nanoclaw.bus import InboundMessage, MessageBus, OutboundMessage
from nanoclaw.command import (
    BUILTIN_COMMAND_SPECS,
    CommandContext,
    CommandRouter,
    command_text,
    normalize_command_text,
    register_builtin_commands,
)
from nanoclaw.command.builtin import builtin_command_palette, builtin_command_starts_agent_turn
from nanoclaw.events import ContextCompactionEvent, EventSink
from nanoclaw.providers import FakeProvider
from nanoclaw.session import SessionManager
from nanoclaw.utils.templates import sync_workspace_templates


def make_ctx(text, loop=None, **kwargs):
    msg = InboundMessage("cli", "user", "direct", text, metadata=kwargs.pop("metadata", {}))
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw=text, loop=loop, **kwargs)


def recorder(name, calls):
    async def handler(ctx):
        calls.append((name, ctx.raw, ctx.args))
        return OutboundMessage(ctx.msg.channel, ctx.msg.chat_id, name)

    return handler


@pytest.fixture
def loop(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    sync_workspace_templates(ws)
    return AgentLoop(FakeProvider([]), ws, sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"))


@pytest.fixture
def router():
    router = CommandRouter()
    register_builtin_commands(router)
    return router


# --- router ----------------------------------------------------------------


def test_normalize_and_command_text():
    assert normalize_command_text("  /new@nanoclaw  ") == "/new"
    assert normalize_command_text("/history@bot 5 @x") == "/history 5 @x"
    assert normalize_command_text("/tmp/file@new") == "/tmp/file"  # as nanobot; still not a command
    assert normalize_command_text("/help@bad-name") == "/help@bad-name"
    assert normalize_command_text(" hello@bot ") == "hello@bot"
    assert command_text(InboundMessage("cli", "u", "d", "  /new ")) == "/new"
    produced = InboundMessage("cli", "u", "d", "/new", metadata={"original_command": "/goal"})
    assert command_text(produced) == ""


async def test_tiers_and_longest_prefix_first():
    calls = []
    router = CommandRouter()
    router.priority("/stop", recorder("stop", calls))
    router.exact("/team", recorder("team", calls))
    router.prefix("/team ", recorder("team-args", calls))
    router.prefix("/team add ", recorder("team-add", calls))

    assert router.is_priority("/STOP@bot") and not router.is_priority("/stop now")
    assert (await router.dispatch_priority(make_ctx("/Stop@bot"))).content == "stop"
    assert await router.dispatch_priority(make_ctx("/team")) is None

    assert (await router.dispatch(make_ctx("/team"))).content == "team"
    assert (await router.dispatch(make_ctx("/team add Bob"))).content == "team-add"
    assert (await router.dispatch(make_ctx("/team list"))).content == "team-args"
    assert calls[-2:] == [("team-add", "/team add Bob", "Bob"), ("team-args", "/team list", "list")]
    assert await router.dispatch(make_ctx("hello")) is None


async def test_dispatch_skips_command_produced_input():
    calls = []
    router = CommandRouter()
    router.exact("/new", recorder("new", calls))
    assert await router.dispatch(make_ctx("/new", metadata={"original_command": "/goal"})) is None
    assert calls == []


def test_session_queue_flag(router):
    assert router.requires_session_queue("/compact")
    assert router.requires_session_queue("/COMPACT@bot")
    assert not router.requires_session_queue("/compact now")
    assert not router.requires_session_queue("/new")
    router.exact("/compact", router._exact["/compact"])  # re-registering without the flag clears it
    assert not router.requires_session_queue("/compact")

    other = CommandRouter()
    other.priority("/x", recorder("x", []))
    other.exact("/x", recorder("x", []), session_queue=True)
    assert not other.requires_session_queue("/x")  # priority wins


@pytest.mark.parametrize(
    "text, priority, dispatchable, name",
    [
        ("/stop", True, False, "/stop"),
        ("/stop now", False, True, "/stop"),
        ("/status", True, False, "/status"),
        ("/new", False, True, "/new"),
        ("/NEW@nanobot", False, True, "/new"),
        ("/history 5", False, True, "/history"),
        ("/history\t5", False, True, "/history"),
        ("/historyx", False, False, None),
        ("/tmp/foo.py explain", False, False, None),
        ("/", False, False, None),
        ("hello", False, False, None),
    ],
)
def test_classification(router, text, priority, dispatchable, name):
    """Same answers as nanobot c4dac64's router with its built-in commands."""
    assert router.is_priority(text) is priority
    assert router.is_dispatchable_command(text) is dispatchable
    assert router.command_name(text) == name


async def test_invalid_use_is_answered_as_text(router):
    ctx = make_ctx("/new@nanoclaw now", metadata={"message_id": 3})
    reply = await router.dispatch(ctx)
    assert reply.content == 'Command "/new" does not accept arguments. Did you mean "/new"?'
    assert reply.metadata == {"message_id": 3, "render_as": "text"}
    reply = await router.dispatch(make_ctx("/HISTORY\t5"))
    assert reply.content == 'Invalid command "/HISTORY". Use "/help" to list available commands.'
    assert router.invalid_command_text("/hello") is None


# --- built-in specs ----------------------------------------------------------


def test_specs_match_nanobot():
    """nanobot command/builtin.py:70 for the commands NanoClaw has."""
    assert [spec.as_dict() for spec in BUILTIN_COMMAND_SPECS] == builtin_command_palette()
    table = {s.command: (s.title, s.icon, s.arg_hint, s.lifecycle, s.accepts_args) for s in BUILTIN_COMMAND_SPECS}
    assert table == {
        "/new": ("New chat", "square-pen", "", "finalize_active_turn", False),
        "/compact": ("Compact context", "archive", "", "side_channel", False),
        "/stop": ("Stop current task", "square", "", "stop_active_turn", False),
        "/status": ("Show status", "activity", "", "side_channel", False),
        "/model": ("Switch model preset", "brain", "[preset]", "side_channel", True),
        "/history": ("Show conversation history", "history", "[n]", "side_channel", True),
        "/dream": ("Run Dream", "sparkles", "", "side_channel", False),
        "/dream-log": ("Show Dream log", "book-open", "", "side_channel", True),
        "/dream-restore": ("Restore memory", "undo-2", "", "side_channel", True),
        "/dream-prompt": ("Dream memory", "file-text", "[init]", "side_channel", True),
        "/evaluator-prompt": ("Heartbeat evaluator", "file-text", "[init]", "side_channel", True),
        "/skill": ("List skills", "wrench", "", "side_channel", False),
        "/help": ("Show help", "circle-help", "", "side_channel", False),
        "/pairing": (
            "Manage pairing", "shield", "[list|approve <code>|deny <code>|revoke <user_id>]", "side_channel", True,
        ),
    }


def test_every_spec_is_registered(router):
    assert {s.command for s in BUILTIN_COMMAND_SPECS} == set(router._registered_commands().values())


def test_starts_agent_turn():
    assert not builtin_command_starts_agent_turn("/new")
    assert not builtin_command_starts_agent_turn("/history 5")
    assert builtin_command_starts_agent_turn("/new now")  # wrong args: not a side channel
    assert builtin_command_starts_agent_turn("hello")


# --- built-in handlers -------------------------------------------------------


async def test_help_and_skill_replies(router, loop):
    reply = await router.dispatch(make_ctx("/help", loop))
    assert reply.content.startswith("nanoclaw commands:\n/new — Reset")
    assert reply.metadata == {"render_as": "text"}
    reply = await router.dispatch(make_ctx("/skill", loop))
    assert reply.content.startswith("Available skills (") and "render_as" not in reply.metadata


async def test_history_and_new(router, loop):
    session = loop.sessions.get_or_create("cli:direct")
    session.add_message("user", "question")
    session.add_message("assistant", "answer")
    loop.sessions.save(session)
    reply = await router.dispatch(make_ctx("/history 1", loop))
    assert reply.content == "Last 1 message(s):\n🤖 Bot: answer" and reply.metadata["render_as"] == "text"
    reply = await router.dispatch(make_ctx("/history x", loop))
    assert reply.content.startswith("Usage: /history [count]")
    assert (await router.dispatch(make_ctx("/new", loop))).content == "New session started."
    assert (await router.dispatch(make_ctx("/history", loop))).content == "No conversation history yet."


async def test_status_is_priority_and_plain_text(router, loop):
    reply = await router.dispatch_priority(make_ctx("/status", loop))
    assert "🧠 Model: fake-model" in reply.content and "⚡ Tasks: 0 active" in reply.content
    assert reply.metadata["render_as"] == "text"


async def test_stop_falls_back_to_subagents(router, loop):
    cancelled = []

    async def cancel_by_session(key):
        cancelled.append(key)
        return 2

    loop.subagents.cancel_by_session = cancel_by_session
    reply = await router.dispatch_priority(make_ctx("/stop", loop))
    assert reply.content == "Stopped 2 task(s)." and cancelled == ["cli:direct"]


async def test_stop_uses_the_loop_hook_and_drains_pending_input(router, loop):
    """Once the loop has ``cancel_active_tasks`` and pending queues (later in layer 14)."""
    calls = []

    async def cancel_active_tasks(key):
        calls.append(key)
        return 1

    loop.cancel_active_tasks = cancel_active_tasks
    pending = asyncio.Queue()
    pending.put_nowait("queued")
    loop._pending_queues = {"cli:direct": pending}
    reply = await router.dispatch_priority(make_ctx("/stop", loop))
    assert reply.content == "Stopped 2 task(s)." and calls == ["cli:direct"]
    assert loop._pending_queues == {}

    async def nothing(key):
        return 0

    loop.cancel_active_tasks = nothing
    assert (await router.dispatch_priority(make_ctx("/stop", loop))).content == "No active task to stop."


async def test_compact_sends_events_and_no_reply(router, loop):
    session = loop.sessions.get_or_create("cli:direct")
    session.add_message("user", "hello")
    loop.sessions.save(session)
    loop.provider = FakeProvider(["- summary"])
    events = []

    async def collect(event):
        events.append(event)

    assert await router.dispatch(make_ctx("/compact", loop, events=EventSink(collect))) is None
    phases = [e.phase for e in events if isinstance(e, ContextCompactionEvent)]
    assert phases == ["started", "succeeded"]


async def test_dream_replies_then_follows_up_on_the_bus(router, loop):
    bus = MessageBus()
    ctx = make_ctx("/dream", loop, bus=bus)
    reply = await router.dispatch(ctx)
    assert reply.content == "Dreaming..."
    assert len(ctx.tasks) == 1
    await asyncio.gather(*ctx.tasks)
    followup = await bus.consume_outbound()
    assert followup.content == DREAM_NO_INPUT_MESSAGE
    assert followup.metadata == {"render_as": "text"}
    assert (followup.channel, followup.chat_id) == ("cli", "direct")


async def test_status_counts_only_unfinished_session_tasks(router, loop):
    """nanobot builtin.py:295: a finished task still in the set is not active."""
    finished = asyncio.create_task(asyncio.sleep(0))
    await finished
    running = asyncio.create_task(asyncio.Event().wait())
    loop._active_tasks["cli:direct"] = {finished, running}
    try:
        reply = await router.dispatch_priority(make_ctx("/status", loop))
    finally:
        running.cancel()
    assert "⚡ Tasks: 1 active" in reply.content


async def test_new_from_the_bus_loop_waits_for_the_session_lock(router, loop):
    """Outside the session's worker (no ``ctx.session``) /new takes the session
    lock itself (research D6), so it cannot clear a session mid-turn."""
    session = loop.sessions.get_or_create("cli:direct")
    session.add_message("user", "hello")
    loop.sessions.save(session)
    lock = loop._get_session_lock("cli:direct")
    await lock.acquire()
    reset = asyncio.create_task(router.dispatch(make_ctx("/new", loop)))
    await asyncio.sleep(0.05)
    assert not reset.done()
    assert [m["content"] for m in loop.sessions.get_or_create("cli:direct").messages] == ["hello"]
    lock.release()
    assert (await asyncio.wait_for(reset, 5)).content == "New session started."
    assert loop.sessions.get_or_create("cli:direct").messages == []
