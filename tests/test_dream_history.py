"""Git history of the memory files and prompt overrides (layer 21.3).

Ported from nanobot tests/command/test_builtin_dream.py (/dream-log,
/dream-restore, /dream-prompt), tests/agent/test_dream.py (Dream prompt
override, diff-grounded commit messages, ``dream_content_diff``) and
nanobot's workspace prompt helpers; plus NanoClaw's own end-to-end checks of
``/dream`` and ``nanoclaw dream`` committing through the real loop.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from types import SimpleNamespace

import pytest

from nanoclaw.__main__ import main
from nanoclaw.agent.loop import DREAM_NO_INPUT_MESSAGE, AgentLoop
from nanoclaw.agent.memory import MemoryStore, commit_dream_changes
from nanoclaw.bus import InboundMessage, MessageBus, OutboundMessage
from nanoclaw.command import CommandContext, CommandRouter, register_builtin_commands
from nanoclaw.command.builtin import (
    build_help_text,
    builtin_command_palette,
    cmd_dream_log,
    cmd_dream_prompt,
    cmd_dream_restore,
)
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.session import SessionManager
from nanoclaw.utils.gitstore import CommitInfo
from nanoclaw.utils.templates import sync_workspace_templates
from nanoclaw.utils.workspace_prompts import (
    WORKSPACE_PROMPT_MAX_CHARS,
    has_workspace_prompt_override,
    initialize_workspace_prompt,
    load_workspace_prompt_override,
    workspace_prompt_file,
)

# --- fakes (nanobot test_builtin_dream.py) -------------------------------------------------


class _FakeStore:
    def __init__(self, git, last_dream_cursor: int = 1):
        self.git = git
        self._last_dream_cursor = last_dream_cursor

    def get_last_dream_cursor(self) -> int:
        return self._last_dream_cursor


class _FakeGit:
    def __init__(
        self,
        *,
        initialized: bool = True,
        commits: list[CommitInfo] | None = None,
        diff_map: dict[str, tuple[CommitInfo, str] | None] | None = None,
        revert_result: str | None = None,
    ):
        self._initialized = initialized
        self._commits = commits or []
        self._diff_map = diff_map or {}
        self._revert_result = revert_result
        self.revert_calls: list[tuple[str, str | None]] = []

    def is_initialized(self) -> bool:
        return self._initialized

    def log(self, max_entries: int = 20, message_prefix: str | None = None) -> list[CommitInfo]:
        commits = self._commits
        if message_prefix is not None:
            commits = [c for c in commits if c.message.startswith(message_prefix)]
        return commits[:max_entries]

    def show_commit_diff(self, sha: str, max_entries: int = 20, message_prefix: str | None = None):
        result = self._diff_map.get(sha)
        if result and message_prefix is not None and not result[0].message.startswith(message_prefix):
            return None
        return result

    def revert(self, sha: str, *, message_prefix: str | None = None) -> str | None:
        self.revert_calls.append((sha, message_prefix))
        return self._revert_result


def _make_ctx(raw: str, git: _FakeGit, *, args: str = "", last_dream_cursor: int = 1) -> CommandContext:
    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="direct", content=raw)
    store = _FakeStore(git, last_dream_cursor=last_dream_cursor)
    loop = SimpleNamespace(consolidator=SimpleNamespace(store=store))
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, args=args, loop=loop)


def _make_dream_prompt_ctx(tmp_path, raw: str = "/dream-prompt", args: str = "") -> CommandContext:
    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="direct", content=raw)
    loop = SimpleNamespace(context=SimpleNamespace(memory=MemoryStore(tmp_path)))
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, args=args, loop=loop)


# --- /dream-log -----------------------------------------------------------------------------


async def test_dream_log_latest_is_more_user_friendly() -> None:
    commit = CommitInfo(sha="abcd1234", message="dream: 2026-04-04, 2 change(s)", timestamp="2026-04-04 12:00")
    diff = "diff --git a/SOUL.md b/SOUL.md\n--- a/SOUL.md\n+++ b/SOUL.md\n@@ -1 +1 @@\n-old\n+new\n"
    git = _FakeGit(commits=[commit], diff_map={commit.sha: (commit, diff)})

    out = await cmd_dream_log(_make_ctx("/dream-log", git))

    assert "## Dream Update" in out.content
    assert "Here is the latest Dream memory change." in out.content
    assert "- Commit: `abcd1234`" in out.content
    assert "- Changed files: `SOUL.md`" in out.content
    assert "Use `/dream-restore abcd1234` to undo this change." in out.content
    assert "```diff" in out.content
    assert out.metadata == {"render_as": "text"}


async def test_dream_log_latest_skips_non_dream_commit() -> None:
    backup = CommitInfo(sha="bbbb2222", message="backup: workspace snapshot", timestamp="2026-04-04 13:00")
    dream = CommitInfo(sha="abcd1234", message="dream: latest", timestamp="2026-04-04 12:00")
    diff = "diff --git a/SOUL.md b/SOUL.md\n"
    git = _FakeGit(commits=[backup, dream], diff_map={dream.sha: (dream, diff), backup.sha: (backup, "unrelated diff")})

    out = await cmd_dream_log(_make_ctx("/dream-log", git))

    assert "`abcd1234`" in out.content
    assert "`bbbb2222`" not in out.content


async def test_dream_log_selected_commit_without_diff() -> None:
    commit = CommitInfo(sha="abcd1234", message="dream: x", timestamp="2026-04-04 12:00")
    git = _FakeGit(diff_map={commit.sha: (commit, "")})

    out = await cmd_dream_log(_make_ctx("/dream-log abcd1234", git, args="abcd1234"))

    assert "Here is the selected Dream memory change." in out.content
    assert "- Changed files: No tracked memory files changed." in out.content
    assert out.content.endswith("Dream recorded this version, but there is no file diff to display.")


async def test_dream_log_missing_commit_guides_user() -> None:
    out = await cmd_dream_log(_make_ctx("/dream-log deadbeef", _FakeGit(diff_map={}), args="deadbeef"))

    assert "Couldn't find Dream change `deadbeef`." in out.content
    assert "Use `/dream-restore` to list recent versions" in out.content


async def test_dream_log_before_first_run_is_clear() -> None:
    out = await cmd_dream_log(_make_ctx("/dream-log", _FakeGit(initialized=False), last_dream_cursor=0))

    assert "Dream has not run yet." in out.content
    assert "Run `/dream`" in out.content
    assert "/dream-prompt" in out.content


async def test_dream_log_without_git_after_a_run() -> None:
    out = await cmd_dream_log(_make_ctx("/dream-log", _FakeGit(initialized=False), last_dream_cursor=3))
    assert out.content == "Dream history is not available because memory versioning is not initialized."


async def test_dream_log_without_saved_versions_mentions_prompt_command() -> None:
    out = await cmd_dream_log(_make_ctx("/dream-log", _FakeGit(initialized=True, commits=[])))

    assert "Dream memory has no saved versions yet." in out.content
    assert "/dream-prompt" in out.content


# --- /dream-restore -------------------------------------------------------------------------


async def test_dream_restore_lists_versions_with_next_steps() -> None:
    commits = [
        CommitInfo(sha="abcd1234", message="dream: latest", timestamp="2026-04-04 12:00"),
        CommitInfo(sha="cccc3333", message="backup: workspace", timestamp="2026-04-04 10:00"),
        CommitInfo(sha="bbbb2222", message="dream: older", timestamp="2026-04-04 08:00"),
    ]

    out = await cmd_dream_restore(_make_ctx("/dream-restore", _FakeGit(commits=commits)))

    assert "## Dream Restore" in out.content
    assert "Choose a Dream memory version to restore." in out.content
    assert "`abcd1234` 2026-04-04 12:00 - dream: latest" in out.content
    assert "`bbbb2222` 2026-04-04 08:00 - dream: older" in out.content
    assert "backup: workspace" not in out.content
    assert "Preview a version with `/dream-log <sha>`" in out.content
    assert "Restore a version with `/dream-restore <sha>`." in out.content


async def test_dream_restore_without_versions_or_git() -> None:
    out = await cmd_dream_restore(_make_ctx("/dream-restore", _FakeGit(commits=[])))
    assert out.content == "Dream memory has no saved versions to restore yet."
    out = await cmd_dream_restore(_make_ctx("/dream-restore", _FakeGit(initialized=False)))
    assert out.content == "Dream history is not available because memory versioning is not initialized."
    assert out.metadata == {}


async def test_dream_restore_success_mentions_files_and_followup() -> None:
    commit = CommitInfo(sha="abcd1234", message="dream: latest", timestamp="2026-04-04 12:00")
    diff = (
        "diff --git a/SOUL.md b/SOUL.md\n--- a/SOUL.md\n+++ b/SOUL.md\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/memory/MEMORY.md b/memory/MEMORY.md\n--- a/memory/MEMORY.md\n+++ b/memory/MEMORY.md\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    git = _FakeGit(diff_map={commit.sha: (commit, diff)}, revert_result="eeee9999")

    out = await cmd_dream_restore(_make_ctx("/dream-restore abcd1234", git, args="abcd1234"))

    assert "Restored Dream memory to the state before `abcd1234`." in out.content
    assert "- New safety commit: `eeee9999`" in out.content
    assert "- Restored files: `SOUL.md`, `memory/MEMORY.md`" in out.content
    assert "Use `/dream-log eeee9999` to inspect the restore diff." in out.content
    assert git.revert_calls == [("abcd1234", "dream:")]


async def test_dream_restore_of_first_version_fails_clearly() -> None:
    commit = CommitInfo(sha="abcd1234", message="dream: first", timestamp="2026-04-04 12:00")
    git = _FakeGit(diff_map={commit.sha: (commit, "")}, revert_result=None)

    out = await cmd_dream_restore(_make_ctx("/dream-restore abcd1234", git, args="abcd1234"))

    assert "It may be the first saved version with no earlier state to restore." in out.content


async def test_dream_restore_rejects_non_dream_commit_clearly() -> None:
    commit = CommitInfo(sha="cccc3333", message="backup: workspace", timestamp="2026-04-04 10:00")
    git = _FakeGit(diff_map={commit.sha: (commit, "unrelated diff")}, revert_result="eeee9999")

    out = await cmd_dream_restore(_make_ctx("/dream-restore cccc3333", git, args="cccc3333"))

    assert "Only Dream memory versions can be restored." in out.content
    assert "Use `/dream-restore` to list recent versions." in out.content
    assert git.revert_calls == []


# --- /dream-prompt --------------------------------------------------------------------------


async def test_dream_prompt_reports_default_prompt(tmp_path) -> None:
    out = await cmd_dream_prompt(_make_dream_prompt_ctx(tmp_path))

    assert "Dream memory instructions: nanoclaw default" in out.content
    assert "prompts/dream.md" in out.content
    assert str(tmp_path) not in out.content
    assert "/dream-prompt init" in out.content
    assert out.metadata == {"render_as": "text"}


async def test_dream_prompt_init_copies_default_prompt(tmp_path) -> None:
    out = await cmd_dream_prompt(_make_dream_prompt_ctx(tmp_path, "/dream-prompt init", "init"))

    prompt_file = tmp_path / "prompts" / "dream.md"
    assert "Created Dream memory instructions" in out.content
    assert "prompts/dream.md" in out.content
    assert str(tmp_path) not in out.content
    assert "fully replaces nanoclaw's default Dream guide" in out.content
    assert prompt_file.read_text(encoding="utf-8") == MemoryStore.default_dream_prompt() + "\n"

    shown = await cmd_dream_prompt(_make_dream_prompt_ctx(tmp_path))
    assert shown.content.startswith("Dream memory instructions: custom for this workspace")


async def test_dream_prompt_init_does_not_overwrite_existing_prompt(tmp_path) -> None:
    prompt_file = tmp_path / "prompts" / "dream.md"
    prompt_file.parent.mkdir()
    prompt_file.write_text("custom", encoding="utf-8")

    out = await cmd_dream_prompt(_make_dream_prompt_ctx(tmp_path, "/dream-prompt init", "init"))

    assert "already exist" in out.content
    assert "prompts/dream.md" in out.content
    assert str(tmp_path) not in out.content
    assert prompt_file.read_text(encoding="utf-8") == "custom"


async def test_dream_prompt_init_recreates_empty_prompt(tmp_path) -> None:
    prompt_file = tmp_path / "prompts" / "dream.md"
    prompt_file.parent.mkdir()
    prompt_file.write_text("  \n", encoding="utf-8")

    out = await cmd_dream_prompt(_make_dream_prompt_ctx(tmp_path, "/dream-prompt init", "init"))

    assert "Created Dream memory instructions" in out.content
    assert prompt_file.read_text(encoding="utf-8") == MemoryStore.default_dream_prompt() + "\n"


async def test_dream_prompt_unknown_argument_shows_usage(tmp_path) -> None:
    out = await cmd_dream_prompt(_make_dream_prompt_ctx(tmp_path, "/dream-prompt reset", "reset"))
    assert out.content == "Usage: /dream-prompt [init]"


def test_dream_prompt_command_in_help_and_palette() -> None:
    palette = builtin_command_palette()
    dream_prompt = next(item for item in palette if item["command"] == "/dream-prompt")

    assert dream_prompt["arg_hint"] == "[init]"
    assert dream_prompt["lifecycle"] == "side_channel"
    assert dream_prompt["accepts_args"] is True
    assert "/dream-prompt [init]" in build_help_text()


# --- workspace prompt helpers (nanobot utils/workspace_prompts.py) ------------------------------


def test_workspace_prompt_helpers(tmp_path) -> None:
    path = workspace_prompt_file(tmp_path, "dream")
    assert path == tmp_path / "prompts" / "dream.md"
    assert load_workspace_prompt_override(path) == (None, 0)
    assert not has_workspace_prompt_override(path)
    assert initialize_workspace_prompt(path, "default") is True
    assert path.read_text(encoding="utf-8") == "default\n"
    assert load_workspace_prompt_override(path) == ("default", 7)
    assert has_workspace_prompt_override(path)
    assert initialize_workspace_prompt(path, "other") is False
    path.write_text("y" * (WORKSPACE_PROMPT_MAX_CHARS + 5) + "\n\n", encoding="utf-8")
    text, original = load_workspace_prompt_override(path)
    assert original == WORKSPACE_PROMPT_MAX_CHARS + 5
    assert text == "y" * WORKSPACE_PROMPT_MAX_CHARS + "\n... (truncated)"
    path.write_bytes(b"\xff\xfe")
    assert load_workspace_prompt_override(path) == (None, 0)
    assert initialize_workspace_prompt(path, "x") is False  # unreadable: never overwritten
    path.unlink()
    path.mkdir()
    assert initialize_workspace_prompt(path, "x") is False  # not a file


# --- MemoryStore (nanobot tests/agent/test_dream.py) ----------------------------------------


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path)
    s.soul_file.write_text("# Soul\n- Helpful", encoding="utf-8")
    s.write_memory("# Memory\n- Project X active")
    return s


def test_workspace_dream_prompt_overrides_default(store):
    store.dream_prompt_file.parent.mkdir(parents=True)
    store.dream_prompt_file.write_text("Custom Dream prompt.", encoding="utf-8")
    store.append_history("keep this fact")

    prompt, _ = store.build_dream_prompt()

    assert prompt.startswith("Custom Dream prompt.")
    assert "## Conversation History" in prompt
    assert "keep this fact" in prompt


def test_workspace_dream_prompt_override_is_capped(store):
    store.dream_prompt_file.parent.mkdir(parents=True)
    store.dream_prompt_file.write_text("x" * 40_000, encoding="utf-8")
    store.append_history("keep this fact")

    prompt, _ = store.build_dream_prompt()

    assert "x" * 40_000 not in prompt
    assert "... (truncated)" in prompt
    assert "## Conversation History" in prompt
    assert "keep this fact" in prompt


def test_empty_workspace_dream_prompt_uses_default(store):
    store.dream_prompt_file.parent.mkdir(parents=True)
    store.dream_prompt_file.write_text("  \n", encoding="utf-8")
    store.append_history("test")

    prompt, _ = store.build_dream_prompt()

    assert prompt.startswith(store.default_dream_prompt() + "\n\n## Conversation History\n")


def test_commit_message_reflects_real_diff_not_narrative(tmp_path):
    store = MemoryStore(tmp_path)
    store.soul_file.write_text("# Soul", encoding="utf-8")
    store.write_memory("# Memory")
    store.git.init()
    store.git.auto_commit("initial state")

    store.write_memory("# Memory\n- DMSO research notes")

    lying = "Added a Medical Research section (Mastic Gum, DMSO) to MEMORY.md"
    diff_body = store.dream_content_diff()
    assert "DMSO research notes" in diff_body
    assert lying not in diff_body

    msg = MemoryStore.build_dream_commit_message("dream: periodic memory consolidation", diff_body)
    assert lying not in msg
    assert "DMSO research notes" in msg

    assert store.git.auto_commit(msg) is not None
    log = subprocess.check_output(["git", "log", "-1", "--format=%B"], cwd=str(tmp_path), text=True).strip()
    assert "dream: periodic memory consolidation" in log
    assert "DMSO research notes" in log


def test_commit_message_is_bare_prefix_when_no_changes(tmp_path):
    store = MemoryStore(tmp_path)
    store.soul_file.write_text("# Soul", encoding="utf-8")
    store.write_memory("# Memory")
    store.git.init()
    store.git.auto_commit("initial state")

    assert store.dream_content_diff() == ""
    assert MemoryStore.build_dream_commit_message("dream: manual run", store.dream_content_diff()) == "dream: manual run"


def test_build_commit_message_ignores_none_and_empty_body():
    assert MemoryStore.build_dream_commit_message("dream: x", "") == "dream: x"
    assert MemoryStore.build_dream_commit_message("dream: x", None) == "dream: x"
    assert MemoryStore.build_dream_commit_message("dream: x", "  ") == "dream: x"
    assert MemoryStore.build_dream_commit_message("dream: x", "SOUL.md: +1 -0") == "dream: x\n\nSOUL.md: +1 -0"


def test_content_diff_empty_when_git_not_initialized(store):
    assert store.dream_content_diff() == ""


def test_content_diff_empty_when_no_tracked_changes(store):
    store.git.init()
    store.git.auto_commit("initial")
    assert store.dream_content_diff() == ""


@pytest.mark.parametrize(
    ("before", "after", "changed"),
    [
        (b"# Memory\r\n", b"# Memory\n", False),
        (b"# Memory\n", b"# Memory", True),
        (b"# Memory\r", b"# Memory\n", True),
    ],
)
def test_content_diff_only_ignores_crlf_lf_changes(store, before, after, changed):
    store.memory_file.write_bytes(before)
    store.git.init()
    store.memory_file.write_bytes(after)
    assert bool(store.dream_content_diff()) is changed


def test_content_diff_reflects_real_content_edits(store):
    store.git.init()
    store.git.auto_commit("initial")
    store.write_memory("# Memory\n- DMSO research notes")
    diff = store.dream_content_diff()
    assert "memory/MEMORY.md" in diff
    assert "DMSO research notes" in diff


def test_content_diff_ignores_cursor_only_changes(store):
    store.git.init()
    store.git.auto_commit("initial")
    store.set_last_dream_cursor(99)
    assert store.dream_content_diff() == ""


def test_commit_dream_changes_skips_noop_runs(tmp_path):
    """nanobot gateway_runtime._commit_dream_changes (the scheduled Dream)."""
    sync_workspace_templates(tmp_path)
    store = MemoryStore(tmp_path)
    store.set_last_dream_cursor(4)
    assert commit_dream_changes(store) is None  # only the cursor moved
    store.write_memory("- likes tea\n")
    sha = commit_dream_changes(store)
    [latest] = store.git.log(max_entries=1)
    assert latest.sha == sha
    assert latest.message.startswith("dream: periodic memory consolidation\n\nmemory/MEMORY.md: +1 -")
    assert commit_dream_changes(MemoryStore(tmp_path / "plain")) is None


# --- through the real loop ------------------------------------------------------------------


def _write_memory_call(content: str) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="w1", name="write_file", arguments={"path": "memory/MEMORY.md", "content": content})],
        finish_reason="tool_calls",
    )


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    sync_workspace_templates(path)
    return path


def _loop(ws, tmp_path, *script) -> AgentLoop:
    return AgentLoop(FakeProvider(list(script)), ws, sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"))


def test_new_workspace_gets_nanobots_repository(ws):
    store = MemoryStore(ws)
    assert store.git.is_initialized()
    assert [c.message for c in store.git.log()] == ["init: nanoclaw memory store"]
    assert (ws / ".gitignore").read_text(encoding="utf-8") == (
        "/*\n!memory/\n!SOUL.md\n!USER.md\n!memory/MEMORY.md\n!.gitignore\n"
    )
    assert (ws / "prompts" / "README.md").is_file()


def test_workspace_inside_a_repository_gets_none(tmp_path):
    (tmp_path / ".git").mkdir()
    sync_workspace_templates(tmp_path / "ws")
    assert not (tmp_path / "ws" / ".git").exists()


async def test_dream_commits_the_change_and_names_the_commit(ws, tmp_path):
    loop = _loop(ws, tmp_path, _write_memory_call("# Memory\n- likes tea\n"), "Updated.")
    loop.memory.append_history("- [permanent] likes tea")

    message = await loop.run_dream()

    match = re.fullmatch(r"Dream completed in \d+\.\ds\. \(commit ([0-9a-f]{8})\)", message)
    assert match
    latest = loop.memory.git.log()[0]
    assert latest.sha == match.group(1)
    assert latest.message.startswith("dream: manual run\n\nmemory/MEMORY.md: +2 -")
    assert "+- likes tea" in latest.message


async def test_dream_with_a_custom_prompt_uses_it(ws, tmp_path):
    loop = _loop(ws, tmp_path, "Nothing to change.")
    loop.memory.dream_prompt_file.write_text("Only keep facts about tea.", encoding="utf-8")
    loop.memory.append_history("- small talk")
    await loop.run_dream()
    assert loop.provider.calls[0].messages[1]["content"].startswith("Only keep facts about tea.\n\n## Conversation History\n")


async def test_dream_without_input_still_commits_but_says_nothing_of_it(ws, tmp_path):
    """nanobot's /dream commits in ``finally`` even when there was nothing to do."""
    loop = _loop(ws, tmp_path)
    (ws / "USER.md").write_text("edited by hand\n", encoding="utf-8")
    assert await loop.run_dream() == DREAM_NO_INPUT_MESSAGE
    assert loop.memory.git.log()[0].message == "dream: manual run"


async def test_failed_dream_commits_with_the_bare_prefix(ws, tmp_path):
    loop = _loop(ws, tmp_path)
    loop.memory.append_history("- fact")
    (ws / "SOUL.md").write_text("changed\n", encoding="utf-8")

    async def boom(*args, **kwargs):
        raise RuntimeError("down")

    loop.process_direct = boom
    message = await loop.run_dream()
    assert re.fullmatch(r"Dream failed after \d+\.\ds: down \(commit [0-9a-f]{8}\)", message)
    assert loop.memory.git.log()[0].message == "dream: manual run"


async def test_dream_without_git_reports_no_changes(tmp_path):
    """Without a repository the diff is empty, so nanobot says "no memory changes"."""
    (tmp_path / ".git").mkdir()  # the workspace sits in another repository
    ws = tmp_path / "ws"
    sync_workspace_templates(ws)
    loop = _loop(ws, tmp_path, _write_memory_call("- likes tea\n"), "Updated.")
    loop.memory.append_history("- likes tea")
    message = await loop.run_dream()
    assert message.endswith("; no memory changes.")
    assert (ws / "memory" / "MEMORY.md").read_text(encoding="utf-8") == "- likes tea\n"


async def test_slash_commands_log_and_restore_a_real_dream(ws, tmp_path):
    loop = _loop(ws, tmp_path, _write_memory_call("# Memory\n- likes tea\n"), "Updated.")
    original = (ws / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    loop.memory.append_history("- likes tea")
    router = CommandRouter()
    register_builtin_commands(router)
    bus = MessageBus()

    def ctx(text: str) -> CommandContext:
        msg = InboundMessage("cli", "user", "direct", text)
        _, _, args = text.partition(" ")
        return CommandContext(msg=msg, session=None, key=msg.session_key, raw=text, args=args, loop=loop, bus=bus)

    dream_ctx = ctx("/dream")
    assert (await router.dispatch(dream_ctx)).content == "Dreaming..."
    await asyncio.gather(*dream_ctx.tasks)
    followup: OutboundMessage = await bus.consume_outbound()
    sha = re.search(r"\(commit ([0-9a-f]{8})\)", followup.content).group(1)

    log = await router.dispatch(ctx("/dream-log"))
    assert f"- Commit: `{sha}`" in log.content
    # The moved Dream cursor is committed too (staged by name), as in nanobot.
    assert "- Changed files: `memory/.dream_cursor`, `memory/MEMORY.md`" in log.content
    assert "+- likes tea" in log.content

    listing = await router.dispatch(ctx("/dream-restore"))
    assert f"- `{sha}` " in listing.content and " - dream: manual run" in listing.content

    restored = await router.dispatch(ctx(f"/dream-restore {sha}"))
    assert f"Restored Dream memory to the state before `{sha}`." in restored.content
    assert (ws / "memory" / "MEMORY.md").read_text(encoding="utf-8") == original
    new_sha = re.search(r"New safety commit: `([0-9a-f]{8})`", restored.content).group(1)
    assert loop.memory.git.log()[0].message == f"revert: undo {sha}"

    # The restore is not a Dream commit: /dream-log still shows the Dream run,
    # but the restore can be inspected by its SHA.
    assert f"- Commit: `{sha}`" in (await router.dispatch(ctx("/dream-log"))).content
    assert "Here is the selected" in (await router.dispatch(ctx(f"/dream-log {new_sha}"))).content

    prompt = await router.dispatch(ctx("/dream-prompt init"))
    assert prompt.content.startswith("Created Dream memory instructions at `prompts/dream.md`.")


def test_cli_dream_commits_like_slash_dream(capsys, ws):
    fake = FakeProvider(["hi", "- [permanent] user is Rohit", _write_memory_call("- user is Rohit\n"), "done"])
    assert main(["ask", "I am Rohit", "-w", str(ws)], provider=fake) == 0
    assert main(["compact", "-w", str(ws)], provider=fake) == 0
    capsys.readouterr()
    assert main(["dream", "-w", str(ws)], provider=fake) == 0
    out = capsys.readouterr().out
    assert re.search(r"Dream completed in \d+\.\ds\. \(commit [0-9a-f]{8}\)", out)
    assert MemoryStore(ws).git.log()[0].message.startswith("dream: manual run\n\nmemory/MEMORY.md:")
