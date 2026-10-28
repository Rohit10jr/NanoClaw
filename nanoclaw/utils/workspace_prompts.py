"""Shared file handling for workspace-local prompt overrides.

Mirrors nanobot/utils/workspace_prompts.py. A prompt named ``<name>`` can be
replaced for one workspace by a non-empty ``prompts/<name>.md``; an empty or
missing file means the built-in default. NanoClaw's overridable prompt is
Dream's (``prompts/dream.md``, ``MemoryStore._dream_template``, ``/dream-prompt``).
nanobot's second one, the heartbeat evaluator (``prompts/evaluator.md``),
uses the same helpers and arrives with heartbeat.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from nanoclaw.agent.messages import truncate_text

WORKSPACE_PROMPT_MAX_CHARS = 32_000


def workspace_prompt_file(workspace: Path, name: str) -> Path:
    """Return the conventional path for a named workspace prompt override."""
    return workspace / "prompts" / f"{name}.md"


def load_workspace_prompt_override(
    path: Path,
    *,
    max_chars: int = WORKSPACE_PROMPT_MAX_CHARS,
) -> tuple[str | None, int]:
    """Load and cap a non-empty UTF-8 prompt override.

    Returns the loaded text and its original length. Missing, unreadable, and
    empty files return ``(None, 0)`` so callers can fall back to their default.
    """
    with suppress(OSError, UnicodeDecodeError):
        text = path.read_text(encoding="utf-8").rstrip()
        if text:
            original_chars = len(text)
            return truncate_text(text, max_chars), original_chars
    return None, 0


def has_workspace_prompt_override(path: Path) -> bool:
    """Return whether a path contains a non-empty workspace prompt override."""
    text, _original_chars = load_workspace_prompt_override(path)
    return text is not None


def initialize_workspace_prompt(path: Path, default_prompt: str) -> bool:
    """Create a default prompt copy when the target is missing or empty.

    Returns ``False`` without overwriting a non-empty file, non-file path, or
    path whose current state cannot be read safely.
    """
    try:
        if path.exists() and (
            not path.is_file() or bool(path.read_text(encoding="utf-8").strip())
        ):
            return False
    except (OSError, UnicodeDecodeError):
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_prompt + "\n", encoding="utf-8")
    return True
