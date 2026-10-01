"""Prompt templates and workspace files.

Mirrors nanobot/utils/prompt_templates.py (``render_template``) and
nanobot/utils/helpers.py (``load_bundled_template``, ``sync_workspace_templates``).
Templates live in ``nanoclaw/templates/``: ``agent/*.md`` are prompt pieces
rendered with Jinja2; ``SOUL.md``, ``USER.md`` and ``AGENTS.md`` are the starting
copies of the files the user edits in their workspace.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "templates"

# Files copied into a new workspace (nanobot copies every top-level template).
WORKSPACE_FILES = ("AGENTS.md", "SOUL.md", "USER.md")


@lru_cache
def _environment() -> Environment:
    # Plain-text prompts: no HTML escaping. trim/lstrip keep {% %} lines from
    # leaving blank lines behind. Same settings as nanobot.
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_ROOT)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(name: str, *, strip: bool = False, **kwargs: Any) -> str:
    """Render ``name`` (e.g. ``agent/identity.md``) from the templates folder."""
    text = _environment().get_template(name).render(**kwargs)
    return text.rstrip() if strip else text


def load_bundled_template(name: str) -> str | None:
    """The raw text of a bundled template, or ``None`` if it doesn't exist."""
    path = TEMPLATES_ROOT / name
    return path.read_text(encoding="utf-8") if path.is_file() else None


def sync_workspace_templates(workspace: Path) -> list[str]:
    """Create any missing starting files in ``workspace``. Never overwrites.

    Returns the names of the files created. nanobot also creates
    ``memory/MEMORY.md``, ``memory/history.jsonl`` and ``skills/`` (those come
    with layers 6 and 7), ``HEARTBEAT.md`` and ``prompts/README.md`` (for
    features NanoClaw doesn't have), and a git repository tracking
    ``SOUL.md``/``USER.md``/``MEMORY.md``.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    added: list[str] = []
    for name in WORKSPACE_FILES:
        dest = workspace / name
        if dest.exists():
            continue
        dest.write_text(load_bundled_template(name) or "", encoding="utf-8")
        added.append(name)
    return added
