"""Skill resource locations and validated YAML identity metadata.

Mirrors nanobot/agent/skill_metadata.py: the helpers that agent plugins and
the skills loader share live here, so either can import them without a
cycle. ``nanoclaw.agent.skills`` re-exports them.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Opening ---, YAML body (group 1), closing --- on its own line; supports CRLF.
SKILL_FRONTMATTER = re.compile(
    r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
    re.DOTALL,
)

SKILL_NAME = re.compile(r"^(?!.*--)[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")


def parse_skill_metadata(content: str) -> dict[str, object] | None:
    """The YAML frontmatter of a skill document, or ``None``."""
    if not (match := SKILL_FRONTMATTER.match(content)):
        return None
    try:
        parsed = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(key): value for key, value in parsed.items()}


def valid_skill_metadata(metadata: dict[str, object], name: str) -> bool:
    """Whether the frontmatter meets the Agent Skills rules (name matches the folder, etc.)."""
    description = metadata.get("description")
    return (
        metadata.get("name") == name
        and len(name) <= 64
        and SKILL_NAME.fullmatch(name) is not None
        and isinstance(description, str)
        and 1 <= len(description.strip()) <= 1024
    )
