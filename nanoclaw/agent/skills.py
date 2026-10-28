"""Skills: Markdown procedures the agent loads when it needs them.

Mirrors nanobot/agent/skills.py (``SkillsLoader``). A skill is a folder with a
``SKILL.md`` file: YAML frontmatter (``name``, ``description``, optional
``metadata``) and instructions. The system prompt lists each skill's name,
description and path; the model reads the full file with ``read_file`` when a
task matches. Skills marked ``always`` are put in the prompt in full, and
``$skill-name`` in a message loads that skill for the turn.

Skills come from the workspace's ``skills/`` folder, enabled Agent Plugins
(``<workspace>/plugins/<name>/skills/``, nanoclaw/agent/plugins.py) and the
built-in ``nanoclaw/skills/``, in that order: an earlier source replaces a later
skill of the same name.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from nanoclaw.runtime_context import RuntimeContextBlock

BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Opening ---, YAML body (group 1), closing --- on its own line; supports CRLF.
_STRIP_SKILL_FRONTMATTER = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?", re.DOTALL)
_SKILL_NAME = re.compile(r"^(?!.*--)[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_SKILL_REFERENCE = re.compile(r"(?<![\w$])\$([A-Za-z0-9_-]+)")


def parse_skill_metadata(content: str) -> dict[str, object] | None:
    """The YAML frontmatter of a skill document, or ``None``."""
    if not (match := _STRIP_SKILL_FRONTMATTER.match(content)):
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
        and _SKILL_NAME.fullmatch(name) is not None
        and isinstance(description, str)
        and 1 <= len(description.strip()) <= 1024
    )


class SkillsLoader:
    """Finds, describes and loads skills."""

    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None) -> None:
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR

    @staticmethod
    def _skill_entries_from_dir(base: Path, source: str, *, skip_names: set[str] | None = None) -> list[dict[str, str]]:
        if not base.exists():
            return []
        entries: list[dict[str, str]] = []
        for skill_dir in base.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            if skip_names is not None and skill_dir.name in skip_names:
                continue
            entries.append({"name": skill_dir.name, "path": str(skill_file), "source": source})
        return entries

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """Workspace skills, then enabled Agent Plugin skills, then built-in
        ones, each not overridden by name.

        Each entry has ``name``, ``path`` and ``source`` (``workspace``,
        ``plugin`` or ``builtin``). With ``filter_unavailable``, skills whose
        required programs or environment variables are missing are left out.
        """
        skills = self._discover_skills()
        if filter_unavailable:
            return [skill for skill in skills if self._check_requirements(self._get_skill_meta(skill["name"]))]
        return skills

    def _discover_skills(self, plugin_documents: dict[str, str] | None = None) -> list[dict[str, str]]:
        """Discovery order workspace → plugin → built-in (nanobot skills.py:123-162).

        A plugin skill's text is the one captured while its package
        fingerprint was verified; it goes into ``plugin_documents`` so
        ``load_skill`` never re-reads a file that may have changed since.
        """
        from nanoclaw.agent.plugins import enabled_agent_plugin_skill_documents

        plugin_skills = enabled_agent_plugin_skill_documents(self.workspace)
        skills = self._skill_entries_from_dir(self.workspace_skills, "workspace")
        seen = {entry["name"] for entry in skills}
        for skill in plugin_skills:
            if skill.name in seen:
                continue
            skills.append({"name": skill.name, "path": str(skill.path), "source": "plugin"})
            if plugin_documents is not None:
                plugin_documents[skill.name] = skill.content
            seen.add(skill.name)
        if self.builtin_skills and self.builtin_skills.exists():
            skills.extend(self._skill_entries_from_dir(self.builtin_skills, "builtin", skip_names=seen))
        return skills

    def load_skill(self, name: str) -> str | None:
        """The full ``SKILL.md`` text of a skill, or ``None``."""
        plugin_documents: dict[str, str] = {}
        entry = next((s for s in self._discover_skills(plugin_documents) if s["name"] == name), None)
        if entry is None:
            return None
        if entry["source"] == "plugin":
            return plugin_documents.get(name)
        return Path(entry["path"]).read_text(encoding="utf-8")

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """Skills in full (without frontmatter), for the prompt."""
        parts = [
            f"### Skill: {name}\n\n{self._strip_frontmatter(markdown)}"
            for name in skill_names
            if (markdown := self.load_skill(name))
        ]
        return "\n\n---\n\n".join(parts)

    def get_explicitly_invoked_skills(self, text: str) -> list[str]:
        """Available skills named as ``$skill-name`` in ``text``, in order, once each."""
        if not text:
            return []
        available = {entry["name"] for entry in self.list_skills(filter_unavailable=True)}
        invoked: list[str] = []
        for match in _SKILL_REFERENCE.finditer(text):
            name = match.group(1)
            if name in available and name not in invoked:
                invoked.append(name)
        return invoked

    def build_explicit_skill_runtime_context(self, text: str) -> RuntimeContextBlock | None:
        """A runtime-context block with the skills the message names (except always-on ones)."""
        skill_names = self.get_explicitly_invoked_skills(text)
        if not skill_names:
            return None
        always_active = set(self.get_always_skills())
        content = self.load_skills_for_context([name for name in skill_names if name not in always_active])
        if not content:
            return None
        return RuntimeContextBlock(
            source="explicit_skills",
            content=f"[Active Skills — instructions for this user turn]\n{content}\n[/Active Skills]",
        )

    def build_skills_summary(self, exclude: set[str] | None = None, *, workspace: Path | None = None) -> str:
        """Name, description and path of each available skill, grouped by source.

        Skills whose required programs or environment variables are missing
        are left out (nanobot/agent/skills.py:270-308); ``nanoclaw skills``
        still lists them with what is missing. Paths are relative to the
        group's folder, so the model joins them to read a skill with
        ``read_file``. The workspace folder is shown as ``skills``, the
        Agent Plugins folder as ``plugins``; the built-in one by its absolute path, because directory tools need the
        real path (only ``read_file`` maps ``skills/<name>/...`` to it).

        ``workspace`` is the turn's project folder: when it is not the agent
        workspace, relative paths would resolve against the project, so the
        workspace and plugins folders are shown by their absolute paths too (nanobot
        skills.py:275-298).
        """
        all_skills = self.list_skills(filter_unavailable=True)
        if not all_skills:
            return ""
        sections: list[str] = []
        agent_workspace = self.workspace.expanduser().resolve()
        project_workspace = (workspace or self.workspace).expanduser().resolve()
        workspace_root = (
            Path("skills") if project_workspace == agent_workspace else self.workspace_skills.expanduser().resolve()
        )
        plugins_root = self.workspace.expanduser().resolve() / "plugins"
        groups = (
            ("Workspace skills", "workspace", self.workspace_skills, workspace_root),
            (
                "Agent Plugin skills",
                "plugin",
                plugins_root,
                Path("plugins") if project_workspace == agent_workspace else plugins_root,
            ),
            ("Built-in skills", "builtin", self.builtin_skills, self.builtin_skills.expanduser().resolve()),
        )
        for label, source, root, display_root in groups:
            entries = [e for e in all_skills if e["source"] == source and (not exclude or e["name"] not in exclude)]
            if not entries:
                continue
            lines = [f"### {label} (`{display_root}`)"]
            for entry in entries:
                name = entry["name"]
                relative_path = Path(entry["path"]).relative_to(root).as_posix()
                lines.append(f"- **{name}** — {self.get_skill_description(name)}  `{relative_path}`")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    # --- requirements -------------------------------------------------------------

    @staticmethod
    def _requirement_lists(skill_meta: dict[str, Any]) -> tuple[list[str], list[str]]:
        """``(bins, env)`` from the metadata, ignoring wrong shapes."""
        requires = skill_meta.get("requires") or {}
        if not isinstance(requires, dict):
            return [], []
        bins_raw = requires.get("bins") or []
        env_raw = requires.get("env") or []
        bins = [v for v in bins_raw if isinstance(v, str) and v.strip()] if isinstance(bins_raw, list) else []
        env = [v for v in env_raw if isinstance(v, str) and v.strip()] if isinstance(env_raw, list) else []
        return bins, env

    def _get_missing_requirements(self, skill_meta: dict[str, Any]) -> str:
        bins, env = self._requirement_lists(skill_meta)
        return ", ".join(
            [f"CLI: {b}" for b in bins if not shutil.which(b)] + [f"ENV: {e}" for e in env if not os.environ.get(e)]
        )

    def _check_requirements(self, skill_meta: dict[str, Any]) -> bool:
        bins, env = self._requirement_lists(skill_meta)
        return all(shutil.which(b) for b in bins) and all(os.environ.get(e) for e in env)

    def get_skill_availability(self, name: str) -> tuple[bool, str]:
        """Whether a skill can run, and what is missing if not."""
        meta = self._get_skill_meta(name)
        available = self._check_requirements(meta)
        return available, "" if available else self._get_missing_requirements(meta)

    # --- metadata -------------------------------------------------------------------

    def get_skill_description(self, name: str) -> str:
        meta = self.get_skill_metadata(name)
        description = meta.get("description") if meta else None
        return description if isinstance(description, str) and description else name

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        if not content.startswith("---"):
            return content
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        return content[match.end():].strip() if match else content

    @staticmethod
    def _parse_nanobot_metadata(raw: object) -> dict[str, Any]:
        """The ``nanobot`` (or ``openclaw``) entry of the ``metadata`` field.

        ``metadata`` may be a mapping or a JSON string. NanoClaw also accepts a
        ``nanoclaw`` entry, checked first.
        """
        if isinstance(raw, dict):
            data: Any = raw
        elif isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        else:
            return {}
        if not isinstance(data, dict):
            return {}
        payload = data.get("nanoclaw", data.get("nanobot", data.get("openclaw", {})))
        return payload if isinstance(payload, dict) else {}

    def _get_skill_meta(self, name: str) -> dict[str, Any]:
        raw_meta = self.get_skill_metadata(name) or {}
        return self._parse_nanobot_metadata(raw_meta.get("metadata"))

    def get_always_skills(self) -> list[str]:
        """Available skills marked ``always`` (top level or inside ``metadata``)."""
        return [
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
            if (meta := self.get_skill_metadata(entry["name"]) or {})
            and (self._parse_nanobot_metadata(meta.get("metadata")).get("always") or meta.get("always"))
        ]

    def get_skill_metadata(self, name: str) -> dict[str, object] | None:
        return parse_skill_metadata(self.load_skill(name) or "")
