"""Skills: Markdown procedures the agent loads when it needs them.

Mirrors nanobot/agent/skills.py (``SkillsLoader``). A skill is a folder with a
``SKILL.md`` file: YAML frontmatter (``name``, ``description``, optional
``metadata``) and instructions. The system prompt lists each skill's name,
description and path; the model reads the full file with ``read_file`` when a
task matches. Skills marked ``always`` are put in the prompt in full, and
``$skill-name`` in a message loads that skill for the turn.

Skills come from the workspace's ``skills/`` folder and the built-in
``nanoclaw/skills/``; a workspace skill replaces a built-in one of the same
name. Names in ``disabled_skills`` (config ``agents.defaults.disabledSkills``)
are left out everywhere: the prompt, ``$name`` and ``load_skill``.

``snapshot()`` caches discovery, documents, metadata and ``which`` results
for one prompt build (nanobot 3a28b9e); ``ContextBuilder`` opens one around
``build_system_prompt`` and ``build_messages``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanoclaw.agent.skill_metadata import BUILTIN_SKILLS_DIR as BUILTIN_SKILLS_DIR
from nanoclaw.agent.skill_metadata import SKILL_FRONTMATTER as _STRIP_SKILL_FRONTMATTER
from nanoclaw.agent.skill_metadata import parse_skill_metadata as parse_skill_metadata
from nanoclaw.agent.skill_metadata import valid_skill_metadata as valid_skill_metadata
from nanoclaw.runtime_context import RuntimeContextBlock

_SKILL_REFERENCE = re.compile(r"(?<![\w$])\$([A-Za-z0-9_-]+)")


@dataclass
class _SkillSnapshot:
    """What one prompt build has already looked up (nanobot skills.py:32).

    nanobot also caches CLI-App skill aliases; NanoClaw has no CLI Apps.
    """

    entries: list[dict[str, str]] | None = None
    documents: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, dict[str, object] | None] = field(default_factory=dict)
    binaries: dict[str, bool] = field(default_factory=dict)


class SkillsLoader:
    """Finds, describes and loads skills."""

    def __init__(
        self,
        workspace: Path,
        builtin_skills_dir: Path | None = None,
        disabled_skills: set[str] | None = None,
    ) -> None:
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR
        self.disabled_skills = disabled_skills or set()
        self._snapshot: ContextVar[_SkillSnapshot | None] = ContextVar("skill_snapshot", default=None)

    @contextmanager
    def snapshot(self) -> Generator[None, None, None]:
        """Reuse discovery, documents, metadata and CLI checks until this scope exits.

        A nested scope shares the outer snapshot.
        """
        if self._snapshot.get() is not None:
            yield
            return
        token = self._snapshot.set(_SkillSnapshot())
        try:
            yield
        finally:
            self._snapshot.reset(token)

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
        """Enabled skills: workspace ones, then built-in ones not overridden by name.

        Each entry has ``name``, ``path`` and ``source`` (``workspace`` or
        ``builtin``). With ``filter_unavailable``, skills whose required
        programs or environment variables are missing are left out.
        """
        with self.snapshot():
            return self._list_skills(filter_unavailable)

    def _list_skills(self, filter_unavailable: bool) -> list[dict[str, str]]:
        snapshot = self._snapshot.get()
        if snapshot is not None and snapshot.entries is not None:
            skills = snapshot.entries
        else:
            skills = self._discover_skills()
            if snapshot is not None:
                snapshot.entries = skills
        if filter_unavailable:
            return [skill for skill in skills if self._check_requirements(self._get_skill_meta(skill["name"]))]
        return skills

    def _discover_skills(self) -> list[dict[str, str]]:
        """Workspace skills, then built-in ones; the first source wins a name,
        and disabled names are dropped (nanobot skills.py:123, where Agent
        Plugin skills come between the two)."""
        skills = self._skill_entries_from_dir(self.workspace_skills, "workspace")
        seen_names = {entry["name"] for entry in skills}
        if self.builtin_skills and self.builtin_skills.exists():
            skills.extend(self._skill_entries_from_dir(self.builtin_skills, "builtin", skip_names=seen_names))
        if self.disabled_skills:
            skills = [s for s in skills if s["name"] not in self.disabled_skills]
        return skills

    def load_skill(self, name: str) -> str | None:
        """The full ``SKILL.md`` text of an enabled skill, or ``None``."""
        with self.snapshot():
            return self._load_skill(name)

    def _load_skill(self, name: str) -> str | None:
        entry = next((s for s in self.list_skills(filter_unavailable=False) if s["name"] == name), None)
        if entry is None:
            return None
        snapshot = self._snapshot.get()
        if snapshot is not None and name in snapshot.documents:
            return snapshot.documents[name]
        content = Path(entry["path"]).read_text(encoding="utf-8")
        if snapshot is not None:
            snapshot.documents[name] = content
        return content

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """Skills in full (without frontmatter), for the prompt."""
        parts = [
            f"### Skill: {name}\n\n{self._strip_frontmatter(markdown)}"
            for name in skill_names
            if (markdown := self.load_skill(name))
        ]
        return "\n\n---\n\n".join(parts)

    def get_explicitly_invoked_skills(self, text: str) -> list[str]:
        """Enabled, available skills named as ``$skill-name`` in ``text``, in order, once each."""
        references = list(_SKILL_REFERENCE.finditer(text or ""))
        if not references:
            return []
        available = {entry["name"] for entry in self.list_skills(filter_unavailable=True)}
        invoked: list[str] = []
        for match in references:
            name = match.group(1)
            if name in available and name not in invoked:
                invoked.append(name)
        return invoked

    def build_explicit_skill_runtime_context(self, text: str) -> RuntimeContextBlock | None:
        """A runtime-context block with the skills the message names (except always-on ones)."""
        with self.snapshot():
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
        ``read_file``. The workspace folder is shown as ``skills``; the
        built-in one by its absolute path, because directory tools need the
        real path (only ``read_file`` maps ``skills/<name>/...`` to it).

        ``workspace`` is the turn's project folder: when it is not the agent
        workspace, relative paths would resolve against the project, so the
        workspace folder is shown by its absolute path too (nanobot
        skills.py:275-298).
        """
        with self.snapshot():
            return self._build_skills_summary(exclude, workspace=workspace)

    def _build_skills_summary(self, exclude: set[str] | None, *, workspace: Path | None) -> str:
        all_skills = self.list_skills(filter_unavailable=True)
        if not all_skills:
            return ""
        agent_workspace = self.workspace.expanduser().resolve()
        project_workspace = (workspace or self.workspace).expanduser().resolve()
        use_relative_roots = project_workspace == agent_workspace
        sections: list[str] = []
        groups = (
            ("Workspace skills", "workspace", self.workspace_skills),
            ("Built-in skills", "builtin", self.builtin_skills),
        )
        for label, source, root in groups:
            entries = [e for e in all_skills if e["source"] == source and (not exclude or e["name"] not in exclude)]
            if not entries:
                continue
            if use_relative_roots and source != "builtin":
                display_root = Path("plugins" if source == "plugin" else "skills")
            else:
                display_root = root.expanduser().resolve()
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
            [f"CLI: {b}" for b in bins if not self._binary_available(b)]
            + [f"ENV: {e}" for e in env if not os.environ.get(e)]
        )

    def _check_requirements(self, skill_meta: dict[str, Any]) -> bool:
        bins, env = self._requirement_lists(skill_meta)
        return all(self._binary_available(b) for b in bins) and all(os.environ.get(e) for e in env)

    def _binary_available(self, command: str) -> bool:
        """``shutil.which``, cached for the open snapshot."""
        snapshot = self._snapshot.get()
        if snapshot is not None and command in snapshot.binaries:
            return snapshot.binaries[command]
        available = shutil.which(command) is not None
        if snapshot is not None:
            snapshot.binaries[command] = available
        return available

    def get_skill_availability(self, name: str) -> tuple[bool, str]:
        """Whether a skill can run, and what is missing if not."""
        meta = self._get_skill_meta(name)
        available = self._check_requirements(meta)
        return available, "" if available else self._get_missing_requirements(meta)

    def get_skill_requirements(self, name: str) -> dict[str, list[str]]:
        """Declared program and environment requirements, and the missing ones (web UI skill detail)."""
        bins, env = self._requirement_lists(self._get_skill_meta(name))
        return {
            "bins": bins,
            "env": env,
            "missing_bins": [value for value in bins if not self._binary_available(value)],
            "missing_env": [value for value in env if not os.environ.get(value)],
        }

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
        with self.snapshot():
            return [
                entry["name"]
                for entry in self.list_skills(filter_unavailable=True)
                if (meta := self.get_skill_metadata(entry["name"]) or {})
                and (self._parse_nanobot_metadata(meta.get("metadata")).get("always") or meta.get("always"))
            ]

    def get_skill_metadata(self, name: str) -> dict[str, object] | None:
        """The frontmatter of an enabled skill, or ``None``."""
        snapshot = self._snapshot.get()
        if snapshot is not None and name in snapshot.metadata:
            return snapshot.metadata[name]
        metadata = parse_skill_metadata(self.load_skill(name) or "")
        if snapshot is not None:
            snapshot.metadata[name] = metadata
        return metadata
