"""Building the system prompt and the message list for a turn.

Mirrors nanobot/agent/context.py (``ContextBuilder``). The system prompt is
assembled from Markdown: rendered templates, the user-editable files
(``AGENTS.md`` from the project folder, ``SOUL.md`` and ``USER.md`` from the
agent workspace), long-term memory (``memory/MEMORY.md``), skills, and the
summary of compacted conversation. The project is the workspace unless
``--project`` names another folder.
"""

from __future__ import annotations

import hashlib

import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nanoclaw.runtime_context import (
    RUNTIME_CONTEXT_MESSAGE_META,
    RuntimeContextBlock,
    append_runtime_context,
)
from nanoclaw.agent.memory import MemoryStore
from nanoclaw.agent.skills import SkillsLoader
from nanoclaw.session.summary import SessionSummary
from nanoclaw.utils.templates import load_bundled_template, render_template

SECTION_SEPARATOR = "\n\n---\n\n"


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md"]
    # Left out of the prompt while still identical to the starting template:
    # an unfilled form would only cost tokens. SOUL.md is always included,
    # because the default personality is meant to be used.
    _SKIPPABLE_DEFAULTS = {"AGENTS.md", "USER.md"}

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
        # The tool contract's ``## Scheduling`` section; the loop turns it on
        # when it registers the ``cron`` tool (nanobot: always in the prompt,
        # where the main agent always has the tool).
        self.include_scheduling = False

    # --- system prompt -------------------------------------------------------

    def build_system_prompt(
        self,
        *,
        channel: str | None = None,
        session_summary: SessionSummary | None = None,
        workspace: Path | None = None,
        include_memory: bool = True,
    ) -> str:
        """Identity, workspace files, tool usage notes, memory, skills, then the conversation summary.

        ``workspace`` is the turn's project folder (nanobot: the scope's
        ``project_path``); None means the agent workspace. AGENTS.md comes
        from the project, SOUL.md, USER.md, memory and skills from the agent
        workspace, and a ``# Current Project`` section names the project when
        it differs (nanobot context.py:88-111).
        """
        root = workspace or self.workspace
        parts = [self._get_identity(channel=channel, workspace=root)]
        bootstrap = self._load_bootstrap_files(root)
        if bootstrap:
            parts.append(bootstrap)
        parts.append(render_template("agent/tool_contract.md", scheduling=self.include_scheduling))
        project_path = root.expanduser().resolve()
        if project_path != self.workspace.expanduser().resolve():
            parts.append(
                "# Current Project\n\n"
                f"Working directory: {project_path}\n"
                "Use it as the default root for project files and relative tool paths."
            )
        if include_memory:
            memory = self.memory.read_memory()
            # Bare: MEMORY.md already starts with "# Long-term Memory", so no
            # extra heading (nanobot/agent/context.py:113-116).
            if memory and not self._is_template_content(memory, "memory/MEMORY.md"):
                parts.append(memory)
        active_skills = self.skills.get_always_skills()
        if active_skills:
            active_content = self.skills.load_skills_for_context(active_skills)
            if active_content:
                parts.append(f"# Active Skills\n\n{active_content}")
        skills_summary = self.skills.build_skills_summary(exclude=set(active_skills), workspace=root)
        if skills_summary:
            parts.append(render_template("agent/skills_section.md", skills_summary=skills_summary))
        if session_summary and session_summary["text"] != "(nothing)":
            parts.append(
                "[Archived Context Summary]\n\n"
                f"Previous conversation summary (last active {session_summary['last_active']}):\n"
                f"{session_summary['text']}"
            )
        return SECTION_SEPARATOR.join(parts)

    def _get_identity(self, channel: str | None = None, workspace: Path | None = None) -> str:
        workspace_path = str((workspace or self.workspace).expanduser().resolve())
        agent_workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = (
            f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, "
            f"Python {platform.python_version()}"
        )
        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            agent_workspace_path=agent_workspace_path,
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
        )

    def _load_bootstrap_files(self, workspace: Path | None = None) -> str:
        """Project instructions plus the agent's own profile files (nanobot context.py:177)."""
        parts: list[str] = []
        project_root = workspace or self.workspace
        sources = {"AGENTS.md": project_root, "SOUL.md": self.workspace, "USER.md": self.workspace}
        for filename in self.BOOTSTRAP_FILES:
            file_path = sources[filename] / filename
            if not file_path.exists():
                continue
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                continue
            if filename in self._SKIPPABLE_DEFAULTS and self._is_template_content(content, filename):
                continue
            parts.append(f"## {filename}\n\n{content}")
        return "\n\n".join(parts)

    # SHA-256 of earlier bundled versions (stripped), still unedited starting
    # templates in older workspaces. NanoClaw only: AGENTS.md lacked nanobot's
    # cron and heartbeat sections until layer 21.
    _PREVIOUS_TEMPLATE_SHA256 = {
        "AGENTS.md": {"44fbc0f752e15cdb8fdc8a0a310f3d323b400f9092d5fb826692392473bde5f5"},
    }

    @classmethod
    def _is_template_content(cls, content: str, template_name: str) -> bool:
        """True if ``content`` is still the unedited starting template (this
        version or an earlier bundled one)."""
        template = load_bundled_template(template_name)
        if template is not None and content.strip() == template.strip():
            return True
        digest = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
        return digest in cls._PREVIOUS_TEMPLATE_SHA256.get(template_name, set())

    # --- messages ---------------------------------------------------------------

    def build_current_message(
        self,
        current_message: str,
        *,
        runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None,
    ) -> dict[str, Any]:
        """The user's new message, with runtime context (e.g. the time) appended.

        Skills named as ``$skill-name`` in the message are appended too, unless
        the caller already included that block.
        """
        blocks = list(runtime_context_blocks or ())
        skill_context = self.skills.build_explicit_skill_runtime_context(current_message)
        if skill_context is not None and skill_context not in blocks:
            blocks.append(skill_context)
        content, marker = append_runtime_context(current_message, blocks)
        current: dict[str, Any] = {"role": "user", "content": content}
        if marker is not None:
            # Internal: the session saves it; providers drop ``_meta``.
            current["_meta"] = {RUNTIME_CONTEXT_MESSAGE_META: marker}
        return current

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str | None,
        *,
        channel: str | None = None,
        runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None,
        session_summary: SessionSummary | None = None,
        workspace: Path | None = None,
    ) -> list[dict[str, Any]]:
        """``[system, *history, current user message]``.

        If the history already ends with a user message (for example, the
        previous turn failed before the model answered), the new message is
        merged into it so two user messages are never sent in a row. With
        ``current_message=None`` (used when archiving) there is no new message.
        """
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    channel=channel, session_summary=session_summary, workspace=workspace
                ),
            },
            *history,
        ]
        if current_message is None:
            return messages
        current = self.build_current_message(current_message, runtime_context_blocks=runtime_context_blocks)
        if messages[-1].get("role") == "user":
            merged = dict(messages[-1])
            merged["content"] = self._merge_message_content(merged.get("content"), current["content"])
            if isinstance(current.get("_meta"), dict):
                merged["_meta"] = {**(merged.get("_meta") or {}), **current["_meta"]}
            return [*messages[:-1], merged]
        return [*messages, current]

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            if not left:
                return right
            if not right:
                return left
            return f"{left}\n\n{right}"

        def to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return to_blocks(left) + to_blocks(right)
