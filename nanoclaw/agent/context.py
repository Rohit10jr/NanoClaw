"""Building the system prompt and the message list for a turn.

Mirrors nanobot/agent/context.py (``ContextBuilder``). The system prompt is
assembled from Markdown: rendered templates plus the user-editable files in the
workspace (``AGENTS.md``, ``SOUL.md``, ``USER.md``). Memory (layer 6) and
skills (layer 7) will add more sections.
"""

from __future__ import annotations

import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nanoclaw.runtime_context import RuntimeContextBlock, append_runtime_context
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

    # --- system prompt -------------------------------------------------------

    def build_system_prompt(self, *, channel: str | None = None) -> str:
        """Identity, then the workspace files, then the tool usage notes."""
        parts = [self._get_identity(channel=channel)]
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        parts.append(render_template("agent/tool_contract.md"))
        return SECTION_SEPARATOR.join(parts)

    def _get_identity(self, channel: str | None = None) -> str:
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = (
            f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, "
            f"Python {platform.python_version()}"
        )
        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            # nanobot can keep the agent's own files apart from a project
            # folder; NanoClaw uses one folder for both (see the roadmap).
            agent_workspace_path=workspace_path,
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
        )

    def _load_bootstrap_files(self) -> str:
        parts: list[str] = []
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if not file_path.exists():
                continue
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                continue
            if filename in self._SKIPPABLE_DEFAULTS and self._is_template_content(content, filename):
                continue
            parts.append(f"## {filename}\n\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _is_template_content(content: str, template_name: str) -> bool:
        """True if ``content`` is still the unedited starting template."""
        template = load_bundled_template(template_name)
        return template is not None and content.strip() == template.strip()

    # --- messages ---------------------------------------------------------------

    def build_current_message(
        self,
        current_message: str,
        *,
        runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None,
    ) -> dict[str, Any]:
        """The user's new message, with runtime context (e.g. the time) appended."""
        content = append_runtime_context(current_message, list(runtime_context_blocks or ()))
        return {"role": "user", "content": content}

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        *,
        channel: str | None = None,
        runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None,
    ) -> list[dict[str, Any]]:
        """``[system, *history, current user message]``.

        If the history already ends with a user message (for example, the
        previous turn failed before the model answered), the new message is
        merged into it so two user messages are never sent in a row.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt(channel=channel)},
            *history,
        ]
        current = self.build_current_message(current_message, runtime_context_blocks=runtime_context_blocks)
        if messages[-1].get("role") == "user":
            merged = dict(messages[-1])
            merged["content"] = self._merge_message_content(merged.get("content"), current["content"])
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
