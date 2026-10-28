"""Skill summaries and the enable switch for the web UI's Skills page.

Ported from nanobot/webui/skills_api.py (c4dac64): ``webui_skills_payload``
(``GET /api/webui/skills``), ``webui_skill_detail_payload``
(``GET /api/webui/skills/{name}``) and ``set_webui_skill_enabled`` (the
``skill.update`` mutation). Payloads never include local paths.

The enable switch persists like nanobot's: it loads ``config.json`` as
stored (``${VAR}`` references unresolved), changes
``agents.defaults.disabledSkills`` and saves the file again, so keys
NanoClaw's schema does not know are dropped, as in nanobot.

Not ported: ``delete_webui_skill`` (``skill.delete``) and the skills
marketplace (``skill.install``, search, trending).
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from nanoclaw.agent.skills import SkillsLoader


class SkillManagementError(Exception):
    """A safe skill-management error for the web UI."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def webui_skills_payload(
    workspace_path: Path,
    *,
    disabled_skills: set[str] | None = None,
) -> dict[str, Any]:
    """Every skill (disabled ones too, ``enabled: false``), workspace ones first."""
    loader = SkillsLoader(workspace_path)
    with loader.snapshot():
        entries = sorted(
            loader.list_skills(filter_unavailable=False),
            key=lambda entry: (entry.get("source") != "workspace", entry["name"]),
        )
        return {"skills": [_skill_payload(loader, entry, disabled_skills=disabled_skills) for entry in entries]}


def webui_skill_detail_payload(
    workspace_path: Path,
    name: str,
    *,
    disabled_skills: set[str] | None = None,
) -> dict[str, Any] | None:
    """One skill with its requirements, install commands and Markdown, or ``None``."""
    loader = SkillsLoader(workspace_path)
    with loader.snapshot():
        entries = loader.list_skills(filter_unavailable=False)
        entry = next((item for item in entries if item["name"] == name), None)
        if entry is None:
            return None
        metadata = loader.get_skill_metadata(name)
        return {
            **_skill_payload(loader, entry, metadata=metadata, disabled_skills=disabled_skills),
            "requirements": loader.get_skill_requirements(name),
            "install_options": _install_options(metadata),
            "raw_markdown": loader.load_skill(name) or "",
        }


def set_webui_skill_enabled(
    workspace_path: Path,
    name: str,
    *,
    enabled: bool,
    disabled_skills: set[str],
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Persist and apply one skill's enabled state.

    ``disabled_skills`` is updated in place to the saved set.
    """
    from nanoclaw.config.loader import load_config, save_config

    _require_skill_entry(workspace_path, name)
    config = load_config(config_path)
    next_disabled = set(config.agents.defaults.disabled_skills)
    if enabled:
        next_disabled.discard(name)
    else:
        next_disabled.add(name)
    if next_disabled != set(config.agents.defaults.disabled_skills):
        config.agents.defaults.disabled_skills = sorted(next_disabled)
        save_config(config, config_path)
    disabled_skills.clear()
    disabled_skills.update(next_disabled)
    return {"name": name, "enabled": enabled, "deleted": False}


def _require_skill_entry(workspace_path: Path, name: str) -> dict[str, str]:
    if not name or "/" in name or "\\" in name:
        raise SkillManagementError("invalid skill name")
    entry = next(
        (item for item in SkillsLoader(workspace_path).list_skills(filter_unavailable=False) if item["name"] == name),
        None,
    )
    if entry is None:
        raise SkillManagementError("skill not found", status=404)
    return entry


def _skill_payload(
    loader: SkillsLoader,
    entry: dict[str, str],
    *,
    metadata: dict[str, Any] | None = None,
    disabled_skills: set[str] | None = None,
) -> dict[str, Any]:
    name = entry["name"]
    metadata = metadata if metadata is not None else loader.get_skill_metadata(name)
    available, unavailable_reason = loader.get_skill_availability(name)
    source = entry.get("source", "unknown")
    return {
        "name": name,
        "description": _description(metadata, name),
        "source": source,
        "enabled": name not in (disabled_skills or set()),
        "deletable": source == "workspace",
        "available": available,
        "unavailable_reason": unavailable_reason,
    }


def _description(metadata: dict[str, Any] | None, fallback: str) -> str:
    if metadata is None:
        return fallback
    value = metadata.get("description")
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _nanobot_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    raw = metadata.get("metadata")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(raw, dict):
        return {}
    # The loader's lookup order: ``nanoclaw`` (NanoClaw only), ``nanobot``, ``openclaw``.
    payload = raw.get("nanoclaw", raw.get("nanobot", raw.get("openclaw", {})))
    return payload if isinstance(payload, dict) else {}


def _install_options(metadata: dict[str, Any] | None) -> list[dict[str, str]]:
    """Copyable setup commands a skill declares (``brew`` / ``apt`` only)."""
    install = _nanobot_metadata(metadata).get("install")
    if not isinstance(install, list):
        return []
    options: list[dict[str, str]] = []
    for item in install:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in {"brew", "apt"}:
            continue
        package = item.get("formula") if kind == "brew" else item.get("package")
        if not isinstance(package, str) or not package.strip():
            continue
        if kind == "brew":
            command = f"brew install {shlex.quote(package.strip())}"
        else:
            command = f"sudo apt-get install -y {shlex.quote(package.strip())}"
        option_id = item.get("id")
        label = item.get("label")
        options.append(
            {
                "id": option_id if isinstance(option_id, str) else kind,
                "kind": kind,
                "label": label if isinstance(label, str) else f"Install with {kind}",
                "command": command,
            }
        )
    return options
