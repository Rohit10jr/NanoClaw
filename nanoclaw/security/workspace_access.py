"""The project folder a turn works on, and its file-access mode.

Mirrors nanobot/security/workspace_access.py. The agent's own files (SOUL.md,
USER.md, memory/, skills/) live in its *workspace*; the code it works on can be
a separate *project* folder. A ``WorkspaceScope`` names that project and the
access mode (``restricted``: tools stay inside the project, ``full``: no
workspace restriction). The loop binds it per turn (``bind_workspace_scope``,
nanobot loop.py:1293), subagents bind the scope of the turn that started
them (nanobot subagent.py:824), and every tool asks ``current_tool_workspace``
for the root of relative paths and the restriction.

In nanobot only the WebUI chooses another project (``WorkspaceScopeResolver``,
stored in session metadata); its other channels always use the agent
workspace. NanoClaw takes it from ``--project DIR`` / ``NANOCLAW_PROJECT`` for
the whole run, so the resolver, the session-metadata helpers and the WebUI
loopback rule are left out.
"""

from __future__ import annotations

import json
import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nanoclaw.runtime_context import RuntimeContextBlock, wrap_runtime_context_lines

WorkspaceAccessMode = Literal["restricted", "full"]
_ACCESS_MODES = {"restricted", "full"}

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled", ""}
_PROVIDER_LABELS = {
    "none": "None",
    "unknown": "Unknown system sandbox",
    "macos_app_sandbox": "macOS App Sandbox",
    "bwrap": "Bubblewrap",
    "seatbelt": "macOS Seatbelt",
}
# nanobot reads NANOBOT_WORKSPACE_SANDBOX_ENFORCED / _PROVIDER (and the older
# NANOBOT_SANDBOX_ENFORCED) to report a sandbox the host put nanobot in.
ENV_SANDBOX_ENFORCED = "NANOCLAW_WORKSPACE_SANDBOX_ENFORCED"
ENV_SANDBOX_PROVIDER = "NANOCLAW_WORKSPACE_SANDBOX_PROVIDER"

_CURRENT_WORKSPACE_SCOPE: ContextVar[WorkspaceScope | None] = ContextVar(
    "nanoclaw_workspace_scope",
    default=None,
)


class WorkspaceScopeError(ValueError):
    """An invalid project folder or access mode."""


@dataclass(frozen=True)
class WorkspaceSandboxStatus:
    """How the workspace restriction is enforced (for display and the runtime block)."""

    restrict_to_workspace: bool
    workspace_root: str
    level: str
    enforced: bool
    provider: str
    provider_label: str
    summary: str

    def as_dict(self) -> dict[str, object]:
        return {
            "restrict_to_workspace": self.restrict_to_workspace,
            "workspace_root": self.workspace_root,
            "level": self.level,
            "enforced": self.enforced,
            "provider": self.provider,
            "provider_label": self.provider_label,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class WorkspaceScope:
    """Effective project root and access mode for one agent turn."""

    project_path: Path
    access_mode: WorkspaceAccessMode
    restrict_to_workspace: bool
    sandbox_status: WorkspaceSandboxStatus

    @property
    def project_name(self) -> str:
        return self.project_path.name or str(self.project_path)

    def metadata(self) -> dict[str, str]:
        return {"project_path": str(self.project_path), "access_mode": self.access_mode}

    def payload(self) -> dict[str, Any]:
        return {
            **self.metadata(),
            "project_name": self.project_name,
            "restrict_to_workspace": self.restrict_to_workspace,
            "sandbox_status": self.sandbox_status.as_dict(),
        }


@dataclass(frozen=True)
class ToolWorkspace:
    """Workspace policy resolved for one tool call."""

    project_path: Path | None
    restrict_to_workspace: bool
    scope: WorkspaceScope | None = None

    @property
    def allowed_root(self) -> Path | None:
        if self.restrict_to_workspace and self.project_path is not None:
            return self.project_path
        return None


def workspace_sandbox_status(
    *,
    restrict_to_workspace: bool,
    workspace: str | Path,
    environ: dict[str, str] | None = None,
) -> WorkspaceSandboxStatus:
    """How workspace restriction is enforced on this host (nanobot :156)."""
    workspace_root = str(Path(workspace).expanduser().resolve(strict=False))
    provider = _env_system_provider(environ)
    if not restrict_to_workspace:
        return WorkspaceSandboxStatus(
            restrict_to_workspace=False,
            workspace_root=workspace_root,
            level="off",
            enforced=False,
            provider="none",
            provider_label=_provider_label("none"),
            summary="Workspace restriction is disabled.",
        )
    if provider:
        label = _provider_label(provider)
        return WorkspaceSandboxStatus(
            restrict_to_workspace=True,
            workspace_root=workspace_root,
            level="system",
            enforced=True,
            provider=provider,
            provider_label=label,
            summary=f"Workspace restriction is system-enforced by {label}.",
        )
    return WorkspaceSandboxStatus(
        restrict_to_workspace=True,
        workspace_root=workspace_root,
        level="application",
        enforced=False,
        provider="none",
        provider_label=_provider_label("none"),
        summary="Workspace restriction uses nanoclaw application-level guards.",
    )


def default_access_mode(restrict_to_workspace: bool) -> WorkspaceAccessMode:
    return "restricted" if restrict_to_workspace else "full"


def build_workspace_scope(project_path: str | Path, access_mode: str) -> WorkspaceScope:
    mode = _normalize_access_mode(access_mode)
    root = Path(project_path).expanduser().resolve(strict=False)
    restrict = mode == "restricted"
    return WorkspaceScope(
        project_path=root,
        access_mode=mode,
        restrict_to_workspace=restrict,
        sandbox_status=workspace_sandbox_status(restrict_to_workspace=restrict, workspace=root),
    )


def default_workspace_scope(workspace: str | Path, restrict_to_workspace: bool) -> WorkspaceScope:
    """The agent workspace itself as the project (every nanobot channel but the WebUI)."""
    return build_workspace_scope(workspace, default_access_mode(restrict_to_workspace))


def validate_workspace_scope_payload(
    raw: Any,
    *,
    default_workspace: str | Path,
    default_restrict_to_workspace: bool,
) -> WorkspaceScope:
    """Check a requested ``{"project_path", "access_mode"}`` (nanobot :251).

    The path must be absolute (``~`` is expanded) and an existing directory.
    """
    if raw is None:
        return default_workspace_scope(default_workspace, default_restrict_to_workspace)
    if not isinstance(raw, dict):
        raise WorkspaceScopeError("workspace_scope must be an object")
    raw_path = raw.get("project_path") or raw.get("path")
    if raw_path is None or raw_path == "":
        raw_path = str(Path(default_workspace).expanduser().resolve(strict=False))
    if not isinstance(raw_path, str):
        raise WorkspaceScopeError("project_path must be a string")
    if "\0" in raw_path:
        raise WorkspaceScopeError("project_path contains invalid characters")
    project = Path(raw_path).expanduser()
    if not project.is_absolute():
        raise WorkspaceScopeError("project_path must be absolute")
    project = project.resolve(strict=False)
    if not project.is_dir():
        raise WorkspaceScopeError("project_path must be an existing directory")
    raw_mode = raw.get("access_mode")
    if raw_mode is None:
        raw_mode = default_access_mode(default_restrict_to_workspace)
    if not isinstance(raw_mode, str):
        raise WorkspaceScopeError("access_mode must be a string")
    return build_workspace_scope(project, raw_mode)


def project_workspace_scope(
    project: str | Path | None, *, workspace: str | Path, restrict_to_workspace: bool
) -> WorkspaceScope:
    """The scope for ``--project DIR`` (NanoClaw): the agent workspace when
    ``project`` is None, else ``project`` (relative to the current directory)
    with the access mode that ``restrict_to_workspace`` gives."""
    if project is None:
        return default_workspace_scope(workspace, restrict_to_workspace)
    path = Path(project).expanduser()
    return validate_workspace_scope_payload(
        {"project_path": str(path if path.is_absolute() else Path.cwd() / path)},
        default_workspace=workspace,
        default_restrict_to_workspace=restrict_to_workspace,
    )


def bind_workspace_scope(scope: WorkspaceScope) -> Token[WorkspaceScope | None]:
    return _CURRENT_WORKSPACE_SCOPE.set(scope)


def reset_workspace_scope(token: Token[WorkspaceScope | None]) -> None:
    _CURRENT_WORKSPACE_SCOPE.reset(token)


def current_workspace_scope() -> WorkspaceScope | None:
    return _CURRENT_WORKSPACE_SCOPE.get()


def current_tool_workspace(
    default_workspace: str | Path | None,
    *,
    restrict_to_workspace: bool = False,
    sandbox_restricts_workspace: bool = False,
    scope: WorkspaceScope | None = None,
) -> ToolWorkspace:
    """Tool policy from an explicit scope or the bound one (nanobot :357).

    Without a bound scope (a tool used outside a turn, as in tests), the
    tool's own workspace and restriction apply.
    """
    scope = scope if scope is not None else current_workspace_scope()
    project_path = (
        scope.project_path
        if scope is not None
        else Path(default_workspace).expanduser() if default_workspace is not None else None
    )
    restrict = (
        scope.restrict_to_workspace if scope is not None else bool(restrict_to_workspace)
    ) or sandbox_restricts_workspace
    return ToolWorkspace(project_path=project_path, restrict_to_workspace=restrict, scope=scope)


def workspace_access_block(
    scope: WorkspaceScope,
    *,
    sandbox: str = "",
) -> RuntimeContextBlock:
    """The per-turn ``workspace_access`` runtime context (nanobot
    ``AgentRuntimeControl.workspace_context``, runtime_control.py:188).

    nanobot adds it to every turn, whether or not the project differs from
    the agent workspace. ``sandbox`` is ``tools.exec.sandbox``: a configured
    sandbox forces the restriction (nanobot ``sandbox_restricts_workspace``).
    """
    access = current_tool_workspace(
        None, sandbox_restricts_workspace=bool(sandbox), scope=scope
    )
    assert access.project_path is not None
    status = workspace_sandbox_status(
        workspace=access.project_path, restrict_to_workspace=access.restrict_to_workspace
    )
    # A valid directory can contain newlines or prompt delimiters. Keep it data.
    root = json.dumps(status.workspace_root, ensure_ascii=False)
    root = root.replace("[", "\\u005b").replace("]", "\\u005d")
    lines = [
        f"Current project (JSON path): {root}",
        (
            "File access: project and tool-specific allowed paths only."
            if status.restrict_to_workspace
            else "File access: no workspace restriction."
        ),
        "OS permissions and tool rules still apply.",
    ]
    if sandbox:
        lines.append("Shell sandbox configured; extra binds do not grant file-tool access.")
    return RuntimeContextBlock(source="workspace_access", content=wrap_runtime_context_lines(lines))


def _env_system_provider(environ: dict[str, str] | None = None) -> str | None:
    env = environ if environ is not None else os.environ
    marker = env.get(ENV_SANDBOX_ENFORCED)
    if marker is None:
        return None
    normalized = marker.strip().lower()
    if normalized in _FALSE_VALUES:
        return None
    if normalized in _TRUE_VALUES:
        return _normalize_provider(env.get(ENV_SANDBOX_PROVIDER))
    return _normalize_provider(marker)


def _normalize_provider(value: str | None) -> str:
    if not value:
        return "unknown"
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or "unknown"


def _provider_label(provider: str) -> str:
    if provider in _PROVIDER_LABELS:
        return _PROVIDER_LABELS[provider]
    return provider.replace("_", " ").title()


def _normalize_access_mode(value: str) -> WorkspaceAccessMode:
    mode = value.strip().lower().replace("_", "-")
    if mode == "restrict":
        mode = "restricted"
    if mode == "full-access":
        mode = "full"
    if mode not in _ACCESS_MODES:
        raise WorkspaceScopeError("access_mode must be restricted or full")
    return mode  # type: ignore[return-value]
