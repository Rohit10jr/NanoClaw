"""How blocked tool calls are reported to the model.

Mirrors nanobot/agent/tools/execution.py (``is_ssrf_violation``,
``_classify_violation``, ``_ssrf_soft_payload``) and the workspace-violation
throttle in nanobot/utils/runtime.py (``workspace_violation_signature``,
``repeated_workspace_violation_error``).

- A private-network block gets a fixed "non-bypassable" note instead of the
  usual retry hint, so the model stops trying other ways in.
- A workspace-boundary error keeps the retry hint, but after more than two
  attempts at the same outside target in one run the model gets a firmer
  "stop retrying" message.
"""

from __future__ import annotations

import ntpath
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

# SSRF is a hard security block at the tool boundary, but the agent turn
# should recover conversationally instead of aborting the runtime.
_SSRF_MARKERS: tuple[str, ...] = (
    "internal/private url detected",
    "private/internal address",
    "private address",
)
_SSRF_BOUNDARY_NOTE = (
    "This is a non-bypassable security boundary. Stop trying to access "
    "private/internal URLs. Do not retry with curl, wget, encoded IPs, "
    "alternate DNS, redirects, proxies, or another tool. Ask the user for "
    "local files, logs, screenshots, or an explicit safe public URL instead. "
    "If the user explicitly trusts this private URL, ask them to whitelist "
    "the exact IP/CIDR via NANOCLAW_SSRF_WHITELIST."
)
# Non-SSRF boundary markers returned to the model as recoverable tool errors.
_WORKSPACE_VIOLATION_MARKERS: tuple[str, ...] = (
    "outside the configured workspace",
    "outside allowed directory",
    "working_dir is outside",
    "working_dir could not be resolved",
    "path outside working dir",
    "path traversal detected",
)
_MAX_REPEAT_WORKSPACE_VIOLATIONS = 2
# nanobot utils/runtime.py:136 matches only ``/`` and ``~`` paths. NanoClaw also
# matches Windows drive paths (``C:\x``, ``C:/x``, ``C:\``) and UNC paths
# (``\\server\share``), so repeated ``exec`` attempts on them escalate too.
# The drive letter must start a token and be followed by a slash, so
# ``http://x`` and ``a:b`` are not paths.
_OUTSIDE_PATH_PATTERN = re.compile(
    r"(?:^|[\s|>'\"])("
    r"/[^\s\"'>;|<]+"
    r"|~[^\s\"'>;|<]+"
    r"|[A-Za-z]:[\\/][^\s\"'>;|<]*"
    r"|\\\\[^\s\"'>;|<]+"
    r")"
)
_WINDOWS_PATH_PATTERN = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def is_ssrf_violation(text: str) -> bool:
    """Return whether a tool error describes a blocked private-network request."""
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _SSRF_MARKERS)


def _is_workspace_violation(text: str) -> bool:
    """Return whether text describes any workspace or network boundary rejection."""
    if not text:
        return False
    lowered = text.lower()
    if is_ssrf_violation(lowered):
        return True
    return any(marker in lowered for marker in _WORKSPACE_VIOLATION_MARKERS)


def _ssrf_soft_payload(raw_text: str) -> str:
    text = raw_text.strip() or "Error: request blocked by SSRF guard"
    return f"{text}\n\n{_SSRF_BOUNDARY_NOTE}"


def workspace_violation_signature(tool_name: str, arguments: Any) -> str | None:
    """Return a stable cross-tool signature for the outside-workspace target."""
    if not isinstance(arguments, dict):
        return None
    for key in ("path", "file_path", "target", "source", "destination"):
        val = arguments.get(key)
        if isinstance(val, str) and val.strip():
            return _normalize_violation_target(val.strip())

    if tool_name in {"exec", "shell"}:
        cmd = str(arguments.get("command") or "").strip()
        if cmd:
            match = _OUTSIDE_PATH_PATTERN.search(cmd)
            if match:
                return _normalize_violation_target(match.group(1))
        cwd = str(arguments.get("working_dir") or "").strip()
        if cwd:
            return _normalize_violation_target(cwd)

    return None


def _normalize_violation_target(raw: str) -> str:
    """Normalize *raw* path so that equivalent spellings collide on the same key."""
    if os.name != "nt" and _WINDOWS_PATH_PATTERN.match(raw):
        # A Windows path on a POSIX host: Path would treat it as a relative
        # file name, so ``C:\x`` and ``C:/x`` would not collide (NanoClaw only).
        return f"violation:{ntpath.normpath(raw).replace(chr(92), '/')}".lower()
    try:
        normalized = Path(raw).expanduser().resolve().as_posix()
    except Exception:
        normalized = raw.replace("\\", "/")
    return f"violation:{normalized}".lower()


def repeated_workspace_violation_error(tool_name: str, arguments: Any, seen_counts: dict[str, int]) -> str | None:
    """Return an escalated error after repeated bypass attempts."""
    signature = workspace_violation_signature(tool_name, arguments)
    if signature is None:
        return None
    count = seen_counts.get(signature, 0) + 1
    seen_counts[signature] = count
    if count <= _MAX_REPEAT_WORKSPACE_VIOLATIONS:
        return None
    logger.warning(
        "Escalating repeated workspace bypass attempt {} (attempt {})",
        signature[:160],
        count,
    )
    target = signature.split("violation:", 1)[1] if "violation:" in signature else signature
    return (
        "Error: refusing repeated workspace-bypass attempts.\n"
        f"You have tried to access '{target}' (or an equivalent path) "
        f"{count} times in this turn. This is a hard policy boundary -- "
        "switching tools, shell tricks, working_dir overrides, symlinks, "
        "or base64 piping will NOT change the answer. Stop retrying. "
        "If the user genuinely needs this resource, tell them you cannot "
        "access it and ask how they want to proceed (e.g. copy the file "
        "into the workspace, or disable restrict_to_workspace for this run)."
    )


def classify_violation(
    *,
    raw_text: str,
    soft_payload: str,
    tool_name: str,
    arguments: Any,
    workspace_violation_counts: dict[str, int],
) -> str | None:
    """The text to return for a boundary error, or ``None`` if it isn't one."""
    if is_ssrf_violation(raw_text):
        logger.warning(
            "Tool {} blocked by SSRF guard; returning non-retryable tool error: {}",
            tool_name,
            raw_text.replace("\n", " ").strip()[:200],
        )
        return _ssrf_soft_payload(raw_text)
    if _is_workspace_violation(raw_text):
        escalation = repeated_workspace_violation_error(tool_name, arguments, workspace_violation_counts)
        if escalation is not None:
            logger.warning("Tool {} hit workspace boundary repeatedly; escalating hint", tool_name)
            return escalation
        return soft_payload
    return None
