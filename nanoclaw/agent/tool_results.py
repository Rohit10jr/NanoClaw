"""Keeping tool results within budget without losing them.

Mirrors nanobot's ``ContextGovernor.normalize_tool_result``
(agent/context_governance.py) and ``maybe_persist_tool_result``
(utils/helpers.py). A result longer than ``max_chars`` is saved in full to
``<workspace>/.nanoclaw/tool-results/<session>/`` and the model gets a preview (start
and end) plus the file path, so it can ``read_file`` the rest. Without a
workspace the result is truncated instead.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import time
from pathlib import Path

from nanoclaw.agent.messages import is_blank, truncate_text

TOOL_RESULTS_DIR = ".nanoclaw/tool-results"
# Other sessions' folders are removed after a week, and beyond 32 folders.
TOOL_RESULT_RETENTION_SECS = 7 * 24 * 60 * 60
TOOL_RESULT_MAX_BUCKETS = 32

# read_file pages its own output (128k cap plus a "Use offset=N to continue"
# footer), so it is never cut or offloaded; cutting it would lose the footer.
OFFLOAD_EXEMPT_TOOLS = frozenset({"read_file"})

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')


def safe_filename(name: str) -> str:
    return _UNSAFE_CHARS.sub("_", name).strip()


def ensure_nonempty_tool_result(tool_name: str, content: str) -> str:
    """An empty result becomes a marker, so the model doesn't think the tool hung."""
    return f"({tool_name} completed with no output)" if is_blank(content) else content


def render_tool_result_reference(
    reference_path: str,
    *,
    original_size: int,
    preview: str,
    max_chars: int | None = None,
) -> str:
    result = (
        "[tool output persisted]\n"
        f"Full output saved to workspace path: {reference_path}\n"
        f"Original size: {original_size} chars\n"
        f"Preview:\n{preview}"
        "\n...\nPreview is also truncated."
        "\nResult truncated. Read the saved file if you need the complete output."
    )
    if max_chars and len(result) > max_chars:
        result = f"[truncated: {reference_path}]"
    return result


def _bucket_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def cleanup_tool_result_buckets(root: Path, current_bucket: Path) -> None:
    """Delete old session folders: older than a week, or beyond the newest 32.

    The current session's folder is never deleted. (nanobot ``_cleanup_tool_result_buckets``)
    """
    siblings = [path for path in root.iterdir() if path.is_dir() and path != current_bucket]
    cutoff = time.time() - TOOL_RESULT_RETENTION_SECS
    for path in siblings:
        if _bucket_mtime(path) < cutoff:
            shutil.rmtree(path, ignore_errors=True)
    keep = max(TOOL_RESULT_MAX_BUCKETS - 1, 0)
    siblings = [path for path in siblings if path.exists()]
    if len(siblings) <= keep:
        return
    siblings.sort(key=_bucket_mtime, reverse=True)
    for path in siblings[keep:]:
        shutil.rmtree(path, ignore_errors=True)


def maybe_persist_tool_result(
    workspace: Path | None,
    bucket: str,
    tool_call_id: str,
    content: str,
    *,
    max_chars: int,
) -> str:
    """Return ``content`` if it fits; otherwise save it and return a reference with a preview."""
    if workspace is None or max_chars <= 0 or len(content) <= max_chars:
        return content

    root = workspace / TOOL_RESULTS_DIR
    directory = root / safe_filename(bucket)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        cleanup_tool_result_buckets(root, directory)
    except OSError:
        pass
    # The name includes a hash of the content, so the same tool-call ID with
    # different output can never share a file: every reference in the history
    # keeps pointing at the output it previewed. Identical output reuses the
    # file. (nanobot names files by ID only and never overwrites, so a repeated
    # ID can point the model at stale output: docs/UPSTREAM_BUGS.md, bug 3.)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    path = directory / f"{safe_filename(tool_call_id)}-{digest}.txt"
    if not path.exists():
        path.write_text(content, encoding="utf-8")

    reference_path = str(path.resolve())
    overhead = len(render_tool_result_reference(reference_path, original_size=len(content), preview=""))
    available = max(0, max_chars - overhead)
    separator = "\n...\n"
    tail_chars = min(1200, max(0, (available - len(separator)) // 4))
    if tail_chars:
        head_chars = available - tail_chars - len(separator)
        preview = content[:head_chars] + separator + content[-tail_chars:]
    else:
        preview = content[:available]
    return render_tool_result_reference(
        reference_path, original_size=len(content), preview=preview, max_chars=max_chars
    )


def normalize_tool_result(
    tool_name: str,
    tool_call_id: str,
    content: str,
    *,
    workspace: Path | None,
    max_chars: int,
    bucket: str = "default",
) -> str:
    """Apply the empty-result marker, then keep the result within ``max_chars``."""
    content = ensure_nonempty_tool_result(tool_name, content)
    if tool_name in OFFLOAD_EXEMPT_TOOLS:
        return content
    if workspace is None:
        return truncate_text(content, max_chars)
    return maybe_persist_tool_result(workspace, bucket, tool_call_id, content, max_chars=max_chars)
