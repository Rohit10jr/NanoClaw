"""apply_patch: structured, all-or-nothing edits across one or more files.

Mirrors nanobot/agent/tools/apply_patch.py (c4dac64): ``replace`` swaps one
unique text match, ``add`` creates a file or appends to an existing one; up to
20 edits, applied in order to an in-memory copy, then written together. Any
failure (a bad edit, or a write error) leaves every file as it was.
``dry_run`` validates and summarizes without writing.

Paths go through the same workspace policy as the other file tools
(``_FsTool._resolve_write``). NanoClaw differences: the result is a plain
string (nanobot returns a ``FileEditResult`` carrying diffs for its activity
events). Like nanobot, a successful write is recorded in the file state, so
the next ``read_file`` of a patched file returns its contents.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from nanoclaw.agent.tools.base import ToolResult
from nanoclaw.agent.tools.file_state import FileStates
from nanoclaw.agent.tools.filesystem import _count_changed_lines, _FsTool
from nanoclaw.agent.tools.schema import object_schema


class _PatchError(ValueError):
    pass


class _GuardRefusal(Exception):
    """A ``write_check`` refusal; its text is returned as is."""


def _validate_patch_path(path: str) -> str:
    normalized = path.strip()
    if not normalized:
        raise _PatchError("patch path cannot be empty")
    if "\0" in normalized:
        raise _PatchError(f"patch path contains a null byte: {path!r}")
    return normalized


def _append_text(content: str, addition: str) -> str:
    """Append text without merging it into an unterminated final line."""
    base = content.replace("\r\n", "\n")
    extra = addition.replace("\r\n", "\n")
    if base and extra and not base.endswith("\n") and not extra.startswith("\n"):
        base += "\n"
    combined = base + extra
    if combined and not combined.endswith("\n"):
        combined += "\n"
    return combined


def _read_utf8(source: Path, path: str) -> str:
    raw = source.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise _PatchError(f"file is not UTF-8 text: {path}") from None


class ApplyPatchTool(_FsTool):
    """Apply file edits by providing structured edit instructions.

    ``write_check(raw_path, resolved_path)`` may refuse a target with a full
    error text; Dream uses it for its write scope (nanobot gives Dream's tools
    ``allowed_dir`` / ``extra_write_allowed_files`` instead).
    """

    def __init__(
        self,
        workspace: Path | None = None,
        *,
        restrict_to_workspace: bool = False,
        write_check: Callable[[str, Path], str | None] | None = None,
        file_states: FileStates | None = None,
    ) -> None:
        super().__init__(workspace, restrict_to_workspace=restrict_to_workspace, file_states=file_states)
        self._write_check = write_check

    @property
    def name(self) -> str:
        return "apply_patch"

    @property
    def description(self) -> str:
        return (
            "Default tool for code edits, including multi-file changes. "
            "Returns a file-change summary. "
            "Use edit_file only for small exact replacements on a single file."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return object_schema(
            edits={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Path to the file to edit. Relative paths resolve against the "
                                "workspace; absolute paths and '..' obey the workspace access policy."
                            ),
                        },
                        "action": {
                            "type": "string",
                            "description": (
                                "replace updates a unique text match; add creates a file or appends to an existing one."
                            ),
                            "enum": ["replace", "add"],
                        },
                        "old_text": {
                            "type": ["string", "null"],
                            "description": "Exact text occurring once in the file; required for replace.",
                        },
                        "new_text": {
                            "type": ["string", "null"],
                            "description": "Text to replace with or append. Required for replace and add.",
                        },
                    },
                    "required": ["path", "action"],
                },
                "description": "File edits, applied in order.",
                "minItems": 1,
                "maxItems": 20,
            },
            dry_run={
                "type": "boolean",
                "description": "Validate and summarize the patch without writing files.",
                "default": False,
            },
            required=["edits"],
        )

    def _resolve_target(self, path: str) -> Path:
        source = self._resolve_write(path)
        if self._write_check is not None:
            problem = self._write_check(path, source)
            if problem:
                raise _GuardRefusal(problem)
        return source

    async def execute(self, edits: list[object] | None = None, dry_run: bool = False, **kwargs: Any) -> str:
        try:
            if not edits:
                raise _PatchError("must provide edits")

            writes: dict[Path, str] = {}
            originals: dict[Path, str] = {}
            actions: dict[Path, str] = {}

            for edit_value in edits:
                if not isinstance(edit_value, dict):
                    raise _PatchError("each edit must be an object")
                edit = cast(dict[str, Any], edit_value)
                raw_path = edit.get("path")
                if not isinstance(raw_path, str):
                    raise _PatchError("path required for edit")
                path = _validate_patch_path(raw_path)
                action = edit.get("action")
                if not isinstance(action, str):
                    raise _PatchError(f"action required for edit: {path}")
                source = self._resolve_target(path)

                if action == "add":
                    new_text = edit.get("new_text")
                    if new_text is None:
                        raise _PatchError(f"new_text required for add: {path}")
                    pending = writes.get(source)
                    if pending is not None:
                        content, exists = pending, True
                    elif source.exists():
                        content, exists = _read_utf8(source, path), True
                    else:
                        content, exists = "", False

                    if exists:
                        # Append; a CRLF file stays CRLF.
                        new_norm = _append_text(content, new_text)
                        if "\r\n" in content:
                            new_norm = new_norm.replace("\n", "\r\n")
                        action_name = "update"
                    else:
                        new_norm = new_text.replace("\r\n", "\n")
                        if new_norm and not new_norm.endswith("\n"):
                            new_norm += "\n"
                        action_name = "add"
                    writes[source] = new_norm

                elif action == "replace":
                    old_text = edit.get("old_text") or ""
                    if not old_text:
                        raise _PatchError(f"old_text required for replace: {path}")
                    new_text = edit.get("new_text")
                    if new_text is None:
                        raise _PatchError(f"new_text required for replace: {path}")

                    pending = writes.get(source)
                    if pending is not None:
                        content = pending
                    elif source.exists():
                        content = _read_utf8(source, path)
                    else:
                        raise _PatchError(f"file to update does not exist: {path}")
                    if pending is None and not source.is_file():
                        raise _PatchError(f"path to update is not a file: {path}")

                    uses_crlf = "\r\n" in content
                    norm_content = content.replace("\r\n", "\n")
                    norm_old = old_text.replace("\r\n", "\n")
                    pos = norm_content.find(norm_old)
                    if pos < 0:
                        raise _PatchError(f"old_text not found in {path}")
                    if norm_content.find(norm_old, pos + 1) >= 0:
                        raise _PatchError(f"old_text appears multiple times in {path}")

                    new_norm = norm_content[:pos] + new_text.replace("\r\n", "\n") + norm_content[pos + len(norm_old) :]
                    if new_norm and not new_norm.endswith("\n"):
                        new_norm += "\n"
                    if uses_crlf:
                        new_norm = new_norm.replace("\n", "\r\n")
                    writes[source] = new_norm
                    action_name = "update"

                else:
                    raise _PatchError(f"unknown action: {action}")

                originals.setdefault(source, content)
                actions.setdefault(source, action_name)

            summaries: list[str] = []
            for source, content in writes.items():
                # nanobot FileDiff.from_text compares CRLF-normalized lines.
                added, deleted = _count_changed_lines(
                    originals[source].replace("\r\n", "\n"), content.replace("\r\n", "\n")
                )
                stats = f" (+{added}/-{deleted})" if added or deleted else ""
                summaries.append(f"- {actions[source]} {self._display(source)}{stats}")

            if dry_run:
                return "Patch dry-run succeeded:\n" + "\n".join(summaries)

            backups = {p: (p.read_bytes() if p.exists() else None) for p in writes}
            try:
                for p, content in writes.items():
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(content, encoding="utf-8", newline="")
            except Exception:
                # All or nothing: put every touched file back.
                for p, data in backups.items():
                    if data is None:
                        if p.exists():
                            p.unlink()
                    else:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(data)
                raise

            for p in writes:
                self._file_states.record_write(p)
            return "Patch applied:\n" + "\n".join(summaries)
        except _GuardRefusal as exc:
            return ToolResult.error(str(exc))
        except PermissionError as exc:
            return ToolResult.error(f"Error: {exc}")
        except Exception as exc:
            return ToolResult.error(f"Error applying patch: {exc}")
