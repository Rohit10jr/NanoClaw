"""File tools: read_file, write_file, edit_file, list_dir.

Mirrors nanobot/agent/tools/filesystem.py. Relative paths resolve against the
workspace. With ``restrict_to_workspace`` (off by default, as in nanobot) every
path must stay inside the workspace; reads may also use the skills folders and
the history journal (layer 9).
"""

from __future__ import annotations

import codecs
import difflib
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoclaw.agent.tools.base import Tool, ToolResult
from nanoclaw.agent.tools.schema import object_schema
from nanoclaw.security.workspace_policy import resolve_allowed_path


class _FsTool(Tool):
    """Shared base: workspace-relative path resolution and the workspace boundary.

    Mirrors nanobot ``_FsTool`` (``create``, ``_resolve_read``,
    ``_resolve_write``) and ``path_utils.resolve_workspace_path``. A path
    outside the boundary raises ``WorkspaceBoundaryError`` (a
    ``PermissionError``), which each tool returns as an error.
    """

    def __init__(self, workspace: Path | None = None, *, restrict_to_workspace: bool = False) -> None:
        self._workspace = workspace
        self._allowed_dir = workspace if restrict_to_workspace and workspace is not None else None
        if workspace is not None:
            from nanoclaw.agent.skills import BUILTIN_SKILLS_DIR

            resolved = workspace.expanduser().resolve(strict=False)
            # Agent-owned skills stay readable; of memory/, only the journal.
            self._extra_read_allowed_dirs = [BUILTIN_SKILLS_DIR, resolved / "skills"]
            self._extra_read_allowed_files = [resolved / "memory" / "history.jsonl"]
        else:
            self._extra_read_allowed_dirs, self._extra_read_allowed_files = [], []

    def _resolve_read(self, path: str) -> Path:
        restricted = self._allowed_dir is not None
        return resolve_allowed_path(
            path,
            workspace=self._workspace,
            allowed_root=self._allowed_dir,
            extra_allowed_roots=self._extra_read_allowed_dirs if restricted else None,
            # The journal exception only matters when restricted.
            extra_allowed_files=self._extra_read_allowed_files if restricted else None,
        )

    def _resolve_write(self, path: str) -> Path:
        return resolve_allowed_path(path, workspace=self._workspace, allowed_root=self._allowed_dir)

    def _resolve(self, path: str) -> Path:
        return self._resolve_read(path)

    def _display(self, p: Path) -> str:
        """Path relative to the workspace when inside it (nanobot: display_file_edit_path)."""
        if self._workspace is not None:
            try:
                return p.resolve().relative_to(self._workspace.resolve()).as_posix()
            except ValueError:
                pass
        return p.as_posix()


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

_BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/stdout", "/dev/stderr",
    "/dev/tty", "/dev/console",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})


def _builtin_skill_read_path(path: str) -> Path | None:
    """Map a workspace-relative ``skills/<name>/...`` read onto the built-in skills.

    The skills list in the system prompt shows built-in skills under
    ``skills/`` too; a file missing from the workspace is looked up there.
    (nanobot ``_builtin_skill_read_path``)
    """
    from nanoclaw.agent.skills import BUILTIN_SKILLS_DIR

    requested = Path(path)
    if requested.is_absolute():
        return None
    parts = requested.parts
    if len(parts) < 2 or parts[0] != "skills":
        return None
    root = BUILTIN_SKILLS_DIR.resolve()
    candidate = (root / Path(*parts[1:])).resolve()
    if candidate != root and root not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


def _is_blocked_device(path: str | Path) -> bool:
    """Device files that could hang or produce endless output."""
    raw = str(path)
    try:
        resolved = str(Path(raw).resolve())
    except (OSError, ValueError):
        resolved = raw
    if raw in _BLOCKED_DEVICE_PATHS or resolved in _BLOCKED_DEVICE_PATHS:
        return True
    for candidate in (raw, resolved):
        if re.match(r"/proc/(\d+|self)/fd/[012]$", candidate):
            return True
    return resolved.startswith("/dev/")


# Extensions read as Latin-1 when they aren't valid UTF-8 (nanobot utils/document.py).
_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm",
    ".log", ".yaml", ".yml", ".toml", ".ini", ".cfg",
})


def _decode_bom_text(raw: bytes) -> str | None:
    """Decode text whose byte-order mark names its encoding (UTF-8/16/32)."""
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig")
    # UTF-32 LE starts with the UTF-16 LE BOM, so check UTF-32 first.
    if raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return raw.decode("utf-32")
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16")
    return None


class ReadFileTool(_FsTool):
    _MAX_CHARS = 128_000
    _MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
    _DEFAULT_LIMIT = 2000

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read a text file by path. "
            "Text is line-numbered; use offset/limit for targeted ranges."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return object_schema(
            path={"type": "string", "description": "The file path to read"},
            offset={"type": "integer", "description": "1-based line to start from (default 1)", "minimum": 1},
            limit={"type": "integer", "description": "Maximum lines to return (default 2000)", "minimum": 1},
            required=["path"],
        )

    async def execute(self, path: str | None = None, offset: int = 1, limit: int | None = None, **kwargs: Any) -> str:
        try:
            if not path:
                return ToolResult.error("Error reading file: Unknown path")
            if _is_blocked_device(path):
                return ToolResult.error(
                    f"Error: Reading {path} is blocked (device path that could hang or produce infinite output)."
                )
            fp = self._resolve_read(path)
            if not fp.exists():
                fp = _builtin_skill_read_path(path) or fp
            if _is_blocked_device(fp):
                return ToolResult.error(
                    f"Error: Reading {fp} is blocked (device path that could hang or produce infinite output)."
                )
            if not fp.exists():
                return ToolResult.error(f"Error: File not found: {path}")
            if not fp.is_file():
                return ToolResult.error(f"Error: Not a file: {path}")

            file_size = fp.stat().st_size
            if file_size > self._MAX_FILE_SIZE_BYTES:
                return ToolResult.error(
                    f"Error: File too large to read ({file_size / (1024 * 1024):.1f} MiB). "
                    f"Maximum is {self._MAX_FILE_SIZE_BYTES // (1024 * 1024)} MiB."
                )

            raw = fp.read_bytes()
            if not raw:
                return f"(Empty file: {path})"
            text = _decode_bom_text(raw)
            if text is None:
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    if fp.suffix.lower() not in _TEXT_EXTENSIONS:
                        mime = mimetypes.guess_type(path)[0]
                        return ToolResult.error(
                            f"Error: Cannot read binary file {path} (MIME: {mime or 'unknown'}). "
                            "Only supported text files can be read."
                        )
                    text = raw.decode("latin-1")
            if not text:
                return f"(Empty file: {path})"

            all_lines = text.replace("\r\n", "\n").splitlines()
            total = len(all_lines)
            offset = max(offset, 1)
            if offset > total:
                return ToolResult.error(f"Error: offset {offset} is beyond end of file ({total} lines)")

            start = offset - 1
            end = min(start + (limit or self._DEFAULT_LIMIT), total)
            numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(all_lines[start:end])]
            result = "\n".join(numbered)

            line_truncated = False
            if len(result) > self._MAX_CHARS:
                trimmed: list[str] = []
                chars = 0
                for line in numbered:
                    extra = len(line) + (1 if trimmed else 0)
                    if chars + extra > self._MAX_CHARS:
                        if not trimmed:
                            trimmed.append(line[: self._MAX_CHARS])
                            line_truncated = True
                        break
                    trimmed.append(line)
                    chars += extra
                end = start + len(trimmed)
                result = "\n".join(trimmed)

            if line_truncated:
                result += (
                    f"\n\n(Line {offset} truncated; its remaining characters are not shown. "
                    "Use exec with a targeted command to inspect the omitted content.)"
                )
            if end < total:
                result += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
            else:
                result += f"\n\n(End of file — {total} lines total)"
            return result
        except PermissionError as e:
            return ToolResult.error(f"Error: {e}")
        except Exception as e:
            return ToolResult.error(f"Error reading file: {e}")


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class WriteFileTool(_FsTool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Create a new file or intentionally replace an entire file with "
            "the provided content. Overwrites existing files and creates parent "
            "directories as needed. For partial edits, prefer edit_file."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return object_schema(
            path={"type": "string", "description": "The file path to write to"},
            content={"type": "string", "description": "The content to write"},
            required=["path", "content"],
        )

    async def execute(self, path: str | None = None, content: str | None = None, **kwargs: Any) -> str:
        try:
            if not path:
                raise ValueError("Unknown path")
            if content is None:
                raise ValueError("Unknown content")
            fp = self._resolve_write(path)
            fp.parent.mkdir(parents=True, exist_ok=True)
            # newline="" writes the text as given: no "\n" -> "\r\n" on
            # Windows (nanobot ae6801d, filesystem.py:580).
            fp.write_text(content, encoding="utf-8", newline="")
            return f"Successfully wrote {len(content)} characters to {fp}"
        except PermissionError as e:
            return ToolResult.error(f"Error: {e}")
        except Exception as e:
            return ToolResult.error(f"Error writing file: {e}")


# ---------------------------------------------------------------------------
# edit_file: matching helpers (ported from nanobot)
# ---------------------------------------------------------------------------

_QUOTE_TABLE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _normalize_quotes(s: str) -> str:
    return s.translate(_QUOTE_TABLE)


def _curly_double_quotes(text: str) -> str:
    parts: list[str] = []
    opening = True
    for ch in text:
        if ch == '"':
            parts.append("\u201c" if opening else "\u201d")
            opening = not opening
        else:
            parts.append(ch)
    return "".join(parts)


def _curly_single_quotes(text: str) -> str:
    parts: list[str] = []
    opening = True
    for i, ch in enumerate(text):
        if ch != "'":
            parts.append(ch)
            continue
        prev_ch = text[i - 1] if i > 0 else ""
        next_ch = text[i + 1] if i + 1 < len(text) else ""
        if prev_ch.isalnum() and next_ch.isalnum():  # apostrophe, as in "don't"
            parts.append("\u2019")
            continue
        parts.append("\u2018" if opening else "\u2019")
        opening = not opening
    return "".join(parts)


def _preserve_quote_style(old_text: str, actual_text: str, new_text: str) -> str:
    """If the match only succeeded by treating curly quotes as straight, write curly quotes back."""
    if _normalize_quotes(old_text.strip()) != _normalize_quotes(actual_text.strip()) or old_text == actual_text:
        return new_text
    styled = new_text
    if any(ch in actual_text for ch in ("\u201c", "\u201d")) and '"' in styled:
        styled = _curly_double_quotes(styled)
    if any(ch in actual_text for ch in ("\u2018", "\u2019")) and "'" in styled:
        styled = _curly_single_quotes(styled)
    return styled


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _reindent_like_match(old_text: str, actual_text: str, new_text: str) -> str:
    """If the match only succeeded by ignoring indentation, shift ``new_text`` to the file's indentation.

    Example: the file has ``    x = 1`` inside a function, the model sends
    ``old_text="x = 1"``-style lines without the indent and ``new_text`` without it
    too. Without this, the replacement would land unindented and break the code.
    """
    # A terminal newline does not add a logical line, even at an unterminated
    # EOF (nanobot 042bcbb, filesystem.py:652).
    old_lines = old_text.removesuffix("\n").split("\n")
    actual_lines = actual_text.removesuffix("\n").split("\n")
    if len(old_lines) != len(actual_lines):
        return new_text
    comparable = [
        (old_line, actual_line)
        for old_line, actual_line in zip(old_lines, actual_lines)
        if old_line.strip() and actual_line.strip()
    ]
    if not comparable or any(
        _normalize_quotes(old_line.strip()) != _normalize_quotes(actual_line.strip())
        for old_line, actual_line in comparable
    ):
        return new_text

    old_ws = _leading_ws(comparable[0][0])
    actual_ws = _leading_ws(comparable[0][1])
    if actual_ws == old_ws:
        return new_text
    if old_ws:
        if not actual_ws.startswith(old_ws):
            return new_text
        delta = actual_ws[len(old_ws):]
    else:
        delta = actual_ws
    if not delta:
        return new_text
    return "\n".join((delta + line) if line else line for line in new_text.split("\n"))


def _strip_trailing_ws(text: str, *, preserve_last_line: bool = False) -> str:
    """Strip line-ending whitespace, except a final fragment that continues inline.

    nanobot ``EditFileTool._strip_trailing_ws`` (filesystem.py:900).
    """
    lines = text.split("\n")
    return "\n".join(
        line if preserve_last_line and i == len(lines) - 1 else line.rstrip() for i, line in enumerate(lines)
    )


@dataclass(slots=True)
class _MatchSpan:
    start: int
    end: int
    text: str
    line: int


def _find_exact_matches(content: str, old_text: str) -> list[_MatchSpan]:
    matches: list[_MatchSpan] = []
    start = 0
    while (idx := content.find(old_text, start)) != -1:
        matches.append(_MatchSpan(idx, idx + len(old_text), content[idx : idx + len(old_text)], content.count("\n", 0, idx) + 1))
        start = idx + max(1, len(old_text))
    return matches


def _find_trim_matches(content: str, old_text: str, *, normalize_quotes: bool = False) -> list[_MatchSpan]:
    """Match line by line, ignoring each line's leading/trailing whitespace."""
    old_lines = old_text.splitlines()
    if not old_lines:
        return []
    content_lines = content.splitlines()
    keepends = content.splitlines(keepends=True)
    if len(content_lines) < len(old_lines):
        return []
    offsets = [0]
    for line in keepends:
        offsets.append(offsets[-1] + len(line))

    def norm(line: str) -> str:
        return _normalize_quotes(line.strip()) if normalize_quotes else line.strip()

    wanted = [norm(line) for line in old_lines]
    size = len(wanted)
    matches: list[_MatchSpan] = []
    for i in range(len(content_lines) - size + 1):
        if [norm(line) for line in content_lines[i : i + size]] != wanted:
            continue
        start, end = offsets[i], offsets[i + size]
        # Include the line terminator only when the requested match includes
        # it (nanobot 042bcbb, filesystem.py:758).
        if not old_text.endswith("\n") and keepends[i + size - 1].endswith("\n"):
            end -= 1
        matches.append(_MatchSpan(start, end, content[start:end], i + 1))
    return matches


def _find_quote_matches(content: str, old_text: str) -> list[_MatchSpan]:
    """Exact match after treating curly and straight quotes as equal."""
    norm_content, norm_old = _normalize_quotes(content), _normalize_quotes(old_text)
    matches: list[_MatchSpan] = []
    start = 0
    while (idx := norm_content.find(norm_old, start)) != -1:
        matches.append(_MatchSpan(idx, idx + len(old_text), content[idx : idx + len(old_text)], content.count("\n", 0, idx) + 1))
        start = idx + max(1, len(norm_old))
    return matches


def _find_matches(content: str, old_text: str) -> list[_MatchSpan]:
    """Try progressively looser strategies; return the first that finds anything."""
    for matcher in (
        lambda: _find_exact_matches(content, old_text),
        lambda: _find_trim_matches(content, old_text),
        lambda: _find_trim_matches(content, old_text, normalize_quotes=True),
        lambda: _find_quote_matches(content, old_text),
    ):
        if matches := matcher():
            return matches
    return []


def _collapse_internal_whitespace(text: str) -> str:
    return "\n".join(" ".join(line.split()) for line in text.splitlines())


def _diagnose_near_match(old_text: str, actual_text: str) -> list[str]:
    hints: list[str] = []
    if old_text == actual_text:
        return hints
    if old_text.lower() == actual_text.lower():
        hints.append("letter case differs")
    if _collapse_internal_whitespace(old_text) == _collapse_internal_whitespace(actual_text):
        hints.append("whitespace differs")
    if old_text.rstrip("\n") == actual_text.rstrip("\n"):
        hints.append("trailing newline differs")
    if _normalize_quotes(old_text) == _normalize_quotes(actual_text):
        hints.append("quote style differs")
    return hints


def _best_window(old_text: str, content: str) -> tuple[float, int, list[str], list[str]]:
    """Closest window of lines to ``old_text``: (ratio, start line index, lines, hints)."""
    lines = content.splitlines(keepends=True)
    old_lines = old_text.splitlines(keepends=True)
    window = max(1, len(old_lines))
    best_ratio, best_start, best_lines = -1.0, 0, []
    for i in range(max(1, len(lines) - window + 1)):
        current = lines[i : i + window]
        ratio = difflib.SequenceMatcher(None, old_lines, current).ratio()
        if ratio > best_ratio:
            best_ratio, best_start, best_lines = ratio, i, current
    actual = "".join(best_lines).replace("\r\n", "\n").rstrip("\n")
    hints = _diagnose_near_match(old_text.replace("\r\n", "\n").rstrip("\n"), actual)
    return best_ratio, best_start, best_lines, hints


def _count_changed_lines(before: str, after: str) -> tuple[int, int]:
    """(added, deleted) line counts, for the edit summary."""
    added = deleted = 0
    matcher = difflib.SequenceMatcher(None, before.splitlines(), after.splitlines(), autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            deleted += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, deleted


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


class EditFileTool(_FsTool):
    _MAX_EDIT_FILE_SIZE = 1024 * 1024 * 1024  # 1 GiB
    _MARKDOWN_EXTS = frozenset({".md", ".mdx", ".markdown"})

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Perform a small, exact replacement in one file. "
            "Copy old_text from read_file. occurrence and replace_all=true are mutually exclusive. "
            "With old_text empty, creates a new file."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return object_schema(
            path={"type": "string", "description": "The file path to edit"},
            old_text={"type": "string", "description": "The text to find and replace; copy it from read_file."},
            new_text={
                "type": "string",
                "description": "The replacement text; must differ from old_text for an existing file.",
            },
            replace_all={"type": "boolean", "description": "Replace all occurrences (default false)"},
            occurrence={
                "type": ["integer", "null"],
                "description": "Optional 1-based occurrence to replace when old_text appears multiple times.",
                "minimum": 1,
            },
            required=["path", "old_text", "new_text"],
        )

    def _summary(self, fp: Path, before: str, after: str, *, created: bool = False) -> str:
        added, deleted = _count_changed_lines(before, after)
        stats = f" (+{added}/-{deleted})" if added or deleted else ""
        return f"Patch applied:\n- {'add' if created else 'update'} {self._display(fp)}{stats}"

    async def execute(
        self,
        path: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        replace_all: bool = False,
        occurrence: int | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            if not path:
                raise ValueError("Unknown path")
            if old_text is None:
                raise ValueError("Unknown old_text")
            if new_text is None:
                raise ValueError("Unknown new_text")
            fp = self._resolve_write(path)
            file_exists = fp.exists()
            if file_exists and old_text == new_text:
                return ToolResult.error("Error: new_text must be different from old_text.")

            # old_text="" means "create this file".
            if not file_exists:
                if old_text == "":
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(new_text, encoding="utf-8", newline="")  # nanobot ae6801d
                    return self._summary(fp, "", new_text, created=True)
                return self._file_not_found_msg(path, fp)

            fsize = fp.stat().st_size
            if fsize > self._MAX_EDIT_FILE_SIZE:
                return ToolResult.error(f"Error: File too large to edit ({fsize / (1024**3):.1f} GiB). Maximum is 1 GiB.")

            raw = fp.read_bytes()
            if old_text == "":
                content = raw.decode("utf-8")
                if content.strip():
                    return ToolResult.error(f"Error: Cannot create file — {path} already exists and is not empty.")
                fp.write_text(new_text, encoding="utf-8", newline="")  # nanobot ae6801d
                return self._summary(fp, content, new_text)

            uses_crlf = b"\r\n" in raw
            content = raw.decode("utf-8").replace("\r\n", "\n")
            norm_old = old_text.replace("\r\n", "\n")
            matches = _find_matches(content, norm_old)
            if not matches:
                return self._not_found_msg(old_text, content, path)

            count = len(matches)
            if replace_all and occurrence is not None:
                return ToolResult.error("Error: occurrence cannot be used with replace_all=true.")
            if occurrence is not None and occurrence > count:
                return ToolResult.error(
                    f"Error: occurrence {occurrence} is out of range; old_text appears {count} time(s)."
                )
            if count > 1 and not replace_all and occurrence is None:
                lines = [m.line for m in matches]
                preview = ", ".join(f"line {n}" for n in lines[:3]) + (", ..." if len(lines) > 3 else "")
                # A warning, not an error, exactly as in nanobot: nothing is changed.
                return (
                    f"Warning: old_text appears {count} times at {preview}. "
                    "Provide more context, set occurrence to choose one match, "
                    "or set replace_all=true."
                )

            norm_new = new_text.replace("\r\n", "\n")

            if replace_all:
                selected = matches
            elif occurrence is not None:
                selected = [matches[occurrence - 1]]
            else:
                selected = [matches[0]]

            new_content = content
            for match in reversed(selected):
                replacement = norm_new
                # Strip trailing whitespace from the replacement's line ends, per
                # match, keeping the last line's when text follows on the same
                # line (nanobot e36a016, filesystem.py:1036). Markdown keeps all
                # trailing whitespace: two spaces there mean a line break.
                if fp.suffix.lower() not in self._MARKDOWN_EXTS:
                    line_end = content.find("\n", match.end)
                    if line_end == -1:
                        line_end = len(content)
                    replacement = _strip_trailing_ws(
                        replacement, preserve_last_line=bool(content[match.end : line_end].strip())
                    )
                replacement = _preserve_quote_style(norm_old, match.text, replacement)
                replacement = _reindent_like_match(norm_old, match.text, replacement)
                end = match.end
                # Deleting whole lines also removes their line break.
                if (
                    replacement == ""
                    and (match.start == 0 or content[match.start - 1] == "\n")
                    and not match.text.endswith("\n")
                    and content[end : end + 1] == "\n"
                ):
                    end += 1
                new_content = new_content[: match.start] + replacement + new_content[end:]

            out = new_content.replace("\n", "\r\n") if uses_crlf else new_content
            fp.write_bytes(out.encode("utf-8"))
            return self._summary(fp, content, new_content)
        except PermissionError as e:
            return ToolResult.error(f"Error: {e}")
        except Exception as e:
            return ToolResult.error(f"Error editing file: {e}")

    @staticmethod
    def _file_not_found_msg(path: str, fp: Path) -> ToolResult:
        parts = [f"Error: File not found: {path}"]
        if fp.parent.is_dir():
            siblings = [f.name for f in fp.parent.iterdir() if f.is_file()]
            close = difflib.get_close_matches(fp.name, siblings, n=3, cutoff=0.6)
            if close:
                parts.append("Did you mean: " + ", ".join(str(fp.parent / c) for c in close) + "?")
        return ToolResult.error("\n".join(parts))

    @staticmethod
    def _not_found_msg(old_text: str, content: str, path: str) -> ToolResult:
        ratio, start, window_lines, hints = _best_window(old_text, content)
        if ratio > 0.5:
            diff = "\n".join(
                difflib.unified_diff(
                    old_text.splitlines(keepends=True),
                    window_lines,
                    fromfile="old_text (provided)",
                    tofile=f"{path} (actual, line {start + 1})",
                    lineterm="",
                )
            )
            hint_text = "\nPossible cause: " + ", ".join(hints) + "." if hints else ""
            return ToolResult.error(
                f"Error: old_text not found in {path}.{hint_text}\n"
                f"Best match ({ratio:.0%} similar) at line {start + 1}:\n{diff}"
            )
        if hints:
            return ToolResult.error(
                f"Error: old_text not found in {path}. Possible cause: {', '.join(hints)}. "
                "Copy the exact text from read_file and try again."
            )
        return ToolResult.error(f"Error: old_text not found in {path}. No similar text found. Verify the file content.")


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------


class ListDirTool(_FsTool):
    _DEFAULT_MAX = 200
    _IGNORE_DIRS = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", ".coverage", "htmlcov",
    }

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return (
            "List the contents of a directory. "
            "Set recursive=true to explore nested structure. "
            "Common noise directories (.git, node_modules, __pycache__, etc.) are auto-ignored."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return object_schema(
            path={"type": "string", "description": "The directory path to list"},
            recursive={"type": "boolean", "description": "Recursively list all files (default false)"},
            max_entries={"type": "integer", "description": "Maximum entries to return (default 200)", "minimum": 1},
            required=["path"],
        )

    async def execute(
        self, path: str | None = None, recursive: bool = False, max_entries: int | None = None, **kwargs: Any
    ) -> str:
        try:
            if path is None:
                raise ValueError("Unknown path")
            dp = self._resolve(path)
            if not dp.exists():
                return ToolResult.error(f"Error: Directory not found: {path}")
            if not dp.is_dir():
                return ToolResult.error(f"Error: Not a directory: {path}")

            cap = max_entries or self._DEFAULT_MAX
            items: list[str] = []
            total = 0
            if recursive:
                for item in sorted(dp.rglob("*")):
                    if any(part in self._IGNORE_DIRS for part in item.relative_to(dp).parts):
                        continue
                    total += 1
                    if len(items) < cap:
                        rel = item.relative_to(dp)
                        items.append(f"{rel}/" if item.is_dir() else str(rel))
            else:
                for item in sorted(dp.iterdir()):
                    if item.name in self._IGNORE_DIRS:
                        continue
                    total += 1
                    if len(items) < cap:
                        items.append(f"{'📁 ' if item.is_dir() else '📄 '}{item.name}")

            if total == 0:
                return f"Directory {path} is empty"
            result = "\n".join(items)
            if total > cap:
                result += f"\n\n(truncated, showing first {cap} of {total} entries)"
            return result
        except PermissionError as e:
            return ToolResult.error(f"Error: {e}")
        except Exception as e:
            return ToolResult.error(f"Error listing directory: {e}")
