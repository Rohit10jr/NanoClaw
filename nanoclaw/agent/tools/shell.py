"""The exec tool: run a shell command. Mirrors nanobot/agent/tools/shell.py.

Deliberate difference from nanobot: the deny list and the internal-URL check
run on **every** command. nanobot runs its whole command guard only when
``restrict_to_workspace`` is on, which is off by default, so by default
nanobot blocks nothing (docs/notes/03-tools.md, 09-safety.md). The workspace
checks (``working_dir``, ``../``, absolute paths) apply only in restricted
mode, as in nanobot.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import signal
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from nanoclaw.agent.tools.base import Tool, ToolResult
from nanoclaw.agent.tools.schema import object_schema
from nanoclaw.security.network import contains_internal_url
from nanoclaw.security.workspace_policy import is_path_within

# nanobot shell.py ExecTool.__init__. Lower-cased commands are searched for
# these. A guard against accidents, not a security boundary:
# `rm --recursive --force`, `find -delete` or a Python one-liner get through. nanobot also blocks writes to its memory files;
# those patterns arrive with memory in layer 6.
DEFAULT_DENY_PATTERNS = [
    r"\brm\s+-[rf]{1,2}\b",  # rm -r, rm -rf, rm -fr
    r"\bdel\s+/[fq]\b",  # del /f, del /q
    r"\brmdir\s+/s\b",  # rmdir /s
    r"(?:^|[;&|]\s*)format(?!=)\b",  # format (as a standalone command)
    r"\b(mkfs|diskpart)\b",  # disk operations
    r"\bdd\s+if=",  # dd
    r">\s*/dev/sd",  # write to a disk device
    r"\b(shutdown|reboot|poweroff)\b",  # system power
    r":\(\)\s*\{.*\};\s*:",  # fork bomb
    # Writes to the memory journal and Dream cursor (layer 6) must go through
    # MemoryStore; direct writes corrupt the cursor format (nanobot #2989).
    r">>?\s*\S*(?:history\.jsonl|\.dream_cursor)",  # > / >> redirect
    r"\btee\b[^|;&<>]*(?:history\.jsonl|\.dream_cursor)",  # tee / tee -a
    r"\b(?:cp|mv)\b(?:\s+[^\s|;&<>]+)+\s+\S*(?:history\.jsonl|\.dream_cursor)",  # cp/mv target
    r"\bdd\b[^|;&<>]*\bof=\S*(?:history\.jsonl|\.dream_cursor)",  # dd of=
    r"\bsed\s+-i[^|;&<>]*(?:history\.jsonl|\.dream_cursor)",  # sed -i
]


# Policy note appended to recoverable workspace-boundary guard errors (nanobot shell.py:85).
_WORKSPACE_BOUNDARY_NOTE = (
    "\n\nNote: this is a hard policy boundary, not a transient failure. "
    "Do NOT retry with shell tricks (symlinks, base64 piping, alternative "
    "tools, working_dir overrides). If the user genuinely needs this "
    "resource, tell them you cannot reach it under the current "
    "restrict_to_workspace policy and ask how to proceed."
)


class ExecTool(Tool):
    _MAX_TIMEOUT = 600
    _MAX_OUTPUT = 10_000

    _BENIGN_DEVICE_PATHS: frozenset[str] = frozenset({
        "/dev/null",
        "/dev/zero",
        "/dev/full",
        "/dev/random",
        "/dev/urandom",
        "/dev/stdin",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/tty",
    })

    def __init__(
        self,
        working_dir: Path | None = None,
        timeout: int = 60,
        deny_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
    ) -> None:
        self.working_dir = working_dir
        self.timeout = timeout
        self.deny_patterns = (deny_patterns or []) + DEFAULT_DENY_PATTERNS
        self.restrict_to_workspace = restrict_to_workspace

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command."

    @property
    def parameters(self) -> dict[str, Any]:
        return object_schema(
            command={"type": "string", "description": "The shell command to execute"},
            working_dir={"type": "string", "description": "Optional working directory for the command"},
            timeout={
                "type": "integer",
                "description": "Hard timeout in seconds (default 60, max 600).",
                "minimum": 1,
                "maximum": 600,
            },
            required=["command"],
        )

    async def execute(
        self,
        command: str | None = None,
        working_dir: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> str:
        if not command:
            return ToolResult.error("Error: Missing command. Provide command.")
        cwd = self._resolve_cwd(working_dir)
        # Restricted mode: an LLM-supplied working_dir must not escape the
        # workspace, or absolute paths under it would pass the path check
        # below, which is anchored on cwd (nanobot shell.py:424, #2826).
        if self.restrict_to_workspace and self.working_dir:
            try:
                requested = Path(cwd).expanduser().resolve()
                resolved_root = Path(self.working_dir).expanduser().resolve()
            except Exception:
                return ToolResult.error("Error: working_dir could not be resolved" + _WORKSPACE_BOUNDARY_NOTE)
            if not is_path_within(requested, resolved_root):
                return ToolResult.error("Error: working_dir is outside the configured workspace" + _WORKSPACE_BOUNDARY_NOTE)
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        effective_timeout = self._resolve_timeout(timeout)
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                shutil.which("bash") or "/bin/bash",
                "-c",
                command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=self._build_env(),
                # Own process group, so a timeout can kill the command's children too.
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=effective_timeout)
            except asyncio.TimeoutError:
                await self._kill_process_tree(process)
                return ToolResult.error(f"Error: Command timed out after {effective_timeout} seconds")
            except asyncio.CancelledError:
                await self._kill_process_tree(process)
                raise

            parts: list[str] = []
            if stdout:
                parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    parts.append(f"STDERR:\n{stderr_text}")
            # A non-zero exit code is reported, not treated as a tool error.
            parts.append(f"\nExit code: {process.returncode}")
            result = "\n".join(parts)

            if len(result) > self._MAX_OUTPUT:
                half = self._MAX_OUTPUT // 2
                result = (
                    result[:half]
                    + f"\n\n... ({len(result) - self._MAX_OUTPUT:,} chars truncated) ...\n\n"
                    + result[-half:]
                )
            return result
        except Exception as e:
            if process is not None and process.returncode is None:
                await self._kill_process_tree(process)
            return ToolResult.error(f"Error executing command: {e}")

    def _resolve_timeout(self, timeout: int | None) -> int | None:
        """Effective timeout in seconds; ``None`` means no limit (nanobot: _resolve_timeout).

        A timeout chosen by the model is capped at 600 s. The configured default
        (``self.timeout``) may be higher, and 0 disables the limit entirely.
        """
        if timeout:
            return min(timeout, self._MAX_TIMEOUT)
        if self.timeout and self.timeout > 0:
            return self.timeout
        return None

    def _guard_command(self, command: str, cwd: str) -> ToolResult | None:
        """Best-effort safety guard (nanobot ``_guard_command``, shell.py:840).

        The deny list and the internal-URL check always run; nanobot runs them
        only in restricted mode. The path checks run only in restricted mode.
        """
        cmd = command.strip()
        lower = cmd.lower()
        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return ToolResult.error("Error: Command blocked by deny pattern filter")

        if contains_internal_url(cmd):
            return ToolResult.error("Error: Command blocked by safety guard (internal/private URL detected)")

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return ToolResult.error(
                    "Error: Command blocked by safety guard (path traversal detected)" + _WORKSPACE_BOUNDARY_NOTE
                )
            cwd_path = Path(cwd).resolve()
            resolved_workspace = Path(self.working_dir).expanduser().resolve() if self.working_dir else None
            for raw in self._extract_absolute_paths(cmd):
                try:
                    expanded = os.path.expandvars(raw.strip())
                    # Python's expanduser() doesn't implement the shell's
                    # directory-stack forms: ``~+`` is the cwd; ``~-`` and
                    # indexed forms can point anywhere, so they fail closed.
                    if expanded == "~+":
                        p = cwd_path
                    elif expanded.startswith("~+/"):
                        p = (cwd_path / expanded[3:]).resolve()
                    elif re.match(r"^~(?:-|[+-]\d+)(?:/|$)", expanded):
                        return ToolResult.error(
                            "Error: Command blocked by safety guard (path outside working dir)"
                            + _WORKSPACE_BOUNDARY_NOTE
                        )
                    else:
                        p = Path(expanded).expanduser().resolve()
                    # Check the unresolved path first: /dev/stderr is a symlink
                    # to /proc/self/fd/2, which would hide the device.
                    if self._is_benign_device_path(expanded):
                        continue
                except Exception:
                    continue
                if self._is_benign_device_path(str(p)):
                    continue
                allowed = is_path_within(p, cwd_path)
                if not allowed and resolved_workspace is not None:
                    allowed = is_path_within(p, resolved_workspace)
                if p.is_absolute() and not allowed:
                    return ToolResult.error(
                        "Error: Command blocked by safety guard (path outside working dir)" + _WORKSPACE_BOUNDARY_NOTE
                    )
        return None

    @classmethod
    def _is_benign_device_path(cls, path: str) -> bool:
        """Return True for kernel device files that should never be workspace-blocked."""
        if path in cls._BENIGN_DEVICE_PATHS:
            return True
        return path.startswith("/dev/fd/")

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        # Windows: match drive-root paths like `C:\` as well as `C:\path\to\file`, and UNC paths like `\\server\share`
        # NOTE: `*` is required so `C:\` (nothing after the slash) is still extracted.
        win_paths = re.findall(
            r"(?<![A-Za-z])(?:[A-Za-z]:[^\s\"'|><;]*|\\\\[^\s\"'|><;]+(?:\\[^\s\"'|><;]+)*)",
            command
        )
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars="();<>|&")
            lexer.whitespace_split = True
            lexer.commenters = ""
            tokens = list(lexer)
        except ValueError:
            # Keep malformed quoting fail-closed. The shell will normally reject
            # it too, but a conservative raw scan must not turn it into a bypass.
            tokens = [command]

        paths = [*win_paths]
        seen = set(win_paths)
        for index, token in enumerate(tokens):
            for path in ExecTool._extract_posix_paths_from_token(token):
                if path not in seen:
                    paths.append(path)
                    seen.add(path)
            if index > 0 and tokens[index - 1] in {"-c", "-lc", "--command"}:
                for path in ExecTool._extract_absolute_paths(token):
                    if path not in seen:
                        paths.append(path)
                        seen.add(path)
        return paths

    @staticmethod
    def _extract_posix_paths_from_token(token: str) -> list[str]:
        """Extract local POSIX/home paths from one shell-decoded token.

        ``shlex`` separates real grouping/redirection operators while preserving
        parentheses and spaces that were quoted or escaped as part of a path.
        Embedded scripts (for example ``sh -c \"cat /tmp/x\"``) still need a
        small boundary scan. Colons are not general boundaries: treating them
        as such misclassifies URLs, ``host:/remote`` and ``C:/Windows``. They
        are considered only inside a syntactically valid assignment, where
        shells expand each colon-delimited tilde component.
        """
        paths: list[str] = []
        for match in re.finditer(
            r"file://(?:[^/\s\"']+)?(/[^\s\"'<>|;&]*)",
            token,
            flags=re.IGNORECASE,
        ):
            uri_prefix = token[: match.start()]
            raw_path = match.group(1)
            if uri_prefix.count("(") > uri_prefix.count(")"):
                raw_path = raw_path.split(")", 1)[0]
            if uri_prefix.count("{") > uri_prefix.count("}"):
                raw_path = raw_path.split(",", 1)[0].split("}", 1)[0]
            raw_path = raw_path.split("?", 1)[0].split("#", 1)[0]
            if raw_path:
                paths.append(unquote(raw_path))
        boundary_chars = frozenset(" \t\r\n=({,<>|;&\"'")
        i = 0
        while i < len(token):
            is_posix = token[i] == "/"
            home_match = re.match(
                r"~(?:[+-](?:\d+)?|[A-Za-z0-9_.@-]+)?(?=/|:|$)",
                token[i:],
            )
            is_home = home_match is not None
            if not is_posix and not is_home:
                i += 1
                continue

            prefix = token[:i]
            parameter_default = (
                i >= 2 and token[i - 2] == ":" and token[i - 1] in "-+?="
            )
            word_start = max(
                (prefix.rfind(char) for char in " \t\r\n<>|;&"),
                default=-1,
            ) + 1
            word_prefix = prefix[word_start:]
            assignment_component = bool(
                re.fullmatch(
                    r"(?:[A-Za-z_][A-Za-z0-9_]*|--?[A-Za-z0-9_.-]+)="
                    r"(?:[^:=\s]*:)*",
                    word_prefix,
                )
            )
            at_boundary = i == 0 or token[i - 1] in boundary_chars
            if is_home:
                # A shell word beginning with ``~`` is a separate shlex token.
                # Mid-token expansion is valid only after ``=`` or a colon in
                # an assignment. This avoids PromQL/Loki ``=~`` and ``|~``
                # match operators while covering PATH-like values.
                at_boundary = i == 0 or assignment_component
            if not at_boundary and not parameter_default:
                i += 1
                continue

            if re.search(r"[A-Za-z][A-Za-z0-9+.-]*://", word_prefix) or re.match(
                r"(?:[^/:=\s]+@)?[^/:=\s]+:$",
                word_prefix,
            ):
                # HTTP-style URL path/query fragments and scp-style remote paths
                # are not local filesystem references. ``file://`` paths were
                # decoded above. Windows drive paths are already captured by the
                # platform-specific expression above.
                i += 1
                continue

            assignment_value = assignment_component
            if i == 0 or assignment_value:
                end = len(token)
                if assignment_value:
                    separator = token.find(":", i)
                    if separator >= 0:
                        end = separator
            elif token[i - 1] in {"'", '"'}:
                quote = token[i - 1]
                closing = token.find(quote, i)
                end = len(token) if closing < 0 else closing
            else:
                end_chars = set(" \t\r\n\"'<>|;&")
                if prefix.count("(") > prefix.count(")"):
                    end_chars.add(")")
                if prefix.count("{") > prefix.count("}"):
                    end_chars.update({",", "}"})
                end = i
                while end < len(token) and token[end] not in end_chars:
                    end += 1

            candidate = token[i:end]
            if candidate:
                paths.append(candidate)
            i = max(end, i + 1)
        return paths

    def _resolve_cwd(self, working_dir: str | None) -> str:
        root = str(self.working_dir) if self.working_dir else os.getcwd()
        if not working_dir:
            return root
        requested = Path(working_dir).expanduser()
        return str(requested if requested.is_absolute() else Path(root) / requested)

    @staticmethod
    def _build_env() -> dict[str, str]:
        """A minimal environment, as in nanobot (Unix).

        Only these variables are passed, so API keys and other secrets in the
        agent's own environment are never visible to commands the model runs.
        bash supplies a default PATH.
        """
        return {
            "HOME": os.environ.get("HOME", "/tmp"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "TERM": os.environ.get("TERM", "dumb"),
            "PYTHONUNBUFFERED": "1",
        }

    @staticmethod
    async def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
