"""The exec tool: run a shell command. Mirrors nanobot/agent/tools/shell.py.

Deliberate difference from nanobot: the deny list is checked on **every**
command. nanobot checks it only when ``restrict_to_workspace`` is on, which is
off by default, so by default nanobot blocks nothing. We keep the safety net
until layer 9 adds workspace restriction (see docs/notes/03-tools.md).
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
from pathlib import Path
from typing import Any

from nanoclaw.agent.tools.base import Tool, ToolResult
from nanoclaw.agent.tools.schema import object_schema

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
]


class ExecTool(Tool):
    _MAX_TIMEOUT = 600
    _MAX_OUTPUT = 10_000

    def __init__(
        self,
        working_dir: Path | None = None,
        timeout: int = 60,
        deny_patterns: list[str] | None = None,
    ) -> None:
        self.working_dir = working_dir
        self.timeout = timeout
        self.deny_patterns = (deny_patterns or []) + DEFAULT_DENY_PATTERNS

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
        guard_error = self._guard_command(command)
        if guard_error:
            return guard_error

        cwd = self._resolve_cwd(working_dir)
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

    def _guard_command(self, command: str) -> ToolResult | None:
        lower = command.strip().lower()
        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return ToolResult.error("Error: Command blocked by deny pattern filter")
        return None

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
