"""The rg tool: installed ripgrep, run through exec sessions. Mirrors nanobot/agent/tools/rg.py.

``RgTool`` is ``ExecTool`` with an argument array instead of a shell command:
``args=["-n", "needle", "."]`` runs ``rg -n needle .`` directly (no shell, so
spaces, quotes, ``|`` or ``$(...)`` in an argument reach rg unchanged; on
Windows ``rg.exe`` is found the same way, through ``shutil.which``). It shares
everything else with ``exec``: the workspace checks, the command guard, PATH
(``path_prepend``/``path_append``), the environment, runtime budgets, output
limits and exec sessions (a long search returns a ``session_id``).

It is registered only when ``tools.exec.enable`` is on and ``rg`` is on PATH
(with the exec PATH settings), as in nanobot. nanobot then drops its own
``grep``/``find_files`` tools (loader.py:130); NanoClaw has neither.

The command guard runs only in restricted mode, as in nanobot. ``exec``
runs it on every command (shell.py docstring), but rg runs no shell, so its
deny list and internal-URL check would only block searches for text such as
``rm -rf`` or ``shutdown``. nanobot also requires ``tools.file.enable``, a
setting NanoClaw doesn't have.
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Any

from nanoclaw.agent.tools.base import ToolResult
from nanoclaw.agent.tools.shell import ExecTool

if TYPE_CHECKING:
    from nanoclaw.config.schema import ExecToolConfig

# The exec arguments rg doesn't take: it builds the command itself and runs no shell.
_DROPPED_PARAMETERS = ("command", "cmd", "shell", "login")


class RgTool(ExecTool):
    @classmethod
    def enabled(cls, exec_config: ExecToolConfig) -> bool:
        """Whether to register rg: exec is enabled and ``rg`` is on the exec PATH (nanobot ``RgTool.enabled``)."""
        if not exec_config.enable:
            return False
        path = os.pathsep.join(
            part
            for part in (exec_config.path_prepend, os.environ.get("PATH", ""), exec_config.path_append)
            if part
        )
        return shutil.which("rg", path=path) is not None

    @property
    def name(self) -> str:
        return "rg"

    @property
    def description(self) -> str:
        return (
            "Search file contents and discover files with ripgrep. "
            "Accepts native rg arguments as an array of strings."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        schema = super().parameters
        properties = schema["properties"]
        for name in _DROPPED_PARAMETERS:
            properties.pop(name, None)
        properties["args"] = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "Native rg arguments, with each argument as one array element",
        }
        schema["required"] = ["args"]
        return schema

    def _guard_command(
        self,
        command: str,
        cwd: str,
        *,
        restrict_to_workspace: bool | None = None,
        workspace_root: str | None = None,
    ) -> ToolResult | None:
        # nanobot's rule (shell.py:446): the guard runs only in restricted mode.
        should_restrict = self.restrict_to_workspace if restrict_to_workspace is None else restrict_to_workspace
        if not should_restrict:
            return None
        return super()._guard_command(
            command, cwd, restrict_to_workspace=restrict_to_workspace, workspace_root=workspace_root
        )

    async def execute(self, args: list[str], **kwargs: Any) -> str:  # type: ignore[override]
        if not args:
            return ToolResult.error("Error: Provide rg arguments, such as --help or --files.")
        for name in _DROPPED_PARAMETERS:
            kwargs.pop(name, None)
        return await super().execute(command=["rg", *args], **kwargs)
