"""OS sandbox for shell commands: bubblewrap. Mirrors nanobot/agent/tools/sandbox.py.

``wrap_command("bwrap", ...)`` turns a shell command into a ``bwrap ... -- sh -c
<command>`` line that runs it in a fresh mount view: system folders read-only,
a private ``/tmp``, the workspace's parent hidden behind an empty tmpfs (in
nanobot's default layout that parent holds ``config.json``), and only the
workspace writable. ``tools.exec.sandboxRoBinds`` / ``sandboxRwBinds`` add
read-only / read-write folders. The network is not isolated (no
``--unshare-net``), as in nanobot.

Deliberate differences from nanobot (decision 11, "bubblewrap only"):

* No macOS seatbelt backend. ``"seatbelt"`` is an unknown name here, so it
  raises the same ``ValueError`` nanobot raises for any unknown backend.
* No read-only bind of a media folder (nanobot ``get_media_dir()``):
  NanoClaw has no media folder.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Iterable
from pathlib import Path


def _normalize_bind_paths(
    paths: Iterable[str] | None,
    *,
    workspace: Path | None = None,
) -> list[str]:
    """Absolute, resolved, de-duplicated bind folders (nanobot ``_normalize_bind_paths``).

    Blank and relative entries are dropped, and so is the workspace itself or
    any parent of it: a later bind there would cover the tmpfs that hides the
    workspace's parent.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in paths or []:
        value = str(raw).strip()
        if not value:
            continue
        path = Path(os.path.expandvars(value)).expanduser()
        if not path.is_absolute():
            continue
        resolved_path = path.resolve(strict=False)
        if workspace is not None:
            try:
                workspace.relative_to(resolved_path)
            except ValueError:
                pass
            else:
                continue
        resolved = str(resolved_path)
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _bwrap(
    command: str,
    workspace: str,
    cwd: str,
    *,
    sandbox_ro_binds: Iterable[str] | None = None,
    sandbox_rw_binds: Iterable[str] | None = None,
) -> str:
    """Wrap ``command`` in a bubblewrap sandbox (requires ``bwrap``).

    The workspace is bind-mounted read-write; its parent folder is hidden
    behind a fresh tmpfs. ``cwd`` outside the workspace falls back to the
    workspace. (nanobot ``_bwrap``)
    """
    ws = Path(workspace).resolve()

    try:
        sandbox_cwd = str(ws / Path(cwd).resolve().relative_to(ws))
    except ValueError:
        sandbox_cwd = str(ws)

    required = ["/usr"]
    optional = [
        "/bin",
        "/lib",
        "/lib64",
        "/etc/alternatives",
        "/etc/ssl/certs",
        "/etc/pki/tls/certs",
        "/etc/pki/ca-trust",
        "/etc/crypto-policies",
        "/etc/resolv.conf",
        "/etc/ld.so.cache",
    ]

    args = ["bwrap", "--new-session", "--die-with-parent", "--setenv", "HOME", str(ws)]
    for p in required:
        args += ["--ro-bind", p, p]
    for p in optional:
        args += ["--ro-bind-try", p, p]
    args += [
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--tmpfs", str(ws.parent),  # hide the workspace's parent (config dir)
        "--dir", str(ws),  # recreate the workspace mount point
        "--bind", str(ws), str(ws),
        # nanobot binds its media folder read-only here; NanoClaw has none.
    ]
    for p in _normalize_bind_paths(sandbox_ro_binds, workspace=ws):
        args += ["--ro-bind-try", p, p]
    for p in _normalize_bind_paths(sandbox_rw_binds, workspace=ws):
        args += ["--bind-try", p, p]
    args += ["--chdir", sandbox_cwd, "--", "sh", "-c", command]
    return shlex.join(args)


_BACKENDS = {"bwrap": _bwrap}


def wrap_command(
    sandbox: str,
    command: str,
    workspace: str,
    cwd: str,
    *,
    sandbox_ro_binds: Iterable[str] | None = None,
    sandbox_rw_binds: Iterable[str] | None = None,
) -> str:
    """Wrap ``command`` using the named sandbox backend (nanobot ``wrap_command``)."""
    if backend := _BACKENDS.get(sandbox):
        return backend(
            command,
            workspace,
            cwd,
            sandbox_ro_binds=sandbox_ro_binds,
            sandbox_rw_binds=sandbox_rw_binds,
        )
    raise ValueError(f"Unknown sandbox backend {sandbox!r}. Available: {list(_BACKENDS)}")
