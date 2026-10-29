"""Shared test setup."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nanoclaw.utils import tokens

# The MCP SDK's stdio_client binds ``errlog=sys.stderr`` when its module is
# first imported. Import it now, under pytest's session-wide capture, so a
# test using capsys/capfd cannot make it keep a per-test stream that is
# closed afterwards ("I/O operation on closed file" in later MCP tests).
import mcp.client.stdio  # noqa: E402,F401

# The ``history`` fixture of the WebUI history tests (tests/webui).
from webui_history_fixture import history  # noqa: E402,F401

REAL_GET_TOKEN_ENCODING = tokens.get_token_encoding

WINDOWS = sys.platform == "win32"

# The exec tool runs bash on POSIX and PowerShell on Windows (layer 16, task
# 16.4). Tests that really run a command give Windows a PowerShell variant
# where they can; this marks the few whose command, or what they check, only
# makes sense in a POSIX shell (docs/WINDOWS.md lists them).
def requires_posix_shell(reason: str):
    return pytest.mark.skipif(WINDOWS, reason=f"POSIX shell only: {reason}")


def pid_running(pid: int) -> bool:
    """Whether process ``pid`` is alive (zombies count as gone). ``os.kill(pid, 0)``
    on POSIX, ``tasklist`` on Windows; psutil is not a dependency."""
    if WINDOWS:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, check=False,
        ).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    try:  # Linux: an unreaped zombie still answers os.kill
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def symlink_or_skip(link, target):
    """Create ``link`` -> ``target``; on Windows, skip the test when symlinks need
    admin rights or Developer Mode (docs/WINDOWS.md, cause D)."""
    try:
        link.symlink_to(target)
    except OSError as exc:
        if sys.platform != "win32":
            raise
        pytest.skip(f"cannot create symlinks here (admin rights or Developer Mode needed): {exc}")


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """Point ~ at a temporary folder, so no test touches the real ~/.nanoclaw.
    Windows looks up ~ in USERPROFILE, not HOME, so both are set."""
    home = str(tmp_path_factory.mktemp("home"))
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)


@pytest.fixture(autouse=True)
def _byte_token_counts(monkeypatch):
    """Count tokens by UTF-8 bytes (nanobot's fallback), so token tests are exact
    and never download tiktoken's vocabulary. Tests of the tiktoken path use
    ``REAL_GET_TOKEN_ENCODING``."""
    monkeypatch.setattr(tokens, "get_token_encoding", lambda: None)


# Hostnames tests use, and what the fake resolver returns for them. Anything
# else under .test resolves to a public address; other names fail to resolve.
FAKE_DNS = {
    "localhost": ["127.0.0.1"],
    "internal.test": ["10.0.0.5"],
    "metadata.test": ["169.254.169.254"],
    "rebind.test": ["93.184.216.34", "127.0.0.1"],
    "v6local.test": ["::1"],
    "mapped.test": ["::ffff:127.0.0.1"],
}


@pytest.fixture
def fake_dns(monkeypatch):
    """No real DNS or proxy: resolve test hostnames from ``FAKE_DNS``.

    Returns the dict, so a test can add names; a value may be a function
    called on each lookup (to change the answer between lookups).
    """
    import ipaddress
    import socket

    table = dict(FAKE_DNS)

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
        host = str(host).rstrip(".").lower()
        try:
            ipaddress.ip_address(host.strip("[]"))
            ips = [host.strip("[]")]
        except ValueError:
            entry = table.get(host)
            ips = (entry() if callable(entry) else entry) or (["93.184.216.34"] if host.endswith(".test") else None)
        if ips is None:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        out = []
        for ip in ips:
            v6 = ipaddress.ip_address(ip).version == 6
            fam = socket.AF_INET6 if v6 else socket.AF_INET
            out.append((fam, socket.SOCK_STREAM, 6, "", (ip, port or 0, 0, 0) if v6 else (ip, port or 0)))
        return out

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    return table
