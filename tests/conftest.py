"""Shared test setup."""

import pytest

from nanoclaw.utils import tokens

REAL_GET_TOKEN_ENCODING = tokens.get_token_encoding


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """Point ~ at a temporary folder, so no test touches the real ~/.nanoclaw."""
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))


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
