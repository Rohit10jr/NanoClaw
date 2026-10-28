"""HTTP helpers for the web UI listener (nanobot webui/http_utils.py, a subset).

nanobot's helpers read a ``websockets`` connection and request; these read
an aiohttp request (``request.remote`` is the TCP peer, ``request.headers``
is case-insensitive). Left out: the trusted-proxy assertion check and the
JSON gzip helper.
"""

from __future__ import annotations

import hmac
import re
from collections.abc import Mapping
from typing import Any

from nanoclaw.security.network import is_loopback_host as _is_loopback_ip


def strip_trailing_slash(path: str) -> str:
    if len(path) > 1 and path.endswith("/"):
        return path.rstrip("/") or "/"
    return path or "/"


def normalize_config_path(path: str) -> str:
    return strip_trailing_slash(path)


def header(headers: Mapping[str, str], key: str) -> str:
    """One header value, stripped; ``""`` when missing. (nanobot ``case_insensitive_header``)"""
    try:
        value = headers.get(key)
    except Exception:
        value = None
    return str(value or "").strip()


def combined_list_header(headers: Any, key: str) -> str:
    """Repeated values of a comma-separated list header, joined. (nanobot :61)"""
    getall = getattr(headers, "getall", None)
    if callable(getall):
        values = getall(key, [])
        return ", ".join(str(v).strip() for v in values if str(v).strip())
    return header(headers, key)


def safe_host_header(value: str) -> str:
    """A Host value safe to echo, or ``""``. (nanobot :69)"""
    value = value.strip()
    if not value:
        return ""
    if re.fullmatch(r"\[[0-9A-Fa-f:.]+\](?::\d{1,5})?", value):
        return value
    if re.fullmatch(r"[A-Za-z0-9.-]+(?::\d{1,5})?", value):
        return value
    return ""


def host_for_url(host: str, port: int) -> str:
    """(nanobot :81)"""
    host = host.strip()
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{host}:{port}"


def accepts_gzip(value: str) -> bool:
    """Whether an ``Accept-Encoding`` value allows gzip. (nanobot :89)"""
    wildcard_quality: float | None = None
    for item in value.split(","):
        name, *params = (part.strip() for part in item.split(";"))
        quality = 1.0
        for param in params:
            key, separator, raw_value = param.partition("=")
            if separator and key.strip().lower() == "q":
                try:
                    quality = float(raw_value.strip())
                except ValueError:
                    quality = 0.0
                break
        if name.lower() == "gzip":
            return quality > 0
        if name == "*":
            wildcard_quality = quality
    return wildcard_quality is not None and wildcard_quality > 0


def is_localhost(remote: str | None) -> bool:
    """Whether the TCP peer is this machine. (nanobot ``is_localhost``, :208)"""
    if not remote:
        return False
    host = remote
    if host.startswith("::ffff:"):
        host = host[7:]
    return host in {"127.0.0.1", "::1", "localhost"}


def _host_without_port(value: str) -> str:
    value = value.strip().strip('"').strip("'")
    if not value:
        return ""
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end > 0 else value
    if value.count(":") == 1:
        host, port = value.rsplit(":", 1)
        if port.isdigit():
            return host
    return value


def is_loopback_host(value: str) -> bool:
    """A Host / forwarded value naming this machine, with or without a port. (nanobot :282)"""
    host = _host_without_port(value)
    if host.startswith("::ffff:"):
        host = host[7:]
    host = host.rstrip(".").lower()
    return _is_loopback_ip(host)


def _split_comma_header(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _forwarded_header_values(value: str, key: str) -> list[str]:
    values: list[str] = []
    for entry in _split_comma_header(value):
        for part in entry.split(";"):
            name, sep, raw = part.partition("=")
            if sep and name.strip().lower() == key:
                cleaned = raw.strip().strip('"')
                if cleaned:
                    values.append(cleaned)
    return values


def _all_forwarded_values_are_loopback(headers: Mapping[str, str]) -> bool:
    checks: list[str] = []
    checks.extend(_split_comma_header(header(headers, "X-Forwarded-For")))
    checks.extend(_split_comma_header(header(headers, "X-Real-IP")))
    checks.extend(_split_comma_header(header(headers, "X-Forwarded-Host")))
    forwarded = header(headers, "Forwarded")
    checks.extend(_forwarded_header_values(forwarded, "for"))
    checks.extend(_forwarded_header_values(forwarded, "host"))
    return all(is_loopback_host(value) for value in checks)


def is_local_browser_request(remote: str | None, headers: Mapping[str, str]) -> bool:
    """A local TCP peer presenting a local browser origin. (nanobot :321)"""
    if not is_localhost(remote):
        return False
    if not is_loopback_host(header(headers, "Host")):
        return False
    return _all_forwarded_values_are_loopback(headers)


def bearer_token(headers: Mapping[str, str]) -> str | None:
    auth = header(headers, "Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def issue_route_secret_matches(headers: Mapping[str, str], configured_secret: str) -> bool:
    """``Authorization: Bearer <secret>`` or ``X-Nanobot-Auth: <secret>``. (nanobot :338)"""
    if not configured_secret:
        return True
    authorization = header(headers, "Authorization")
    if authorization and authorization.lower().startswith("bearer "):
        return hmac.compare_digest(authorization[7:].strip(), configured_secret)
    header_token = header(headers, "X-Nanobot-Auth")
    if not header_token:
        return False
    return hmac.compare_digest(header_token, configured_secret)
