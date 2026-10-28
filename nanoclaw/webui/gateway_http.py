"""The plain-HTTP side of the web UI listener: bootstrap, token issue, static files.

Ports of nanobot webui/ws_http.py ``_handle_bootstrap`` (:729),
``_bootstrap_ws_url`` (:807), ``_handle_token_issue`` (:701) and
``_serve_static`` (:2001), answering aiohttp requests. The REST routes
(``/api/sessions``, ...) are nanoclaw/webui/rest_api.py; the socket is
nanoclaw/channels/websocket/runtime.py.
"""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web
from loguru import logger

from nanoclaw.webui.client_contract import gateway_identity
from nanoclaw.webui.gateway_tokens import GatewayTokenStore, token_response_payload
from nanoclaw.webui.http_utils import (
    accepts_gzip,
    combined_list_header,
    header,
    host_for_url,
    is_local_browser_request,
    issue_route_secret_matches,
    normalize_config_path,
    safe_host_header,
)
from nanoclaw.webui.ingress_policy import DEFAULT_WEBUI_INGRESS_POLICY, WebUIIngressPolicy

if TYPE_CHECKING:
    from nanoclaw.config.schema import WebSocketConfig

NO_STORE_HEADERS = {"Cache-Control": "no-store"}
WEBUI_DIST_ENV = "NANOCLAW_WEBUI_DIST"

# nanobot ws_http.py:267: browsers reject ES modules served as text/plain.
_MIME_FIXES: dict[str, str] = {
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".css": "text/css",
    ".html": "text/html",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".wasm": "application/wasm",
}
for _ext, _ctype in _MIME_FIXES.items():
    mimetypes.add_type(_ctype, _ext, strict=True)

# Runtime capabilities of a browser surface (nanobot settings_api.py:44).
RUNTIME_CAPABILITIES = {
    "can_restart_engine": False,
    "can_open_logs": False,
    "can_export_diagnostics": False,
}


def http_error(status: int, message: str | None = None) -> web.Response:
    """A plain-text error (nanobot http_utils.py ``http_error``)."""
    from http import HTTPStatus

    return web.Response(
        status=status, text=message or HTTPStatus(status).phrase, content_type="text/plain", charset="utf-8"
    )


def json_response(data: Any, *, status: int = 200, no_store: bool = True) -> web.Response:
    return web.Response(
        status=status,
        body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        content_type="application/json",
        charset="utf-8",
        headers=NO_STORE_HEADERS if no_store else None,
    )


def resolve_webui_dist(configured: str, env: dict[str, str] | None = None) -> Path | None:
    """The built frontend directory: ``channels.websocket.webuiDist``, else
    ``NANOCLAW_WEBUI_DIST``; None when unset or not a directory."""
    import os

    raw = configured.strip() or (env if env is not None else os.environ).get(WEBUI_DIST_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        logger.warning("web UI build directory {} does not exist; static files are off", path)
        return None
    return path


class WebUIGatewayHTTP:
    """Bootstrap, token issue and static routes of one web UI listener."""

    def __init__(
        self,
        config: WebSocketConfig,
        tokens: GatewayTokenStore,
        *,
        static_dist_path: Path | None,
        runtime_model_name: Callable[[], str | None] | None = None,
        ingress: WebUIIngressPolicy = DEFAULT_WEBUI_INGRESS_POLICY,
    ) -> None:
        self.config = config
        self.tokens = tokens
        self.static_dist_path = static_dist_path
        self.runtime_model_name = runtime_model_name
        self.ingress = ingress

    @property
    def secret(self) -> str:
        return self.config.token_issue_secret.strip() or self.config.token.strip()

    def _model_name(self) -> str:
        if self.runtime_model_name is not None:
            try:
                raw = self.runtime_model_name()
            except Exception as e:
                logger.debug("bootstrap runtime model resolver failed: {}", e)
            else:
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
        return ""

    def handle_token_issue(self, request: web.Request) -> web.Response:
        """``GET <tokenIssuePath>``: a one-time WebSocket token. (nanobot ws_http.py:701)"""
        secret = self.secret
        if secret:
            if not issue_route_secret_matches(request.headers, secret):
                return http_error(401, "Unauthorized")
        else:
            logger.warning(
                "token_issue_path is set but token_issue_secret is empty; "
                "any client can obtain connection tokens — set token_issue_secret for production."
            )
        if not self.tokens.can_issue():
            logger.error("too many outstanding issued tokens ({}), rejecting issuance", len(self.tokens.issued_tokens))
            return json_response({"error": "too many outstanding tokens"}, status=429)
        token_value = self.tokens.issue_token(self.config.token_ttl_s)
        return json_response(token_response_payload(token_value, self.config.token_ttl_s))

    def handle_bootstrap(self, request: web.Request, *, terminal_probe: bool = False) -> web.Response:
        """``GET /webui/bootstrap``: tokens and limits for the browser. (nanobot ws_http.py:729)

        Without a secret only a local browser (loopback peer, loopback Host,
        no non-loopback forwarding headers) is served (403 otherwise); with
        one, the request must carry it (401 otherwise). The response has a
        one-time WebSocket token (audience ``webui``) and an API token for
        ``/api/*``, both valid ``tokenTtlS`` seconds.
        """
        secret = self.secret
        is_local_browser = is_local_browser_request(request.remote, request.headers)
        if secret:
            if not issue_route_secret_matches(request.headers, secret):
                return http_error(401, "Unauthorized")
        elif not is_local_browser:
            return http_error(403, "bootstrap is localhost-only")

        terminal = gateway_identity(self.tokens.instance_id)
        if terminal_probe:
            return json_response(terminal)

        api_token_allowed = bool(secret) or is_local_browser
        if not self.tokens.can_issue(include_api_token=api_token_allowed):
            return json_response({"error": "too many outstanding tokens"}, status=429)
        token = self.tokens.issue_token(self.config.token_ttl_s, audience="webui")
        api_token = self.tokens.issue_api_token(self.config.token_ttl_s) if api_token_allowed else None
        payload: dict[str, Any] = {
            "token": token,
            "terminal": terminal,
            "ws_path": normalize_config_path(self.config.path),
            "ws_url": self.bootstrap_ws_url(request),
            "expires_in": self.config.token_ttl_s,
            "limits": self.ingress.bootstrap_limits(max_frame_bytes=self.config.max_message_bytes),
            "model_name": self._model_name(),
            "runtime_surface": "browser",
            "runtime_capabilities": dict(RUNTIME_CAPABILITIES),
        }
        if api_token is not None:
            payload["api_token"] = api_token
        return json_response(payload)

    def bootstrap_ws_url(self, request: web.Request) -> str:
        """(nanobot ws_http.py:807)"""
        if self.config.public_ws_url:
            return self.config.public_ws_url
        host = safe_host_header(header(request.headers, "Host"))
        if not host:
            host = host_for_url(self.config.host, self.config.port)
        proto = header(request.headers, "X-Forwarded-Proto").split(",", 1)[0].strip().lower()
        secure = proto in {"https", "wss"} or bool(self.config.ssl_certfile.strip())
        scheme = "wss" if secure else "ws"
        return f"{scheme}://{host}{normalize_config_path(self.config.path)}"

    def serve_static(self, request_path: str, *, accept_encoding: str = "") -> web.Response | None:
        """One file of the built frontend, ``index.html`` for unknown paths
        (hash routing); None without a build. (nanobot ws_http.py:2001)"""
        assert self.static_dist_path is not None
        rel = request_path.lstrip("/") or "index.html"
        if ".." in rel.split("/") or rel.startswith("/"):
            return http_error(403, "Forbidden")
        candidate = (self.static_dist_path / rel).resolve()
        try:
            candidate.relative_to(self.static_dist_path)
        except ValueError:
            return http_error(403, "Forbidden")
        if not candidate.is_file():
            index = self.static_dist_path / "index.html"
            if not index.is_file():
                return None
            candidate = index
        ctype, _ = mimetypes.guess_type(candidate.name)
        ctype = ctype or "application/octet-stream"
        utf8_text = ctype.startswith("text/") or ctype in {"application/javascript", "application/json"}
        compressible = utf8_text or ctype == "image/svg+xml"
        response_path = candidate
        headers: dict[str, str] = {}
        if compressible:
            headers["Vary"] = "Accept-Encoding"
            gzip_candidate = candidate.with_name(f"{candidate.name}.gz")
            if accepts_gzip(accept_encoding) and gzip_candidate.is_file():
                response_path = gzip_candidate
                headers["Content-Encoding"] = "gzip"
        try:
            body = response_path.read_bytes()
        except OSError as e:
            logger.warning("static: failed to read {}: {}", response_path, e)
            return http_error(500, "Internal Server Error")
        headers["Cache-Control"] = "no-cache" if candidate.name == "index.html" else "public, max-age=31536000, immutable"
        headers["Content-Type"] = f"{ctype}; charset=utf-8" if utf8_text else ctype
        return web.Response(status=200, body=body, headers=headers)

    def static_or_404(self, request: web.Request, path: str) -> web.Response:
        if self.static_dist_path is not None:
            response = self.serve_static(path, accept_encoding=combined_list_header(request.headers, "Accept-Encoding"))
            if response is not None:
                return response
        return http_error(404, "Not Found")
