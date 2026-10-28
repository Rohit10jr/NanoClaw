"""Telegram setup validation: is the bot token real? (nanobot channels/telegram/validation.py)

nanobot runs this from the WebUI settings ("channel-validate",
channels/validation.py ``validate_channel_config``); NanoClaw's web UI has no
channel settings, so ``nanoclaw channels validate telegram`` runs it. The
payload has nanobot's shape (``status`` connected / configured / invalid /
needs_setup, ``checks``, ``identity``, ``can_enable``, ...) and never holds
the token. NanoClaw has no shared ``network.proxy``, so ``useProxy`` is not
checked and ``getMe`` goes through the environment's proxies.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import Any, cast

import httpx

_TIMEOUT_SECONDS = 4.0
_ENV_REF_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# -- payload helpers (nanobot channels/validation.py) ---------------------------------


def check(check_id: str, label: str, status: str, message: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": check_id, "label": label, "status": status}
    if message:
        payload["message"] = message
    return payload


def _status_message(status: str) -> str:
    return {
        "connected": "Connection verified.",
        "configured": "Configuration is present, but full verification was not possible.",
        "needs_setup": "Required setup is missing.",
        "invalid": "Configuration was checked and looks invalid.",
        "unsupported": "This channel is not supported by the WebUI setup checker.",
    }.get(status, "Channel checked.")


def payload(
    name: str,
    status: str,
    checks: list[dict[str, Any]],
    *,
    identity: dict[str, Any] | None = None,
    missing_fields: list[str] | None = None,
    can_enable: bool | None = None,
) -> dict[str, Any]:
    missing = missing_fields or []
    return {
        "name": name,
        "status": status,
        "checks": checks,
        "identity": {key: value for key, value in (identity or {}).items() if value},
        "missing_fields": missing,
        "can_enable": status not in {"needs_setup", "invalid", "unsupported"} and not missing
        if can_enable is None
        else can_enable,
        "requires_restart": False,
        "checked_at": datetime.now(UTC).isoformat(),
        "message": _status_message(status),
    }


def status_from_checks(name: str, checks: list[dict[str, Any]], missing: list[str]) -> dict[str, Any]:
    if missing:
        return payload(name, "needs_setup", checks, missing_fields=missing, can_enable=False)
    if any(c["status"] == "fail" for c in checks):
        return payload(name, "invalid", checks, missing_fields=missing, can_enable=False)
    if any(c["status"] in ("warn", "skipped") for c in checks):
        return payload(name, "configured", checks, missing_fields=missing)
    return payload(name, "connected", checks, missing_fields=missing)


def string_value(value: Any) -> str:
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else str(value).strip()


def message_from_response(data: dict[str, Any], fallback: str) -> str:
    error = data.get("error") or data.get("description") or data.get("message")
    return str(error) if error else fallback


def resolve_env_refs(value: str) -> str:
    """``${VAR}`` resolved, or ``""`` when a variable is unset. (nanobot loader.py:220)"""
    names = _ENV_REF_PATTERN.findall(value)
    if any(name not in os.environ for name in names):
        return ""
    return _ENV_REF_PATTERN.sub(lambda m: os.environ[m.group(1)], value)


# -- Telegram ---------------------------------------------------------------------------


def _get_me(token: str, proxy: str | None) -> dict[str, Any]:
    if proxy and "://" not in proxy:
        proxy = f"http://{proxy}"
    client_kwargs: dict[str, Any] = {"timeout": _TIMEOUT_SECONDS}
    if proxy:
        client_kwargs.update(proxy=proxy, trust_env=False)
    with httpx.Client(**client_kwargs) as client:
        response = client.get(f"https://api.telegram.org/bot{token}/getMe")
        response.raise_for_status()
        data = response.json()
    return cast(dict[str, Any], data) if isinstance(data, dict) else {}


def validate(values: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Check ``values`` (the ``channels.telegram`` section, camelCase). (nanobot validation.py:35)"""
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    raw_token = string_value(values.get("token"))
    if raw_token:
        checks.append(check("field:token", "Token", "pass", "Configured."))
    else:
        missing.append("token")
        checks.append(check("field:token", "Token", "fail", "Required."))
    token = string_value(resolve_env_refs(raw_token))
    if raw_token and not token:
        checks.append(
            check(
                "token_env",
                "Token environment variable",
                "fail",
                "Set every environment variable referenced by the bot token.",
            )
        )
        return status_from_checks("telegram", checks, missing)
    if token:
        if not re.match(r"^\d+:[A-Za-z0-9_-]{20,}$", token):
            checks.append(check("token_format", "Token format", "fail", "Telegram tokens look like 123456:ABC..."))
        else:
            checks.append(check("token_format", "Token format", "pass", "Looks like a BotFather token."))
            try:
                data = _get_me(token, None)
                if data.get("ok") and isinstance(data.get("result"), dict):
                    bot = cast(dict[str, Any], data["result"])
                    identity = {
                        "name": bot.get("username") or bot.get("first_name"),
                        "account": str(bot.get("id") or ""),
                    }
                    checks.append(check("get_me", "Bot identity", "pass", "Telegram accepted the bot token."))
                    return payload("telegram", "connected", checks, identity=identity, missing_fields=missing)
                checks.append(
                    check(
                        "get_me", "Bot identity", "fail", message_from_response(data, "Telegram rejected the token.")
                    )
                )
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                rejected = status_code in {400, 401, 403, 404}
                checks.append(
                    check(
                        "get_me",
                        "Bot identity",
                        "fail" if rejected else "warn",
                        f"Telegram rejected the token: HTTP {status_code}."
                        if rejected
                        else f"Telegram could not verify the token: HTTP {status_code}.",
                    )
                )
            except httpx.TransportError:
                checks.append(check("get_me", "Bot identity", "warn", "Could not reach Telegram now. Try again later."))
            except Exception:
                checks.append(check("get_me", "Bot identity", "warn", "Could not verify Telegram now. Try again later."))
    return status_from_checks("telegram", checks, missing)


def validate_channel_config(name: str, raw_values: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate a channel's saved config plus ``channels.<name>.<field>`` overrides
    without writing anything. (nanobot channels/validation.py:52, Telegram only)"""
    from nanoclaw.config.loader import load_config

    if name != "telegram":
        return payload(name or "unknown", "unsupported", [check("support", "Setup check", "skipped", "Not supported.")])
    values = load_config().channels.telegram.model_dump(mode="json", by_alias=True)
    prefix = f"channels.{name}."
    for raw_key, raw_value in (raw_values or {}).items():
        field = raw_key[len(prefix) :] if raw_key.startswith(prefix) else raw_key
        # A blank secret in a form keeps the saved one (nanobot _merge_form_values).
        if field in ("token", "webhookSecretToken") and raw_value is not None and not string_value(raw_value):
            continue
        values[field] = raw_value
    result = validate(values)
    result["name"] = name
    result.setdefault("requires_restart", True)
    return result


__all__ = ["validate", "validate_channel_config"]
