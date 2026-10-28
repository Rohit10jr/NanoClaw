"""Telegram token validation: nanobot channels/telegram/tests/test_validation.py
(c4dac64). nanobot's five shared-proxy tests are left out (NanoClaw has no
``network.proxy``); the env-ref test keeps its token half."""

from __future__ import annotations

import httpx
import pytest

from nanoclaw.channels.telegram import validation as telegram_validation
from nanoclaw.channels.telegram.validation import validate_channel_config
from nanoclaw.config.loader import save_config
from nanoclaw.config.schema import Config, TelegramConfig


def test_telegram_setup_exposes_proxy_opt_in_as_an_optional_field() -> None:
    # NanoClaw has no setup manifest: the config field is the contract.
    assert TelegramConfig().use_proxy is False
    assert TelegramConfig.model_validate({"token": "x"}).model_dump(by_alias=True)["useProxy"] is False


def test_get_me_builds_http_client_with_explicit_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "123456:abcdefghijklmnopqrstuvwxyz"
    proxy = "socks5://proxy-user:proxy-pass@127.0.0.1:1080"
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"ok": True, "result": {"id": 42}}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            captured["url"] = url
            return FakeResponse()

    monkeypatch.setattr(telegram_validation.httpx, "Client", FakeClient)

    result = telegram_validation._get_me(token, proxy)

    assert result["ok"] is True
    assert captured["kwargs"] == {
        "timeout": 4.0,
        "proxy": proxy,
        "trust_env": False,
    }
    assert captured["url"] == f"https://api.telegram.org/bot{token}/getMe"


def test_get_me_normalizes_proxy_without_a_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"ok": True, "result": {"id": 42}}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def get(self, _url: str) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(telegram_validation.httpx, "Client", FakeClient)

    telegram_validation._get_me("123456:abcdefghijklmnopqrstuvwxyz", "127.0.0.1:7890")

    assert captured["proxy"] == "http://127.0.0.1:7890"


def test_validate_telegram_bad_token_is_invalid(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(), config_path)
    monkeypatch.setattr("nanoclaw.config.loader._current_config_path", config_path)

    result = validate_channel_config("telegram", {"channels.telegram.token": "not-a-token"})

    assert result["status"] == "invalid"
    assert result["can_enable"] is False
    assert result["missing_fields"] == []


@pytest.mark.parametrize("status_code", [401, 404])
def test_validate_telegram_rejects_denied_tokens_without_exposing_them(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    token = "123456:abcdefghijklmnopqrstuvwxyz"
    config_path = tmp_path / "config.json"
    save_config(
        Config.model_validate({"channels": {"telegram": {"token": token}}}),
        config_path,
    )
    monkeypatch.setattr("nanoclaw.config.loader._current_config_path", config_path)

    def raise_http_error(token_value: str, _proxy: str | None) -> dict:
        request = httpx.Request("GET", f"https://api.telegram.org/bot{token_value}/getMe")
        response = httpx.Response(status_code, request=request)
        raise httpx.HTTPStatusError("rejected", request=request, response=response)

    monkeypatch.setattr(telegram_validation, "_get_me", raise_http_error)

    result = validate_channel_config("telegram", {"channels.telegram.token": ""})

    assert result["status"] == "invalid"
    assert result["can_enable"] is False
    assert token not in str(result)
    assert any(
        f"HTTP {status_code}" in check.get("message", "") for check in result["checks"]
    )


def test_validate_telegram_keeps_transient_http_failures_retryable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "123456:abcdefghijklmnopqrstuvwxyz"
    config_path = tmp_path / "config.json"
    save_config(
        Config.model_validate({"channels": {"telegram": {"token": token}}}),
        config_path,
    )
    monkeypatch.setattr("nanoclaw.config.loader._current_config_path", config_path)

    def raise_http_error(token_value: str, _proxy: str | None) -> dict:
        request = httpx.Request("GET", f"https://api.telegram.org/bot{token_value}/getMe")
        response = httpx.Response(503, request=request)
        raise httpx.HTTPStatusError("unavailable", request=request, response=response)

    monkeypatch.setattr(telegram_validation, "_get_me", raise_http_error)

    result = validate_channel_config("telegram", {"channels.telegram.token": ""})

    assert result["status"] == "configured"
    assert result["can_enable"] is True
    assert token not in str(result)
    assert any("HTTP 503" in check.get("message", "") for check in result["checks"])


def test_validate_telegram_resolves_saved_secret_env_refs_without_exposing_them(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "123456:abcdefghijklmnopqrstuvwxyz"
    token_ref = "${TELEGRAM_TOKEN_TEST}"
    monkeypatch.setenv("TELEGRAM_TOKEN_TEST", token)
    config_path = tmp_path / "config.json"
    save_config(
        Config.model_validate(
            {"channels": {"telegram": {"token": token_ref, "useProxy": True}}}
        ),
        config_path,
    )
    monkeypatch.setattr("nanoclaw.config.loader._current_config_path", config_path)
    captured: dict[str, str | None] = {}

    def fake_get_me(token_value: str, proxy_value: str | None) -> dict:
        captured.update(token=token_value, proxy=proxy_value)
        return {"ok": True, "result": {"id": 42, "username": "working_bot"}}

    monkeypatch.setattr(telegram_validation, "_get_me", fake_get_me)

    result = validate_channel_config("telegram")

    assert result["status"] == "connected"
    assert captured == {"token": token, "proxy": None}
    assert token_ref not in str(result)
    assert token not in str(result)




def test_validate_telegram_reports_missing_token(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(), config_path)
    monkeypatch.setattr("nanoclaw.config.loader._current_config_path", config_path)

    result = validate_channel_config("telegram")

    assert result["status"] == "needs_setup"
    assert result["missing_fields"] == ["token"]
    assert result["can_enable"] is False
