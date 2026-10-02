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
