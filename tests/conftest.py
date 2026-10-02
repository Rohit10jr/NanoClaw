"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """Point ~ at a temporary folder, so no test touches the real ~/.nanoclaw."""
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
