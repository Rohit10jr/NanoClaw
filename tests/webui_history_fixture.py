"""The ``history`` fixture for WebUI history tests (nanobot conftest.py ``DisplayHistoryFixture``).

A module, not ``tests/webui/conftest.py``: a second ``conftest`` would shadow
``from conftest import ...`` in the top-level tests. Test modules import ``history``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanoclaw.session.manager import SessionManager
from nanoclaw.webui.history import WebUIHistoryStore


class _Reopened:
    def __init__(self, directory: Path) -> None:
        self.store = WebUIHistoryStore(directory)

    def project_history(self, key, project):
        return project(self.store.reader(key))


class DisplayHistoryFixture:
    """Explicit offline setup plus snapshot reads for history projection tests."""

    def __init__(self, root: Path) -> None:
        workspace = root / "workspace"
        workspace.mkdir(exist_ok=True)
        self.sessions = SessionManager(workspace, sessions_root=root / "sessions")
        self.store = WebUIHistoryStore.for_sessions(self.sessions)

    def append(self, key, record):
        from nanoclaw.webui.transcript import append_transcript_object

        append_transcript_object(key, record, store=self.store)

    def reopen(self) -> _Reopened:
        return _Reopened(self.store.directory)

    def _project(self, name, key, *args, **kwargs):
        from nanoclaw.webui import transcript

        return getattr(transcript, name)(key, *args, history=self.store.reader(key), **kwargs)

    def read(self, key):
        return self._project("read_transcript_lines", key)

    def build(self, key, **kwargs):
        return self._project("build_webui_thread_response", key, **kwargs)

    def trace(self, key, detail_ref):
        return self._project("build_webui_trace_detail_response", key, detail_ref)

    def revision(self, key, **kwargs):
        return self._project("webui_transcript_revision", key, **kwargs)

    def prompts(self, key, **kwargs):
        return self._project("build_prompt_index", key, **kwargs)

    def prompt_response(self, key, previous_revision, **kwargs):
        return self._project("prompt_index_response", key, previous_revision, **kwargs)


@pytest.fixture
def history(tmp_path: Path) -> DisplayHistoryFixture:
    return DisplayHistoryFixture(tmp_path)
