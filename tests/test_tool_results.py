"""Oversized tool results: saved to the workspace with a preview, read_file exempt."""

import hashlib
from pathlib import Path

from nanoclaw.agent.messages import TRUNCATED_SUFFIX
from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.tool_results import TOOL_RESULTS_DIR, maybe_persist_tool_result, normalize_tool_result
from nanoclaw.agent.tools import build_default_tools
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest


def test_small_results_pass_through(tmp_path):
    assert normalize_tool_result("exec", "c1", "ok", workspace=tmp_path, max_chars=100) == "ok"
    assert not (tmp_path / TOOL_RESULTS_DIR).exists()


def test_empty_result_gets_marker():
    assert normalize_tool_result("exec", "c1", "  ", workspace=None, max_chars=100) == "(exec completed with no output)"


def test_without_workspace_results_are_truncated():
    out = normalize_tool_result("exec", "c1", "x" * 50, workspace=None, max_chars=10)
    assert out == "x" * 10 + TRUNCATED_SUFFIX


def test_large_result_is_saved_with_head_and_tail_preview(tmp_path):
    content = "HEAD" + "m" * 5000 + "TAIL"
    out = maybe_persist_tool_result(tmp_path, "default", "call/1", content, max_chars=1000)

    digest = hashlib.sha256(content.encode()).hexdigest()[:12]
    saved = tmp_path / TOOL_RESULTS_DIR / "default" / f"call_1-{digest}.txt"  # unsafe "/" replaced
    assert saved.read_text() == content
    assert out.startswith(f"[tool output persisted]\nFull output saved to workspace path: {saved.resolve()}\n")
    assert "Original size: 5008 chars" in out
    assert "HEAD" in out and "TAIL" in out
    assert out.endswith("Result truncated. Read the saved file if you need the complete output.")
    assert len(out) <= 1000


def _saved_path(reference: str) -> Path:
    return Path(reference.split("workspace path: ")[1].split("\n")[0])


def test_repeated_call_id_never_points_to_stale_output(tmp_path):
    """Regression test for the NanoClaw side of docs/UPSTREAM_BUGS.md bug 3."""
    old, new = "OLD-" + "a" * 3000, "NEW-" + "b" * 3000
    r_old = maybe_persist_tool_result(tmp_path, "default", "call_0", old, max_chars=1000)
    r_new = maybe_persist_tool_result(tmp_path, "default", "call_0", new, max_chars=1000)

    # Each reference shows a preview of, and points to, its own output.
    assert r_old.split("Preview:\n")[1].startswith("OLD-")
    assert r_new.split("Preview:\n")[1].startswith("NEW-")
    assert _saved_path(r_old).read_text() == old  # the earlier reference is not overwritten
    assert _saved_path(r_new).read_text() == new
    assert _saved_path(r_old) != _saved_path(r_new)


def test_identical_output_reuses_one_file(tmp_path):
    content = "same" * 1000
    r1 = maybe_persist_tool_result(tmp_path, "default", "c1", content, max_chars=500)
    r2 = maybe_persist_tool_result(tmp_path, "default", "c1", content, max_chars=500)
    assert r1 == r2
    assert len(list((tmp_path / TOOL_RESULTS_DIR / "default").iterdir())) == 1


def test_tiny_budget_falls_back_to_bare_reference(tmp_path):
    out = maybe_persist_tool_result(tmp_path, "default", "c1", "x" * 500, max_chars=50)
    assert out.startswith("[truncated: ") and str(tmp_path) in out


def test_read_file_is_never_cut(tmp_path):
    big = "1| line\n" * 5000 + "\n(Showing lines 1-5000 of 9000. Use offset=5001 to continue.)"
    assert normalize_tool_result("read_file", "c1", big, workspace=tmp_path, max_chars=100) == big
    assert normalize_tool_result("read_file", "c1", big, workspace=None, max_chars=100) == big


def _run(tmp_path: Path, tool_name: str, args: dict, *, workspace: Path | None, max_chars: int = 16_000):
    fake = FakeProvider(
        [
            LLMResponse(content=None, tool_calls=[ToolCallRequest("c1", tool_name, args)], finish_reason="tool_calls"),
            "done",
        ]
    )
    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "go"}],
        tools=build_default_tools(tmp_path),
        provider=fake,
        workspace=workspace,
        max_tool_result_chars=max_chars,
    )
    return AgentRunner().run(spec)


async def test_runner_keeps_read_file_footer_on_large_files(tmp_path):
    (tmp_path / "big.txt").write_text("\n".join(f"line {i:05d} " + "x" * 20 for i in range(3000)))
    result = await _run(tmp_path, "read_file", {"path": "big.txt"}, workspace=tmp_path)
    content = result.messages[2]["content"]
    assert len(content) > 16_000
    assert content.endswith("(Showing lines 1-2000 of 3000. Use offset=2001 to continue.)")


async def test_runner_offloads_large_exec_output_to_workspace(tmp_path):
    result = await _run(
        tmp_path, "exec", {"command": "seq 1 3000"}, workspace=tmp_path, max_chars=2000
    )
    content = result.messages[2]["content"]
    assert content.startswith("[tool output persisted]")
    saved = _saved_path(content)
    assert saved.parent == tmp_path / TOOL_RESULTS_DIR / "default"
    assert saved.name.startswith("c1-") and saved.suffix == ".txt"
    assert saved.read_text().startswith("1\n2\n3\n")
    assert "Exit code: 0" in saved.read_text()


async def test_runner_normalizes_error_results_too(tmp_path):
    # Error results (here: list_dir on a missing, very long path) are kept in budget too.
    result = await _run(tmp_path, "list_dir", {"path": "x" * 300}, workspace=None, max_chars=40)
    assert result.messages[2]["content"].endswith(TRUNCATED_SUFFIX)
