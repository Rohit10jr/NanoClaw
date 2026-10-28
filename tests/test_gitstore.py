"""Tests for GitStore core operations.

Ported from nanobot tests/utils/test_gitstore.py; the last class adds the
log/diff/revert paths /dream-log and /dream-restore use.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from nanoclaw.utils.gitstore import GitStore


@pytest.fixture
def git(tmp_path):
    """Create an initialized GitStore with tracked MEMORY.md."""
    g = GitStore(tmp_path, tracked_files=["MEMORY.md", "SOUL.md"])
    g.init()
    return g


class TestSummarizeWorkingTree:
    """Ground-truth diff summary used to keep Dream audit records honest."""

    def test_empty_when_not_initialized(self, tmp_path):
        git = GitStore(tmp_path, tracked_files=["MEMORY.md"])
        assert git.summarize_working_tree(["MEMORY.md"]) == ""

    def test_empty_when_no_changes(self, git):
        assert git.summarize_working_tree(["MEMORY.md", "SOUL.md"]) == ""

    def test_summarizes_real_change(self, git, tmp_path):
        (tmp_path / "MEMORY.md").write_text("# Memory\n- new fact\n", encoding="utf-8")
        summary = git.summarize_working_tree(["MEMORY.md"])
        assert "MEMORY.md: +2 -0" in summary
        assert "new fact" in summary
        assert "1 file changed, 2 insertions(+), 0 deletions(-)" in summary

    def test_only_reports_requested_paths(self, git, tmp_path):
        # MEMORY.md changes, but we only ask about the unchanged SOUL.md.
        (tmp_path / "MEMORY.md").write_text("changed\n", encoding="utf-8")
        assert git.summarize_working_tree(["SOUL.md"]) == ""

    def test_counts_additions_and_removals(self, git, tmp_path):
        (tmp_path / "MEMORY.md").write_text("# M\n- keep\n- new\n", encoding="utf-8")
        summary = git.summarize_working_tree(["MEMORY.md"])
        assert "MEMORY.md: +3 -0" in summary

    def test_detects_deletion(self, git, tmp_path):
        # File removed from the working tree (must have content first; the
        # fixture's tracked files start empty, so an empty-file delete is a no-op).
        (tmp_path / "MEMORY.md").write_text("has content\n", encoding="utf-8")
        git.auto_commit("add content")
        (tmp_path / "MEMORY.md").unlink()
        summary = git.summarize_working_tree(["MEMORY.md"])
        assert summary  # a removal is still a change
        assert "deletion" in summary

    def test_non_utf8_file_marked_binary_without_replacement_chars(self, git, tmp_path):
        # Invalid UTF-8 must not leak replacement chars into the audit record.
        (tmp_path / "MEMORY.md").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\x00\x01")
        summary = git.summarize_working_tree(["MEMORY.md"])
        assert "MEMORY.md: binary or non-UTF-8 file changed" in summary
        assert "\ufffd" not in summary  # no U+FFFD replacement chars leaked


class TestNestedRepoProtection:
    """Regression tests for GitHub issue #2980: nested repo protection."""

    def test_init_refuses_inside_git_repo(self, tmp_path):
        """init() should detect it's inside an existing git repo and refuse."""
        project = tmp_path / "project"
        project.mkdir()
        (project / ".git").mkdir()

        workspace = project / "workspace"
        workspace.mkdir()

        g = GitStore(workspace, tracked_files=["MEMORY.md"])
        result = g.init()

        assert result is False
        assert not (workspace / ".git").is_dir()

    def test_init_preserves_existing_gitignore(self, tmp_path):
        """init() should preserve existing .gitignore entries and append new ones."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        existing = "*.pyc\n__pycache__/\n"
        (workspace / ".gitignore").write_text(existing, encoding="utf-8")

        g = GitStore(workspace, tracked_files=["MEMORY.md"])
        result = g.init()

        assert result is True
        gitignore = (workspace / ".gitignore").read_text(encoding="utf-8")
        assert "*.pyc" in gitignore
        assert "__pycache__/" in gitignore
        assert "!MEMORY.md" in gitignore
        assert "!.gitignore" in gitignore

    def test_init_no_gitignore_creates_new(self, tmp_path):
        """init() should create .gitignore with Dream content when none exists."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        g = GitStore(workspace, tracked_files=["MEMORY.md"])
        result = g.init()

        assert result is True
        gitignore = (workspace / ".gitignore").read_text(encoding="utf-8")
        expected = g._build_gitignore()
        assert gitignore == expected

    def test_init_gitignore_merge_idempotent(self, tmp_path):
        """init() should not duplicate Dream entries already in .gitignore."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Pre-existing .gitignore that already has some Dream entries
        existing = "*.pyc\n/*\n!MEMORY.md\n"
        (workspace / ".gitignore").write_text(existing, encoding="utf-8")

        g = GitStore(workspace, tracked_files=["MEMORY.md"])
        result = g.init()

        assert result is True
        gitignore = (workspace / ".gitignore").read_text(encoding="utf-8")
        # No duplicate lines
        lines = gitignore.splitlines()
        assert lines.count("/*") == 1
        assert lines.count("!MEMORY.md") == 1
        # Existing entry preserved, new Dream entries appended
        assert "*.pyc" in gitignore
        assert "!.gitignore" in gitignore

    def test_init_outside_git_repo_works_normally(self, tmp_path):
        """init() should succeed and create .git when not inside a git repo."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        g = GitStore(workspace, tracked_files=["MEMORY.md"])
        result = g.init()

        assert result is True
        assert (workspace / ".git").is_dir()

    def test_staging_paths_are_absolute_from_workspace(self, tmp_path, monkeypatch):
        """Git operations should not depend on the process working directory."""
        from dulwich import porcelain

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(tmp_path)

        git = GitStore(workspace, tracked_files=["MEMORY.md"])

        with patch.object(porcelain, "add", wraps=porcelain.add) as mock_add:
            assert git.init() is True
            assert len(git.log()) == 1

            (workspace / "MEMORY.md").write_text("updated\n", encoding="utf-8")
            assert git.auto_commit("update memory") is not None
            assert len(git.log()) == 2

        assert len(mock_add.call_args_list) == 2
        for call in mock_add.call_args_list:
            staging_paths = [Path(path) for path in call.kwargs["paths"]]
            assert all(path.is_absolute() for path in staging_paths)
            assert all(path.is_relative_to(workspace) for path in staging_paths)

    def test_staging_paths_preserve_symlinks(self, tmp_path):
        """Absolute staging paths should still identify the tracked symlink itself."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = tmp_path / "shared-memory.md"
        target.write_text("shared\n", encoding="utf-8")
        link = workspace / "MEMORY.md"
        try:
            link.symlink_to(target)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        git = GitStore(workspace, tracked_files=["MEMORY.md"])

        staging_path = Path(git._staging_paths("MEMORY.md")[0])
        assert staging_path == link.absolute()
        assert staging_path.is_symlink()

    def test_init_refuses_inside_git_worktree(self, tmp_path):
        """init() should refuse when the parent checkout is a git worktree."""
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "README.md").write_text("x\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-q",
                "-m",
                "init",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(repo), "branch", "wt-branch"], check=True)

        worktree = tmp_path / "worktree"
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "-q", str(worktree), "wt-branch"],
            check=True,
        )
        assert (worktree / ".git").is_file()

        workspace = worktree / "workspace"
        workspace.mkdir()

        g = GitStore(workspace, tracked_files=["MEMORY.md"])
        result = g.init()

        assert result is False
        assert not (workspace / ".git").exists()


class TestCommitIdEncoding:
    """Commit ids must be usable with git, not hex-of-hex."""

    def test_auto_commit_returns_the_real_short_sha(self, git, tmp_path):
        (tmp_path / "MEMORY.md").write_text("- a fact\n", encoding="utf-8")
        sha = git.auto_commit("memory update")
        expected = subprocess.run(
            ["git", "-C", str(tmp_path), "log", "-1", "--format=%h", "--abbrev=8"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert sha == expected

    def test_a_real_git_sha_resolves(self, git, tmp_path):
        (tmp_path / "MEMORY.md").write_text("- a fact\n", encoding="utf-8")
        git.auto_commit("memory update")
        real = subprocess.run(
            ["git", "-C", str(tmp_path), "log", "-1", "--format=%h", "--abbrev=8"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert git._resolve_sha(real) is not None


class TestHistoryAndRestore:
    """log / show_commit_diff / revert, as /dream-log and /dream-restore use them."""

    def test_init_commit_names_nanoclaw(self, git, tmp_path):
        [first] = git.log()
        assert first.message == "init: nanoclaw memory store"
        assert len(first.sha) == 8
        author = subprocess.run(
            ["git", "-C", str(tmp_path), "log", "-1", "--format=%an <%ae>|%cn <%ce>"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert author == "nanoclaw <nanoclaw@dream>|nanoclaw <nanoclaw@dream>"

    def test_init_is_idempotent_and_writes_nanobots_gitignore(self, tmp_path):
        g = GitStore(tmp_path, tracked_files=["SOUL.md", "USER.md", "memory/MEMORY.md"])
        assert g.init() is True
        assert g.init() is False
        assert (tmp_path / ".gitignore").read_text(encoding="utf-8") == (
            "/*\n!memory/\n!SOUL.md\n!USER.md\n!memory/MEMORY.md\n!.gitignore\n"
        )
        # Missing tracked files are created empty for the first commit.
        assert (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8") == ""

    def test_auto_commit_without_changes_is_none(self, git):
        assert git.auto_commit("dream: nothing") is None
        assert len(git.log()) == 1

    def test_not_initialized_is_inert(self, tmp_path):
        g = GitStore(tmp_path, tracked_files=["MEMORY.md"])
        assert g.auto_commit("x") is None
        assert g.log() == []
        assert g.diff_commits("a", "b") == ""
        assert g.revert("abc") is None

    def test_log_filters_by_prefix_and_counts_matches(self, git, tmp_path):
        for i, msg in enumerate(["dream: one", "other: two", "dream: three"]):
            (tmp_path / "MEMORY.md").write_text(f"v{i}\n", encoding="utf-8")
            git.auto_commit(msg)
        assert [c.message for c in git.log()] == [
            "dream: three", "other: two", "dream: one", "init: nanoclaw memory store",
        ]
        assert [c.message for c in git.log(max_entries=2, message_prefix="dream:")] == [
            "dream: three", "dream: one",
        ]
        assert git.log()[0].subject() == "dream: three"

    def test_show_commit_diff_against_parent(self, git, tmp_path):
        (tmp_path / "MEMORY.md").write_text("old\n", encoding="utf-8")
        git.auto_commit("dream: first")
        (tmp_path / "MEMORY.md").write_text("new\n", encoding="utf-8")
        sha = git.auto_commit("dream: second\n\nbody")
        commit, diff = git.show_commit_diff(sha)
        assert commit.sha == sha and commit.subject() == "dream: second"
        assert "diff --git a/MEMORY.md b/MEMORY.md" in diff
        assert "-old" in diff and "+new" in diff
        assert git.show_commit_diff("deadbeef") is None
        assert git.show_commit_diff(sha, message_prefix="other:") is None
        # The root commit has no parent: an empty diff.
        root = git.log()[-1]
        assert git.show_commit_diff(root.sha) == (root, "")

    def test_revert_restores_parent_state_and_commits(self, git, tmp_path):
        (tmp_path / "MEMORY.md").write_text("before\n", encoding="utf-8")
        (tmp_path / "SOUL.md").write_text("soul\n", encoding="utf-8")
        git.auto_commit("dream: base")
        (tmp_path / "MEMORY.md").write_text("after\n", encoding="utf-8")
        sha = git.auto_commit("dream: change")
        new_sha = git.revert(sha, message_prefix="dream:")
        assert new_sha is not None
        assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8") == "before\n"
        assert (tmp_path / "SOUL.md").read_text(encoding="utf-8") == "soul\n"
        assert git.log()[0].message == f"revert: undo {sha}"
        assert git.log()[0].sha == new_sha

    def test_revert_refuses_root_other_prefix_and_unknown(self, git, tmp_path):
        root = git.log()[0].sha
        assert git.revert(root) is None
        (tmp_path / "MEMORY.md").write_text("x\n", encoding="utf-8")
        sha = git.auto_commit("backup: snapshot")
        assert git.revert(sha, message_prefix="dream:") is None
        assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8") == "x\n"
        assert git.revert("deadbeef") is None

    def test_explicitly_tracked_ignored_file_is_still_committed(self, tmp_path):
        """MemoryStore lists memory/.dream_cursor, which .gitignore leaves out;
        staging it by name commits it anyway, as in nanobot."""
        GitStore(tmp_path, tracked_files=["SOUL.md", "USER.md", "memory/MEMORY.md"]).init()
        g = GitStore(tmp_path, tracked_files=["SOUL.md", "USER.md", "memory/MEMORY.md", "memory/.dream_cursor"])
        (tmp_path / "memory" / ".dream_cursor").write_text("3", encoding="utf-8")
        assert g.auto_commit("dream: manual run") is not None
        files = subprocess.run(
            ["git", "-C", str(tmp_path), "ls-files"], capture_output=True, text=True, check=True,
        ).stdout.split()
        assert files == [".gitignore", "SOUL.md", "USER.md", "memory/.dream_cursor", "memory/MEMORY.md"]
