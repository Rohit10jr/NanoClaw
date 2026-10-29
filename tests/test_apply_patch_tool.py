"""apply_patch (ported from nanobot tests/tools/test_apply_patch_tool.py, c4dac64)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanoclaw.agent.tools import build_default_tools, is_tool_error
from nanoclaw.agent.tools.apply_patch import ApplyPatchTool
from nanoclaw.agent.tools.filesystem import EditFileTool, ListDirTool, WriteFileTool


async def test_apply_patch_edits_replace(tmp_path):
    target = tmp_path / "calc.py"
    target.write_text("def add(a, b):\n    return a + b\n")
    tool = ApplyPatchTool(tmp_path)

    result = await tool.execute(
        edits=[{"path": "calc.py", "action": "replace", "old_text": "    return a + b", "new_text": "    return a - b"}]
    )

    assert "update calc.py" in result
    assert target.read_text() == "def add(a, b):\n    return a - b\n"


async def test_apply_patch_edits_add_new_file(tmp_path):
    result = await ApplyPatchTool(tmp_path).execute(edits=[{"path": "config.py", "action": "add", "new_text": "DEBUG = True"}])

    assert result == "Patch applied:\n- add config.py (+1/-0)"
    assert (tmp_path / "config.py").read_text() == "DEBUG = True\n"


async def test_apply_patch_edits_preserves_new_file_trailing_blank_lines(tmp_path):
    result = await ApplyPatchTool(tmp_path).execute(edits=[{"path": "notes.txt", "action": "add", "new_text": "one\n\n"}])

    assert "add notes.txt" in result
    assert (tmp_path / "notes.txt").read_text() == "one\n\n"


async def test_apply_patch_edits_add_to_existing_file(tmp_path):
    target = tmp_path / "log.py"
    target.write_text("import logging\n\nlogger = logging.getLogger(__name__)\n")

    result = await ApplyPatchTool(tmp_path).execute(
        edits=[{"path": "log.py", "action": "add", "new_text": "def debug(msg):\n    logger.debug(msg)"}]
    )

    assert "update log.py" in result
    assert target.read_text() == "import logging\n\nlogger = logging.getLogger(__name__)\ndef debug(msg):\n    logger.debug(msg)\n"


@pytest.mark.parametrize("new_text", ["beta", "\nbeta"], ids=["without_final_newline", "respects_leading_newline"])
async def test_apply_patch_add_separates_existing_file_without_final_newline(tmp_path, new_text):
    target = tmp_path / "notes.txt"
    target.write_text("alpha", encoding="utf-8")

    result = await ApplyPatchTool(tmp_path).execute(edits=[{"path": "notes.txt", "action": "add", "new_text": new_text}])

    assert "update notes.txt" in result
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"


async def test_apply_patch_edits_add_to_existing_crlf_file_without_final_newline(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"alpha\r\nbravo")

    result = await ApplyPatchTool(tmp_path).execute(edits=[{"path": "notes.txt", "action": "add", "new_text": "charlie"}])

    assert "update notes.txt" in result
    assert target.read_bytes() == b"alpha\r\nbravo\r\ncharlie\r\n"


async def test_apply_patch_replace_keeps_crlf(tmp_path):
    target = tmp_path / "win.txt"
    target.write_bytes(b"one\r\ntwo\r\nthree\r\n")

    result = await ApplyPatchTool(tmp_path).execute(
        edits=[{"path": "win.txt", "action": "replace", "old_text": "two\nthree", "new_text": "2\n3"}]
    )

    assert result == "Patch applied:\n- update win.txt (+2/-2)"
    assert target.read_bytes() == b"one\r\n2\r\n3\r\n"


async def test_apply_patch_rejects_delete_action(tmp_path):
    target = tmp_path / "utils.py"
    target.write_text("def unused():\n    pass\ndef used():\n    return 1\n")

    result = await ApplyPatchTool(tmp_path).execute(
        edits=[{"path": "utils.py", "action": "delete", "old_text": "def unused():\n    pass\n"}]
    )

    assert is_tool_error(result)
    assert "unknown action: delete" in result
    assert target.read_text() == "def unused():\n    pass\ndef used():\n    return 1\n"


async def test_apply_patch_edits_batch_multiple_files(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("X = 1\n")
    b = tmp_path / "b.py"
    b.write_text("from a import X\nprint(X)\n")

    result = await ApplyPatchTool(tmp_path).execute(
        edits=[
            {"path": "a.py", "action": "replace", "old_text": "X = 1", "new_text": "Y = 1"},
            {"path": "b.py", "action": "replace", "old_text": "from a import X", "new_text": "from a import Y"},
        ]
    )

    assert "update a.py" in result
    assert "update b.py" in result
    assert a.read_text() == "Y = 1\n"
    assert b.read_text() == "from a import Y\nprint(X)\n"


async def test_apply_patch_later_edit_sees_earlier_edit(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("a\n")

    result = await ApplyPatchTool(tmp_path).execute(
        edits=[
            {"path": "f.txt", "action": "replace", "old_text": "a", "new_text": "b"},
            {"path": "f.txt", "action": "replace", "old_text": "b", "new_text": "c"},
            {"path": "f.txt", "action": "add", "new_text": "d"},
        ]
    )

    assert result == "Patch applied:\n- update f.txt (+2/-1)"
    assert target.read_text() == "c\nd\n"


async def test_apply_patch_edits_rejects_ambiguous_old_text(tmp_path):
    target = tmp_path / "repeated.txt"
    target.write_text("target\nmiddle\ntarget\n")

    result = await ApplyPatchTool(tmp_path).execute(
        edits=[{"path": "repeated.txt", "action": "replace", "old_text": "target", "new_text": "changed"}]
    )

    assert result == "Error applying patch: old_text appears multiple times in repeated.txt"
    assert target.read_text() == "target\nmiddle\ntarget\n"


async def test_apply_patch_edits_dry_run_validates_without_writing(tmp_path):
    target = tmp_path / "dry.txt"
    target.write_text("before\n")

    result = await ApplyPatchTool(tmp_path).execute(
        edits=[
            {"path": "dry.txt", "action": "replace", "old_text": "before", "new_text": "after"},
            {"path": "added.txt", "action": "add", "new_text": "new"},
        ],
        dry_run=True,
    )

    assert result == "Patch dry-run succeeded:\n- update dry.txt (+1/-1)\n- add added.txt (+1/-0)"
    assert target.read_text() == "before\n"
    assert not (tmp_path / "added.txt").exists()


async def test_apply_patch_edits_allows_absolute_and_parent_paths_when_unrestricted(tmp_path):
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    tool = ApplyPatchTool(workspace, restrict_to_workspace=False)

    absolute = await tool.execute(edits=[{"path": str(outside / "absolute.txt"), "action": "add", "new_text": "absolute"}])
    parent = await tool.execute(edits=[{"path": "../outside/parent.txt", "action": "add", "new_text": "parent"}])

    assert "Patch applied" in absolute
    assert "Patch applied" in parent
    assert (outside / "absolute.txt").read_text() == "absolute\n"
    assert (outside / "parent.txt").read_text() == "parent\n"


async def test_apply_patch_edits_rejects_outside_paths_when_restricted(tmp_path):
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    tool = ApplyPatchTool(workspace, restrict_to_workspace=True)

    absolute = await tool.execute(edits=[{"path": str(outside / "absolute.txt"), "action": "add", "new_text": "nope"}])
    parent = await tool.execute(edits=[{"path": "../outside/parent.txt", "action": "add", "new_text": "nope"}])

    assert is_tool_error(absolute) and absolute.startswith("Error: ")
    assert "outside allowed directory" in absolute
    assert "outside allowed directory" in parent
    assert not (outside / "absolute.txt").exists()
    assert not (outside / "parent.txt").exists()


async def test_apply_patch_restricted_cannot_write_skill_read_roots(tmp_path):
    """Reads may use extra roots when restricted; apply_patch writes may not."""
    from nanoclaw.agent.skills import BUILTIN_SKILLS_DIR

    target = next(BUILTIN_SKILLS_DIR.rglob("SKILL.md"))
    before = target.read_bytes()
    tool = ApplyPatchTool(tmp_path, restrict_to_workspace=True)

    result = await tool.execute(edits=[{"path": str(target), "action": "add", "new_text": "x"}])

    assert "outside allowed directory" in result
    assert target.read_bytes() == before


async def test_apply_patch_edits_reports_invalid_edit_shapes(tmp_path):
    tool = ApplyPatchTool(tmp_path)

    missing_path = await tool.execute(edits=[{"action": "add", "new_text": "x"}])
    missing_action = await tool.execute(edits=[{"path": "x.txt", "new_text": "x"}])
    non_object = await tool.execute(edits=["not an object"])
    empty_path = await tool.execute(edits=[{"path": "  ", "action": "add", "new_text": "x"}])
    no_edits = await tool.execute(edits=[])

    assert "path required for edit" in missing_path
    assert "action required for edit: x.txt" in missing_action
    assert "each edit must be an object" in non_object
    assert "patch path cannot be empty" in empty_path
    assert no_edits == "Error applying patch: must provide edits"


async def test_apply_patch_schema_limits_edits_to_twenty(tmp_path):
    registry = build_default_tools(tmp_path)
    edits = [{"path": f"f{i}.txt", "action": "add", "new_text": "x"} for i in range(21)]

    result = await registry.execute("apply_patch", {"edits": edits})

    assert "edits must be at most 20 items" in result
    assert not any(tmp_path.iterdir())


async def test_apply_patch_preflight_rejects_missing_later_file_without_partial_write(tmp_path):
    first = tmp_path / "first.txt"
    first.write_text("before\n")

    result = await ApplyPatchTool(tmp_path).execute(
        edits=[
            {"path": "first.txt", "action": "replace", "old_text": "before", "new_text": "after"},
            {"path": "missing.txt", "action": "replace", "old_text": "remove me", "new_text": "removed"},
        ]
    )

    assert "file to update does not exist: missing.txt" in result
    assert first.read_text() == "before\n"


async def test_apply_patch_rejects_non_utf8_file(tmp_path):
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00x")

    result = await ApplyPatchTool(tmp_path).execute(edits=[{"path": "bin.dat", "action": "add", "new_text": "x"}])

    assert result == "Error applying patch: file is not UTF-8 text: bin.dat"


@pytest.mark.parametrize("first_exists", [True, False], ids=["existing-file", "added-file"])
async def test_apply_patch_edits_rolls_back_when_later_write_fails(tmp_path, monkeypatch, first_exists):
    first = tmp_path / "first.txt"
    first_original = b"before\r\nfirst tail"
    if first_exists:
        first.write_bytes(first_original)
    second = tmp_path / "second.txt"
    second_original = b"before\r\nsecond tail"
    second.write_bytes(second_original)
    write_text = Path.write_text
    failed = False

    def _fail_second_write(path, *args, **kwargs):
        nonlocal failed
        written = write_text(path, *args, **kwargs)
        if path == second and not failed:
            failed = True
            # A persistence error can occur after the file has already changed.
            raise OSError("injected second write failure")
        return written

    monkeypatch.setattr(Path, "write_text", _fail_second_write)

    result = await ApplyPatchTool(tmp_path).execute(
        edits=[
            {"path": "first.txt", "action": "add", "new_text": "after"},
            {"path": "second.txt", "action": "replace", "old_text": "before", "new_text": "after"},
        ]
    )

    assert failed
    assert is_tool_error(result)
    assert "Error applying patch: injected second write failure" in result
    if first_exists:
        assert first.read_bytes() == first_original
    else:
        assert not first.exists()
    assert second.read_bytes() == second_original


def test_apply_patch_registered_with_file_tools(tmp_path):
    names = build_default_tools(tmp_path).tool_names
    assert "apply_patch" in names


async def test_default_apply_patch_follows_workspace_restriction(tmp_path):
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    registry = build_default_tools(workspace, restrict_to_workspace=True)

    result = await registry.execute(
        "apply_patch", {"edits": [{"path": str(outside / "x.txt"), "action": "add", "new_text": "x"}]}
    )

    assert "outside allowed directory" in result
    assert not (outside / "x.txt").exists()


def test_coding_tool_descriptions_steer_editing_priority():
    """nanobot tests/tools/test_tool_descriptions.py (editing part)."""
    apply_patch = ApplyPatchTool().description.lower()
    edit_tool = EditFileTool()
    edit_file = edit_tool.description.lower()
    edit_parameters = edit_tool.parameters["properties"]
    write_file = WriteFileTool().description.lower()

    assert "default tool for code edits" in apply_patch
    assert "multi-file" in apply_patch
    assert "without writing files" in ApplyPatchTool().parameters["properties"]["dry_run"]["description"]
    assert "edit_file only for small exact replacements" in apply_patch

    assert "small, exact replacement" in edit_file
    assert "prefer apply_patch" in edit_file
    assert "occurrence, line_hint, and replace_all=true are mutually exclusive" in edit_file
    assert "copy it from read_file" in edit_parameters["old_text"]["description"].lower()
    assert "must differ from old_text" in edit_parameters["new_text"]["description"].lower()

    assert "replace an entire file" in write_file
    assert "prefer apply_patch" in write_file
    assert ListDirTool().description == (
        "List directory contents, skipping common noise directories (.git, node_modules, __pycache__, etc.)."
    )


# ---------------------------------------------------------------------------
# Edge cases pinned by mutation checks (layer 17)
# ---------------------------------------------------------------------------


async def test_apply_patch_rejects_null_byte_path(tmp_path):
    result = await ApplyPatchTool(tmp_path).execute(edits=[{"path": "a\0b.txt", "action": "add", "new_text": "x"}])

    assert result == "Error applying patch: patch path contains a null byte: 'a\\x00b.txt'"
    assert not any(tmp_path.iterdir())


async def test_apply_patch_reports_missing_text_fields(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("alpha\n")
    tool = ApplyPatchTool(tmp_path)

    add_without_text = await tool.execute(edits=[{"path": "new.txt", "action": "add"}])
    replace_without_new = await tool.execute(edits=[{"path": "f.txt", "action": "replace", "old_text": "alpha"}])
    replace_without_old = await tool.execute(edits=[{"path": "f.txt", "action": "replace", "new_text": "beta"}])
    replace_empty_old = await tool.execute(
        edits=[{"path": "f.txt", "action": "replace", "old_text": "", "new_text": "beta"}]
    )

    assert add_without_text == "Error applying patch: new_text required for add: new.txt"
    assert replace_without_new == "Error applying patch: new_text required for replace: f.txt"
    assert replace_without_old == "Error applying patch: old_text required for replace: f.txt"
    assert replace_empty_old == "Error applying patch: old_text required for replace: f.txt"
    assert target.read_text() == "alpha\n"
    assert not (tmp_path / "new.txt").exists()


async def test_apply_patch_add_new_file_normalizes_crlf_text(tmp_path):
    result = await ApplyPatchTool(tmp_path).execute(edits=[{"path": "n.txt", "action": "add", "new_text": "a\r\nb"}])

    assert result == "Patch applied:\n- add n.txt (+2/-0)"
    assert (tmp_path / "n.txt").read_bytes() == b"a\nb\n"


async def test_apply_patch_replace_reports_missing_old_text(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\n")

    result = await ApplyPatchTool(tmp_path).execute(
        edits=[{"path": "f.txt", "action": "replace", "old_text": "gamma", "new_text": "delta"}]
    )

    assert result == "Error applying patch: old_text not found in f.txt"
    assert target.read_text() == "alpha\nbeta\n"


async def test_apply_patch_replace_matches_crlf_old_text_in_lf_file(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("one\ntwo\nthree\n")

    result = await ApplyPatchTool(tmp_path).execute(
        edits=[{"path": "f.txt", "action": "replace", "old_text": "one\r\ntwo", "new_text": "1\r\n2"}]
    )

    assert result == "Patch applied:\n- update f.txt (+2/-2)"
    assert target.read_bytes() == b"1\n2\nthree\n"


async def test_apply_patch_replace_terminates_last_line(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta")

    result = await ApplyPatchTool(tmp_path).execute(
        edits=[{"path": "f.txt", "action": "replace", "old_text": "beta", "new_text": "gamma"}]
    )

    assert "update f.txt" in result
    assert target.read_text() == "alpha\ngamma\n"


async def test_apply_patch_new_file_edited_again_is_still_an_add(tmp_path):
    result = await ApplyPatchTool(tmp_path).execute(
        edits=[
            {"path": "n.txt", "action": "add", "new_text": "alpha"},
            {"path": "n.txt", "action": "replace", "old_text": "alpha", "new_text": "beta"},
        ]
    )

    assert result == "Patch applied:\n- add n.txt (+1/-0)"
    assert (tmp_path / "n.txt").read_text() == "beta\n"


async def test_apply_patch_stats_ignore_line_ending_conversion(tmp_path):
    """A mixed file becomes CRLF throughout; only the edited line counts as changed."""
    target = tmp_path / "mixed.txt"
    target.write_bytes(b"one\r\ntwo\nthree\n")

    result = await ApplyPatchTool(tmp_path).execute(
        edits=[{"path": "mixed.txt", "action": "replace", "old_text": "three", "new_text": "3"}]
    )

    assert result == "Patch applied:\n- update mixed.txt (+1/-1)"
    assert target.read_bytes() == b"one\r\ntwo\r\n3\r\n"


async def test_apply_patch_schema_requires_an_edit(tmp_path):
    result = await build_default_tools(tmp_path).execute("apply_patch", {"edits": []})

    assert is_tool_error(result)
    assert "at least 1" in result


async def test_apply_patch_schema_limits_actions(tmp_path):
    (tmp_path / "f.txt").write_text("alpha\n")

    result = await build_default_tools(tmp_path).execute(
        "apply_patch", {"edits": [{"path": "f.txt", "action": "delete"}]}
    )

    assert is_tool_error(result)
    assert "must be one of" in result
    assert "unknown action" not in result
