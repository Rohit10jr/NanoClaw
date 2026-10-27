"""edit_file occurrence / line_hint / expected_replacements.

Ported from nanobot tests/tools/test_file_edit_coding_enhancements.py (c4dac64).
"""

from __future__ import annotations

from nanoclaw.agent.tools import is_tool_error
from nanoclaw.agent.tools.filesystem import EditFileTool


async def _edit(tmp_path, initial: str, **kwargs):
    target = tmp_path / "duplicate.txt"
    target.write_text(initial)
    result = await EditFileTool(tmp_path).execute(path=str(target), **kwargs)
    return result, target.read_text()


async def test_edit_file_can_select_occurrence(tmp_path):
    result, text = await _edit(tmp_path, "one\nsame\ntwo\nsame\n", old_text="same", new_text="changed", occurrence=2)
    assert "Patch applied:" in result
    assert text == "one\nsame\ntwo\nchanged\n"


async def test_edit_file_expected_replacements_guards_replace_all(tmp_path):
    result, text = await _edit(
        tmp_path, "same\nsame\n", old_text="same", new_text="changed", replace_all=True, expected_replacements=1
    )
    assert is_tool_error(result)
    assert result == "Error: expected 1 replacements but would make 2."
    assert text == "same\nsame\n"


async def test_edit_file_expected_replacements_allows_replace_all_when_count_matches(tmp_path):
    result, text = await _edit(
        tmp_path, "same\nsame\n", old_text="same", new_text="changed", replace_all=True, expected_replacements=2
    )
    assert "Patch applied:" in result
    assert text == "changed\nchanged\n"


async def test_edit_file_expected_replacements_guards_single_replacement(tmp_path):
    result, text = await _edit(tmp_path, "same\nother\n", old_text="same", new_text="x", expected_replacements=2)
    assert result == "Error: expected 2 replacements but would make 1."
    assert text == "same\nother\n"


async def test_edit_file_line_hint_selects_matching_occurrence(tmp_path):
    result, text = await _edit(tmp_path, "one\nsame\ntwo\nsame\n", old_text="same", new_text="changed", line_hint=4)
    assert "Patch applied:" in result
    assert text == "one\nsame\ntwo\nchanged\n"


async def test_edit_file_rejects_unique_match_outside_line_hint(tmp_path):
    result, text = await _edit(tmp_path, "one\nsame\ntwo\nother\n", old_text="same", new_text="changed", line_hint=4)
    assert "line_hint 4 does not match the old_text location" in result
    assert "old_text appears at line 2" in result
    assert text == "one\nsame\ntwo\nother\n"


async def test_edit_file_line_hint_can_cover_multiline_match(tmp_path):
    result, text = await _edit(
        tmp_path,
        "before\nstart\nmiddle\nend\nafter\n",
        old_text="start\nmiddle\nend",
        new_text="start\nchanged\nend",
        line_hint=3,
    )
    assert "Patch applied:" in result
    assert text == "before\nstart\nchanged\nend\nafter\n"


async def test_edit_file_line_hint_excludes_line_after_trailing_newline(tmp_path):
    """A match ending in "\\n" doesn't cover the next line (nanobot _match_end_line)."""
    result, text = await _edit(tmp_path, "a\nb\nc\n", old_text="a\n", new_text="x\n", line_hint=2)
    assert "line_hint 2 does not match" in result
    assert text == "a\nb\nc\n"


async def test_edit_file_multiple_match_hint_mentions_occurrence(tmp_path):
    result, text = await _edit(tmp_path, "same\nsame\n", old_text="same", new_text="changed")
    assert "old_text appears 2 times" in result
    assert "occurrence" in result
    assert text == "same\nsame\n"


async def test_edit_file_rejects_ambiguous_line_hint(tmp_path):
    result, text = await _edit(tmp_path, "same same\n", old_text="same", new_text="changed", line_hint=1)
    assert "line_hint 1 is ambiguous" in result
    assert text == "same same\n"


async def test_edit_file_rejects_occurrence_with_replace_all(tmp_path):
    result, text = await _edit(
        tmp_path, "same\nsame\n", old_text="same", new_text="changed", occurrence=1, replace_all=True
    )
    assert "occurrence cannot be used with replace_all" in result
    assert text == "same\nsame\n"


async def test_edit_file_rejects_line_hint_with_replace_all(tmp_path):
    result, text = await _edit(
        tmp_path, "same\nsame\n", old_text="same", new_text="changed", line_hint=1, replace_all=True
    )
    assert "line_hint cannot be used with replace_all" in result
    assert text == "same\nsame\n"


async def test_edit_file_rejects_line_hint_with_occurrence(tmp_path):
    result, text = await _edit(tmp_path, "same\nsame\n", old_text="same", new_text="changed", occurrence=1, line_hint=1)
    assert "line_hint cannot be used with occurrence" in result
    assert text == "same\nsame\n"


async def test_edit_file_rejects_zero_occurrence(tmp_path):
    result, text = await _edit(tmp_path, "same\n", old_text="same", new_text="changed", occurrence=0)
    assert "occurrence must be >= 1" in result
    assert text == "same\n"


async def test_edit_file_rejects_zero_line_hint(tmp_path):
    result, text = await _edit(tmp_path, "same\n", old_text="same", new_text="changed", line_hint=0)
    assert "line_hint must be >= 1" in result
    assert text == "same\n"


async def test_edit_file_rejects_zero_expected_replacements(tmp_path):
    result, text = await _edit(tmp_path, "same\n", old_text="same", new_text="changed", expected_replacements=0)
    assert result == "Error: expected_replacements must be >= 1."
    assert text == "same\n"


async def test_edit_file_line_hint_with_crlf_file(tmp_path):
    target = tmp_path / "win.txt"
    target.write_bytes(b"x\r\nsame\r\ny\r\nsame\r\n")
    result = await EditFileTool(tmp_path).execute(path="win.txt", old_text="same", new_text="z", line_hint=2)
    assert "Patch applied:" in result
    assert target.read_bytes() == b"x\r\nz\r\ny\r\nsame\r\n"
