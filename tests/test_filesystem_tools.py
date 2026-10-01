"""File tools against real files in a temporary workspace."""

import pytest

from nanoclaw.agent.tools import is_tool_error
from nanoclaw.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool


@pytest.fixture
def ws(tmp_path):
    return tmp_path


def write(ws, name, text):
    p = ws / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# --- read_file ----------------------------------------------------------------


async def test_read_numbers_lines_and_marks_end(ws):
    write(ws, "a.txt", "one\ntwo\nthree\n")
    out = await ReadFileTool(ws).execute(path="a.txt")
    assert out == "1| one\n2| two\n3| three\n\n(End of file — 3 lines total)"


async def test_read_paging(ws):
    write(ws, "a.txt", "\n".join(f"line {i}" for i in range(1, 11)))
    out = await ReadFileTool(ws).execute(path="a.txt", offset=3, limit=2)
    assert out == "3| line 3\n4| line 4\n\n(Showing lines 3-4 of 10. Use offset=5 to continue.)"


async def test_read_absolute_path_and_crlf(ws):
    p = ws / "win.txt"
    p.write_bytes(b"a\r\nb\r\n")
    assert (await ReadFileTool(ws).execute(path=str(p))).startswith("1| a\n2| b")


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
async def test_read_utf16_and_utf32_with_bom(ws, encoding):
    (ws / "win.txt").write_bytes("caf\u00e9\nna\u00efve".encode(encoding))  # Python adds the BOM
    assert (await ReadFileTool(ws).execute(path="win.txt")).startswith("1| caf\u00e9\n2| na\u00efve")


async def test_read_latin1_text_file(ws):
    (ws / "old.csv").write_bytes("name\ncaf\u00e9".encode("latin-1"))
    assert (await ReadFileTool(ws).execute(path="old.csv")).startswith("1| name\n2| caf\u00e9")


async def test_non_utf8_without_text_extension_is_binary(ws):
    (ws / "blob.bin").write_bytes("caf\u00e9".encode("latin-1"))
    out = await ReadFileTool(ws).execute(path="blob.bin")
    assert is_tool_error(out) and out.startswith("Error: Cannot read binary file blob.bin")


async def test_read_utf8_bom_is_dropped(ws):
    (ws / "bom.txt").write_bytes("﻿hello".encode("utf-8"))
    assert (await ReadFileTool(ws).execute(path="bom.txt")).startswith("1| hello")


@pytest.mark.parametrize(
    ("setup", "path", "expected"),
    [
        (None, "missing.txt", "Error: File not found: missing.txt"),
        ("dir", "sub", "Error: Not a file: sub"),
        ("binary", "img.png", "Error: Cannot read binary file img.png (MIME: image/png)"),
        (None, "/dev/zero", "Error: Reading /dev/zero is blocked"),
    ],
)
async def test_read_errors(ws, setup, path, expected):
    if setup == "dir":
        (ws / "sub").mkdir()
    if setup == "binary":
        (ws / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\x00")
    out = await ReadFileTool(ws).execute(path=path)
    assert is_tool_error(out)
    assert out.startswith(expected)


async def test_read_empty_and_offset_past_end(ws):
    write(ws, "empty.txt", "")
    assert await ReadFileTool(ws).execute(path="empty.txt") == "(Empty file: empty.txt)"
    write(ws, "short.txt", "x\n")
    out = await ReadFileTool(ws).execute(path="short.txt", offset=5)
    assert out == "Error: offset 5 is beyond end of file (1 lines)"


async def test_read_caps_characters(ws, monkeypatch):
    monkeypatch.setattr(ReadFileTool, "_MAX_CHARS", 25)  # two numbered lines = 21 chars, three = 32
    write(ws, "long.txt", "aaaaaaa\nbbbbbbb\nccccccc\n")
    out = await ReadFileTool(ws).execute(path="long.txt")
    assert out.startswith("1| aaaaaaa\n2| bbbbbbb\n\n(Showing lines 1-2 of 3. Use offset=3 to continue.)")


async def test_read_single_huge_line_is_cut(ws, monkeypatch):
    monkeypatch.setattr(ReadFileTool, "_MAX_CHARS", 10)
    write(ws, "one.txt", "x" * 50)
    out = await ReadFileTool(ws).execute(path="one.txt")
    assert out.startswith("1| xxxxxxx\n\n(Line 1 truncated;")


# --- write_file -------------------------------------------------------------------


async def test_write_creates_parents_and_overwrites(ws):
    tool = WriteFileTool(ws)
    out = await tool.execute(path="deep/dir/f.txt", content="hello")
    target = ws / "deep/dir/f.txt"
    assert out == f"Successfully wrote 5 characters to {target}"
    assert target.read_text() == "hello"
    await tool.execute(path="deep/dir/f.txt", content="bye")
    assert target.read_text() == "bye"


async def test_write_into_a_directory_fails_cleanly(ws):
    (ws / "d").mkdir()
    out = await WriteFileTool(ws).execute(path="d", content="x")
    assert is_tool_error(out) and out.startswith("Error")


# --- edit_file ------------------------------------------------------------------------


async def test_edit_exact_replacement_and_summary(ws):
    p = write(ws, "app.py", "a = 1\nb = 2\n")
    out = await EditFileTool(ws).execute(path="app.py", old_text="b = 2", new_text="b = 3")
    assert p.read_text() == "a = 1\nb = 3\n"
    assert out == "Patch applied:\n- update app.py (+1/-1)"


async def test_edit_matches_despite_different_indentation(ws):
    p = write(ws, "f.py", "def f():\n    return 1\n")
    await EditFileTool(ws).execute(path="f.py", old_text="return 1", new_text="return 2")
    assert p.read_text() == "def f():\n    return 2\n"
    p = write(ws, "g.py", "def g():\n    x = 1\n    return x\n")
    # Multi-line old_text with the wrong indentation still matches line by line,
    # and new_text (written in the same unindented style) is shifted to match the file.
    await EditFileTool(ws).execute(path="g.py", old_text="x = 1\nreturn x", new_text="y = 2\nreturn y")
    assert p.read_text() == "def g():\n    y = 2\n    return y\n"


async def test_edit_reindents_nested_blocks_relative_to_the_match(ws):
    p = write(ws, "h.py", "class A:\n    def f(self):\n        if x:\n            go()\n")
    old = "if x:\n    go()"  # model dropped the 8-space indent
    new = "if x:\n    go()\nelse:\n    stop()"
    await EditFileTool(ws).execute(path="h.py", old_text=old, new_text=new)
    assert p.read_text() == (
        "class A:\n    def f(self):\n        if x:\n            go()\n        else:\n            stop()\n"
    )


async def test_edit_keeps_indentation_when_old_text_was_exact(ws):
    p = write(ws, "k.py", "def k():\n    a = 1\n")
    await EditFileTool(ws).execute(path="k.py", old_text="    a = 1", new_text="    a = 2")
    assert p.read_text() == "def k():\n    a = 2\n"


async def test_edit_matches_curly_quotes(ws):
    p = write(ws, "q.txt", "say “hello” now\n")
    await EditFileTool(ws).execute(path="q.txt", old_text='say "hello" now', new_text="say bye now")
    assert p.read_text() == "say bye now\n"


async def test_edit_writes_curly_quotes_back_when_the_file_used_them(ws):
    p = write(ws, "q.md", "She said \u201chi\u201d and it\u2019s fine.\n")
    await EditFileTool(ws).execute(
        path="q.md", old_text='She said "hi" and it\'s fine.', new_text='She said "bye" and it\'s \'ok\'.'
    )
    assert p.read_text() == "She said \u201cbye\u201d and it\u2019s \u2018ok\u2019.\n"


async def test_edit_multiple_matches_warns_and_changes_nothing(ws):
    p = write(ws, "m.txt", "x\ny\nx\nx\nx\n")
    out = await EditFileTool(ws).execute(path="m.txt", old_text="x", new_text="z")
    assert not is_tool_error(out)
    assert out.startswith("Warning: old_text appears 4 times at line 1, line 3, line 4, ...")
    assert p.read_text() == "x\ny\nx\nx\nx\n"


async def test_edit_occurrence_and_replace_all(ws):
    p = write(ws, "m.txt", "x\nx\nx\n")
    await EditFileTool(ws).execute(path="m.txt", old_text="x", new_text="second", occurrence=2)
    assert p.read_text() == "x\nsecond\nx\n"
    await EditFileTool(ws).execute(path="m.txt", old_text="x", new_text="y", replace_all=True)
    assert p.read_text() == "y\nsecond\ny\n"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"old_text": "x", "new_text": "y", "occurrence": 9}, "Error: occurrence 9 is out of range; old_text appears 2 time(s)."),
        ({"old_text": "x", "new_text": "y", "occurrence": 1, "replace_all": True}, "Error: occurrence cannot be used with replace_all=true."),
        ({"old_text": "x", "new_text": "x"}, "Error: new_text must be different from old_text."),
        ({"old_text": "", "new_text": "y"}, "Error: Cannot create file — m.txt already exists and is not empty."),
    ],
)
async def test_edit_argument_errors(ws, kwargs, expected):
    write(ws, "m.txt", "x\nx\n")
    out = await EditFileTool(ws).execute(path="m.txt", **kwargs)
    assert is_tool_error(out)
    assert out == expected


async def test_edit_not_found_shows_best_match_diff(ws):
    write(ws, "c.py", "import os\n\ndef greet(name):\n    print('hello', name)\n    return name\n")
    # Two of the three lines match exactly; one has a typo.
    old = "def greet(name):\n    print('helo', name)\n    return name\n"
    out = await EditFileTool(ws).execute(path="c.py", old_text=old, new_text="x")
    assert is_tool_error(out)
    assert out.startswith("Error: old_text not found in c.py.\nBest match (67% similar) at line 3:")
    assert "-    print('helo', name)" in out and "+    print('hello', name)" in out


async def test_edit_single_line_typo_has_no_best_match(ws):
    # nanobot compares whole lines, so a one-line typo scores 0% similar.
    write(ws, "c.py", "def greet(name):\n")
    out = await EditFileTool(ws).execute(path="c.py", old_text="def greet(nam):", new_text="x")
    assert out == "Error: old_text not found in c.py. No similar text found. Verify the file content."


async def test_edit_not_found_with_hint(ws):
    write(ws, "c.txt", "Hello World\n")
    out = await EditFileTool(ws).execute(path="c.txt", old_text="hello world", new_text="x")
    assert "Possible cause: letter case differs." in out


async def test_edit_missing_file_suggests_names(ws):
    write(ws, "config.yaml", "a: 1\n")
    out = await EditFileTool(ws).execute(path="config.yml", old_text="a", new_text="b")
    assert out.startswith("Error: File not found: config.yml\nDid you mean: ")
    assert "config.yaml" in out


async def test_edit_creates_file_with_empty_old_text(ws):
    out = await EditFileTool(ws).execute(path="new/n.txt", old_text="", new_text="one\ntwo\n")
    assert (ws / "new/n.txt").read_text() == "one\ntwo\n"
    assert out == "Patch applied:\n- add new/n.txt (+2/-0)"


async def test_edit_keeps_crlf_line_endings(ws):
    p = ws / "w.txt"
    p.write_bytes(b"a\r\nb\r\n")
    await EditFileTool(ws).execute(path="w.txt", old_text="b", new_text="c")
    assert p.read_bytes() == b"a\r\nc\r\n"


async def test_edit_deleting_a_whole_line_removes_its_newline(ws):
    p = write(ws, "d.txt", "keep\ndrop me\nkeep too\n")
    await EditFileTool(ws).execute(path="d.txt", old_text="drop me", new_text="")
    assert p.read_text() == "keep\nkeep too\n"


async def test_edit_strips_trailing_spaces_except_markdown(ws):
    p = write(ws, "s.py", "x\n")
    await EditFileTool(ws).execute(path="s.py", old_text="x", new_text="y   ")
    assert p.read_text() == "y\n"
    m = write(ws, "s.md", "x\n")
    await EditFileTool(ws).execute(path="s.md", old_text="x", new_text="y  ")
    assert m.read_text() == "y  \n"


# --- list_dir ---------------------------------------------------------------------------


async def test_list_dir_flat_ignores_noise(ws):
    write(ws, "b.txt", "")
    write(ws, "src/main.py", "")
    (ws / ".git").mkdir()
    (ws / "node_modules").mkdir()
    out = await ListDirTool(ws).execute(path=".")
    assert out == "📄 b.txt\n📁 src"


async def test_list_dir_recursive(ws):
    write(ws, "src/main.py", "")
    write(ws, "src/__pycache__/main.pyc", "")
    out = await ListDirTool(ws).execute(path=".", recursive=True)
    assert out == "src/\nsrc/main.py"


async def test_list_dir_ignore_rules_apply_below_the_listed_dir_only(tmp_path):
    # A workspace that itself lives under a "build" folder is still listable.
    ws = tmp_path / "build" / "proj"
    write(ws, "a.txt", "")
    assert await ListDirTool(ws).execute(path=".", recursive=True) == "a.txt"


async def test_list_dir_truncation_and_errors(ws):
    for i in range(5):
        write(ws, f"f{i}.txt", "")
    out = await ListDirTool(ws).execute(path=".", max_entries=2)
    assert out.endswith("\n\n(truncated, showing first 2 of 5 entries)")
    (ws / "empty").mkdir()
    assert await ListDirTool(ws).execute(path="empty") == "Directory empty is empty"
    assert await ListDirTool(ws).execute(path="nope") == "Error: Directory not found: nope"
    assert await ListDirTool(ws).execute(path="f0.txt") == "Error: Not a directory: f0.txt"
