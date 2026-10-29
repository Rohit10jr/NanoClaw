"""Windows paths in ``exec`` commands count toward the repeated-violation escalation (layer 16.2).

nanobot utils/runtime.py ``_OUTSIDE_PATH_PATTERN`` matches only ``/`` and ``~``
paths, so in nanobot ``exec("type C:\\secret.txt")`` has no signature and is
never escalated. NanoClaw also matches drive and UNC paths
(docs/WINDOWS.md, "Found while fixing the tests").
"""

import os

import pytest

from nanoclaw.agent.tools.violations import repeated_workspace_violation_error, workspace_violation_signature


def sig(command: str) -> str | None:
    return workspace_violation_signature("exec", {"command": command})


@pytest.mark.parametrize(
    ("command", "target"),
    [
        (r"type C:\Users\me\secret.txt", "c:/users/me/secret.txt"),
        (r"dir C:\\", "c:/"),
        ("Get-Content C:/Users/me/secret.txt", "c:/users/me/secret.txt"),
        (r"Get-Content D:\data\..\keys.txt", "d:/keys.txt"),
        (r'type "C:\Windows\win.ini"', "c:/windows/win.ini"),
        (r"type 'C:\Windows\win.ini'", "c:/windows/win.ini"),
        (r"type C:\x | more", "c:/x"),
        (r"echo hi > C:\out.txt", "c:/out.txt"),
        (r"dir \\server\share\folder", "//server/share/folder"),
        (r'copy "\\server\share\a.txt" .', "//server/share/a.txt"),
    ],
)
@pytest.mark.skipif(os.name == "nt", reason="POSIX-host normalization; Windows resolves through Path")
def test_windows_paths_get_a_signature(command, target):
    assert sig(command) == f"violation:{target}"


@pytest.mark.parametrize("command", [r"type C:\x\y.txt", "type C:/x/y.txt", r"type c:\X\Y.TXT", r"type C:\x\.\y.txt"])
def test_drive_spellings_collide(command):
    assert sig(command) == sig(r"type C:\x\y.txt")


@pytest.mark.parametrize(
    "command",
    [
        "curl http://example.com/x",
        "curl https://example.com",
        "echo a:b",
        "echo key:value c:d",
        "python -c print(1)",
        "echo C:",
        "git log --format=%h:%s",
        "echo hello",
    ],
)
def test_no_false_positives(command):
    assert sig(command) is None


def test_posix_paths_unchanged():
    assert sig("cat /etc/passwd") is not None
    assert sig("cat /etc/passwd").endswith("/etc/passwd")
    assert sig("ls ~/x") is not None


def test_repeated_windows_exec_escalates_and_crosses_tools():
    counts: dict[str, int] = {}
    assert repeated_workspace_violation_error("exec", {"command": r"type C:\secret.txt"}, counts) is None
    assert repeated_workspace_violation_error("read_file", {"path": "C:/secret.txt"}, counts) is None
    third = repeated_workspace_violation_error("exec", {"command": r'Get-Content "C:\secret.txt"'}, counts)
    assert third is not None and third.startswith("Error: refusing repeated workspace-bypass attempts.")
    assert "c:/secret.txt" in third


def test_repeated_unc_exec_escalates():
    counts: dict[str, int] = {}
    outs = [repeated_workspace_violation_error("exec", {"command": r"dir \\srv\share"}, counts) for _ in range(3)]
    assert outs[:2] == [None, None] and outs[2] is not None
