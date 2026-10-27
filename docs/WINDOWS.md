# Running on Windows: test failures and options

On Windows 11 (Python 3.12.12, `uv run pytest`, 2026-10-17) the suite gave
**29 failed, 543 passed**. The same commit (`712d649`) passes fully on Linux
(572 passed; checked in a fresh clone, also with a fresh home folder and
offline, where the 2 tiktoken tests skip).

All 29 failures come from Linux/macOS assumptions. None is a regression, and
where a test fails on paths, the tool's output matches nanobot's on Windows
(checked against nanobot's code).

## The 29 failures by cause

| # | Cause | Failures | Whose problem | Fix |
|---|---|---|---|---|
| A | **The shell tool runs `bash`.** On Windows that finds WSL's `bash.exe`, which fails with "RPC call contains a handle…". The UTF-16 text (`T\x00h\x00e\x00`) is WSL's error message | 16 (`test_shell_tool` 11, `test_safety` 3, `test_tool_results` 1, `test_tools_end_to_end` 1) | **Real limitation, already recorded:** layer 3 left out Windows shell support (`notes/03-tools.md`, phase 2). nanobot uses PowerShell on Windows | Port nanobot's Windows shell support (bigger job), or skip these tests on Windows |
| B | **Tests read files without saying UTF-8**, so Windows uses `cp1252` and `—` / curly quotes break | 3 (`test_context`, curly quotes in `test_filesystem_tools`, bundled skills in `test_skills`) | **Test bug.** The app itself always reads and writes with `encoding="utf-8"` | Add `encoding="utf-8"` in the tests |
| C | **Tests expect `/` in paths**; Windows gives `\` | 3 (`list_dir` recursive, tool-results folder, repeated violations) | **Test assumption.** The output matches nanobot on Windows | Compare paths in a platform-neutral way |
| D | **Creating symlinks** needs admin rights or Developer Mode on Windows | 2 (`test_safety`, `test_session`) | Test environment | Skip when a symlink can't be created |
| E | **Folder permission `0o700`** has no meaning on Windows | 1 (`test_sessions_root_is_private`) | Test environment | Skip on Windows |
| F | **The test uses `signal.SIGPIPE` directly**, which Windows doesn't have | 1 (`test_exit_signal_handler`) | **Test bug.** The app already checks `hasattr(signal, "SIGPIPE")` | Same check in the test |
| G | **prompt_toolkit needs a real console**, and pytest captures output | 2 (`test_chat` prompt tests) | Test environment. Real `nanoclaw chat` in a terminal is fine | Give the test a dummy console |
| H | **The test needs the program `sh`**, which isn't on Windows | 1 (`test_requirements`) | Test assumption | Use a program every OS has (e.g. `python`) |

**What works on Windows today:** chat, ask, memory, skills and the file
tools. **What doesn't:** whenever the agent runs a shell command (the `exec`
tool), it fails (cause A).

## Options

| Option | What you get | Effort |
|---|---|---|
| **1. Fix the tests (B–H) and skip shell tests on Windows (A)** | `uv run pytest` passes on Windows, with skips that say why; `exec` still doesn't work there | Small |
| **2. Port nanobot's Windows shell support** | `exec` works on Windows with PowerShell (or `cmd`), like nanobot: the Windows environment, killing the process tree, PowerShell command handling | Medium. A phase 2 item brought forward; the final check is a test run on Windows |
| **3. Use WSL now** | Clone and run NanoClaw inside WSL (Ubuntu). Everything works and all tests pass, since it's Linux | Nothing to build |

Recommended: 1 so the suite is honest on Windows, and 3 for the full agent
today; then 2 as a planned piece of work. Running in Docker or on Ubuntu is
the next thing to try.

## Status

| Item | Status |
|---|---|
| Diagnosis (this page) | Done |
| Option 1: test fixes and Windows skips | Done in layer 11 (task 11.7): causes B–H fixed in the tests; 21 tests skipped on Windows (18 run the shell tool, 2 need symlink rights, 1 POSIX permissions); not yet run on Windows |
| Option 2: Windows shell support | Planned for layer 16 (PowerShell with `taskkill`) |
| Docker / Ubuntu run | Planned |

## Found while fixing the tests (for layer 16)

| Item | Where | Note |
|---|---|---|
| `exec` paths like `C:\...` don't count toward the repeated-violation escalation | `nanoclaw/agent/tools/violations.py` `_OUTSIDE_PATH_PATTERN` (only `/` and `~` paths) | **Fixed in layer 16 (task 16.2):** drive (`C:\x`, `C:/x`, `C:\`) and UNC (`\\server\share`) paths now match, and on a POSIX host `C:\x` and `C:/x` give the same signature. nanobot c4dac64 has the same gap (`utils/runtime.py:136`); this is a NanoClaw difference (tests/test_violations_windows.py) |
| `write_text` without an encoding | `nanoclaw/skills/skill-creator/scripts/init_skill.py` (copied from nanobot) | Writes cp1252 on Windows; same in nanobot |
| `list_dir` recursive mixes `/` after folder names with `\` inside paths | `ListDirTool` | Same as nanobot; cosmetic |
