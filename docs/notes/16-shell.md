# 16 — Shell: exec sessions and Windows

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package.

## The concept

Until now `exec` ran a command to the end and returned its output, with a
60-second default limit (600 s at most). A dev server, a long build or a
command waiting for input either blocked the turn or was killed. nanobot
changed this upstream (`6310a39`, `26e1f41`): **every `exec` call runs as an
exec session**. `exec` waits a short time (1 s by default); a command that is
still running keeps running in the background and the model gets a session id.
It then reads more output, sends input, waits for a string or the exit, or
stops the command with the `exec_session` tool. This layer ports that, and
nanobot's Windows shell (PowerShell or `cmd`), which phase 1 left out.

```
exec("npm run dev")  ──▶ ExecSessionManager.start ──▶ wait yield_time_ms
                              │                         ├─ finished → output, Exit code, Elapsed
                              │                         └─ running  → output so far, "Process running. session_id: …"
exec_session(id, input / wait_for / until_exit / terminate) ──▶ poll ──▶ new output, status
list_exec_sessions ──▶ this chat's sessions
```

## The pieces

| Piece | What it does | nanobot |
|---|---|---|
| `ExecSessionManager` | Up to 8 sessions (a finished one is evicted when full); finished results kept 30 min; sessions belong to the chat that started them; `terminate_by_owner`, `close_all`, `has_pending_work` | `agent/tools/exec_session.py:425` |
| `_ExecSession` | One process: bounded output buffers (head and tail kept), a supervisor that kills at the hard limit (default 1 hour, at most 24 hours), an idle check (300 s without output: `report` adds a note, `terminate` stops it), stdin writes | `exec_session.py:151` |
| `exec_session`, `list_exec_sessions` tools | Poll, write stdin, close stdin, wait for a string (10 s default) or the exit (10 min), terminate; list | `exec_session.py:777, 936` |
| `ShellOutputFilter` | Strips ANSI/OSC escape codes, even split across chunks | `agent/tools/shell_output.py` |
| `ExecTool` on sessions | `yield_time_ms`, `max_runtime` (alias `timeout`, not both), output format: stdout, `STDERR:`, status lines, always `Elapsed: N.Ns`; stdin closed unless `yield_time_ms` is given; PATH prepend/append, `allowedEnvKeys`, `allowPatterns` | `agent/tools/shell.py` |
| `tools.exec` settings | nanobot's `ExecToolConfig`: `enable`, `maxRuntime`, `maxRuntimeLimit`, `idleTimeout`, `idleAction`, `pathPrepend`, `pathAppend`, `allowedEnvKeys`, `allowPatterns`, `denyPatterns` (sandbox fields read, unused until layer 18) | `agent/tools/shell_config.py` |
| Ownership and cleanup | The loop owns a manager: `/new` and `/stop` end that chat's sessions (counted in "Stopped N task(s)."), shutdown ends all. Each subagent gets its own manager, closed before its report is sent | `agent/loop.py:397, 983, 1819`; `agent/subagent.py:106, 274, 318` |
| Windows shell | PowerShell (`pwsh`, else `powershell`) with UTF-8 output and the exit code passed on, or `cmd` via `COMSPEC`; the Windows environment variables; a quoted `.exe` path gets `& `; the process tree is put in a Job Object and killed with it, `taskkill /T /F` otherwise | `shell.py:541-770`, `agent/tools/_windows_job.py` |

## What you see

| Situation | Behaviour |
|---|---|
| A quick command | Same output as before, plus an `Elapsed:` line |
| A command still running after 1 s | Its output so far and `Process running. session_id: …`; the model checks back with `exec_session` |
| `/new`, `/stop` | Running commands of that chat are stopped |
| Leaving `chat`, end of `ask` | All commands started by the agent are stopped; nothing is left running |
| `"tools": {"exec": {"enable": false}}` | No shell tools at all |
| Windows | `exec` runs PowerShell (or `cmd` with `shell: "cmd"`) instead of failing on WSL's `bash.exe` (WINDOWS.md cause A) |

## How it was built

Phase A in parallel: exec sessions and the session-based `exec` (16.1); the
`tools.exec` settings and Windows paths in the violation check (16.2). Phase
B: wiring into the loop, subagents, config and CLI (16.3). Phase C: the
Windows shell (16.4).

| Task | Checked against nanobot |
|---|---|
| 16.1 Exec sessions | 44 scenarios through both tools (quick commands, yield then poll, stdin, `wait_for`, `until_exit`, terminate, hard limit, idle report/terminate, ANSI, owners, capacity, limits, guards, schemas): 41 identical, 3 deliberate (the guard, below); re-checked by the main session |
| 16.2 Settings | 16 config files (valid and invalid, legacy `timeout`) through both loaders: identical |
| 16.3 Wiring | 21 checks through both `AgentLoop`s with real processes (`/new` ends only that chat's commands, `/stop` count, shutdown, a subagent's commands end with it, `enable=false`): identical; a real `nanoclaw ask` leaves no process behind |
| 16.4 Windows | 486 cases with nanobot's Windows monkeypatches (spawn arguments for each shell choice, shell resolution, environment, PowerShell normalization, tree kill): identical. **Not yet run on Windows** |

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | Guard | Deny list, internal-URL check and `allowPatterns` only with `restrictToWorkspace` | Always | Layer 3/9 decision; the 3 differing scenarios |
| 2 | Violation check | `/` and `~` paths only | Also `C:\`, `C:/`, `\\server\share` | nanobot has the same gap (found in 16.2) |
| 3 | Job Object creation fails | The command fails | Runs without a job; `taskkill` stops it | A restricted Windows account still gets a shell |
| 4 | Sandbox, media folder, web UI loopback, workspace scopes | Yes | No | Layers 18, 23 |
| 5 | User `!cmd` | Web UI only | No | No web UI yet (layer 23) |
| 6 | A child's sessions | Owned by `subagent:<id>` | Owned by the parent chat | No visible effect: each child has its own manager |
| 7 | `exclusive`/`read_only` tool flags | Yes | No | NanoClaw runs tool calls one at a time |

## Tests

Layer 16 adds 287 tests (1,309 → 1,596; 6 Windows-only tests skip on Linux):
`test_exec_session_tools.py`, `test_shell_execution_budgets.py`,
`test_shell_reap.py`, `test_exec_session_startup.py`, `test_exec_env.py`,
`test_shell_output.py`, `test_exec_allow_patterns.py`,
`test_exec_session_isolation.py`, `test_exec_session_capacity.py`,
`test_exec_session_timeout.py`, `test_exec_session_wiring.py`,
`test_exec_config.py`, `test_violations_windows.py`, `test_exec_platform.py`,
adapted from nanobot's. The real-process tests no longer read `/proc`, and
most shell tests now have PowerShell variants instead of Windows skips (list
in [`WINDOWS.md`](../WINDOWS.md)).

### Mutation checks

MUTATION_RESULTS
