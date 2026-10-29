# 18 — Isolation: bubblewrap sandbox and a separate project folder

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package.

## The concept

Two ways to keep the agent's work contained:

- **An OS sandbox for shell commands.** Layer 9's guards (deny list,
  private-network check, workspace restriction) look at the command text, so
  `python -c "..."` or a shell trick can still get around them. nanobot can
  run every `exec` (and `rg`) inside **bubblewrap**: the command sees system
  folders read-only, the project read-write, and nothing else of the home
  folder (sessions, config, keys).
- **A project folder separate from the agent's own workspace.** The workspace
  holds the agent's memory, skills, `SOUL.md` and `USER.md`; until now the
  file tools and the shell also worked there. With a **workspace scope** the
  tools work on another folder (a code repository) while the agent's own files
  stay where they are. In nanobot the project is chosen in the web UI; NanoClaw
  takes `--project DIR`.

```
                agent workspace (~/.nanoclaw/workspace)      project (--project ~/code/app)
system prompt:  SOUL.md, USER.md, memory, skills             AGENTS.md, "# Current Project"
tools:          (skills/ and memory/history.jsonl readable)  read_file, write_file, edit_file, apply_patch,
                                                             list_dir, exec cwd, rg cwd, sandbox root
```

## Sandbox (18.1)

| Piece | What it does | nanobot |
|---|---|---|
| `wrap_command`, `_bwrap` | `bwrap --new-session --die-with-parent`, `HOME` set to the project; `/usr` and the usual system paths read-only; fresh `/proc`, `/dev`, `/tmp`; an empty tmpfs over the project's parent (hides neighbouring folders); the project read-write; `sandboxRoBinds`/`sandboxRwBinds` (binds covering the project or its parents are dropped); `--chdir`; `sh -c`. Network is not cut off | `agent/tools/sandbox.py` |
| `exec`/`rg` wrapping | `tools.exec.sandbox: "bwrap"` wraps string commands and `rg`'s argument list; skipped with a warning on Windows | `agent/tools/shell.py:437-470` |
| Forced restriction | A configured sandbox turns on the workspace restriction for `exec` and the file tools, also under a project scope; extra binds count as allowed roots for `exec`'s path check only | `shell.py:1175`, `filesystem.py:83`, `workspace_access.py:357` |
| Prompt | The per-turn access block adds "Shell sandbox configured; extra binds do not grant file-tool access." | `agent/runtime_control.py:203` |

```json
{ "tools": { "exec": { "sandbox": "bwrap", "sandboxRoBinds": ["~/.local/bin"], "sandboxRwBinds": [] } } }
```

## Project folder (18.2)

| Piece | What it does | nanobot |
|---|---|---|
| `WorkspaceScope`, `bind_workspace_scope`, `current_tool_workspace` | The turn's project and access mode, bound for the turn and read by every tool; subagents inherit it; Dream stays on the workspace | `security/workspace_access.py`, `agent/loop.py:1293`, `agent/subagent.py:824` |
| File tools | Relative paths resolve against the project; restricted mode keeps paths inside it, plus reads of the built-in and workspace `skills/` and `memory/history.jsonl` | `agent/tools/filesystem.py:85-95` |
| `exec`, `rg` | Default working folder: the project | `shell.py:388` |
| System prompt | `AGENTS.md` from the project; `SOUL.md`, `USER.md`, memory and skills from the workspace (absolute paths); `# Current Project` when the two differ | `agent/context.py:105` |
| Runtime block `workspace_access` | Every user turn: "Current project (JSON path): …" and the file-access mode | `runtime_control.py:186` |
| CLI | `--project`/`-p DIR` on `chat`, `ask`, `prompt`; `NANOCLAW_PROJECT`; checked before anything is created | — (web UI in nanobot) |

```bash
uv run nanoclaw chat --project ~/code/app             # work on ~/code/app; memory stays in the workspace
uv run nanoclaw ask -p . --restrict "fix the failing test"
```

## How it was built

Two tasks in parallel. At integration, one conflict in `exec`'s command
preparation, and one real gap between the two: under a bound project scope the
tools take their restriction from the scope, so the sandbox's forced
restriction (passed by 18.1 as the tools' own flag) was lost. Fixed by passing
`sandbox_restricts_workspace` to `current_tool_workspace`, as nanobot does,
with tests that fail without it; the sandbox line added to the runtime block.

| Task | Checked against nanobot |
|---|---|
| 18.1 Sandbox | 62 cases (43 bwrap command lines for many workspace and bind layouts, 19 wrapped `exec`/`rg` commands): 60 identical, 2 deliberate (no macOS seatbelt); bubblewrap 0.9.0 installed in the container and live tests run; re-run after integration |
| 18.2 Project folder | 564 checks on the same layouts (prompt sections and sources, runtime block in 12 modes, 496 file-tool allow/deny decisions, `exec`/`rg` working folders): identical; re-run after integration |

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | macOS sandbox (`seatbelt`) | Yes | No ("Unknown sandbox backend") | Decision 11: bubblewrap only |
| 2 | Media folder bind | Read-only bind | None | No media folder |
| 3 | Choosing the project | Per chat in the web UI, saved in session metadata | One setting per run (`--project`, `NANOCLAW_PROJECT`), not saved | No web UI; nanobot's CLI never has a project either |
| 4 | Dream and the project | Never sees a project (only web UI turns have one) | Explicitly kept on the workspace | Same result |

## Tests

Layer 18 adds 115 tests (1,986 → 2,101): `tests/test_sandbox.py` (nanobot's
bwrap, unknown-backend and platform tests, bind roots, forced restriction,
live runs, sandbox under a project scope) and `tests/test_workspace_scope.py`
(scopes, file tools, prompt, runtime block, subagents, CLI, an end-to-end
`ask --project`).

### Mutation checks

214 rules were broken on purpose:

| Area | Rules | Caught at once | Gaps closed with tests | Equivalent | Result |
|---|---|---|---|---|---|
| Sandbox and its wiring | 76 | 59 | 16 | 1 | 75 caught |
| Workspace scope | 59 | 37 | 22 | 0 | 59 caught |
| Project wiring (file tools, prompt, loop, subagents, CLI) | 79 | 58 | 14 | 7 | 72 caught |

26 new tests close the gaps, for example: `$VAR`/`~` expansion of bind paths
and a relative bind becoming an absolute mount; the full list of optional
system mounts; `edit_file`, `apply_patch` and `list_dir` keeping the
sandbox's restriction; scope validation (NUL in the path, wrong types); the
host-sandbox environment markers; both bracket escapes in the runtime block;
Dream keeping its restriction and never seeing `# Current Project`;
subagents binding the given scope without a parent turn; `chat -p`.

Equivalent (8): values that a later check drops anyway (a blank bind path
becomes `.`, which is relative), extra read roots never consulted when
unrestricted, scopes that always equal the loop's on the paths concerned, and
a context reset in a task that ends right after. The split `wait_for` test
failed four times; it now waits for a file the test creates instead of a
sleep (the real cause, found in layer 20: its text "ab" sometimes appeared in
the random hex session id).

