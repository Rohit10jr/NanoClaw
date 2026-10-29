# 09 — Safety

Reference: nanobot at commit `d3d70f2`. Paths are relative to its `nanobot/` package.

## The concept

The agent's tools can read and write files, run shell commands and fetch web
pages. "Safety" here means **limits on what those tools can reach**. nanobot
has two kinds, and NanoClaw now has both:

| Kind | What it stops | nanobot | NanoClaw |
|---|---|---|---|
| **1. Private-network guard (SSRF)** | A web page or message tricking the agent into fetching an internal address: your router's admin page, a local service on `localhost`, or the cloud metadata service (`169.254.169.254`), which can hand out credentials | Always on for `web_fetch`; for `exec` only in restricted mode | Always on for `web_fetch` **and** `exec` (difference #2) |
| **2. Workspace restriction** (`restrict_to_workspace`) | File tools and shell commands reaching files outside the workspace folder | **Off by default**; turned on in the config | Off by default; turned on with `--restrict` or `NANOCLAW_RESTRICT_TO_WORKSPACE=1` |

Both are **application-level guards**: they check paths, commands and
addresses before a tool runs. They are not an operating-system sandbox; a
determined program can still find ways around text checks (nanobot says the
same in `security/workspace_policy.py`). Layer 18 added the real boundary for
shell commands: an optional **bubblewrap sandbox** (section 3).

## 1. Private-network guard

| Rule | nanobot | NanoClaw |
|---|---|---|
| Blocked ranges: `0.0.0.0/8`, `10/8`, `100.64/10` (carrier NAT), `127/8`, `169.254/16` (link-local, cloud metadata), `172.16/12`, `192.168/16`, `::`, `::1`, `fc00::/7`, `fe80::/10`. IPv6 forms of IPv4 addresses (`::ffff:127.0.0.1`) are checked as IPv4 | `security/network.py:16`, `_normalize_addr` `:56` | Same |
| The host is **resolved** and every address checked; one private address is enough to block. Only `http`/`https`; a host that can't be resolved is blocked | `resolve_url_target` `:78` | Same |
| **DNS pinning**: the connection uses exactly the addresses that were checked. Otherwise a hostile DNS server could answer "public" for the check and `127.0.0.1` for the connection (DNS rebinding) | `PinnedDNSAsyncTransport` `:264`, `pin_resolved_url_dns` `:215` | Same |
| Every **redirect** target is checked before it is requested | `web.py:208` `_get_with_safe_redirects` | Same |
| Proxy environment variables (`HTTPS_PROXY`, `NO_PROXY`, …) are honoured | `httpx_env_proxy_mounts` `:174`, `web.py:133` | Same |
| `exec`: a command containing a URL to a private address is blocked (`curl http://169.254.169.254/…`) | `contains_internal_url` `:323`, called in `_guard_command` (`shell.py:869`) | Same check, but always on (difference #2) |
| Trusted private ranges (e.g. Tailscale `100.64.0.0/10`) can be allowed | config `tools.ssrfWhitelist` | `NANOCLAW_SSRF_WHITELIST=100.64.0.0/10,…` (difference #3) |

Error texts are nanobot's, for example:

```
{"error": "URL validation failed: Blocked: metadata.test resolves to private/internal address 169.254.169.254", "url": "..."}
{"error": "Redirect blocked: Blocked: 127.0.0.1 resolves to private/internal address 127.0.0.1", "url": "..."}
Error: Command blocked by safety guard (internal/private URL detected)
```

## 2. Workspace restriction

Turn it on per call or for the whole shell:

```bash
uv run nanoclaw ask --restrict "tidy up the notes folder"
export NANOCLAW_RESTRICT_TO_WORKSPACE=1     # 1/true/yes/on
```

| Tool | Rule | nanobot |
|---|---|---|
| `read_file`, `list_dir` | The path, with symlinks resolved, must be inside the workspace. Also allowed: the built-in skills folder, the workspace `skills/` folder and `memory/history.jsonl` | `_FsTool.create` (`filesystem.py:84`), `_resolve_read` `:151`, `resolve_allowed_path` (`workspace_policy.py:96`) |
| `write_file`, `edit_file` | Inside the workspace only (no exceptions) | `_resolve_write` `:182` |
| `exec` | `working_dir` must be inside the workspace | `shell.py:424` |
| `exec` | Blocked: `../` anywhere in the command; absolute and `~` paths outside the workspace (including paths inside `sh -c '…'`, `file://` URLs and `X=/path` assignments); `~-` and `~N` (the shell's directory stack) always. Allowed: `/dev/null`, `/dev/stderr` and other harmless devices | `_guard_command` `:840`, `_extract_absolute_paths` `:1032` |
| Subagents | Get the same restriction as the main agent | `subagent.py:218` `_subagent_tools_config` |

Errors, as in nanobot:

```
Error: Path /etc/passwd is outside allowed directory /home/me/ws (this is a hard policy boundary, not a transient failure; do not retry with shell tricks or alternative tools, and ask the user how to proceed if the resource is genuinely required)
Error: Command blocked by safety guard (path outside working dir)

Note: this is a hard policy boundary, not a transient failure. Do NOT retry with shell tricks (...)
```

Dream's tools were already limited to its own files (layer 6); this setting doesn't change them.

## 3. OS sandbox: bubblewrap (layer 18, task 18.1)

Set `tools.exec.sandbox` to `"bwrap"` in `~/.nanoclaw/config.json` (Linux,
needs the `bwrap` program and user namespaces). Every `exec` and `rg` command is
then run as `bwrap ... -- sh -c '<command>'` (`agent/tools/sandbox.py`, a port
of nanobot's `_bwrap` and `wrap_command`):

| Inside the sandbox | bwrap argument | nanobot |
|---|---|---|
| `/usr` read-only; `/bin`, `/lib`, `/lib64`, `/etc/alternatives`, certificates, `/etc/resolv.conf`, `/etc/ld.so.cache` read-only if they exist | `--ro-bind`, `--ro-bind-try` | `sandbox.py:71` |
| Fresh `/proc`, `/dev`, an empty private `/tmp` | `--proc`, `--dev`, `--tmpfs /tmp` | same |
| The workspace's **parent** folder replaced by an empty tmpfs (in nanobot's layout it holds `config.json`), then the workspace mounted back **read-write** | `--tmpfs <parent>`, `--dir`, `--bind <ws> <ws>` | same |
| `HOME` is the workspace; the command starts in `working_dir`, or the workspace when that is outside it | `--setenv HOME`, `--chdir` | same |
| `tools.exec.sandboxRoBinds` / `sandboxRwBinds`: extra folders, read-only / read-write. Relative or blank entries, duplicates, and the workspace or any of its parents are dropped (a bind there would undo the hidden parent); missing folders are skipped by bwrap (`-try`) | `--ro-bind-try`, `--bind-try` | `_normalize_bind_paths` |
| Everything else (`/home`, `/root`, `/var`, ...) does not exist | — | same |
| Network: not isolated | (no `--unshare-net`) | same |

What else the setting changes, as in nanobot:

- **It forces workspace restriction**, even without `--restrict`: `exec` gets
  the `working_dir` and path checks of section 2 (nanobot
  `sandbox_restricts_workspace`, `shell.py:391`), and the file tools stay inside
  the workspace (nanobot `_FsTool.create`, `filesystem.py:83`). Subagents get
  the same, since they share the exec settings.
- **Bind folders count as allowed in `exec`'s path check** (nanobot
  `_active_sandbox_bind_roots`, `shell.py:1175`), so `~/.local/bin/uv` works
  with that folder in `sandboxRoBinds`. A bind that contains the workspace is
  ignored there too. Binds never grant **file-tool** access.
- `rg`'s argument list is joined, wrapped and split back, so it still runs
  without a shell around `bwrap` (`shell.py:445`).
- On Windows the setting is ignored with nanobot's warning ("Sandbox 'bwrap'
  is not supported on Windows; running unsandboxed"); restriction is still forced.
- Only `"bwrap"` is accepted. Any other name, `"seatbelt"` included, fails the
  command with nanobot's error for an unknown backend:
  `Unknown sandbox backend 'seatbelt'. Available: ['bwrap']`.

Deliberate differences: no macOS **seatbelt** backend (phase-2 decision 11:
bubblewrap only), and no read-only bind of nanobot's **media folder**
(`~/.nanobot/media`): NanoClaw has none (difference #4). nanobot's `my` tool
also tells the model "Shell sandbox configured; extra binds do not grant
file-tool access." in its workspace-access runtime-context block
(`runtime_control.py:203`); NanoClaw has no such block yet.

Checked against nanobot (`scratchpad/t181/compare.py`): 62 cases (43 bwrap
argument lists over workspace layouts at `/`, under `/tmp`, nested, symlinked,
missing or relative; binds overlapping the workspace or its parents, missing,
relative, duplicated, `~`/`$VAR`, symlinked; cwd inside, outside and escaping;
and 19 sandboxed `exec`/`rg` commands in string and argument-list form):
60 identical, 2 different, both the unknown-backend message (`['bwrap']`
instead of `['bwrap', 'seatbelt']`) and `seatbelt` itself. Tested live with
bubblewrap 0.9.0 (`tests/test_sandbox.py::TestBwrapLive`, skipped where
`bwrap --ro-bind / / true` fails).

## How a blocked call reaches the model

As in nanobot's tool execution (`agent/tools/execution.py:26–60`, `_classify_violation` `:245`):

| Blocked by | What the model gets |
|---|---|
| The private-network guard | The error plus a fixed note: "This is a non-bypassable security boundary. Stop trying to access private/internal URLs. Do not retry with curl, wget, encoded IPs, …", **without** the usual retry hint |
| The workspace boundary | The error with the usual retry hint. From the **third** attempt at the same target in one run (any tool: `read_file /x`, then `exec cat /x` counts as the same), a firmer message: "Error: refusing repeated workspace-bypass attempts. You have tried to access '/x' … 3 times in this turn. … Stop retrying." (`utils/runtime.py:138`, `:173`) |

The count starts again with each run. Dream's own boundary errors use the same wording, so they are throttled the same way.

## Decisions (chosen with the user)

| # | Question | Decision |
|---|---|---|
| 1 | Shell deny list (deferred from layer 3) | **Keep it always on** (nanobot: only in restricted mode) |
| 2 | Restriction default | **Off**, as in nanobot; on with `--restrict` / `NANOCLAW_RESTRICT_TO_WORKSPACE` |
| 3 | OS sandbox | **Phase 2** → layer 18: bubblewrap only (phase-2 decision 11) |

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why | Later |
|---|---|---|---|---|---|
| 1 | Shell deny list | Only in restricted mode (`shell.py:446`), so by default nothing is blocked | Always | Decision 1 (layer 3 difference #1, now settled) | — |
| 2 | `exec` internal-URL check | Inside the same restricted-only guard | Always | Built with the deny list as part of the always-on guard (the plan described it as always on; it turned out to be restricted-only in nanobot while building). Side effect, as in nanobot's restricted mode: a URL whose host can't be resolved also blocks the command | — |
| 3 | Whitelist and restriction settings | Config file (`tools.ssrfWhitelist`, `tools.restrictToWorkspace`); the boundary note says "via tools.ssrfWhitelist" | Environment variables and `--restrict`; the note names `NANOCLAW_SSRF_WHITELIST` | NanoClaw has no config file | Phase 2: configuration |
| 4 | Media folder | Restricted reads (and `exec` paths) may also use nanobot's media folder (`~/.nanobot/media`, chat attachments); the bwrap sandbox mounts it read-only | No exception, no sandbox mount | NanoClaw has no attachments | Phase 2 (with channels) |
| 5 | Requests per fetch | `web_fetch` first checks whether the page is an image, then fetches it again (2 requests) | 1 request | Image detection and Jina Reader were left out in layer 3; the safety checks on each request are the same | — |

## What we left out (for later phases)

| nanobot feature | Where | What it does | Later phase |
|---|---|---|---|
| OS sandbox for `exec` | `agent/tools/sandbox.py` (bubblewrap on Linux, seatbelt on macOS), `sandbox_ro_binds`/`rw_binds` | Runs commands in an isolated filesystem view with only the workspace writable | **bubblewrap done in task 18.1** (section 3); seatbelt left out (phase-2 decision 11) |
| Allow patterns | `exec.allow_patterns` | Commands that bypass the deny list (or an allow-only mode) | **Phase 2: configuration** |
| Extra allowed folders | `extra_read_allowed_dirs`, `extra_write_allowed_dirs` | More folders the file tools may use in restricted mode | **Phase 2: configuration** |
| Local-service access for the Web UI | `allow_loopback`, `webui_allow_local_service_access` | Lets the Web UI's preview reach `localhost` | **Phase 2: Web UI** |
| Trusting a proxy's DNS | `trust_remote_dns` | For image generation behind a trusted proxy | **Phase 2** |
| Configured fetch proxy | `web.proxy` | A proxy just for `web_fetch` (environment proxies already work) | **Phase 2: configuration** |
| Workspace scopes / separate project folder | `security/workspace_access.py` | Per-chat project folders, each with its own restriction | **Done in layer 18 (task 18.2)**: `--project DIR`; restricted mode then means the project, plus reads of the agent's skills and `memory/history.jsonl` |
| SSRF checks for MCP servers | `agent/tools/mcp.py` | Same guard for MCP connections | **Done in task 17.4** (URL check, per-request hook incl. redirects, DNS pinning) |
| Repeated-lookup throttle | `repeated_external_lookup_error` | Stops the model repeating the same search or fetch | **Phase 2** (skipped since layer 2) |

## Files

| File | Contents | nanobot |
|---|---|---|
| `nanoclaw/security/network.py` (new) | Blocked ranges, `resolve_url_target`, `validate_url_target`, whitelist (+ env var), DNS pinning, proxy mounts, `contains_internal_url` | `security/network.py` |
| `nanoclaw/security/workspace_policy.py` (new) | `resolve_allowed_path`, `is_path_within`, `WORKSPACE_BOUNDARY_NOTE` (moved here from `memory.py`) | `security/workspace_policy.py` (copied unchanged) |
| `nanoclaw/agent/tools/violations.py` (new) | How blocked calls are reported; the repeated-violation message | `agent/tools/execution.py`, `utils/runtime.py` |
| `nanoclaw/agent/tools/web.py` | Safe validation, pinned transport, proxy mounts, redirect checks | `agent/tools/web.py` |
| `nanoclaw/agent/tools/shell.py` | `working_dir` check; guard with internal-URL and path checks; path extraction copied from nanobot | `agent/tools/shell.py` |
| `nanoclaw/agent/tools/sandbox.py` (layer 18) | `wrap_command`, `_bwrap`, `_normalize_bind_paths` | `agent/tools/sandbox.py` (seatbelt left out) |
| `nanoclaw/agent/tools/filesystem.py` | `_resolve_read` / `_resolve_write` with the boundary | `agent/tools/filesystem.py`, `path_utils.py` |
| `nanoclaw/agent/runner.py` | Uses `classify_violation`; per-run counts | `agent/tools/execution.py`, `runner.py:385` |
| `agent/tools/__init__.py`, `loop.py`, `subagent.py` | `restrict_to_workspace` passed through | config wiring |
| `nanoclaw/__main__.py` | `--restrict`, `NANOCLAW_RESTRICT_TO_WORKSPACE`, `NANOCLAW_SSRF_WHITELIST` | `config/loader.py:141`, config schema |

## Tests

`tests/test_safety.py` (54 tests), with a fake DNS resolver and no real network
(the `fake_dns` fixture in `tests/conftest.py`, which also clears proxy
variables):

- every blocked range (and the edges of `172.16/12`), IPv6 and IPv4-in-IPv6, unresolvable and non-http URLs, public addresses; the whitelist and its variable;
- `contains_internal_url`; proxy mounts and `NO_PROXY`;
- `web_fetch`: blocked before any request, blocked redirects, DNS that changes between the check and the connection, pinned addresses, proxy settings;
- `exec`: internal URLs blocked even unrestricted; restricted `../`, absolute, `~`, `sh -c`, `file://` and assignment paths; allowed devices and workspace paths; `working_dir`;
- restricted file tools: inside/outside, symlink escapes, the skills and journal exceptions, writes without exceptions;
- the SSRF note, escalation after repeated attempts (across tools, reset each run);
- wiring: loop, subagents, `--restrict`, the environment variables;
- (second review, ported from nanobot's tests) two fetches at the same time each pinned to their own addresses; a rebound name rejected before the connection.

The web tests (`tests/test_web_tools.py`) now use the fake resolver too.

Every rule was checked by breaking it on purpose and confirming a test fails
(57 rules). The first run found five weak tests, which were fixed. It also
showed that one test could write into the package's own skills folder when
restriction was broken: that test now uses a copy, and the mutation script
stops if a package file changes. Two changes cannot fail because they change
nothing today:

- checking a redirect target before the next loop step checks it again (kept as in nanobot, as defence in depth);
- the journal exception for restricted reads: the journal is always inside the workspace while NanoClaw has one workspace folder. It matters once a separate project folder exists (phase 2). *(Layer 18: it does now; `tests/test_workspace_scope.py` covers it.)*

## Review against nanobot

Checked by **running both projects' code** on the same inputs, with the same fake DNS:

| Check | Result |
|---|---|
| 27 URLs (all ranges, IPv6, mapped, decimal/hex IPs, credentials in URL, bad schemes), each with 3 whitelists: `validate_url_target` and `resolve_url_target` | Identical |
| `contains_internal_url` on 8 commands | Identical |
| Proxy mounts for 3 proxy setups | Identical |
| `resolve_allowed_path`: 12 paths (relative, `../`, absolute, symlinks, journal, empty) × 4 settings | Identical (results and error texts) |
| `read_file`, `write_file`, `list_dir`, restricted and not | Identical |
| `exec` guard in restricted mode: 42 hand-written commands and **3,000 random commands** | Identical |
| `exec` `working_dir` in restricted mode | Identical |
| Blocked-call reporting for 12 errors in sequence (SSRF note, escalation counts) | Identical (apart from the whitelist name, difference #3) |
| `web_fetch` end to end: private target, redirects to `127.0.0.1` and to a private name, a normal redirect, a bad scheme, DNS changing between check and connection | Identical results; nanobot makes one extra request for the image check (difference #5) |

No new nanobot bugs were found.

### Second review (after the build)

| Check | How | Result |
|---|---|---|
| Other places nanobot reacts to restriction | Searched nanobot for `restrict_to_workspace` / sandbox status | Only the file tools, `exec`, subagents and nanobot's `my` tool (self-inspection, not ported) use it. No prompt or tool-description changes. The Web UI loopback exception applies only to Web UI turns |
| Order of checks in `read_file`, `write_file`, `edit_file` | Read side by side | Same (device check, then boundary, then built-in skill fallback) |
| **nanobot's own tests run against NanoClaw** | Copied `tests/security/test_security_network.py`, `test_workspace_policy.py` and `tests/tools/test_exec_security.py`, pointed them at NanoClaw's modules (a small adapter for `_guard_command`'s signature) | **112 pass.** All 37 failures test features we left out (workspace scopes, Web UI loopback, sandbox bind folders, proxy-DNS trust, `is_loopback_host`) or our differences #1–#2 (nanobot skips the guard in full-access mode) |
| nanobot's `web_fetch` safety tests | Ported the two that don't need Jina Reader: concurrent pinned fetches, and a rebound name | Both pass; **added to our suite** |
| nanobot's runner safety tests | Read against ours | Already covered (SSRF note, run continues, escalation on repeated `exec cat /outside`) |

Correction made: **a test for the DNS-pinning lock.** Pinning works by
temporarily replacing Python's process-wide resolver, so two fetches at the same
time must take turns (`PinnedDNSAsyncTransport._resolver_lock`). No test
covered that: removing the lock passed the whole suite. The ported nanobot test
now catches it (checked by breaking it on purpose). No code changes were
needed; no new nanobot bugs were found.

