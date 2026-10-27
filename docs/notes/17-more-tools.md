# 17 — More tools: `apply_patch`, `rg`, file state, MCP

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package.

## The concept

Four additions that make the agent better at working on code, all from
nanobot's current default tool set:

- **`apply_patch`**: nanobot's default edit tool now. Several edits across
  several files in one call, applied all or nothing.
- **`rg`**: ripgrep for searching file contents and listing files, run
  through exec sessions with an argument list instead of a shell command.
- **File state**: when the model reads a file it has already read and the
  file hasn't changed, it gets a one-line note instead of the whole file
  again, as long as the earlier result is still in its context.
- **MCP** (Model Context Protocol): tools from external servers (a database,
  a browser, an issue tracker), configured in the config file.

## The pieces

| Piece | What it does | nanobot |
|---|---|---|
| `ApplyPatchTool` (`apply_patch`) | 1–20 edits, `replace` (one unique match) or `add` (create or append), applied in order to an in-memory copy and then written together; any failure leaves every file as it was; `dry_run`; CRLF kept. Main agent, subagents and Dream (limited to the memory files) | `agent/tools/apply_patch.py` |
| `edit_file` `line_hint`, `expected_replacements` | Pick one of several matches by line number; refuse when the number of matches isn't the expected one. `write_file`/`edit_file`/`list_dir` descriptions updated (they point to `apply_patch`) | `agent/tools/filesystem.py` |
| `RgTool` (`rg`) | `ExecTool` with `args` (one array element per argument) and no shell; registered when exec is enabled and `rg` is on the exec PATH; main agent and subagents | `agent/tools/rg.py` |
| `FileStates`, `file_read_context` | Records each read (path, range, content hash). A repeated read with the same range and bytes returns `[File unchanged since last read: <path>]` only if the earlier result is unchanged in the request the model just answered; compaction, truncation or a write (`write_file`, `edit_file`, `apply_patch`) make the next read return the contents; `force=true` always does. One state per session (cleared by `/new`); subagents and Dream have their own | `agent/tools/file_state.py`, `agent/runner.py` |
| `MCPProvider`, `connect_mcp_servers`, `MCPToolWrapper` | Connect each server in `tools.mcpServers` (stdio when `command` is set, SSE for URLs ending in `/sse`, streamable HTTP otherwise), list its tools (paginated, only if it offers tools), wrap them as `mcp_<server>_<tool>` (64 characters at most, with a hash), call with a timeout (30 s), one retry after a transient error, reconnect after "session terminated"; a server that fails is skipped | `agent/tools/mcp.py` |
| MCP safety | URL checked before connecting and on every request and redirect (private addresses blocked unless in `ssrfWhitelist`); pinned DNS; credentials redacted in logs | `agent/tools/mcp.py`, `security/network.py` |
| Tool order | Built-in tools sorted, then `mcp_*` sorted, cached: connecting a server doesn't change the start of the tool list (prompt cache) | `agent/tools/registry.py:97` |

## Configuration

```json
{
  "tools": {
    "mcpServers": {
      "files": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"] },
      "tracker": { "url": "https://mcp.example.com/mcp", "headers": { "Authorization": "Bearer ${TRACKER_TOKEN}" },
                   "enabledTools": ["search_issues"], "toolTimeout": 60 }
    }
  }
}
```

nanobot's `MCPServerConfig` fields: `type`, `command`, `args`, `env`, `cwd`,
`url`, `headers`, `toolTimeout`, `enabledTools` (`["*"]` by default; raw or
`mcp_…` names), `auth`, `useProxy`. `nanoclaw config show` masks `headers`
and `env` values. `chat` and `ask` connect before the first turn and close the
servers on exit.

## How it was built

Four independent tasks in parallel, each by a sub-agent in its own worktree.
Integration: two conflicts in the tool lists and notes; `apply_patch` wired
into the file state (it records its writes, as nanobot `apply_patch.py:243`);
`rg`'s guard changed to nanobot's rule (below).

| Task | Checked against nanobot |
|---|---|
| 17.1 `apply_patch`, `edit_file` options | 78 cases through both tool registries (47 `apply_patch`, 5 Dream, 26 `edit_file`), comparing results and file bytes: identical; re-run after integration |
| 17.2 `rg` | 45 cases on one workspace (search options, outside paths, truncation, `--pre` and other argument tricks, errors): identical after integration |
| 17.3 File state | 19 scripted multi-step runs through both runners (repeat, modified, touched, ranges, five spellings of one path, writes, compaction removing the earlier read, subagent): identical; re-run after integration |
| 17.4 MCP | One fixture server (8 tools) through both projects: tool definitions 10/10 identical in the same order; calls 19/20 identical, the image tool differs (below) |

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | `apply_patch` result | `FileEditResult` with diffs for activity events | Plain text | No activity events (web UI, layer 23) |
| 2 | Dream's write scope for `apply_patch` | `allowed_dir`, `extra_write_allowed_files` | A `write_check` hook | Same refusals in the same order |
| 3 | `rg` enable | Also needs `tools.file.enable` | Exec enable only | NanoClaw has no file-tools switch |
| 4 | `rg` and `grep`/`find_files` | `rg` replaces them | Added alone | NanoClaw never had them |
| 5 | Provider-side compaction and file state | Tracked | Not needed | No provider-side compaction |
| 6 | MCP images | Saved as artifacts for the `message` tool | nanobot's fallback text "(MCP tool returned an image that could not be stored)" | No artifacts or `message` tool |
| 7 | MCP `useProxy`, `auth: "oauth"` | Proxy routing; OAuth | Warning; proxy ignored, OAuth server skipped | Left out |
| 8 | MCP resources, prompts, hot reload, status | Yes | No | Tools only (decision 12) |

Note on `rg`: the sub-agent first gave `rg` `exec`'s always-on guard
(NanoClaw difference in layers 3 and 9). As `rg` runs no shell, that only
blocked searches for text such as `rm -rf` or `shutdown`; it now runs the
guard only in restricted mode, as nanobot.

## Tests

Layer 17 adds 273 tests (1,634 → 1,907): `test_apply_patch_tool.py`,
`test_edit_file_line_hint.py`, `test_rg_tool.py`, `test_file_state.py`,
`test_mcp_tool.py`, `test_mcp_probe.py`, `test_mcp_transient_retry.py`,
`test_mcp_connection.py`, `test_mcp_reconnect_crash.py`,
`test_mcp_real_server.py` and `test_mcp_wiring.py` (a real FastMCP server in
`tests/fixtures/mcp_demo_server.py`, over stdio and streamable HTTP), adapted
from nanobot's.

### Mutation checks

MUTATION_RESULTS
