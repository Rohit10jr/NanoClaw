# 25 — Skills and plugins

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package.

The last layer of phase 2 brings NanoClaw's skills up to nanobot's current
loader, adds **Agent Plugins** (packages that bring skills and MCP servers),
and ports nanobot's **`my`** tool, which lets the agent look at (and, if
allowed, adjust) its own runtime settings.

## Skills catch-up (25.1)

| Piece | What it does | nanobot |
|---|---|---|
| `skill_metadata.py` | `BUILTIN_SKILLS_DIR`, `SKILL_FRONTMATTER`, `SKILL_NAME`, `parse_skill_metadata`, `valid_skill_metadata` (re-exported from `skills.py`) | `agent/skill_metadata.py` |
| `SkillsLoader.snapshot()` | One cache per prompt build (skill list, documents, metadata, `which` results); `ContextBuilder` opens it around the system prompt and the messages | `agent/skills.py:32-67` |
| `disabledSkills` | `agents.defaults.disabledSkills`: names left out of the prompt, `$name`, `load_skill`, always-on skills and `/skill`; `nanoclaw skills` marks them `(disabled)`; the web UI skills page lists them and toggles them | `agent/skills.py:48`, `config/schema.py:198` |
| Web UI skills | `/api/webui/skills` with `enabled`, `/api/webui/skills/{name}` (requirements, install options, Markdown), the `skill.update` toggle | `webui/skills_api.py` |
| `cron` skill | nanobot's built-in skill for the cron tool (layer 21) | `skills/cron/SKILL.md` |

The toggle saves `config.json` by editing only `agents.defaults.disabledSkills`
in the stored JSON: unknown sections and keys, key order and `${VAR}`
references stay as they are (nanobot re-saves the whole config through its
schema, which would drop the nanobot-only sections NanoClaw keeps since
layer 12). The `cron` skill's text drops nanobot's `message` tool and "change
the chat in the web UI" instructions, which NanoClaw doesn't have; it says a
task's reply goes to its own chat. Still left out: `image-generation`,
`clawhub`, `nanobot-document`, `update-setup` (features NanoClaw lacks), and
CLI-app skill aliases.

## Agent Plugins (25.2)


An **Agent Plugin** is a folder you drop into `<workspace>/plugins/<name>/`
that brings skills and MCP servers together, in the
[Agent Plugins v1](https://agent-plugins.org) layout:

```
<workspace>/plugins/demo/
├── plugin.json            {"$schema": ".../1.0.0/plugin.schema.json", "name": "demo", ...}
├── skills/<skill>/SKILL.md  ordinary skills (name must match the folder)
└── mcp.json               {"$schema": ".../1.0.0/mcp.schema.json", "mcpServers": {...}}  stdio only
```

Nothing in it is used until you enable it:

```bash
nanoclaw plugins list                 # what is installed, enabled or not, its skills and servers
nanoclaw plugins enable demo          # shows fingerprint, permissions, skills, server commands; asks y/N
nanoclaw plugins enable demo --yes    # no question (scripts)
nanoclaw plugins disable demo
```

Once enabled, its skills appear in the prompt under `### Agent Plugin skills
(`plugins`)` (between workspace and built-in skills; a workspace skill of the
same name wins, a plugin skill beats a built-in one), `read_file` may read
the skill folders even when the tools are restricted to a project, and its
MCP servers start with the next `chat`, `ask` or `gateway`, named `<plugin>`
(one server) or `<plugin>--<server>` (several). A server in your own
`tools.mcpServers` with the same name wins.

## Trust model

Enabling a plugin is a decision to trust that exact package: its skills
become instructions to the agent and its MCP servers run as local programs
with your permissions (any command on `PATH`, or a file inside the package).
`enable` therefore shows what you are approving and records it as a
**fingerprint**: a SHA-256 over every path in the package, every symlink's
target text and every file's bytes. The approval is a marker file
`<config dir>/plugin-data/<sha256(workspace)[:12]>/<name>/enabled` (mode 0600,
directories 0700) holding `{"fingerprint", "root"}`; it lives outside the
workspace, so the agent cannot write it while its tools are restricted
(`--restrict`; unrestricted tools and `exec` can write anywhere you can, this
marker included), and a symlinked `plugin-data` that leads outside the config
folder is refused. Every use
re-computes the fingerprint: if any byte, file or link in the package
changed, or the package moved, the marker is deleted and the plugin is off
until you enable it again (the CLI also refuses if the package changed
between the review it printed and the write). Skill text is taken from the
same bytes that were hashed, so a skill swapped during a check is never
shown to the model. Paths are resolved and must stay inside the package
(`plugins/` inside the workspace, skills inside the package, `./command`
and `cwd` inside the package or its data folder), so a symlink pointing
outside is ignored; two packages claiming the same name are both ignored.
What the fingerprint cannot cover: programs the servers start from `PATH`
(e.g. `python`), files outside the package that a symlink points to (the
link text is hashed, not the target), and a change made after a check and
before a server process starts. Those are part of what you trust when you
enable.

## The pieces

| Piece | What it does | nanobot |
|---|---|---|
| `nanoclaw/agent/plugins.py` | Discovery, manifest validation, fingerprints, markers, skill documents, `read_file` skill dirs (cached, re-verified per package), MCP server translation | `agent/plugins.py` (ported as is) |
| `SkillsLoader._discover_skills`, `build_skills_summary` | Order workspace → plugin → built-in; "Agent Plugin skills" group | `agent/skills.py:123`, `:281` |
| `_FsTool._resolve_read` | Enabled plugin skill folders join the restricted read roots | `agent/tools/filesystem.py:145` |
| `nanoclaw.agent.tools.mcp.configured_servers` | `tools.mcpServers` + plugin servers, for gateway, chat and ask | `agent/tools/mcp.py:1409` |
| `nanoclaw/cli/plugins.py` | `nanoclaw plugins list/enable/disable` | web UI Apps page (`webui/mcp_presets_api.py`, `plugin-<name>` rows) |

## Differences from nanobot

1. **Enabling from the CLI.** nanobot enables plugins only from its web UI
   Apps page, through the MCP presets API (`/api/settings/mcp-presets`),
   which NanoClaw does not have; porting it would mean porting that whole
   API, so the web UI endpoints are left out. The CLI shows the review and
   asks for confirmation.
2. **No hot reload.** nanobot's web UI reloads MCP connections after a
   toggle; NanoClaw resolves plugin servers when `chat`, `ask` or `gateway`
   starts. Skills change from the next turn.
3. **Summary paths** are relative to the resolved `plugins` folder (safer
   when the workspace path contains a symlink).

At integration, plugin skills were wired into 25.1's snapshot exactly as in
nanobot: the texts and metadata captured during the fingerprint check are
stored in the open snapshot, so the approved text is the one in the prompt.

## The `my` tool (25.3)

nanobot's `my` tool has two actions, `check` and `set` (`inspect`/`modify`
still accepted). `check` reads the agent's runtime: model, presets, limits,
workspace, sandbox, tool names, web and exec settings, the current request
(channel, chat, sender) and a small scratchpad. `set` changes in-memory
settings only: the session's model preset (like `/model`), the model and
context window outside a session, the iteration limit (subagents too), the
retry mode, the tool-result limit, and scratchpad values (64 keys, 10 levels).
Nothing writes `config.json`, restarts the process or touches security
settings: the bus, provider, tools, sessions, restriction, channels and every
secret-looking name are blocked with nanobot's messages, and `set` is off
unless `tools.my.allowSet` is true. Main agent only; the `my` skill describes
it. Differences: a custom tool registry is used as given; `web_config` is
built from the registered web tools; nanobot's old `myEnabled`/`mySet` config
names aren't migrated.

## How it was built

Three tasks in parallel. 25.1 was sent back once (the skill toggle had
re-saved the whole config, dropping unknown keys; the cron skill pointed at
tools NanoClaw lacks). Integration: conflicts in the CLI, loop, skills README
and tests (keep both), and the skills loader (plugins wired into the
snapshot, above).

| Task | Checked against nanobot |
|---|---|
| 25.1 Skills | 30 workspace setups × 13 checks (lists, always-on, summaries, `load_skill`, requirements, metadata, `$name`, the prompt's skills part): 390 identical; re-run after integration |
| 25.2 Plugins | 26 cases (discovery and validation, fingerprints, markers and revocation, skill documents and summary groups, MCP server validation and merging): identical; re-run after integration. End to end with real `nanoclaw ask` runs: a plugin's skill and MCP tool appear only after `enable` and disappear when the package changes |
| 25.3 `my` | 80 tool calls (and the description, schema and final loop state): identical; re-run after integration |

## Tests

Layer 25 adds 204 tests (4,207 → 4,411): `test_skills_snapshot_disabled.py`,
`test_agent_plugins.py`, `test_agent_plugins_cli.py`, `test_my_tool.py`.

### Mutation checks

MUTATION_RESULTS
