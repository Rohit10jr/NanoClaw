# NanoClaw

A minimal personal AI agent, rebuilt layer by layer to understand how
[nanobot](https://github.com/HKUDS/nanobot) works.

nanobot is a framework plus agent of about 130k lines, most of it chat-app
integrations, a web UI and many model providers. NanoClaw rebuilds only the
**agent core**: the loop, provider interface, tools, context, sessions, memory,
skills, subagents, safety guards and a terminal chat. It keeps nanobot's
concepts and class names, so the two codebases can be read side by side, and
leaves the production features (multi-channel routing, hooks, plugins,
recovery, governance) for later.

## Status

| # | Layer | Status |
|---|---|---|
| 0 | Architecture recon + project skeleton | done |
| 1 | Provider interface (OpenAI-compatible) | done |
| 2 | Agent loop (runner) | done |
| 3 | Tools (files, shell, web) | done |
| 4 | Context / system prompt | done |
| 5 | Sessions + agent loop | done |
| 6 | Memory + compaction | done |
| 7 | Skills | done |
| 8 | Subagents | done |
| 9 | Safety (workspace limits, shell guards) | done |
| 10 | Terminal chat (CLI) | done |
| 11 | Catch up with upstream nanobot (`c4dac64`) | done |
| 12 | Foundations: config file, logging, retries, token usage | done |
| 13 | Hooks, events, streaming replies | done |
| 14 | Message bus, session workers, `/stop`, injections | done |
| 15 | Crash recovery, compaction before every model request | done |
| 16 | Shell: background exec sessions, Windows PowerShell | done |
| 17 | More tools: `apply_patch`, `rg`, unchanged-file reads, MCP servers | done |
| 18 | Isolation: bubblewrap sandbox, separate project folder (`--project`) | done |
| 19 | Subagents v2: the `subagent` tool (create, send, cancel, check) | done |
| 20 | Providers: registry, model presets, `/model`, Anthropic (Claude), fallback models | done |
| 21 | Runtime: `nanoclaw gateway`, cron, heartbeat, scheduled Dream, idle compaction, memory history | done |
| 22 | Session history for the web UI (wire events, per-chat display history) | done |
| 23 | Web UI backend: nanobot's own web UI works against `nanoclaw gateway` | done |

Phase 1 (the core agent) is complete. Phase 2 builds the production backend
(layers 11–25; progress in [`docs/PHASE2_PROGRESS.md`](docs/PHASE2_PROGRESS.md)).

- [`docs/ROADMAP.md`](docs/ROADMAP.md): every layer in build order, plus the
  production features we'll review afterwards.
- [`docs/CODE_MAP.md`](docs/CODE_MAP.md): everything built so far, mapped
  symbol by symbol to the nanobot code it comes from.
- [`docs/notes/`](docs/notes/): one note per layer explaining the concept, how
  nanobot does it, what we built and what we left out.
- [`docs/UPSTREAM_BUGS.md`](docs/UPSTREAM_BUGS.md): bugs found in nanobot while
  porting it, with reproductions and suggested fixes, for issues/PRs upstream.

## Development

```bash
uv sync
uv run pytest
uv run nanoclaw --version
```

## Working on a project folder

By default the agent works in its workspace (`~/.nanoclaw/workspace`), which
also holds its own files. `nanoclaw chat --project DIR` / `nanoclaw ask
--project DIR "..."` (or `NANOCLAW_PROJECT=DIR`) point it at a code folder
instead, as nanobot's WebUI project picker does: relative paths in the file
tools, `exec`'s and `rg`'s default directory and `AGENTS.md` come from the
project, while memory, skills, `SOUL.md` and `USER.md` stay in the workspace
(Dream only ever edits the workspace). With `--restrict` the tools are kept
inside the project, though they can still read the agent's skills and
`memory/history.jsonl`. `nanoclaw prompt --project DIR` shows the system
prompt with its `# Current Project` section. The choice lasts for one run; it
is not saved with the session.

## Scheduled tasks

Since layer 21 the agent has nanobot's `cron` tool: "remind me every day at
8am" becomes a job in `~/.nanoclaw/workspace/cron/jobs.json`, bound to the chat
session it was created in (`every_seconds`, a cron expression with an IANA
timezone, defaulting to `agents.defaults.timezone`, or a one-time `at`). The
tool also lists, updates, enables, disables and removes jobs and shows their
run history. As in nanobot, jobs fire only while the long-running gateway
process runs; `nanoclaw chat` and `ask` only edit them. When a job is due, its
message runs as a normal turn of its session, and the reply goes back to that
chat.

## Configuration

NanoClaw talks to any OpenAI-compatible API (OpenAI, OpenRouter, DeepSeek,
Groq, Ollama, vLLM, ...):

```bash
export NANOCLAW_API_KEY=sk-...
export NANOCLAW_MODEL=gpt-4o-mini                  # default
export NANOCLAW_API_BASE=https://openrouter.ai/api/v1   # omit for OpenAI
uv run nanoclaw chat                      # chat in the terminal (type exit or Ctrl+C to quit)
uv run nanoclaw ask "My name is Rohit"
uv run nanoclaw ask "What's my name?"     # the conversation continues
uv run nanoclaw ask --new "hello"         # forget it and start over
uv run nanoclaw prompt        # show the system prompt (no key needed)
```

Or put the settings in a config file, `~/.nanoclaw/config.json`, with
nanobot's field names (a nanobot `config.json` loads too; sections NanoClaw
doesn't use are ignored). Environment variables override the file, and flags
override both:

```bash
uv run nanoclaw config init      # write a default config.json (or add missing settings)
uv run nanoclaw config show      # the settings a run would use, API keys masked
uv run nanoclaw config path      # where the file is (-c FILE or NANOCLAW_CONFIG to use another)
```

```json
{
  "agents": { "defaults": { "model": "openai/gpt-4o-mini", "maxToolIterations": 200 } },
  "providers": { "openrouter": { "apiKey": "${OPENROUTER_API_KEY}" } }
}
```

The provider is chosen as nanobot does: `agents.defaults.provider` if set,
else the model's prefix or name (`deepseek-chat` → DeepSeek, `claude-…` →
Anthropic), else the first provider with a key (gateways such as OpenRouter
first). Known providers (OpenRouter, OpenAI, Anthropic, DeepSeek, Gemini,
Groq, Moonshot, Mistral, Ollama, vLLM, LM Studio, …) need only an `apiKey`
(local ones an `apiBase`); any other name under `providers` is a custom
OpenAI-compatible endpoint with its own `apiBase`. `extraHeaders`,
`extraBody` and `extraQuery` are sent with every request.

Model presets name a model and its settings; `agents.defaults.modelPreset`
picks one, `--preset NAME` (on `chat` and `ask`) overrides it for a run, and
`/model NAME` switches the current conversation (`/model` lists them;
`/model default` goes back to `agents.defaults`):

```json
{
  "agents": { "defaults": { "model": "deepseek-chat", "modelPreset": "fast" } },
  "providers": { "deepseek": { "apiKey": "${DEEPSEEK_API_KEY}" },
                 "openrouter": { "apiKey": "${OPENROUTER_API_KEY}" } },
  "modelPresets": {
    "fast": { "model": "deepseek-chat", "maxTokens": 4096, "temperature": 0.3 },
    "deep": { "model": "anthropic/claude-opus-4-5", "provider": "openrouter",
              "reasoningEffort": "high", "contextWindowTokens": 200000 }
  }
}
```

Failed model requests are retried (rate limits, timeouts, server errors; 1, 2,
4 seconds or the server's Retry-After); quota and authentication errors stop
at once. `"providerRetryMode": "persistent"` keeps retrying.

`fallbackModels` lists backup models, tried in order once the model's retries
run out (timeouts, server errors, rate limits, quota and authentication
errors; not content-filter, context-length or other bad-request errors). Each
entry is a preset name or an inline model; omitted limits come from the
active model, and the context budget is the smallest window of the chain.
After 3 failed requests in a row the model is skipped for 60 seconds. The
chat shows `↳ Using fallback model …` when a backup answered:

```json
{ "agents": { "defaults": { "modelPreset": "fast", "fallbackModels": [
    "deep", { "provider": "deepseek", "model": "deepseek-reasoner", "maxTokens": 4096 } ] } } }
``` Add `--logs` to
`chat`, `ask`, `compact` or `dream` to see what the agent does internally.

Conversations are saved in `~/.nanoclaw/sessions/`, outside the workspace, so
the agent's file tools can't edit them. If the process is killed in the middle
of a turn, the next turn starts with the interrupted turn restored (finished
tool results kept, unfinished calls marked as interrupted). Use `--session NAME` to keep several
separate conversations (default `cli:direct`).

Replies stream word by word, with the model's reasoning shown as dim `✻` lines
(as in nanobot; turn lines off with `channels.sendProgress`, `sendToolHints`,
`showReasoning` in the config). `nanoclaw ask` keeps stdout for the reply only.

In the chat, slash commands: `/new` (start over), `/compact`, `/stop` (cancel
the running turn and background subagents; finished work is kept), `/status`,
`/model [preset]`, `/history [n]`, `/dream`, `/dream-log`, `/dream-restore`, `/dream-prompt`, `/evaluator-prompt`, `/skill`, `/help`. While the agent works you see what
it does (`↳ $ ls`, `↳ read notes.txt`), and you can keep typing: a new message
is folded into the running turn, as in nanobot.

Memory, as in nanobot:

```bash
uv run nanoclaw compact     # summarize the conversation so far (also automatic, before any model request that would be too long)
uv run nanoclaw dream       # fold the summaries into long-term memory (memory/MEMORY.md, USER.md, SOUL.md)
export NANOCLAW_CONTEXT_WINDOW=128000   # your model's context window in tokens (default 128000)
export NANOCLAW_COMPACTION=nanobot      # or "fixed": NanoClaw's fix for upstream bug 4 (A/B testing it:
                                        # uv run python scripts/compare_compaction.py --dry-run)
```

Skills, as in nanobot: Markdown procedures in `skills/<name>/SKILL.md`
(your workspace) or built in (`github`, `memory`, `skill-creator`,
`summarize`, `tmux`, `weather`). The agent sees a one-line list and reads a
skill when it needs it; `$name` in a message loads one for that turn.

```bash
uv run nanoclaw skills                                   # list skills
uv run nanoclaw ask 'What is the $weather in Paris?'     # use a skill now
```

Summaries are kept in a journal, `memory/history.jsonl` in the workspace;
`nanoclaw dream` reads the new entries and updates the memory files, which the
agent sees in every system prompt.

Memory history, as in nanobot: the workspace is a small git repository
(dulwich, no `git` program needed) that tracks only `SOUL.md`, `USER.md` and
`memory/MEMORY.md`. Every Dream run (`/dream`, `nanoclaw dream`) commits what
really changed, and its reply names the commit. In the chat, `/dream-log
[sha]` shows a Dream change as a diff, `/dream-restore` lists Dream versions
and `/dream-restore <sha>` undoes one (as a new commit, so it can be undone
too). `/dream-prompt init` copies Dream's instructions to `prompts/dream.md`,
where you can change how Dream organizes memory; delete or empty the file to
go back to the default. A workspace that is already inside a git repository
gets no memory repository.

The gateway, as in nanobot: `nanoclaw gateway` keeps the agent running in the
foreground until Ctrl+C or SIGTERM. It answers `GET /health` on
`gateway.host`:`gateway.port` (default `127.0.0.1:18790`, `--port` to change)
and summarizes conversations idle for `agents.defaults.idleCompactAfterMinutes`
(default 15, 0 = off) in the background; `nanoclaw chat` does that too while
it runs. It also runs the scheduled jobs: your `cron` jobs, Dream every 2 hours
(`agents.defaults.dream`: `intervalH`, a legacy `cron` expression, or
`enabled: false`; each run commits as `dream: periodic memory consolidation`)
and the heartbeat every 30 minutes (`gateway.heartbeat`: `intervalS`,
`enabled`). The heartbeat reads the tasks under `## Active Tasks` in the
workspace's `HEARTBEAT.md`, runs them, and lets a small model call decide
whether the result is worth sending to your most recent chat
(`/evaluator-prompt init` copies that decision's instructions to
`prompts/evaluator.md`). Without a channel nothing reaches the gateway from
outside and the heartbeat has nowhere to deliver (it then does nothing, as
nanobot does with only the CLI).

```bash
uv run nanoclaw gateway            # logs on stderr (--no-logs to hide, -v for debug)
curl http://127.0.0.1:18790/health
```

The web UI: the gateway can serve nanobot's own React web UI (nanobot's
`webui/`, built unchanged with `bun run build`; its output is
`nanobot/web/dist`). Turn it on in `config.json` and point it at the build:

```json
{"channels": {"websocket": {"enabled": true, "webuiDist": "/path/to/nanobot/web/dist"}}}
```

(or set `NANOCLAW_WEBUI_DIST` instead of `webuiDist`), then open
`http://127.0.0.1:8765` (`host`, `port`). One port serves the page, its
`/api` routes and the WebSocket; chats are saved as `websocket:<id>` sessions
and the heartbeat can deliver to them. By default only browsers on this
machine are let in. For remote access set a secret, `"tokenIssueSecret": "..."`
(and `"host": "0.0.0.0"` or a reverse proxy); the page then asks for it once.
The UI shows nanobot's core chat (`webui.core.v1`): streaming replies, tool
activity, history, titles, delete, and a chat's scheduled jobs (the cron
jobs the agent bound to it) in the Session details popover, where they can be
paused, run or removed; deleting such a chat also deletes its jobs, after the
confirm dialog lists them. Settings, skills, apps, editing automations,
channels, attachments and subagent panels are nanobot features NanoClaw does
not serve; those pages show empty or error states.

Subagents, as in nanobot: the agent can hand a task to a child agent with the
`subagent` tool (`action=create`). With `wait=true` it waits for the answer;
otherwise the child works in the background (up to 4 at once) and its report
comes back by itself, so the agent tells you the result when it's done. While
a child runs, the agent can `send` it a message, `cancel` it, or `check` its
status and results (each task belongs to the conversation that started it).
`nanoclaw ask` waits for background children before exiting.

Safety, as in nanobot: `web_fetch` and shell commands can't reach private or
internal network addresses (`localhost`, `192.168.x.x`, cloud metadata, ...),
and a short deny list blocks commands like `rm -rf`. With `--restrict` the file
tools and shell commands also stay inside the workspace:

```bash
uv run nanoclaw ask --restrict "clean up the notes"   # or NANOCLAW_RESTRICT_TO_WORKSPACE=1
export NANOCLAW_SSRF_WHITELIST=100.64.0.0/10          # private ranges you trust (e.g. Tailscale)
```

These are guards, not a sandbox: run the agent somewhere you don't mind it
working, or, on Linux, set `"tools": {"exec": {"sandbox": "bwrap"}}` in the
config to run every shell command inside bubblewrap, where only the workspace
is writable and nothing outside it but system folders is visible (this also
turns on the workspace restriction).

Shell commands, as in nanobot: a command still running after a second keeps
running in the background (up to an hour by default) and the agent checks on
it, sends it input or stops it. `/new`, `/stop` and leaving the chat stop the
commands it started. On Windows commands run in PowerShell (or `cmd`). Limits
and allowed commands go in the config file under `tools.exec` (`maxRuntime`,
`idleTimeout`, `allowPatterns`, `denyPatterns`, `enable`, ...).

MCP servers, as in nanobot, go under `tools.mcpServers`: a `command` (stdio)
or a `url` (streamable HTTP; SSE for URLs ending in `/sse`). Their tools appear
as `mcp_<server>_<tool>`; `enabledTools` limits which, `toolTimeout` (30 s)
bounds each call. Private-network URLs are blocked unless listed in
`tools.ssrfWhitelist`. OAuth servers and `useProxy` are not supported.

```json
{ "tools": { "mcpServers": {
  "fs": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"] },
  "remote": { "url": "https://mcp.example.com/mcp", "headers": { "Authorization": "Bearer ${MCP_TOKEN}" } }
} } }
```

The agent's personality and instructions live in editable Markdown files in
the workspace (`~/.nanoclaw/workspace` by default, change with `-w`):
`SOUL.md` (personality), `USER.md` (about you), `AGENTS.md` (workspace rules).
They are created on first run and never overwritten.
