# NanoClaw Roadmap

The plan for rebuilding nanobot's agent core, layer by layer. Each layer is
small enough to read in one sitting, has tests that need no API key, and ends
with a note in [`notes/`](notes/).

- nanobot reference: [HKUDS/nanobot](https://github.com/HKUDS/nanobot) at commit `d3d70f2` (2026-10-01).
- nanobot paths are relative to its `nanobot/` package, except `docs/`, `.agent/`,
  `webui/` and `tui/`, which are at its repo root.
- What is already built, symbol by symbol: [`CODE_MAP.md`](CODE_MAP.md).

## How every layer works

1. **Read.** Study the nanobot files for the layer.
2. **Explain.** Agree in chat on what we build and what we skip.
3. **Build.** Write the minimal version, keeping nanobot's names.
4. **Test.** Use `FakeProvider` and toy tools, with no network.
5. **Document.** Write `docs/notes/NN-<layer>.md` and update `CODE_MAP.md`.
6. **Review.** Push, you review, then the next layer.

## Phase 1 — the core agent

| # | Layer | What it adds | nanobot reference | NanoClaw module | Status |
|---|---|---|---|---|---|
| 0 | Architecture + skeleton | Map of how nanobot works; Python project | `docs/architecture.md`, `.agent/design.md` | `pyproject.toml`, `docs/notes/00` | ✅ done |
| 1 | Provider | One interface for calling a model | `providers/base.py`, `providers/openai_compat_provider.py` | `nanoclaw/providers/` | ✅ done |
| 2 | Runner | The agent loop: call model → run tools → repeat | `agent/runner.py`, `agent/tools/execution.py`, `agent/tools/base.py`, `agent/tools/registry.py` | `nanoclaw/agent/runner.py`, `nanoclaw/agent/tools/` | ✅ done |
| 3 | Tools | Real capabilities: files, shell, web | `agent/tools/filesystem.py`, `shell.py`, `web.py`, `schema.py` | `nanoclaw/agent/tools/` | ✅ done |
| 4 | Context | Building the system prompt | `agent/context.py`, `templates/` | `nanoclaw/agent/context.py`, `nanoclaw/templates/` | ⏭ next |
| 5 | Sessions + AgentLoop | Conversations saved and resumed; one turn = context → runner → save | `session/manager.py`, `agent/loop.py` (`process_direct`) | `nanoclaw/session/`, `nanoclaw/agent/loop.py` | planned |
| 6 | Memory | Long-term memory; compacting old history | `agent/memory.py`, `agent/autocompact.py` | `nanoclaw/agent/memory.py` | planned |
| 7 | Skills | Markdown procedures loaded on demand | `agent/skills.py`, `skills/` | `nanoclaw/agent/skills.py`, `nanoclaw/skills/` | planned |
| 8 | Subagents | Hand a task to a child agent | `agent/subagent.py`, `agent/tools/spawn.py` | `nanoclaw/agent/subagent.py` | planned |
| 9 | Safety | Keep tools inside the workspace; guard shell commands | `security/workspace_access.py`, `security/workspace_policy.py`, `security/network.py`, `agent/tools/sandbox.py` | `nanoclaw/security/` | planned |
| 10 | Terminal chat | Interactive REPL using everything above | `cli/agent.py`, `cli/commands.py` | `nanoclaw/cli.py` | planned |

### Layer details

**0 — Architecture + skeleton.** Message flow: channel → bus → `AgentLoop` →
`AgentRunner` ⇄ provider/tools → reply. The loop/runner split. State on disk.
*Note:* `notes/00-architecture.md`.

**1 — Provider.** `LLMProvider.chat()` returns an `LLMResponse` (text, tool
calls, finish reason, usage). OpenAI message format everywhere. Errors are
returned, not raised. Tool arguments are never guessed.
*Note:* `notes/01-provider.md`.

**2 — Runner.** `AgentRunner.run(AgentRunSpec) -> AgentRunResult`. Iteration
limit; tools only run after a normal stop; one retry on an empty reply; a final
no-tools call when iterations run out; trimming long tool output; tool failures
reported back to the model as text. Also a minimal `Tool` + `ToolRegistry`,
because the loop needs something to call.
*Skipped:* hooks, mid-turn injections, checkpoints, length recovery, parallel
tools, repeated-lookup throttling, streaming.

**3 — Tools.** `read_file`, `write_file`, `edit_file`, `list_dir`, `exec`
(shell with timeout and minimal environment), `web_fetch`, `web_search`
(DuckDuckGo). Full parameter schema validation in the registry.
**Decision:** the `exec` deny list is applied to every command, unlike nanobot,
which only applies it when workspace restriction is on (off by default).
See `notes/03-tools.md`, "Differences from nanobot".
*Skipped:* `apply_patch`, ripgrep search, MCP, cron, image generation,
long-running shell sessions, PDF/Office/image reading.

**4 — Context.** `ContextBuilder` assembles the system prompt from identity
(`SOUL.md`, `USER.md`), project instructions (`AGENTS.md`), runtime info
(time, OS, workspace) and the tool list; then appends history and the new user
message.

**5 — Sessions + AgentLoop.** `Session` / `SessionManager` stores each
conversation as JSONL and replays recent messages. A small `AgentLoop` with
`process_direct()` ties one turn together: load session → build context → run
runner → save new messages.
*Skipped:* the message bus, channels, per-session queues, cancellation, the
workspace-ID scheme.

**6 — Memory.** `MemoryStore` (`memory/MEMORY.md` + `history.jsonl`), a memory
tool or prompt section, and compaction: when the session grows too large,
summarize old turns.
*Skipped:* "Dream" scheduled consolidation, provider-native compaction.

**7 — Skills.** `SkillsLoader` lists `skills/*/SKILL.md` (name + description
in the prompt); the model reads the full skill with `read_file` when needed.
One or two bundled example skills.

**8 — Subagents.** `SubagentManager` + `spawn` tool: run a child `AgentRunner`
with its own task and a restricted toolset; return its result to the parent.

**9 — Safety.** `restrict_to_workspace`: path checks so file tools and
`exec` stay in the workspace; blocking private-network addresses in
`web_fetch`. (The deny list and timeout already exist from layer 3.)
**To revisit here:** keep the deny list always on, or switch to nanobot's
behaviour of applying it only in restricted mode.
*Skipped:* bubblewrap / macOS seatbelt sandboxing (maybe later).

**10 — Terminal chat.** `nanoclaw chat`: REPL with session resume, `/new`,
`/exit`, showing tool calls as they happen. `nanoclaw ask` stays for one-shot
use.

## Phase 2 — review production features

After phase 1 works end to end, we look at each of these, decide whether
NanoClaw needs it, and build only the ones we pick.

| Feature | nanobot reference | Why it exists |
|---|---|---|
| Message bus + channels | `bus/`, `channels/` (Telegram, Slack, Discord, …) | Same agent from many chat apps |
| Hooks | `agent/hook.py`, `agent/turn_hooks.py`, `agent/progress_hook.py` | Plugins and UIs observe or change each step |
| Plugins | `agent/plugins.py` | Installable bundles of skills + MCP servers |
| Recovery + checkpoints | `session/recovery.py`, runner checkpoint callbacks | Resume a turn after a crash |
| Mid-turn injections | runner `injection_callback` | Fold new user messages into a running turn |
| Context governance | `agent/context_governance.py` | Keep requests within the context window |
| Provider retries | `providers/base.py` `chat_with_retry` | Rate limits, flaky gateways |
| Streaming | `chat_stream`, `cli/stream.py` | Replies appear word by word |
| Model presets + fallback | `agent/model_presets.py`, `providers/fallback_provider.py` | Named model setups, backup models |
| Anthropic adapter | `providers/anthropic_provider.py` | Non-OpenAI wire format |
| MCP | `agent/tools/mcp.py` | Tools from external MCP servers |
| Cron + heartbeat | `cron/`, `agent/tools/cron.py`, `templates/HEARTBEAT.md` | Scheduled and proactive tasks |
| Usage accounting | `llm_usage/` | Token and cost tracking |
| Web UI / TUI | `webui/`, `tui/` | Browser and full-screen terminal interfaces |
