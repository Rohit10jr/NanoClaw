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
6. **Compare.** Run nanobot and NanoClaw side by side on the same inputs; fix
   any difference or record it as deliberate; record nanobot bugs in
   `UPSTREAM_BUGS.md`. Break each rule on purpose to confirm a test catches it.
7. **Review.** Push, you review, then the next layer.

## Phase 1 — the core agent

| # | Layer | What it adds | nanobot reference | NanoClaw module | Status |
|---|---|---|---|---|---|
| 0 | Architecture + skeleton | Map of how nanobot works; Python project | `docs/architecture.md`, `.agent/design.md` | `pyproject.toml`, `docs/notes/00` | ✅ done |
| 1 | Provider | One interface for calling a model | `providers/base.py`, `providers/openai_compat_provider.py` | `nanoclaw/providers/` | ✅ done |
| 2 | Runner | The agent loop: call model → run tools → repeat | `agent/runner.py`, `agent/tools/execution.py`, `agent/tools/base.py`, `agent/tools/registry.py` | `nanoclaw/agent/runner.py`, `nanoclaw/agent/tools/` | ✅ done |
| 3 | Tools | Real capabilities: files, shell, web | `agent/tools/filesystem.py`, `shell.py`, `web.py`, `schema.py` | `nanoclaw/agent/tools/` | ✅ done |
| 4 | Context | Building the system prompt | `agent/context.py`, `templates/` | `nanoclaw/agent/context.py`, `nanoclaw/templates/` | ✅ done |
| 5 | Sessions + AgentLoop | Conversations saved and resumed; one turn = context → runner → save | `session/manager.py`, `agent/loop.py` (`process_direct`) | `nanoclaw/session/`, `nanoclaw/agent/loop.py` | ✅ done |
| 6 | Memory | Long-term memory; compacting old history | `agent/memory.py`, `agent/autocompact.py`, `session/summary.py`, `agent/context_governance.py` (compaction) | `nanoclaw/agent/memory.py`, `nanoclaw/session/summary.py`, `nanoclaw/utils/tokens.py` | ✅ done |
| 7 | Skills | Markdown procedures loaded on demand | `agent/skills.py`, `skills/` | `nanoclaw/agent/skills.py`, `nanoclaw/skills/` | ⏭ next |
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

**4 — Context.** `ContextBuilder` assembles the system prompt from Jinja2
templates (runtime, workspace, platform policy, channel format hint, untrusted
content rule) and the editable workspace files (`AGENTS.md`, `SOUL.md`,
`USER.md`; unedited forms skipped), then the tool usage notes; then appends
history and the new user message. `nanoclaw prompt` prints it.
**Decisions:** the current time is added as a runtime-context block on the user
message (nanobot gives no date/time); one workspace folder for everything (the
separate project folder is listed under phase 2). See `notes/04-context.md`.

**5 — Sessions + AgentLoop.** `Session` / `SessionManager` store each
conversation as JSONL in `~/.nanoclaw/sessions/<workspace-id>/` (outside the
workspace, as nanobot requires) and replay it on the next turn. `AgentLoop`
with `process_direct()` ties one turn together: load session → save the user
message → build context → run runner → save new messages. `nanoclaw ask` now
continues the conversation; `--new` starts over, `--session` picks another.
See `notes/05-sessions.md`.
A turn that crashed is closed with an error reply on the next turn.
*Skipped:* the message bus, channels, per-session queues, cancellation,
resuming a crashed turn from a checkpoint, cross-process file locks,
workspace-ID migration.

**6 — Memory.** nanobot's three tiers. `MemoryStore`: `memory/MEMORY.md`
(long-term facts, `# Memory` in the system prompt) and the journal
`memory/history.jsonl`. Compaction (`Consolidator`, `MemoryArchiver`): when a
request would not fit the context window, old messages are summarized by the
model (or dumped raw if that fails) into `[Archived Context Summary]` and the
journal. `nanoclaw compact`, `ask --new` (archives first) and `nanoclaw dream`
(folds journal entries into `MEMORY.md`/`USER.md`/`SOUL.md` with restricted
file tools). Token counting with tiktoken. See `notes/06-memory.md`.
**Decisions:** context window default 128,000 (`NANOCLAW_CONTEXT_WINDOW`;
nanobot: 200,000); tiktoken added; Dream included as a manual command.
**Difference:** compaction is checked once before a turn; nanobot also checks
between tool calls in a turn (phase 2, context governance).
**Found:** compaction usually falls back to a raw dump in nanobot too
(`UPSTREAM_BUGS.md`, bug 4; kept faithful, fix open).
*Deferred (see phase 2):* mid-turn compaction, idle auto-compact, scheduled
Dream, git history of memory files, provider-native compaction, Dream prompt
override, Dream model override, compaction events.

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
| Message bus + channels | `bus/`, `channels/` (Telegram, Slack, Discord, …) | Same agent from many chat apps. A long-running process would also allow **idle auto-compact** (`agent/autocompact.py`: summarize a chat after 15 idle minutes; deferred from layer 6) |
| Hooks | `agent/hook.py`, `agent/turn_hooks.py`, `agent/progress_hook.py` | Plugins and UIs observe or change each step; also **compaction events** (`ContextCompactionEvent`; deferred from layer 6) |
| Plugins | `agent/plugins.py` | Installable bundles of skills + MCP servers |
| Recovery + checkpoints | `session/recovery.py`, runner checkpoint callbacks | Resume a turn after a crash (closing a crashed turn is already ported, layer 5) |
| Mid-turn injections | runner `injection_callback` | Fold new user messages into a running turn |
| Context governance | `agent/context_governance.py` | Keep requests within the context window: token budgets, tool-result budgets, and **compaction in the middle of a turn** (between tool calls). Already ported: the adjacent-user-message merge (layer 5) and compaction before a turn (layer 6) |
| Provider retries | `providers/base.py` `chat_with_retry` | Rate limits, flaky gateways |
| Streaming | `chat_stream`, `cli/stream.py` | Replies appear word by word |
| Model presets + fallback | `agent/model_presets.py`, `providers/fallback_provider.py` | Named model setups, backup models; also the **Dream model override** (run Dream on another model; deferred from layer 6) |
| Anthropic adapter | `providers/anthropic_provider.py` | Non-OpenAI wire format |
| MCP | `agent/tools/mcp.py` | Tools from external MCP servers |
| Cron + heartbeat | `cron/`, `agent/tools/cron.py`, `templates/HEARTBEAT.md` | Scheduled and proactive tasks, including **Dream every 2 hours** (`cli/gateway_runtime.py`, `DreamConfig`; deferred from layer 6, where Dream is the manual `nanoclaw dream`) |
| Usage accounting | `llm_usage/` | Token and cost tracking |
| Web UI / TUI | `webui/`, `tui/` | Browser and full-screen terminal interfaces |
| Memory history (git) | `utils/gitstore.py`, `/dream-log`, `/dream-restore` | Commits `SOUL.md`/`USER.md`/`MEMORY.md` after each Dream so changes can be viewed and undone (deferred from layer 6) |
| Provider conversation state | `providers/conversation_state.py`, `summarize_provider_compaction` | Server-side conversation state (e.g. OpenAI Responses API), including **provider-native compaction** (deferred from layer 6) |
| Prompt overrides | `utils/workspace_prompts.py`, `templates/prompts/`, `/dream-prompt` | Replace Dream's (and the heartbeat's) instructions with a file in the workspace (deferred from layer 6) |
| Logging | `loguru` throughout | Warnings for degraded archives, corrupt files, ... (NanoClaw has no logging yet) |
| Separate project folder | `agent/context.py` (`workspace` vs project), `security/workspace_access.py` | Keep `SOUL.md`/`USER.md`/memory/skills in the agent's home while tools and `AGENTS.md` use a chosen project folder; would add a `--project` CLI option |
