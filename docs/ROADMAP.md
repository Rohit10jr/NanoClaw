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
| 7 | Skills | Markdown procedures loaded on demand | `agent/skills.py`, `skills/` | `nanoclaw/agent/skills.py`, `nanoclaw/skills/` | ✅ done |
| 8 | Subagents | Hand a task to a child agent | `agent/subagent.py`, `agent/tools/spawn.py`, `agent/tools/context.py` | `nanoclaw/agent/subagent.py`, `nanoclaw/agent/tools/spawn.py` | ✅ done |
| 9 | Safety | Keep tools inside the workspace; block private-network addresses | `security/workspace_policy.py`, `security/network.py`, `agent/tools/execution.py` (violations), `agent/tools/shell.py` (guard) | `nanoclaw/security/`, `nanoclaw/agent/tools/violations.py` | ✅ done |
| 10 | Terminal chat | Interactive REPL using everything above | `cli/agent.py`, `cli/commands.py` | `nanoclaw/cli.py` | ⏭ next |

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

**7 — Skills.** `SkillsLoader` lists `skills/*/SKILL.md` from the workspace
and the built-in folder (workspace wins by name) as one line each in the
`# Skills` section; the model reads a skill with `read_file` (built-in skills
are found through a `skills/` fallback). Requirements (`bins`, `env`) mark a
skill unavailable; `always: true` puts it in `# Active Skills`; `$name` in a
message loads it for that turn. Dream may now write skills. `nanoclaw skills`.
See `notes/07-skills.md`.
**Decisions:** 6 built-in skills (`skill-creator`, `memory`, `weather`,
`github`, `tmux`, `summarize`); PyYAML added; `nanoclaw skills` command.
**Result:** the whole system prompt is now identical to nanobot's (apart from
the two layer 4 differences).
*Deferred (see phase 2):* plugin skills, CLI-app aliases, disabling skills,
ClawHub; the other 5 nanobot built-in skills.

**8 — Subagents.** `SubagentManager` + the `spawn` tool (main agent only): a
child agent with its own short prompt, the task as its only message and the 7
standard tools (no `spawn`). `wait=true` returns the child's answer as the tool
result; otherwise it runs in the background and its report becomes a follow-up
turn (saved as an assistant message, given to the model as new input). At most
4 children at once. See `notes/08-subagents.md`.
**Decisions:** `nanoclaw ask` waits for background children and reports their
results (nanobot's one-shot mode cancels them); iteration limit stays 20 for
the main agent and children (nanobot: 200) for now, to be updated later.
**Differences:** a report arriving during a main turn waits for the next turn
(no mid-turn injection); a queue instead of the message bus; several `spawn`
calls in one reply run one after another (no parallel tools).
**Found:** a repeated report that triggers compaction puts nanobot's summary
marker one message too early (`UPSTREAM_BUGS.md`, bug 5; correct in NanoClaw).
*Deferred (see phase 2):* live status and logging of children (hooks),
compaction inside a child's run, project-folder scope, shared background
shell sessions, usage attribution, stop commands (layer 10).

**9 — Safety.** Private-network (SSRF) guard, always on: `web_fetch` resolves
every host and redirect target and refuses private/internal addresses, with
DNS pinning and proxy support; `exec` refuses commands with internal URLs.
Workspace restriction, opt-in (`--restrict`, `NANOCLAW_RESTRICT_TO_WORKSPACE`):
file tools and `exec` paths stay inside the workspace (reads may also use the
skills folders and the journal), for subagents too. Blocked calls are reported
as nanobot does (a fixed note for private-network blocks; a firmer message after
repeated attempts at the same outside path). See `notes/09-safety.md`.
**Decisions:** deny list stays always on (layer 3 question settled); restriction
off by default, as in nanobot; OS sandbox in phase 2.
**Differences:** the `exec` internal-URL check also runs unrestricted (nanobot:
restricted only); settings come from environment variables and `--restrict`.
*Deferred (see phase 2):* OS sandbox, allow patterns, extra allowed folders,
configured fetch proxy, Web UI loopback access, MCP SSRF checks.

**10 — Terminal chat.** `nanoclaw chat`: REPL with session resume, `/new`,
`/exit`, showing tool calls as they happen. `nanoclaw ask` stays for one-shot
use. Background subagent reports appear as they finish; `/stop` cancels a
conversation's subagents (`cancel_by_session`, built in layer 8).

## Phase 2 — review production features

After phase 1 works end to end, we look at each of these, decide whether
NanoClaw needs it, and build only the ones we pick.

| Feature | nanobot reference | Why it exists |
|---|---|---|
| Message bus + channels | `bus/`, `channels/` (Telegram, Slack, Discord, …) | Same agent from many chat apps. A long-running process would also allow **idle auto-compact** (`agent/autocompact.py`: summarize a chat after 15 idle minutes; deferred from layer 6). Subagent reports travel on the bus in nanobot (NanoClaw: a queue, layer 8) |
| Hooks | `agent/hook.py`, `agent/turn_hooks.py`, `agent/progress_hook.py` | Plugins and UIs observe or change each step; also **compaction events** (`ContextCompactionEvent`; deferred from layer 6) and **live subagent status** (`_SubagentHook`: each child's iteration, tool calls and token use, and a log line per child tool call; deferred from layer 8) |
| Plugins | `agent/plugins.py` | Installable bundles of skills + MCP servers (plugin skills appear as an "Agent Plugin skills" group; deferred from layer 7) |
| Recovery + checkpoints | `session/recovery.py`, runner checkpoint callbacks | Resume a turn after a crash (closing a crashed turn is already ported, layer 5) |
| Mid-turn injections | runner `injection_callback` | Fold new user messages into a running turn; also **subagent reports that arrive during a turn** (NanoClaw runs them after the turn; layer 8 difference #3) |
| Context governance | `agent/context_governance.py` | Keep requests within the context window: token budgets, tool-result budgets, and **compaction in the middle of a turn** (between tool calls), including **inside a subagent's run** (deferred from layer 8). Already ported: the adjacent-user-message merge (layer 5) and compaction before a turn (layer 6) |
| Provider retries | `providers/base.py` `chat_with_retry` | Rate limits, flaky gateways |
| Streaming | `chat_stream`, `cli/stream.py` | Replies appear word by word |
| Model presets + fallback | `agent/model_presets.py`, `providers/fallback_provider.py` | Named model setups, backup models; also the **Dream model override** (run Dream on another model; deferred from layer 6) |
| Anthropic adapter | `providers/anthropic_provider.py` | Non-OpenAI wire format |
| MCP | `agent/tools/mcp.py` | Tools from external MCP servers |
| Cron + heartbeat | `cron/`, `agent/tools/cron.py`, `templates/HEARTBEAT.md` | Scheduled and proactive tasks, including **Dream every 2 hours** (`cli/gateway_runtime.py`, `DreamConfig`; deferred from layer 6, where Dream is the manual `nanoclaw dream`) |
| Usage accounting | `llm_usage/` | Token and cost tracking (including a subagent's tokens counted under the conversation that started it; deferred from layer 8) |
| Web UI / TUI | `webui/`, `tui/` | Browser and full-screen terminal interfaces |
| Memory history (git) | `utils/gitstore.py`, `/dream-log`, `/dream-restore` | Commits `SOUL.md`/`USER.md`/`MEMORY.md` after each Dream so changes can be viewed and undone (deferred from layer 6) |
| Provider conversation state | `providers/conversation_state.py`, `summarize_provider_compaction` | Server-side conversation state (e.g. OpenAI Responses API), including **provider-native compaction** (deferred from layer 6) and adding subagent reports to it (deferred from layer 8) |
| Prompt overrides | `utils/workspace_prompts.py`, `templates/prompts/`, `/dream-prompt` | Replace Dream's (and the heartbeat's) instructions with a file in the workspace (deferred from layer 6) |
| More skills features | `skills/clawhub`, `disabled_skills` config, `apps/cli` aliases, nanobot's `cron`/`image-generation`/`my`/`update-setup` skills | Install community skills (after layer 9), hide skills, CLI-app aliases, and the built-in skills that need features NanoClaw lacks (deferred from layer 7) |
| OS sandbox for `exec` | `agent/tools/sandbox.py` (bubblewrap, macOS seatbelt) | Commands run in an isolated filesystem view with only the workspace writable; the text guards of layer 9 are not a security boundary (deferred from layer 9) |
| Configuration file | `config/schema.py`, `config/loader.py` | One place for settings now spread over environment variables: SSRF whitelist, restriction, `exec` allow patterns, extra allowed folders, a fetch proxy (deferred from layer 9), disabled skills (layer 7) |
| Logging | `loguru` throughout | Warnings for degraded archives, corrupt files, ... (NanoClaw has no logging yet) |
| Separate project folder | `agent/context.py` (`workspace` vs project), `security/workspace_access.py` | Keep `SOUL.md`/`USER.md`/memory/skills in the agent's home while tools and `AGENTS.md` use a chosen project folder; would add a `--project` CLI option; subagents would then run in the project folder (`workspace_scope`; deferred from layer 8) |
| Long-running shell sessions | `agent/tools/exec_session.py`, `ExecSessionManager` | Background shell commands the agent checks on later, shared with subagents (deferred from layers 3 and 8) |
| Iteration limit | `config/schema.py` `max_tool_iterations` (200) | NanoClaw uses 20 for the main agent and subagents; to be revisited (layer 8 decision) |
