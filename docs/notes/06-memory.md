# 06 — Memory and compaction

Reference: nanobot at commit `d3d70f2`. Paths are relative to its `nanobot/` package.

## The concept

A model can only read so much at once: its **context window** (128,000 tokens
for `gpt-4o-mini`). Layer 5 sends the whole conversation every turn, so a long
chat would eventually stop fitting. Layer 6 adds nanobot's three tiers of memory:

| Tier | What | Where it lives | When it changes |
|---|---|---|---|
| 1. Session | The conversation itself (layer 5) | `~/.nanoclaw/sessions/...` | Every turn |
| 2. **Compaction** | Old messages are replaced by a short summary, which goes into the system prompt as `[Archived Context Summary]`. Each summary is also written to a journal | Summary: session metadata. Journal: `memory/history.jsonl` | When a request would not fit; `nanoclaw compact`; `nanoclaw ask --new` |
| 3. **Long-term memory** | Facts worth keeping for weeks: preferences, projects, decisions. Shown in every system prompt as `# Memory` | `memory/MEMORY.md` (also `USER.md`, `SOUL.md`) | `nanoclaw dream` |

```
conversation ──(too long)──► summary ──► history.jsonl ──(nanoclaw dream)──► MEMORY.md
   tier 1                     tier 2        journal                           tier 3
```

The normal agent is told it may **not** edit `MEMORY.md`, `USER.md` or
`SOUL.md` ("Only Dream memory-consolidation tasks may edit ..."). Only Dream
writes them, so long-term memory is curated in one place.

## Compaction

### When

Before each turn, NanoClaw estimates the size of the request it is about to
send (system prompt + history + new message + tool definitions). If it is at or
over the **input budget**, the history is compacted first:

```
input budget = context window − max reply tokens (8,192) − safety margin (1,024)
             = 128,000 − 8,192 − 1,024 = 118,784 tokens by default
```

(nanobot `ContextGovernor.request_pressure` / `input_budget`, `providers/base.py` `CONTEXT_SAFETY_BUFFER`.)

### How

1. The request so far, minus the new message (system prompt + history), is
   sent to the model with nanobot's archive prompt (`templates/agent/consolidator_archive.md`):
   *"Create a compact replacement checkpoint for this session..."*. It asks
   for one fact per line, each marked `[permanent]`, `[durable]`, `[ephemeral]`
   or `[correction]`.
2. The answer is appended to the journal `memory/history.jsonl`.
3. The turn is sent as: system prompt with the summary, then only the new message.
4. When the turn is saved, a hidden marker is inserted where the summary ends,
   `last_archived` moves to it, and the summary is stored in the session
   metadata. Old messages stay in the file; they are just no longer sent.

```
saved session:   [old user] [old assistant] ... [marker] [new user] [assistant]
                  └──────── summarized ───────┘   ↑ last_archived
next request:    [system + summary] [new user] [assistant] [next user]
```

If compaction happens again later, the previous summary is in the system
prompt of the archive request, and the prompt tells the model to update it.

### The raw fallback

If the model can't produce a summary (error, cut-off answer, empty answer,
keeps calling tools), or the archive request itself doesn't fit, a **raw
checkpoint** is used instead: the messages as text,
`[RAW] 4 messages\n[2026-10-01T10:00] USER: ...`, written to the journal in
chunks and cut to a token limit for the system prompt.

**In practice the raw fallback is common.** The archive request contains the
same history as the request that just overflowed, plus the archive prompt
(~400 tokens), and its budget is only 1,024 tokens larger. So the model is only
asked for a summary when the overflow is small (about 600 tokens or less);
beyond that the result is a raw dump. We checked this by running nanobot
itself, and NanoClaw matches it exactly. It is recorded as
[upstream bug 4](../UPSTREAM_BUGS.md#4-compaction-usually-falls-back-to-a-raw-dump).
By default NanoClaw keeps nanobot's behaviour; a fix can be switched on and
compared (next section).

### Compaction A/B: `NANOCLAW_COMPACTION`

| Mode | When the archive request doesn't fit | Raw dump (model failed) keeps |
|---|---|---|
| `nanobot` (default) | Raw dump, exactly like upstream | The oldest part |
| `fixed` | The history is split into parts that each fit one request; each part is summarized; the summaries are journaled oldest first and joined into the checkpoint | The newest part |

When the archive request fits, both modes do the same thing (one model call).
`fixed` costs one extra model call per extra part. If one turn alone is too
big for a request, or the model fails on any part, `fixed` uses the raw dump
(nothing is journaled before that decision, so nothing is written twice).
`fixed` applies to automatic compaction, `nanoclaw compact` and `ask --new`.

**Comparing them** — `scripts/compare_compaction.py` runs the same
conversation in both modes and writes a report:

```bash
uv run python scripts/compare_compaction.py --dry-run          # simulated model, no API
uv run python scripts/compare_compaction.py --runs 3           # real API: NANOCLAW_API_KEY / _API_BASE / _MODEL
uv run python scripts/compare_compaction.py --runs 3 --model gpt-4o-mini --out ab-results
```

The conversation: three facts (early, middle, late), each at a random place in
a stretch of turns where the agent fetches a long log file with a `fetch_log`
tool (3,000 tokens on average, 50–150 %), then a quiz, then Dream. The
window is 12,000 tokens, so it compacts several times. The agent has no other
tools, so it can't look facts up in the journal. Both modes get the same
messages and log files in each run.

Options: `--runs`, `--model`, `--dry-run`, `--window`, `--max-tokens`,
`--filler-tokens`, `--fillers-per-fact`, `--max-dreams`, `--modes`, `--out`.

| Report row | Meaning |
|---|---|
| Model summaries | Compactions where the model wrote the checkpoint (not a raw dump) |
| Quiz recall (early / mid / late / total) | The quiz answer contains the fact |
| Fact still in context at the quiz (per run) | The fact was somewhere in the quiz request; a miss there is the model's, not compaction's |
| In MEMORY.md after Dream | Dream wrote the fact to long-term memory. Dream sees only the first 1,000 characters of each journal entry, so a fact deep in a raw dump can't get there |
| Compaction calls / tokens | The cost of compaction |
| All calls / tokens, errors | Whole runs; errors are model errors plus turns that failed |

Each run folder (`run1/nanobot/`, `run1/fixed/`, ...) has the checkpoints, the
quiz answer, the Dream log, the journal (`workspace/memory/history.jsonl`),
`MEMORY.md` and the session file side by side; `results.json` has the raw
numbers.

Dry run (simulated model that "remembers" exactly what is in its request; 5
runs): model summaries 25 % vs 100 %, quiz recall 7/15 vs 15/15, MEMORY.md
6/15 vs 12/15, compaction calls 3 vs 20. This only shows the mechanics; the
decision waits for a real-model run.

Note: the scenario uses long **tool results** because those trigger the bug.
A long *user* message does not: the archive request leaves the new message
out, so it fits.

### Other triggers

| Command | What it does | nanobot |
|---|---|---|
| `nanoclaw compact` | Summarizes the whole conversation now | `/compact` → `Consolidator.compact_idle_session` |
| `nanoclaw ask --new "..."` | Summarizes the old conversation into the journal (so Dream can still learn from it), then starts over | `/new` → `archive_session` (in the background in nanobot) |

## Long-term memory: Dream

`nanoclaw dream` runs nanobot's Dream job once:

1. Reads up to 20 journal entries Dream hasn't seen yet (`memory/.dream_cursor`).
2. Runs one **temporary** agent turn (nothing saved) with Dream's instructions
   (`templates/agent/dream.md`): which facts go into `SOUL.md`, `USER.md` or
   `memory/MEMORY.md`, how to treat the `[permanent]`/`[ephemeral]`/... marks,
   what to drop.
3. Dream has restricted tools: `read_file` inside the workspace, and
   `edit_file`/`write_file` for those three files only.
4. If the run completed, the cursor moves past the entries; otherwise they are
   read again next time. Old journal entries beyond 1,000 are dropped, but
   never ones Dream hasn't read.

The updated `MEMORY.md` appears in the system prompt of every later turn
(skipped while it is still the unedited template).

## Files in the workspace

| File | Content |
|---|---|
| `memory/MEMORY.md` | Long-term facts (created from a template on first run) |
| `memory/history.jsonl` | The journal: `{"cursor": 1, "timestamp": "2026-10-01 10:00", "content": "...", "session_key": "cli:direct"}` per line |
| `memory/.cursor` | The last journal number used |
| `memory/.dream_cursor` | The last journal number Dream has processed |

## Settings

| Setting | Default | Notes |
|---|---|---|
| `NANOCLAW_CONTEXT_WINDOW` | 128,000 | Tokens. Set it to your model's window; `0` turns automatic compaction off |
| `NANOCLAW_COMPACTION` | `nanobot` | `nanobot` (upstream behaviour) or `fixed` (bug 4 fix, being A/B tested); see "Compaction A/B" |
| Token counting | tiktoken (`cl100k_base`) | Same as nanobot. The vocabulary is downloaded once to `~/.nanoclaw/cache/tiktoken`; without it, UTF-8 bytes are counted (overestimates, so compacts early) |

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why | Later |
|---|---|---|---|---|---|
| 1 | **When compaction is checked** | Before every model request, also in the middle of a turn (between tool calls) | Once, before the turn's first request | The mid-turn version rewrites the request while the runner is working; it is part of nanobot's larger context governance (token budgets, tool-result budgets). For conversations that grow over many turns the result is identical (checked). Only a single turn that overflows by itself is not handled: the request is sent and the model returns an error | Phase 2, "Context governance" |
| 2 | **Context window default** | 200,000 | 128,000, set with `NANOCLAW_CONTEXT_WINDOW` | Our default model `gpt-4o-mini` has 128,000; with 200,000 requests would fail before compaction started. Chosen with the user | — |
| 3 | `/new` archiving | In the background; the reply comes first | Waits for it (one extra model call before the turn) | The CLI exits after the command, which would kill a background task | — |
| 4 | Where the summary is stored when a turn compacts | Committed when the turn is saved | Same, but the compaction runs before the runner rather than inside it | Same saved result (checked) | — |
| 5 | Dream's tools | `read_file`, `edit_file`, `write_file`, `apply_patch`; may also write under `skills/` | `read_file`, `edit_file`, `write_file`; only `MEMORY.md`, `USER.md`, `SOUL.md`. Error text names the allowed files | No `apply_patch` (layer 3); skills arrive in layer 7. General path restriction is layer 9; here a small wrapper (`_PathGuard`) does the check | Layer 7 (skills), layer 9. *(Layer 9: general restriction added for the normal tools (opt-in); Dream keeps its own guard, as in nanobot.)* *(Layer 7: `skills/` writes, built-in skill reads and nanobot's exact error text added; only `apply_patch` still differs)* |
| 6 | Dream's instructions | Include a section on creating skills | Same text without the skills section and table row | Skills arrive in layer 7 | Layer 7 *(done: the full nanobot text is used)* |
| 7 | "No memory changes" in Dream's report | Read from a git diff of the memory files | Compares the three files before and after | No git store (see below) | — |
| 8 | Token counter loading | Loaded in a background thread; byte counts until ready | Loaded on first use | Simpler; same counts once loaded | — |
| 9 | Dream's "nothing to process" message | Points to nanobot settings (`idleCompactAfterMinutes`, `/dream-prompt`) | Points to `nanoclaw compact` and `nanoclaw ask --new` | Those are the ways a NanoClaw conversation reaches the journal | Phase 2 (idle auto-compact, prompt overrides) |

## What we left out (and when it could come)

| nanobot feature | Where | What it does | Why not now | Later phase |
|---|---|---|---|---|
| Compaction in the middle of a turn | `agent/context_governance.py` `_compact_request_history`, `runner.py` | Summarizes while a long tool loop grows, keeping the newest steps | Needs the rest of context governance | **Phase 2: Context governance** |
| Idle auto-compact | `agent/autocompact.py`, `loop.py` `_check_expired_sessions_if_due` | Summarizes a conversation after 15 idle minutes, in the background | Needs an always-running process; nanobot's own one-shot CLI doesn't run it either | **Phase 2: Message bus + channels** (long-running mode) |
| Dream every 2 hours | `cli/gateway_runtime.py` cron job, `DreamConfig` | Runs Dream on a schedule | Same: needs the always-running gateway | **Phase 2: Cron + heartbeat** |
| Git history of memory files | `utils/gitstore.py`, `/dream-log`, `/dream-restore` | Commits `SOUL.md`/`USER.md`/`MEMORY.md` after each Dream, so changes can be viewed and undone | An audit trail, not needed for memory to work; adds a `.git` folder to the workspace | **Phase 2: Memory history (git)** |
| Provider-native compaction | `summarize_provider_compaction`, provider conversation state | Lets APIs with server-side conversation state compact on their side | Our adapter uses the standard chat API | **Phase 2: Provider conversation state** |
| `prompts/dream.md` override | `utils/workspace_prompts.py`, `/dream-prompt` | Replace Dream's instructions with your own file | A customisation option | **Phase 2: Prompt overrides** |
| Dream model override | `DreamConfig.model_override`, `dream_runtime` | Run Dream on another model | We have one model setting | **Phase 2: Model presets + fallback** |
| Compaction events | `ContextCompactionEvent` | "Compaction started/finished" notices for the web UI | Nothing listens yet | **Phase 2: Hooks / Web UI** |
| Legacy `HISTORY.md` migration | `MemoryStore._maybe_migrate_legacy_history` | Converts the journal format of old nanobot versions | NanoClaw has no old installs | Not needed |
| Pruning old Dream sessions | `MemoryStore.prune_dream_sessions` | Deletes saved Dream sessions beyond 10 | Our Dream turns are never saved | Not needed |
| Logging | warnings throughout `memory.py` | Notes degraded archives, corrupt journal lines | No logging yet | Phase 2 |

## Files

| File | Contents |
|---|---|
| `nanoclaw/agent/memory.py` | `MemoryStore` (MEMORY.md, journal, Dream prompt/tools, raw fallback), `MemoryArchiver`, `Consolidator`, `_PathGuard` |
| `nanoclaw/agent/loop.py` | Compaction before a turn (`_compact_if_needed`), `new_session` (archives first), `compact_session`, `run_dream`, ephemeral turns, `ContextWindowExceededError` |
| `nanoclaw/agent/context.py` | `# Memory` and `[Archived Context Summary]` sections; `build_messages` with no new message |
| `nanoclaw/session/manager.py` | `commit_summary_checkpoint`; `get_history` skips the marker, takes `max_tokens`, trims assistant replies |
| `nanoclaw/session/summary.py` | `SUMMARY_CONTINUATION_TEXT`, `is_summary_checkpoint`, `session_summary_from_metadata` |
| `nanoclaw/utils/tokens.py` | `estimate_prompt_tokens`, `estimate_message_tokens`, `truncate_text_to_tokens` |
| `nanoclaw/utils/llm_runtime.py` | `LLMRuntime` (provider, model, max tokens, temperature, window) |
| `nanoclaw/runtime_context.py` | `public_history_message` (removes the time block from saved messages for the raw dump) |
| `nanoclaw/templates/` | `memory/MEMORY.md`, `agent/consolidator_archive.md`, `agent/dream.md`; `identity.md` gets nanobot's "Dream" wording back |
| `nanoclaw/__main__.py` | `compact` and `dream` commands, `NANOCLAW_CONTEXT_WINDOW` |

## Tests

- `tests/test_memory.py` (30 tests): journal numbering, cleaning, corrupt lines,
  trimming without losing unread entries; Dream prompt and tool limits; raw
  fallback and combining with an earlier summary; archiving with the model
  (request shape, `(nothing)`, failures, tool calls refused once, too big);
  never summarizing only part of a stretch; summary checkpoints and the token
  limit in `get_history`; system prompt sections; token counting (bytes and
  tiktoken).
- `tests/test_compaction.py` (21 tests): a turn that compacts (archive request,
  turn request, saved session, journal); the next turn; no compaction under
  budget; window 0; a second compaction updating the first; a large overflow
  giving a raw checkpoint (as in nanobot); still too big after compacting; a
  crashed turn; `compact`; ephemeral turns; Dream (nothing to do, memory
  updated, no changes, incomplete run, journal trimmed); CLI commands and
  `NANOCLAW_CONTEXT_WINDOW`.
- `tests/conftest.py` counts tokens by bytes in tests, so results are exact and
  nothing is downloaded (one test uses tiktoken when available).

Every rule was checked by breaking it on purpose and confirming a test fails
(26 rules; the 25 from layer 5 still hold). One more change, passing
`persist=True` for temporary turns at the call site, cannot change behaviour
(a temporary turn has no history to journal); the `persist` flag itself is
tested directly.

## Review against nanobot

Checked by **running both projects** with the same scripted model:

| Check | Result |
|---|---|
| Token counts (tiktoken and byte fallback), token truncation | Identical |
| `get_history` with token limits, summary marker, `_command` messages; `commit_summary_checkpoint` | Identical (after porting the reply trimming below) |
| System prompt with memory and summary sections; `MEMORY.md` template | Identical (apart from nanobot's skills section, layer 7) |
| Journal: entries, cursors, trimming; Dream prompt; raw checkpoint; combining with an earlier summary | Identical (Dream prompt without the skills section) |
| `/new` then a turn: archive request, journal, next request, saved session | Identical |
| `/compact` then a turn: archive request, saved session, summary, journal, next request | Identical (except a timestamp) |
| A turn that overflows by 1, 300 and 3,000 tokens: model calls, archive request, turn request, summary, saved session, journal | Identical in all three (summary by the model for 1 and 300, raw dump for 3,000, in both projects) |
| Dream: request, tool result, `MEMORY.md`, cursor | Identical (except the workspace path) |

One correction came out of the review: nanobot trims whitespace from assistant
replies (and removes a few internal markers) before replaying them
(`_sanitize_assistant_replay_text`). Layer 5 had left that out; it is now
ported, and the layer 5 comparisons still pass.

### Second review (after the build)

A second pass ran both projects side by side on paths the first review had not
compared, with the same tool set (only `echo`) on both sides, comparing every
request sent (body and summary section), the saved session, the session
metadata and the journal:

| Scenario | Result |
|---|---|
| Tool calls in the history, then a compacting turn | Same (12 checks) |
| A second compaction updating the first summary | Same (14 checks) |
| `/compact`, a turn, then `/new` (archiving on top of an existing summary), then a turn | Same (14 checks) |
| A crashed turn, then an overflowing turn (crash closed, then compacted) | Same (10 checks) |
| A model error, then a normal turn | Same (8 checks) |
| An oversized temporary turn (as Dream uses) | **Different** before the fix: nanobot refuses it with `ContextWindowExceededError` before calling the model; NanoClaw skipped the check for temporary turns and sent it. Now both refuse it |

Also checked: nanobot's other model-facing cleanups in `prepare_for_model`
(removing `[Previous assistant message omitted.]` placeholders, nameless tool
calls, orphan tool results; backfilling missing tool results; re-budgeting
tool results). NanoClaw never writes such messages (the layer 2 and 5 save
rules prevent them), so they change nothing here; they stay with the rest of
context governance in phase 2.

| # | Correction | nanobot | Fix |
|---|---|---|---|
| 1 | Temporary turns skipped the context-window check | Checks every turn; a temporary turn compacts without journaling, or is refused | Temporary turns go through the same check with `persist=False` (nothing journaled, no summary saved). Tests: `test_oversized_ephemeral_turn_is_refused_like_nanobot`, `test_compaction_without_persist_journals_nothing` |
| 2 | Dream's "nothing to process" message differed without being recorded | — | Recorded as difference #9 |

### Completed in the end-of-layer docs check

Checking every doc after layer 6 found two promises from earlier layers that
had not been kept. Both are now built, as in nanobot:

| Promise | Where it was made | Now |
|---|---|---|
| Deny patterns for writes to `history.jsonl` / `.dream_cursor` "come with memory in layer 6" | `03-tools.md`, difference #2 | nanobot's five patterns added to `DEFAULT_DENY_PATTERNS`; the list is identical to nanobot's (checked on 15 commands). Reading the journal stays allowed |
| Saved tool results get "a folder per session and clean-up with sessions in layer 5" | `03-tools.md`, difference #5 | `.nanoclaw/tool-results/<session>/`; other sessions' folders removed after 7 days and beyond 32 (nanobot `_cleanup_tool_result_buckets`) |

