# 05 — Sessions and the AgentLoop

Reference: nanobot at commit `d3d70f2`. Paths are relative to its `nanobot/` package.

## The concept

Until layer 4 every `nanoclaw ask` started from nothing. Now the agent
**remembers the conversation**:

```bash
nanoclaw ask "My name is Rohit"
nanoclaw ask "What's my name?"      # → "Rohit"
nanoclaw ask --new "Hi"             # forget it and start over
```

Two pieces make this work:

- A **session** is one conversation: the list of messages (user, assistant,
  tool) saved in a file. On the next turn they are sent to the model again.
- The **`AgentLoop`** runs one turn around the runner: load the session, save
  the new message, build the request, run the runner, save what it added.

## One turn

`AgentLoop.process_direct(text, session_key="cli:direct")` runs four steps,
the same as nanobot's `loop.py`:

| Step | What happens | nanobot |
|---|---|---|
| 1. Build | Load the session and take its history. **Save the user's message now**, so it is not lost if the turn crashes | `_build_turn`, `_persist_user_message_early` |
| 2. Run | `ContextBuilder.build_messages(history, text)` → `AgentRunner.run` (layers 2–4) | `_run_turn` |
| 3. Save | Append everything the runner added (assistant replies, tool calls, tool results) and write the file | `_persist_turn`, `_save_turn` |
| 4. Reply | Return the result; a blank reply becomes "I completed the tool steps but couldn't produce a final answer..." | `_persist_turn` |

Two turns in the same session never run at the same time (one `asyncio.Lock`
per session key).

## What is saved, and what is sent back

The file keeps more than the model needs; the history sent to the model is
cleaned up by `Session.get_history()`:

| Saved in the file | Sent to the model? |
|---|---|
| `role`, `content`, `tool_calls`, `tool_call_id`, `name`, `reasoning_content` | Yes |
| `timestamp` (every message) | No |
| `_runtime_context` (marker of the time block on user messages) | No |

Rules when **saving** (`_save_turn`):

- Empty assistant messages are skipped; they confuse later turns.
- A tool result whose call ID is unknown, or already answered, is dropped;
  providers reject those.

Rules when **replaying** (`get_history`):

- Start at the first user message, never in the middle of a turn.
- Drop tool results at the start whose tool call is missing.
- Skip assistant messages with blank text and nothing else.

### The time block is saved too

Each user message is saved exactly as the model saw it, including the
`Current Time` block from layer 4, plus a marker recording which part was
appended. So old turns are replayed **byte for byte** as they were first sent:

```
[user]       My name is Rohit           ← turn 1, with its 14:05 time block
[assistant]  Nice to meet you, Rohit.
[user]       What's my name?            ← turn 2, with its 14:07 time block
```

Two benefits: the start of each request stays identical from turn to turn
(providers cache it), and the model can see when each earlier message was sent.
This is nanobot's own design (`_persist_user_message_early` stores the
runtime context and a `_runtime_context` marker); only the time block itself
is our addition (layer 4, difference #1).

## Where sessions are stored

```
~/.nanoclaw/
├── workspace/                    the agent's folder (file tools work here)
│   └── .nanoclaw/workspace-id    random ID, e.g. b7fa0680f574b876391d07f096ae9ef6
└── sessions/                     outside the workspace, private (mode 700)
    └── b7fa0680.../              one folder per workspace, named by its ID
        └── Y2xpOmRpcmVjdA.jsonl  session "cli:direct" (file name = base64 of the key)
```

- **Outside the workspace**, so the agent's file tools can't read or rewrite
  its own history. Like nanobot, NanoClaw **refuses** to store sessions inside
  the workspace.
- **One folder per workspace**, so `-w project-a` and `-w project-b` keep
  separate conversations.
- **The folder is named by an ID saved in the workspace**, not by its path, so
  the history follows the workspace when you move or rename it.
- **Base64 file names** turn any key (`telegram:-100123`, `a/b`) into a safe
  file name.

File format (JSON Lines): one metadata line, then one line per message.

```json
{"_type": "metadata", "key": "cli:direct", "created_at": "...", "updated_at": "...", "metadata": {}, "last_archived": 0}
{"role": "user", "content": "My name is Rohit\n\n[Runtime Context — ...]...", "timestamp": "...", "_runtime_context": {"version": 1, "sources": ["time"], "suffix": "..."}}
{"role": "assistant", "content": "Nice to meet you, Rohit.", "timestamp": "..."}
```

The whole file is rewritten on each save, through a temporary file and a
rename, so a crash never leaves a half-written file. Unreadable lines are
skipped when loading. The format is the same as nanobot's: each project can
read the other's session files (checked).

## CLI

| Command | Effect |
|---|---|
| `nanoclaw ask "..."` | Continues session `cli:direct` (the key nanobot's terminal chat uses) |
| `nanoclaw ask --new "..."` | Clears the session first (nanobot's `/new` command) |
| `nanoclaw ask -s work "..."` | Uses another session, `work` |

## Limits until layer 6

A session grows without limit: `get_history()` replays everything since
`last_archived`, which nanobot also does at this commit. In nanobot, memory and
compaction (summarizing old turns, then moving `last_archived` forward) keep
the request small; that is layer 6. Until then, use `--new` when a
conversation gets long.

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | Loading a damaged file | Strict read first; on error, a `repair` pass skips bad lines and logs a warning | Always skips bad lines, without logging | Same result, simpler; there is no logging yet |
| 2 | Return value of `process_direct` | An `OutboundMessage` for the bus | The runner's `AgentRunResult` | No message bus yet; the CLI needs `tools_used` and `stop_reason` |
| 3 | `/new` | Also archives the old messages into memory in the background | Only clears | Memory arrives in layer 6 |

## What we left out

| nanobot feature | Where | Why it exists |
|---|---|---|
| Message bus, channels, per-session queues | `bus/`, `channels/`, `loop.py` | Many chat apps feeding one agent (phase 2) |
| Cancellation, crash recovery, runtime checkpoints | `loop.py`, `session/recovery.py`, `save_runtime_checkpoint` | Resume or stop a turn mid-way (phase 2) |
| Cross-process file locks | `FileLock` in `JsonlSessionStore` | Several nanobot processes on one session folder |
| Workspace-ID recovery and migration | `_find_workspace_namespace`, `_claim_workspace_namespace`, legacy `~/.nanobot/sessions` | Old installs; copied workspaces |
| Summary checkpoints, history limits | `commit_summary_checkpoint`, `get_history(max_messages, max_tokens)` | Compaction (**layer 6**) |
| Provider conversation state | `provider_state` | Provider-side conversation IDs |
| Image and CLI-app breadcrumbs, replay-text cleanup | `get_history` | Features we don't have (images, CLI apps, `message` tool) |
| `latency_ms`, `_last_usage` | `_save_turn`, `_persist_turn` | UI statistics |
| Session listing, deleting, metadata-only updates | `list_sessions`, `delete_session`, `update_metadata` | Web UI and chat commands |
| Session policy (transient sessions) | `SessionPolicy`, `get_or_create_transient` | Sessions that are never saved |

## Files

| File | Contents |
|---|---|
| `nanoclaw/session/manager.py` | `Session`, `SessionManager`, `find_legal_message_start` |
| `nanoclaw/agent/loop.py` | `AgentLoop` (`process_direct`, `new_session`, `_persist_user_message_early`, `_save_turn`) |
| `nanoclaw/runtime_context.py` | `append_runtime_context` now also returns the marker |
| `nanoclaw/agent/context.py` | The current message carries the marker in `_meta` |
| `nanoclaw/providers/base.py` | `sanitize_messages` drops `_meta`, as nanobot's provider does |
| `nanoclaw/__main__.py` | `ask` uses `AgentLoop`; `--session`, `--new` |

About 440 lines of code, versus about 4,400 in nanobot's `session/manager.py`
and `agent/loop.py` (most of which serve features we left out).

## Tests

- `tests/test_session.py` (34 tests): replay rules (allowed keys, first user
  message, missing tool calls, blank assistant messages, archived messages, bad
  offsets), `clear`, save/load round trip, file format, safe file names, no temp
  files left, bad lines skipped, cache and eviction, default location outside the
  workspace, refusal inside it, workspace ID (created once, separate per
  workspace, follows a moved folder, invalid or symlinked ID refused), private
  folder.
- `tests/test_loop.py` (16 tests): the second turn sees the first; history after
  a restart; byte-identical replay; the saved marker; tool messages saved and
  replayed without bookkeeping keys; user message saved before a crash; a
  message after an interrupted turn; model errors; blank replies; save rules;
  separate sessions; `--new`; the per-session lock; CLI options.
- `tests/conftest.py` points `~` at a temporary folder, so tests never touch
  your real `~/.nanoclaw`.

Every rule was also checked by breaking it on purpose and confirming a test
fails (15 rules).

## Review against nanobot

Checked by **running both projects** (nanobot installed from `d3d70f2`):

| Check | How | Result |
|---|---|---|
| History replay | The same 13 message lists (tool calls, orphans, blank messages, markers, archived offsets, ...) through both `Session.get_history()` | Identical in all 13 |
| A full two-turn conversation | Same scripted model (a tool call, then answers) and the same runtime block in both `AgentLoop`s | Saved sessions identical (except nanobot's `latency_ms`); the second turn's request identical after the system prompt |
| File compatibility | Session written by nanobot read by NanoClaw, then written back and read by nanobot | Both work; same file name for the same key |
| Interrupted turn | A session ending with an unanswered user message, then a new turn with a tool call | Both save the full turn correctly. This was a suspected nanobot bug (its save boundary counts messages differently); running it showed it is not one |
