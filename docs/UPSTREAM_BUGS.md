# Upstream bugs (nanobot)

Bugs we found in [nanobot](https://github.com/HKUDS/nanobot) while porting it,
recorded so we can open issues or pull requests there later. "Upstream" means
the original project NanoClaw is based on.

- nanobot commit checked: `d3d70f2` (2026-10-01, the latest `main` at the time).
- Each bug was **reproduced by running nanobot's own code** (installed from that
  commit in a separate virtual environment), not only by reading it.

## Before filing

1. Search nanobot's open and closed issues and PRs for the same problem.
2. Re-run the reproduction on nanobot's latest `main`; it may already be fixed.
3. Keep the PR small: the fix plus a regression test in nanobot's `tests/`.
4. Follow nanobot's `CONTRIBUTING.md` and `AGENTS.md` (e.g. they ask contributors not to run `ruff format`).

## Summary

| # | Bug | Confidence | Severity | Status in NanoClaw |
|---|---|---|---|---|
| 1 | `list_dir(recursive=true)` reports an empty directory when the workspace is inside a folder named `build`, `dist`, `venv`, ... | Confirmed | Medium | Fixed (layer 3) |
| 2 | The configured timezone has no effect: no current date/time reaches the model, although the docs say it does | Confirmed | Medium | Added a current-time block (layer 4) |
| 3 | A saved oversized tool result can point the model to stale content when a tool-call ID repeats in a session | Mechanism confirmed; real-world trigger not yet confirmed | Low–medium | Fixed with the suggested fix (content-hash file names) |
| 4 | Compaction usually falls back to a raw dump: the model is only asked for a summary when the request overflows by less than ~600 tokens | Confirmed | Medium | Same as nanobot by default; an opt-in fix (`NANOCLAW_COMPACTION=fixed`) is being A/B tested |
| 5 | When a **repeated** subagent report triggers compaction, the summary marker is placed one message too early, so an already-summarized message is replayed again | Confirmed (only for a report delivered twice) | Low | Correct in NanoClaw (layer 8, difference #8) |
| 6 | Terminal progress lines are printed as rich markup: a shell command containing `[/x]` **aborts the turn** in `nanobot agent -m`; `[red]`, `[a-z]` etc. silently disappear from the displayed line | Confirmed | Medium | Fixed in NanoClaw (text escaped; layer 10) |

---

## 1. `list_dir` recursive listing is empty under a `build`/`dist`/`venv` folder

**Where:** `nanobot/agent/tools/filesystem.py`, `ListDirTool.execute`, line 1161:

```python
for item in sorted(dp.rglob("*")):
    if any(p in self._IGNORE_DIRS for p in item.parts):   # item.parts is the ABSOLUTE path
        continue
```

**Problem:** the noise-folder check (`.git`, `node_modules`, `build`, `dist`,
`venv`, `.venv`, `.tox`, `htmlcov`, ...) looks at *every* part of the absolute
path, including the folders *above* the one being listed. If the workspace
itself is anywhere under a folder with one of those names, every entry is
skipped. The non-recursive listing checks only `item.name`, so it works, which
makes the result inconsistent.

**Realistic trigger:** Docker images and CI runners often build in `/build` or
`/dist`; projects kept under `~/build/...`; a workspace inside a `venv` folder.

**Reproduction** (with nanobot installed):

```python
import asyncio
from pathlib import Path
from nanobot.agent.tools.filesystem import ListDirTool

ws = Path("/tmp/repro/build/myproject")
(ws / "src").mkdir(parents=True, exist_ok=True)
(ws / "README.md").write_text("x")
(ws / "src" / "app.py").write_text("y")

async def main():
    tool = ListDirTool(workspace=ws)
    print(await tool.execute(path="."))                  # 📄 README.md / 📁 src   (correct)
    print(await tool.execute(path=".", recursive=True))  # "Directory . is empty"  (wrong)
asyncio.run(main())
```

Observed on `d3d70f2`: flat listing `📄 README.md | 📁 src`; recursive listing
`Directory . is empty`. The same project under `/tmp/repro/normal/myproject`
lists `README.md | src/ | src/app.py` correctly.

**Suggested fix:** check only the parts below the listed directory.

```diff
-                    if any(p in self._IGNORE_DIRS for p in item.parts):
+                    if any(p in self._IGNORE_DIRS for p in item.relative_to(dp).parts):
```

**Regression test:** create a workspace at `<tmp>/build/proj` with one file,
assert that `list_dir(".", recursive=True)` returns that file.

**NanoClaw:** fixed in `nanoclaw/agent/tools/filesystem.py` (`ListDirTool`);
test `test_list_dir_ignore_rules_apply_below_the_listed_dir_only` in
`tests/test_filesystem_tools.py`.

---

## 2. Timezone setting has no effect; the model never gets the current date/time

**What the docs say:** `docs/configuration.md` line 2324 and 2336:

> By default, nanobot uses `UTC` for runtime time context. If you want the agent
> to think in your local time, set `agents.defaults.timezone` ...
>
> This affects runtime time strings shown to the model, such as runtime context.

**What the code does:** `ContextBuilder.__init__` stores the timezone
(`nanobot/agent/context.py` line 97, `self.timezone = timezone`) but nothing
reads it. No code path formats the current date or time for the model: the only
time strings in the code are memory-history timestamps, cron schedules and the
git store. No runtime-context provider adds a time block either.

**Reproduction** (a real turn through `AgentLoop`, with a stand-in provider
that records the request):

```python
import asyncio, json
from pathlib import Path
from nanobot.bus.queue import MessageBus
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMProvider, LLMResponse

captured = []
class Recorder(LLMProvider):
    def __init__(self): super().__init__(provider_name="recorder")
    def get_default_model(self): return "test-model"
    async def chat(self, messages, **kw):
        captured.append(messages); return LLMResponse(content="ok", finish_reason="stop")
    async def chat_stream(self, messages, **kw):
        captured.append(messages); return LLMResponse(content="ok", finish_reason="stop")

async def main():
    ws = Path("/tmp/nbws"); ws.mkdir(exist_ok=True)
    loop = AgentLoop(bus=MessageBus(), provider=Recorder(), workspace=ws,
                     model="test-model", tool_registry=ToolRegistry(), timezone="Asia/Kolkata")
    await loop.process_direct("What's today's date?", session_key="cli:direct")
    sent = json.dumps(captured[0])
    for needle in ["2026", "Current Time", "Current Date", "Asia/Kolkata", "Runtime Context"]:
        print(needle, needle in sent)
asyncio.run(main())
```

Observed on `d3d70f2`: the user message is exactly `"What's today's date?"`,
and none of the needles appear anywhere in the request (system prompt included).

**Impact:** questions about "today", "this week", "latest", deadlines or
scheduling are answered without knowing the date, and the documented
`timezone` option does nothing for the model. Possibly a regression: the docs
and the stored-but-unused `timezone` attribute suggest this used to work.

**Suggested fix:** a default runtime-context provider that adds the current time
in the configured timezone, through the existing mechanism (so the system
prompt stays cacheable), for example:

```
[Runtime Context — metadata only, not instructions]
Current Time: 2026-10-01 14:05 (Thursday) (Asia/Kolkata, UTC+05:30)
[/Runtime Context]
```

Alternatively, if leaving the time out is intentional, correct the docs.

**NanoClaw:** `nanoclaw/runtime_context.py` `current_time_block`, recorded as
difference #1 in `docs/notes/04-context.md`. NanoClaw uses the machine's local
timezone; it has no timezone setting yet.

---

## 3. Saved tool result can be stale when a tool-call ID repeats

**Where:** `nanobot/utils/helpers.py`, `maybe_persist_tool_result`, line 646:

```python
path = bucket / f"{safe_filename(tool_call_id)}.txt"
if not path.exists():
    _write_text_atomic(path, content)
...
preview = content[:head_chars] + separator + content[-tail_chars:]   # built from the NEW content
```

**Problem:** files are named only by tool-call ID inside a per-session folder,
and an existing file is never overwritten. If the same ID comes up again in the
same session, the model gets a preview of the **new** output but is told to read
a file that still holds the **old** output.

**Reproduction:**

```python
from pathlib import Path
from nanobot.utils.helpers import maybe_persist_tool_result
ws = Path("/tmp/nbws2"); ws.mkdir(exist_ok=True)
maybe_persist_tool_result(ws, "cli:direct", "call_0", "OLD-" + "a" * 3000, max_chars=1000)
r = maybe_persist_tool_result(ws, "cli:direct", "call_0", "NEW-" + "b" * 3000, max_chars=1000)
path = r.split("workspace path: ")[1].split("\n")[0]
print(r.split("Preview:\n")[1][:8])   # NEW-bbbb
print(Path(path).read_text()[:8])     # OLD-aaaa
```

**Confidence:** the mechanism is confirmed. Whether it happens in practice
depends on a provider (or local model server) reusing tool-call IDs across
turns in one session; **we have not confirmed one that does**. nanobot
de-duplicates IDs within one response but not across turns. Confirm a real
trigger before filing, or file it as a robustness issue.

**Suggested fix:** make the file name depend on the content as well, e.g.
`f"{safe_filename(tool_call_id)}-{sha256(content)[:12]}.txt"`. Writing stays
idempotent for identical content (which is presumably why nanobot never
overwrites) and different content can never share a file.

**NanoClaw:** fixed with exactly this suggested fix. Before the fix, NanoClaw
overwrote the file instead, which had the mirror-image problem: an *older*
reference in the history would then point to the newer content. Now
`nanoclaw/agent/tool_results.py` (`maybe_persist_tool_result`) names files
`<call id>-<first 12 hex chars of sha256(content)>.txt` and writes only if the
file doesn't exist. Regression tests in `tests/test_tool_results.py`:
`test_repeated_call_id_never_points_to_stale_output` (both references keep
their own content) and `test_identical_output_reuses_one_file`. Removing the
hash from the name makes three tests fail. This doubles as a tested version of
the upstream fix for the PR.

---

## 4. Compaction usually falls back to a raw dump

**Where:** `nanobot/agent/memory.py`, `MemoryArchiver.archive` (input check
around line 896), reached from `Consolidator.summarize_transcript` (line 1118)
when `ContextGovernor.prepare_request` finds a request over budget
(`agent/context_governance.py`, `request_pressure` line 414).

**Problem:** compaction starts when a request reaches the input budget
(`window − max_tokens − 1024`). The summary request then contains the same
system prompt, history and tool definitions, plus the archive prompt
(~400 tokens with tiktoken, ~1,650 by the byte fallback), and its budget is
only `window − max_tokens`, i.e. 1,024 tokens more. So the model is asked for a
summary only if the request overflowed by less than roughly
`1024 − archive prompt + new message` tokens, about 600 in practice. Any larger
overflow (for example one long message, or a big tool result) takes the
"degraded" path: a raw transcript dump.

The raw dump keeps the **start** of the archived conversation: it is cut to
16,000 characters, then to a token limit, from the front. So after a large
overflow the summary in the system prompt holds the oldest part of the
conversation, and the most recent context before the new message is lost.

**Reproduction** (two long turns, then a third request that overflows by a
chosen amount; nanobot's own default tool set):

```python
import asyncio, copy, json, tempfile
from pathlib import Path
from nanobot.bus.queue import MessageBus
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMProvider, LLMResponse, GenerationSettings
from nanobot.utils.helpers import estimate_prompt_tokens_chain
from nanobot.utils import token_encoding

BIG = "x" * 40000
calls = []
class P(LLMProvider):
    def __init__(self, replies):
        super().__init__(provider_name="p"); self.r = replies
        self.generation = GenerationSettings(max_tokens=100)
    def get_default_model(self): return "m"
    async def chat(self, messages, **kw):
        calls.append(messages); return LLMResponse(content=self.r.pop(0), finish_reason="stop")
    async def chat_stream(self, messages, **kw): return await self.chat(messages)

async def run(overflow):
    ws = Path(tempfile.mkdtemp())
    loop = AgentLoop(bus=MessageBus(), provider=P([BIG, BIG]), workspace=ws, model="m",
                     tool_registry=ToolRegistry(), context_window_tokens=10_000_000)
    await loop.process_direct("first " + BIG, session_key="cli:direct")
    await loop.process_direct("second " + BIG, session_key="cli:direct")
    s = loop.sessions.get_or_create("cli:direct")
    msgs = loop.context.build_messages(history=s.get_history(), current_message="third", channel="cli")
    full, _ = estimate_prompt_tokens_chain(loop.provider, "m", msgs, loop.tools.get_definitions())
    loop2 = AgentLoop(bus=MessageBus(), provider=P(["- SUMMARY", "answer"]), workspace=ws, model="m",
                      tool_registry=ToolRegistry(), context_window_tokens=full - overflow + 100 + 1024)
    calls.clear()
    await loop2.process_direct("third", session_key="cli:direct")
    journal = [json.loads(l)["content"][:16] for l in (ws / "memory/history.jsonl").read_text().splitlines()]
    print(f"overflow {overflow}: model calls {len(calls)}, journal {sorted(set(journal))}")

async def main():
    token_encoding.warmup_token_encoding(); token_encoding._warmup_thread.join()
    for overflow in (1, 300, 600, 900, 2000):
        await run(overflow)
asyncio.run(main())
```

Observed on `d3d70f2`:

```
overflow 1: model calls 2, journal ['- SUMMARY']
overflow 300: model calls 2, journal ['- SUMMARY']
overflow 600: model calls 2, journal ['- SUMMARY']
overflow 900: model calls 1, journal ['[RAW] 4 messages']
overflow 2000: model calls 1, journal ['[RAW] 4 messages']
```

From 900 tokens of overflow on, the model is never asked to summarize (one call:
the turn itself), and the journal gets a raw dump. Logged as
"Memory archive input does not fit ... using raw checkpoint" and "Memory
consolidation degraded".

**Impact:** the "smart" summary (facts marked `[permanent]`, `[durable]`, ...)
that Dream and later turns rely on is skipped whenever a conversation crosses
the budget by more than a few hundred tokens, which a single long message or
tool result easily does. The conversation then continues with a truncated raw
transcript of its oldest part.

**Suggested fixes** (any one):

1. **Compact earlier**: start compaction at a threshold below the budget (for
   example 80–90 % of it), so the summary request still fits.
2. **Summarize what fits**: when the archive request is too big, summarize the
   most recent part that fits and raw-archive only the older remainder, or
   summarize in chunks.
3. **Keep the end, not the start, of the raw dump**, so at least the most recent
   context survives.

Regression test: a session whose next request overflows by a few thousand
tokens should produce a model-written summary (a call with the archive prompt).

**NanoClaw:** behaves exactly like nanobot (checked by running both with
overflows of 1, 300 and 3,000 tokens; see `docs/notes/06-memory.md`, "The raw
fallback"). The test `test_large_overflow_gets_a_raw_checkpoint_like_nanobot`
in `tests/test_compaction.py` documents that behaviour.

Re-checked on nanobot's latest `main` (`9dc0aba`, 2026-10-04): same results,
the archive code is unchanged. Not fixed upstream.

**Opt-in fix, being A/B tested:** `NANOCLAW_COMPACTION=fixed` (default
`nanobot`, unchanged). It combines fixes 2 and 3:

- **Summarize in parts**: when the archive request doesn't fit, the history
  is split into parts that each fit one request (each starting at a user
  message), and each part is summarized. The summaries are journaled oldest
  first and joined into the checkpoint.
- **Raw dump keeps the end**: if one turn alone can't fit, or the model fails
  on any part, the raw dump keeps the newest part instead of the oldest.

A first version summarized only the newest part that fit and journaled the
older rest raw. The dry run of the A/B script showed that facts in that older
rest dropped out of the agent's context (recall 1/9 against nanobot's 7/9), so
the older rest is now summarized too.

Compare both on a real model with `scripts/compare_compaction.py` (see
`docs/notes/06-memory.md`, "Compaction A/B"). If `fixed` wins, it becomes the
default and this bug is marked fixed in NanoClaw.

---

## 5. Summary marker one message too early on a repeated subagent report

**Where:** `nanobot/agent/loop.py`:

```python
# _build_turn, line 2088: set for every subagent turn, also when the report
# was a duplicate and _persist_subagent_followup() saved nothing
ctx.input_persisted_early = True

# _save_turn, line 2357: assumes the last saved row is this turn's input
insert_at = len(session.messages) - (1 if input_persisted_early else 0)
```

**Problem:** `_persist_subagent_followup` (line 2454) skips a report whose
`subagent_task_id` is already saved, but the turn still runs (and may
compact). When it compacts, `_save_turn` places the summary marker one row
before the end, assuming that row is this turn's input. For a duplicate it is
not: it is the previous turn's last reply, which **was** included in the
summary. That reply stays after the marker, so every later turn sends it again
next to the summary that already contains it, and `last_archived` is one too low.

**Reproduction** (scripted model, empty tool registry, as in
`docs/notes/08-subagents.md`, "Second review"):

1. Two long user turns, then a subagent report (task `abcd1234`) processed as
   a system turn; the model replies `first-reply`.
2. Reopen with a context window that the next request overflows by ~200 tokens.
3. Deliver the **same** report again (same `subagent_task_id`).

Saved session afterwards:

```
... ok2, [Subagent report], <summary marker>, first-reply, Fine.    ← nanobot (last_archived 5)
... ok2, [Subagent report], first-reply, <summary marker>, Fine.    ← expected (last_archived 6)
```

The next user turn's request in nanobot contains `first-reply` after the
`[Archived Context Summary]`, although the summary request already covered it.

**Severity:** low. It needs a report delivered twice (which the
deduplication exists for, e.g. a redelivered report after a crash or one
already injected into a running turn) **and** compaction in that same turn.
Effect: one duplicated message in context.

**Suggested fix:** record the history boundary before the follow-up is
persisted and use it for `insert_at` (as `_persist_user_message_early`'s
caller effectively does for user turns), or subtract 1 only when
`_persist_subagent_followup` returned `True`.

**NanoClaw:** correct. `AgentLoop._process_turn` records `boundary =
len(session.messages)` before saving the report and inserts the marker there.
Regression test: `test_compacting_on_a_duplicate_report_keeps_the_summary_boundary_right`
in `tests/test_subagents.py` (using nanobot's formula makes it fail).

---

## 6. Progress lines are printed as rich markup

**Where:** `nanobot/cli/terminal.py`, `_print_cli_progress_line` (line 276) and
`_print_interactive_line`:

```python
target.print(f"  [dim]↳ {text}[/dim]")
```

**Problem:** `text` is the model's text before a tool call, or a tool hint such
as `$ echo "[/x]"`. rich reads square brackets as markup, so:

- a closing tag that matches nothing (`[/x]`, `[/]`) raises `rich.errors.MarkupError`;
- a valid tag (`[red]`, `[bold]`) or a style-like word (`[a-z]`, `[it]`) is
  silently removed from what the user sees (`$ grep -E "[a-z]+" f` is shown as
  `$ grep -E "+" f`).

In one-shot mode (`nanobot agent -m`) the progress callback runs inside the
agent turn, so the exception **aborts the turn**: the tool never runs, there is
no reply, and only the user's message is saved. Square brackets are common in
shell commands (regex classes, `[[ … ]]` tests, array indexes, JSON), so this is
easy to hit.

**Reproduction** (nanobot `d3d70f2`; the second part uses a scripted model):

```python
from nanobot.cli import terminal as t
t._print_cli_progress_line('$ echo "[/x]"', None, None)
# rich.errors.MarkupError: closing tag '[/x]' at position 17 doesn't match any open tag

# A turn whose model calls exec with command 'echo "[/x]"', with
# on_progress=lambda c, **k: t._print_cli_progress_line(c, None, None)
# (what `nanobot agent -m` passes):
#   AgentLoop.process_direct(...) raises MarkupError; saved session: [user "go"] only
```

**Suggested fix:** escape the text: `from rich.markup import escape`, then
`f"  [dim]↳ {escape(text)}[/dim]"` (also in `_print_interactive_line` and the
reasoning printer). Regression test: a progress line with `[/x]`, `[red]` and
`[a-z]` prints them literally.

**NanoClaw:** fixed this way (`nanoclaw/cli/terminal.py`,
`progress_line_markup`); tests `test_progress_text_is_not_treated_as_markup`
in `tests/test_chat.py` (removing the escape makes it fail). Not checked: the
same pattern in nanobot's interactive classic chat, where the line is printed
from the outbound-consumer task; that task only catches `TimeoutError` and
`CancelledError`, so the error would likely end it, but this wasn't run.

---

## Checked and not bugs

| Observation | Why it is not a bug |
|---|---|
| `exec` deny list (`rm -rf`, `mkfs`, ...) is not applied by default | Intentional: it only applies in restricted mode (`shell.py` line 446, with a comment explaining "full access is an explicit trust decision"). NanoClaw applies it always, by choice (`docs/notes/03-tools.md`) |
| `edit_file` reports "No similar text found" for a one-line `old_text` with a typo | Near-match search compares whole lines by design, so a single changed line scores 0%. Could be an enhancement (character-level similarity for short `old_text`), not a bug |
| `list_dir` and `read_file` resolve paths outside the workspace by default | Intentional: workspace restriction is opt-in (`restrict_to_workspace`, default `false`) |
