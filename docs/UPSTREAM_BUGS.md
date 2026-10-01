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
| 3 | A saved oversized tool result can point the model to stale content when a tool-call ID repeats in a session | Mechanism confirmed; real-world trigger not yet confirmed | Low–medium | Has the mirror-image issue; fix proposed below |

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

**NanoClaw:** `nanoclaw/agent/tool_results.py` overwrites the file instead,
which has the mirror-image problem: an *older* reference in the history would
then point to the newer content. The same content-hash file name fixes both;
not yet applied.

---

## Checked and not bugs

| Observation | Why it is not a bug |
|---|---|
| `exec` deny list (`rm -rf`, `mkfs`, ...) is not applied by default | Intentional: it only applies in restricted mode (`shell.py` line 446, with a comment explaining "full access is an explicit trust decision"). NanoClaw applies it always, by choice (`docs/notes/03-tools.md`) |
| `edit_file` reports "No similar text found" for a one-line `old_text` with a typo | Near-match search compares whole lines by design, so a single changed line scores 0%. Could be an enhancement (character-level similarity for short `old_text`), not a bug |
| `list_dir` and `read_file` resolve paths outside the workspace by default | Intentional: workspace restriction is opt-in (`restrict_to_workspace`, default `false`) |
