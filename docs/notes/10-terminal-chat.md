# 10 — Terminal chat

Reference: nanobot at commit `d3d70f2`. Paths are relative to its `nanobot/` package.

## The concept

`nanoclaw ask` sends one message and exits. `nanoclaw chat` is a chat that
stays open: you type, the agent replies, the conversation continues until you
leave. It uses everything from layers 1–9.

```
$ uv run nanoclaw chat
nanoclaw Interactive mode (gpt-4o-mini) — type exit or Ctrl+C to quit

You: what's in this folder?
⠋ nanoclaw is thinking...
  ↳ Let me look.
  ↳ ls .

nanoclaw
There are three notes and a `skills` folder.

You: /history 2
...
You: exit

Goodbye!
```

## How nanobot does it

nanobot has two terminal chats. The default is a separate **native
full-screen app** (`cli/tui_launcher.py`). The **classic** chat
(`nanobot agent --classic`, `cli/agent.py:266–443`) is a Python prompt built
with **prompt_toolkit** (input, history) and **rich** (Markdown, spinner).
NanoClaw builds the classic chat; the full-screen app is phase 2.

| Feature | nanobot | NanoClaw |
|---|---|---|
| Banner `… Interactive mode (model) — type exit or Ctrl+C to quit` | `agent.py:271` | Same (`nanoclaw` instead of `🐈`) |
| Prompt `You:`; ↑/↓ input history saved to a file; multi-line input box: **Enter sends, Alt+Enter (or Shift+Enter where the terminal reports it) adds a line**; multi-line paste | `terminal.py:172` (`~/.nanobot/history/cli_history`), `:132`, `:415` | Same (`~/.nanoclaw/history/cli_history`) |
| Keys typed while the agent works are dropped before the next prompt (so they aren't sent by accident); terminal settings saved at start and restored on every exit; broken Unicode surrogates (emoji from some consoles) repaired in input and the history file | `_flush_pending_tty_input` `:98`, `_restore_terminal` `:122`, `SafeFileHistory` `:86`, `sanitize_surrogates` | Same |
| Leave with `exit`, `quit`, `/exit`, `/quit`, `:q` (any case), Ctrl+C or Ctrl+D at the prompt → "Goodbye!" | `terminal.py:51`, `agent.py:417–425` | Same |
| Ctrl+C (or another termination signal) **while the agent works** ends the chat: "Received SIGINT, goodbye!" | `agent.py:283` | Same (decision 2). The interrupted message gets the "interrupted" reply on the next turn (layer 5) |
| "… is thinking..." spinner while the agent works | `stream.py:50` `ThinkingSpinner` | Same |
| **Progress lines**: the model's text before tool calls, then short tool hints, e.g. `↳ $ ls -la`, `↳ read notes.txt × 2`, `↳ fetch example.com/…/page.html`, `↳ search "…"` | `progress_hook.py:135`, `utils/tool_hints.py`, `terminal.py:276` | Same (tool hints copied); the text is escaped (difference #2) |
| Reply under the bot's name, rendered as **Markdown** (`--no-markdown` for plain text); command output as plain text | `terminal.py:216`, `:233` | Same |
| Background **subagent results** printed above the prompt when they finish | `agent.py:296–350` (bus) | Same idea; read from the subagent queue (layer 8) |

## Slash commands

The commands NanoClaw has, with nanobot's descriptions and texts
(`command/builtin.py`, `command/router.py`):

| Command | What it does | nanobot |
|---|---|---|
| `/new` | Cancels this chat's subagents, archives the conversation into the journal, starts over: "New session started." | `cmd_new` `:314` |
| `/compact` | Summarizes the conversation now: `↳ Compressing context…`, `↳ Context compacted.` (or `Unable to compact context.`); nothing if there is nothing new | `cmd_compact` `:349`, compaction events (`bus/outbound_events.py:155`) |
| `/stop` | Cancels this chat's background subagents: "Stopped N task(s)." / "No active task to stop." | `cmd_stop` `:214` |
| `/status` | Model, last turn's tokens, context use, messages, uptime, running tasks | `cmd_status` `:267`, `build_status_content` (`utils/helpers.py:900`) |
| `/history [n]` | The last n messages (default 10, max 50; long ones cut at 200 characters), without the time block | `cmd_history` `:869` |
| `/dream` | "Dreaming...", then Dream's result | `cmd_dream` `:453` |
| `/skill` | Lists the skills | `cmd_skill` `:954` |
| `/help` | Lists these commands | `cmd_help` `:1022` |
| A registered command with wrong arguments | Never sent to the model: `Command "/new" does not accept arguments. Did you mean "/new"?` | `router.py:130` |
| Other text starting with `/` (paths, unknown names) | Since layer 11 sent to the model like any message, as nanobot `c4dac64` (`7e00d6a`); the "Unknown command … Did you mean" suggestions are gone. A path named like a command (`/new`) is still the command: write `/new/` or `look at /new` | `c4dac64 router.py:98-168` |

## What changed underneath

| Change | Why | nanobot |
|---|---|---|
| Runner: `on_progress` callback, called before tools run with the model's text (`<think>` removed) and the tool hints | The progress lines | `AgentProgressHook.before_execute_tools` (`progress_hook.py:135`); a small slice of hooks (phase 2) |
| Loop: `process_direct(on_progress=)`, `process_subagent_result(on_progress=)` | Pass it through | `process_direct(on_progress=)` |
| Loop: saves the last turn's token use in the session (`_last_usage`) | `/status` | `loop.py:2217` |
| Loop: `start_time` | `/status` uptime | `_start_time` |
| `Session.get_history(include_runtime_context=False)` | `/history` shows messages without the time block | `session/manager.py:289` |
| `nanoclaw ask` prints progress lines on stderr instead of `[tools used: …]` | Decision 3, as nanobot's `agent -m` | `cli/agent.py:199` |
| `prompt_toolkit` and `rich` added | Decision 1 | same libraries |

## Decisions (chosen with the user)

| # | Question | Decision |
|---|---|---|
| 1 | Libraries | `prompt_toolkit` + `rich`, as nanobot |
| 2 | Ctrl+C while the agent works | Ends the chat, as nanobot |
| 3 | Progress lines in `ask` | Yes, on stderr (stdout keeps only the reply) |
| 4 | Name in the banner and header | `nanoclaw`, no emoji (as in layer 4) |
| — | Streaming | Phase 2 (it changes the provider, the runner and needs hooks; see the plan discussion) |

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why | Later |
|---|---|---|---|---|---|
| 1 | Default chat | Full-screen native app; the Python prompt with `--classic` | The Python prompt (classic) only | The native app is a separate program | Phase 2: TUI |
| 2 | Progress-line text | Printed as rich markup: `[/x]` crashes, `[red]`/`[a-z]` vanish ([upstream bug 6](../UPSTREAM_BUGS.md#6-progress-lines-are-printed-as-rich-markup)) | Escaped, shown as typed | Bug fix | — |
| 3 | Replies appear | Streamed word by word | Whole, after the spinner | No streaming yet | Phase 2: Streaming |
| 4 | Commands | 17, plus `!shell` | The 8 above | The others need model presets, goals, triggers, Dream history, prompt overrides, pairing, the Web UI | Phase 2, with those features |
| 5 | `/stop` | Also cancels a running turn (from other channels) | Cancels subagents only | In the classic chat you can't type during a turn in either project | Phase 2: cancellation |
| 6 | `/dream` | Runs in the background; "Dreaming..." then a later message; also commits memory to git | Waits for Dream with the spinner, then prints the result | Same output, simpler; no git history (layer 6) | — |
| 7 | `/status` | Also cache hit rate and web-search quota | Model, tokens, context, messages, uptime, tasks | We don't track those | Phase 2: usage accounting |
| 8 | `ask` output | Reply rendered with the header and Markdown; progress on the console | Plain reply on stdout, progress on stderr | Keeps `ask` easy to use in scripts | — |
| 9 | How the chat reaches the agent | Over the message bus (`InboundMessage`, outbound events) | Direct calls to the loop | No bus (layer 5) | Phase 2: Message bus |
| 10 | A subagent result arriving during a turn | Added to the running turn | Printed after the turn, as its own follow-up | Layer 8 difference #3 | Phase 2: Mid-turn injections |
| 11 | Piped input (not a terminal) | Classic chat needs a terminal | Reads plain lines from stdin | Scripts and tests | — |

## What we left out (for later phases)

| nanobot feature | Where | Later phase |
|---|---|---|
| Full-screen native terminal app, themes | `cli/tui_launcher.py`, `--theme` | **Phase 2: Web UI / TUI** |
| Streaming replies | `cli/stream.py` `StreamRenderer`, provider `chat_stream` | **Phase 2: Streaming** |
| Showing the model's reasoning | `_ReasoningBuffer`, `show_reasoning` | **Phase 2: Streaming** |
| `/restart`, `/model`, `/goal`, `/trigger`, `/dream-log`, `/dream-restore`, `/dream-prompt`, `/evaluator-prompt`, `/pairing`, `!shell` | `command/builtin.py` | **Phase 2**, with model presets, goals, cron triggers, memory git history, prompt overrides, channels, Web UI |
| Cancelling a running turn | `_cancel_active_tasks` | **Phase 2: cancellation** |
| `--logs`, per-channel progress settings (`send_progress`, `send_tool_hints`) | `cli/agent.py`, `ChannelsConfig` | **Phase 2: Logging, configuration** |
| Restart notices | `utils/restart.py` | **Phase 2** (with `/restart`) |

## Files

| File | Contents | nanobot |
|---|---|---|
| `nanoclaw/cli/chat.py` (new) | `Chat` (loop, turns, subagent results, commands), command specs, help, malformed-command text, `/history` format, `/status` text, exit signals | `cli/agent.py`, `command/builtin.py`, `command/router.py`, `utils/helpers.py` |
| `nanoclaw/cli/terminal.py` (new) | Exit words, reply and progress printing, spinner, prompt with history | `cli/terminal.py`, `cli/stream.py` |
| `nanoclaw/utils/tool_hints.py`, `utils/path.py` (new) | Tool hints and path shortening (copied) | `utils/tool_hints.py`, `utils/path.py` |
| `agent/runner.py`, `agent/loop.py`, `session/manager.py` | `on_progress`, `_last_usage`, `start_time`, `include_runtime_context` | see above |
| `nanoclaw/__main__.py` | `nanoclaw chat [-s] [-w] [--restrict] [--no-markdown]`; `ask` progress lines | `cli/agent.py:54` |
| `pyproject.toml` | `prompt_toolkit`, `rich` | same |

## Tests

`tests/test_chat.py` (47 tests), the chat driven by scripted input with the
output captured:

- turns continue the conversation; blank lines skipped; the exit words in any case; Ctrl+C and Ctrl+D; the termination-signal handler;
- progress lines (model text, tool hints, repeats), escaped markup, blank lines; tool hints for each tool; the runner's progress callback;
- Markdown on and off;
- every slash command; unknown and malformed commands never reach the model; `/history` limits, cut-off and roles; `/new` and `/status` with a running subagent; `/compact` success, nothing to do, and failure;
- a background subagent result printed while waiting for input;
- the CLI: `chat` with piped input, `-s`, `--restrict`, `--no-markdown`; `ask` progress on stderr; `_last_usage`; history without the time block;
- (second review) the real prompt_toolkit input driven through a pipe: Enter sends, Alt+Enter adds a line, multi-line box, history file; surrogate repair; typed-ahead keys dropped; the terminal restored on every exit; `/dream`'s nothing-to-do text; the spinner clearing its line.

Also run by hand in a **real pseudo-terminal**: prompt_toolkit input,
spinner, progress lines, Markdown, `/help`, unknown-command suggestion, the
saved input history, exit.

Every rule was checked by breaking it on purpose (47 rules). The first run
showed 9 weak tests, now fixed. One change cannot fail: removing the `<think>`
strip from the progress text, because the runner already removes `<think>`
blocks when the reply arrives (kept, as in nanobot).

## Review against nanobot

Checked by running both projects' code:

| Check | Result |
|---|---|
| Tool hints for 2,000 random lists of calls (all tools, paths, URLs, commands, long and odd arguments) | Identical |
| `/help` (nanobot's lines for our commands), command descriptions and argument hints | Identical (apart from the name) |
| Unknown and malformed commands, through nanobot's real router (13 inputs) | Identical |
| `/history` message format (7 kinds of message) | Identical |
| `/status` text for 3 situations | Identical (apart from the first line, the name and version) |
| Exit words and the exit check | Identical |
| Reply rendering (Markdown, plain, command output, empty) | Identical (apart from the name) |
| Progress lines | Identical, except text with square brackets (bug 6) |
| Progress events of whole turns (text + tool, two calls, `<think>` block, no tools, `spawn`) and the `/history` lines afterwards | Identical, except that nanobot also sends an empty event after the tools ran (its chat prints nothing for it) |

**Found:** [upstream bug 6](../UPSTREAM_BUGS.md#6-progress-lines-are-printed-as-rich-markup):
a shell command containing `[/x]` aborts the turn in `nanobot agent -m`, and
bracketed text vanishes from progress lines. Fixed in NanoClaw.

### Second review (after the build)

Read nanobot's classic chat line by line (`cli/terminal.py`, `cli/stream.py`,
`cli/agent.py`) against ours. Six things were missing and are now added:

| # | Missing | nanobot | Fix |
|---|---|---|---|
| 1 | **Enter / Alt+Enter**: a multi-line input box where Enter sends and Alt+Enter (Shift+Enter on CSI-u terminals) adds a line | `_build_cli_key_bindings` (`terminal.py:132`), `multiline=True` | Same key bindings (compared: identical) |
| 2 | **Typed-ahead keys dropped** before each prompt, so keys pressed while the agent worked aren't sent as the next message | `_flush_pending_tty_input` (`:98`) | Same |
| 3 | **Terminal restored** (echo, line mode) on exit, Ctrl+C/Ctrl+D and termination signals | `_init_prompt_session` saves, `_restore_terminal` (`:122`) | Same |
| 4 | **Broken Unicode surrogates** (emoji from Windows consoles) repaired in input and in the history file | `sanitize_surrogates`, `SafeFileHistory` (`:86`, issue #2846) | Same |
| 5 | **Spinner line cleared** before printing | `_clear_current_line` (`stream.py:26`) | Same |
| 6 | **`/dream` with nothing to do** shown as plain text (its line breaks and backticks kept) | `cmd_dream`, `render_as: text` (`builtin.py:480`) | Same |

Checked again in a real pseudo-terminal: "hello", Alt+Enter, "second line"
is sent as one two-line message and saved as one history entry. The 12 new
rules were checked by breaking them on purpose (3 needed stronger tests, now
added). No new nanobot bugs found.

