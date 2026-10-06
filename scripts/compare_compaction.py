"""A/B test of compaction: nanobot's upstream method vs NanoClaw's fix (UPSTREAM_BUGS.md, bug 4).

Runs the same scripted conversation through two agents that differ only in
``compaction``:

  nanobot  exactly like upstream: if the history is too big for one archive
           request, a raw dump that keeps the oldest part
  fixed    summarize it in parts that each fit one request; only if that
           fails, a raw dump that keeps the newest part

Three facts are planted (early, middle, late), each at a random place in a
stretch of turns where the agent fetches a long log file with a tool, so the conversation compacts
several times. (Long tool results are what make the history outgrow the
archive request; a long *user* message doesn't, because the archive request
leaves out the new message.) Then a quiz asks for the
facts, and Dream folds the journal into MEMORY.md. The report compares how
often the model was asked for a summary, how many facts the agent still knew,
how many reached MEMORY.md, and what compaction cost.

    uv run python scripts/compare_compaction.py --dry-run            # no API, simulated model
    uv run python scripts/compare_compaction.py --runs 3             # real API (NANOCLAW_* variables)
    uv run python scripts/compare_compaction.py --runs 3 --model gpt-4o-mini

Results go to ``--out`` (default ``compaction-ab/<time>``): ``report.md``,
``results.json``, and per run and mode the checkpoints, the quiz answer, the
journal, MEMORY.md and the session file. The agent's only tool is
``fetch_log``, so it can't look the facts up in the journal.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoclaw.agent.loop import DREAM_NO_INPUT_MESSAGE, AgentLoop  # noqa: E402
from nanoclaw.agent.memory import COMPACTION_MODES  # noqa: E402
from nanoclaw.agent.tools import Tool, ToolRegistry  # noqa: E402
from nanoclaw.providers import LLMProvider, LLMResponse, ToolCallRequest, provider_from_env  # noqa: E402
from nanoclaw.providers.base import LLMUsage  # noqa: E402
from nanoclaw.session import SessionManager  # noqa: E402
from nanoclaw.utils.templates import sync_workspace_templates  # noqa: E402
from nanoclaw.utils.tokens import estimate_prompt_tokens, get_token_encoding  # noqa: E402

SESSION = "cli:direct"
ARCHIVE_MARKER = "Create a compact replacement checkpoint"  # start of agent/consolidator_archive.md
RAW_CHECKPOINT_STARTS = ("[RAW]", "[Previous archived context]")

# (position, question, answer to find, the message that plants it)
FACTS = [
    ("early", "What is my locker code?", "VELVET-4817",
     "Something to remember about me: my locker code is VELVET-4817."),
    ("mid", "What is my sister's name?", "Odalys",
     "Another thing to remember: my sister's name is Odalys."),
    ("late", "What is the project codename?", "BLUEHERON",
     "One more thing to remember: the project codename is BLUEHERON."),
]
QUIZ = (
    "Quiz time. From what you remember of our conversation (don't guess), answer briefly:\n"
    + "\n".join(f"{i}. {q}" for i, (_, q, _, _) in enumerate(FACTS, start=1))
    + "\nWrite 'unknown' for anything you don't remember."
)


# --- the conversation ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    enc = get_token_encoding()
    return len(enc.encode_ordinary(text)) if enc is not None else len(text.encode("utf-8"))  # same fallback as the estimator


def log_file(rng: random.Random, tokens: int) -> str:
    """A long log file (no facts in it)."""
    services = ["api", "worker", "scheduler", "billing", "search", "auth"]
    lines: list[str] = []
    text = ""
    while _count_tokens(text) < tokens:
        for _ in range(20):
            lines.append(
                f"2026-09-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:"
                f"{rng.randint(0, 59):02d} INFO {rng.choice(services)}-{rng.randint(1, 9)} "
                f"batch={rng.randint(10000, 99999)} items={rng.randint(1, 500)} took={rng.randint(5, 900)}ms status=ok"
            )
        text = "\n".join(lines)
    return text


def fetch_request(number: int) -> str:
    return f"Please fetch log file #{number} with the fetch_log tool. Don't analyse it; just reply 'noted'."


def conversation(seed: int, filler_tokens: int, fillers_per_fact: int) -> tuple[list[str], dict[int, str]]:
    """The user messages and the log files: three stretches of log fetches, each
    with one fact at a random place in it (the quiz comes after)."""
    rng = random.Random(seed)
    turns: list[str] = []
    logs: dict[int, str] = {}
    for _, _, _, plant in FACTS:
        at = rng.randint(0, fillers_per_fact)
        for slot in range(fillers_per_fact + 1):
            if slot == at:
                turns.append(plant)
                continue
            number = len(logs) + 1
            # Sizes vary (50-150%), like real tool results, so compaction overflows by different amounts.
            logs[number] = log_file(rng, int(filler_tokens * rng.uniform(0.5, 1.5)))
            turns.append(fetch_request(number))
    return turns, logs


class FetchLogTool(Tool):
    """Returns one of the run's log files (the only tool the agent has)."""

    def __init__(self, logs: dict[int, str]) -> None:
        self._logs = logs

    @property
    def name(self) -> str:
        return "fetch_log"

    @property
    def description(self) -> str:
        return "Fetch a log file by its number."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]}

    async def execute(self, number: int) -> str:
        return self._logs.get(int(number), f"Error: no log file #{number}")


# --- a provider wrapper that counts calls and tokens --------------------------------------------


@dataclass
class CallStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0

    def add(self, other: CallStats) -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.errors += other.errors


@dataclass
class Recorder(LLMProvider):
    """Wraps the real (or simulated) provider and sorts each call by phase."""

    inner: LLMProvider
    phase: str = "turn"
    stats: dict[str, CallStats] = field(default_factory=dict)
    last_request: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(default_model=self.inner.default_model)

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **kwargs: Any):
        last = messages[-1].get("content") if messages else ""
        phase = "compaction" if isinstance(last, str) and last.startswith(ARCHIVE_MARKER) else self.phase
        if phase != "compaction":
            self.last_request = messages
        response = await self.inner.chat(messages, tools, **kwargs)
        stats = self.stats.setdefault(phase, CallStats())
        stats.calls += 1
        if response.usage and response.usage.total_tokens:
            stats.input_tokens += response.usage.input_tokens
            stats.output_tokens += response.usage.output_tokens
        else:
            stats.input_tokens += estimate_prompt_tokens(messages, tools)[0]
            stats.output_tokens += _count_tokens(response.content or "")
        if response.finish_reason == "error":
            stats.errors += 1
        return response


class SimulatedModel(LLMProvider):
    """--dry-run: a stand-in model that "remembers" only what is in its request.

    Summaries, quiz answers and Dream's MEMORY.md list the facts whose answers
    appear in the request, so the two modes can be compared without an API.
    """

    def __init__(self) -> None:
        super().__init__(default_model="simulated")
        self.phase = "turn"
        self._dream_wrote = False

    @staticmethod
    def _found(messages: list[dict[str, Any]]) -> list[str]:
        text = json.dumps(messages, ensure_ascii=False)
        return [answer for _, _, answer, _ in FACTS if answer in text]

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **kwargs: Any):
        usage = LLMUsage(input_tokens=estimate_prompt_tokens(messages, tools)[0], output_tokens=20)
        last = messages[-1].get("content") if messages else ""
        found = self._found(messages)
        if isinstance(last, str) and last.startswith(ARCHIVE_MARKER):
            reply = "\n".join(f"- [permanent] the user said: {a}" for a in found) or "(nothing)"
            return LLMResponse(content=reply, usage=usage)
        if self.phase == "dream":
            if messages[-1].get("role") == "tool":
                return LLMResponse(content="Memory updated.", usage=usage)
            content = "# Memory\n" + "".join(f"- {a}\n" for a in found)
            call = ToolCallRequest(id="w1", name="write_file", arguments={"path": "memory/MEMORY.md", "content": content})
            return LLMResponse(content=None, tool_calls=[call], finish_reason="tool_calls", usage=usage)
        if self.phase == "quiz":
            return LLMResponse(content="\n".join(found) or "unknown", usage=usage)
        wanted = re.search(r"fetch log file #(\d+)", last) if isinstance(last, str) else None
        if wanted and messages[-1].get("role") == "user":
            call = ToolCallRequest(id=f"f{wanted.group(1)}", name="fetch_log", arguments={"number": int(wanted.group(1))})
            return LLMResponse(content=None, tool_calls=[call], finish_reason="tool_calls", usage=usage)
        return LLMResponse(content="noted", usage=usage)


# --- one run ------------------------------------------------------------------------------------


@dataclass
class RunResult:
    mode: str
    run: int
    compactions: int = 0
    model_summaries: int = 0
    recalled: dict[str, bool] = field(default_factory=dict)
    in_context: dict[str, bool] = field(default_factory=dict)
    in_memory: dict[str, bool] = field(default_factory=dict)
    stats: dict[str, CallStats] = field(default_factory=dict)
    turn_errors: list[str] = field(default_factory=list)
    seconds: float = 0.0


async def run_one(
    mode: str,
    run: int,
    turns: list[str],
    logs: dict[int, str],
    provider: LLMProvider,
    args: argparse.Namespace,
    out: Path,
) -> RunResult:
    result = RunResult(mode=mode, run=run)
    root = out / f"run{run}" / mode
    workspace = root / "workspace"
    sync_workspace_templates(workspace)
    recorder = Recorder(provider)
    tools = ToolRegistry()
    tools.register(FetchLogTool(logs))
    loop = AgentLoop(
        recorder,
        workspace,
        model=args.model,
        tools=tools,  # only fetch_log: the agent can't read the facts back from the journal
        sessions=SessionManager(workspace, sessions_root=root / "sessions"),
        max_tokens=args.max_tokens,
        temperature=0.1,
        context_window_tokens=args.window,
        compaction=mode,
    )
    checkpoints: list[str] = []
    started = time.monotonic()

    def set_phase(phase: str) -> None:
        recorder.phase = phase
        if isinstance(provider, SimulatedModel):
            provider.phase = phase

    def note_checkpoint() -> None:
        summary = loop.sessions.get_or_create(SESSION).metadata.get("_last_summary")
        text = summary["text"] if summary else None
        if text and (not checkpoints or checkpoints[-1] != text):
            checkpoints.append(text)

    for number, message in enumerate(turns, start=1):
        set_phase("turn")
        try:
            await loop.process_direct(message, session_key=SESSION)
        except Exception as exc:  # e.g. the request still doesn't fit after compaction
            result.turn_errors.append(f"turn {number}: {type(exc).__name__}: {exc}")
        note_checkpoint()
        print(f"  [{mode} run {run}] turn {number}/{len(turns)} done, {len(checkpoints)} compactions", flush=True)

    set_phase("quiz")
    try:
        answer = (await loop.process_direct(QUIZ, session_key=SESSION)).final_content or ""
    except Exception as exc:
        answer = ""
        result.turn_errors.append(f"quiz: {type(exc).__name__}: {exc}")
    note_checkpoint()
    quiz_request = json.dumps(recorder.last_request, ensure_ascii=False)
    for position, _, expected, _ in FACTS:
        result.recalled[position] = expected.lower() in answer.lower()
        result.in_context[position] = expected in quiz_request

    set_phase("dream")
    dream_log = []
    for _ in range(args.max_dreams):
        status = await loop.run_dream()
        dream_log.append(status)
        if status == DREAM_NO_INPUT_MESSAGE or not status.startswith("Dream completed"):
            break
    memory = loop.memory.read_memory()
    for position, _, expected, _ in FACTS:
        result.in_memory[position] = expected.lower() in memory.lower()

    result.compactions = len(checkpoints)
    # A raw dump starts with "[RAW]", or "[Previous archived context]" when it keeps the old summary.
    result.model_summaries = sum(1 for c in checkpoints if not c.startswith(RAW_CHECKPOINT_STARTS))
    result.stats = recorder.stats
    result.seconds = time.monotonic() - started

    sep = "\n\n" + "=" * 80 + "\n\n"
    (root / "checkpoints.md").write_text(
        sep.join(f"## Compaction {i}\n\n{c}" for i, c in enumerate(checkpoints, start=1)) or "(no compaction)\n",
        encoding="utf-8",
    )
    (root / "quiz.md").write_text(f"## Question\n\n{QUIZ}\n\n## Answer\n\n{answer}\n", encoding="utf-8")
    (root / "dream.log").write_text("\n".join(dream_log) + "\n", encoding="utf-8")
    await loop.close()
    return result


# --- report -------------------------------------------------------------------------------------


def _pct(hits: int, total: int) -> str:
    return f"{hits}/{total} ({100 * hits / total:.0f}%)" if total else "-"


def report(results: list[RunResult], args: argparse.Namespace) -> str:
    lines = [
        "# Compaction A/B: nanobot vs fixed",
        "",
        f"{datetime.now():%Y-%m-%d %H:%M} · model `{args.model or 'default'}`"
        f"{' (simulated, --dry-run)' if args.dry_run else ''} · {args.runs} run(s) · window {args.window:,}"
        f" · max_tokens {args.max_tokens:,} · {args.fillers_per_fact} log fetches of ~{args.filler_tokens:,} (50-150%)"
        " tokens after each fact",
        "",
        "| | " + " | ".join(args.modes) + " |",
        "|---|" + "---|" * len(args.modes),
    ]
    by_mode = {m: [r for r in results if r.mode == m] for m in args.modes}

    def row(label: str, cell) -> None:
        lines.append(f"| {label} | " + " | ".join(cell(by_mode[m]) for m in args.modes) + " |")

    row("Compactions", lambda rs: str(sum(r.compactions for r in rs)))
    row("Model summaries (not a raw dump)", lambda rs: _pct(sum(r.model_summaries for r in rs), sum(r.compactions for r in rs)))
    for position, _, _, _ in FACTS:
        row(f"Quiz: {position} fact recalled", lambda rs, p=position: _pct(sum(r.recalled[p] for r in rs), len(rs)))
    row("Quiz: total recall", lambda rs: _pct(sum(sum(r.recalled.values()) for r in rs), len(FACTS) * len(rs)))
    for position, _, _, _ in FACTS:
        row(f"In MEMORY.md after Dream: {position}", lambda rs, p=position: _pct(sum(r.in_memory[p] for r in rs), len(rs)))
    row("In MEMORY.md: total", lambda rs: _pct(sum(sum(r.in_memory.values()) for r in rs), len(FACTS) * len(rs)))

    def totals(rs: list[RunResult], phase: str) -> CallStats:
        total = CallStats()
        for r in rs:
            if phase in r.stats:
                total.add(r.stats[phase])
        return total

    row("Compaction calls", lambda rs: str(totals(rs, "compaction").calls))
    row("Compaction tokens (in / out)", lambda rs: f"{totals(rs, 'compaction').input_tokens:,} / {totals(rs, 'compaction').output_tokens:,}")
    row("All model calls", lambda rs: str(sum(totals(rs, p).calls for p in ("turn", "compaction", "quiz", "dream"))))
    row("All tokens (in / out)", lambda rs: "{:,} / {:,}".format(
        sum(totals(rs, p).input_tokens for p in ("turn", "compaction", "quiz", "dream")),
        sum(totals(rs, p).output_tokens for p in ("turn", "compaction", "quiz", "dream")),
    ))
    row("Errors (model errors + failed turns)", lambda rs: str(
        sum(totals(rs, p).errors for p in ("turn", "compaction", "quiz", "dream")) + sum(len(r.turn_errors) for r in rs)
    ))

    lines += [
        "",
        "## Per run",
        "",
        "| Run | Mode | Compactions | Model summaries | Recalled | Fact still in context at the quiz | In MEMORY.md | Errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        def flags(d: dict[str, bool]) -> str:
            return " ".join(f"{p}:{'✓' if d[p] else '✗'}" for p, _, _, _ in FACTS)

        errors = sum(s.errors for s in r.stats.values()) + len(r.turn_errors)
        lines.append(
            f"| {r.run} | {r.mode} | {r.compactions} | {r.model_summaries} | {flags(r.recalled)} | "
            f"{flags(r.in_context)} | {flags(r.in_memory)} | {errors} |"
        )
    failures = [f"- run {r.run} {r.mode}: {e}" for r in results for e in r.turn_errors]
    if failures:
        lines += ["", "## Failed turns", "", *failures]
    lines += [
        "",
        "## How to read it",
        "",
        "- **Model summaries**: share of compactions where the model wrote a summary. nanobot falls back to a"
        " raw dump when the history overflows the archive request by more than about 600 tokens.",
        "- **Recalled**: the quiz answer contains the fact. **Still in context**: the fact was somewhere in"
        " the quiz request (history or checkpoint), so a miss there is the model's, not compaction's.",
        "- **In MEMORY.md**: Dream sees only the first 1,000 characters of each journal entry, so facts deep"
        " inside a raw dump don't reach long-term memory.",
        "- Per run folders hold the checkpoints, the quiz answer, the journal (`workspace/memory/history.jsonl`),"
        " `MEMORY.md` and the session file, side by side.",
    ]
    return "\n".join(lines) + "\n"


# --- main ---------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3, help="conversations per mode (default 3)")
    parser.add_argument("--model", default=None, help="model name (default: NANOCLAW_MODEL or the provider default)")
    parser.add_argument("--dry-run", action="store_true", help="use a simulated model, no API calls")
    parser.add_argument("--window", type=int, default=12_000, help="context window in tokens (default 12,000)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="reply limit in tokens (default 1,024)")
    parser.add_argument("--filler-tokens", type=int, default=3000, help="average size of a log file in tokens (default 3,000)")
    parser.add_argument("--fillers-per-fact", type=int, default=3, help="log fetches after each fact (default 3)")
    parser.add_argument("--max-dreams", type=int, default=5, help="Dream runs at most (default 5)")
    parser.add_argument("--modes", default=",".join(COMPACTION_MODES), help="modes to compare (default nanobot,fixed)")
    parser.add_argument("--out", type=Path, default=None, help="results folder (default compaction-ab/<time>)")
    args = parser.parse_args(argv)
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in args.modes if m not in COMPACTION_MODES]
    if unknown:
        parser.error(f"unknown mode(s) {unknown}; choose from {', '.join(COMPACTION_MODES)}")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    return args


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out or Path("compaction-ab") / datetime.now().strftime("%Y%m%d-%H%M%S")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    results: list[RunResult] = []
    for run in range(1, args.runs + 1):
        # Both modes get the same messages; only the model's replies differ.
        turns, logs = conversation(seed=run, filler_tokens=args.filler_tokens, fillers_per_fact=args.fillers_per_fact)
        for mode in args.modes:
            provider = SimulatedModel() if args.dry_run else provider_from_env()
            print(f"run {run}, mode {mode}: {len(turns)} turns + quiz + Dream", flush=True)
            results.append(await run_one(mode, run, turns, logs, provider, args, out))
    text = report(results, args)
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(
            [
                {
                    **{k: v for k, v in r.__dict__.items() if k != "stats"},
                    "stats": {phase: s.__dict__ for phase, s in r.stats.items()},
                }
                for r in results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(text)
    print(f"Saved to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
