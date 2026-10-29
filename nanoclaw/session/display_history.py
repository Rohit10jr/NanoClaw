"""Durable display facts, separate from provider/model messages and transports.

Mirrors nanobot/session/display_history.py. nanobot keeps display facts in
SQLite tables (``history_turns``/``history_entries``) and reads them through a
``HistoryReader`` bound to one read transaction. NanoClaw keeps them in one
append-only JSONL file per chat (``nanoclaw/webui/history.py``), so this
``HistoryReader`` is built from an in-memory list of facts read from that file
(or synthesised from session messages). Its interface is nanobot's: turn
ordinals, per-turn record/byte/user counts, bounded entry reads, navigation.

A turn is the run of facts up to and including a ``COMPLETION`` fact (nanobot
starts a new turn ordinal after each completion).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

MAX_HISTORY_FACT_BYTES = 8 * 1024 * 1024


class HistoryKind(StrEnum):
    INPUT = "input"
    ANSWER = "answer"
    ANSWER_SEGMENT = "answer_segment"
    REASONING = "reasoning"
    REASONING_SEGMENT = "reasoning_segment"
    PROGRESS = "progress"
    TOOL_ACTIVITY = "tool_activity"
    FILE_ACTIVITY = "file_activity"
    COMPLETION = "completion"
    FORK_BOUNDARY = "fork_boundary"
    COMPACTION = "compaction"
    # Only old transcripts can contain unfinished transport fragments.
    ANSWER_FRAGMENT = "answer_fragment"
    REASONING_FRAGMENT = "reasoning_fragment"


@dataclass(frozen=True)
class HistoryFact:
    kind: HistoryKind
    content: Mapping[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))
    turn_id: str | None = None
    phase: str | None = None
    sequence: int | None = None
    created_at_ms: int = 0

    def encode(self) -> str:
        record = json.dumps({
            "kind": self.kind.value, "content": dict(self.content),
            "turn_id": self.turn_id, "phase": self.phase,
            "sequence": self.sequence, "created_at_ms": self.created_at_ms,
        }, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(record.encode("utf-8")) > MAX_HISTORY_FACT_BYTES:
            raise ValueError("history fact exceeds the 8 MiB storage limit")
        return record

    @classmethod
    def decode(cls, record: str) -> HistoryFact:
        value = cast(dict[str, Any], json.loads(record))
        return cls(
            HistoryKind(value["kind"]), value["content"], value["turn_id"],
            value["phase"], value["sequence"], value["created_at_ms"],
        )


@dataclass(frozen=True)
class HistoryTurn:
    ordinal: int
    record_count: int
    byte_count: int
    user_count: int


_NAVIGATION_KINDS = frozenset({
    HistoryKind.INPUT, HistoryKind.ANSWER, HistoryKind.ANSWER_SEGMENT,
    HistoryKind.ANSWER_FRAGMENT, HistoryKind.COMPLETION,
})
_ANSWER_KINDS = (HistoryKind.ANSWER, HistoryKind.ANSWER_SEGMENT, HistoryKind.ANSWER_FRAGMENT)


def split_fact_turns(facts: Sequence[HistoryFact]) -> list[list[HistoryFact]]:
    """Group facts into turns; a completion closes the current turn."""
    turns: list[list[HistoryFact]] = []
    current: list[HistoryFact] = []
    for fact in facts:
        current.append(fact)
        if fact.kind is HistoryKind.COMPLETION:
            turns.append(current)
            current = []
    if current:
        turns.append(current)
    return turns


class HistoryReader:
    """A single-session snapshot of display facts.

    ``revision`` is ``None`` when the session has no display history store at
    all (nanobot: no session row); otherwise an opaque ``(identity, count)``
    pair that changes with every append.
    """

    def __init__(
        self,
        key: str,
        facts: Sequence[HistoryFact],
        *,
        storage_path: Path | None = None,
        revision: tuple[str, int] | None = None,
    ) -> None:
        self.key = key
        self.storage_path = storage_path or Path()
        self._revision = revision
        self._turns: list[list[tuple[HistoryFact, int]]] = [
            [(fact, len(fact.encode().encode("utf-8"))) for fact in turn]
            for turn in split_fact_turns(facts)
        ]

    @property
    def revision(self) -> tuple[str, int] | None:
        return self._revision

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def _summary(self, ordinal: int) -> HistoryTurn:
        turn = self._turns[ordinal]
        return HistoryTurn(
            ordinal,
            len(turn),
            sum(size for _, size in turn),
            sum(1 for fact, _ in turn if fact.kind is HistoryKind.INPUT),
        )

    def turns(
        self, *, before: int | None = None, after: int = 0, reverse: bool = False,
    ) -> Iterator[HistoryTurn]:
        upper = self.turn_count if before is None else min(before, self.turn_count)
        ordinals = range(max(0, after), upper)
        for ordinal in (reversed(ordinals) if reverse else ordinals):
            yield self._summary(ordinal)

    def read_entries(
        self, ordinal: int, *, max_records: int | None = None, max_bytes: int | None = None,
    ) -> list[tuple[int, HistoryFact]]:
        if not 0 <= ordinal < self.turn_count:
            return []
        turn = self._turns[ordinal]
        if max_records is None or max_bytes is None:
            return [(position, fact) for position, (fact, _) in enumerate(turn)]
        # Preserve the first input and the useful tail of a pathological turn
        # (nanobot reads sizes first, then decodes only the selected rows).
        priorities: list[tuple[int, int]] = []
        for kinds, reverse in (((HistoryKind.INPUT,), False), ((HistoryKind.COMPLETION,), True),
                               (_ANSWER_KINDS, True)):
            indexed = list(enumerate(turn))
            for position, (fact, size) in (reversed(indexed) if reverse else indexed):
                if fact.kind in kinds:
                    priorities.append((position, size))
                    break
        candidates = [(position, size) for position, (_, size) in reversed(list(enumerate(turn)))][:max_records]
        positions: set[int] = set()
        used = 0
        for position, size in (*priorities, *candidates):
            if position not in positions and len(positions) < max_records and used + size <= max_bytes:
                positions.add(position)
                used += size
        return [(position, turn[position][0]) for position in sorted(positions)]

    def user_count_before(self, ordinal: int) -> int:
        return sum(self._summary(index).user_count for index in range(min(ordinal, self.turn_count)))

    def navigation(self) -> Iterator[tuple[int, int, HistoryFact]]:
        """Read input/answer records only, never file diffs or tool result bodies."""
        for ordinal, turn in enumerate(self._turns):
            for position, (fact, _) in enumerate(turn):
                if fact.kind in _NAVIGATION_KINDS:
                    yield ordinal, position, fact

    def all_facts(self) -> list[HistoryFact]:
        return [fact for turn in self._turns for fact, _ in turn]
