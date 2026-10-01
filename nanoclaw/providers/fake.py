"""A scripted provider for tests: no network, no API key."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from nanoclaw.providers.base import LLMProvider, LLMResponse


@dataclass
class RecordedCall:
    """What the agent sent on one ``chat`` call."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    model: str | None


class FakeProvider(LLMProvider):
    """Returns queued responses in order and records every request.

    A plain string in the script is shorthand for a text-only reply.
    """

    def __init__(self, script: list[LLMResponse | str] | None = None) -> None:
        super().__init__(default_model="fake-model")
        self._script: list[LLMResponse] = [
            LLMResponse(content=item) if isinstance(item, str) else item
            for item in (script or [])
        ]
        self.calls: list[RecordedCall] = []

    def queue(self, *items: LLMResponse | str) -> None:
        for item in items:
            self._script.append(LLMResponse(content=item) if isinstance(item, str) else item)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        # Deep-copy so later mutation of the caller's history can't rewrite
        # what the test sees as "sent".
        self.calls.append(
            RecordedCall(copy.deepcopy(messages), copy.deepcopy(tools), model or self.default_model)
        )
        if not self._script:
            return LLMResponse(content="FakeProvider: script exhausted", finish_reason="error")
        return self._script.pop(0)
