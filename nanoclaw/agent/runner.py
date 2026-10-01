"""The agent loop: call the model, run the tools it asks for, repeat.

Mirrors nanobot/agent/runner.py (AgentRunner.run / _run_core) and
nanobot/agent/tools/execution.py (execute_tool_calls). The runner is
model-facing only: it knows nothing about sessions, channels or prompts.
Its caller builds ``initial_messages`` and decides what to do with the result.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from nanoclaw.agent.messages import (
    BUDGET_EXHAUSTED_FINALIZATION_PROMPT,
    DEFAULT_ERROR_MESSAGE,
    EMPTY_FINAL_RESPONSE_MESSAGE,
    FINALIZATION_RETRY_PROMPT,
    MAX_ITERATIONS_MESSAGE,
    MODEL_ERROR_PLACEHOLDER,
    build_assistant_message,
    build_tool_message,
    is_blank,
    truncate_text,
)
from nanoclaw.agent.tools import ToolRegistry, is_tool_error
from nanoclaw.providers import LLMProvider, LLMResponse, LLMUsage, ToolCallRequest

# After this many empty replies in a row, ask once more with tools off.
_MAX_EMPTY_RETRIES = 2


@dataclass
class AgentRunSpec:
    """Everything one run needs."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    provider: LLMProvider
    model: str | None = None
    max_iterations: int = 20  # nanobot's default is 200
    max_tool_result_chars: int = 16_000
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class AgentRunResult:
    """What a run produced.

    ``stop_reason`` is one of ``completed``, ``error``,
    ``empty_final_response`` or ``max_iterations``.
    """

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: LLMUsage | None = None
    stop_reason: str = "completed"
    error: str | None = None


def _add_usage(total: LLMUsage | None, new: LLMUsage | None) -> LLMUsage | None:
    if new is None:
        return total
    if total is None:
        return new
    return LLMUsage(
        input_tokens=total.input_tokens + new.input_tokens,
        output_tokens=total.output_tokens + new.output_tokens,
    )


class AgentRunner:
    """Runs one agent turn against a provider and a tool registry."""

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        # Work on a copy so the caller's list is never modified.
        messages = copy.deepcopy(spec.initial_messages)
        tools_used: list[str] = []
        usage: LLMUsage | None = None
        empty_retries = 0

        for _iteration in range(spec.max_iterations):
            response = await self._request(spec, messages, with_tools=True)
            usage = _add_usage(usage, response.usage)

            # 1. The model wants tools: run them and go round again.
            if response.should_execute_tools:
                messages.append(
                    build_assistant_message(
                        response.content,
                        tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                        reasoning_content=response.reasoning_content,
                    )
                )
                for call in response.tool_calls:
                    content, ok = await self._execute_tool_call(spec, call)
                    messages.append(build_tool_message(call.id, call.name, content))
                    if ok:
                        tools_used.append(call.name)
                empty_retries = 0
                continue

            # Tool calls that arrive with a cut-off or filtered reply may be
            # incomplete; they are dropped and the reply is treated as text.
            content = response.content

            # 2. Empty reply: retry, then ask once more with tools off.
            cut_off = response.finish_reason in ("length", "content_filter")
            if not response.is_error and not cut_off and is_blank(content):
                empty_retries += 1
                if empty_retries < _MAX_EMPTY_RETRIES:
                    continue
                response = await self._request(
                    spec,
                    messages + [{"role": "user", "content": FINALIZATION_RETRY_PROMPT}],
                    with_tools=False,
                )
                usage = _add_usage(usage, response.usage)
                content = response.content

            # 3. Model error: stop, keeping history well formed.
            if response.is_error:
                final = content or DEFAULT_ERROR_MESSAGE
                self._append_model_error_placeholder(messages)
                return AgentRunResult(final, messages, tools_used, usage, "error", final)

            # 4. Still nothing to say.
            if is_blank(content):
                messages.append(build_assistant_message(EMPTY_FINAL_RESPONSE_MESSAGE))
                return AgentRunResult(
                    EMPTY_FINAL_RESPONSE_MESSAGE,
                    messages,
                    tools_used,
                    usage,
                    "empty_final_response",
                    EMPTY_FINAL_RESPONSE_MESSAGE,
                )

            # 5. A normal final answer.
            messages.append(
                build_assistant_message(content, reasoning_content=response.reasoning_content)
            )
            return AgentRunResult(content, messages, tools_used, usage, "completed")

        # Out of iterations: ask for a summary with tools off, else a fixed message.
        final, finalize_usage = await self._finalize_after_max_iterations(spec, messages)
        usage = _add_usage(usage, finalize_usage)
        messages.append(build_assistant_message(final))
        return AgentRunResult(final, messages, tools_used, usage, "max_iterations")

    async def _request(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        with_tools: bool,
    ) -> LLMResponse:
        tools = spec.tools.get_definitions() if with_tools else []
        return await spec.provider.safe_chat(
            messages=messages,
            tools=tools or None,
            model=spec.model,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
        )

    async def _execute_tool_call(
        self,
        spec: AgentRunSpec,
        call: ToolCallRequest,
    ) -> tuple[str, bool]:
        """Run one call; return the text for the tool message and whether it succeeded."""
        result = await spec.tools.execute(call.name, call.arguments)
        ok = not is_tool_error(result)
        text = "" if result is None else str(result)
        if is_blank(text):
            text = f"({call.name} completed with no output)"
        return truncate_text(text, spec.max_tool_result_chars), ok

    async def _finalize_after_max_iterations(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ) -> tuple[str, LLMUsage | None]:
        response = await self._request(
            spec,
            messages + [{"role": "user", "content": BUDGET_EXHAUSTED_FINALIZATION_PROMPT}],
            with_tools=False,
        )
        if response.is_error or response.has_tool_calls or is_blank(response.content):
            return MAX_ITERATIONS_MESSAGE.format(max_iterations=spec.max_iterations), response.usage
        return response.content or "", response.usage

    @staticmethod
    def _append_model_error_placeholder(messages: list[dict[str, Any]]) -> None:
        """Close the turn with an assistant message so the history can be replayed."""
        last = messages[-1] if messages else None
        if last and last.get("role") == "assistant" and not last.get("tool_calls"):
            return
        messages.append(build_assistant_message(MODEL_ERROR_PLACEHOLDER))
