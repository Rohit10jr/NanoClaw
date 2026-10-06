"""The agent loop: call the model, run the tools it asks for, repeat.

Mirrors nanobot/agent/runner.py (AgentRunner.run / _run_core) and
nanobot/agent/tools/execution.py (execute_tool_calls). The runner is
model-facing only: it knows nothing about sessions, channels or prompts.
Its caller builds ``initial_messages`` and decides what to do with the result.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanoclaw.agent.context_governance import merge_adjacent_user_messages_for_model
from nanoclaw.agent.messages import (
    BUDGET_EXHAUSTED_FINALIZATION_PROMPT,
    DEFAULT_ERROR_MESSAGE,
    EMPTY_FINAL_RESPONSE_MESSAGE,
    FINALIZATION_RETRY_PROMPT,
    MAX_ITERATIONS_MESSAGE,
    MODEL_ERROR_PLACEHOLDER,
    build_assistant_message,
    build_malformed_tool_call_note,
    build_tool_message,
    is_blank,
    strip_think,
)
from nanoclaw.agent.tool_results import normalize_tool_result
from nanoclaw.agent.tools import ToolRegistry, is_tool_error
from nanoclaw.agent.tools.registry import with_retry_hint
from nanoclaw.agent.tools.violations import classify_violation
from nanoclaw.providers import LLMProvider, LLMResponse, LLMUsage, ToolCallRequest
from nanoclaw.utils.tool_hints import format_tool_hints

# Called as on_progress(text) for the model's text before tool calls, and
# on_progress(hint, tool_hint=True) with the tool hints (nanobot's progress events).
ProgressCallback = Callable[..., Awaitable[None]]

# After this many empty replies in a row, ask once more with tools off.
_MAX_EMPTY_RETRIES = 2

# finish_reasons after which an empty reply is final rather than retried.
_NO_RETRY_FINISH_REASONS = frozenset({"error", "length", "refusal", "content_filter"})


@dataclass
class AgentRunSpec:
    """Everything one run needs."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    provider: LLMProvider
    model: str | None = None
    max_iterations: int = 20  # nanobot's default is 200
    max_tool_result_chars: int = 16_000
    # Where oversized tool results are saved; without it they are truncated.
    workspace: Path | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    # Saved tool results go in a folder per session (nanobot: per session key).
    session_key: str | None = None
    # When the iterations run out: ask the model for a summary with tools off
    # (main agent), or just end with the fixed message (subagents set False).
    finalize_on_max_iterations: bool = True
    # Replaces the default out-of-iterations message ({max_iterations} is filled in).
    max_iterations_message: str | None = None
    # Progress reports while the run works (nanobot ``on_progress``); see ProgressCallback.
    on_progress: ProgressCallback | None = None


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
        # Per-run throttle for repeated attempts at the same outside target.
        workspace_violation_counts: dict[str, int] = {}

        for _iteration in range(spec.max_iterations):
            response, request_usage = await self._request_model(spec, messages)
            usage = _add_usage(usage, request_usage)

            # 1. The model wants tools: run them and go round again.
            if response.should_execute_tools:
                if spec.on_progress is not None:
                    await self._report_progress(spec.on_progress, response)
                messages.append(
                    build_assistant_message(
                        response.content,
                        tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                        reasoning_content=response.reasoning_content,
                    )
                )
                for call in response.tool_calls:
                    raw, ok = await self._execute_tool_call(spec, call, workspace_violation_counts)
                    content = normalize_tool_result(
                        call.name,
                        call.id,
                        raw,
                        workspace=spec.workspace,
                        max_chars=spec.max_tool_result_chars,
                        bucket=spec.session_key or "default",
                    )
                    messages.append(build_tool_message(call.id, call.name, content))
                    if ok:
                        tools_used.append(call.name)
                empty_retries = 0
                continue

            # Tool calls that arrive with a cut-off or filtered reply may be
            # incomplete; they are dropped and the reply is treated as text.
            content = response.content

            # 2. Empty reply: retry, then ask once more with tools off.
            if response.finish_reason not in _NO_RETRY_FINISH_REASONS and is_blank(content):
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

    @staticmethod
    async def _report_progress(on_progress: ProgressCallback, response: LLMResponse) -> None:
        """The model's text, then the tool hints, before the tools run.

        (nanobot ``AgentProgressHook.before_execute_tools``, without streaming)
        """
        thought = strip_think(response.content) if response.content else None
        if thought:
            await on_progress(thought)
        hint = strip_think(format_tool_hints(response.tool_calls))
        await on_progress(hint or "", tool_hint=True)

    async def _request(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        with_tools: bool,
    ) -> LLMResponse:
        """One provider call. Inline <think> blocks are removed from the answer."""
        tools = spec.tools.get_definitions() if with_tools else []
        response = await spec.provider.safe_chat(
            messages=merge_adjacent_user_messages_for_model(messages),
            tools=tools or None,
            model=spec.model,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
        )
        if response.content and not response.is_error:
            response.content = strip_think(response.content)
        return response

    async def _request_model(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        malformed_retry: bool = False,
    ) -> tuple[LLMResponse, LLMUsage | None]:
        """Request with tools, recovering from tool calls that have no name.

        Mirrors nanobot runner.py _request_model: nameless calls are dropped.
        If that leaves none, the request is retried once with a note; if the
        retry is malformed too, one final request is made with tools off.
        """
        response = await self._request(spec, messages, with_tools=True)
        usage = response.usage
        all_dropped, original_finish = self._drop_malformed_tool_calls(response)
        if not all_dropped or original_finish not in ("tool_calls", "function_call"):
            return response, usage

        retry_messages = messages + [build_malformed_tool_call_note(response.content)]
        if not malformed_retry:
            retry, retry_usage = await self._request_model(spec, retry_messages, malformed_retry=True)
            return retry, _add_usage(usage, retry_usage)
        fallback = await self._request(spec, retry_messages, with_tools=False)
        return fallback, _add_usage(usage, fallback.usage)

    @staticmethod
    def _drop_malformed_tool_calls(response: LLMResponse) -> tuple[bool, str]:
        """Remove tool calls with an empty name. Returns (all_dropped, original_finish_reason).

        A nameless call can't run, and if saved to history the API would reject
        every later request in the conversation. (nanobot runner.py:1068)
        """
        original_finish = response.finish_reason
        if not response.tool_calls:
            return False, original_finish
        valid = [tc for tc in response.tool_calls if tc.has_valid_name()]
        if len(valid) == len(response.tool_calls):
            return False, original_finish
        response.tool_calls = valid
        if not valid:
            response.finish_reason = "stop"
        return not valid, original_finish

    async def _execute_tool_call(
        self,
        spec: AgentRunSpec,
        call: ToolCallRequest,
        workspace_violation_counts: dict[str, int] | None = None,
    ) -> tuple[str, bool]:
        """Run one call; return the text for the tool message and whether it succeeded.

        Mirrors nanobot agent/tools/execution.py _execute_tool_call. Boundary
        errors (private network, outside the workspace) are reported as nanobot
        does (``classify_violation``).
        """
        counts = workspace_violation_counts if workspace_violation_counts is not None else {}

        def failed(raw_text: str, payload: str) -> tuple[str, bool]:
            handled = classify_violation(
                raw_text=raw_text,
                soft_payload=payload,
                tool_name=call.name,
                arguments=call.arguments,
                workspace_violation_counts=counts,
            )
            return (handled if handled is not None else payload), False

        tool, params, error = spec.tools.prepare_call(call.name, call.arguments)
        if error:
            return failed(error, with_retry_hint(error))
        assert tool is not None
        try:
            result = await tool.execute(**params)
        except Exception as exc:
            return failed(str(exc), with_retry_hint(f"Error: {type(exc).__name__}: {exc}"))
        if is_tool_error(result):
            return failed(str(result), with_retry_hint(str(result)))
        return ("" if result is None else str(result)), True

    async def _finalize_after_max_iterations(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ) -> tuple[str, LLMUsage | None]:
        if not spec.finalize_on_max_iterations:
            return self._max_iterations_fallback(spec), None
        response = await self._request(
            spec,
            messages + [{"role": "user", "content": BUDGET_EXHAUSTED_FINALIZATION_PROMPT}],
            with_tools=False,
        )
        if response.is_error or response.has_tool_calls or is_blank(response.content):
            return self._max_iterations_fallback(spec), response.usage
        return response.content or "", response.usage

    @staticmethod
    def _max_iterations_fallback(spec: AgentRunSpec) -> str:
        """The fixed out-of-iterations message. (nanobot runner.py:1249)"""
        return (spec.max_iterations_message or MAX_ITERATIONS_MESSAGE).format(max_iterations=spec.max_iterations)

    @staticmethod
    def _append_model_error_placeholder(messages: list[dict[str, Any]]) -> None:
        """Close the turn with an assistant message so the history can be replayed."""
        last = messages[-1] if messages else None
        if last and last.get("role") == "assistant" and not last.get("tool_calls"):
            return
        messages.append(build_assistant_message(MODEL_ERROR_PLACEHOLDER))
