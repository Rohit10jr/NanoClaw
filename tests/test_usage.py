"""LLMUsage and usage estimation.

Adapted from nanobot tests/providers/test_usage_contract.py (without the
streaming timing fields NanoClaw doesn't have).
"""

import pytest

from nanoclaw.providers import FakeProvider, LLMResponse, LLMUsage, ToolCallRequest
from nanoclaw.utils.tokens import estimate_message_tokens, estimate_prompt_tokens


def test_plain_construction_counts_as_reported():
    usage = LLMUsage(input_tokens=12, output_tokens=3)
    assert usage.total_tokens == 15
    assert usage.reported_tokens == 15 and usage.estimated_tokens == 0
    assert usage.cache_read_tokens is None and usage.request_count == 0


def test_reported_usage_derives_total_and_preserves_unreported_cache():
    usage = LLMUsage.reported(input_tokens=12, output_tokens=3)
    assert usage.total_tokens == 15
    assert usage.cache_read_tokens is None and usage.cache_write_tokens is None
    assert usage.source == "reported"
    assert usage.context_tokens == 12 and usage.request_count == 1


def test_reported_usage_preserves_explicit_total_across_operations():
    usage = LLMUsage.reported(input_tokens=15, output_tokens=18, total_tokens=175)
    assert usage.total_tokens == 175 and usage.reported_tokens == 175 and usage.estimated_tokens == 0
    assert LLMUsage.from_dict(usage.to_dict()) == usage
    combined = usage + LLMUsage.estimated(input_tokens=2, output_tokens=1)
    assert (combined.total_tokens, combined.reported_tokens, combined.estimated_tokens) == (178, 175, 3)


def test_reported_usage_normalizes_missing_or_underreported_total():
    assert LLMUsage.reported(input_tokens=15, output_tokens=18).total_tokens == 33
    under = LLMUsage.reported(input_tokens=15, output_tokens=18, total_tokens=12)
    assert under.total_tokens == 33 and under.reported_tokens == 33


def test_reported_usage_preserves_explicit_zero_cache():
    usage = LLMUsage.reported(input_tokens=12, output_tokens=3, cache_read_tokens=0, cache_write_tokens=0)
    assert usage.cache_read_tokens == 0 and usage.cache_write_tokens == 0


def test_usage_rejects_inconsistent_values():
    with pytest.raises(ValueError, match="must equal"):
        LLMUsage(input_tokens=10, output_tokens=2, total_tokens=12, reported_tokens=11)
    with pytest.raises(ValueError, match="at least"):
        LLMUsage(input_tokens=10, output_tokens=2, total_tokens=11, reported_tokens=11)
    with pytest.raises(ValueError, match="cache token counts"):
        LLMUsage.reported(input_tokens=10, output_tokens=2, cache_read_tokens=11)
    with pytest.raises(ValueError, match="non-negative"):
        LLMUsage(input_tokens=-1)


def test_serialization_is_strict():
    usage = LLMUsage.estimated(input_tokens=10, output_tokens=2)
    serialized = usage.to_dict()
    assert serialized["source"] == "estimated"
    assert LLMUsage.from_dict(serialized) == usage
    assert LLMUsage.from_dict({"prompt_tokens": 10, "completion_tokens": 2}) is None
    assert LLMUsage.from_dict({"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}) is None  # pre-12.3 shape
    assert LLMUsage.from_dict({**serialized, "total_tokens": 99}) is None
    assert LLMUsage.from_dict({**serialized, "source": "reported"}) is None
    assert LLMUsage.from_dict({**serialized, "legacy_alias": 12}) is None
    assert LLMUsage.from_dict({**serialized, "request_count": True}) is None


def test_aggregation_keeps_reported_estimated_split_and_unknown_cache():
    reported = LLMUsage.reported(input_tokens=10, output_tokens=2, total_tokens=20, cache_read_tokens=4)
    estimated = LLMUsage.estimated(input_tokens=5, output_tokens=1)
    combined = reported + estimated
    assert (combined.input_tokens, combined.output_tokens, combined.total_tokens) == (15, 3, 26)
    assert (combined.reported_tokens, combined.estimated_tokens) == (20, 6)
    assert combined.source == "mixed"
    assert combined.cache_read_tokens is None  # only one side reported it
    assert combined.context_tokens == 5
    assert combined.request_count == 2


def test_aggregation_sums_cache_reported_on_both_sides():
    a = LLMUsage.reported(input_tokens=10, output_tokens=1, cache_read_tokens=4, cache_write_tokens=0)
    b = LLMUsage.reported(input_tokens=20, output_tokens=1, cache_read_tokens=6, cache_write_tokens=2)
    assert ((a + b).cache_read_tokens, (a + b).cache_write_tokens) == (10, 2)


def test_empty_request_counts_without_replacing_last_context():
    usage = LLMUsage.reported(input_tokens=12, output_tokens=3) + LLMUsage.empty_request()
    assert usage.total_tokens == 15
    assert usage.context_tokens == 12
    assert usage.request_count == 2


# --- estimation (nanobot base.py:449 _usage_for_call) ---------------------------------


MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello there"}]
TOOLS = [{"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}}]


def test_estimates_missing_usage():
    response = LLMResponse(
        content="hi",
        tool_calls=[ToolCallRequest(id="c1", name="echo", arguments={"text": "x"})],
        reasoning_content="thinking",
    )
    usage = FakeProvider().usage_for_call(response, {"messages": MESSAGES, "tools": TOOLS})
    expected_in, _ = estimate_prompt_tokens(MESSAGES, TOOLS)
    expected_out = estimate_message_tokens(
        {
            "role": "assistant",
            "content": "hi",
            "tool_calls": [response.tool_calls[0].to_openai_tool_call()],
            "reasoning_content": "thinking",
        }
    )
    assert usage == LLMUsage.estimated(input_tokens=expected_in, output_tokens=expected_out)


def test_keeps_reported_usage():
    reported = LLMUsage.reported(input_tokens=5, output_tokens=1)
    response = LLMResponse(content="hi", usage=reported)
    assert FakeProvider().usage_for_call(response, {"messages": MESSAGES}) is reported


def test_zero_reported_usage_is_estimated():
    response = LLMResponse(content="hi", usage=LLMUsage.reported(input_tokens=0, output_tokens=0))
    assert FakeProvider().usage_for_call(response, {"messages": MESSAGES}).source == "estimated"


def test_errors_are_not_estimated():
    response = LLMResponse(content="Error calling LLM: x", finish_reason="error")
    assert FakeProvider().usage_for_call(response, {"messages": MESSAGES}) is None
