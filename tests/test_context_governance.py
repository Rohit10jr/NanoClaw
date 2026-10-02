"""Merging adjacent user messages before sending (nanobot context_governance.py)."""

import copy

from nanoclaw.agent.context_governance import merge_adjacent_user_messages_for_model as merge
from nanoclaw.runtime_context import (
    RuntimeContextBlock,
    append_runtime_context,
    detach_runtime_context,
    reattach_runtime_context,
)


def user(text, *blocks):
    content, marker = append_runtime_context(text, [RuntimeContextBlock(s, t) for s, t in blocks])
    message = {"role": "user", "content": content}
    if marker:
        message["_meta"] = {"runtime_context": marker}
    return message


def test_runtime_context_stays_at_the_end_after_a_merge():
    out = merge([user("hi", ("time", "[T]")), {"role": "user", "content": "Please answer"}])
    assert len(out) == 1
    assert out[0]["content"] == "hi\n\nPlease answer\n\n[T]"
    assert out[0]["_meta"]["runtime_context"]["suffix"] == "[T]"


def test_two_messages_with_runtime_context_keep_both_blocks_in_order():
    out = merge([user("a", ("time", "[T1]")), user("b", ("time", "[T2]"))])
    assert out[0]["content"] == "a\n\nb\n\n[T1]\n\n[T2]"
    assert out[0]["_meta"]["runtime_context"]["sources"] == ["time", "time"]


def test_plain_merges_and_non_adjacent_messages():
    assert merge([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]) == [
        {"role": "user", "content": "a\n\nb"}
    ]
    assert merge([{"role": "user", "content": ""}, {"role": "user", "content": "b"}])[0]["content"] == "b"
    apart = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "b"}]
    assert merge(apart) == apart


def test_a_marker_that_no_longer_matches_falls_back_to_a_plain_merge():
    stale = {"role": "user", "content": "edited", "_meta": {"runtime_context": {"version": 1, "sources": ["t"], "suffix": "[T]"}}}
    assert merge([stale, {"role": "user", "content": "b"}])[0]["content"] == "edited\n\nb"


def test_merge_does_not_change_the_input():
    messages = [user("hi", ("time", "[T]")), {"role": "user", "content": "x"}]
    before = copy.deepcopy(messages)
    merge(messages)
    assert messages == before


def test_detach_and_reattach_round_trip():
    content, marker = append_runtime_context("hi", [RuntimeContextBlock("t", "[T]")])
    clean, sources, blocks = detach_runtime_context(content, marker)
    assert (clean, sources, blocks) == ("hi", ["t"], [{"type": "text", "text": "[T]"}])
    assert reattach_runtime_context(clean, sources, blocks) == (content, marker)
    assert detach_runtime_context("changed", marker) is None
    assert detach_runtime_context("hi[T]", marker) is None  # the blank-line separator must be there
    assert detach_runtime_context("[T]", marker)[0] == ""
    parts, marker = append_runtime_context([{"type": "text", "text": "x"}], [RuntimeContextBlock("t", "[T]")])
    assert detach_runtime_context(parts, marker)[0] == [{"type": "text", "text": "x"}]
