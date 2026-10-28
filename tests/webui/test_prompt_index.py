"""Complete, lightweight navigation uses the same IDs and cursors as replay.

Ported from nanobot tests/webui/test_prompt_index.py (its HTTP-route tests wait for layer 23).
"""

import pytest

from nanoclaw.webui import transcript
from nanoclaw.webui.session_list_index import delete_webui_session


def test_complete_index_and_bidirectional_pages_without_loading_model_messages(history, tmp_path, monkeypatch):
    key = "websocket:navigation"
    for i in range(120):
        history.append(key, {"event": "user", "chat_id": "navigation", "text": f"question {i}", "turn_id": f"t{i}"})
        history.append(key, {"event": "message", "chat_id": "navigation", "text": "answer " + "x" * 200, "turn_id": f"t{i}"})
        history.append(key, {"event": "turn_end", "chat_id": "navigation", "turn_id": f"t{i}"})
    index = history.prompts(key)
    assert len(index) == 120
    latest = history.build(key, limit=3)
    assert len(latest["events"]) < len(index)
    target = index[42]
    stats = transcript.TranscriptReplayStats()
    page = history.build(key, after=target["cursor"], limit=6, stats=stats)
    assert page["events"][0]["projection_id"] == target["id"]
    assert page["events"][0]["text"] == "question 42"
    assert page["page"]["user_message_offset"] == 42
    assert page["page"]["has_more_before"] and page["page"]["has_more_after"]
    assert stats.parsed_records < 40  # Not the 360-record transcript.
    newer = history.build(key, after=page["page"]["after_cursor"], limit=6)
    older = history.build(key, before=newer["page"]["before_cursor"], limit=6)
    assert older["events"] == page["events"]
    # Warm navigation uses the committed revision and cached summaries.
    with monkeypatch.context() as patch:
        patch.setattr("nanoclaw.session.display_history.HistoryReader.navigation", lambda _: (_ for _ in ()).throw(AssertionError("replayed history")))
        assert history.prompts(key) == index
    history.append(key, {"event": "user", "chat_id": "navigation", "text": "new question"})
    assert len(history.prompts(key)) == 121


def test_oversized_turn_keeps_matching_navigation_identity(history, tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "_MAX_TRANSCRIPT_PAGE_RECORDS", 6)
    key = "websocket:large-turn"
    history.append(key, {"event": "user", "chat_id": "large-turn", "text": "target"})
    for i in range(20):
        history.append(key, {"event": "message", "chat_id": "large-turn", "kind": "tool_hint", "text": str(i)})
    history.append(key, {"event": "message", "chat_id": "large-turn", "text": "done"})
    history.append(key, {"event": "turn_end", "chat_id": "large-turn"})
    target = history.prompts(key)[0]
    page = history.build(key, after=target["cursor"], limit=40)
    assert page["page"]["truncated_oversized_turn"]
    assert page["events"][0]["projection_id"] == target["id"]
    assert len(target["answerPreview"]) <= 240




def test_summary_deltas_rebuild_after_append_rewrite_eviction_and_delete(history, tmp_path, monkeypatch):
    key = "websocket:index-delta"
    for i in range(100):
        history.append(key, {"event": "user", "text": f"question {i}", "turn_id": f"t{i}"})
        history.append(key, {"event": "turn_end", "turn_id": f"t{i}"})
    initial = history.prompt_response(key, None)
    assert len(initial["prompts"]) == 100
    history.append(key, {"event": "user", "text": "next", "turn_id": "next"})
    appended = history.prompt_response(key, initial["revision"])
    assert appended["replaceFrom"] == 100
    assert len(appended["prompts"]) == 1
    assert [*initial["prompts"], *appended["prompts"]] == history.prompts(key)
    # A late answer changes the last preview; it must replace that prompt, not append it.
    history.append(key, {"event": "message", "text": "answer", "turn_id": "next"})
    changed = history.prompt_response(key, appended["revision"])
    assert changed["replaceFrom"] == 100
    assert changed["prompts"][0]["answerPreview"] == "answer"
    delete_webui_session(history.sessions, history.store, key)
    history.append(key, {"event": "user", "text": "replacement"})
    replaced = history.prompt_response(key, changed["revision"])
    assert replaced["replaceFrom"] == 0
    assert len(replaced["prompts"]) == 1
    transcript._PROMPT_SNAPSHOTS.clear()  # Restart/eviction sends a complete response.
    assert history.prompt_response(key, replaced["revision"])["replaceFrom"] == 0
    delete_webui_session(history.sessions, history.store, key)
    assert history.prompt_response(key, replaced["revision"])["prompts"] == ()


@pytest.mark.parametrize("input_id", [None, "injected-guidance"])
def test_recovered_prompt_identity_time_and_cursor_survive_pages(history, tmp_path, monkeypatch, input_id):
    key = "websocket:recovered-navigation"
    messages = [{"role": "system", "content": "context"}]
    for i in range(2):
        user = {"role": "user", "content": "same guidance", "timestamp": f"2026-06-02T10:0{i}:00+00:00"}
        if input_id:
            user["webui_turn_id"] = f"{input_id}-{i}"
        messages.extend([user, {"role": "assistant", "content": f"answer {i}"}])
        history.append(key, {"event": "message", "text": f"answer {i}", "turn_id": f"execution-{i}"})
        history.append(key, {"event": "turn_end", "turn_id": f"execution-{i}"})
    def loader():
        return messages

    cold_page = history.build(key, limit=1, session_messages_loader=loader)
    index = history.prompts(key, session_messages_loader=loader)
    assert history.build(key, limit=1, session_messages_loader=loader) == cold_page
    assert len(index) == 2
    assert index[0]["id"] != index[1]["id"]
    assert [prompt["createdAt"] for prompt in index] == [1780394400000, 1780394460000]
    for position, prompt in enumerate(index):
        page = history.build(key, after=prompt["cursor"], limit=1, session_messages_loader=loader)
        user = next(event for event in page["events"] if event["event"] == "user_message")
        assert user["projection_id"] == prompt["id"]
        assert user["created_at_ms"] == prompt["createdAt"]
        assert user.get("turn_id") == (f"{input_id}-{position}" if input_id else None)
    assert history.prompts(key, session_messages_loader=loader) == index



@pytest.mark.parametrize("with_anchor", [False, True])
def test_indexed_backfill_preserves_unique_in_suffix_matching(history, tmp_path, monkeypatch, with_anchor):
    key = "websocket:repeated-answers"
    answers = ["same answer", "anchor answer", "same answer"] if with_anchor else ["same answer"] * 2
    messages = []
    for position, answer in enumerate(answers):
        user = {"role": "user", "content": f"question {position}", "webui_turn_id": f"input-{position}"}
        messages.extend([user, {"role": "assistant", "content": answer}])
        if position < len(answers) - 1:
            history.append(key, {"event": "user", "text": user["content"], "turn_id": user["webui_turn_id"]})
        history.append(key, {"event": "message", "text": answer})
        history.append(key, {"event": "turn_end"})
    prompts = history.prompts(key, session_messages_loader=lambda: messages)
    # A unique intervening answer advances the alignment past the first duplicate.
    assert len(prompts) == (len(answers) if with_anchor else 1)
    page = history.build(key, limit=1, session_messages=messages)
    users = [event for event in page["events"] if event["event"] == "user_message"]
    assert [user["text"] for user in users] == (["question 2"] if with_anchor else [])
    if with_anchor:
        assert users[0]["projection_id"] == prompts[-1]["id"]
