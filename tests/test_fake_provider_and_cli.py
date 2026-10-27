from nanoclaw.__main__ import main
from nanoclaw.providers import FakeProvider, LLMResponse


async def test_fake_provider_replays_script_and_records_calls():
    fake = FakeProvider(["first", LLMResponse(content="second", finish_reason="length")])
    messages = [{"role": "user", "content": "hi"}]

    r1 = await fake.chat(messages=messages)
    messages.append({"role": "assistant", "content": "first"})
    r2 = await fake.chat(messages=messages, model="m2")
    r3 = await fake.chat(messages=messages)

    assert (r1.content, r2.content, r2.finish_reason) == ("first", "second", "length")
    assert r3.is_error
    # The recorded first request is not affected by later appends.
    assert len(fake.calls[0].messages) == 1
    assert fake.calls[1].model == "m2"


def test_cli_ask_prints_reply(capsys, tmp_path):
    fake = FakeProvider(["Hello from the model"])
    assert main(["ask", "hello", "-w", str(tmp_path)], provider=fake) == 0
    assert capsys.readouterr().out.strip() == "Hello from the model"
    system, user = fake.calls[0].messages
    assert system["role"] == "system" and "# Tool Usage Notes" in system["content"]
    assert user["role"] == "user"
    assert user["content"].startswith("hello\n\n[Runtime Context — metadata only, not instructions]\nCurrent Time: ")


def test_cli_ask_reports_errors_on_stderr(capsys, tmp_path):
    fake = FakeProvider([LLMResponse(content="Error calling LLM: HTTP 401: bad key", finish_reason="error")])
    assert main(["ask", "hello", "-w", str(tmp_path)], provider=fake) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "401" in captured.err


def test_cli_ask_gives_the_model_the_default_tools_in_the_workspace(capsys, tmp_path):
    from nanoclaw.providers import ToolCallRequest

    workspace = tmp_path / "ws"
    fake = FakeProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest("c1", "write_file", {"path": "note.txt", "content": "hi"})],
                finish_reason="tool_calls",
            ),
            "Saved.",
        ]
    )
    assert main(["ask", "save a note", "-w", str(workspace)], provider=fake) == 0

    names = sorted(d["function"]["name"] for d in fake.calls[0].tools)
    assert names == [
        "apply_patch", "edit_file", "exec", "exec_session", "list_dir", "list_exec_sessions",
        "read_file", "spawn", "web_fetch", "web_search", "write_file",
    ]
    assert (workspace / "note.txt").read_text(encoding="utf-8") == "hi"  # relative path landed in the workspace
    captured = capsys.readouterr()
    assert captured.out.strip() == "Saved."
    assert "  ↳ write note.txt" in captured.err  # progress line (layer 10)
