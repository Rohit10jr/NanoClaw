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


def test_cli_ask_prints_reply(capsys):
    fake = FakeProvider(["Hello from the model"])
    assert main(["ask", "hello"], provider=fake) == 0
    assert capsys.readouterr().out.strip() == "Hello from the model"
    assert fake.calls[0].messages == [{"role": "user", "content": "hello"}]


def test_cli_ask_reports_errors_on_stderr(capsys):
    fake = FakeProvider([LLMResponse(content="Error calling LLM: HTTP 401: bad key", finish_reason="error")])
    assert main(["ask", "hello"], provider=fake) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "401" in captured.err
