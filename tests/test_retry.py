"""LLMProvider.chat_with_retry and error classification.

Adapted from nanobot tests/providers/test_provider_retry.py and
test_provider_error_metadata.py. Sleeping is patched out, as there.
"""

import asyncio

import pytest

from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.memory import MemoryArchiver, MemoryStore
from nanoclaw.providers import FakeProvider, LLMProvider, LLMResponse
from nanoclaw.providers.base import RETRY_AFTER_BUFFER
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.utils.templates import sync_workspace_templates

USER = [{"role": "user", "content": "hello"}]


class ScriptedProvider(LLMProvider):
    def __init__(self, responses):
        super().__init__(default_model="test-model")
        self._responses = list(responses)
        self.calls = 0
        self.last_kwargs: dict = {}

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.calls += 1
        self.last_kwargs = kwargs
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def error(content="boom", **kwargs) -> LLMResponse:
    return LLMResponse(content=content, finish_reason="error", **kwargs)


@pytest.fixture
def delays(monkeypatch):
    recorded: list[float] = []

    async def fake_sleep(delay):
        recorded.append(delay)

    monkeypatch.setattr("nanoclaw.providers.base.asyncio.sleep", fake_sleep)
    return recorded


# --- standard mode -------------------------------------------------------------


async def test_retries_transient_error_then_succeeds(delays):
    provider = ScriptedProvider([error("429 rate limit"), LLMResponse(content="ok")])
    response = await provider.chat_with_retry(messages=USER)
    assert response.content == "ok"
    assert provider.calls == 2
    assert delays == [1]


async def test_does_not_retry_non_transient_error(delays):
    provider = ScriptedProvider([error("401 unauthorized")])
    response = await provider.chat_with_retry(messages=USER)
    assert response.content == "401 unauthorized"
    assert provider.calls == 1 and delays == []


async def test_returns_final_error_after_four_attempts(delays):
    provider = ScriptedProvider(
        [error("429 rate limit a"), error("429 rate limit b"), error("429 rate limit c"), error("503 final server error")]
    )
    progress: list[str] = []

    async def on_wait(text):
        progress.append(text)

    response = await provider.chat_with_retry(messages=USER, on_retry_wait=on_wait)
    assert response.content == "503 final server error"
    assert provider.calls == 4
    assert delays == [1, 2, 4]
    assert progress[0] == "Model request failed, retry in 1s (attempt 1)."
    assert progress[-1] == "Model request failed after 4 attempts, giving up."


async def test_cancellation_propagates():
    provider = ScriptedProvider([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await provider.chat_with_retry(messages=USER)


async def test_exceptions_are_classified_and_retried(delays):
    class APITimeoutError(Exception):
        pass

    provider = ScriptedProvider([APITimeoutError("slow"), LLMResponse(content="ok")])
    response = await provider.chat_with_retry(messages=USER)
    assert response.content == "ok" and delays == [1]


async def test_passes_request_settings_through():
    provider = ScriptedProvider([LLMResponse(content="ok")])
    await provider.chat_with_retry(messages=USER, model="m", max_tokens=10, temperature=0.1, tool_choice="auto")
    assert provider.last_kwargs == {
        "messages": USER,
        "tools": None,
        "model": "m",
        "max_tokens": 10,
        "temperature": 0.1,
        "tool_choice": "auto",
    }


# --- structured metadata ---------------------------------------------------------


async def test_uses_retry_after_text_and_reports_the_wait(delays):
    provider = ScriptedProvider([error("429 rate limit, retry after 7s"), LLMResponse(content="ok")])
    progress: list[str] = []

    async def on_wait(text):
        progress.append(text)

    response = await provider.chat_with_retry(messages=USER, on_retry_wait=on_wait)
    assert response.content == "ok"
    assert delays == [7.0 + RETRY_AFTER_BUFFER]
    assert f"{int(7 + RETRY_AFTER_BUFFER)}s" in progress[0]


async def test_prefers_structured_retry_after(delays):
    provider = ScriptedProvider(
        [error("429 rate limit, retry after 99s", error_retry_after_s=0.2), LLMResponse(content="ok")]
    )
    await provider.chat_with_retry(messages=USER)
    assert delays == [0.2 + RETRY_AFTER_BUFFER]


async def test_uses_response_retry_after(delays):
    provider = ScriptedProvider([error("429 rate limit", retry_after=9.0), LLMResponse(content="ok")])
    await provider.chat_with_retry(messages=USER)
    assert delays == [9.0 + RETRY_AFTER_BUFFER]


async def test_long_waits_are_chunked_with_heartbeats(delays):
    provider = ScriptedProvider([error("overloaded", retry_after=60), LLMResponse(content="ok")])
    progress: list[str] = []

    async def on_wait(text):
        progress.append(text)

    await provider.chat_with_retry(messages=USER, on_retry_wait=on_wait)
    assert delays == [30, 30, 1]
    assert [p.split(" in ")[1] for p in progress] == ["61s (attempt 1).", "31s (attempt 1).", "1s (attempt 1)."]


@pytest.mark.parametrize(
    ("response", "retried"),
    [
        (error("request failed", error_status_code=409), True),
        (error("request failed", error_kind="timeout"), True),
        (error("429 rate limit", error_should_retry=False), False),
        (error('{"error":{"type":"insufficient_quota"}}', error_status_code=429, error_type="insufficient_quota"), False),
        (error("rate limited", error_status_code=429, error_code="rate_limit_exceeded", error_retry_after_s=0.2), True),
        # ZhiPu's rate limit: code 1302 with Chinese text.
        (error("Error: {'code': '1302', 'message': '您的账户已达到速率限制，请您控制请求频率'}"), True),
        (error("Error: {'code': '1302', 'message': '速率限制'}", error_status_code=429, error_code="1302"), True),
        (error("forbidden", error_status_code=403), False),
        (error("bad key", error_kind="authentication"), False),
        (error("payment required", error_status_code=402), False),
    ],
)
async def test_structured_classification(delays, response, retried):
    provider = ScriptedProvider([response, LLMResponse(content="ok")])
    result = await provider.chat_with_retry(messages=USER)
    assert provider.calls == (2 if retried else 1)
    assert (result.content == "ok") is retried


@pytest.mark.parametrize(
    ("expected", "kwargs"),
    [
        (True, {"error_status_code": 402}),
        (True, {"error_type": "insufficient_quota"}),
        (True, {"content": "429 You exceeded your current quota"}),
        (False, {"error_status_code": 429, "error_type": "rate_limit_exceeded"}),
    ],
)
def test_is_arrearage_response(expected, kwargs):
    response = LLMResponse(finish_reason="error", **{"content": "boom", **kwargs})
    assert LLMProvider.is_arrearage_response(response) is expected


def test_error_response_from_exception_classification():
    class RateLimitError(Exception):
        status_code = 429

    class AuthenticationError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    rate = LLMProvider._error_response_from_exception(RateLimitError("slow down"))
    assert (rate.error_kind, rate.error_should_retry, rate.error_status_code) == ("rate_limit", True, 429)
    auth = LLMProvider._error_response_from_exception(AuthenticationError("no"))
    assert (auth.error_kind, auth.error_should_retry) == ("authentication", None)
    conn = LLMProvider._error_response_from_exception(APIConnectionError(""))
    assert (conn.error_kind, conn.content) == ("connection", "Error calling LLM: APIConnectionError")


# --- persistent mode --------------------------------------------------------------


async def test_persistent_stops_after_ten_identical_errors(delays):
    provider = ScriptedProvider([*[error("429 rate limit") for _ in range(10)], LLMResponse(content="ok")])
    progress: list[str] = []

    async def on_wait(text):
        progress.append(text)

    response = await provider.chat_with_retry(messages=USER, retry_mode="persistent", on_retry_wait=on_wait)
    assert response.content == "429 rate limit"
    assert provider.calls == 10
    assert delays == [1, 2, 4, 4, 4, 4, 4, 4, 4]
    assert progress[-1] == "Persistent retry stopped after 10 identical errors."
    assert progress[3] == "Model request failed, persistent retry in 4s (attempt 4)."


async def test_persistent_keeps_going_past_four_attempts_while_errors_change(delays):
    provider = ScriptedProvider([*[error(f"503 busy {i}") for i in range(12)], LLMResponse(content="ok")])
    response = await provider.chat_with_retry(messages=USER, retry_mode="persistent")
    assert response.content == "ok" and provider.calls == 13


async def test_persistent_caps_waits_at_sixty_seconds(delays):
    provider = ScriptedProvider([error("429 rate limit", retry_after=300), LLMResponse(content="ok")])
    await provider.chat_with_retry(messages=USER, retry_mode="persistent")
    assert sum(delays) == 60


async def test_standard_mode_does_not_cap_retry_after(delays):
    provider = ScriptedProvider([error("429 rate limit", retry_after=300), LLMResponse(content="ok")])
    await provider.chat_with_retry(messages=USER)
    assert sum(delays) == 301


# --- Retry-After parsing (nanobot base.py:1441-1525) --------------------------------


def test_extract_retry_after_common_formats():
    assert LLMProvider._extract_retry_after('{"error":{"retry_after":20}}') == 20.0
    assert LLMProvider._extract_retry_after("Rate limit reached, please try again in 20s") == 20.0
    assert LLMProvider._extract_retry_after("retry-after: 20") == 20.0
    assert LLMProvider._extract_retry_after("please wait 500ms before retry") == 0.5
    assert LLMProvider._extract_retry_after("no hint here") is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Rate limit reached for gpt-x. Please try again in 1m30s.", 90.0),
        ("Please try again in 2m0.5s", 120.5),
        ("Please try again in 1h2m3s", 3723.0),
        ("Please try again in 6m0s. Visit https://example.com", 360.0),
        ("Please try again in 1.5s", 1.5),
        ("Please try again in 20ms", 0.1),
        ("Please try again in 30 seconds", 30.0),
        ("Please try again in 2 minutes", 120.0),
    ],
)
def test_extract_retry_after_compound_durations(message, expected):
    assert LLMProvider._extract_retry_after(message) == pytest.approx(expected)


def test_retry_after_beats_try_again_in():
    assert LLMProvider._extract_retry_after("retry after 5s; otherwise try again in 1m30s") == 5.0


def test_retry_after_headers():
    assert LLMProvider._extract_retry_after_from_headers({"Retry-After": "20"}) == 20.0
    assert LLMProvider._extract_retry_after_from_headers({"retry-after": "20"}) == 20.0
    assert LLMProvider._extract_retry_after_from_headers({"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}) == 0.1
    assert LLMProvider._extract_retry_after_from_headers({"retry-after-ms": "250"}) == 0.25
    assert LLMProvider._extract_retry_after_from_headers({"Retry-After-Ms": "1000"}) == 1.0
    assert LLMProvider._extract_retry_after_from_headers({"retry-after-ms": "500", "retry-after": "10"}) == 0.5
    assert LLMProvider._extract_retry_after_from_headers({"retry-after": "soon"}) is None
    assert LLMProvider._extract_retry_after_from_headers(None) is None


# --- callers ------------------------------------------------------------------------


async def test_fake_provider_makes_one_call_by_default():
    fake = FakeProvider([error("503 busy"), "ok"])
    response = await fake.chat_with_retry(messages=USER)
    assert response.content == "503 busy" and len(fake.calls) == 1


async def test_memory_archive_retries(tmp_path, delays):
    ws = tmp_path / "ws"
    sync_workspace_templates(ws)
    store = MemoryStore(ws)
    fake = FakeProvider([error("Error calling LLM: HTTP 502: bad gateway"), "- [durable] likes tea"], retries=True)
    runtime = LLMRuntime(provider=fake, model=None, max_tokens=1000, temperature=0.1, context_window_tokens=128_000)
    source = [{"role": "user", "content": "I like tea"}, {"role": "assistant", "content": "Noted."}]
    summary = await MemoryArchiver(store, ContextBuilder(ws).build_messages, lambda: []).archive(
        source, runtime=runtime, session_key="k", history=source, request_tools=[]
    )
    assert summary == "- [durable] likes tea"
    assert delays == [1] and fake.last_retry_mode == "standard"


async def test_loop_passes_its_retry_mode_to_the_main_run(tmp_path):
    ws = tmp_path / "ws"
    sync_workspace_templates(ws)
    fake = FakeProvider(["hi"])
    loop = AgentLoop(fake, ws, provider_retry_mode="persistent")
    await loop.process_direct("hello")
    assert fake.last_retry_mode == "persistent"
