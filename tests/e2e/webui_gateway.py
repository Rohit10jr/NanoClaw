"""Run ``nanoclaw gateway`` with a scripted provider for the browser e2e test.

Usage: ``python webui_gateway.py ROOT WEBUI_DIST PORT [SECRET]``

Everything (config, workspace, sessions) lives under ROOT; run it with
``HOME=ROOT/home`` so nothing touches the real ``~/.nanoclaw``. The provider
picks its reply from keywords in the latest user message, so the browser
script can drive every scenario without a fixed script order:

- ``markdown``   -> a streamed Markdown reply (heading, code block, list)
- ``inspect``    -> ``<think>`` reasoning + a ``read_file`` call on notes.txt, then an answer
- ``sleepy``     -> an ``exec`` call running ``sleep 8`` in the foreground (for ``/stop``)
- ``explode``    -> the model call raises (a failed turn)
- ``slowstream`` -> a reply streamed over ~6 s (for reload during a turn)
- anything else  -> ``Echo: <text>``

Title requests get ``<first three words> topic``.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from nanoclaw.__main__ import main
from nanoclaw.providers import FakeProvider
from nanoclaw.providers.base import LLMResponse, ToolCallRequest

MARKDOWN_REPLY = (
    "# Release notes\n\n"
    "Here is a sample:\n\n"
    "```python\n"
    "print('hello from nanoclaw')\n"
    "```\n\n"
    "- first item\n"
    "- second item\n"
)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


class RoutingProvider(FakeProvider):
    """A FakeProvider whose reply depends on the conversation, with delays."""

    def __init__(self) -> None:
        super().__init__([])
        self._call_seq = 0

    def _plan(self, messages: list[dict[str, Any]]) -> tuple[list[str], LLMResponse, float]:
        system = _text(messages[0].get("content")) if messages else ""
        if system.startswith("You write short, neutral chat titles"):
            prompt = _text(messages[-1].get("content"))
            match = re.search(r"^User: (.*)$", prompt, flags=re.MULTILINE)
            words = re.findall(r"[A-Za-z0-9]+", match.group(1) if match else "chat")[:3]
            title = " ".join(words).capitalize() + " topic"
            return [], LLMResponse(content=title), 0.0
        user = next((_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
        user = user.split("[Runtime Context")[0].strip()
        lowered = user.lower()
        after_tool = bool(messages) and messages[-1].get("role") == "tool"
        self._call_seq += 1
        call_id = f"call_{self._call_seq}"
        if "inspect" in lowered:
            if after_tool:
                text = "The notes file says: **the launch code is 42**."
                return _chunks(text, 8), LLMResponse(content=text), 0.2
            text = "<think>The user wants the notes. I should read notes.txt first.</think>Let me read the notes."
            call = ToolCallRequest(call_id, "read_file", {"path": "notes.txt"})
            return _chunks(text, 10), LLMResponse(content=text, tool_calls=[call], finish_reason="tool_calls"), 0.2
        if "sleepy" in lowered:
            if after_tool:
                return ["Slept well."], LLMResponse(content="Slept well."), 0.0
            call = ToolCallRequest(call_id, "exec", {"command": "sleep 8", "yield_time_ms": 15000})
            return [], LLMResponse(content="", tool_calls=[call], finish_reason="tool_calls"), 0.0
        if "explode" in lowered:
            raise RuntimeError("scripted provider failure")
        if "markdown" in lowered:
            return _chunks(MARKDOWN_REPLY, 12), LLMResponse(content=MARKDOWN_REPLY), 0.12
        if "slowstream" in lowered:
            text = " ".join(f"word{i}" for i in range(1, 21)) + " done-slow."
            return re.findall(r"\S+\s*", text), LLMResponse(content=text), 0.3
        text = f"Echo: {user.strip()}"
        return re.findall(r"\S+\s*", text), LLMResponse(content=text), 0.03

    async def chat_stream(  # type: ignore[override]
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta=None,
        on_thinking_delta=None,
        on_tool_call_delta=None,
    ) -> LLMResponse:
        deltas, response, delay = self._plan(messages)
        for text in deltas:
            if delay:
                await asyncio.sleep(delay)
            if on_content_delta:
                await on_content_delta(text)
        for index, call in enumerate(response.tool_calls):
            if on_tool_call_delta:
                await on_tool_call_delta(
                    {
                        "index": index,
                        "call_id": call.id,
                        "name": call.name,
                        "arguments_delta": json.dumps(call.arguments),
                    }
                )
        return response

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7, tool_choice=None):  # type: ignore[override]
        return self._plan(messages)[1]


def run(root: Path, dist: str, port: int, secret: str = "") -> int:
    root.mkdir(parents=True, exist_ok=True)
    ws_dir = root / "ws"
    ws_dir.mkdir(exist_ok=True)
    (ws_dir / "notes.txt").write_text("the launch code is 42\n")
    websocket: dict[str, Any] = {"enabled": True, "port": port, "webuiDist": dist}
    if secret:
        websocket["tokenIssueSecret"] = secret
    cfg = {
        "channels": {"websocket": websocket},
        "gateway": {"port": port + 1, "heartbeat": {"enabled": False}},
        "agents": {"defaults": {"dream": {"enabled": False}}},
        "providers": {"custom": {"apiKey": "sk-e2e", "apiBase": "http://127.0.0.1:9/v1"}},
    }
    (root / "config.json").write_text(json.dumps(cfg))
    return main(["gateway", "-c", str(root / "config.json"), "-w", str(ws_dir)], provider=RoutingProvider())


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else ""))
