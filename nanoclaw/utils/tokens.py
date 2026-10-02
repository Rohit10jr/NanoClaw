"""Estimating how many tokens a request uses.

Mirrors nanobot's ``utils/token_encoding.py`` and the token helpers in
``utils/helpers.py``. Counts use tiktoken's ``cl100k_base`` encoding (which may
differ slightly from the model's own tokenizer). If tiktoken can't load its
vocabulary (it is downloaded once, then cached), the UTF-8 byte count is used
instead, which overestimates and so compacts early rather than late.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

TRUNCATED_SUFFIX = "\n... (truncated)"
# nanobot caches the vocabulary under its data directory.
DEFAULT_TIKTOKEN_CACHE = Path("~/.nanoclaw/cache/tiktoken")


@lru_cache(maxsize=1)
def get_token_encoding() -> Any | None:
    """The ``cl100k_base`` encoding, or ``None`` if it can't be loaded."""
    try:
        if "TIKTOKEN_CACHE_DIR" not in os.environ and "DATA_GYM_CACHE_DIR" not in os.environ:
            os.environ["TIKTOKEN_CACHE_DIR"] = str(DEFAULT_TIKTOKEN_CACHE.expanduser())
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _message_text_parts(msg: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    content = msg.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if isinstance(text, str) and text:
                    parts.append(text)
    if msg.get("tool_calls"):
        parts.append(json.dumps(msg["tool_calls"], ensure_ascii=False))
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        parts.append(reasoning)
    for key in ("name", "tool_call_id"):
        value = msg.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    return parts


def estimate_prompt_tokens(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> tuple[int, str]:
    """Estimated input tokens of a request, and how they were counted.

    Counts everything providers send: content, tool calls, reasoning, names,
    tool definitions, plus 4 tokens of framing per message.
    (nanobot ``_estimate_prompt_tokens_with_source``)
    """
    parts = [part for msg in messages for part in _message_text_parts(msg)]
    payload = "\n".join(parts)
    overhead = len(messages) * 4
    enc = get_token_encoding()
    if enc is not None:
        try:
            tool_tokens = 0
            if tools:
                rendered = json.dumps(tools, ensure_ascii=False)
                tool_tokens = len(enc.encode_ordinary(("\n" if parts else "") + rendered))
            message_tokens = len(enc.encode_ordinary(payload)) if payload else 0
            return message_tokens + tool_tokens + overhead, "tiktoken"
        except Exception:
            pass
    tool_payload = ("\n" if payload else "") + json.dumps(tools, ensure_ascii=False) if tools else ""
    return len((payload + tool_payload).encode("utf-8")) + overhead, "heuristic"


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimated tokens of one saved message (at least 4). (nanobot ``estimate_message_tokens``)"""
    content = message.get("content")
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if isinstance(text, str) and text:
                    parts.append(text)
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
    elif content is not None:
        parts.append(json.dumps(content, ensure_ascii=False))
    for key in ("name", "tool_call_id"):
        value = message.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        parts.append(reasoning)
    payload = "\n".join(parts)
    if not payload:
        return 4
    enc = get_token_encoding()
    if enc is not None:
        try:
            return max(4, len(enc.encode_ordinary(payload)) + 4)
        except Exception:
            pass
    return max(4, len(payload.encode("utf-8")) + 4)


def _truncate_to_utf8_bytes(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    return text if len(encoded) <= max_bytes else encoded[:max_bytes].decode("utf-8", errors="ignore")


def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Cut ``text`` to at most ``max_tokens`` tokens, ending with a truncation marker."""
    if max_tokens <= 0:
        return text
    enc = get_token_encoding()
    if enc is not None:
        try:
            tokens = enc.encode_ordinary(text)
            if len(tokens) <= max_tokens:
                return text
            body_budget = max_tokens - len(enc.encode_ordinary(TRUNCATED_SUFFIX))
            if body_budget <= 0:
                return enc.decode(tokens[:max_tokens])
            for candidate in range(body_budget, -1, -1):
                result = enc.decode(tokens[:candidate]) + TRUNCATED_SUFFIX
                if len(enc.encode_ordinary(result)) <= max_tokens:
                    return result
            return enc.decode(tokens[:max_tokens])
        except Exception:
            pass
    if len(text.encode("utf-8")) <= max_tokens:
        return text
    suffix_bytes = len(TRUNCATED_SUFFIX.encode("utf-8"))
    if max_tokens <= suffix_bytes:
        return _truncate_to_utf8_bytes(text, max_tokens)
    return _truncate_to_utf8_bytes(text, max_tokens - suffix_bytes) + TRUNCATED_SUFFIX
