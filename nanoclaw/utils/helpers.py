"""Text helpers shared by chat channels. (nanobot utils/helpers.py, ``split_message`` only)"""

from __future__ import annotations


def split_message(content: str, max_len: int = 2000) -> list[str]:
    """
    Split content into chunks within max_len, preferring line breaks.

    Args:
        content: The text content to split.
        max_len: Maximum length per chunk (default 2000 for Discord compatibility).

    Returns:
        List of message chunks, each within max_len.
    """
    if not content:
        return []
    # Non-positive max_len cannot advance the cut pointer; return unsplit.
    if max_len <= 0:
        return [content]
    if len(content) <= max_len:
        return [content]
    original_content = content
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            if content.strip():
                chunks.append(content)
            break
        cut = content[:max_len]
        # Consume only the newline itself so indentation starts the next chunk.
        newline_pos = cut.rfind("\n")
        if newline_pos >= 0:
            # Exclude both bytes of a CRLF boundary from the emitted chunk.
            line_end = newline_pos
            if line_end > 0 and content[line_end - 1] == "\r":
                line_end -= 1
            chunk = content[:line_end]
            if chunk.strip():
                chunks.append(chunk)
            content = content[newline_pos + 1 :]
            continue

        # Keep the existing word-boundary behavior, but avoid emitting a
        # whitespace-only chunk when an indented line exceeds max_len.
        space_pos = cut.rfind(" ")
        if space_pos > 0 and cut[:space_pos].strip():
            chunks.append(content[:space_pos])
            content = content[space_pos:].lstrip(" \t")
            # A space boundary may sit immediately before a line break. Drop
            # that delimiter too, without stripping the next line's indent.
            if content.startswith("\r\n"):
                content = content[2:]
            elif content.startswith("\n"):
                content = content[1:]
            continue

        # Do not split between the two code points of a CRLF delimiter.
        if cut.endswith("\r") and content[max_len : max_len + 1] == "\n":
            chunk = cut[:-1]
            if chunk.strip():
                chunks.append(chunk)
            content = content[max_len + 1 :]
            continue

        chunk = content[:max_len]
        if chunk.strip():
            chunks.append(chunk)
        content = content[max_len:]
        if not chunk.strip():
            # Keep any remaining indentation so the final non-blank chunk can
            # retain as much of it as the channel limit permits.
            continue
        # A delimiter can sit immediately after the hard-break boundary. Keep
        # ordinary space trimming, but consume only the newline so indentation
        # on the following line is preserved.
        content = content.lstrip(" \t")
        if content.startswith("\r\n"):
            content = content[2:]
        elif content.startswith("\n"):
            content = content[1:]
    # Preserve the historical non-empty-input contract for callers that take
    # the first chunk directly. This fallback is only reachable for content
    # made entirely of whitespace.
    return chunks or [original_content[:max_len]]
