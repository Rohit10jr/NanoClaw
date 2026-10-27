"""A small MCP server for tests and the nanobot comparison (task 17.4).

Run ``python mcp_demo_server.py`` for stdio, or
``python mcp_demo_server.py http PORT`` for streamable HTTP on 127.0.0.1.
Only JSON-RPC may go to stdout in stdio mode, so nothing here prints.
"""

from __future__ import annotations

import asyncio
import sys

from mcp.server.fastmcp import FastMCP, Image
from pydantic import BaseModel

server = FastMCP("demo")


class Point(BaseModel):
    x: int
    y: int


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def echo(text: str, repeat: int = 1) -> str:
    """Return the text, repeated."""
    return " ".join([text] * repeat)


@server.tool()
def fail(message: str) -> str:
    """Always fails (the result has isError set)."""
    raise ValueError(message)


@server.tool()
async def slow(seconds: float) -> str:
    """Sleep, then answer."""
    await asyncio.sleep(seconds)
    return f"slept {seconds}"


@server.tool()
def describe_point(point: Point, label: str | None = None) -> str:
    """A schema with a $ref (Point) and an anyOf-with-null (label)."""
    return f"{label or 'point'}: ({point.x}, {point.y})"


@server.tool(name="a_tool_with_a_rather_long_name_that_needs_the_hash_suffix_to_fit")
def long_named() -> str:
    """A name longer than 64 characters once prefixed."""
    return "long ok"


@server.tool()
def two_blocks() -> list[str]:
    """Returns two text blocks."""
    return ["first", "second"]


@server.tool()
def tiny_image() -> Image:
    """Returns a 1x1 PNG."""
    import base64

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    return Image(data=png, format="png")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "http":
        server.settings.port = int(sys.argv[2])
        server.settings.host = "127.0.0.1"
        server.run(transport="streamable-http")
    else:
        server.run()
