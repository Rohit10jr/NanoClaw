"""web_fetch over a mocked HTTP transport; web_search with a fake search function."""

import json

import httpx
import pytest

from nanoclaw.agent.tools import is_tool_error
from nanoclaw.agent.tools.web import UNTRUSTED_BANNER, WebFetchTool, WebSearchTool

ARTICLE = """<!doctype html><html><head><title>Cats</title><style>p{}</style></head><body>
<nav>Menu Home About Contact</nav>
<article><h1>All about cats</h1>
<p>Cats are small carnivorous mammals. They have been living with humans for thousands of years,
and are valued for companionship and their ability to hunt vermin.</p>
<p>See <a href="https://example.org/more">more facts</a> about cats and their long history with people.</p>
<ul><li>Whiskers</li><li>Tails</li></ul></article>
<script>alert('x')</script></body></html>"""


@pytest.fixture(autouse=True)
def _dns(fake_dns):
    return fake_dns


def site(routes):
    """A mock transport serving ``routes``: url -> (status, headers, body)."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        status, headers, body = routes[str(request.url)]
        return httpx.Response(status, headers=headers, content=body.encode() if isinstance(body, str) else body)

    return httpx.MockTransport(handler), seen


async def fetch(routes, url, **kwargs):
    transport, seen = site(routes)
    out = await WebFetchTool(transport=transport).execute(url=url, **kwargs)
    return out, seen


async def test_html_is_extracted_as_markdown_with_banner():
    out, _ = await fetch({"https://cats.test/": (200, {"content-type": "text/html"}, ARTICLE)}, "https://cats.test/")
    data = json.loads(out)
    assert data["extractor"] == "readability"
    assert data["status"] == 200 and data["untrusted"] is True and data["truncated"] is False
    text = data["text"]
    assert text.startswith(UNTRUSTED_BANNER + "\n\n# Cats")
    assert "Cats are small carnivorous mammals." in text
    assert "[more facts](https://example.org/more)" in text
    assert "Menu Home" not in text  # readability dropped the navigation
    assert "alert(" not in text and "p{}" not in text
    assert data["length"] == len(text)


def test_to_markdown_conversions():
    html_in = '<h2>Title</h2><p>Hi &amp; <a href="https://x.test">link</a></p><ul><li>one</li><li>two</li></ul>a<br>b'
    assert WebFetchTool._to_markdown(html_in) == "## Title\nHi & [link](https://x.test)\n\n- one\n- twoa\nb"


async def test_text_mode_has_no_markdown_links():
    out, _ = await fetch(
        {"https://cats.test/": (200, {"content-type": "text/html"}, ARTICLE)}, "https://cats.test/", extractMode="text"
    )
    text = json.loads(out)["text"]
    assert "more facts" in text and "](" not in text


async def test_json_is_pretty_printed():
    out, _ = await fetch({"https://api.test/x": (200, {"content-type": "application/json"}, '{"a": [1, 2]}')}, "https://api.test/x")
    data = json.loads(out)
    assert data["extractor"] == "json"
    assert data["text"].endswith('{\n  "a": [\n    1,\n    2\n  ]\n}')


async def test_plain_text_and_truncation():
    out, _ = await fetch({"https://t.test/": (200, {"content-type": "text/plain"}, "x" * 500)}, "https://t.test/", maxChars=100)
    data = json.loads(out)
    assert data["extractor"] == "raw" and data["truncated"] is True
    assert data["text"] == f"{UNTRUSTED_BANNER}\n\n" + "x" * 100


async def test_redirects_are_followed():
    routes = {
        "https://a.test/": (301, {"location": "/b"}, ""),
        "https://a.test/b": (200, {"content-type": "text/plain"}, "landed"),
    }
    out, seen = await fetch(routes, "https://a.test/")
    data = json.loads(out)
    assert data["finalUrl"] == "https://a.test/b" and data["text"].endswith("landed")
    assert seen == ["https://a.test/", "https://a.test/b"]


async def test_redirect_to_non_http_scheme_is_blocked():
    out, _ = await fetch({"https://a.test/": (302, {"location": "file:///etc/passwd"}, "")}, "https://a.test/")
    assert is_tool_error(out)
    assert json.loads(out)["error"] == "Redirect blocked: Only http/https allowed, got 'file'"


async def test_too_many_redirects():
    routes = {f"https://a.test/{i}": (302, {"location": f"/{i + 1}"}, "") for i in range(10)}
    out, seen = await fetch(routes, "https://a.test/0")
    assert json.loads(out)["error"] == "Too many redirects: exceeded limit of 5"
    assert len(seen) == 6


@pytest.mark.parametrize(
    ("url", "error"),
    [
        ("ftp://x.test/f", "URL validation failed: Only http/https allowed, got 'ftp'"),
        ("not a url", "URL validation failed: Only http/https allowed, got 'none'"),
        ("https://", "URL validation failed: Missing domain"),
    ],
)
async def test_invalid_urls(url, error):
    out, seen = await fetch({}, url)
    assert is_tool_error(out) and json.loads(out)["error"] == error
    assert seen == []


async def test_http_error_status():
    out, _ = await fetch({"https://a.test/": (404, {}, "nope")}, "https://a.test/")
    assert is_tool_error(out)
    assert "404" in json.loads(out)["error"]


async def test_url_is_cleaned_of_quotes_and_backticks():
    out, seen = await fetch({"https://a.test/": (200, {"content-type": "text/plain"}, "ok")}, " `https://a.test/` ")
    assert seen == ["https://a.test/"] and not is_tool_error(out)


# --- web_search -------------------------------------------------------------------


async def test_search_formats_results_like_nanobot():
    def fake_search(query, n):
        assert (query, n) == ("cats", 2)
        return [
            {"title": "<b>Cat</b> facts", "url": "https://a.test", "content": "Cats   are\n\n\n\ngreat"},
            {"title": "More", "url": "https://b.test", "content": ""},
        ]

    out = await WebSearchTool(search_fn=fake_search).execute(query="cats", count=2)
    assert out == "Results for: cats\n\n1. Cat facts\n   https://a.test\n   Cats are\n\ngreat\n2. More\n   https://b.test"


async def test_search_count_is_clamped_and_defaults_to_5():
    counts = []
    tool = WebSearchTool(search_fn=lambda q, n: counts.append(n) or [])
    assert await tool.execute(query="x") == "No results for: x"
    await tool.execute(query="x", count=50)
    assert counts == [5, 10]


async def test_search_failure_is_an_error():
    def broken(query, n):
        raise RuntimeError("rate limited")

    out = await WebSearchTool(search_fn=broken).execute(query="x")
    assert is_tool_error(out) and out == "Error: DuckDuckGo search failed (rate limited)"
