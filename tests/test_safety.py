"""Layer 9: private-network blocking and workspace restriction."""

import json
import os
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from nanoclaw.__main__ import main
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.subagent import SubagentManager
from nanoclaw.agent.tools import ToolRegistry, build_default_tools, is_tool_error
from nanoclaw.agent.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool, EditFileTool
from nanoclaw.agent.tools.registry import RETRY_HINT
from nanoclaw.agent.tools.shell import ExecTool
from nanoclaw.agent.tools.web import WebFetchTool
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.security import network
from nanoclaw.security.network import (
    PinnedDNSAsyncTransport,
    UnsafeURLRequestError,
    configure_ssrf_whitelist,
    configure_ssrf_whitelist_from_env,
    contains_internal_url,
    httpx_env_proxy_mounts,
    pin_resolved_url_dns,
    resolve_url_target,
    validate_url_target,
)
from nanoclaw.security.workspace_policy import WORKSPACE_BOUNDARY_NOTE

NOTE_SHELL = (
    "\n\nNote: this is a hard policy boundary, not a transient failure. "
    "Do NOT retry with shell tricks (symlinks, base64 piping, alternative "
    "tools, working_dir overrides). If the user genuinely needs this "
    "resource, tell them you cannot reach it under the current "
    "restrict_to_workspace policy and ask how to proceed."
)


@pytest.fixture(autouse=True)
def _dns_and_whitelist(fake_dns):
    configure_ssrf_whitelist([])
    yield fake_dns
    configure_ssrf_whitelist([])


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    (path / "memory").mkdir(parents=True)
    (path / "memory" / "history.jsonl").write_text('{"cursor": 1}\n')
    (path / "notes.txt").write_text("inside\n")
    return path


@pytest.fixture
def outside(tmp_path):
    path = tmp_path / "secret.txt"
    path.write_text("outside\n")
    return path


# --- network: which addresses are blocked ----------------------------------


@pytest.mark.parametrize(
    "url, error",
    [
        ("http://localhost/x", "Blocked: localhost resolves to private/internal address 127.0.0.1"),
        ("http://127.0.0.1:8080/", "Blocked: 127.0.0.1 resolves to private/internal address 127.0.0.1"),
        ("http://internal.test/", "Blocked: internal.test resolves to private/internal address 10.0.0.5"),
        ("http://metadata.test/", "Blocked: metadata.test resolves to private/internal address 169.254.169.254"),
        ("http://rebind.test/", "Blocked: rebind.test resolves to private/internal address 127.0.0.1"),
        ("http://v6local.test/", "Blocked: v6local.test resolves to private/internal address ::1"),
        ("http://mapped.test/", "Blocked: mapped.test resolves to private/internal address ::ffff:127.0.0.1"),
        ("http://[::1]/", "Blocked: ::1 resolves to private/internal address ::1"),
        ("http://192.168.1.1/", "Blocked: 192.168.1.1 resolves to private/internal address 192.168.1.1"),
        ("http://100.100.1.1/", "Blocked: 100.100.1.1 resolves to private/internal address 100.100.1.1"),
        ("http://172.16.0.1/", "Blocked: 172.16.0.1 resolves to private/internal address 172.16.0.1"),
        ("http://172.31.255.1/", "Blocked: 172.31.255.1 resolves to private/internal address 172.31.255.1"),
        ("http://0.0.0.0/", "Blocked: 0.0.0.0 resolves to private/internal address 0.0.0.0"),
        ("http://unknown.invalid/", "Cannot resolve hostname: unknown.invalid"),
        ("ftp://public.test/", "Only http/https allowed, got 'ftp'"),
        ("https://", "Missing domain"),
    ],
)
def test_blocked_targets(url, error):
    assert validate_url_target(url) == (False, error)


def test_public_targets_return_the_checked_addresses():
    assert resolve_url_target("https://public.test/page") == (True, "", ("93.184.216.34",))
    assert validate_url_target("http://8.8.8.8/") == (True, "")
    assert validate_url_target("http://172.32.0.1/") == (True, "")  # just outside 172.16/12


def test_whitelist(monkeypatch):
    configure_ssrf_whitelist(["10.0.0.0/8", "not-a-cidr"])
    assert validate_url_target("http://internal.test/") == (True, "")
    assert validate_url_target("http://metadata.test/")[0] is False
    monkeypatch.setenv("NANOCLAW_SSRF_WHITELIST", " 169.254.0.0/16 , ,junk")
    configure_ssrf_whitelist_from_env()
    assert validate_url_target("http://metadata.test/") == (True, "")
    assert validate_url_target("http://internal.test/")[0] is False  # replaced, not added


def test_contains_internal_url():
    assert contains_internal_url("curl -s http://169.254.169.254/latest/meta-data/")
    assert contains_internal_url("wget 'http://localhost:3000/admin'")
    assert contains_internal_url("curl https://unknown.invalid/")  # unresolvable counts as unsafe, as in nanobot
    assert not contains_internal_url("curl https://public.test/api && echo done")
    assert not contains_internal_url("ls -la /etc")


def test_proxy_mounts(monkeypatch):
    assert httpx_env_proxy_mounts() == {}
    monkeypatch.setenv("HTTPS_PROXY", "proxy.corp:3128")
    monkeypatch.setenv("NO_PROXY", "localhost,10.1.2.3,.internal.corp")
    mounts = httpx_env_proxy_mounts()
    assert isinstance(mounts["https://"], httpx.AsyncHTTPTransport)
    assert mounts["all://localhost"] is None
    assert mounts["all://10.1.2.3"] is None
    assert mounts["all://*.internal.corp"] is None
    monkeypatch.setenv("NO_PROXY", "*")
    assert httpx_env_proxy_mounts() == {}


# --- web_fetch -------------------------------------------------------------


def site(routes):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        status, headers, body = routes[str(request.url)]
        return httpx.Response(status, headers=headers, content=body.encode())

    return httpx.MockTransport(handler), seen


async def test_web_fetch_blocks_private_targets_before_any_request():
    transport, seen = site({})
    out = await WebFetchTool(transport=transport).execute(url="http://metadata.test/latest/meta-data/")
    assert is_tool_error(out)
    assert json.loads(out) == {
        "error": "URL validation failed: Blocked: metadata.test resolves to private/internal address 169.254.169.254",
        "url": "http://metadata.test/latest/meta-data/",
    }
    assert seen == []


async def test_web_fetch_blocks_redirects_to_private_targets():
    transport, seen = site(
        {
            "https://public.test/": (302, {"location": "http://127.0.0.1/admin"}, ""),
            "https://public.test/b": (301, {"location": "http://internal.test/x"}, ""),
        }
    )
    tool = WebFetchTool(transport=transport)
    out = json.loads(await tool.execute(url="https://public.test/"))
    assert out["error"] == "Redirect blocked: Blocked: 127.0.0.1 resolves to private/internal address 127.0.0.1"
    out = json.loads(await tool.execute(url="https://public.test/b"))
    assert out["error"] == "Redirect blocked: Blocked: internal.test resolves to private/internal address 10.0.0.5"
    assert seen == ["https://public.test/", "https://public.test/b"]  # the private targets were never requested


async def test_web_fetch_uses_the_pinned_transport(fake_dns):
    """DNS answers public for the checks, then private for the connection."""
    answers = iter([["93.184.216.34"], ["93.184.216.34"]])
    fake_dns["flip.test"] = lambda: next(answers, ["127.0.0.1"])
    transport, seen = site({"https://flip.test/": (200, {"content-type": "text/plain"}, "secret")})
    out = json.loads(await WebFetchTool(transport=transport).execute(url="https://flip.test/"))
    assert out == {
        "error": "Redirect blocked: Blocked: flip.test resolves to private/internal address 127.0.0.1",
        "url": "https://flip.test/",
    }
    assert seen == []


def test_web_fetch_honours_proxy_variables(monkeypatch):
    from nanoclaw.agent.tools.web import _fetch_client_kwargs

    assert set(_fetch_client_kwargs(30.0, None)) == {"timeout", "transport"}
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp:3128")
    kwargs = _fetch_client_kwargs(30.0, None)
    assert isinstance(kwargs["transport"], PinnedDNSAsyncTransport)
    assert set(kwargs["mounts"]) == {"https://"}


async def test_web_fetch_public_page_still_works():
    transport, seen = site({"https://public.test/": (200, {"content-type": "text/plain"}, "hello")})
    out = json.loads(await WebFetchTool(transport=transport).execute(url="https://public.test/"))
    assert out["status"] == 200 and out["text"].endswith("hello")


async def test_dns_rebinding_between_check_and_request_is_blocked(fake_dns):
    """The name is public when checked, private when the connection is made."""

    class Rebinding(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise AssertionError("must not be reached")

    transport = PinnedDNSAsyncTransport(inner=Rebinding())
    assert validate_url_target("http://flip.test/") == (True, "")
    fake_dns["flip.test"] = ["127.0.0.1"]
    with pytest.raises(UnsafeURLRequestError, match="flip.test resolves to private/internal address 127.0.0.1"):
        await transport.handle_async_request(httpx.Request("GET", "http://flip.test/"))


async def test_pinned_transport_connects_to_the_checked_addresses(fake_dns):
    import socket

    looked_up = []

    class Inner(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            fake_dns["pin.test"] = ["127.0.0.1"]  # DNS changes after the check...
            looked_up.append(socket.getaddrinfo("pin.test", 443)[0][4][0])  # ...the connection doesn't
            return httpx.Response(200)

    fake_dns["pin.test"] = ["93.184.216.40"]
    transport = PinnedDNSAsyncTransport(inner=Inner())
    original = network.socket.getaddrinfo
    assert (await transport.handle_async_request(httpx.Request("GET", "https://pin.test/"))).status_code == 200
    assert looked_up == ["93.184.216.40"]
    assert network.socket.getaddrinfo is original  # restored afterwards


# --- exec: always-on checks ------------------------------------------------


async def test_exec_blocks_internal_urls_even_unrestricted(ws):
    tool = ExecTool(working_dir=ws)
    out = await tool.execute(command="curl -s http://169.254.169.254/latest/meta-data/")
    assert out == "Error: Command blocked by safety guard (internal/private URL detected)"
    assert is_tool_error(out)
    out = await tool.execute(command="echo https://public.test/ok")
    assert "https://public.test/ok" in out


async def test_unrestricted_exec_and_files_can_leave_the_workspace(ws, outside):
    assert "outside" in await ExecTool(working_dir=ws).execute(command=f"cat {outside}")
    assert "outside" in await ExecTool(working_dir=ws).execute(command="cat ../secret.txt")
    assert "outside" in await ReadFileTool(ws).execute(path=str(outside))


# --- restricted file tools -------------------------------------------------


def boundary(path, ws):
    return f"Error: Path {path} is outside allowed directory {ws}{WORKSPACE_BOUNDARY_NOTE}"


async def test_restricted_reads_and_writes_stay_inside(ws, outside):
    read = ReadFileTool(ws, restrict_to_workspace=True)
    assert "inside" in await read.execute(path="notes.txt")
    assert "inside" in await read.execute(path=str(ws / "notes.txt"))
    assert await read.execute(path=str(outside)) == boundary(outside, ws)
    assert await read.execute(path="../secret.txt") == boundary("../secret.txt", ws)
    assert is_tool_error(await read.execute(path=str(outside)))

    write = WriteFileTool(ws, restrict_to_workspace=True)
    assert await write.execute(path=str(outside), content="x") == boundary(outside, ws)
    assert outside.read_text() == "outside\n"
    assert "Successfully wrote" in await write.execute(path="sub/new.txt", content="x")

    edit = EditFileTool(ws, restrict_to_workspace=True)
    assert await edit.execute(path=str(outside), old_text="outside", new_text="x") == boundary(outside, ws)

    listing = ListDirTool(ws, restrict_to_workspace=True)
    assert await listing.execute(path=str(outside.parent)) == boundary(outside.parent, ws)
    assert "notes.txt" in await listing.execute(path=".")


async def test_restricted_symlink_escape_is_blocked(ws, outside):
    (ws / "link.txt").symlink_to(outside)
    (ws / "linkdir").symlink_to(outside.parent)
    read = ReadFileTool(ws, restrict_to_workspace=True)
    assert await read.execute(path="link.txt") == boundary("link.txt", ws)
    assert await read.execute(path="linkdir/secret.txt") == boundary("linkdir/secret.txt", ws)


async def test_restricted_read_exceptions(ws, tmp_path, monkeypatch):
    import shutil

    import nanoclaw.agent.skills as skills

    # A copy, so a failing write check can never touch the package's own skills.
    BUILTIN_SKILLS_DIR = tmp_path / "builtin-skills"
    shutil.copytree(skills.BUILTIN_SKILLS_DIR, BUILTIN_SKILLS_DIR)
    monkeypatch.setattr(skills, "BUILTIN_SKILLS_DIR", BUILTIN_SKILLS_DIR)

    read = ReadFileTool(ws, restrict_to_workspace=True)
    for path in ("skills/weather/SKILL.md", str(BUILTIN_SKILLS_DIR / "weather" / "SKILL.md")):
        out = await read.execute(path=path)  # built-in fallback, and the folder itself
        assert not is_tool_error(out) and "# Weather" in out
    assert "cursor" in await read.execute(path="memory/history.jsonl")

    # Writes get no exceptions.
    write = WriteFileTool(ws, restrict_to_workspace=True)
    target = BUILTIN_SKILLS_DIR / "weather" / "SKILL.md"
    assert await write.execute(path=str(target), content="x") == boundary(target, ws)


# --- restricted exec -------------------------------------------------------


@pytest.mark.parametrize(
    "command, error",
    [
        ("cat ../secret.txt", "path traversal detected"),
        ("cat /etc/hostname", "path outside working dir"),
        ("ls ~-", "path outside working dir"),
        ("ls ~/", "path outside working dir"),
        ("sh -c 'cat /etc/hostname'", "path outside working dir"),
        ("cat file:///etc/hostname", "path outside working dir"),
        ("X=/etc/passwd; echo $X", "path outside working dir"),
    ],
)
async def test_restricted_exec_blocks(ws, command, error):
    out = await ExecTool(working_dir=ws, restrict_to_workspace=True).execute(command=command)
    assert out == f"Error: Command blocked by safety guard ({error}){NOTE_SHELL}"


@pytest.mark.parametrize(
    "command",
    [
        "cat notes.txt",
        "echo hi > /dev/null 2>/dev/stderr",
        "cat {ws}/notes.txt",
        "curl https://public.test/a/b",
        "ls ~+",
    ],
)
async def test_restricted_exec_allows(ws, command):
    out = await ExecTool(working_dir=ws, restrict_to_workspace=True).execute(command=command.format(ws=ws))
    assert "Command blocked" not in out


# nanobot 1d43b46 (tests/tools/test_tool_validation.py:408): a "~user" path
# that cannot be expanded is blocked, not skipped unchecked.
@pytest.mark.skipif(sys.platform == "win32", reason="Unix resolves named homes through passwd")
@pytest.mark.parametrize("command", ["cat {p}", "cat --config={p}", "cat <{p}"])
async def test_restricted_exec_blocks_unknown_user_home(ws, tmp_path, command):
    home_path = f"~nanoclaw-missing-{tmp_path.name}/file.txt"
    with pytest.raises(RuntimeError):
        Path(home_path).expanduser()
    out = await ExecTool(working_dir=ws, restrict_to_workspace=True).execute(command=command.format(p=home_path))
    assert out == f"Error: Command blocked by safety guard (path outside working dir){NOTE_SHELL}"


# nanobot 1d43b46 (tests/tools/test_tool_validation.py:426): "~" with no home
# directory at all is blocked too.
async def test_restricted_exec_blocks_home_path_without_home_directory(ws, monkeypatch):
    for key in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH"):
        monkeypatch.delenv(key, raising=False)
    if sys.platform != "win32":
        import pwd

        def missing_passwd_entry(uid):
            raise KeyError(uid)  # a container UID may have no passwd entry

        monkeypatch.setattr(pwd, "getpwuid", missing_passwd_entry)
    with pytest.raises(RuntimeError):
        Path("~/file.txt").expanduser()
    out = await ExecTool(working_dir=ws, restrict_to_workspace=True).execute(command="cat ~/file.txt")
    assert out == f"Error: Command blocked by safety guard (path outside working dir){NOTE_SHELL}"


# nanobot 1d43b46 (tests/tools/test_tool_validation.py:1191, :1212)
@pytest.mark.parametrize(
    "command", ["cat ~unknown/file.txt", "cat --config=~unknown/file.txt", "cat <~unknown/file.txt", "cat ~/file.txt"]
)
def test_exec_guard_rejects_unresolvable_home_path(ws, monkeypatch, command):
    original = Path.expanduser

    def expanduser(path):
        if str(path).startswith("~"):
            raise RuntimeError("Could not determine home directory.")
        return original(path)

    monkeypatch.setattr(Path, "expanduser", expanduser)
    error = ExecTool(working_dir=ws, restrict_to_workspace=True)._guard_command(command, str(ws))
    assert error == f"Error: Command blocked by safety guard (path outside working dir){NOTE_SHELL}"


def test_exec_guard_allows_resolved_home_inside_workspace(ws, monkeypatch):
    original = Path.expanduser

    def expanduser(path):
        return ws / "notes.txt" if str(path).startswith("~") else original(path)

    monkeypatch.setattr(Path, "expanduser", expanduser)
    assert ExecTool(working_dir=ws, restrict_to_workspace=True)._guard_command("cat ~/notes.txt", str(ws)) is None


async def test_restricted_exec_allows_resolved_home_inside_workspace(ws, monkeypatch):
    monkeypatch.setenv("HOME", str(ws))
    out = await ExecTool(working_dir=ws, restrict_to_workspace=True).execute(command="cat ~/notes.txt")
    assert "inside" in out and "Command blocked" not in out


async def test_restricted_exec_working_dir(ws, tmp_path):
    tool = ExecTool(working_dir=ws, restrict_to_workspace=True)
    assert await tool.execute(command="pwd", working_dir=str(tmp_path)) == (
        "Error: working_dir is outside the configured workspace" + NOTE_SHELL
    )
    (ws / "sub").mkdir()
    assert str(ws / "sub") in await tool.execute(command="pwd", working_dir="sub")
    # From a subfolder, the rest of the workspace is still allowed.
    assert "inside" in await tool.execute(command=f"cat {ws}/notes.txt", working_dir="sub")
    assert "Error" not in await ExecTool(working_dir=ws).execute(command="pwd", working_dir=str(tmp_path))


async def test_deny_list_still_applies_unrestricted(ws):
    assert await ExecTool(working_dir=ws).execute(command="rm -rf build") == (
        "Error: Command blocked by deny pattern filter"
    )


# --- how blocked calls reach the model -------------------------------------


def calls(*items):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(items)],
        finish_reason="tool_calls",
    )


async def test_ssrf_errors_get_the_boundary_note_not_the_retry_hint(ws):
    provider = FakeProvider([calls(("web_fetch", {"url": "http://localhost:8080/"})), "ok"])
    result = await AgentRunner().run(
        AgentRunSpec([{"role": "user", "content": "go"}], build_default_tools(ws), provider)
    )
    tool_message = result.messages[2]["content"]
    assert tool_message.startswith('{"error": "URL validation failed: Blocked: localhost resolves to private/internal')
    assert tool_message.endswith("ask them to whitelist the exact IP/CIDR via NANOCLAW_SSRF_WHITELIST.")
    assert "This is a non-bypassable security boundary." in tool_message
    assert RETRY_HINT not in tool_message


async def test_repeated_workspace_violations_escalate(ws, outside):
    tools = build_default_tools(ws, restrict_to_workspace=True)
    third = calls(("read_file", {"path": str(outside)}), ("exec", {"command": f"cat {outside}"}))
    provider = FakeProvider(
        [calls(("read_file", {"path": str(outside)})), calls(("list_dir", {"path": str(outside)})), third, "ok"]
    )
    result = await AgentRunner().run(AgentRunSpec([{"role": "user", "content": "go"}], tools, provider))
    tool_messages = [m["content"] for m in result.messages if m["role"] == "tool"]
    assert tool_messages[0] == boundary(outside, ws) + RETRY_HINT
    assert tool_messages[1] == boundary(outside, ws) + RETRY_HINT
    assert tool_messages[2].startswith("Error: refusing repeated workspace-bypass attempts.\n")
    assert f"You have tried to access '{str(outside).lower()}' (or an equivalent path) 3 times" in tool_messages[2]
    assert tool_messages[3].startswith("Error: refusing repeated workspace-bypass attempts.\n")  # exec, same target
    assert "4 times" in tool_messages[3]


async def test_escalation_counts_reset_each_run(ws, outside):
    tools = build_default_tools(ws, restrict_to_workspace=True)
    for _ in range(3):
        provider = FakeProvider([calls(("read_file", {"path": str(outside)})), "ok"])
        result = await AgentRunner().run(AgentRunSpec([{"role": "user", "content": "go"}], tools, provider))
        assert result.messages[2]["content"] == boundary(outside, ws) + RETRY_HINT


# --- wiring ----------------------------------------------------------------


def _restricted(registry):
    return [registry.get(n).restrict_to_workspace if n == "exec" else registry.get(n)._allowed_dir is not None
            for n in ("read_file", "write_file", "edit_file", "list_dir", "exec")]


def test_loop_and_subagents_pass_restriction_on(ws):
    loop = AgentLoop(FakeProvider([]), ws, restrict_to_workspace=True)
    assert _restricted(loop.tools) == [True] * 5
    assert _restricted(loop.subagents._build_tools()) == [True] * 5
    loop = AgentLoop(FakeProvider([]), ws)
    assert _restricted(loop.tools) == [False] * 5
    assert _restricted(SubagentManager(ws)._build_tools()) == [False] * 5


def test_cli_restrict_flag_and_env(ws, monkeypatch):
    seen = []
    import nanoclaw.__main__ as cli

    original = cli.AgentLoop

    def spy(*args, **kwargs):
        seen.append(kwargs.get("restrict_to_workspace"))
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "AgentLoop", spy)
    assert main(["ask", "hi", "-w", str(ws)], provider=FakeProvider(["a"])) == 0
    assert main(["ask", "hi", "-w", str(ws), "--restrict"], provider=FakeProvider(["a"])) == 0
    monkeypatch.setenv("NANOCLAW_RESTRICT_TO_WORKSPACE", "yes")
    assert main(["ask", "hi", "-w", str(ws)], provider=FakeProvider(["a"])) == 0
    monkeypatch.setenv("NANOCLAW_RESTRICT_TO_WORKSPACE", "0")
    assert main(["ask", "hi", "-w", str(ws)], provider=FakeProvider(["a"])) == 0
    assert seen == [False, True, True, False]


def test_cli_reads_the_ssrf_whitelist(ws, monkeypatch):
    monkeypatch.setenv("NANOCLAW_SSRF_WHITELIST", "10.0.0.0/8")
    main(["skills", "-w", str(ws)])
    assert validate_url_target("http://internal.test/") == (True, "")


# --- ported from nanobot tests/tools/test_web_fetch_security.py --------------


async def test_concurrent_fetches_each_use_their_own_pinned_addresses(monkeypatch):
    """DNS pinning overrides the process-wide resolver, so two fetches at the
    same time must not see each other's pins (the transport's lock)."""
    import asyncio
    import socket

    public_ips = {"a.example": "93.184.216.34", "b.example": "93.184.216.35"}
    calls = {host: 0 for host in public_ips}
    seen = {}

    def rebinding_resolver(hostname, port, family=0, type_=0, proto=0, flags=0):
        host = str(hostname).rstrip(".").lower()
        calls[host] += 1
        ip = public_ips[host] if calls[host] <= 2 else "169.254.169.254"
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]

    class Resolving(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            await asyncio.sleep(0)
            infos = socket.getaddrinfo(request.url.host, request.url.port or 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
            seen[str(request.url)] = infos[0][4][0]
            return httpx.Response(200, request=request)

    tool = WebFetchTool()

    async def fetch(url):
        async with httpx.AsyncClient(transport=PinnedDNSAsyncTransport(inner=Resolving())) as client:
            return await tool._get_with_redirects(client, url)

    monkeypatch.setattr("nanoclaw.security.network.socket.getaddrinfo", rebinding_resolver)
    results = await asyncio.gather(fetch("https://a.example/"), fetch("https://b.example/"))
    assert all(error is None and response is not None for response, error in results)
    assert seen == {"https://a.example/": "93.184.216.34", "https://b.example/": "93.184.216.35"}
    assert calls == {"a.example": 2, "b.example": 2}


async def test_rebound_target_is_rejected_before_the_connection(monkeypatch):
    import socket

    calls = {"n": 0}

    def rebinding_resolver(hostname, port, family=0, type_=0, proto=0, flags=0):
        calls["n"] += 1
        ip = "93.184.216.34" if calls["n"] <= 2 else "169.254.169.254"
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]

    class Fail(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise AssertionError("rebound target must be rejected before transport")

    monkeypatch.setattr("nanoclaw.security.network.socket.getaddrinfo", rebinding_resolver)
    result = await WebFetchTool(transport=Fail()).execute(url="http://evil.example/page")
    assert is_tool_error(result) and "blocked" in json.loads(result)["error"].lower()
    assert calls["n"] == 3  # first check, redirect-loop check, pinned transport


# nanobot cb03b09 (tests/security/test_security_network.py:246): the pinned
# resolver also answers for a hostname given as bytes, as callers may pass it.
@pytest.mark.parametrize("host", ["example.com", b"example.com", "EXAMPLE.COM.", b"EXAMPLE.COM."])
def test_pin_resolved_url_dns_prevents_second_resolution_rebind(host):
    def rebinding_resolver(hostname, port, family=0, type_=0, proto=0, flags=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", 0))]

    with patch("nanoclaw.security.network.socket.getaddrinfo", side_effect=rebinding_resolver) as resolver:
        with pin_resolved_url_dns("http://example.com/page", ("93.184.216.34",)):
            infos = socket.getaddrinfo(host, 80, socket.AF_UNSPEC, socket.SOCK_STREAM)
        assert socket.getaddrinfo is resolver
        resolver.assert_not_called()
    assert infos[0][4][0] == "93.184.216.34"


# nanobot cb03b09 (tests/security/test_security_network.py:263)
@pytest.mark.parametrize("host", ["other.example", b"other.example"])
def test_pin_resolved_url_dns_preserves_other_host_lookups(host):
    expected = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.35", 443))]
    with patch("nanoclaw.security.network.socket.getaddrinfo", return_value=expected) as resolver:
        with pin_resolved_url_dns("https://example.com/", ("93.184.216.34",)):
            infos = socket.getaddrinfo(
                host, 443, socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, socket.AI_CANONNAME
            )
        assert socket.getaddrinfo is resolver
        resolver.assert_called_once_with(
            host, 443, socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, socket.AI_CANONNAME
        )
    assert infos == expected
