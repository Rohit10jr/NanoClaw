"""nanobot's real web UI in Chromium against ``nanoclaw gateway`` (task 23.4).

Skipped unless ``NANOCLAW_E2E_WEBUI_DIST`` points at a built nanobot web UI
(the directory holding ``index.html``) and Node can ``require('playwright')``
with a Chromium it can launch. Each run starts ``webui_gateway.py`` (a
scripted provider, a temporary HOME/config/workspace) and drives the
scenarios in ``webui_e2e.js``; then it restarts the gateway with a
``tokenIssueSecret`` for the bootstrap-secret scenario. Screenshots go to
``NANOCLAW_E2E_OUT`` when set, else to the test's temporary directory.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).parent
DIST = os.environ.get("NANOCLAW_E2E_WEBUI_DIST", "")

pytestmark = pytest.mark.skipif(
    not DIST or not (Path(DIST) / "index.html").is_file() or shutil.which("node") is None,
    reason="set NANOCLAW_E2E_WEBUI_DIST to a built nanobot web UI (and install node + playwright)",
)


def _free_port() -> int:
    # The gateway takes PORT for the UI and PORT + 1 for its health server.
    for _ in range(50):
        with socket.socket() as a, socket.socket() as b:
            a.bind(("127.0.0.1", 0))
            port = a.getsockname()[1]
            try:
                b.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port pair")


class _Gateway:
    def __init__(self, root: Path, secret: str = "") -> None:
        self.root = root
        self.port = _free_port()
        home = root / "home"
        home.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "HOME": str(home)}
        env.pop("NANOCLAW_CONFIG", None)
        self.log = (root / "gateway.log").open("w")
        self.proc = subprocess.Popen(
            [sys.executable, str(HERE / "webui_gateway.py"), str(root), DIST, str(self.port), secret],
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        self._wait_ready()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"gateway exited: {(self.root / 'gateway.log').read_text()[-2000:]}")
            try:
                with urllib.request.urlopen(self.base + "/", timeout=2) as resp:
                    if resp.status == 200:
                        return
            except OSError:
                time.sleep(0.3)
        raise RuntimeError("gateway did not start")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()


def _run_scenarios(base: str, out: Path, state: Path, secret: str = "") -> list[dict]:
    args = ["node", str(HERE / "webui_e2e.js"), base, str(out), str(state)]
    if secret:
        args.append(secret)
    proc = subprocess.run(args, capture_output=True, text=True, timeout=900)
    results = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    if not results:
        pytest.fail(f"no scenario results (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return results


def _check_node_playwright() -> None:
    probe = subprocess.run(
        ["node", "-e", "require('playwright')"], capture_output=True, text=True, cwd=HERE,
    )
    if probe.returncode != 0:
        pytest.skip("node cannot require('playwright')")


def test_webui_scenarios_in_browser(tmp_path: Path) -> None:
    _check_node_playwright()
    out = Path(os.environ.get("NANOCLAW_E2E_OUT") or tmp_path / "screenshots")
    results: list[dict] = []
    gateway = _Gateway(tmp_path / "plain")
    try:
        results += _run_scenarios(gateway.base, out, gateway.root)
    finally:
        gateway.stop()
    secret = "e2e-bootstrap-secret"
    gateway = _Gateway(tmp_path / "secret", secret=secret)
    try:
        results += _run_scenarios(gateway.base, out, gateway.root, secret)
    finally:
        gateway.stop()
    failed = [r for r in results if not r["ok"]]
    report = "\n".join(f"{'ok  ' if r['ok'] else 'FAIL'} {r['scenario']}: {r['detail']}" for r in results)
    print(report)
    assert not failed, report
    assert {r["scenario"] for r in results} >= {
        "s1_markdown", "s2_reasoning_tools", "s3_two_chats", "s4_delete", "s5_stop_and_slash",
        "s6_failed_turn", "s7_commands", "s8_reload_midstream", "s9_secret",
    }
