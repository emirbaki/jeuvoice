"""Engine server tests (fake backend, no torch). Start the real server on port 0."""

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def start_server(token="secret", backend="fake"):
    data_dir = Path(tempfile.mkdtemp(prefix="jeuvoice_engine_test_"))
    proc = subprocess.Popen(
        [sys.executable, "-m", "jeuvoice_engine", "--port", "0",
         "--token", token, "--data-dir", str(data_dir), "--backend", backend],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    assert line.startswith("JEUVOICE_ENGINE_READY port="), f"ready line: {line!r}"
    port = int(line.split("=", 1)[1])
    return proc, port, data_dir


def call(port, path, token=None, method="GET"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    if token is not None:
        req.add_header("X-JeuVoice-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def wait_healthy(port, token, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            code, body = call(port, "/health", token)
            if code == 200 and body.get("ok"):
                return body
        except Exception:
            pass
        time.sleep(0.1)
    raise AssertionError("server did not become healthy")


def test_ready_line_and_health():
    proc, port, data_dir = start_server()
    try:
        body = wait_healthy(port, "secret")
        assert body["backend"] == "fake"
        assert body["version"]
        assert (data_dir / "logs" / "engine.log").exists()
    finally:
        call(port, "/shutdown", "secret", method="POST")
        proc.wait(timeout=15)


def test_wrong_token_is_401():
    proc, port, _ = start_server(token="right")
    try:
        wait_healthy(port, "right")
        code, _ = call(port, "/health", "wrong")
        assert code == 401
        code, _ = call(port, "/health")
        assert code == 401
    finally:
        call(port, "/shutdown", "right", method="POST")
        proc.wait(timeout=15)


def test_shutdown_stops_server():
    proc, port, _ = start_server()
    try:
        wait_healthy(port, "secret")
        code, body = call(port, "/shutdown", "secret", method="POST")
        assert code == 200 and body.get("ok") is True
        proc.wait(timeout=15)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()
