"""Adapter fetch tests (fake backend; the failure path needs the network)."""

import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def start_server(token="secret"):
    data_dir = Path(tempfile.mkdtemp(prefix="jeuvoice_fetch_test_"))
    proc = subprocess.Popen(
        [sys.executable, "-m", "jeuvoice_engine", "--port", "0",
         "--token", token, "--data-dir", str(data_dir), "--backend", "fake"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    assert line.startswith("JEUVOICE_ENGINE_READY port="), f"ready line: {line!r}"
    return proc, int(line.split("=", 1)[1])


def post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"X-JeuVoice-Token": "secret", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def test_unknown_repo_gives_a_plain_error():
    # Hugging Face answers 401 for unknown repos too (it hides their existence),
    # so this takes the token-message path. Both messages are plain words.
    proc, port = start_server()
    try:
        body = post(port, "/fetch_adapter", {"repo": "jeuvoice-such-repo-does-not-exist-xyz/abc", "subfolder": None, "token": None})
        assert body.get("ok") is False
        assert body.get("error") == "Add your Hugging Face token in Settings."
        assert "Traceback" not in (body.get("error") or "")
    finally:
        try:
            post(port, "/shutdown", {})
        except Exception:
            pass
        proc.wait(timeout=15)
