"""Job API tests (fake backend over real HTTP, no torch)."""

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "Birinci paragraf burada bitiyor.\n\nİkinci paragraf şimdi başlıyor."


def start_server(token="secret"):
    data_dir = Path(tempfile.mkdtemp(prefix="jeuvoice_jobs_test_"))
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
    return proc, int(line.split("=", 1)[1]), data_dir


def call(port, path, token="secret", method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
    req.add_header("X-JeuVoice-Token", token)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def wait_state(port, job_id, states, timeout=30.0):
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        _, last = call(port, f"/jobs/{job_id}")
        if last.get("state") in states:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {states}: {last}")


@pytest.fixture()
def server():
    proc, port, data_dir = start_server()
    yield port, data_dir
    try:
        call(port, "/shutdown", method="POST")
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()


def render_body(data_dir, paragraphs=None, overrides=None):
    return {
        "script": SCRIPT,
        "paragraphs": paragraphs,
        "preset": {"pause": 0.1, "para_pause": "0.3", "guidance": 2.5, "num_step": 32, "speed": 1.0},
        "terms": {"en": [], "quote": [], "respell": []},
        "base_seed": 1,
        "part_overrides": overrides or {},
        "out_dir": str(data_dir / "out"),
    }


def test_render_two_paragraphs(server):
    port, data_dir = server
    _, created = call(port, "/render", method="POST", body=render_body(data_dir))
    done = wait_state(port, created["job"], ("done",))
    assert done["progress"] == 1.0
    parts = done["result"]["parts"]
    assert len(parts) == 2
    # Seeds follow seed_for: no prompt in the fake preset, so base seed... but the
    # fake backend gets no prompt -> has_prompt False -> seed == base_seed.
    assert [p["seed"] for p in parts] == [1, 1]
    assert done["result"]["pause"] == 0.1 and done["result"]["para_pause"] == 0.3
    for p in parts:
        assert Path(p["file"]).exists()
        assert p["audio_s"] > 0


def test_part_overrides_win(server):
    port, data_dir = server
    _, created = call(port, "/render", method="POST", body=render_body(data_dir, overrides={"1": {"seed": 77, "guidance": 3.0}}))
    done = wait_state(port, created["job"], ("done",))
    first = done["result"]["parts"][0]
    assert first["seed"] == 77 and first["guidance"] == 3.0


def test_cancel_between_parts(server):
    port, data_dir = server
    long_script = "\n\n".join(f"Paragraf numarası {i} burada bitiyor." for i in range(200))
    body = render_body(data_dir)
    body["script"] = long_script
    _, created = call(port, "/render", method="POST", body=body)
    time.sleep(0.3)
    code, _ = call(port, f"/jobs/{created['job']}/cancel", method="POST")
    assert code == 200
    final = wait_state(port, created["job"], ("cancelled", "done"))
    assert final["state"] == "cancelled"


def test_unload_refused_while_render_runs(server):
    port, data_dir = server
    long_script = "\n\n".join(f"Paragraf numarası {i} burada bitiyor." for i in range(200))
    body = render_body(data_dir)
    body["script"] = long_script
    _, created = call(port, "/render", method="POST", body=body)
    time.sleep(0.3)
    code, _ = call(port, "/unload", method="POST")
    assert code in (200, 409)
    if code == 409:
        final = wait_state(port, created["job"], ("done", "cancelled", "failed"))
        assert final["state"] == "done"
        code, body = call(port, "/unload", method="POST")
        assert code == 200 and body.get("ok") is True
    else:
        wait_state(port, created["job"], ("done", "cancelled", "failed"))


def test_backend_error_is_plain(server):
    port, data_dir = server
    body = render_body(data_dir)
    # Force a failure the server cannot fix: point out_dir into a file, not a folder.
    blocker = data_dir / "blocker"
    blocker.write_text("x")
    body["out_dir"] = str(blocker / "out")
    _, created = call(port, "/render", method="POST", body=body)
    final = wait_state(port, created["job"], ("failed",))
    assert final["error"] == "The render failed. Try again."
    assert "Traceback" not in (final["error"] or "")
