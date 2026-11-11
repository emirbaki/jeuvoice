"""Adapter fetch tests (fake backend; the failure path needs the network)."""

import json
import logging
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


def test_a_missing_subfolder_falls_back_to_the_repo_root(monkeypatch, tmp_path):
    # upload_adapter.py puts the files at the repo root, while a preset may
    # name checkpoint-6000. The fetch must find the root files.
    import huggingface_hub

    from jeuvoice_engine.fetch import fetch_adapter

    calls = []

    def fake_snapshot_download(repo_id, allow_patterns, token, local_dir, local_dir_use_symlinks):
        calls.append(allow_patterns)
        if allow_patterns is None:
            dest = Path(local_dir)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "adapter_config.json").write_text("{}")
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    path = fetch_adapter("u/r", "checkpoint-6000", None, tmp_path, logging.getLogger("test"))
    assert Path(path) == tmp_path / "models" / "u__r"
    assert (Path(path) / "adapter_config.json").is_file()
    assert calls == [["checkpoint-6000/*"], None]
