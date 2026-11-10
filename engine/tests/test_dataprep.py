"""Data-prep tests: cleaning rules, manifests, HF reads and the /prep/* endpoints."""

import json
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def try_import(name):
    try:
        return __import__(name)
    except ImportError as e:
        pytest.skip(f"missing test dependency: {e.name or name}")


def sine_wav(path: Path, secs: float = 1.0, freq: float = 440.0, amp: float = 0.5, sr: int = 24000, clip: bool = False):
    n = int(secs * sr)
    t = np.arange(n, dtype=np.float64) / sr
    y = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if clip:
        y = np.clip(y * 3.0, -1.0, 1.0).astype(np.float32)
    sf.write(path, y, sr)


def test_clipped_clip_is_excluded_with_a_plain_reason():
    pytest.importorskip("scipy")
    from jeuvoice_engine.dataprep import clean_clip

    tmp = Path(tempfile.mkdtemp(prefix="dataprep_test_"))
    sine_wav(tmp / "loud.wav", amp=0.9, clip=True)
    out = clean_clip("loud", str(tmp / "loud.wav"), str(tmp / "out"))
    assert out["excluded"] == "The clip is clipped."
    assert out["audio"] == ""
    assert "Traceback" not in (out["excluded"] or "")


def test_clean_sine_passes_with_24k_mono_output():
    pytest.importorskip("scipy")
    from jeuvoice_engine.dataprep import clean_clip

    tmp = Path(tempfile.mkdtemp(prefix="dataprep_test_"))
    sine_wav(tmp / "sine.wav", secs=1.0)
    out = clean_clip("sine", str(tmp / "sine.wav"), str(tmp / "out"))
    assert out["excluded"] is None
    assert Path(out["audio"]).exists()
    with wave.open(out["audio"], "rb") as w:
        assert (w.getnchannels(), w.getframerate()) == (1, 24000)
    assert abs(out["duration_s"] - 1.0) < 0.05
    assert out["band_khz"] > 0.0


def test_short_clip_still_cleans():
    pytest.importorskip("scipy")
    from jeuvoice_engine.dataprep import clean_clip

    tmp = Path(tempfile.mkdtemp(prefix="dataprep_test_"))
    sine_wav(tmp / "short.wav", secs=0.4)
    out = clean_clip("short", str(tmp / "short.wav"), str(tmp / "out"))
    assert out["excluded"] is None
    assert Path(out["audio"]).exists()


def _manifest_clips(n_heavy=2, n_light=1):
    clips, sources = [], [{"name": "H", "weight": 2}, {"name": "L", "weight": 1}]
    for i in range(n_heavy):
        clips.append({"id": f"h{i}", "audio": f"h{i}.wav", "text": "merhaba", "duration_s": 1.0, "source": "H", "excluded": None})
    for i in range(n_light):
        clips.append({"id": f"l{i}", "audio": f"l{i}.wav", "text": "selam", "duration_s": 2.0, "source": "L", "excluded": None})
    return clips, sources


def test_manifest_repeats_weight_two_with_x2_ids():
    from jeuvoice_engine.dataprep import manifest_rows

    clips, sources = _manifest_clips()
    train, dev = manifest_rows(clips, sources, 0.0)
    assert dev == []
    assert len(train) == 2 * 2 + 1, [r["id"] for r in train]
    ids = sorted(r["id"] for r in train)
    assert ids == ["h0", "h0-x2", "h1", "h1-x2", "l0"], ids
    by_id = {r["id"]: r for r in train}
    assert by_id["h0-x2"]["text"] == "merhaba"
    assert all(r["language_id"] == "tr" for r in train)
    # Deterministic: the same input always gives the same order.
    again, _ = manifest_rows(clips, sources, 0.0)
    assert [r["id"] for r in again] == [r["id"] for r in train]


def test_manifest_dev_split_keeps_every_row():
    from jeuvoice_engine.dataprep import manifest_rows

    clips = [dict(id=f"c{i}", audio=f"c{i}.wav", text="x", duration_s=1.0, source="S", excluded=None) for i in range(10)]
    sources = [{"name": "S", "weight": 1}]
    train, dev = manifest_rows(clips, sources, 0.2)
    assert len(train) == 8 and len(dev) == 2
    assert len({r["id"] for r in train + dev}) == 10


def test_prep_hf_reads_a_tiny_local_dataset():
    try_import("datasets")
    from datasets import Dataset

    from jeuvoice_engine.dataprep import stream_hf

    # Plain string paths (no Audio cast): the pipeline reads them with soundfile.
    # (datasets' Audio decoding needs torchcodec, whose Windows wheels don't load.)
    tmp = Path(tempfile.mkdtemp(prefix="dataprep_hf_"))
    sine_wav(tmp / "a.wav", secs=0.5)
    sine_wav(tmp / "b.wav", secs=0.5)
    ds = Dataset.from_dict({"audio": [str(tmp / "a.wav"), str(tmp / "b.wav")], "text": ["bir", "iki"]})
    ds.save_to_disk(str(tmp / "ds"))
    rows = stream_hf(str(tmp / "ds"), None, "train", "audio", "text", None, None)
    assert len(rows) == 2
    assert [t for _, t in rows] == ["bir", "iki"]
    assert all(isinstance(a, str) and Path(a).exists() for a, _ in rows)


# --- HTTP-level tests against the real server (fake backend) ---


def start_server(token="secret"):
    data_dir = Path(tempfile.mkdtemp(prefix="dataprep_jobs_test_"))
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


def call(port, path, body=None, method="GET"):
    import urllib.error
    import urllib.request

    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
    req.add_header("X-JeuVoice-Token", "secret")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def wait_state(port, job_id, states, timeout=120.0):
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        _, last = call(port, f"/jobs/{job_id}")
        if last.get("state") in states:
            return last
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} never reached {states}: {last}")


def test_prep_clean_job_writes_wavs():
    proc, port = start_server()
    try:
        tmp = Path(tempfile.mkdtemp(prefix="dataprep_clean_"))
        sine_wav(tmp / "a.wav", secs=0.5)
        sine_wav(tmp / "b.wav", secs=0.5)
        _, created = call(port, "/prep/clean", {
            "clips": [{"id": "a", "audio": str(tmp / "a.wav")}, {"id": "b", "audio": str(tmp / "b.wav")}],
            "out_dir": str(tmp / "out"),
        }, method="POST")
        done = wait_state(port, created["job"], ("done",))
        assert done["progress"] == 1.0
        clips = {c["id"]: c for c in done["result"]["clips"]}
        assert set(clips) == {"a", "b"}
        assert all(c["excluded"] is None for c in clips.values())
        assert all(Path(c["audio"]).exists() for c in clips.values())
    finally:
        call(port, "/shutdown", {}, method="POST")
        proc.wait(timeout=30)


def test_prep_manifest_job_writes_train_and_dev():
    proc, port = start_server()
    try:
        tmp = Path(tempfile.mkdtemp(prefix="dataprep_manifest_"))
        clips, sources = _manifest_clips(n_heavy=4, n_light=2)
        _, created = call(port, "/prep/manifest", {
            "clips": clips, "sources": sources, "dev_ratio": 0.2, "out_dir": str(tmp / "m"),
        }, method="POST")
        done = wait_state(port, created["job"], ("done",))
        assert done["result"] == {"train": 8, "dev": 2}
        train = (tmp / "m" / "train.jsonl").read_text(encoding="utf-8").strip().split("\n")
        dev = (tmp / "m" / "dev.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(train) == 8 and len(dev) == 2
        row = json.loads(train[0])
        assert set(row) == {"id", "audio_path", "text", "language_id"}
    finally:
        call(port, "/shutdown", {}, method="POST")
        proc.wait(timeout=30)


def test_prep_cancel_between_clips():
    proc, port = start_server()
    try:
        tmp = Path(tempfile.mkdtemp(prefix="dataprep_cancel_"))
        clips = []
        for i in range(12):
            p = tmp / f"c{i}.wav"
            sine_wav(p, secs=20.0)
            clips.append({"id": f"c{i}", "audio": str(p)})
        _, created = call(port, "/prep/clean", {"clips": clips, "out_dir": str(tmp / "out")}, method="POST")
        time.sleep(0.5)
        code, _ = call(port, f"/jobs/{created['job']}/cancel", method="POST")
        assert code == 200
        final = wait_state(port, created["job"], ("cancelled", "done"))
        assert final["state"] == "cancelled"
    finally:
        call(port, "/shutdown", {}, method="POST")
        proc.wait(timeout=30)


def test_prep_backend_error_is_plain():
    proc, port = start_server()
    try:
        tmp = Path(tempfile.mkdtemp(prefix="dataprep_fail_"))
        blocker = tmp / "blocker"
        blocker.write_text("x")
        _, created = call(port, "/prep/manifest", {
            "clips": [], "sources": [], "dev_ratio": 0.0, "out_dir": str(blocker / "nope"),
        }, method="POST")
        failed = wait_state(port, created["job"], ("failed",))
        assert failed["error"] == "Preparing the data failed. Try again."
        assert "Traceback" not in (failed["error"] or "")
    finally:
        call(port, "/shutdown", {}, method="POST")
        proc.wait(timeout=30)
