"""Training data preparation jobs: clean clips, read HF datasets, write manifests.

Port of the cleaning rules in `tts_lab/scripts/clean_my_voice.py` (same constants) and
the manifest format of `tts_lab/scripts/prep_omnivoice_data.py`. Runs as engine jobs:
one worker thread, cancellable between clips.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 24_000

# From clean_my_voice.py: burst peaks above LIMIT x the median speech level come down,
# the first ONSET_S seconds use the lower ONSET_LIMIT. MAX_CLIPPED of clipped samples
# throws the clip out. Clean clips are normalized to PEAK.
LIMIT = 2.5
ONSET_LIMIT = 1.6
ONSET_S = 0.6
MAX_CLIPPED = 0.0005
PEAK = 0.72

CLIPPED_MSG = "The clip is clipped."


def load_mono(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
        sr = TARGET_SR
    return y, sr


def is_clipped(y: np.ndarray) -> bool:
    return bool(np.mean(np.abs(y) > 0.99) > MAX_CLIPPED)


def limit_bursts(y: np.ndarray, sr: int) -> np.ndarray:
    """Limit burst peaks like clean_my_voice.limit, then normalize to PEAK."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    from scipy.signal import butter, sosfiltfilt

    y = sosfiltfilt(butter(4, 70, "hp", fs=sr, output="sos"), y)
    rms = np.sqrt(uniform_filter1d(y * y, int(0.05 * sr)) + 1e-12)
    level = float(np.median(rms[rms > 0.02])) if np.any(rms > 0.02) else float(rms.max())
    ceiling = np.full(len(y), LIMIT)
    # Short clips may end before the onset ramp does.
    a = min(int(ONSET_S * sr), len(y))
    b = min(int((ONSET_S + 0.2) * sr), len(y))
    ceiling[:a] = ONSET_LIMIT
    if b > a:
        ceiling[a:b] = np.linspace(ONSET_LIMIT, LIMIT, b - a)
    gain = np.minimum(1.0, ceiling * level / np.maximum(rms, 1e-12))
    gain = uniform_filter1d(1.0 - maximum_filter1d(1.0 - gain, int(0.03 * sr)), int(0.02 * sr))
    out = y * gain
    return (out * PEAK / max(float(np.abs(out).max()), 1e-6)).astype(np.float32)


def band_khz(y: np.ndarray, sr: int) -> float:
    """Median 99th-percentile spectral roll-off across frames, in kHz."""
    n_fft, hop = 2048, 1024
    if len(y) < n_fft:
        return 0.0
    window = np.hanning(n_fft)
    rolls = []
    for start in range(0, len(y) - n_fft + 1, hop):
        mag = np.abs(np.fft.rfft(window * y[start : start + n_fft]))
        total = float(mag.sum())
        if total <= 0.0:
            continue
        idx = int(np.searchsorted(np.cumsum(mag), 0.99 * total))
        rolls.append(idx * sr / n_fft / 1000.0)
    return float(np.median(rolls)) if rolls else 0.0


def clean_clip(clip_id: str, path: str, out_dir: str) -> dict:
    """Clean one clip: write a 24 kHz mono WAV, or record why it is excluded."""
    import os

    try:
        y, _ = load_mono(path)
    except Exception:
        return {"id": clip_id, "audio": "", "duration_s": 0.0, "band_khz": 0.0, "excluded": "The clip could not be read."}
    if is_clipped(y):
        return {"id": clip_id, "audio": "", "duration_s": 0.0, "band_khz": 0.0, "excluded": CLIPPED_MSG}
    os.makedirs(out_dir, exist_ok=True)
    out = limit_bursts(y, TARGET_SR)
    dest = str(Path(out_dir) / f"{clip_id}.wav")
    sf.write(dest, out, TARGET_SR)
    return {
        "id": clip_id,
        "audio": dest,
        "duration_s": round(len(out) / TARGET_SR, 2),
        "band_khz": round(band_khz(out, TARGET_SR), 1),
        "excluded": None,
    }


def manifest_rows(clips: list[dict], sources: list[dict], dev_ratio: float) -> tuple[list[dict], list[dict]]:
    """Expand clips by source weight (`-x2`, `-x3`, …), shuffle with seed 0, split dev."""
    by_source = {s["name"]: int(s.get("weight", 1)) for s in sources}
    rows: list[dict] = []
    for clip in clips:
        if clip.get("excluded"):
            continue
        times = by_source.get(clip.get("source", ""), 1)
        for k in range(max(times, 1)):
            suffix = f"-x{k + 1}" if k else ""
            rows.append({
                "id": f"{clip['id']}{suffix}",
                "audio_path": clip["audio"],
                "text": clip.get("text", ""),
                "language_id": "tr",
            })
    rnd = random.Random(0)
    rnd.shuffle(rows)
    dev_n = int(round(dev_ratio * len(rows)))
    return rows[dev_n:], rows[:dev_n]


def write_manifests(rows: tuple[list[dict], list[dict]], out_dir: str) -> None:
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", rows[0]), ("dev", rows[1])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_clean_run(store, clips: list[dict], out_dir: str):
    def _run(job):
        results = []
        total = len(clips)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for n, clip in enumerate(clips, 1):
            if job.cancelled:
                break
            job.message = f"Clip {n} of {total}"
            results.append(clean_clip(str(clip.get("id", f"clip-{n}")), str(clip.get("audio", "")), str(out)))
            job.progress = n / total if total else 1.0
        job.progress = len(results) / total if total else 1.0
        return {"clips": results}

    return _run


def make_hf_run(store, req: dict):
    def _run(job):
        import os

        repo = str(req.get("repo", ""))
        out = Path(str(req.get("out_dir", "")))
        out.mkdir(parents=True, exist_ok=True)
        limit = req.get("limit")
        limit = int(limit) if limit is not None else None
        rows = stream_hf(repo, req.get("config"), str(req.get("split", "train")), str(req.get("audio_col", "audio")),
                         req.get("text_col"), req.get("token"), limit)
        results = []
        for n, (audio, text) in enumerate(rows, 1):
            if job.cancelled:
                break
            job.message = f"Clip {n}" + (f" of {limit}" if limit else "")
            clip_id = f"{Path(repo.replace('/', '__')).name}-{n:05d}"
            dest = str(out / f"{clip_id}.wav")
            if isinstance(audio, dict) and "array" in audio:
                import librosa

                data = np.asarray(audio["array"], dtype=np.float32)
                if data.ndim > 1:
                    data = data.mean(axis=0)
                sr = int(audio.get("sampling_rate", TARGET_SR) or TARGET_SR)
                if sr != TARGET_SR:
                    data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
            else:
                data, _ = load_mono(str(audio))
            secs = len(data) / TARGET_SR
            sf.write(dest, data, TARGET_SR)
            results.append({"id": clip_id, "audio": dest, "duration_s": round(secs, 2), "band_khz": round(band_khz(data, TARGET_SR), 1), "text": text or "", "excluded": None})
            if limit is not None:
                job.progress = min(n / limit, 1.0)
        job.progress = 1.0
        return {"clips": results}

    return _run


def stream_hf(repo: str, config, split: str, audio_col: str, text_col, token, limit: int | None):
    """Yield (audio, text) pairs: a local folder straight, a Hub repo streamed."""
    import os

    if os.path.isdir(repo):
        from datasets import DatasetDict, load_from_disk

        loaded = load_from_disk(repo)
        if isinstance(loaded, DatasetDict):
            ds = loaded[split if split in loaded else list(loaded.keys())[0]]
        else:
            ds = loaded
    else:
        from datasets import load_dataset

        kwargs: dict = {"split": split, "streaming": True}
        if config:
            kwargs["config_name"] = config
        if token:
            kwargs["token"] = token
        ds = load_dataset(repo, **kwargs)
    out = []
    for row in ds:
        out.append((row.get(audio_col), (row.get(text_col) if text_col else "") or ""))
        if limit is not None and len(out) >= limit:
            break
    return out


def make_manifest_run(store, clips: list[dict], sources: list[dict], dev_ratio: float, out_dir: str):
    def _run(job):
        job.message = "Writing the manifest…"
        train, dev = manifest_rows(clips, sources, float(dev_ratio or 0.0))
        write_manifests((train, dev), out_dir)
        job.progress = 1.0
        return {"train": len(train), "dev": len(dev)}

    return _run
