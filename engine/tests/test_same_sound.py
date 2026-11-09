"""Same-sound test (GPU): engine render matches the CLI reference run sample for sample.

Needs the downloaded model, the tr2 adapter and the reference clip on this
machine. Run: `engine/.venv/Scripts/python -m pytest engine/tests -q -m gpu --gpu`.
The reference run writes into tts_lab/outputs/jeuvoice_check_* (the only
allowed write there); the test deletes that folder after. Ask the user once
before the first run.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.gpu

REF_SCRIPT = "İstanbul'da sabah erken saatlerde Joel ve Ellie tünele giriyor."
REF_WAV_NAME = "ses_001_0007.wav"
REF_TXT_NAME = "ses_001_0007.txt"
TTS_LAB = Path(r"D:/MLProjelerim/tts_lab")
ADAPTER = TTS_LAB / "outputs" / "omnivoice_tr2" / "checkpoint-6000"


def test_same_sound_as_cli():
    from jeuvoice_engine.omnivoice_backend import OmniVoiceBackend
    from jeuvoice_engine.text import plan_parts

    ref_wav = TTS_LAB / "data" / "ref" / "sesim" / REF_WAV_NAME
    ref_txt = TTS_LAB / "data" / "ref" / "sesim" / REF_TXT_NAME
    for p in (ref_wav, ref_txt, ADAPTER):
        if not p.exists():
            pytest.skip(f"missing reference file: {p}")
    out_name = "jeuvoice_check_same_sound"
    out_dir = TTS_LAB / "outputs" / out_name

    parts = plan_parts(REF_SCRIPT, [], ["The Last of Us"], [])
    assert parts, "one paragraph gives at least one part"
    first = parts[0]

    backend = OmniVoiceBackend()
    backend.load("k2-fsa/OmniVoice", str(ADAPTER), "cuda")
    try:
        prompt = backend.make_prompt(str(ref_wav), ref_txt.read_text(encoding="utf-8").strip())
        wav = backend.render(first, prompt, None, seed=2, guidance=2.5, num_step=32, speed=1.0)
    finally:
        backend.unload()

    script_file = out_dir.parent / f"{out_name}.txt"
    try:
        script_file.write_text(REF_SCRIPT, encoding="utf-8")
        venv_python = TTS_LAB / ".." / ".venv_ml" / "Scripts" / "python.exe"
        # NOTE: omnivoice_src (pinned source checkout, has utils.lora), not
        # omnivoice_env (older, no LoRA support). Hashes match the pinned commit.
        # omnivoice_env stays second on the path for deps (pydub, ...).
        import os

        env = {"PYTHONPATH": os.pathsep.join([
            str(TTS_LAB / "third_party" / "omnivoice_src"),
            str(TTS_LAB / "third_party" / "omnivoice_env"),
        ])}

        cmd = [
            str(venv_python), "scripts/try_omnivoice.py", str(script_file),
            "--by-paragraph", "--pause", "0.1", "--para-pause", "0.3",
            "--guidance-scale", "2.5", "--quote-term", "The Last of Us",
            "--lora-adapter", str(ADAPTER),
            "--ref", str(ref_wav), "--ref-text", str(ref_txt),
            "--out", out_name,
        ]
        subprocess.run(cmd, cwd=TTS_LAB, env={**os.environ, **env}, check=True)
        ref, _ = sf.read(out_dir / "sentence_001.wav", dtype="float32")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        if script_file.exists():
            script_file.unlink()

    n = min(len(wav), len(ref))
    diff = float(np.max(np.abs(wav[:n] - ref[:n])))
    print(f"max abs difference: {diff}")
    assert diff <= 1e-4, f"engine render differs from the CLI: {diff}"
