"""Write the OmniVoice text vectors: the tokenizer output and the step-0 layout.

Developer tool. It uses the engine's own tokenizer and helpers, so the Rust text path
can be compared with them exactly. It writes `crates/jeuvoice-omnivoice/tests/data/
text_vectors.json`, which is committed (about 60 KB).

    engine\\.venv\\Scripts\\python.exe -I tools\\reference\\gen_text_vectors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "target/reference/omnivoice"
OUT = ROOT / "crates/jeuvoice-omnivoice/tests/data/text_vectors.json"

sys.path.insert(0, str(ROOT / "engine/.venv/Lib/site-packages"))

import math  # noqa: E402

from omnivoice.models.omnivoice import (  # noqa: E402
    _combine_text,
    _get_time_steps,
    _tokenize_with_nonverbal_tags,
)
from transformers import AutoTokenizer  # noqa: E402


def main() -> None:
    specs = json.loads((ROOT / "tools/reference/omnivoice_scripts.json").read_text(encoding="utf-8"))
    model_dir = next(
        (Path.home() / ".cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots").glob("*/")
    )
    tok = AutoTokenizer.from_pretrained(str(model_dir))

    language = specs["language"]
    speed = specs["speed"]
    ref_text = specs["ref_text"]

    out: dict = {"language": language, "speed": speed, "ref_text": ref_text,
                "num_step": 32, "guidance": 2.5, "t_shift": 0.1, "layer_penalty": 5.0,
                "scripts": []}
    for script in specs["scripts"]:
        det = np.load(REF / f"{script['id']}.det.npz")
        prompt = det["prompt_tokens"]  # [8, 186]
        target_len = int(det["target_len"])
        prompt_len = prompt.shape[1]

        style_text = (
            "<|denoise|>"
            f"<|lang_start|>{language}<|lang_end|>"
            "<|instruct_start|>None<|instruct_end|>"
        )
        style_ids = tok(style_text, add_special_tokens=False).input_ids
        full_text = _combine_text(ref_text=ref_text, text=script["text"])
        text_ids = _tokenize_with_nonverbal_tags(f"<|text_start|>{full_text}<|text_end|>", tok)[0].tolist()

        taps = safe_load(REF / "s1.taps.safetensors") if script["id"] == "s1" else {}
        times = _get_time_steps(0.0, 1.0, out["num_step"], out["t_shift"]).tolist()
        total = target_len * 8
        remaining = total
        schedule = []
        for step in range(out["num_step"]):
            count = remaining if step == out["num_step"] - 1 else min(
                math.ceil(total * (times[step + 1] - times[step])), remaining
            )
            schedule.append(int(count))
            remaining -= int(count)
        entry = {
            "id": script["id"],
            "schedule": schedule,
            "tokens": det["tokens"].astype(int).tolist(),
            "tokens32": np.load(REF / "f32tokens" / f"{script['id']}.det.npz")["tokens"].astype(int).tolist(),
            "text": script["text"],
            "target_len": target_len,
            "prompt_len": prompt_len,
            "style_ids": style_ids,
            "text_ids": text_ids,
            "prompt_ids": prompt.astype(int).tolist(),
        }
        if taps:
            entry["ids0"] = taps["ids0"].astype(int).tolist()
            entry["mask0"] = taps["audio_mask0"].astype(int).tolist()
        out["scripts"].append(entry)
        print(f"{script['id']}: prefix {len(style_ids) + len(text_ids)}, prompt {prompt_len}, "
              f"target {target_len}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} B)")


def safe_load(path: Path) -> dict:
    from safetensors.numpy import load_file

    return load_file(str(path))


if __name__ == "__main__":
    main()
