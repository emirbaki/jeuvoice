"""Write Whisper large-v3-turbo decoder reference tensors for the Rust GPU tests.

Developer tool only. It never ships, and the app never calls it.

    engine\\.venv\\Scripts\\python.exe -I tools\\reference\\whisper_decoder_reference.py <in.f32> <in.rs.json> <out.safetensors>

<in.f32> holds the 16 kHz samples (example `dump_samples`). <in.rs.json> is the whisper-rs
fixture of the same clip (example `dump_whisper_rs`). The clip must be shorter than 30 s: the
Hugging Face feature extractor normalizes each 30 s window, whisper.cpp the whole file, and
the two agree only for one window.

The decoder runs teacher-forced on two token lists:

- `steps`: `[SOT, tr, transcribe]` and the window tokens of whisper-rs. whisper-rs leaves out
  the second timestamp of a pair, so the tool puts the timestamp in twice. Any token list is
  good for a numeric test.
- `dtw_tokens`: `[SOT, tr, NOT]`, the text tokens, EOT (the DTW pass of whisper.cpp).

Keys (f32 unless noted):

| Key | Shape | Content |
|---|---|---|
| `steps`, `dtw_tokens` | `[T]`, `[Td]` int64 | the token lists |
| `enc_out` | `[1500, 1280]` | the fp32 encoder output, the decoder input |
| `logits32`, `logits16` | `[T - 2, 51866]` | the logits after token 2, 3, ... (fp32 and fp16 model) |
| `align32`, `align16` | `[6, Td, 1500]` | the cross-attention softmax of the 6 alignment heads |
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import WhisperFeatureExtractor, WhisperModel

MODEL_ID = "openai/whisper-large-v3-turbo"
EOT, SOT, TR, TRANSCRIBE, NOT, BEG = 50257, 50258, 50259 + 9, 50360, 50364, 50365
# (decoder layer, head), as g_aheads_large_v3_turbo in whisper.cpp.
ALIGNMENT_HEADS = [(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)]


def token_lists(fixture: dict) -> tuple[list[int], list[int]]:
    segs = fixture["segments"]
    if any(s["t1"] > 3000 for s in segs):
        sys.exit("the clip is longer than one 30 s window: use a clip shorter than 30 s")
    seq: list[int] = []
    for i, s in enumerate(segs):
        ids = [t["id"] for t in s["tokens"]]
        seq += ids
        if i + 1 < len(segs) and ids[-1] >= BEG:
            seq.append(ids[-1])
    steps = [SOT, TR, TRANSCRIBE] + seq
    text = [t for t in seq if t < EOT]
    return steps, [SOT, TR, NOT] + text + [EOT]


def run(model: WhisperModel, enc: torch.Tensor, steps: list[int], dtw: list[int], device: str):
    dec = model.decoder
    w = dec.embed_tokens.weight
    with torch.no_grad():
        out = dec(input_ids=torch.tensor([steps], device=device), encoder_hidden_states=enc)
        logits = (out.last_hidden_state[0, 2:].float() @ w.float().T).cpu().numpy()
        out = dec(input_ids=torch.tensor([dtw], device=device), encoder_hidden_states=enc, output_attentions=True)
        ca = out.cross_attentions
        align = torch.stack([ca[layer][0, head] for layer, head in ALIGNMENT_HEADS]).float().cpu().numpy()
    return logits, align


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit("usage: whisper_decoder_reference.py <in.f32> <in.rs.json> <out.safetensors>")
    samples = np.fromfile(sys.argv[1], dtype="<f4")
    fixture = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    steps, dtw = token_lists(fixture)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fe = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    mel = fe(samples, sampling_rate=16_000, return_tensors="pt").input_features.to(device)
    # Eager attention: the other kernels do not return the attention weights.
    model = WhisperModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager").to(device).eval()
    with torch.no_grad():
        enc = model.encoder(mel).last_hidden_state  # [1, 1500, 1280]
    logits32, align32 = run(model, enc, steps, dtw, device)
    # The fp16 model gets the same encoder output, rounded to f16, as the Rust decoder does.
    model = model.half()
    logits16, align16 = run(model, enc.half(), steps, dtw, device)

    tensors = {
        "steps": np.array(steps, dtype=np.int64),
        "dtw_tokens": np.array(dtw, dtype=np.int64),
        "enc_out": np.ascontiguousarray(enc[0].float().cpu().numpy(), dtype=np.float32),
        "logits32": np.ascontiguousarray(logits32, dtype=np.float32),
        "logits16": np.ascontiguousarray(logits16, dtype=np.float32),
        "align32": np.ascontiguousarray(align32, dtype=np.float32),
        "align16": np.ascontiguousarray(align16, dtype=np.float32),
    }
    Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, sys.argv[3])
    for k, v in tensors.items():
        print(f"{k:<11} {tuple(v.shape)} {v.dtype}")
    agree = (logits32.argmax(1) == logits16.argmax(1)).mean()
    print(f"argmax agreement fp16 vs fp32: {agree:.4f}")
    print(f"max |logits16 - logits32|: {np.abs(logits16 - logits32).max():.4f}")


if __name__ == "__main__":
    main()
