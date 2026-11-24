"""Write Whisper large-v3-turbo reference tensors for the Rust GPU tests.

Developer tool only. It never ships, and the app never calls it.

    engine\\.venv\\Scripts\\python.exe -I tools\\reference\\whisper_reference.py <in.f32> <out.safetensors>

<in.f32> holds raw little-endian f32 samples at 16 kHz, written by
`cargo run -p jeuvoice-whisper --example dump_samples`. The output keys match the tap names
of the Rust encoder (crates/jeuvoice-whisper/src/cuda/encoder.rs). All tensors are f32.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import WhisperFeatureExtractor, WhisperModel

MODEL_ID = "openai/whisper-large-v3-turbo"
TAP_BLOCKS = (0, 15, 31)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: whisper_reference.py <in.f32> <out.safetensors>")
    samples = np.fromfile(sys.argv[1], dtype="<f4")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fe = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    # The first 30 s window, padded with zeros to 30 s (480,000 samples).
    mel = fe(samples, sampling_rate=16_000, return_tensors="pt").input_features  # [1, 128, 3000]

    model = WhisperModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device).eval()
    enc = model.encoder

    taps: dict[str, torch.Tensor] = {}

    def keep(name, post=None):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            taps[name] = post(out) if post else out
        return hook

    gelu = torch.nn.functional.gelu
    hooks = [
        enc.conv1.register_forward_hook(keep("conv1", gelu)),
        enc.conv2.register_forward_hook(keep("conv2", gelu)),
    ]
    for i in TAP_BLOCKS:
        hooks.append(enc.layers[i].register_forward_hook(keep(f"block_{i:02d}")))

    with torch.no_grad():
        out = enc(mel.to(device), output_hidden_states=True)
    for h in hooks:
        h.remove()

    tensors = {
        "samples": samples.astype(np.float32),
        "mel": mel[0],
        "conv1": taps["conv1"][0],  # [1280, 3000]
        "conv2": taps["conv2"][0],  # [1280, 1500]
        # hidden_states[0] is the conv output after the positional embedding.
        "embed": out.hidden_states[0][0],  # [1500, 1280]
        "ln_post": out.last_hidden_state[0],  # [1500, 1280]
    }
    for i in TAP_BLOCKS:
        tensors[f"block_{i:02d}"] = taps[f"block_{i:02d}"][0]

    as_np = {
        k: (v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else v)
        for k, v in tensors.items()
    }
    as_np = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in as_np.items()}
    Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
    save_file(as_np, sys.argv[2])
    for k, v in as_np.items():
        print(f"{k:<10} {tuple(v.shape)}")


if __name__ == "__main__":
    main()
