"""Mac spike: render one fixed paragraph with OmniVoice on Apple Silicon (MPS).

The user runs this on a Mac. It makes nothing outside its own temp folder:
it renders into a temp dir, writes spike.wav there, and prints the numbers
to send back.

Run (macOS, 15 minutes):
    uv venv engine/.venv --python 3.12
    uv pip install torch transformers==5.19.0 peft==0.21.2 accelerate==1.15.0 \
        "omnivoice[lora] @ git+https://github.com/k2-fsa/OmniVoice@08be0b4ccbac3e13e374e86fbfead4b4cac343e2" \
        "trnorm @ git+https://github.com/ozguradmin/ahenk-turkish-tts" soundfile numpy
    uv run python engine/tools/mac_spike.py --adapter-dir <tr2-checkpoint-6000> \
        --ref-wav <ses_001_0007.wav> --ref-text <ses_001_0007.txt>

The fixed paragraph is in docs/mac-spike.md. Default preset settings:
paragraph mode, pause 0.1, para_pause 0.3, guidance_scale 2.5, num_step 32,
speed 1.0, seed 1, part seed = seed + i (1-based) with a reference clip.
Sample rate 24000. Quote term "The Last of Us".
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

FIXED_PARAGRAPH = (
    "İstanbul'da sabah erken saatlerde Joel ve Ellie tünele giriyor. "
    'The Last of Us\'ta bu sahne herkesin aklında kaldı.'
)

DEFAULTS = {
    "model": "k2-fsa/OmniVoice",
    "guidance_scale": 2.5,
    "num_step": 32,
    "speed": 1.0,
    "seed": 1,
}


def clean(text: str) -> str:
    """Same cleaning as the CLI: trnorm normalize, apostrophe rule, I/İ, lowercase."""
    from trnorm import normalize

    import re

    text = normalize(text)
    text = re.sub(r"(?<=\w)['’](?=\w)", "", text)
    text = text.replace("I", "ı").replace("İ", "i").lower()
    text = text.replace("the last of us", '"The Last of Us"')
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description="OmniVoice Mac spike (MPS, fp16 with fp32 fallback).")
    ap.add_argument("--adapter-dir", required=True, help="tr2 checkpoint folder (checkpoint-6000)")
    ap.add_argument("--ref-wav", required=True, help="reference WAV, 3-10 s (ses_001_0007.wav)")
    ap.add_argument("--ref-text", required=True, help="exact transcript of --ref-wav (.txt path or text)")
    ap.add_argument("--model", default=DEFAULTS["model"])
    args = ap.parse_args()

    ref_text = args.ref_text
    if ref_text.endswith(".txt"):
        ref_text = Path(ref_text).read_text(encoding="utf-8").strip()

    tmp = Path(tempfile.mkdtemp(prefix="jeuvoice_mac_spike_"))
    out_wav = tmp / "spike.wav"

    import numpy as np
    import soundfile as sf
    import torch

    if not torch.backends.mps.is_available():
        print("MPS is not available on this Mac. The voice engine does not run on this Mac yet.")
        return 2

    from omnivoice import OmniVoice

    dtype = torch.float16
    t0 = time.perf_counter()
    try:
        model = OmniVoice.from_pretrained(args.model, device_map="mps", dtype=dtype)
        dtype_name = "fp16"
    except Exception as e:
        print(f"fp16 load failed ({e}), falling back to fp32.")
        dtype = torch.float32
        model = OmniVoice.from_pretrained(args.model, device_map="mps", dtype=dtype)
        dtype_name = "fp32"
    print(f"device=mps dtype={dtype_name}")

    from omnivoice.utils.lora import load_lora_adapter

    model = load_lora_adapter(model, args.adapter_dir)
    # The 800 MB audio tokenizer runs out of memory beside the main model, so it runs on the CPU.
    model.audio_tokenizer = model.audio_tokenizer.to("cpu")
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    load_s = time.perf_counter() - t0

    prompt = model.create_voice_clone_prompt(ref_audio=args.ref_wav, ref_text=ref_text)
    text = clean(FIXED_PARAGRAPH)
    print(f"text: {text}")

    torch.manual_seed(DEFAULTS["seed"] + 1)  # part seed = seed + i, i = 1
    t0 = time.perf_counter()
    wav = model.generate(
        text=text,
        voice_clone_prompt=prompt,
        language="tr",
        num_step=DEFAULTS["num_step"],
        speed=DEFAULTS["speed"],
        guidance_scale=DEFAULTS["guidance_scale"],
    )[0]
    render_s = time.perf_counter() - t0
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    sf.write(out_wav, wav, 24000)

    audio_s = len(wav) / 24000
    rtf = render_s / audio_s if audio_s > 0 else 0.0
    print(f"wrote {out_wav}")
    print(f"load_s={load_s:.1f} render_s={render_s:.1f} audio_s={audio_s:.2f} rtf={rtf:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
