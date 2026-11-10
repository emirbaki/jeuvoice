# Mac spike (Apple Silicon, ~15 minutes)

A teammate with an Apple Silicon Mac runs this. The agent cannot run it.
Send back the printed numbers and `spike.wav`.

## What it does

Renders one fixed paragraph with OmniVoice + the tr2 adapter on `mps`
(fp16, fp32 fallback), audio tokenizer on CPU. Nothing is written
outside its own temp folder except `spike.wav` in that temp folder.

Fixed paragraph:

> İstanbul'da sabah erken saatlerde Joel ve Ellie tünele giriyor. The Last of Us'ta bu sahne herkesin aklında kaldı.

Settings (the default preset): base `k2-fsa/OmniVoice` + adapter tr2
`checkpoint-6000`, reference `ses_001_0007.wav` + `ses_001_0007.txt`,
`guidance_scale = 2.5`, `num_step = 32`, `speed = 1.0`, `seed = 1`,
part seed = `seed + 1`. Sample rate 24000.

## Steps

1. Install uv: `curl -LsSf astral.sh/uv/install.sh | sh`.
2. From the repo root:
   `uv venv engine/.venv --python 3.12`
3. Install the macOS packages (default index for torch):
   `uv pip install torch transformers==5.19.0 peft==0.21.2 accelerate==1.15.0 "omnivoice[lora] @ git+https://github.com/k2-fsa/OmniVoice@08be0b4ccbac3e13e374e86fbfead4b4cac343e2" "trnorm @ git+https://github.com/ozguradmin/ahenk-turkish-tts" soundfile numpy`
4. Copy the adapter folder (`checkpoint-6000` from `tts_lab/outputs/omnivoice_tr2/`)
   and `ses_001_0007.wav` + `ses_001_0007.txt` (`tts_lab/data/ref/sesim/`) onto the Mac.
5. Run:
   `uv run python engine/tools/mac_spike.py --adapter-dir <checkpoint-6000> --ref-wav <ses_001_0007.wav> --ref-text <ses_001_0007.txt>`

## Send back

- Which dtype printed (`fp16` or `fp32`, plus any fallback line).
- The `load_s`, `render_s`, `audio_s`, `rtf` numbers.
- `spike.wav` (from the printed temp path).

## Result

Pending. When the numbers arrive, record them here (device, dtype,
load/render/audio/RTF, macOS version). If the spike fails, the Voice tab
shows "The voice engine does not run on this Mac yet." on macOS.
