# Reference tensors

This tool is for developers only. It never ships, and the app never calls it.

The Rust GPU tests compare each kernel and each encoder layer with values from PyTorch. This folder writes those values.

## Make the Whisper reference files

Run from the repo root. The first run downloads `openai/whisper-large-v3-turbo` from Hugging Face (about 1.6 GB).

1. Write the 16 kHz samples, as the app gives them to Whisper:

   ```powershell
   cargo run -p jeuvoice-whisper --example dump_samples -- D:\MLProjelerim\tts_lab\data\ref\my_voice.ogg target\reference\whisper\my_voice.f32
   ```

2. Write the reference tensors:

   ```powershell
   engine\.venv\Scripts\python.exe -I tools\reference\whisper_reference.py target\reference\whisper\my_voice.f32 target\reference\whisper\my_voice.safetensors
   ```

3. Do the same for `my_voice2.mp4` and `sesim\ses_001_0007.wav`.

The files go to `target\reference\whisper\` and are not committed.

## Keys

| Key | Shape | Content |
|---|---|---|
| `samples` | `[n]` | the input samples |
| `mel` | `[128, 3000]` | `WhisperFeatureExtractor` output of the first 30 s window |
| `conv1`, `conv2` | `[1280, 3000]`, `[1280, 1500]` | after GELU |
| `embed` | `[1500, 1280]` | after the positional embedding |
| `block_00`, `block_15`, `block_31` | `[1500, 1280]` | hidden states after those blocks |
| `ln_post` | `[1500, 1280]` | the encoder output |
