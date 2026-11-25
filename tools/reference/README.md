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

## Make the Whisper decoder reference files

The decoder test feeds the PyTorch encoder output into the Rust decoder, so it checks the decoder alone. Use a clip shorter than 30 s: the Hugging Face feature extractor normalizes each 30 s window, whisper.cpp the whole file.

1. Write the samples (step 1 above) and the whisper-rs fixture:

   ```powershell
   scripts\cargo.cmd run -p jeuvoice-whisper --release --example dump_whisper_rs -- D:\MLProjelerim\tts_lab\data\ref\my_voice.ogg target\reference\whisper\my_voice.rs.json
   ```

2. Write the decoder reference:

   ```powershell
   engine\.venv\Scripts\python.exe -I tools\reference\whisper_decoder_reference.py target\reference\whisper\my_voice.f32 target\reference\whisper\my_voice.rs.json target\reference\whisper\my_voice.decoder.safetensors
   ```

3. Do the same for `sesim\ses_001_0007.wav`.

| Key | Shape | Content |
|---|---|---|
| `steps`, `dtw_tokens` | `[T]`, `[Td]` int64 | the teacher-forced token lists |
| `enc_out` | `[1500, 1280]` | the fp32 encoder output |
| `logits32`, `logits16` | `[T - 2, 51866]` | the logits after each token from the third on |
| `align32`, `align16` | `[6, Td, 1500]` | the cross-attention softmax of the 6 alignment heads |
