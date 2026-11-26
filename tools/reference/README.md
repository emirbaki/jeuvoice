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

## Make the OmniVoice reference files (Phase 2)

The model files are already on this machine: the `k2-fsa/OmniVoice` snapshot in the Hugging Face cache and the LoRA adapter in the app data folder. Nothing is downloaded.

```powershell
engine\.venv\Scripts\python.exe -I tools\reference\omnivoice_reference.py --adapter "$env:APPDATA\Jeu\JeuVoice\data\models\emirbakid__jeuvoice-omnivoice-tr2"
```

It reads `tools\reference\omnivoice_scripts.json` (3 Turkish scripts, the 8 s voice prompt `target\reference\omnivoice\prompt8.wav` and its transcript) and writes into `target\reference\omnivoice\`, which is not committed:

| File | Content |
|---|---|
| `<id>.det.npz` | the deterministic tokens `[8, T]`, the prompt frames `[8, T_ref]`, the text, the language, the target length |
| `<id>.product.npz` | the same from the product mode (`position_temperature` 5.0) |
| `<id>.taps.safetensors` | `embed` and `hidden_00` … `hidden_27` `[T, 1024]` f32 of the first pass, `cond_logits_00` / `uncond_logits_00` `[8, T_target, 1025]`, `conf00` / `pick00` `[8, T_target]`, `order00` with the first step's top-k target positions and `topk00` `[8, T_target, 103]` with the first step's top token ids |

The tool captures taps for the first script unless `--taps-script` selects others. The quality gate needs s1, s2 and s3. Generate the missing ones in fp16 and f32:

```powershell
engine\.venv\Scripts\python.exe -I tools\reference\omnivoice_reference.py --adapter "$env:APPDATA\Jeu\JeuVoice\data\models\emirbakid__jeuvoice-omnivoice-tr2" --skip-product --taps-script s2,s3
engine\.venv\Scripts\python.exe -I tools\reference\omnivoice_reference.py --adapter "$env:APPDATA\Jeu\JeuVoice\data\models\emirbakid__jeuvoice-omnivoice-tr2" --out target\reference\omnivoice\f32 --dtype fp32 --skip-product --taps-script s2,s3
```

The deterministic mode (`position_temperature` 0, `class_temperature` 0) repeats exactly. The product mode does not: the Gumbel noise on the position score changes which positions are unmasked first, and the tokens differ on almost every position. Compare the product mode by frame count only.

The engine baseline, stage by stage, with the peak VRAM of the card:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\tts-baseline.ps1
```

The audio tokenizer runs on the CPU, as in the engine. With it on the GPU beside the model the 4 GB card thrashes and one render takes 202 s instead of 6 s. A voice prompt longer than about 10 s has the same effect (the model recommends 3 to 10 s).
