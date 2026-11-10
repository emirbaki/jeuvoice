# JeuVoice

## Build on Windows

The `whisper-rs` crate builds whisper.cpp with CMake and CUDA. It needs Visual Studio with the C++ tools, the CUDA toolkit, and some environment settings. The script `scripts\cargo.cmd` sets them. Use it in place of `cargo`:

```
scripts\cargo.cmd build
scripts\cargo.cmd test
```

The script sets the CMake generator to `Visual Studio 17 2022`. It sets `CMAKE_CUDA_ARCHITECTURES` to 86 (RTX 30 series) unless you set it first.

## Run

Build the release app, then start it:

```
scripts\cargo.cmd build --release -p jeuvoice-app
target\release\jeuvoice-app.exe
```

The first release build takes about 11 minutes. The first time you transcribe, JeuVoice downloads the speech model (about 1.6 GB).

## Status

Plan 1 (Transcribe) is done.

### What works

- Drop audio or video files (`.mp4`, `.ogg`, `.wav`, `.mp3` and more). JeuVoice reads the sound itself. It needs no ffmpeg.
- A batch runs one file at a time. If one file is bad, the others still finish. The bad file shows a plain error.
- Paths with Turkish letters and spaces work.
- Words come with start and end times. You can play the take, click a word to jump to it, and fix a word.
- You can paste or open a script. JeuVoice marks the words that differ from what it heard.
- You can export SRT, VTT or plain text.

### Measured speed

Checked on 2026-11-09 with the release build. Model: `large-v3-turbo`. Language: Turkish.

| File | Sound | Time | Speed |
|---|---|---|---|
| my_voice2.mp4 | 33.6 s | 2.3 s | 14.9 min per min |
| my_voice.ogg | 27.0 s | 1.2 s | 22.1 min per min |
| çekim 1.wav | 8.4 s | 0.8 s | 10.7 min per min |
| All three | 69.0 s | 4.3 s | 16.2 min per min |

"min per min" means minutes of sound for each minute of work. The model ran on the GPU (NVIDIA, CUDA). Short files look slower, because each file has a fixed start cost. The model load time is not in these numbers.

The script check of one take against its own text gave 20 words the same and 0 different.

To repeat the check:

```
scripts\cargo.cmd test -p jeuvoice-app --release --test real_check -- --ignored --nocapture
```

### Known limits

- Word start times can be 0.1 to 0.3 seconds late.
- The macOS build is not tested.
- The RunPod, Voice and Train tabs are not in this plan.
- A file that is not audio gives a message with a technical detail at the end.
- An automatic check cannot see the screen or hear sound. The window opens and stays open for 10 seconds without a crash. The list below needs your eyes and ears.

### Try it

1. Start the app. Download the model when it asks.
2. Drop an `.mp4`, an `.ogg`, and a file from a folder with Turkish letters.
3. Wait until all three are done.
4. Press play. Check that the sound and the cursor move together.
5. Click a word. Check that playback jumps to that word.
6. Fix one wrong word. Check that the new text stays.
7. Paste a script. Check that the marked words are the real differences.
8. Export SRT. Open it next to the video in a player. Check the timing.

## Status (Plan 2, Voice)

The Voice tab generates Turkish voice-overs with the OmniVoice model and the tr2 adapter, keeps projects in `Documents/JeuVoice`, and exports WAV + SRT.

### First start: model files

- The Hugging Face token lives in the OS keychain. Open Settings and paste a token with read permission, then Save. The token never lands in `settings.json`.
- Without a token, point "Use a local adapter folder" at a tr2 checkpoint folder on disk (for example `tts_lab/outputs/omnivoice_tr2/checkpoint-6000`). The local folder wins over every Hugging Face adapter.
- The default preset `ses_0007` installs on first start. It clones `ses_001_0007.wav` with the tr2 `checkpoint-6000` adapter.

### How to use it

1. Open the Voice tab. Pick a preset, paste a two-paragraph script, press **Generate**. The engine loads once ("Loading the voice model…"), then renders paragraph by paragraph.
2. Press play. Click a word to play from it.
3. Drag a region on the wave and press **Retake region**: three new takes appear. Press a take to hear it, then **Use this take** to splice it in. **+ Try 3 more** renders three more.
4. Double-click a word to fix how it sounds ("Say it as"): the fix is stored as a respell and the paragraph renders again.
5. A name in Latin letters gets an "Add to English terms?" hint next to the script check. Adding it puts the name in the project's English terms.
6. **Undo** and **Redo** (or Ctrl+Z / Ctrl+Y) step through project changes. Every change saves `project.json` at once.
7. **Export WAV** writes the 24 kHz timeline. **Export SRT** writes the timeline words.
8. The **Advanced** panel (closed by default) holds guidance, base seed, pause, paragraph pause, the term lists and per-paragraph guidance and seed. **Apply** saves them into the project; the next render uses them.

### Measured speed

Measured on this machine (Windows, NVIDIA RTX 3050 Ti Laptop GPU, 4.29 GB VRAM; 16 GB RAM, no page file) with the release build. Default preset: base `k2-fsa/OmniVoice` + tr2 adapter `checkpoint-6000`, reference clip `ses_001_0007`, `guidance_scale = 2.5`, `num_step = 32`:

| Step | Measured |
|---|---|
| Engine start + voice model load on CUDA | 25.5 s (2.37 GB VRAM used) |
| Render 2 paragraphs (5.6 s of audio) | 12.1 s (0.46 audio-sec per wall-sec) |
| Whisper `large-v3-turbo` on the CPU next to the loaded engine | 52.6 s, no out-of-memory |
| Retake one paragraph with a new seed + splice planning | 33.2 s |
| Export | 140,640-sample WAV + non-empty SRT |

Whisper ran on the CPU while OmniVoice stayed on CUDA: no CUDA out-of-memory on the 4 GB card. VRAM stayed at about 2.4 GB of 4.3 GB through render and retake.

Run it yourself:

```
scripts\cargo.cmd test -p jeuvoice-app --release --test real_voice -- --ignored --nocapture
```

### Known limits

- Retakes render whole paragraphs; a retake never renders a lone sentence.
- The term hint only flags Latin-letter words with inner capitals (`iPhone`, `NATO`).
- The macOS build is not tested, and the voice engine has no Mac spike result yet: the Voice tab stays off there.
- GPU cards under 8 GB run Whisper on the CPU while the voice model is loaded. Starting a Transcribe batch unloads the engine first; the next Voice job loads it again.

### Try it by ear

1. Set the Hugging Face token or the local adapter folder in Settings.
2. Paste your real script (two paragraphs to start). Press **Generate** and listen from start to end.
3. Compare a paragraph with `tts_lab/outputs/omni_tr2_para_gs25_s4/full.wav`. They should sound like the same voice.
4. Drag a region over one sentence and press **Retake region**. Use the take that sounds best.
5. Open **Advanced**, set guidance to 3, and retake the word "damara". Keep the take that reads it cleanly.
