# JeuVoice — design (MVP)

Date: 2026-11-09. Status: draft, waiting for approval. Sections 10 and 11 come after the builder check and change parts of sections 4 and 5. Where they differ, sections 10 and 11 win.

Notation: **[user]** = the user decided this. **[call]** = my decision, the user can change it. Text without a mark is a fact about the current setup.

## 1. Intent

The team makes many content videos. Today the voice work goes like this:

- They write scripts by hand or with an LLM.
- They make voice takes with ElevenLabs (paid credits).
- They record their own voice-overs and need transcripts of them all the time.
- Agents (Claude Code and others) or a video editor then build the video.

The user fine-tuned their own Turkish voice model (OmniVoice + LoRA, run "tr2"). Today only the user can run it, from the terminal, with many flags.

JeuVoice is a native desktop app that gives this model and Whisper to people who do not code. **[user]** Later it can become a product or a paid service. **[user]**

Success for the MVP:

1. A teammate who does not code makes a full voice-over from a script, fixes bad words or parts, and exports it. They need no help and no terminal.
2. A teammate transcribes a batch of recorded takes, exports subtitles, and checks each take against its script.
3. The user starts an OmniVoice LoRA fine-tune on Kaggle from the app, without a hand-made bundle. Then they test the checkpoints by ear in the app.

## 2. Goals, non-goals, anti-goals

Goals (MVP):

- Three tabs: **Voice**, **Transcribe**, **Train**. **[user]**
- Voice model: OmniVoice tr2 only. The engine has a slot for more models later. **[user]**
- Remote training: Kaggle only. **[user]**
- Platforms: Windows with an NVIDIA GPU, and macOS. **[user]** A GPU is required. **[user]** On macOS this means Apple Silicon (Metal / MPS). **[call]**

Non-goals (MVP):

- An API, CLI or MCP server for agents. This comes in version 2. **[user]**
- RunPod and other cloud platforms. **[user]**
- VoxCPM2 GGUF / llama.cpp-omni and other voice models.
- A native Rust port of OmniVoice. This comes later, mainly for the product. **[user]**
- CPU-only machines, Linux, Intel Macs.
- User accounts, billing, cloud hosting, multi-user projects.

Anti-goals (the MVP fails if one of these is true, even when it works):

- The app renders sound worse than the current CLI render with the same settings.
- A teammate must open a terminal or edit a config file.
- The app breaks the lab setup (`D:\MLProjelerim\.venv_ml`, `tts_lab`).

## 3. Constraints

- UI: **Rune** (github.com/anomalyco/rune), Rust, GPU-drawn. **[user]** Rune is pre-release. Pin one rev of the `dev` branch. Rune can draw vector shapes (`canvas(shape)`, polylines) for the waveform, and it has a `drop-files` example. Rune apps run headless, so the builder can render a screen to PNG and check the element tree.
- OmniVoice runs only in Python/PyTorch today (Qwen3-style model, iterative diffusion decoding, Higgs Audio V2 tokenizer). Python is required for the MVP. **[user]** (accepted a Python engine process).
- Training is Python only. It runs on Kaggle (T4 ×2). The user's PC has an RTX 3050 Ti with 4 GB. That is too small for an OmniVoice LoRA.
- No ffmpeg on the user's PATH. Do not depend on a system ffmpeg.
- The current setup that the user chose by ear (memory note, 2026-11-07):
  - Model: base OmniVoice + tr2 LoRA adapter `outputs/omnivoice_tr2/checkpoint-6000`.
  - Reference clip: `data/ref/sesim/ses_001_0007`.
  - Flags: `--by-paragraph --pause 0.1 --para-pause 0.3 --guidance-scale 2.5`, plus `--en-term`, `--quote-term "..."`, `--respell`.
  - Per-part fixes: `--part-guidance N=3 --part-seed N=4`, `--only N`, `--try-seeds N=1,2,3`.
  - Code: `tts_lab/scripts/try_omnivoice.py`.
- Whisper and CER do not show stress or vowel stretch. Only the ear does. The app must make listening and comparing takes fast.

## 4. Architecture

```
┌──────────────── JeuVoice (Rust, Rune) ────────────────┐
│ Voice tab   Transcribe tab   Train tab   Settings     │
│ waveform view · audio playback · project files        │
│ whisper-rs (Whisper in-process)                       │
│ Kaggle client (HTTP)                                  │
└───────────────┬───────────────────────────────────────┘
                │ local HTTP on 127.0.0.1 + token
┌───────────────▼───────────────┐
│ jeuvoice-engine (Python)      │
│ OmniVoice + LoRA, data prep   │
│ /tts /retake /prep /health    │
└───────────────────────────────┘
```

### 4.1 Rust app

- One Cargo workspace. New git repo at `D:\MLProjelerim\jeuvoice`. **[call]** `tts_lab` stays the lab and is not changed.
- Crates **[call]**: `jeuvoice-app` (Rune UI), `jeuvoice-core` (project model, edit list, audio splice, script diff), `jeuvoice-whisper`, `jeuvoice-kaggle`, `jeuvoice-engine-client`.
- Audio playback and decode: `cpal` + `symphonia` (no ffmpeg). **[call]** For video input (mp4, mov), use symphonia for the audio track if it can. If it cannot, the engine decodes with PyAV, as tts_lab does today. **[call]**
- Secrets (the Kaggle API key) go in the OS keychain (`keyring` crate). **[call]**

### 4.2 Whisper

- `whisper-rs` (whisper.cpp bindings) in the app process. **[call]** CUDA on Windows, Metal on macOS.
- Model: `large-v3-turbo` GGML. The app downloads it on first use. **[call]**
- Word timestamps: use whisper.cpp token timestamps with DTW alignment for better word times. **[call]**
- Language: Turkish by default. The user can change it. **[call]**
- The Voice tab also uses Whisper. It gives the word times under the waveform and flags words that the model said wrong.

### 4.3 Python engine (`jeuvoice-engine`)

- The app starts the engine as a child process and restarts it if it crashes. The UI shows engine state: starting, ready, busy, error.
- Transport: HTTP on `127.0.0.1` on a free port, with a random token for each start. **[call]** Long jobs return a job id. The app reads progress with server-sent events or polling. **[call]** The version 2 agent API can use the same HTTP surface later.
- The engine wraps the logic of `try_omnivoice.py` as a module. Do not call the script as a CLI. Rendering with the same inputs and seed must give the same audio as the CLI (see section 8).
- Model slot: an `Engine` interface with `load`, `synthesize(text, ref, params, seed)`, and `capabilities`. OmniVoice is the only implementation in the MVP. **[call]**
- Device: CUDA on Windows, MPS on macOS. **Risk:** nobody has checked OmniVoice on MPS. The first milestone checks it (section 9).

### 4.4 Engine install

- The installer carries a portable Python (python-build-standalone) and `uv`. On first start, the app makes the engine environment in the app data folder. It installs PyTorch (CUDA build on Windows, default build on macOS) and the OmniVoice source at the pinned commit `08be0b4ccbac3e13e374e86fbfead4b4cac343e2`. **[call]**
- First start shows a setup screen with progress. It takes about 10–15 minutes and uses 3–5 GB.
- Model files: base OmniVoice from Hugging Face (`k2-fsa/OmniVoice`). The tr2 adapter comes from a private Hugging Face repo of the user. The reference clip ships in the app. See section 11.1.
- The app never uses or changes `D:\MLProjelerim\.venv_ml`.

### 4.5 Installers

- Windows: an MSI or NSIS installer. macOS: a signed `.dmg`. **[call]** The Apple signing and notarization account is left to the builder (it is needed only to share the app outside the user's own Mac).

## 5. Screens

All tabs share one waveform view (the "layout B" that the user chose). **[user]** The team learns one screen. Mockups: `tts_lab/.superpowers/brainstorm/1196-1794235908/content/editor-layout.html` (option B) and `train-tab.html`.

### 5.1 Voice tab

Flow **[user]**:

1. The user pastes a script or opens a `.txt` / `.md` file, then clicks **Generate**.
2. The engine renders with the setup in section 3. The result shows as one long waveform, with the words under it (word times from Whisper).
3. The user drags over a region of the waveform, or selects words. Then they click **Retake region**.
4. The side panel shows takes 1, 2, 3. The user plays each one and picks one. The pick goes into the waveform. The panel has a "+ try 3 more" button.
5. **Fix word:** the user selects a word and types how to say it ("say it as"). This uses the respell feature. Then the app retakes the region.
6. **Export:** WAV, plus SRT made from the final word times. **[call]** (SRT on export was my guess, and the user did not object.)

How retake works **[call]**:

- The app does not regenerate only the selected words. The model sounds bad when it starts in the middle of a sentence. The app regenerates the full sentence (or sentences) around the region. Then it puts the new audio in place with a short crossfade at the sentence edges (about 20–40 ms, at silence where it can).
- Each retake uses a new seed. The take keeps its seed and parameters, so the user can repeat it.
- Edits do not destroy audio. The project keeps the first render plus an edit list. Undo and redo work on the edit list.
- After a retake, Whisper runs again on the new sentence only, to update the word times.
- Whisper flags: if Whisper hears a word that is different from the script, the app marks that word. (Example: "damara" heard as "damaraa".) This is a hint only. Stress and vowel stretch need the ear.

Advanced panel (hidden by default) **[call]**: reference clip, guidance, seed, pause lengths, `en-term` and `quote-term` lists, per-part guidance and seed. A teammate never needs it. The user can use it.

### 5.2 Transcribe tab

1. The user drops one or more audio or video files. A batch is normal. **[user]**
2. Whisper makes the text with word times. The same waveform view shows the file.
3. The user can fix a wrong word in the text.
4. **Subtitles:** export SRT or VTT. **[user]**
5. **Script check [user]:** the user pastes a script, or picks the script of a Voice project. The app shows a word diff. Red: a script word that the speaker skipped or said differently. Green: a word that the speaker added. A click on a difference jumps to that point in the audio.
6. The diff ignores small differences before it compares **[call]**: case, punctuation, and numbers written as digits or as words ("2023" and "iki bin yirmi üç"). Turkish number-to-word logic is needed for this.

### 5.3 Train tab

One screen with four steps, top to bottom. **[user]** (approved the mockup).

1. **Data:** the user drops audio files or a folder, or pastes Hugging Face dataset links (section 11.2). The app does the prep that tts_lab scripts do today:
   - Whisper makes the text for each clip.
   - The app removes clipped clips, limits peak bursts (see `clean_my_voice.py`), and shows the bandwidth. See section 10.
   - The app writes the manifests (see `prep_omnivoice_data.py`).
   - The app shows counts and total minutes. The user can play clips and remove them.
2. **Base model:** OmniVoice + LoRA. The user can start from base OmniVoice or from a tr2 checkpoint.
3. **Where to train:** cards for This PC, Kaggle and RunPod. In the MVP only Kaggle works. **[user]** "This PC" shows a warning and is off when the GPU memory is too small. RunPod shows "coming later". **[call]**
4. **Script and run:** the main settings show in plain view (steps, LoRA r and alpha, batch tokens, save steps). Defaults are from tr2: 6,000 steps, r=32, alpha=64. "View full script" shows the real notebook. **Start on Kaggle** uploads and runs it. "Download bundle only" gives the zip and notebook for a manual run, like `make_omni_kaggle.py` today.

Kaggle flow **[call]**:

- Use the Kaggle API (the same calls as the `kaggle` CLI). Upload the data as a private dataset (new version for each run), push the notebook as a private kernel with GPU T4 ×2 and internet on, then poll the status.
- Make the audio tokens on Kaggle, not on the user's PC. This avoids the 4 GB GPU limit and keeps one code path. Upload the clean audio and the manifests.
- Keep the current Kaggle workarounds from `make_omni_kaggle.py`: pack the shards in one file with a different suffix, because Kaggle unpacks `.tar` files. Save only the LoRA adapters, because a full checkpoint is 2.8 GB.
- When the run ends, the app downloads the adapters. Each checkpoint gets a **test** button. It renders a fixed short test text with the Voice tab setup, so the user chooses by ear.
- Live progress: the Kaggle API gives the run status, but it may not give live logs while the kernel runs. The mockup showed a live loss bar. If the API cannot give it, the MVP shows status and elapsed time, and shows the loss curve after the run ends.

### 5.4 Settings

Kaggle API key, Hugging Face read token, engine status and logs, model files and disk use, default reference clip, Whisper language.

## 6. Projects and files

- A project is a folder with `project.json` (script, settings, edit list, takes with their seeds and parameters) and an `audio/` folder. **[call]**
- Default place: `Documents/JeuVoice/<project name>`. **[call]**
- Recent projects show on the start screen. **[call]**

## 7. Left to the builder

- Crate layout details, the HTTP routes, the JSON shapes, and the job and progress format.
- The waveform drawing method in Rune, and zoom and scroll behavior.
- Crossfade length and the exact cut-point search.
- The Rune visual style. Use the Rune kit components and theme.
- Error texts. Write them in plain words that a teammate can act on.

## 8. Tests that matter

- **Same sound as the CLI:** a test renders a fixed paragraph with the engine and with `try_omnivoice.py`, with the same seed and settings, and compares the samples. They must match within float tolerance on the same device.
- Retake splice: unit tests on the edit list and crossfade (no clicks, correct length, undo and redo).
- Script diff: unit tests with Turkish text, numbers as digits and words, and punctuation.
- Kaggle client: tests against recorded API responses. Do one real run by hand before release.
- UI: Rune headless renders of each tab to PNG, plus element-tree checks for the main flows.

## 9. Order of work [call]

1. **Spike (1–2 days):** run OmniVoice tr2 on a Mac with MPS. Check speed and sound. If it fails, macOS gets the Transcribe tab only in the MVP, and the user decides what to do.
2. Python engine with `/tts` and `/retake`, plus the "same sound as the CLI" test.
3. Rust app shell with Rune: tabs, engine start and restart, settings.
4. Transcribe tab (Whisper, waveform, SRT, script check).
5. Voice tab (generate, retake, takes panel, fix word, export).
6. Train tab with Kaggle.
7. Installers and the first-start setup screen.

## 10. Calls made after the builder check

- **GPU memory [call].** Teammate PCs are like the user's PC (4 GB card) **[user]**. Design for 4 GB:
  - The engine keeps OmniVoice loaded on the GPU. The audio tokenizer stays on the CPU on every card, as in the CLI. This keeps "same sound as the CLI".
  - Whisper uses the GPU only when it fits next to the engine. Otherwise Whisper runs on the CPU. A retake needs Whisper on one sentence only, so the CPU is fast enough.
  - In the Transcribe tab with no Voice job, the app can ask the engine to unload the model, so Whisper gets the GPU.
  - macOS minimum: Apple Silicon with 16 GB unified memory.
- **Retake unit [call].** The user found that sentence mode loses liveliness and stress (memory note, 2026-11-07). So a retake renders the **whole paragraph** with a new seed, in the same paragraph mode as the first render. The app then cuts the sentence or sentences that cover the region from the new render (with the Whisper word times) and splices them in. The edit list stores takes for each paragraph, with the seed and parameters. This maps to the CLI's `--only N` and `--part-seed N=S`. The Mac spike (section 9, step 1) also checks by ear that the splice sounds right.
- **Data cleaning [call].** The MVP does what `clean_my_voice.py` does (remove clipped clips, limit peak bursts), plus a bandwidth check and a Whisper confidence check. There is no new noise or music detector in the MVP.
- **Term lists [call].** Settings has a shared term list for the team: English terms, quoted titles and respellings. It starts with the user's current lists. Each project can add its own terms. The Voice tab suggests a term when Whisper flags a Latin-script word that the model read badly.
- **Exact render preset [call].** Before step 2 of the work, the builder writes the exact CLI command of the chosen setup into a preset file: `--ref`, `--ref-text`, `--seed` (default 1), `--num-step` (default 32) and the flags in section 3. The engine always passes `--ref-text` with the reference clip, so OmniVoice never needs its own speech recognition model. The "same sound as the CLI" test uses this preset.
- **Kaggle live progress.** Settled in section 5.3: if the API gives no live logs, the MVP shows status and elapsed time, then the loss curve after the run.

## 11. Answers from the user after the builder check

### 11.1 Model files and voice presets

- The reference clip `ses_001_0007` ships inside the app as the first voice preset. **[user]**
- The tr2 adapter and other model files come from a private Hugging Face repo of the user. **[user]** Each teammate enters a read token in Settings. **[call]**
- **Voice presets are a must-have for the MVP. [user]** A voice preset is a saved voice that the user picks in the Voice tab. Two kinds **[user]**:
  - **Clone preset:** a reference clip (3–10 s) plus its exact text (`--ref-text`). The user makes one by dropping a clip. Whisper makes the text, and the user checks it.
  - **Seed preset:** no reference clip. A seed plus optional voice design attributes (`--instruct`, for example "male, young adult"). The user tries seeds, listens, and saves the one they like.
- A preset also stores the model (base or LoRA checkpoint) and the render settings (guidance, pauses). **[call]**
- Presets are files in a presets folder. The user can share a preset file with a teammate. **[call]**

This changes section 5.1: the Voice tab has a preset picker next to **Generate**. The default preset is `ses_0007` with the section 3 setup.

### 11.2 Training data sources and the recipe page

- **Sources [user]:** local files with drag and drop, and Hugging Face dataset links. The user pastes a link and picks the split, the audio column and the text column. **[call]** If the dataset has no text, Whisper makes it.
- **Mix [call]:** each source gets a repeat weight (the `*N` of `prep_omnivoice_data.py`), shown as "×3". The data step shows minutes for each source after the weights.
- **Recipe page [user]:** a page in the Train tab that shows how tr2 was made: each dataset, its minutes, its weight, the base model, and the settings (6,000 steps, r=32, alpha=64). The user can copy the recipe and change it.
- **"Train like tr2" button [user]:** it fills the data step with the tr2 recipe and downloads the sources that have a link. Then the user adds their own clips and starts.
- **Builder note [call]:** the exact tr2 mix is not written down in one place. Rebuild it from `tts_lab/data/omni2/train.jsonl` (count rows for each source and repeat) and the tr2 notebook. Antalia (`data/antalia`) is a local corpus with no public link that I know of. If a tr2 source has no link, the recipe page says so and asks the user to add it from disk.
- **Licenses [call]:** the recipe page shows the license of each source (for example YODAS2). This matters when the app becomes a paid product.
