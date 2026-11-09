# JeuVoice Plan 2 — Python Engine, Voice Presets and the Voice Tab

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A teammate opens the Voice tab, pastes a script, picks a voice preset, clicks **Generate**, fixes bad words or regions with retakes, and exports WAV + SRT. The sound is the same as the user's CLI render with OmniVoice tr2.

**Architecture:** A Python process (`jeuvoice-engine`) runs OmniVoice + the tr2 LoRA. The Rust app starts it, talks to it over local HTTP with a token, and restarts it if it crashes. All edit logic (paragraph takes, sentence splices, undo) is pure Rust in `jeuvoice-core`. The Voice tab reuses the waveform, playback and Whisper from Plan 1.

**Tech Stack:** Rust (existing workspace, Rune pinned), Python 3.12 + PyTorch + OmniVoice (pinned commit) + peft + trnorm, `uv` for the engine environment, stdlib `http.server` for the engine API, `ureq` in Rust, `huggingface_hub` for model files, `keyring` for the HF token.

**Spec:** `docs/design.md` — sections 3, 4.3, 4.4, 5.1, 6, 9, 10, 11.1. Where 10/11 differ from 4/5, 10/11 win.

## Before you start

- Plan 1 lives on branch `plan1-transcribe`. Ask the user to merge it into `main` first (the user decides. Never merge yourself). Then branch `plan2-voice` from `main`.
- If the working tree has uncommitted changes from another agent, stop and ask the user what to do with them.
- Read `README.md` and `docs/plans/1-transcribe.md` "Global Constraints" once.

## Global Constraints

- Repo `D:\MLProjelerim\jeuvoice`. Never write into `D:\MLProjelerim\tts_lab` (read-only source of facts and test media). Never install into or run from `D:\MLProjelerim\.venv_ml`.
- Rune pin stays `rev = "b3a33a0f3a5c35d78f188046a040696193cfb6bf"`.
- Build and test Rust only through `scripts\cargo.cmd` (plain cargo fails on this machine).
- Engine Python environment: `engine/.venv` (git-ignored) made with `uv`. Python 3.12. Pinned versions (copy exactly): `torch==2.14.0` from the CUDA 12.6 index on Windows (`https://download.pytorch.org/whl/cu126`), default index on macOS; `transformers==5.19.0`; `peft==0.21.2`; `accelerate==1.15.0`; `omnivoice[lora] @ git+https://github.com/k2-fsa/OmniVoice@08be0b4ccbac3e13e374e86fbfead4b4cac343e2`; `trnorm @ git+https://github.com/ozguradmin/ahenk-turkish-tts` (find the right subdirectory/ref in that repo. Tts_lab uses trnorm 0.1.0); `soundfile`, `numpy`, `huggingface_hub`.
- The render setup the user chose by ear (default preset): base `k2-fsa/OmniVoice` + adapter tr2 `checkpoint-6000`. Reference `ses_001_0007.wav` + its exact text `ses_001_0007.txt`. Paragraph mode; `pause = 0.1`, `para_pause = 0.3`, `guidance_scale = 2.5`, `num_step = 32`, `speed = 1.0`, `seed = 1`. Part seed = `seed + i` (1-based part index) when a reference clip is used. Sample rate 24000. Quote term "The Last of Us".
- The audio tokenizer always runs on the CPU (as in `try_omnivoice.py`). This keeps "same sound as the CLI" and fits a 4 GB GPU.
- A retake renders the **whole paragraph** with a new seed and splices in only the sentences that cover the user's region (spec 10, "Retake unit"). Never render a sentence alone.
- GPU sharing on cards under 8 GB: while the engine holds OmniVoice on CUDA, Whisper runs on the CPU. In the Transcribe tab with no Voice job, the app may ask the engine to unload.
- All user-visible text: plain short English sentences that say what to do. No raw exception text in the UI (keep details in the engine log).
- Projects: `Documents/JeuVoice/<project name>/project.json` + `audio/`. Presets: `<app data>/presets/<name>/preset.json` (+ `ref.wav`).
- Secrets (HF token) go in the OS keychain (`keyring` crate, service `JeuVoice`).

## Review Focus

1. **Same sound as the CLI:** a render through the engine must match `tts_lab/scripts/try_omnivoice.py` sample for sample (same device, seed, text, settings). Test in Task 4.
2. **Turkish text in, Turkish text out:** `İ/ı`, apostrophes in English titles ("The Last of Us'ta"), and digits go through `clean()` exactly as in the CLI. Test in Task 3.
3. **Engine crash or slow start:** the first start loads 3 GB of weights. The UI shows "Starting the voice engine…" and never freezes. A crash shows a plain message and a **Restart** button. Test in Task 6 and Task 11.
4. **Retake at a paragraph edge and at the clip end:** a region that touches the first or last sentence, or spans two paragraphs, splices without clicks and keeps the total length correct. Test in Task 8.
5. **Engine and Whisper on one 4 GB GPU:** Voice tab Whisper runs on the CPU while OmniVoice is on CUDA. No CUDA out-of-memory. Test in Task 7 (rule) and Task 14 (real run).

---

### Task 1: Mac spike script (the user runs it on a Mac)

**Files:** Create `engine/tools/mac_spike.py`, `docs/mac-spike.md`.

The agent cannot run this. Write it so a teammate with an Apple Silicon Mac can run it in 15 minutes.

- [ ] **Step 1:** `mac_spike.py`: makes nothing outside its own temp folder. Loads OmniVoice on `mps` with fp16 (fall back to fp32 if fp16 fails, and print which), loads the tr2 adapter from a folder path argument, keeps the audio tokenizer on CPU, renders the fixed paragraph in `docs/mac-spike.md` with the default preset settings, writes `spike.wav`, and prints load time, render time, audio seconds and RTF.
- [ ] **Step 2:** `docs/mac-spike.md`: exact steps (install uv, `uv venv`, install the pinned packages for macOS, copy the adapter and `ses_001_0007.*`, run the script), and what to send back (the printed numbers and `spike.wav`).
- [ ] **Step 3:** Commit: `docs: Mac spike script`. Tell the user it is ready. Do not wait for the result. Continue with Task 2. Record the result later in `docs/mac-spike.md` when the user gives it. If the spike fails, Task 11 shows "The voice engine does not run on this Mac yet." on macOS and the Voice tab stays off there.

### Task 2: Engine package, environment and health endpoint

**Files:**
- Create: `engine/pyproject.toml`, `engine/jeuvoice_engine/__init__.py`, `engine/jeuvoice_engine/__main__.py`, `engine/jeuvoice_engine/server.py`, `engine/tests/test_server.py`, `scripts/engine-dev.cmd` (makes `engine/.venv` with uv and the pinned packages)
- Modify: `.gitignore` (`engine/.venv/`, `__pycache__/`)

**Interfaces:**
- Produces: `python -m jeuvoice_engine --port 0 --token <T> --data-dir <D> [--backend omnivoice|fake]`. On start it prints exactly one line to stdout `JEUVOICE_ENGINE_READY port=<N>` after the HTTP server listens. Every request needs header `X-JeuVoice-Token: <T>` (else 401). `GET /health` → `{"ok": true, "backend": "...", "device": "cuda|mps|cpu|none", "vram_total_gb": float|null, "model_loaded": bool, "version": "<engine version>"}`. `POST /shutdown` → stops the process. Logs go to `<D>/logs/engine.log`.
- The `fake` backend never imports torch. It exists so the Rust client and app tests run without a GPU.

- [ ] **Step 1: Write failing tests** (`pytest`, fake backend, start the server on port 0 in a thread): `test_ready_line_and_health`, `test_wrong_token_is_401`, `test_shutdown_stops_server`.
- [ ] **Step 2:** Run `engine\.venv\Scripts\python -m pytest engine/tests -q`. Expected: FAIL.
- [ ] **Step 3:** Implement with `http.server.ThreadingHTTPServer` and JSON bodies. `scripts/engine-dev.cmd` runs `uv venv engine\.venv --python 3.12` and `uv pip install` with the pins from Global Constraints (and `pytest`).
- [ ] **Step 4:** Run the tests. Expected: PASS. Run `scripts\engine-dev.cmd` once for real (the torch download is about 2.5 GB. Run it in the background).
- [ ] **Step 5:** Commit: `feat(engine): package, health endpoint, dev environment`.

### Task 3: Engine text pipeline (port of the CLI)

**Files:** Create `engine/jeuvoice_engine/text.py`, `engine/tests/test_text.py`, `engine/tests/fixtures/` (copies of 3 short scripts).

**Interfaces:**
- Produces: `@dataclass Part(index: int, paragraph: int, language: str, text: str)` (index is 1-based, as in the CLI), `def plan_parts(script: str, en_terms: list[str], quote_terms: list[str], respell: list[tuple[str, str]]) -> list[Part]`. Behavior = `try_omnivoice.py` with `--by-paragraph`: `split_paragraphs` (from `render_script_gguf.py`), `split_by_language`, `clean` (trnorm `normalize`, apostrophe rule, `I→ı`, `İ→i`, lowercase, quote terms), then respell `OLD=NEW` on the cleaned text.

- [ ] **Step 1: Write failing tests:** for each fixture script, `plan_parts` equals the parts the CLI functions give. Import the CLI functions read-only from `D:/MLProjelerim/tts_lab/scripts` with `sys.path` in the test only (skip the test with a clear reason if that folder is missing). Add exact tests: `"The Last of Us'ta Joel"` with quote term keeps the title in quotes in an `en` part; `"İSTANBUL IŞIK"` becomes `"istanbul ışık"`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement (copy the logic. Do not import tts_lab at runtime).
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(engine): text pipeline identical to the CLI`.

### Task 4: OmniVoice backend and the "same sound" test

**Files:** Create `engine/jeuvoice_engine/backend.py` (interface + fake), `engine/jeuvoice_engine/omnivoice_backend.py`, `engine/tests/test_backend_fake.py`, `engine/tests/test_same_sound.py` (marked `@pytest.mark.gpu`, skipped unless `--gpu`).

**Interfaces:**
- Produces:
  - `class Backend(Protocol)`: `load(model: str, adapter_dir: str | None, device: str) -> None`, `unload() -> None`, `make_prompt(ref_wav: str | None, ref_text: str | None) -> object | None`, `render(part: Part, prompt, instruct: str | None, seed: int, guidance: float, num_step: int, speed: float, duration: float | None) -> np.ndarray` (float32 mono, 24 kHz), `device: str`, `loaded: bool`.
  - `FakeBackend`: `render` returns a deterministic 220 Hz tone, length `0.06 s × len(part.text)`, seeded amplitude, so tests can tell seeds apart.
  - `OmniVoiceBackend`: `OmniVoice.from_pretrained(model, device_map=device, dtype=float16)`, `load_lora_adapter`, audio tokenizer moved to CPU, `torch.cuda.empty_cache()`; `render` does `torch.manual_seed(seed)` then `model.generate(...)` exactly as `try_omnivoice.render`.
  - `def seed_for(part_index: int, base_seed: int, has_prompt: bool) -> int` (= `base_seed + part_index` with a prompt, else `base_seed`).

- [ ] **Step 1: Write failing tests:** fake backend tests (length rule, seed changes the samples, `seed_for`). `test_same_sound` (GPU): render the first paragraph of a fixture script with the default preset through `OmniVoiceBackend`, then run `try_omnivoice.py` on the same paragraph with the same flags (`--by-paragraph --pause 0.1 --para-pause 0.3 --guidance-scale 2.5 --quote-term "The Last of Us" --lora-adapter <tts_lab ckpt-6000> --ref <ses_001_0007.wav> --ref-text <ses_001_0007.txt> --out <tmp name>`) with `PYTHONPATH=third_party/omnivoice_env` and the `.venv_ml` python **only as a read-only reference run** (it writes into `tts_lab/outputs/<tmp name>`: use a name starting with `jeuvoice_check_` and delete that folder after the test). Assert max abs difference ≤ 1e-4 for every part.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the fake tests, then `pytest -m gpu --gpu`. Expected: PASS. If the reference run differs, find why before you continue (dtype, tokenizer device, seed order). Write the max difference in the report. NOTE: the reference CLI needs `PYTHONPATH=third_party/omnivoice_src` (the pinned source checkout with `utils/lora`); `omnivoice_env` is older and has no LoRA support.
- [ ] **Step 5:** Commit: `feat(engine): OmniVoice backend matching the CLI`.

Note: the reference run writes into tts_lab/outputs. This is the only allowed write there. Ask the user once before the first run.

### Task 5: Engine job API

**Files:** Create `engine/jeuvoice_engine/jobs.py`, modify `server.py`, create `engine/tests/test_jobs.py`.

**Interfaces:**
- Produces HTTP endpoints (all JSON, all need the token):
  - `POST /load {"model": str, "adapter_dir": str|null, "device": "auto"|"cuda"|"mps"|"cpu"}` → job id. `POST /unload` → `{"ok": true}` (refused with 409 while a render runs).
  - `POST /render {"script": str, "paragraphs": [int]|null, "preset": {...preset.json...}, "terms": {"en": [...], "quote": [...], "respell": [[old,new],...]}, "base_seed": int, "part_overrides": {"<i>": {"seed": int, "guidance": float}}, "out_dir": str}` → `{"job": id}`. Renders the listed paragraphs (null = all). Writes `out_dir/part_<i>.wav` (24 kHz float WAV) per part and returns, in the job result, `{"parts": [{"index", "paragraph", "language", "text", "seed", "guidance", "file", "audio_s"}], "pause": float, "para_pause": float}`.
  - `GET /jobs/<id>` → `{"state": "waiting"|"running"|"done"|"failed"|"cancelled", "progress": 0..1, "message": str, "result": {...}|null, "error": str|null}`. `POST /jobs/<id>/cancel` (takes effect between parts).
  - One GPU worker thread runs jobs one at a time, in order.
- Errors in `error` are plain words. The traceback goes only to the log.

- [ ] **Step 1: Write failing tests** with the fake backend: render two paragraphs → files exist, seeds follow `seed_for`, `part_overrides` wins, progress reaches 1.0. Cancel between parts → state `cancelled`; `/unload` during a render → 409. A backend exception → `failed` with a plain message.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(engine): job API for load, render and cancel`.

### Task 6: Rust engine client

**Files:** Create `crates/jeuvoice-engine-client/{Cargo.toml,src/lib.rs,src/process.rs,src/api.rs,tests/fake_engine.rs}`. Add to the workspace.

**Interfaces:**
- Produces:
  - `pub struct EngineConfig { pub python: PathBuf, pub engine_dir: PathBuf, pub data_dir: PathBuf, pub backend: String }` (`backend` is `"omnivoice"` or `"fake"`).
  - `pub struct EngineProcess` with `fn start(cfg: &EngineConfig) -> Result<Self, EngineError>` (spawns `python -m jeuvoice_engine --port 0 --token <random 32 hex> ...` with `PYTHONPATH=engine_dir`, waits up to 60 s for the ready line on a reader thread, never on the UI thread), `fn is_alive(&mut self) -> bool`, `fn stop(self)` (POST /shutdown, then kill after 3 s).
  - `pub trait EngineApi: Send + Sync` with `health`, `load`, `unload`, `render(&RenderRequest) -> Result<JobId>`, `job(&JobId) -> Result<JobStatus>`, `cancel(&JobId)`. `pub struct HttpEngine` implements it with `ureq`. `pub struct FakeEngineApi` (in-memory, for app tests) too.
  - Serde types mirror Task 5 exactly: `RenderRequest`, `Preset`, `PartOverride`, `JobStatus`, `JobState`, `RenderResult`, `PartResult`, `Health`.
  - `Terms { en: Vec<String>, quote: Vec<String>, respell: Vec<(String, String)> }` lives in a new small module `jeuvoice-core/src/terms.rs`, made in this task. The client, the project model (Task 8) and Settings (Task 9) all use this one type.
  - `pub enum EngineError` with plain `Display` texts, for example "The voice engine did not start. Open Settings and click Restart engine." and "The voice engine stopped. Click Restart engine."

- [ ] **Step 1: Write failing tests** (`tests/fake_engine.rs`): start the real Python engine with `backend = "fake"` (python = `engine/.venv/Scripts/python.exe` on Windows, `engine/.venv/bin/python` elsewhere. Skip with a message if missing), health ok, render one paragraph, poll to done, files exist. Kill the process → `is_alive` false and calls give the "stopped" error. A wrong token → error.
- [ ] **Step 2:** Run `scripts\cargo.cmd test -p jeuvoice-engine-client`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(engine-client): start, talk to and stop the voice engine`.

### Task 7: App services refactor (no behavior change)

The final review of Plan 1 asked for this before the Voice tab.

**Files:**
- Create: `crates/jeuvoice-app/src/services/{mod.rs,speech.rs,player.rs,engine.rs}`
- Modify: `state.rs` (split. Target under 400 lines), `transcribe_tab.rs`, `settings_tab.rs`, `main.rs`, `waveform.rs` (add a `key: &str` parameter used for the wave element and the layout read. The Transcribe tab passes `"wave"`)

**Interfaces:**
- Produces:
  - `services::speech::SpeechService`: owns the single Whisper worker and job queue. `fn submit(&mut self, job: SpeechJob) -> JobTicket` where `SpeechJob` is `File { path }` (Transcribe tab) or `Samples { samples_16k: Arc<Vec<f32>>, tag: u64 }` (Voice tab paragraphs); `fn poll(&mut self) -> Vec<(JobTicket, Update)>`; `fn set_force_cpu(&mut self, on: bool)` (applied before the next job. Reloads the model like the GPU switch does).
  - `services::player::SharedPlayer`: one `Playback` for the whole app; `fn load(&mut self, owner: PlayerOwner, audio: Audio)` (takes `Audio` by value. No clone), `owner()`, so the tab that does not own the player shows its play bar as stopped.
  - `services::engine::EngineService`: wraps `EngineProcess` + `HttpEngine`. State `Off | Starting | Ready(Health) | Failed(String)`; `fn ensure_started(&mut self)` (on a worker), `fn restart(&mut self)`, `fn api(&self) -> Option<Arc<dyn EngineApi>>`. Not started until the Voice tab first needs it.
  - GPU rule: `fn whisper_must_use_cpu(engine: &EngineState) -> bool` = engine `Ready` with `device == "cuda"` and `model_loaded` and `vram_total_gb < 8.0`.
- [ ] **Step 1: Write failing tests:** `whisper_must_use_cpu` table test; `SharedPlayer` owner switch. Existing Transcribe and Settings tests still pass unchanged (they are the safety net).
- [ ] **Step 2:** Run. Expected: the new tests FAIL.
- [ ] **Step 3:** Refactor. Move code. Do not change behavior. Fix the Plan 1 ledger minors that live in moved code only if trivial (export target index capture, cache save order through one writer thread).
- [ ] **Step 4:** Run `scripts\cargo.cmd test --workspace`. Expected: PASS, same test count or more.
- [ ] **Step 5:** Commit: `refactor(app): shared speech, player and engine services`.

### Task 8: Project model, edit list and splice (pure core)

**Files:** Create `crates/jeuvoice-core/src/project.rs`, `crates/jeuvoice-core/src/splice.rs`, tests `tests/project.rs`, `tests/splice.rs`.

**Interfaces:**
- Produces:
  - `struct Project { version: u32 (=1), name: String, script: String, preset: String, terms: Terms, base_seed: u64, paragraphs: Vec<Paragraph> }`; `struct Paragraph { index: usize, text: String, takes: Vec<Take>, base_take: usize, splices: Vec<Splice>, words: Vec<Word> }`; `struct Take { id: u32, seed: u64, guidance: f32, parts: Vec<TakePart> /* file, language, text, audio_s */, file: String /* joined paragraph wav, relative to audio/ */ }`; `struct Splice { take: u32, src: (f64, f64), dst: (f64, f64) }` (seconds; `dst` in the paragraph timeline before this splice).
  - `fn paragraph_audio(p: &Paragraph, load: impl Fn(&str) -> Audio) -> Audio` (base take + splices in order, each with a crossfade of `CROSSFADE_S = 0.03` at both edges).
  - `fn timeline(project: &Project, ...) -> (Audio, Vec<(usize, f64)>)` = paragraphs joined with `para_pause` silence. Returns each paragraph's start time.
  - `fn sentence_spans(words: &[Word], paragraph_text: &str) -> Vec<(f64, f64)>`: one span per script sentence (split on `.?!`), mapped through `script_diff`. A cut point between two sentences is the lowest-RMS 20 ms window inside the gap between their words (pass the audio in a variant `cut_points(audio, spans)`).
  - `fn plan_splice(old: &Paragraph, old_audio: &Audio, new_take: &Take, new_audio: &Audio, new_words: &[Word], region: (f64, f64)) -> Splice`: covers every sentence that overlaps `region`. Source and destination use the cut points of the same sentence indices.
  - `struct History { undo: Vec<Project>, redo: Vec<Project> }` with `push`, `undo`, `redo` (snapshots. Projects are small JSON).
  - `fn save(project: &Project, dir: &Path) -> io::Result<()>` (write to `project.json.tmp`, then rename) and `fn load(dir: &Path) -> Result<Project, ProjectError>`.
- [ ] **Step 1: Write failing tests:**
  - A splice in the middle keeps total length = old − old span + new span.
  - The crossfade has no jump > 0.05 at the seams (two sine waves).
  - A region at the first sentence and a region at the last sentence (Review Focus 4).
  - A region across a paragraph boundary gives two splices, one for each paragraph.
  - `sentence_spans` on a 3-sentence fixture with words.
  - Save/load round trip, and History undo/redo order.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(core): project model, sentence spans and splices`.

### Task 9: Voice presets and model files

**Files:** Create `crates/jeuvoice-core/src/preset.rs` (+ tests), `crates/jeuvoice-app/src/services/models.rs`, `assets/presets/ses_0007/{preset.json,ref.wav,ref.txt}` (copy from `tts_lab/data/ref/sesim/ses_001_0007.*`), `engine/tools/upload_adapter.py`. Modify `settings.rs`.

**Interfaces:**
- Produces:
  - `struct Preset { name: String, kind: PresetKind /* Clone | Seed */, ref_audio: Option<String>, ref_text: Option<String>, seed: u64, instruct: Option<String>, model: ModelRef { base: String, adapter: AdapterRef }, guidance: f32, pause: f32, para_pause: f32, num_step: u32, speed: f32 }`; `enum AdapterRef { None, HfRepo { repo: String, subfolder: String }, Local { dir: String } }`. JSON field names match Task 5's `preset`.
  - `fn list_presets(dir: &Path) -> Vec<Preset>`, `fn save_preset(dir: &Path, p: &Preset, ref_wav: Option<&Path>)`, `fn install_default_presets(app_assets: &Path, presets_dir: &Path)` (copies `ses_0007` on first start only).
  - Default preset `ses_0007`: kind Clone, adapter `HfRepo { repo: "emirbakid/jeuvoice-omnivoice-tr2", subfolder: "checkpoint-6000" }`, settings from Global Constraints.
  - Settings gains: `hf_token` (keychain, never in settings.json), `local_adapter_dir: Option<String>` (when set, overrides every HfRepo adapter — for development with `tts_lab/outputs/omnivoice_tr2/checkpoint-6000`), `terms: Terms` (seeded with quote term "The Last of Us").
  - `services::models::resolve_adapter(&AdapterRef, &Settings) -> Result<PathBuf>`: Local → the folder. HfRepo → engine downloads with `huggingface_hub.snapshot_download(repo, allow_patterns=[subfolder/*], token)` into `<app data>/models/` (add `POST /fetch_adapter` to the engine for this. A missing token or a 401 gives "Add your Hugging Face token in Settings.").
  - `upload_adapter.py`: uploads a local checkpoint folder to a private HF repo. **The user runs it**. The agent never uploads.
- [ ] **Step 1: Write failing tests:** preset JSON round trip; `install_default_presets` copies once and does not overwrite user edits; `local_adapter_dir` override wins. Token is read from a fake keyring store in tests.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. In Settings add: "Hugging Face token" field (saved to the keychain), "Use a local adapter folder" picker, "Restart engine" button, engine status line.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: voice presets and model files`. Tell the user: run `engine/tools/upload_adapter.py` to put the tr2 adapter on Hugging Face, or set the local adapter folder.

### Task 10: Preset editor (make clone and seed presets)

**Files:** Create `crates/jeuvoice-app/src/voice/preset_editor.rs` (+ tests).

- Clone preset: the user drops a 3–10 s clip. The app trims nothing, checks the length (outside 3–10 s → "Use a clip between 3 and 10 seconds."), runs Whisper on it (SpeechService, `Samples` job) and shows the text in a field the user must check. **Try** renders the fixed test sentence "Bugün size kısa bir hikâye anlatacağım." with the clip. **Save** writes the preset.
- Seed preset: the user types optional voice attributes (`instruct`), clicks **Try 3 voices** (seeds base, base+1, base+2), plays each, picks one, names it, saves.
- [ ] **Step 1: Write failing headless tests** with `FakeEngineApi` and `FakeTranscriber`: a 2 s clip gives the length message. A 5 s clip fills the text field from the fake transcript. Try 3 voices sends three renders with three seeds. Save writes a preset file that `list_presets` returns.
- [ ] **Step 2–4:** Implement, run, PASS.
- [ ] **Step 5:** Commit: `feat(app): preset editor for clone and seed voices`.

### Task 11: Voice tab, part 1 — generate, play, export, projects

**Files:** Create `crates/jeuvoice-app/src/voice/{mod.rs,state.rs,view.rs,tests.rs}`. Modify `view.rs` (enable the Voice tab), `main.rs`.

Layout (layout B, as in `docs/mockups/editor-layout.html`): top bar with project name, **Open…**, **New**, preset picker, **Generate**, **Export WAV**, **Export SRT**. Main area: the script field (before the first render) or the timeline waveform with words. Side panel: takes (Task 12). Bottom: play bar.

- **Generate** does these steps in order:
  1. Start the engine if needed. Show "Starting the voice engine… This takes about a minute the first time."
  2. Load the preset model (`/load`).
  3. Render all paragraphs (`/render`). Show "Paragraph 3 of 12".
  4. Join each paragraph's parts with `pause` into `audio/p<NN>_t<take>.wav`.
  5. Run Whisper on each paragraph (CPU when `whisper_must_use_cpu`). Make word times and marks with `script_diff`, as in the Transcribe tab.
- **Projects:** New asks for a name. Every change saves `project.json` (on a worker). Open lists `Documents/JeuVoice/*/project.json`.
- **Export WAV:** 24 kHz 16-bit WAV of the timeline. **Export SRT:** from the timeline words.
- Engine failure: a plain message and **Restart engine**. The project stays.
- macOS without a passing spike: the tab shows "The voice engine does not run on this Mac yet." (a cfg flag set by Task 1's result. Default on).
- [ ] **Step 1: Write failing headless tests** with `FakeEngineApi` + `FakeTranscriber` + temp dirs: paste a 2-paragraph script, Generate → progress, then a timeline with words. Project.json exists and reopens to the same state. Export WAV writes a file with the timeline length. An engine failure shows the message and the Restart button.
- [ ] **Step 2–4:** Implement, run, PASS. Snap the tab to PNG and look at it.
- [ ] **Step 5:** Commit: `feat(app): Voice tab generate, play, export and projects`.

### Task 12: Voice tab, part 2 — retake, takes panel, fix word, undo

**Files:** Modify `crates/jeuvoice-app/src/voice/*`.

- **Retake region:** the user drags a region (or selects words). **Retake region** (key `retake`) renders the covering paragraph(s) with 3 new seeds (one job per seed. Seeds = base_seed + 1000 × take id + part index), runs Whisper on each new take, and fills the side panel with "Take 1 / Take 2 / Take 3" (keys `take-<n>`), each with ▶. **+ Try 3 more** adds three. **Use this take** applies `plan_splice` and pushes History.
- **Fix word:** select a word → "Say it as" field (prefilled with the word) → adds a respell `OLD=NEW` to the project terms (lowercase, as in the CLI) → retakes the covering paragraph.
- **Term hint** (spec 10): when Whisper flags a word written in Latin letters that is not Turkish (for example an English name), show "Add to English terms?" next to it. A click adds it to the project terms.
- **Undo / Redo** (keys `undo`, `redo`, and Ctrl+Z / Ctrl+Y).
- **Advanced** panel (closed by default): guidance, base seed, pause, para pause, en terms, quote terms, respell list, per-paragraph guidance and seed (`part_overrides`).
- [ ] **Step 1: Write failing headless tests:** select a region in paragraph 2 → Retake → three takes appear → Use take 2 → the timeline length changes by (new span − old span) and paragraph 1 is unchanged. Undo restores the old timeline. Fix word sends a render with the respell in `terms.respell`. Advanced guidance change reaches the next render request.
- [ ] **Step 2–4:** Implement, run, PASS. Snap and look.
- [ ] **Step 5:** Commit: `feat(app): retakes, fix word, undo and advanced settings`.

### Task 13: Transcribe tab unloads the engine when idle

**Files:** Modify `services/engine.rs`, `transcribe_tab.rs`.

- When the user starts a Transcribe batch and no Voice job runs and the GPU is under 8 GB, the app calls `/unload` first, so Whisper gets the GPU. The next Voice job loads the model again ("Loading the voice model…").
- [ ] **Step 1: Write a failing test** with `FakeEngineApi` that counts unload/load calls.
- [ ] **Step 2–4:** Implement, run, PASS.
- [ ] **Step 5:** Commit: `feat(app): share the GPU between the voice engine and Whisper`.

### Task 14: Real check and README

- [ ] **Step 1:** `scripts\cargo.cmd build --release -p jeuvoice-app` and `scripts\cargo.cmd test --workspace`; `engine\.venv\Scripts\python -m pytest engine/tests -q` and `-m gpu --gpu`.
- [ ] **Step 2:** An ignored Rust end-to-end test with the real engine + the local adapter folder (`tts_lab/outputs/omnivoice_tr2/checkpoint-6000`, read-only): render a 2-paragraph Turkish script, run Whisper on CPU next to the loaded engine (no CUDA OOM on the 4 GB card), retake one sentence region, export WAV + SRT. Print load time, render RTF, retake time, peak VRAM from `/health`.
- [ ] **Step 3:** README: "Voice tab" section (how to set the token or the local adapter folder, how to make a preset, how to retake), measured numbers, known limits, and a "Try it by ear" list for the user (generate the user's real script, compare with `tts_lab/outputs/omni_tr2_para_gs25_s4/full.wav`, retake "damara" with guidance 3 in Advanced).
- [ ] **Step 4:** Commit: `docs: Voice tab status`. Tell the user the by-ear check is theirs.
