# JeuVoice Plan 3 — Train Tab with Kaggle

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The user drops clips or pastes Hugging Face dataset links, picks a base model, clicks **Start on Kaggle**, watches the run, and tests each checkpoint by ear in the app. A recipe page shows how tr2 was made, and **Train like tr2** loads that recipe.

**Architecture:** Rust owns the UI, the Kaggle API client, the bundle and notebook builder, and Whisper for clip text. The Python engine (Plan 2) does audio cleaning, bandwidth checks, Hugging Face dataset reads and manifest writing. Audio tokens are made on Kaggle, not on the user's PC. Checkpoint tests use the Voice engine from Plan 2.

**Tech Stack:** Rust (workspace from Plans 1–2), `ureq` + `serde_json` for the Kaggle REST API, `keyring` for Kaggle credentials, `zip` for bundles, Python engine with `datasets`/`huggingface_hub` and `soundfile`.

**Spec:** `docs/design.md` — sections 5.3, 10 ("Data cleaning"), 11.2.

## Before you start

- Plan 2 must be merged into `main` (the user decides). Branch `plan3-train` from `main`.
- Read the Plan 2 engine API (`engine/jeuvoice_engine/server.py`, `jobs.py`) and `crates/jeuvoice-engine-client`.

## Global Constraints

- All Plan 2 Global Constraints still apply (repo, Rune pin, `scripts\cargo.cmd`, read-only tts_lab, no `.venv_ml`, plain words, keychain for secrets).
- Remote training: Kaggle only in this plan. RunPod shows a card that says "Coming later".
- Kaggle run settings (copy exactly from tr2, `tts_lab/kaggle/omnivoice-tr2.zip` `config/train.json`): `init_from_checkpoint "k2-fsa/OmniVoice"`, `use_lora true`, `lora_r 32`, `lora_alpha 64`, `lora_dropout 0.05`, `lora_modules_to_save ["audio_embeddings","audio_heads"]`, `learning_rate 1e-4`, `weight_decay 0.01`, `steps 6000`, `warmup_ratio 0.01`, `batch_tokens 2048`, `gradient_accumulation_steps 2`, `max_batch_size 32`, `mixed_precision "fp16"`, `eval_steps 250`, `save_steps 1000`, `logging_steps 50`, `keep_last_n_checkpoints -1`, seed 42. Other keys: copy the file from the zip as the template.
- Kaggle notebook: GPU **T4 ×2**, internet on, pip line `omnivoice[lora] @ git+https://github.com/k2-fsa/OmniVoice@08be0b4ccbac3e13e374e86fbfead4b4cac343e2`, `transformers==5.19.0`, `peft==0.21.2`, `accelerate==1.15.0`, and `pip uninstall -y torchao` first.
- Keep the two Kaggle workarounds from `make_omni_kaggle.py`: put tar shards in one file with another suffix (`.pack`), because Kaggle unpacks every `.tar` in an upload. And keep only `adapter_config.json` + `adapter_model.safetensors` of older checkpoints, because a full checkpoint is about 3 GB.
- The first real upload to Kaggle is outward-facing. The agent asks the user before it, every time in this plan.
- Data cleaning (spec 10): the logic of `tts_lab/scripts/clean_my_voice.py` (clipped clips out, burst peaks limited) plus a bandwidth check plus a Whisper confidence check. No new noise or music detector.

## Review Focus

1. **Large uploads:** a 40-minute dataset is about 230 MB of WAV. Upload must show progress, survive a slow link, and give a plain error with **Try again** when it fails. Test in Task 4.
2. **Kaggle API changes or bad credentials:** a 401/403 gives "Check your Kaggle user name and key in Settings." A 429 waits and tries again. Test in Task 4.
3. **Hugging Face dataset with other column names:** the user picks the audio and text columns. A missing text column means Whisper makes the text. Test in Task 3.
4. **Notebook failure on Kaggle:** status `error` shows the last 30 lines of the Kaggle log and keeps the bundle for a manual run. Test in Task 7.
5. **App closed during a run:** the run keeps going on Kaggle. On the next start, the Train tab finds the run again from the saved run file and continues to poll. Test in Task 7.

---

### Task 1: Training data model and the tr2 recipe

**Files:** Create `crates/jeuvoice-core/src/train.rs`, `crates/jeuvoice-core/tests/train.rs`, `assets/recipes/tr2.json`.

**Interfaces:**
- Produces:
  - `enum SourceKind { LocalFiles { paths: Vec<String> }, LocalFolder { dir: String }, HfDataset { repo: String, config: Option<String>, split: String, audio_col: String, text_col: Option<String> } }`
  - `struct Source { name: String, kind: SourceKind, weight: u32, license: Option<String>, note: Option<String> }` (weight = the `*N` repeat of `prep_omnivoice_data.py`)
  - `struct Clip { id: String, audio: String, text: String, duration_s: f32, source: String, excluded: Option<String> /* plain reason */ }`
  - `struct TrainSettings { steps: u32, lora_r: u32, lora_alpha: u32, batch_tokens: u32, save_steps: u32, learning_rate: f64, start_from: StartFrom }`, `enum StartFrom { Base, Adapter { repo_or_dir: String } }`
  - `struct Recipe { name: String, sources: Vec<Source>, settings: TrainSettings, base_model: String, notes: String }`
  - `fn minutes_by_source(clips: &[Clip], sources: &[Source]) -> Vec<(String, f32 /* raw min */, f32 /* min × weight */)>`
- `assets/recipes/tr2.json`: rebuild the tr2 mix from `tts_lab/data/omni2/train.jsonl` (read-only). Measured rows there: `alania_a8_tr2` 14505 ×1, `yodas_tr2` 9171 ×1, `antalia` 994 ×1, `my_voice_a8` 710 ×2, `my_voice_a8_clean` 309 ×2. For each source, find the public link and license from `tts_lab/data/<source>/README.md` and the prep scripts (`scripts/prep_yodas.py` uses `espnet/yodas2` + the clip list `Ertugrul1611/akil-tts-tr`. Alania: read `scripts/prep_alania_audio8.py`. Antalia has a local README with license cc-by-4.0 and maybe no public link). A source with no public link gets `LocalFolder` and a `note` that says "Add this folder from your own disk." The user's own voice sources are `LocalFolder` with the note "Your own recordings."
- [ ] **Step 1: Write failing tests:** recipe JSON round trip; `tr2.json` loads and has 5 sources with the weights above; `minutes_by_source` with weights.
- [ ] **Step 2–4:** Implement, run (`scripts\cargo.cmd test -p jeuvoice-core`), PASS.
- [ ] **Step 5:** Commit: `feat(core): training data model and the tr2 recipe`.

### Task 2: Engine data preparation jobs

**Files:** Create `engine/jeuvoice_engine/dataprep.py`, `engine/tests/test_dataprep.py`. Modify `server.py`.

**Interfaces:**
- Produces engine endpoints:
  - `POST /prep/clean {"clips": [{"id","audio"}], "out_dir": str}` → job. For each clip: clipped samples check, burst limit (logic and constants from `clean_my_voice.py`), measured bandwidth in kHz (logic from tts_lab's band measurement — find it via the memory note "Veri gerçek bandı" in `scripts/`. If not found, use the 99th-percentile spectral roll-off), write a cleaned 24 kHz mono WAV. Result per clip: `{"id","audio","duration_s","band_khz","excluded": null|"The clip is clipped." ...}`.
  - `POST /prep/hf {"repo","config","split","audio_col","text_col","limit": int|null, "out_dir"}` → job. Streams the dataset with `datasets` (token from the request, never logged), writes clips as WAV, returns clips with text (or empty text when `text_col` is null).
  - `POST /prep/manifest {"clips": [...], "sources": [{"name","weight"}], "dev_ratio": 0.02, "out_dir"}` → writes `train.jsonl` and `dev.jsonl` in the format of `prep_omnivoice_data.py` (`id`, `audio_path`, `text`, `language_id: "tr"`. Repeated rows get `-x2`, `-x3` ids. Shuffled with `random.Random(0)`).
- [ ] **Step 1: Write failing tests:** a synthetic clipped clip is excluded with the plain reason. A clean sine passes. Manifest repeats a weight-2 source twice with `-x2` ids and stays deterministic; `/prep/hf` on a tiny local `datasets` fixture (made in the test with `Dataset.from_dict` + audio column) maps the chosen columns.
- [ ] **Step 2–4:** Implement, run `pytest`, PASS.
- [ ] **Step 5:** Commit: `feat(engine): data cleaning, HF datasets and manifests`.

### Task 3: Train tab, step 1 — Data

**Files:** Create `crates/jeuvoice-app/src/train/{mod.rs,state.rs,data_step.rs,tests.rs}`. Modify `view.rs` (enable the Train tab).

- Drop zone for files or a folder (the same audio formats as the Transcribe tab). **Add Hugging Face dataset…** opens a small form: link or repo id, split, audio column, text column (optional), "Use at most N clips" (optional).
- Each source shows as a row: name, clip count, minutes, minutes × weight, a weight stepper (×1 … ×5), and **Remove**.
- Pipeline per source: local clips → Whisper text (SpeechService, `Samples` jobs) → `/prep/clean`. HF source → `/prep/hf` → Whisper only when text is empty → `/prep/clean`.
- **Review clips** opens a list with ▶ (SharedPlayer), the text (editable), the band, and the excluded reason. The user can exclude or include a clip.
- Chips: "212 clips · 41 min", "9 clips left out", "Band: 18 kHz".
- Data lives in a training project folder: `Documents/JeuVoice/Training/<name>/` with `train_project.json` (sources, clips, settings, run).
- [ ] **Step 1: Write failing headless tests** (fake engine + fake transcriber + temp dirs): drop 3 clips → 3 clips with text after the jobs. An HF form with no text column runs Whisper. Changing the weight changes "minutes × weight". Excluding a clip in Review updates the counts. Reopening the training project restores everything.
- [ ] **Step 2–4:** Implement, run, PASS. Snap and look.
- [ ] **Step 5:** Commit: `feat(app): Train tab data step`.

### Task 4: Kaggle client crate

**Files:** Create `crates/jeuvoice-kaggle/{Cargo.toml,src/lib.rs,src/auth.rs,src/datasets.rs,src/kernels.rs,tests/recorded.rs,tests/fixtures/*.json}`.

**Interfaces:**
- Produces:
  - `struct KaggleAuth { username: String, key: String }`, `fn load_auth() -> Option<KaggleAuth>` (keychain service `JeuVoice-Kaggle`), `fn save_auth(&KaggleAuth)`, `fn import_kaggle_json(path: &Path) -> Result<KaggleAuth>`.
  - `struct Kaggle { base: String /* https://www.kaggle.com/api/v1 */, auth: KaggleAuth }`
  - `fn upload_dataset(&self, owner_slug: &str, title: &str, files: &[PathBuf], private: bool, progress: &mut dyn FnMut(u64, u64)) -> Result<DatasetRef>` (creates the dataset, or a new version when it exists)
  - `fn push_kernel(&self, spec: &KernelSpec) -> Result<KernelRef>` where `KernelSpec { slug, title, notebook_json: String, dataset_sources: Vec<String>, enable_gpu: true, enable_internet: true, accelerator: "nvidiaTeslaT4" /* verify the exact value for T4 ×2 */, private: true }`
  - `fn kernel_status(&self, k: &KernelRef) -> Result<KernelStatus { state: Queued|Running|Complete|Error|Cancelled, message: Option<String> }>`
  - `fn kernel_log_tail(&self, k: &KernelRef, lines: usize) -> Result<String>` and `fn download_output(&self, k: &KernelRef, dest: &Path, progress) -> Result<Vec<PathBuf>>`
  - `enum KaggleError` with plain texts (bad credentials, rate limit, network, quota, "Kaggle refused the notebook: <reason>").
- Verify every endpoint, auth scheme (basic auth with the key, or a newer bearer token) and request body against the current official client source (`github.com/Kaggle/kaggle-api`, file `kaggle/api/kaggle_api_extended.py` and its generated client). Write down the verified endpoints in a doc comment at the top of `lib.rs` with the source commit.
- [ ] **Step 1: Write failing tests** against recorded responses with a local mock server (`mockito` dev-dependency): create dataset, new version, blob upload with progress, push kernel, status in each state, 401 → bad-credentials text, 429 → waits and tries again (inject a short delay in tests), output download.
- [ ] **Step 2–4:** Implement, run, PASS.
- [ ] **Step 5:** Commit: `feat(kaggle): Kaggle API client`.

### Task 5: Bundle and notebook builder

**Files:** Create `crates/jeuvoice-core/src/kaggle_bundle.rs` (pure: notebook JSON and configs) + `tests/kaggle_bundle.rs`; `crates/jeuvoice-app/src/train/bundle.rs` (file work on a worker).

**Interfaces:**
- Produces:
  - `fn train_config(settings: &TrainSettings, template: &serde_json::Value) -> serde_json::Value` (template = the tr2 `config/train.json`, shipped as `assets/kaggle/train_template.json`)
  - `fn data_config(manifest_dir: &str) -> serde_json::Value`
  - `fn notebook(name: &str, bundle_marker: &str) -> String` — cells: (1) nvidia-smi + pip lines from Global Constraints. (2) find the bundle by `bundle_<name>.txt`, copy to `/kaggle/working/run`, unpack `audio.pack`. (3) **extract audio tokens on Kaggle** with `python -m omnivoice.scripts.extract_audio_tokens` on `train.jsonl` and `dev.jsonl` (read its arguments from `third_party/omnivoice_src/omnivoice/scripts/extract_audio_tokens.py` in tts_lab, read-only), writing the token manifests the trainer reads. (4) train with `accelerate launch ... -m omnivoice.cli.train` and the prune loop from `make_omni_kaggle.py`. (5) copy adapters + `train.log` to `/kaggle/working/result`, delete the rest. (6) print a final line `JEUVOICE_RUN_DONE`.
  - `fn build_bundle(dir: &Path, name: &str, clips_dir: &Path, manifests: &Path, cfg: &Value, data: &Value) -> Result<PathBuf>` → `<name>.zip` with `bundle_<name>.txt`, `config/train.json`, `config/data.json`, `manifests/*.jsonl`, and `audio.pack` (a tar of the WAVs with a non-`.tar` suffix). Paths inside the manifests are relative to the run folder.
- [ ] **Step 1: Write failing tests:** `train_config` keeps every template key and changes only the settings fields. The notebook JSON is valid nbformat 4 and contains the pins, the `.pack` unpack, the token extraction, and `JEUVOICE_RUN_DONE`. A bundle of 3 tiny WAVs has the expected entries and no `.tar` name.
- [ ] **Step 2–4:** Implement, run, PASS. Compare the generated notebook text with `tts_lab/kaggle/omnivoice-tr2.ipynb` by eye and list the differences in the report (token extraction is the expected new part).
- [ ] **Step 5:** Commit: `feat: Kaggle bundle and notebook builder`.

### Task 6: Train tab, steps 2 and 3 — base model and where to train

**Files:** Create `crates/jeuvoice-app/src/train/{model_step.rs,target_step.rs}`.

- Step 2: "Start from" — **Base OmniVoice** or **A tr2 checkpoint** (list of adapters the app knows: presets' adapters, local folders, finished runs).
- Step 3: cards **This PC**, **Kaggle**, **RunPod**. This PC shows the GPU name and memory from the engine `/health`. It is off with "Your graphics card has 4 GB. Training needs at least 12 GB." when under 12 GB. Kaggle shows "T4 ×2 · free · about 6 hours for 6,000 steps" (scale the hours with steps: 1 h per 1,000 steps). RunPod shows "Coming later" and is off.
- Kaggle card without credentials shows **Add Kaggle key** → a form for user name + key, or **Import kaggle.json**.
- [ ] **Step 1: Write failing headless tests:** a 4 GB fake health turns This PC off with the message. Steps 3000 shows "about 3 hours". Saving credentials calls `save_auth` (fake store).
- [ ] **Step 2–4:** Implement, run, PASS.
- [ ] **Step 5:** Commit: `feat(app): Train tab model and target steps`.

### Task 7: Train tab, step 4 — script, run and progress

**Files:** Create `crates/jeuvoice-app/src/train/run_step.rs`, `crates/jeuvoice-app/src/train/run.rs`.

- Plain view of steps, LoRA r and alpha, batch tokens, save steps (editable). **View full script** shows the notebook text (read-only, monospace, scrollable).
- **Start on Kaggle:** a confirm dialog that names what goes up ("Upload 230 MB of audio to a private Kaggle dataset and start a notebook on your Kaggle account?"). Then: write manifests (`/prep/manifest`), build the bundle, `upload_dataset` with a progress bar, `push_kernel`, save `run.json` in the training project (`dataset`, `kernel`, `started_at`, `state`).
- Poll `kernel_status` every 60 s on a worker. Show state, elapsed time and the estimate. Live loss only if the log tail gives it (parse `train.log` lines in the tail when available). Otherwise show the loss curve after the run.
- On `Complete`: `download_output` into `runs/<name>/`, list `checkpoint-N` folders, draw the loss curve from `train.log` (a canvas polyline), show a checkpoint row each with **Test** (Task 8).
- On `Error`: show the last 30 log lines, keep the bundle, offer **Download bundle only** for a manual run.
- **Download bundle only** works at any time: it saves the zip and the `.ipynb` with a Save dialog.
- On app start: any training project with `run.json` in state Queued/Running resumes polling (Review Focus 5).
- [ ] **Step 1: Write failing headless tests** with a fake Kaggle (trait over the client): the confirm dialog appears before any upload. Upload progress shows. Status Running → Complete downloads outputs and lists 3 checkpoints. Error shows the log tail. A saved Running `run.json` resumes polling after a restart of the runtime.
- [ ] **Step 2–4:** Implement, run, PASS. Snap and look.
- [ ] **Step 5:** Commit: `feat(app): start, follow and collect Kaggle runs`.

### Task 8: Checkpoint test and "Use as preset model"

**Files:** Create `crates/jeuvoice-app/src/train/checkpoints.rs`.

- **Test** on a checkpoint renders the fixed text "Bugün size kısa bir hikâye anlatacağım. Bu hikâye bir yolculukla başlıyor." with the default preset's reference clip and settings, but with this checkpoint as `AdapterRef::Local`. It plays the result and keeps it, so the user can compare checkpoints one after another (A/B buttons on two rows).
- **Use as preset model** makes a new preset (copy of the chosen preset with this adapter) and opens it in the preset editor.
- [ ] **Step 1: Write failing tests** with the fake engine: Test sends `/load` with the checkpoint folder and a render with the fixed text. Use as preset model writes a preset whose adapter is the checkpoint folder.
- [ ] **Step 2–4:** Implement, run, PASS.
- [ ] **Step 5:** Commit: `feat(app): test checkpoints by ear and turn them into presets`.

### Task 9: Recipe page and "Train like tr2"

**Files:** Create `crates/jeuvoice-app/src/train/recipe_page.rs`.

- A page (button **How tr2 was made** in the Train tab) shows `assets/recipes/tr2.json`: each source with clips, minutes, weight, license and note, the base model, and the settings. **Copy recipe** puts the JSON on the clipboard.
- **Train like tr2** fills the data step with the recipe sources (HF sources start downloading through `/prep/hf`. Local sources ask the user to pick the folder) and fills the settings. Then the user adds their own clips.
- [ ] **Step 1: Write failing headless tests:** the page shows 5 sources with their weights and licenses. Train like tr2 creates 5 source rows and starts one `/prep/hf` job per HF source. A local source row says "Add this folder from your own disk."
- [ ] **Step 2–4:** Implement, run, PASS.
- [ ] **Step 5:** Commit: `feat(app): tr2 recipe page`.

### Task 10: Real check and README

- [ ] **Step 1:** Full tests: `scripts\cargo.cmd test --workspace`, engine `pytest`.
- [ ] **Step 2:** Ask the user for permission and for their Kaggle credentials to be set in Settings by them. Then run a **tiny** real run from the app or an ignored test: 20 clips of the user's own voice (`tts_lab/data/my_voice_a8_clean`, read-only copies), `steps 50`, `save_steps 25`. Check: upload, kernel start, token extraction, training, two checkpoints download, Test plays a render.
- [ ] **Step 3:** README "Train tab" section: credentials, what goes up to Kaggle, time and cost, how to resume, known limits, and a "Try it" list.
- [ ] **Step 4:** Commit: `docs: Train tab status`.
