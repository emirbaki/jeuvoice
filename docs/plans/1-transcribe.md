# JeuVoice Plan 1 — App Shell and Transcribe Tab

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A JeuVoice desktop app (Rune) where a teammate drops a batch of audio or video files, gets Whisper transcripts with word times, fixes words, exports SRT/VTT/TXT, and checks each take against its script.

**Architecture:** One Cargo workspace at `D:\MLProjelerim\jeuvoice`. `jeuvoice-core` holds pure logic (decode, peaks, transcript, subtitles, script diff, queue) with no UI and no model. `jeuvoice-whisper` wraps `whisper-rs` behind a `Transcriber` trait. `jeuvoice-app` is the Rune UI. It plays audio with `cpal` and runs Whisper on a worker thread. The Python engine is not in this plan.

**Tech Stack:** Rust (stable), Rune (git, pinned rev), whisper-rs (CUDA on Windows, Metal on macOS), symphonia, rubato, cpal, hound (tests), similar (diff), serde/serde_json, directories.

**Spec:** `D:\MLProjelerim\tts_lab\docs\superpowers\specs\2026-11-09-jeuvoice-design.md` (sections 4.1, 4.2, 5.2, 5.4, 6, 10).

## Plan series

The spec has four parts that each give working software. This is plan 1. The other plans come after this one ships.

| Plan | Delivers |
|---|---|
| **1 (this)** | Repo, app shell with tabs, Transcribe tab, Settings (Whisper part) |
| 2 | Mac spike (OmniVoice on MPS), Python engine, voice presets, Voice tab (generate, retake, fix word, export) |
| 3 | Train tab: data step, HF dataset links, recipe page, Kaggle run, checkpoint test |
| 4 | First-start engine setup (portable Python + uv), installers for Windows and macOS |

## Global Constraints

- Repo: `D:\MLProjelerim\jeuvoice`, a new git repo. Do not change anything in `D:\MLProjelerim\tts_lab` or `D:\MLProjelerim\.venv_ml`.
- Rune: `rune = { git = "https://github.com/anomalyco/rune", rev = "b3a33a0f3a5c35d78f188046a040696193cfb6bf" }` (tip of `dev` on 2026-11-09). Never use a branch name.
- Platforms: Windows with an NVIDIA GPU, and Apple Silicon macOS. GPU required.
- Whisper model: `large-v3-turbo` GGML, downloaded on first use to the app data folder. Default language: Turkish (`tr`). The user can change it.
- No system ffmpeg. Decode with symphonia.
- Audio playback: `cpal` on both platforms. Rune's `Player` plays audio only on macOS, so do not use it.
- Projects: `Documents/JeuVoice/<project name>/project.json` plus `audio/`.
- Settings and secrets: settings JSON in the OS config folder (`directories::ProjectDirs::from("com", "Jeu", "JeuVoice")`). Plan 1 stores no secrets.
- Every user-visible text and error is plain words that a teammate can act on (spec section 7).
- The build of `whisper-rs` compiles whisper.cpp with CMake and needs libclang. The auto-mode classifier blocked a C++ build in tts_lab before. If a build command is blocked, give the exact command to the user to run.

## Review Focus

1. **Paths with Turkish letters and spaces** (`C:\Users\Emir\Masaüstü\çekim 1.mp4`): decode, transcribe and export must work. Test in Task 2.
2. **A video with no audio track, or a codec symphonia cannot read:** that file fails with a clear message, and the rest of the batch goes on. Test in Task 7.
3. **Turkish case rules in the script check:** `İSTANBUL` = `istanbul`, `IŞIK` = `ışık`. Plain `to_lowercase` gives `i̇stanbul` and `işik`, which are wrong. Test in Task 5.
4. **Turkish number forms:** `2.000` (dot = thousands), `1,5` (comma = decimal), `%50`, `3.` (ordinal). Each must match its spoken words. Test in Task 5.
5. **Silence or music only:** the transcript is empty, the export writes an empty file, and nothing crashes. Test in Task 4.

---

### Task 1: Workspace and an empty app with three tabs

**Files:**
- Create: `Cargo.toml` (workspace), `.gitignore`, `rust-toolchain.toml` (stable)
- Create: `crates/jeuvoice-core/Cargo.toml`, `crates/jeuvoice-core/src/lib.rs`
- Create: `crates/jeuvoice-whisper/Cargo.toml`, `crates/jeuvoice-whisper/src/lib.rs`
- Create: `crates/jeuvoice-app/Cargo.toml`, `crates/jeuvoice-app/src/main.rs`, `crates/jeuvoice-app/src/state.rs`, `crates/jeuvoice-app/src/view.rs`
- Create: `docs/design.md` (copy of the spec), `docs/plans/1-transcribe.md` (copy of this plan)

**Interfaces:**
- Produces: `enum Tab { Voice, Transcribe, Train, Settings }`, `struct App { tab: Tab, .. }` in `state.rs`, `fn view(s: &App, cx: &mut Cx<App>) -> El` in `view.rs`. Each tab button has key `tab-voice`, `tab-transcribe`, `tab-train`, `tab-settings`. Voice and Train show the text "Coming in a later version".

- [ ] **Step 1:** `git init` in `D:\MLProjelerim\jeuvoice`. Make the workspace with the three crates and the Rune dependency from Global Constraints. Start from Rune's `examples/hello`.
- [ ] **Step 2: Write the failing test** in `crates/jeuvoice-app/src/view.rs`:

```rust
#[test]
fn transcribe_is_the_first_tab_and_tabs_switch() {
    let mut rt = Runtime::new(App::default(), view, (1100.0, 720.0), Env { system_dark: true, scale: 2.0 });
    rt.frame(0.0);
    assert_eq!(rt.state.tab, Tab::Transcribe);
    let (x, y) = rt.find("tab-settings").unwrap();
    rt.pointer_down(x, y); rt.pointer_up(x, y); rt.frame(1.0 / 60.0);
    assert_eq!(rt.state.tab, Tab::Settings);
}
```

- [ ] **Step 3:** Run `cargo test -p jeuvoice-app`. Expected: FAIL (no `Tab`).
- [ ] **Step 4:** Implement `Tab`, `App` (default tab `Transcribe`) and the tab bar. Window title "JeuVoice", size 1100×720.
- [ ] **Step 5:** Run `cargo test -p jeuvoice-app`. Expected: PASS. Run `cargo run -p jeuvoice-app -- --snap shell.png` and look at the PNG.
- [ ] **Step 6:** Commit: `feat: workspace and app shell with tabs`.

### Task 2: Decode any audio or video file

**Files:**
- Create: `crates/jeuvoice-core/src/audio.rs`
- Test: `crates/jeuvoice-core/tests/audio.rs`, fixtures made by the test with `hound`

**Interfaces:**
- Produces:
  - `struct Audio { pub samples: Vec<f32>, pub rate: u32 }` (mono)
  - `fn decode(path: &Path) -> Result<Audio, AudioError>` (native rate, channels mixed to mono)
  - `fn resample(a: &Audio, rate: u32) -> Audio`
  - `fn to_whisper(a: &Audio) -> Vec<f32>` (16 kHz mono)
  - `enum AudioError { NotFound(PathBuf), NoAudioTrack(PathBuf), Unsupported { path: PathBuf, detail: String } }` with `Display` texts: "Cannot find the file {path}.", "The file {name} has no sound track.", "JeuVoice cannot read the sound in {name} ({detail})."

- [ ] **Step 1: Write the failing tests:**
  - `decodes_stereo_wav_to_mono`: write a 48 kHz stereo WAV, left = 0.5, right = −0.5 for 1 s. Assert `rate == 48000`, `samples.len() == 48000`, every sample within 1e-3 of 0.0.
  - `resamples_to_16k`: 1 s of 440 Hz at 48 kHz → `to_whisper` gives 16000 ± 16 samples.
  - `reads_paths_with_turkish_letters`: write the WAV to `<tmp>/Masaüstü/çekim 1.wav`. Decode works.
  - `missing_file_is_not_found`: assert `AudioError::NotFound`.
- [ ] **Step 2:** Run `cargo test -p jeuvoice-core --test audio`. Expected: FAIL.
- [ ] **Step 3:** Implement with symphonia (features `mp3`, `aac`, `isomp4`, `ogg`, `vorbis`, `flac`, `wav`, `pcm`) and rubato. Choose the first audio track. No audio track gives `NoAudioTrack`.
- [ ] **Step 4:** Run the tests. Expected: PASS. Also decode by hand one real `.mp4` and one `.ogg` from `D:\MLProjelerim\tts_lab\data` (read only).
- [ ] **Step 5:** Commit: `feat(core): decode audio and video files to mono`.

### Task 3: Waveform peaks

**Files:**
- Create: `crates/jeuvoice-core/src/peaks.rs`

**Interfaces:**
- Produces: `struct Peaks { pub per_second: u32, pub min: Vec<f32>, pub max: Vec<f32> }`, `fn peaks(a: &Audio, per_second: u32) -> Peaks`, `fn Peaks::range(&self, start_s: f64, end_s: f64, buckets: usize) -> Vec<(f32, f32)>` (folds stored buckets into `buckets` columns for the visible range).

- [ ] **Step 1: Write the failing tests:**
  - `peaks_of_a_square_wave`: 1 s at 8 kHz alternating ±0.8, `per_second = 100` → 100 buckets, each `(−0.8, 0.8)`.
  - `range_folds_buckets`: `range(0.0, 1.0, 10)` → 10 columns, each `(−0.8, 0.8)`.
  - `range_past_the_end_is_flat`: `range(0.9, 1.5, 6)` → the last columns are `(0.0, 0.0)`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Store 200 peaks per second (the app uses `per_second = 200`).
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(core): waveform peaks`.

### Task 4: Transcript model and subtitle export

**Files:**
- Create: `crates/jeuvoice-core/src/transcript.rs`, `crates/jeuvoice-core/src/subtitles.rs`

**Interfaces:**
- Produces:
  - `#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)] struct Word { pub text: String, pub start: f64, pub end: f64, pub prob: f32 }`
  - `struct Transcript { pub language: String, pub words: Vec<Word> }`, `fn Transcript::text(&self) -> String`, `fn Transcript::set_word(&mut self, i: usize, text: &str)` (keeps the times)
  - `fn to_srt(t: &Transcript) -> String`, `fn to_vtt(t: &Transcript) -> String`, `fn to_txt(t: &Transcript) -> String`
- Cue rules: a cue holds at most 2 lines of at most 42 characters and lasts at most 6.0 s. A new cue starts after a word that ends with `.`, `?` or `!`, or after a gap of 0.7 s or more.

- [ ] **Step 1: Write the failing tests:**
  - `srt_of_two_sentences`: words "Merhaba." (0.0–0.6), "Bugün" (1.5–1.9), "oyun" (1.9–2.2), "konuşuyoruz." (2.2–3.0). Assert exact output:

```
1
00:00:00,000 --> 00:00:00,600
Merhaba.

2
00:00:01,500 --> 00:00:03,000
Bugün oyun konuşuyoruz.

```

  - `vtt_has_header_and_dot_millis`: starts with `WEBVTT\n\n`, times use `.` (`00:00:01.500`).
  - `long_text_wraps_at_42_and_splits_cues`: 30 words of 6 letters, 0.3 s each → no line over 42 chars, no cue over 2 lines or 6.0 s.
  - `empty_transcript_gives_empty_srt`: `to_srt` of no words is `""` (Review Focus 5).
  - `set_word_keeps_times`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(core): transcript and SRT/VTT/TXT export`.

### Task 5: Script check (Turkish normalize and word diff)

**Files:**
- Create: `crates/jeuvoice-core/src/turkish.rs`, `crates/jeuvoice-core/src/script_diff.rs`

**Interfaces:**
- Produces:
  - `fn tr_lower(s: &str) -> String` (`I`→`ı`, `İ`→`i`, then normal lowercase)
  - `fn number_to_words(n: u64) -> String` (Turkish)
  - `fn normalize(text: &str) -> Vec<String>`: lowercase with `tr_lower`, remove punctuation, expand numbers to words (one token for each word).
  - `enum DiffOp { Same { script: usize, heard: usize }, Missing { script: usize }, Added { heard: usize }, Changed { script: usize, heard: usize } }` — indices point to words of the original script and to `Transcript::words`.
  - `fn script_diff(script: &str, t: &Transcript) -> Vec<DiffOp>`
- Number rules:
  - "bir" is dropped before "yüz" and "bin" (100 = "yüz", 1000 = "bin", 2023 = "iki bin yirmi üç"), but stays before "milyon" (1,000,000 = "bir milyon").
  - `2.000` = 2000. `1,5` = "bir virgül beş". `%50` = "yüzde elli". `3.` at the end of a number token = "üçüncü" (ordinals: birinci, ikinci, üçüncü, dördüncü, beşinci, altıncı, yedinci, sekizinci, dokuzuncu, onuncu; for others add the suffix by vowel harmony of the last word: -ıncı/-inci/-uncu/-üncü).
- A number in the script that becomes several words maps all its normalized tokens back to the one original script word. A `Changed` op is a 1-to-1 swap next to each other in the diff.

- [ ] **Step 1: Write the failing tests:**
  - `turkish_lowercase`: `tr_lower("İSTANBUL IŞIK") == "istanbul ışık"` (Review Focus 3).
  - `numbers`: 0 "sıfır", 21 "yirmi bir", 100 "yüz", 1000 "bin", 2023 "iki bin yirmi üç", 1_000_000 "bir milyon", 1_250_000 "bir milyon iki yüz elli bin".
  - `number_forms` (Review Focus 4): `normalize("2.000 kişi, %50 indirim, 1,5 saat, 3. bölüm")` == `["iki","bin","kişi","yüzde","elli","indirim","bir","virgül","beş","saat","üçüncü","bölüm"]`.
  - `digits_match_words`: script "2023 yılında çıktı." against heard words "iki bin yirmi üç yılında çıktı" → all `Same`, no `Missing` or `Added`.
  - `skipped_and_added_words`: script "Joel ve Ellie tünele giriyor" against heard "Joel Ellie hemen tünele giriyor" → `Missing{script:1}` ("ve") and `Added{heard:2}` ("hemen").
  - `changed_word`: script "damara" against heard "damaraa" → `Changed{script:0, heard:0}`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Use the `similar` crate for the word diff on normalized tokens, then map back to original indices.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(core): Turkish script check`.

### Task 6: Whisper wrapper and model download

**Files:**
- Create: `crates/jeuvoice-whisper/src/lib.rs`, `crates/jeuvoice-whisper/src/model.rs`, `crates/jeuvoice-whisper/src/words.rs`
- Test: `crates/jeuvoice-whisper/tests/words.rs`, `crates/jeuvoice-whisper/tests/real_model.rs` (`#[ignore]`)

**Interfaces:**
- Consumes: `Word`, `Transcript` (Task 4).
- Produces:
  - `struct TranscribeOptions { pub language: String, pub use_gpu: bool }`
  - `trait Transcriber: Send + Sync { fn transcribe(&self, samples_16k: &[f32], opts: &TranscribeOptions, progress: &mut dyn FnMut(f32)) -> Result<Transcript, WhisperError>; }`
  - `struct WhisperRs` with `fn WhisperRs::load(model: &Path, use_gpu: bool) -> Result<Self, WhisperError>`. If GPU load fails, it loads on the CPU and sets `pub on_gpu: bool` to false.
  - `struct FakeTranscriber(pub Transcript)` (returns the fixed transcript, for app tests)
  - `fn model_path() -> PathBuf` (`<data dir>/models/ggml-large-v3-turbo.bin`), `fn download_model(progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf, WhisperError>` from `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin`. Download to a `.part` file, then rename.
  - `fn tokens_to_words(tokens: &[(String, f64, f64, f32)]) -> Vec<Word>`: a token that starts with a space starts a new word. Skip special tokens (`[_BEG_]`, `[_TT_…]`, text that starts with `[_`).
- Pin the newest `whisper-rs` release on crates.io. Features: `cuda` on Windows, `metal` on macOS (target-specific dependency tables). Turn on token timestamps. Turn on DTW with the large-v3-turbo preset if the pinned version has it.

- [ ] **Step 1: Write the failing tests** in `tests/words.rs`:
  - `tokens_join_into_words`: tokens `[(" Mer",0.0,0.2,0.9), ("haba",0.2,0.5,0.8), (" dünya",0.6,1.0,0.95)]` → words "Merhaba" (0.0–0.5, prob 0.8 = min) and "dünya" (0.6–1.0).
  - `special_tokens_are_skipped`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement `words.rs`, `model.rs` and `WhisperRs`.
- [ ] **Step 4:** Run `cargo test -p jeuvoice-whisper`. Expected: PASS.
- [ ] **Step 5:** `real_model.rs` (ignored test): transcribe `D:\MLProjelerim\tts_lab\data\ref\sesim\` clip `ses_001_0007` (find the file). Assert at least 3 words, times increase, language `tr`. Run `cargo test -p jeuvoice-whisper -- --ignored` with the model present. Expected: PASS. Write in the commit message whether it ran on the GPU.
- [ ] **Step 6:** Commit: `feat(whisper): whisper-rs transcriber with word times`.

### Task 7: Transcription queue

**Files:**
- Create: `crates/jeuvoice-core/src/queue.rs`

**Interfaces:**
- Consumes: `decode`, `to_whisper` (Task 2), `Transcriber` (Task 6; the core crate gets the trait through a generic closure so it does not depend on whisper: `type Job = Box<dyn FnOnce(&mut dyn FnMut(f32)) -> Result<Transcript, String> + Send>`).
- Produces:
  - `enum ItemState { Waiting, Running(f32), Done(Transcript), Failed(String) }`
  - `struct Queue { pub items: Vec<(PathBuf, ItemState)> }`, `fn add(&mut self, paths: &[PathBuf])` (skips a path already in the queue), `fn next_waiting(&self) -> Option<usize>`, `fn apply(&mut self, i: usize, update: Update)`
  - `enum Update { Progress(f32), Done(Transcript), Failed(String) }`
  - `fn spawn_worker(jobs: Receiver<(usize, Job)>, updates: Sender<(usize, Update)>, wake: impl Fn() + Send + 'static) -> JoinHandle<()>` (one item at a time, because of GPU memory)

- [ ] **Step 1: Write the failing tests:**
  - `items_run_in_order_and_a_failure_does_not_stop_the_batch` (Review Focus 2): three jobs, the second returns `Err("The file b.mp4 has no sound track.")`. Assert the final states are Done, Failed (with that text), Done.
  - `adding_the_same_file_twice_keeps_one`.
  - `progress_updates_reach_the_item`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement with `std::thread` and `std::sync::mpsc`.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(core): transcription queue`.

### Task 8: Audio playback

**Files:**
- Create: `crates/jeuvoice-app/src/playback.rs`

**Interfaces:**
- Consumes: `Audio` (Task 2).
- Produces: `struct Playback` with `fn new() -> Result<Self, String>`, `fn load(&mut self, a: Arc<Audio>)`, `fn play(&mut self)`, `fn pause(&mut self)`, `fn seek(&mut self, t: f64)`, `fn position(&self) -> f64`, `fn is_playing(&self) -> bool`. The position is an `Arc<AtomicU64>` sample counter that the cpal callback moves. `struct Cursor` holds the pure position logic (seek clamp, advance, end stop) so tests do not need a sound device.

- [ ] **Step 1: Write the failing tests** on `Cursor`: `seek_clamps_to_length`, `advance_stops_at_end_and_pauses`, `position_in_seconds_uses_the_source_rate` (a 44.1 kHz source on a 48 kHz device plays at the right speed: resample on load to the device rate).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement `Cursor` and `Playback` with cpal (default output device). No device gives the error "No sound output found. Connect speakers or headphones."
- [ ] **Step 4:** Run. Expected: PASS. Play a file by hand.
- [ ] **Step 5:** Commit: `feat(app): audio playback with cpal`.

### Task 9: Waveform view with words

**Files:**
- Create: `crates/jeuvoice-app/src/waveform.rs`

**Interfaces:**
- Consumes: `Peaks::range` (Task 3), `Word` (Task 4).
- Produces: `struct WaveView { pub start: f64, pub span: f64, pub selection: Option<(f64, f64)>, pub playhead: f64 }` and `fn waveform<S: 'static>(v: &WaveView, peaks: &Peaks, words: &[Word], marks: &[(usize, Mark)], lens: impl Fn(&mut S) -> &mut WaveView + Clone + 'static, on_word: impl Fn(&mut S, usize) + Clone + 'static) -> El`. `enum Mark { Missing, Added, Changed, LowProb }` colors words: red for script words the speaker missed or changed, green for added words, orange underline for `prob < 0.5`.
- Behavior: the wave is a `canvas` of vertical min/max strokes (one per column). Drag on the wave sets `selection`. A click on the wave sets `playhead`. A click on a word calls `on_word`. The mouse wheel zooms around the pointer (`span` between 2 s and the full length). Element keys: `wave`, `word-{i}`.

- [ ] **Step 1: Write the failing headless tests:**
  - `drag_selects_a_time_range`: 10 s of audio, `span = 10`, wave 1000 points wide → drag from x=100 to x=300 gives selection ≈ (1.0, 3.0) ± 0.05.
  - `click_on_word_calls_handler`: tap `word-2`, assert the handler got 2.
  - `wheel_zooms_around_pointer`: the time under the pointer stays the same within 0.05 s.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Use `tx.live_only()` for drag and wheel updates where the README allows it.
- [ ] **Step 4:** Run. Expected: PASS. Run `--snap wave.png` on a fixture scene and look at the PNG.
- [ ] **Step 5:** Commit: `feat(app): waveform view with words`.

### Task 10: Transcribe tab

**Files:**
- Create: `crates/jeuvoice-app/src/transcribe_tab.rs`
- Modify: `crates/jeuvoice-app/src/state.rs`, `crates/jeuvoice-app/src/view.rs`, `crates/jeuvoice-app/src/main.rs`

**Interfaces:**
- Consumes: `Queue`, `spawn_worker` (Task 7), `Transcriber`, `FakeTranscriber` (Task 6), `waveform` (Task 9), `Playback` (Task 8), `to_srt`/`to_vtt`/`to_txt` (Task 4), `script_diff` (Task 5).
- Produces: `struct TranscribeTab { queue: Queue, selected: Option<usize>, view: WaveView, script: String, diff: Vec<DiffOp>, editing: Option<usize> }` in `App`. `App` takes the transcriber as `Arc<dyn Transcriber>` so tests inject `FakeTranscriber`.
- Layout (spec 5.2, layout B):
  - Left column: a drop zone (key `drop`, accepts `wav mp3 m4a aac ogg opus flac mp4 mov mkv webm`) and the file list with a status for each file (waiting, progress %, done, error text).
  - Main area: the waveform of the selected file with its words.
  - Bottom bar: play/pause (key `play`) and the time.
  - Right panel: "Script check" with a text field (key `script`) and an "Open script…" button (`.txt`, `.md`). Picking the script of a Voice project comes in Plan 2, when projects exist. It shows counts: "3 missing, 1 added, 2 changed". A click on a difference jumps the playhead there.
  - Export buttons: `export-srt`, `export-vtt`, `export-txt`. Each opens a Save dialog, with the default name `<file name>.srt` next to the source file.
  - Double-click on a word opens a text field to fix it (`Transcript::set_word`).
- Transcripts are kept in `Documents/JeuVoice/Transcripts/<file name>.json`, so a file dropped again opens at once. **[plan call]** (A full project format comes in Plan 2.)

- [ ] **Step 1: Write the failing headless tests** (with `FakeTranscriber` and fixture WAVs):
  - `dropping_files_transcribes_them`: drop two WAVs on `drop`, advance frames until both are done (the worker wakes the runtime), assert both are `Done`.
  - `script_check_marks_missing_word`: paste "Joel ve Ellie" into `script`, the fake transcript is "Joel Ellie" → `word` marks show one Missing, and the counts text says "1 missing".
  - `export_srt_writes_the_file`: tap `export-srt`, answer the dialog with `Runtime::resolve_file_dialog(Some(tmp/"a.srt"))`, assert the file has the SRT text.
  - `fixing_a_word_changes_the_export`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Wire the real `WhisperRs` in `main.rs`. Load the model on first use (Settings shows the download).
- [ ] **Step 4:** Run. Expected: PASS. Snap the tab to PNG and look at it.
- [ ] **Step 5:** Commit: `feat(app): Transcribe tab`.

### Task 11: Settings (Whisper part)

**Files:**
- Create: `crates/jeuvoice-app/src/settings.rs`, `crates/jeuvoice-app/src/settings_tab.rs`

**Interfaces:**
- Produces: `#[derive(Serialize, Deserialize)] struct Settings { pub whisper_language: String, pub use_gpu: bool }` (defaults `"tr"`, `true`), `fn Settings::load() -> Settings` (defaults when the file is missing or bad), `fn Settings::save(&self) -> io::Result<()>`.
- Settings tab: language picker (`tr`, `en`, `auto`), "Use GPU" switch, Whisper model status: "Not downloaded" with a **Download** button (key `download-model`) and a progress bar, or "Ready (on GPU)" / "Ready (on CPU)". Spec 5.4 lists more settings. They come in later plans.

- [ ] **Step 1: Write the failing tests:** `settings_round_trip`, `bad_settings_file_gives_defaults`, `language_change_is_saved` (headless: pick `en`, reload, assert `en`).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. The download runs on a worker and updates the progress bar.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(app): settings for Whisper`.

### Task 12: Release build and a real check

**Files:**
- Create: `README.md` (how to build and run, in plain words)

- [ ] **Step 1:** Run `cargo build --release -p jeuvoice-app`. If the auto-mode classifier blocks it, give the command to the user.
- [ ] **Step 2:** Run all tests: `cargo test --workspace`. Expected: PASS.
- [ ] **Step 3:** Start the app. Download the model. Drop three real takes (one `.mp4`, one `.ogg`, one in a folder with Turkish letters). Paste the script of one take. Export SRT. Open the SRT next to the video in a player and check the timing by eye.
- [ ] **Step 4:** Write the result in the README "Status" section: what works, the transcription speed (minutes of audio for each minute), GPU or CPU.
- [ ] **Step 5:** Commit: `docs: build and status`.
