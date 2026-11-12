# Voices Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Voices tab to clone or describe a voice, listen to tries, save it as a preset, and delete it. The built-in `ses_0007` becomes Sami.

**Architecture:** The core preset module gets the Sami name, a migration and a delete. A small `voices` module in the app maps attribute chips to the OmniVoice `instruct` string, owns the tab state and runs `PresetEditor` on a worker. The view uses the Frost style. The Voice tab and the Voices tab share one "engine busy" rule.

**Tech Stack:** Rust 2024, rune (pinned), `scripts\cargo.cmd` for every build and test.

**Spec:** `docs/superpowers/specs/2026-11-12-voices-tab-design.md`

## Global Constraints

- Built-in voice name: `Sami`. Old name: `ses_0007`. Sami cannot be deleted.
- New voices store the `model` of the Sami preset on disk (base `k2-fsa/OmniVoice` + tr2 adapter). Never `AdapterRef::None` for a new voice.
- Instruct attributes (English, comma + space, this order): gender `male`/`female`, age `child`/`teenager`/`young adult`/`middle-aged`/`elderly`, pitch `very low pitch`/`low pitch`/`moderate pitch`/`high pitch`/`very high pitch`, style `whisper`. No accents.
- Clip length 3–10 s. Test sentence: `TRY_SENTENCE` in `voice/preset_editor.rs`.
- Defaults: guidance 2.5, speed 1.0, base seed 1. "Try 3 voices" uses base, +1, +2. "Try 3 more" adds the next three.
- Delete text: "Delete the voice {name}? This cannot be undone." / "{name} sesi silinsin mi? Bu geri alınamaz."
- Keys: `tab-voices`, `page-voices`, every new key starts with `voices-`. At most 4 `glass-` elements on a screen.
- Every visible text uses `crate::i18n::tr!` / `trf!` with English and Turkish. Frost style only (`crate::style`). Engine and Whisper calls on workers.
- Card light: the top of the glass border gradient becomes `white(0.22)` (was `white(0.16)`).

## Review Focus

1. **A name with characters Windows refuses** (`a/b`, `x:y`, `CON`, a trailing dot): save shows a plain error and writes nothing. Test in Task 3.
2. **A name that differs only in case from a saved voice** (`sami`): save says the name is taken. Test in Task 3.
3. **Both `ses_0007` and `Sami` folders exist** (a user made a voice called Sami earlier): the migration changes nothing and loses no file. Test in Task 1.
4. **A project that uses a deleted voice:** it opens with Sami and a notice, not an error. Test in Task 4.
5. **A tab switch while tries render:** the tries go on, the results show on return, and Generate in the Voice tab stays off until they end. Test in Task 4.

---

### Task 1: Sami in the core presets

**Files:**
- Modify: `crates/jeuvoice-core/src/preset.rs`, rename folder `assets/presets/ses_0007` to `assets/presets/Sami` (its `preset.json` `name` becomes `Sami`), `crates/jeuvoice-app/src/main.rs:93-96`, `crates/jeuvoice-app/src/voice/state.rs:148-152`, `crates/jeuvoice-app/src/train/checkpoints.rs:23-26`
- Test: `crates/jeuvoice-core/tests/preset.rs`

**Interfaces:**
- Produces (in `jeuvoice_core::preset`):
  - `pub const BUILTIN_PRESET: &str = "Sami";` and `pub const LEGACY_BUILTIN_PRESET: &str = "ses_0007";`
  - `pub fn canonical_preset_name(name: &str) -> &str` — `ses_0007` gives `Sami`, any other name gives itself.
  - `pub fn migrate_legacy_builtin(presets_dir: &Path) -> std::io::Result<bool>` — renames the folder and the `name` field when `ses_0007/` exists and `Sami/` does not. Returns true when it renamed.
  - `pub fn delete_preset(presets_dir: &Path, name: &str) -> Result<(), PresetDeleteError>` with `pub enum PresetDeleteError { Builtin, Missing, Io(std::io::Error) }`.
  - `default_preset().name == "Sami"`.

- [ ] **Step 1: Write the failing tests** in `crates/jeuvoice-core/tests/preset.rs`:
  - `the_builtin_voice_is_sami`: `default_preset().name == "Sami"`, `canonical_preset_name("ses_0007") == "Sami"`, `canonical_preset_name("Ece") == "Ece"`.
  - `an_old_install_moves_to_sami`: save `default_preset()` renamed `ses_0007` with a `ref.wav` into a temp dir. `migrate_legacy_builtin` returns true. `list_presets` gives one preset named `Sami`, and `Sami/ref.wav` exists.
  - `both_folders_stay_when_sami_exists` (Review Focus 3): make `ses_0007/` and `Sami/` with different seeds. `migrate_legacy_builtin` returns false, and both `preset.json` files are byte-for-byte the same as before.
  - `sami_cannot_be_deleted`: `delete_preset(dir, "Sami")` is `Err(PresetDeleteError::Builtin)` and the folder still exists.
  - `delete_removes_the_folder`: save a preset `Ece`. `delete_preset(dir, "Ece")` is `Ok`, the folder is gone, and a second call gives `Missing`.
- [ ] **Step 2:** Run `scripts\cargo.cmd test -p jeuvoice-core --test preset` — Expected: FAIL (items missing).
- [ ] **Step 3:** Implement. Rename the asset folder with `git mv`. In `main.rs` call `migrate_legacy_builtin` before `install_default_presets`. Replace the `"ses_0007"` lookups in `voice/state.rs` and `train/checkpoints.rs` with `BUILTIN_PRESET`. Where a project or a setting stores a preset name, compare through `canonical_preset_name`.
- [ ] **Step 4:** Run `scripts\cargo.cmd test --workspace` — Expected: PASS. Fix tests that name `ses_0007` only when they test the built-in voice. The engine-client test at `api.rs:615` uses the name as plain data: leave it.
- [ ] **Step 5: Commit** `feat(core): the built-in voice is Sami`.

### Task 2: Attributes to the instruct string

**Files:**
- Create: `crates/jeuvoice-app/src/voices/mod.rs` (declares `pub mod traits;` for now), `crates/jeuvoice-app/src/voices/traits.rs`
- Modify: `crates/jeuvoice-app/src/main.rs` (`mod voices;`)

**Interfaces:**
- Produces:
  - `pub enum Gender { Male, Female }`, `pub enum Age { Child, Teenager, YoungAdult, MiddleAged, Elderly }`, `pub enum Pitch { VeryLow, Low, Moderate, High, VeryHigh }` (all `Clone, Copy, Debug, PartialEq, Eq`, with `pub const ALL: [Self; N]` and `pub fn label(self) -> &'static str` through `tr!`).
  - `#[derive(Clone, Debug, Default, PartialEq)] pub struct VoiceTraits { pub gender: Option<Gender>, pub age: Option<Age>, pub pitch: Option<Pitch>, pub whisper: bool }`
  - `impl VoiceTraits { pub fn instruct(&self) -> Option<String> }`

- [ ] **Step 1: Write the failing tests** in `traits.rs`:
  - `all_traits_make_the_omnivoice_string`: female + young adult + high + whisper gives `Some("female, young adult, high pitch, whisper")`.
  - `no_traits_make_no_instruct`: `VoiceTraits::default().instruct() == None`.
  - `one_trait_alone`: only `Pitch::VeryLow` gives `Some("very low pitch")`.
  - `every_value_is_an_omnivoice_word`: every gender, age and pitch value alone gives a string from the Global Constraints list.
- [ ] **Step 2:** Run `scripts\cargo.cmd test -p jeuvoice-app --bin jeuvoice-app voices::traits` — Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same command — Expected: PASS.
- [ ] **Step 5: Commit** `feat(app): voice attributes for OmniVoice voice design`.

### Task 3: Preset editor for the tab

**Files:**
- Modify: `crates/jeuvoice-app/src/voice/preset_editor.rs`, `crates/jeuvoice-app/src/voice/state.rs` (make `resolved_adapter` and `wait_job` `pub(crate)`)

**Interfaces:**
- Consumes: `VoiceTraits` (Task 2), `BUILTIN_PRESET`, `canonical_preset_name`, `list_presets` (Task 1).
- Produces (on `PresetEditor`):
  - fields `pub traits: VoiceTraits` (replaces the `instruct: String` field), `pub guidance: f32` (2.5), `pub speed: f32` (1.0).
  - `pub fn try_three(&mut self, out_dir: &Path)` keeps its name and uses `traits.instruct()`, guidance and speed. It replaces `tries`.
  - `pub fn try_more(&mut self, out_dir: &Path)` — adds three tries with the next three seeds, keeps the old tries.
  - `pub fn save(&mut self, presets_dir: &Path, model: ModelRef) -> Result<Preset, String>` — stores `model`, `guidance`, `speed`, the picked seed and `traits.instruct()`.
  - `pub fn name_problem(name: &str, presets_dir: &Path) -> Option<String>` — the plain error for an empty name, a taken name (case-insensitive, `canonical_preset_name` applied), or a name Windows refuses (`< > : " / \ | ? *`, control characters, a trailing dot or space, `CON PRN AUX NUL COM1-9 LPT1-9` in any case).
  - `pub(crate) fn load_model(api: &Arc<dyn EngineApi>, model: &ModelRef, settings: &Settings, store: &dyn TokenStore) -> Result<(), String>` — resolve the adapter, `load`, wait (reuse `resolved_adapter` and `wait_job` from `voice/state.rs`).

- [ ] **Step 1: Write the failing tests** in the `preset_editor.rs` test module (it has a `Spy` engine):
  - `save_stores_the_given_model_and_settings`: set guidance 3.0, speed 1.1, pick try 1 of `try_three`. `save(dir, sami_model)` writes `model == sami_model`, guidance 3.0, speed 1.1, seed `base_seed + 1`.
  - `described_voice_saves_its_instruct`: traits female + elderly, kind `Seed`. The saved `instruct` is `Some("female, elderly")`.
  - `try_more_adds_the_next_seeds`: `try_three` then `try_more` gives 6 tries with seeds 1..=6, and the spy saw 6 renders.
  - `names_windows_refuses_are_errors` (Review Focus 1): `name_problem` is `Some` for `"a/b"`, `"x:y"`, `"CON"`, `"com3"`, `"Ece."`, `""`, `"   "`. Save with `"a/b"` returns `Err` and the presets folder stays empty.
  - `a_taken_name_is_an_error_in_any_case` (Review Focus 2): with `Sami` saved, `name_problem("sami", dir)` and `name_problem("ses_0007", dir)` are `Some`.
  - `load_model_loads_base_and_adapter`: with a spy, `load_model` for a `Local { dir }` adapter calls `load("k2-fsa/OmniVoice", Some(dir), "auto")`.
- [ ] **Step 2:** Run `scripts\cargo.cmd test -p jeuvoice-app --bin jeuvoice-app preset_editor` — Expected: FAIL.
- [ ] **Step 3:** Implement. `save` calls `name_problem` first.
- [ ] **Step 4:** Run `scripts\cargo.cmd test -p jeuvoice-app` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(app): preset editor saves the model, settings and safe names`.

### Task 4: Voices tab state and the shared engine rule

**Files:**
- Create: `crates/jeuvoice-app/src/voices/state.rs`, `crates/jeuvoice-app/src/voices/tests.rs`
- Modify: `crates/jeuvoice-app/src/voices/mod.rs`, `crates/jeuvoice-app/src/state.rs` (`Tab::Voices`, `App::voices`, poll), `crates/jeuvoice-app/src/voice/state.rs` (Generate off while Voices works, deleted-voice fallback)

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `Tab::Voices` (between `Voice` and `Transcribe`).
  - `pub enum VoicesMode { Clone, Describe }`, `pub enum VoicesStatus { Idle, CheckingClip, LoadingModel, Rendering, Failed(String) }`.
  - `pub struct VoicesTab { pub mode: VoicesMode, pub editor: Option<PresetEditor>, pub selected: Option<String>, pub confirm_delete: Option<String>, pub status: VoicesStatus, pub notice: Option<String>, pub name_field: TextEdit, pub clip_text_field: TextEdit, … }`
  - Methods: `pub fn is_busy(&self) -> bool`, `pub fn drop_clip(&mut self, path: PathBuf, …)`, `pub fn try_voices(&mut self, more: bool, …)`, `pub fn listen(&mut self, name: &str, …)`, `pub fn save(&mut self, …) -> Option<String>` (the saved name), `pub fn ask_delete(&mut self, name: &str)`, `pub fn confirm_delete(&mut self, …)`, `pub fn poll(&mut self) -> bool`. Each engine method takes the shared engine, settings and token store the way `VoiceTab::generate` does, and runs on a worker.
  - `VoiceTab::refresh_presets` runs after a Voices save or delete, and `VoiceTab::preset_name` becomes the saved voice.

- [ ] **Step 1: Write the failing tests** in `voices/tests.rs` with `FakeEngineApi` (same rig style as `voice/tests.rs`):
  - `describe_try_save_shows_in_the_voice_tab`: pick female + young adult, `try_voices(false)`, wait for 3 done tries, pick one, name `Ece`, save. `app.voice.presets` holds `Ece`, `app.voice.preset_name == "Ece"`, and `Ece/preset.json` has the Sami model.
  - `tries_load_the_model_first`: the spy engine sees `load` before the first `render`.
  - `listen_loads_the_voice_model_and_renders`: with a saved voice that has `AdapterRef::None`, `listen` makes the spy see `load("k2-fsa/OmniVoice", None, "auto")`, then one render of `TRY_SENTENCE`, and the result file plays through the shared player.
  - `generate_waits_while_voices_render` (Review Focus 5): start tries, switch to `Tab::Voice`. `voice-generate` is off. After the tries end it is on again, and the tries are still in the editor on return.
  - `try_waits_while_the_voice_tab_generates`: with the Voice tab busy, `voices-try` is off.
  - `delete_asks_then_removes`: save `Ece`. `ask_delete("Ece")` sets `confirm_delete`, nothing is removed yet. `confirm_delete` removes the folder, and `Ece` leaves both preset lists. `ask_delete("Sami")` does nothing.
  - `a_project_with_a_deleted_voice_opens_with_sami` (Review Focus 4): a project with `preset: "Ece"`, `Ece` deleted. `open_project` gives `preset_name == "Sami"` and a notice that names Ece.
- [ ] **Step 2:** Run `scripts\cargo.cmd test -p jeuvoice-app --bin jeuvoice-app voices::` — Expected: FAIL.
- [ ] **Step 3:** Implement. Workers send events through a channel, as `voice/state.rs` does. Try files go to `<app data>/voice-tries/`, removed at app start.
- [ ] **Step 4:** Run `scripts\cargo.cmd test -p jeuvoice-app` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(app): Voices tab state and one engine user at a time`.

### Task 5: Voices tab view and the card light

**Files:**
- Create: `crates/jeuvoice-app/src/voices/view.rs`
- Modify: `crates/jeuvoice-app/src/view.rs` (tab list, page), `crates/jeuvoice-app/src/style.rs` (top light), `crates/jeuvoice-app/src/voices/tests.rs` (view tests, `snap_voices`)

**Interfaces:**
- Consumes: Task 4.
- Produces: keys `tab-voices`, `page-voices`, `glass-voices-list`, `glass-voices-maker`, `voices-item-<name>`, `voices-new`, `voices-mode-clone`, `voices-mode-describe`, `voices-chip-<group>-<value>` (for example `voices-chip-gender-female`), `voices-drop`, `voices-clip-text`, `voices-try`, `voices-try-more`, `voices-take-<i>`, `voices-play-<i>`, `voices-settings`, `voices-name`, `voices-save`, `voices-listen`, `voices-delete-<name>`, `voices-delete-yes`, `voices-delete-no`.

- [ ] **Step 1: Write the failing tests** in `voices/tests.rs`:
  - `the_voices_tab_lists_sami_without_delete`: tap `tab-voices`. `voices-item-Sami` is on screen, `voices-delete-Sami` is not.
  - `chips_build_the_instruct`: tap `voices-mode-describe`, `voices-chip-gender-female`, `voices-chip-pitch-high`. The editor traits give `Some("female, high pitch")`. A second tap on the same chip clears it.
  - `try_is_the_main_button`: `crate::style::take_main_keys()` after one frame of the describe mode is `["voices-try"]`.
  - `the_voices_page_keeps_the_blur_budget`: at most 4 visible `glass-` elements with the delete question open.
  - `the_top_light_is_brighter` (in `style.rs` tests): the highlight color is `white(0.22)`, and the contrast tests still pass.
- [ ] **Step 2:** Run `scripts\cargo.cmd test -p jeuvoice-app --bin jeuvoice-app voices:: style::` — Expected: FAIL.
- [ ] **Step 3:** Implement the view as in the spec: left list in `glass_blur("glass-voices-list", …)`, right maker in `glass_blur("glass-voices-maker", …)`. While setup is not ready, show `setup::view::waiting_hint()`.
- [ ] **Step 4:** Run `scripts\cargo.cmd test -p jeuvoice-app` — Expected: PASS. Add `#[ignore] fn snap_voices` (Turkish UI, describe mode, 3 tries, one picked) that writes `target/voices.png`. Run it with `-- --ignored` and look at the picture.
- [ ] **Step 5: Commit** `feat(app): Voices tab and brighter card light`.

### Task 6: Final check

- [ ] **Step 1:** Run `scripts\cargo.cmd test --workspace` and `engine\.venv\Scripts\python -m pytest engine/tests -q` — Expected: PASS.
- [ ] **Step 2:** Real check with the engine (about 5 minutes): start the debug app, make one described voice and one cloned voice from `assets/presets/Sami/ref.wav`, listen, save both, and generate one paragraph with each in the Voice tab.
- [ ] **Step 3:** Render `snap_voices`, `snap_voice` and `snap_train`. Show them to the user.
- [ ] **Step 4:** Update `docs/user-guide.md`: a "Voices" section (make, listen, save, delete) and the Sami name.
- [ ] **Step 5: Commit** `docs: Voices tab in the user guide`.
