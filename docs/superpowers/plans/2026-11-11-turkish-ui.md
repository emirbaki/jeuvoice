# Turkish UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the whole JeuVoice UI in English or Turkish, with the choice saved in `settings.json`.

**Architecture:** A module `i18n.rs` keeps the current language (a global atomic in the app, a thread-local value in tests). A macro `tr!(en, tr)` picks the text at the call site. Settings gets `ui_language` and a new "Interface language" row.

**Tech Stack:** Rust, rune UI toolkit, serde.

**Spec:** `docs/superpowers/specs/2026-11-11-turkish-ui-design.md`

## Global Constraints

- Build and test only with `scripts\cargo.cmd`.
- Keep every element `key` unchanged.
- English is the default. `ui_language` is `"en"` or `"tr"`, with `#[serde(default)]`.
- English text and comments follow ASD-STE100. Turkish text is plain and neutral, with the glossary in the spec.
- Text from outside the app (`{e}` parts) is not translated.

## Review Focus

- An old `settings.json` without `ui_language` loads in English. Test: `old_settings_load_in_english`.
- An unknown `ui_language` value ("de") falls back to English. Test: `unknown_ui_language_is_english`.
- Parallel tests: a Turkish test does not change the language of an English test. Test: `language_is_per_test_thread`.
- The language applies before the first frame after a restart. Test: `saved_turkish_shows_on_first_frame`.
- Long Turkish text in the 190 pt Train side column and the diff pills. Check: `snap_train_tr`, `snap_train_data_tr`.

---

### Task 1: The i18n module and the switch

**Files:**
- Create: `crates/jeuvoice-app/src/i18n.rs`
- Modify: `src/main.rs` (mod), `src/settings.rs` (field), `src/state.rs` (`App::new`, `set_ui_language`), `src/settings_tab.rs` (row)
- Test: `src/i18n.rs`, `src/settings_tab_tests.rs`

**Interfaces:**
- Produces: `pub enum Lang { En, Tr }`, `Lang::from_code(&str) -> Lang`, `Lang::code(self) -> &'static str`, `pub fn lang() -> Lang`, `pub fn set_lang(Lang)`, macro `tr!(en, tr)` returning `&'static str`, and `tr!(en_fmt, tr_fmt, args..)` returning `String`. `App::set_ui_language(&mut self, code: &str)`. Keys `ui-lang-en`, `ui-lang-tr`.

- [ ] Write tests: `unknown_ui_language_is_english`, `language_is_per_test_thread`, `old_settings_load_in_english`, `the_interface_switch_saves_turkish` (press `ui-lang-tr`, then `settings.json` has `"ui_language": "tr"` and `tab-settings` shows "Ayarlar"), `saved_turkish_shows_on_first_frame`.
- [ ] Implement, then run `scripts\cargo.cmd test -p jeuvoice-app` and make sure all pass.

### Task 2: Root, Settings, setup and services texts

**Files:** `src/view.rs`, `src/settings_tab.rs`, `src/state.rs`, `src/setup/*.rs`, `src/version.rs`, `src/services/*.rs`, `src/runtime_paths.rs`, `src/settings.rs`, `src/cache.rs`, `src/playback.rs`, `src/waveform.rs`, `src/marks.rs`

- [ ] Wrap each user-facing text in `tr!`. Constants that tests compare become functions.
- [ ] Add test `setup_shows_turkish` (one setup text in Turkish).

### Task 3: Transcribe and Voice texts

**Files:** `src/transcribe.rs`, `src/transcribe_tab.rs`, `src/voice/*.rs`

- [ ] Wrap each text in `tr!`. Add tests `transcribe_shows_turkish`, `voice_shows_turkish`.

### Task 4: Train texts

**Files:** `src/train/*.rs`

- [ ] Wrap each text in `tr!`. Add test `train_shows_turkish`.

### Task 5: Turkish snapshots and the full run

- [ ] Add `snap_settings_tr`, `snap_setup_tr`, `snap_train_tr`, `snap_train_data_tr`, `snap_train_run_tr`, `snap_transcribe_tr`, `snap_voice_tr`.
- [ ] Run them with `--ignored`. Look for overflow. Shorten Turkish text before a layout changes.
- [ ] Run `scripts\cargo.cmd test --workspace`. Commit. Send the images to the user for approval.
