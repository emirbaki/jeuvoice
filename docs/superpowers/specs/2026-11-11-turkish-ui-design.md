# Turkish UI language: design

## Intent

The user is Turkish. JeuVoice shows every UI text in English now. The app must
show the whole UI in Turkish or in English. The user selects the language in
Settings, and the choice stays in `settings.json`.

## Decisions by the user

- Default language: always English on the first start. The app does not read
  the system locale.
- The switch: a separate "Interface language" row on the Settings page. It is
  not part of the Whisper "Speech language" picker.
- The user guide `docs/user-guide.md` stays English. No Turkish copy now.
- Tone: plain and neutral, like Windows in Turkish. Short impersonal labels
  ("Kaydet", "Model indiriliyor…"). No "sen" and no "siz" address where a
  label can do without it. Where a sentence must address the user, use the
  polite imperative form ("Tokeni yapıştırın").

## Goals

- Every user-facing string in `crates/jeuvoice-app` has an English and a
  Turkish text: tab names, buttons, tile titles, hints, status lines,
  placeholders, accessibility labels (`.label(...)`), and the plain error
  texts in `setup/steps.rs`, `version.rs`, `services/gpu.rs`,
  `services/engine.rs`, `state.rs` and the tab views.
- A language change shows at once on the next frame. No restart.
- Every element `key` stays unchanged.

## Non-goals

- Turkish copies of docs, the installer, or the README.
- Translation of text that comes from outside the app: Python engine messages,
  Kaggle API errors, OS error texts (`{e}` parts). These stay as they come.
- Translation of user data: preset names, recipe file content, transcripts.
- More languages than two. The design must not block a third language, but no
  work goes into that now.

## Anti-goals

- Text that runs out of a tile, a pill or the 190 pt Train side column.
- English tests that break, or keys that change.
- A Turkish text that a Turkish speaker reads as a machine translation.

## Constraints

- Build and test only with `scripts\cargo.cmd`.
- Colors and styles stay in `style.rs`. Pages do not call `tok::theme(true)`.
- English UI text and comments follow ASD-STE100.

## Design (my calls, open for change)

1. **Module `i18n.rs`.** It holds `enum Lang { En, Tr }` and the current
   language. Production code keeps the language in one global atomic, so
   worker threads (setup steps, engine start) also make their error texts in
   the current language. Under `cfg(test)`, the language is thread-local. Each
   test thread starts in English, so parallel tests do not affect each other.
2. **Strings at the call site.** A small macro `tr!("Download", "İndir")`
   returns the text for the current language. The English and Turkish texts
   stay next to each other in the code. No separate key table. Texts with
   values use `tr!` with `format!`-style arguments for both languages, because
   Turkish word order and suffixes differ ("{n} files" → "{n} dosya").
3. **Settings.** `Settings` gets `ui_language: String` with
   `#[serde(default)]` and the value `"en"`, so old `settings.json` files load
   without change. `App::set_ui_language(code)` saves at once, the same way
   `set_language` saves the Whisper language. `App::new` and `Settings::load`
   apply the saved language before the first frame.
4. **Switch.** In the Settings "Speech" tile area, a new row "Interface
   language" with two radio pills: "English" and "Türkçe". Each name shows in
   its own language. New keys: `ui-lang-en`, `ui-lang-tr`.
5. **Stored messages.** Status texts that the app keeps in state (for example
   `files_status`) are made in the language that is current at that time. A
   language change does not translate a message that is already on the screen.
   That is acceptable, because the next action makes a new message.
6. **Turkish words for app terms.** Fixed glossary, used the same everywhere:
   Transcribe → Yazıya dök, Voice → Ses, Train → Eğitim, Settings → Ayarlar,
   preset → hazır ayar, checkpoint → kontrol noktası, voice engine → ses
   motoru, speech model → konuşma modeli, token → token, profile → profil,
   clip → klip, run (a Kaggle run) → çalıştırma. The builder may adjust a word
   where it reads badly, and lists each change for the user.

## Tests

- All existing tests stay in English and keep passing without change.
- New tests: switch to Turkish through the `ui-lang-tr` pill and check that
  `settings.json` has `"ui_language": "tr"` and that the tab names show
  "Ayarlar" and "Eğitim". A few more tests check one Turkish text in Train,
  Voice, Transcribe and setup.
- New ignored snapshot tests `snap_*_tr` for Settings, Setup, Train (board,
  Data, Run), Transcribe and Voice. The user reviews these images.
- Overflow check: in the Turkish snapshots, look at the 190 pt Train side
  column, the diff pills, the tab bar and every button row. Shorten a Turkish
  text before a layout is changed.

## Left to the builder

- The exact macro shape and the name of the module.
- Turkish wording inside the tone and glossary rules above.
- Which few texts the new Turkish tests check.
