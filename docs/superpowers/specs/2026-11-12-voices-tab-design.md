# Voices tab — design

Date: 2026-11-12. Status: waiting for approval. Reviewed by a builder check.

## Why

The app has one voice: the preset `ses_0007` (a clone of one reference clip, rendered with the base model and the tr2 adapter). The user wants to make more voices in the app, listen to them, and save the good ones as presets. The logic for this exists in `crates/jeuvoice-app/src/voice/preset_editor.rs` (`PresetEditor`, with tests), but no screen uses it.

## Goals

1. A new **Voices** tab ("Sesler" in Turkish) where the user makes, listens to and saves voices.
2. Two ways to make a voice (decided by the user: both):
   - **Clone:** from a 3–10 s clip of a voice.
   - **Describe:** from voice attributes (OmniVoice voice design, `instruct`) and a seed.
3. Listen to each try before saving. Pick one try, name it, save it.
4. A saved voice shows at once in the Voice tab preset list.
5. A stronger thin light line on the top edge of every glass card, in the whole app (decided by the user: option 1 of the card light mockup).

## Non-goals

- Free-text voice prompts. OmniVoice accepts only fixed attributes (see below). Any other text is an error in OmniVoice.
- Training a new adapter. That stays in the Train tab.
- Editing a saved voice in place. To change a voice, make a new one. (Claude's call.)
- Renaming a voice after save, except the one rename of `ses_0007` below.

## Anti-goals

- A saved voice that sounds different in the Voice tab than in its try. The saved preset must hold every value the try used: kind, clip, clip text, attributes, seed, model and adapter, guidance, speed.
- Removing or breaking `ses_0007`.
- A voice maker that blocks the UI while Whisper or the engine works.

## Facts the design rests on (checked on 2026-11-12)

- `OmniVoice.generate` has three modes: clone (reference clip), design (`instruct`), auto (neither).
- Design attributes (`omnivoice/models/omnivoice.py`, `_resolve_instruct`), English, comma + space separated:
  - gender: `male`, `female`
  - age: `child`, `teenager`, `young adult`, `middle-aged`, `elderly`
  - pitch: `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch`
  - style: `whisper`
  - English accents (only for English speech). Leave these out: the app speaks Turkish.
- The prompts still work with the tr2 adapter. Median pitch, same sentence, seed 4: "female, young adult, high pitch" gives 377 Hz on base and 301 Hz with tr2. "male, middle-aged, low pitch" gives 148 Hz on both. The files are in `target/design-test/`.
- The engine already reads `instruct` and an optional `ref_audio` from the render preset (`engine/jeuvoice_engine/jobs.py`, `make_render_run`). No engine change is needed.
- `PresetEditor::save` today writes `AdapterRef::None`, so a new voice would use the base model without tr2. This must change (see "Model").

## The Voices tab (decided by the user: layout A)

Mockup: `.superpowers/brainstorm/769-1794498780/content/voice-maker-layout.html`, option A.

A new tab between Voice and Transcribe. Two glass panels:

**Left: My voices.** A list of the saved presets (`list_presets`), with the kind ("clone" or "described") under each name. At the bottom, "+ New voice". A click on a saved voice shows its values on the right and a **Listen** button that renders the test sentence with it. (Claude's guess, not contradicted by the user.)

**Right: New voice.** A switch with two modes, **Clone** and **Describe**.

- **Clone:**
  1. Drop a clip (or click to pick one). The app checks the length (3–10 s) and runs Whisper on a worker.
  2. The clip text shows in a field. The user checks and fixes it.
  3. **Try** renders the test sentence (`TRY_SENTENCE`) with the clip.
- **Describe:**
  1. Pick attributes as chips: gender (one), age (one), pitch (one), whisper (on or off). Each group can stay empty.
  2. **Try 3 voices** renders the test sentence with three seeds: the base seed, +1, +2.
  3. **Try 3 more** renders the next three seeds. (Claude's call: the Voice tab already has the same idea for takes.)
- **Tries:** each try is a row with a play button, a small wave, its seed, and a pick mark. Play uses the shared player, as the Voice tab does. The picked try has the orange outline.
- **Settings** (closed by default, Claude's call): base seed, guidance (default 2.5), speed (default 1.0). The user asked for "seed and other settings".
- **Save:** a name field and **Save**. The name must not be empty and must not match a saved voice. After save, the new voice is selected in the left list and in the Voice tab preset list.

The **Try** buttons are the orange main buttons of the panel. All other buttons are gray glass.

## Model (decided by the user)

A new voice uses the tr2 adapter by default, the same model as Sami: the base model plus the tr2 adapter. `PresetEditor::save` copies the `model` field of the Sami preset as it is on disk. If that preset is missing, it uses `default_preset().model`. It never writes `AdapterRef::None` for a new voice.

## Sami: the built-in voice (decided by the user)

- The built-in preset `ses_0007` gets the name **Sami**. The asset in `assets/presets/` and `default_preset()` use the new name.
- Sami cannot be deleted. Its row has no delete button.
- Existing installs (Claude's call): on start, if `<presets>/ses_0007/` exists and `<presets>/Sami/` does not, the app renames the folder and the `name` field to Sami. A project or setting that names `ses_0007` resolves to Sami, so old projects still open with their voice.

## Delete (decided by the user)

- Every saved voice except Sami has a delete button in the My voices list.
- Delete asks first: "Delete the voice <name>? This cannot be undone." with **Delete** and **Cancel**.
- Delete removes `<presets>/<name>/`. The voice leaves the Voices list and the Voice tab preset list.
- A project that used a deleted voice falls back to Sami when it opens, with a short notice. (Claude's call.)

## Engine use (Claude's calls, from the builder check)

- **Load before render:** before a Try and before Listen, the tab loads the voice's model and adapter with the same flow as the Voice tab (`resolve_adapter`, then `load`, with a "Loading the voice model" state). A try never renders with whatever model the engine has loaded by chance.
- **Listen on a saved voice** loads that preset's own model, also for an older preset without an adapter.
- **One engine user at a time:** while the Voice tab generates, the Try and Listen buttons are off with a short reason. While the Voices tab renders or loads, Generate in the Voice tab is off in the same way. No job queue.

## Card light (decided by the user: option 1)

Every glass panel (`style::glass`, `style::glass_blur`) gets a clearly visible thin light line on its top edge. Today the top of the border gradient uses `white(0.16)`. Make it brighter (about `white(0.22)`, as in the mockup `inset 0 1px 0 rgba(255,255,255,.22)`). Text contrast tests in `style.rs` must still pass.

## Constraints

- Use the Frost style module (`style.rs`). Do not call `tok::theme(true)` in a page.
- Every user-visible text goes through the i18n macros (`crate::i18n::tr!`, `trf!`) with English and Turkish.
- Keys: `tab-voices`, `page-voices`, and `voices-` as the prefix for all new keys.
- At most 4 blurred (`glass-`) elements on a screen.
- Whisper and engine calls run on workers. `PresetEditor` already blocks, so the tab calls it from a worker and polls it on the UI thread.
- The voice maker needs the engine. While setup has not finished, the tab shows the same waiting hint as the Train tab.
- Prose follows the ASD-STE100 rule.

## Left to the builder

- How the tab owns the editor state (a `VoicesTab` struct in `crates/jeuvoice-app/src/voices/` is the natural place).
- How the chips map to the `instruct` string (order: gender, age, pitch, whisper).
- Where the try files go (a temp folder under the app data folder is fine) and when they are removed.
- Exact sizes and spacing, inside the Frost style.

## Done when

1. A user makes a cloned voice and a described voice, listens to the tries, saves each, and uses each in the Voice tab to generate a paragraph.
2. A saved voice renders with the same model, adapter, seed and attributes as its try.
3. `scripts\cargo.cmd test --workspace` passes, with tests for the chip-to-`instruct` mapping, save validation (empty name, name taken), the saved model fields (tr2, copied from Sami), the `ses_0007` to Sami migration, Sami protection, and delete.
4. A `snap_voices` picture shows the tab in Turkish and the card light, and the user approves it.
