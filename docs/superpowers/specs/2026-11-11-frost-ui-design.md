# Frost UI redesign — design

Date: 2026-11-11. Status: waiting for approval. Reviewed by a builder check.

## Why

The app works, but it does not look good. The user finds the Train tab the worst: it is one long scroll, every part has the same gray card style, and nothing shows where you are or what to do next. The other tabs need smaller fixes. The user wants a compact, modular layout, a frosted-glass look, and smooth motion between pages.

## Goals

1. **Train tab as a tile board** with no page scroll at the default window size.
2. **One look for the whole app:** the "Frost" style (see below).
3. **Smooth motion:** tab changes, tile changes and panels move. Nothing jumps.
4. **A wave that shows the sound:** the color follows the sound pitch, and the words show as separated blocks.

## Non-goals

- **Turkish UI (i18n):** a separate project after this one, when the texts are final. This project changes no text that it does not have to change.
- **Light mode.** The app stays dark only.
- **New features.** Each button and each flow stays as it is. Only the layout and the look change.
- **macOS native materials** (`system_material`). We use rune's own blur, which looks the same on Windows and macOS.

## Anti-goals (a failure, even if it works)

- The app gets slower to draw, or the wave lags during playback.
- A teammate cannot find a button that was there before.
- A colorful, busy look. Frost is calm: little color, one orange accent.
- Text with low contrast on glass. Body text must stay easy to read.

## Constraints

- **Toolkit:** rune at the pinned revision (`Cargo.toml`). No new UI library.
- **Blur limit:** rune blurs at most 4 elements per frame (`MAX_BACKDROP_BLURS = 4` in `rune/src/render/backdrop_blur.rs`). Elements past 4 draw only their fill. The design must not depend on more than 4 blurred elements on one screen.
- **Tests:** the existing UI tests find elements by `key` (for example `tab-train`, `train-run-start`, `voice-generate`). Keep all keys. A test that breaks because of a layout change gets a fix in the test only if the behavior is the same.
- **Snapshots:** the `snap_*` tests (`--ignored`) render each page to `target/*.png`. Use them to check each page, and include the new pictures in the final report.
- Window default size: 1100 × 720 points (see the snap tests).

## The Frost style (decided by the user)

Mockup: `.superpowers/brainstorm/*/content/visual-style.html`, option B.

- **Background:** dark gray, almost black, with a soft light spot at the top left. Approximate CSS reference: `radial-gradient(at 20% 10%, rgba(255,255,255,.08), transparent 60%)` over `linear-gradient(#18191d, #101114)`.
- **Glass panel:** white fill at about 5.5 % opacity, a 1 pt white border at about 9 % opacity, a thin inner highlight at the top edge, and a large corner radius (about 14 pt).
- **Accent:** one warm orange (about `#ff9f5a`). Use it for the main action of a page (for example **Generate**, **Start on Kaggle**), the active state, and progress bars. All other buttons are gray glass.
- **Text:** near-white for body text, gray (about `#8d8e96`) for secondary text.
- **Errors and success:** keep red and green, but tune them to fit the dark gray.
- **Put all of this in one place:** a style module (for example `crates/jeuvoice-app/src/style.rs`) with the colors, the glass panel function, the button kinds (main, normal, quiet) and the motion values. The 58 direct calls to `tok::theme(true)` and the `red()/orange()/green()` helpers move to this module. One place makes the i18n project and later style changes easy.

## Tab bar and page change (proposed by Claude, accepted by the user)

- **Tab bar:** a pill-shaped glass bar at the top left. A bright pill slides under the active tab (rune `matched` or a layout spring). The tab bar is one of the blurred elements.
- **Page change:** the new page fades in and moves up about 8 pt, in about 0.2 s (rune `Transition::offset(..).combined(Transition::opacity())`). Keep motion off when the system asks for reduced motion (rune does this for matched motion).

## Train tab (decided by the user)

Mockups: `train-layout.html` option C, `tile-open.html` option 3.

**Board (nothing open):** a header row with the title "Training" and a small quiet button "How tr2 was made". Under it, a grid of glass tiles that fills the page:

| Tile | Place | Shows |
|---|---|---|
| Data | big, on the left, full height | the totals as pills (clips, minutes, left out, band) and the source names with their clip counts |
| Model and Where | top right | the model to start from, and the training place with its short facts (for example "Kaggle · T4 ×2 · free") |
| Run | bottom right | changes with the run state: before a run, the time estimate and "Ready to start". During a run, the step count and the progress bar. After a run, the number of checkpoints. |

**Board tiles are summaries only (decided by the user).** A board tile has no buttons and no text fields. A click anywhere on a tile opens it. All controls, and all their `key`s, live in the open tiles only. Tests that tap a Train control first open its tile.

**My call (Claude):** files dropped on the Train page while the board shows open the Data tile and add the files, as a drop on the Data step does today.

**Open a tile:** a click on a tile makes it wide. The other tiles slide into a narrow column on the right, and they show only their title and one line of state. A click on a tile in the column opens that tile. A close button, or a click on the open tile title, goes back to the board. The motion is a layout spring, so the tiles slide and do not jump. Use rune `matched` ids per tile, or `animate_layout`.

- The opened Data tile holds what the Data step has today: sources, Hugging Face dataset, and the clip review (play, text, Exclude/Include).
- The opened Model and Where tile holds the model step and the target step (Kaggle profile, key, This PC reason, RunPod "Coming later").
- The opened Run tile holds the run step: settings, start, log, progress, loss curve, checkpoints and **Test**.
- "How tr2 was made" opens the recipe as before. The builder decides if it uses a tile-sized panel or an overlay.

**My call (Claude):** the open tile is kept in the Train state (for example `open_tile: Option<TrainTile>`) and is not saved to disk. The board shows on each app start.

## Wave colors (decided by the user)

Mockup: `wave-color.html` option 3. This applies to both waves: Voice and Transcribe (`waveform.rs`).

- **Color by sound pitch:** low sounds (vowels) are warm orange, middle sounds are pink, and high hissing sounds (s, ş, ç) are teal. The color changes smoothly between them.
- **Played and not played:** the wave is bright before the play position and about 30 % bright after it.
- **Word gaps:** a small gap (a few points) shows where each word ends, when the words are known.
- **The pitch value** must come from the audio, computed with the peaks. My call (Claude): a zero-crossing rate per peaks bucket is enough and cheap, because hissing sounds have a high rate. Add it next to `min` and `max` in `jeuvoice_core::peaks::Peaks`. The builder can use a better measure if it stays cheap.
- The selection tint and the playhead stay. The playhead gets a soft glow.
- The wave must stay as fast as today during playback.

## Other pages (proposed by Claude, accepted by the user)

- **Voice:** two glass panels: takes on the left, editor on the right. **Generate** is the only orange button. The other buttons are gray glass. The play bar at the bottom is a glass bar.
- **Transcribe:** the same three columns (files, wave, script check), in glass panels. The script check shows the differences as colored pills (missing, changed, added).
- **Settings:** small glass tiles in two columns, not one long column. Groups: Speech (language, speech model), Graphics card, Voice engine (status, restart or Repair, adapter folder), Accounts (Hugging Face token, Kaggle profiles).
- **Setup screen:** one glass card in the center, with the step list (ticks in orange or green), the progress bar and the time hint.

## Blur budget per screen (Claude's call)

At most 4 blurred elements on a screen. Use blur for the tab bar and the largest panels only, for example: tab bar + up to 3 main panels. Small tiles, buttons and pills use the glass fill without blur. On the plain dark Frost background the difference is small, so this costs little.

## Review point (Claude's call)

Build the style module, the tab bar and the Train board first. Then stop and show the user the snapshots. Continue with the other pages only after the user approves the look. A wrong look then costs one page of work, not seven.

## Left to the builder

- The exact numbers: sizes, spacing, radii, spring values and colors, inside the limits above.
- How the style module is split, and the names of its functions.
- How the tiles animate in rune (`matched`, `animate_layout`, or springs on size).
- Small layout details on each page, if each button and key stays.

## Done when

1. All workspace tests pass (`scripts\cargo.cmd test --workspace`).
2. The `snap_*` pictures show the Frost style, and the user approves them: Transcribe, Voice, Train board, Train with the Data tile open (clip review), Train with the Run tile open (before a run), Settings and Setup. That is seven pictures.
3. The Train board shows with no page scroll at 1100 × 720.
4. The wave shows pitch colors and word gaps in Voice and Transcribe.
