# macOS app bundle and disk image

A teammate with an Apple Silicon Mac runs these steps. The agent has no Mac.
Time: about 30 minutes for the first build, then about 5 minutes.

## What you need

- An Apple Silicon Mac with macOS 13 or newer.
- Xcode command line tools: `xcode-select --install`.
- Rust (rustup), with the toolchain from `rust-toolchain.toml`.
- CMake: `brew install cmake`.
- cargo-packager 0.11.8 or newer: `cargo install cargo-packager --locked`.

## Make the app and the dmg

1. Open a terminal in the repo root.
2. Run `scripts/make-dmg.sh`.
3. Find `dist/macos/JeuVoice.app` and `dist/macos/JeuVoice_<version>_aarch64.dmg`.

The script builds the release app (whisper uses Metal), downloads the pinned
Python 3.12 and `uv` once, and checks their SHA-256. It puts these files in
`JeuVoice.app/Contents/Resources`:

| File | What it is |
|---|---|
| `python-3.12.16-20261110.tar.gz` | python-build-standalone, `aarch64-apple-darwin`, `install_only` |
| `uv` | uv 0.13.7 for `aarch64-apple-darwin` |
| `requirements-macos.txt` | The locked engine packages for macOS |
| `engine/` | The engine source, without `.venv` and tests |
| `assets/` | Presets, recipes and the Kaggle template |

There are no CUDA files on macOS. The app finds `Contents/Resources` from
`Contents/MacOS/jeuvoice-app` (see `runtime_paths.rs`).

## Sign and notarize

Use your own Developer ID certificate and an App Store Connect API key or an
app-specific password. Replace `TEAMID`, `Your Name` and `jeuvoice-notary`.

1. Save the notary credentials once:

   ```bash
   xcrun notarytool store-credentials jeuvoice-notary --apple-id you@example.com --team-id TEAMID
   ```

2. Sign the app (hardened runtime, every nested binary):

   ```bash
   codesign --force --deep --options runtime --timestamp --sign "Developer ID Application: Your Name (TEAMID)" dist/macos/JeuVoice.app
   ```

3. Make the dmg again from the signed app, or sign the dmg:

   ```bash
   codesign --force --timestamp --sign "Developer ID Application: Your Name (TEAMID)" dist/macos/JeuVoice_*_aarch64.dmg
   ```

4. Send the dmg to Apple and wait for the result:

   ```bash
   xcrun notarytool submit dist/macos/JeuVoice_*_aarch64.dmg --keychain-profile jeuvoice-notary --wait
   ```

5. Attach the ticket to the dmg:

   ```bash
   xcrun stapler staple dist/macos/JeuVoice_*_aarch64.dmg
   ```

6. Check the result: `spctl -a -vv -t install dist/macos/JeuVoice_*_aarch64.dmg`.

`uv` and the Python archive are signed by their makers. The Python files that
setup unpacks go to the app data folder, outside the app, so they do not
change the app signature.

## Known limits on macOS

- The Voice tab is off on macOS until the Mac spike passes
  (`VOICE_SUPPORTED` in `voice/state.rs`, see `docs/mac-spike.md`). The app
  shows the Plan 2 message on the Voice tab. Setup still installs the engine,
  because the Train tab uses it for data prep.
- The free-space check reads 0 on macOS, so it never blocks setup. Make sure
  that the Mac has 8 GB free.
- The build is for Apple Silicon only. An Intel Mac cannot start it.

## Test record

Run these on a clean macOS user account. Write the result of each step here.

| Step | Date | Result |
|---|---|---|
| `scripts/make-dmg.sh` ends with "Done". | | |
| The signed dmg passes `spctl`. | | |
| Drag the app to Applications. It starts with no Gatekeeper warning. | | |
| First start shows the setup screen. Setup reaches "Ready". | | |
| Transcribe one file (Metal). | | |
| The Voice tab shows "The voice engine does not run on this Mac yet." | | |

## Send back

- The last 30 lines of the `make-dmg.sh` output.
- The filled test record above.
- If setup fails: a screenshot of the setup screen with the error text.
  The app data folder is `~/Library/Application Support/com.Jeu.JeuVoice/`.
