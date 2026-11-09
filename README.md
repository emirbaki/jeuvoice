# JeuVoice

## Build on Windows

The `whisper-rs` crate builds whisper.cpp with CMake and CUDA. It needs Visual Studio with the C++ tools, the CUDA toolkit, and some environment settings. The script `scripts\cargo.cmd` sets them. Use it in place of `cargo`:

```
scripts\cargo.cmd build
scripts\cargo.cmd test
```

The script sets the CMake generator to `Visual Studio 17 2022`. It sets `CMAKE_CUDA_ARCHITECTURES` to 86 (RTX 30 series) unless you set it first.

## Run

Build the release app, then start it:

```
scripts\cargo.cmd build --release -p jeuvoice-app
target\release\jeuvoice-app.exe
```

The first release build takes about 11 minutes. The first time you transcribe, JeuVoice downloads the speech model (about 1.6 GB).

## Status

Plan 1 (Transcribe) is done.

### What works

- Drop audio or video files (`.mp4`, `.ogg`, `.wav`, `.mp3` and more). JeuVoice reads the sound itself. It needs no ffmpeg.
- A batch runs one file at a time. If one file is bad, the others still finish. The bad file shows a plain error.
- Paths with Turkish letters and spaces work.
- Words come with start and end times. You can play the take, click a word to jump to it, and fix a word.
- You can paste or open a script. JeuVoice marks the words that differ from what it heard.
- You can export SRT, VTT or plain text.

### Measured speed

Checked on 2026-11-09 with the release build. Model: `large-v3-turbo`. Language: Turkish.

| File | Sound | Time | Speed |
|---|---|---|---|
| my_voice2.mp4 | 33.6 s | 2.3 s | 14.9 min per min |
| my_voice.ogg | 27.0 s | 1.2 s | 22.1 min per min |
| çekim 1.wav | 8.4 s | 0.8 s | 10.7 min per min |
| All three | 69.0 s | 4.3 s | 16.2 min per min |

"min per min" means minutes of sound for each minute of work. The model ran on the GPU (NVIDIA, CUDA). Short files look slower, because each file has a fixed start cost. The model load time is not in these numbers.

The script check of one take against its own text gave 20 words the same and 0 different.

To repeat the check:

```
scripts\cargo.cmd test -p jeuvoice-app --release --test real_check -- --ignored --nocapture
```

### Known limits

- Word start times can be 0.1 to 0.3 seconds late.
- The macOS build is not tested.
- The RunPod, Voice and Train tabs are not in this plan.
- A file that is not audio gives a message with a technical detail at the end.
- An automatic check cannot see the screen or hear sound. The window opens and stays open for 10 seconds without a crash. The list below needs your eyes and ears.

### Try it

1. Start the app. Download the model when it asks.
2. Drop an `.mp4`, an `.ogg`, and a file from a folder with Turkish letters.
3. Wait until all three are done.
4. Press play. Check that the sound and the cursor move together.
5. Click a word. Check that playback jumps to that word.
6. Fix one wrong word. Check that the new text stays.
7. Paste a script. Check that the marked words are the real differences.
8. Export SRT. Open it next to the video in a player. Check the timing.
