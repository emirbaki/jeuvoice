# JeuVoice user guide

JeuVoice does three things in one window:

- **Transcribe**: it writes down what people say in an audio or video file.
- **Voice**: it reads your script aloud in a trained voice.
- **Train**: it teaches the voice model a new voice on Kaggle.

You do not need a terminal or config files.

## Install

### Windows

1. Double-click `jeuvoice-app_<version>_x64-setup.exe`.
2. Click **Next** until the install ends. You do not need admin rights.
3. Start JeuVoice from the Start menu.

The app goes to `%LOCALAPPDATA%\Programs\JeuVoice`.

### macOS (Apple Silicon only)

1. Open `JeuVoice_<version>_aarch64.dmg`.
2. Drag **JeuVoice** to **Applications**.
3. Start JeuVoice from Applications.

## First start

On the first start, JeuVoice installs its voice engine. This takes about 10 to 15 minutes. It needs 8 GB of free disk space and an internet connection.

1. Open the **Voice** tab. You see the setup screen with a list of steps.
2. Wait until all steps have a tick.
3. You can use the **Transcribe** tab during setup.

If setup stops, the next start cleans up and starts again. Click **Try again** to start it at once.

## Transcribe

1. Open the **Transcribe** tab.
2. Drop one or more files on the window. JeuVoice reads `.mp4`, `.ogg`, `.wav`, `.mp3` and more.
3. On the first use, JeuVoice downloads the speech model (about 1.6 GB). Wait for it.
4. Wait until each file shows **Done**. A bad file shows a message. The other files still finish.
5. Press play. Click a word to jump to it.
6. To fix a word, click it and type the right text.
7. To check a script, paste or open it. JeuVoice marks the words that are different.
8. Click **Export SRT…**, **Export VTT…** or **Export TXT…** to save the text.

## Voice

### Before the first render

The voice needs model files. Do one of these in **Settings**:

- Paste a Hugging Face token with read permission and click **Save**. The token goes into the system keychain, not into a file.
- Or, under **Voice model files**, type a checkpoint folder on your disk and click **Save folder**.

### Make a voice-over

1. Open the **Voice** tab.
2. Pick a voice. The built-in voice is **Sami**.
3. Paste your script. Put an empty line between paragraphs.
4. Click **Generate**. The first time, the model loads (about 30 seconds). Then JeuVoice reads one paragraph at a time.
5. Press play to listen.

### Retake a part

1. Drag over a part of the wave.
2. Click **Retake region**. Three new takes appear.
3. Click a take to hear it.
4. Click **Use this take** to put it in. Click **+ Try 3 more** for more takes.

### Fix a word

1. Double-click a word.
2. Type how it must sound in **Say it as**.
3. JeuVoice reads that paragraph again.

A name in Latin letters (for example `iPhone`) gets an **Add to English terms?** hint. Click it to keep the English sound for that name.

### Undo and save

- **Undo** and **Redo** (Ctrl+Z and Ctrl+Y) step through your changes.
- JeuVoice saves the project after each change. Projects are in `Documents/JeuVoice`.

### Export

- **Export WAV** saves the sound (24 kHz).
- **Export SRT** saves the subtitles.

### Presets

A preset is a saved voice. Make and manage voices in the **Voices** tab.

## Voices

The Voices tab lists your voices on the left. **Sami** is the built-in voice. You cannot delete Sami.

### Make a voice from a description

1. Click **+ New voice**, then **Describe**.
2. Pick the gender, age and pitch. You can leave a group empty. **Whisper** makes a whispering voice.
3. Click **Try 3 voices**. The first time, the voice model loads (about 30 seconds).
4. Press ▶ on each try to listen. Click the try you like. It gets an orange outline.
5. Click **+ Try 3 more** for three more tries.

### Make a voice from a clip (clone)

1. Click **+ New voice**, then **Clone**.
2. Drop a clean clip of 3 to 10 seconds, or click the area to pick a file.
3. Whisper writes what the clip says. Read the text and fix it.
4. Click **Try** and listen.

### Save, listen, delete

- **Settings** (closed at first) holds these settings. The panel scrolls when it is longer than the window.
  - **Model**: **tr2 (Turkish)** speaks Turkish well. **Base model** follows gender, age and pitch better, but speaks Turkish less well.
  - **Pitch shift**: makes the voice lower (−) or higher (+), in semitones. A pitch chip sets it too (Very low −4 … Very high +4).
  - **Variety**: 0 gives the likeliest sound. A higher value gives more different voices from the same seed.
  - **Speed**, **Base seed**, **Guidance** and **Quality steps** (16, 32 or 64: more is slower and cleaner).
  - **Pauses**: the silence between sentences and between paragraphs.
- The voice settings use fixed English words for the model (for example `female, elderly`). The model does not always follow **Elderly**.
- Type a name and click **Save**. The new voice shows in the Voice tab at once and is selected there.
- New voices use the same model as Sami (the tr2 adapter).
- Click a voice in the list, then **Listen**, to hear the test sentence with it.
- Click ✕ next to a voice to delete it. The app asks first. A project that used a deleted voice opens with Sami.
- While the Voice tab generates, **Try** and **Listen** wait. While the Voices tab works, **Generate** waits.

## Train

Training runs on Kaggle, not on your computer. Kaggle gives free GPU time each week.

1. **Kaggle key**: on the Train tab, click **Add Kaggle key**. Type your Kaggle user name and the key from `kaggle.com/settings` (API, Create New Token). Click **Save key**. Or click **Import kaggle.json**.
2. **Data**: drop voice clips, or click **Add Hugging Face dataset…**. Check the text of each clip. Exclude bad clips.
3. **Model**: pick the model to start from.
4. **Where to train**: pick **Kaggle**.
5. **Start**: click **Start on Kaggle**, then **Start**. About 1,000 steps take one hour. 6,000 steps take about 6 hours.

You can close JeuVoice during a run. The run continues on Kaggle. On the next start, the Train tab shows the progress again.

### Test a checkpoint

1. When the run makes a checkpoint, it shows in the list.
2. Click **Test** next to a checkpoint. JeuVoice reads a test sentence with it.
3. Listen and keep the best checkpoint.

## When something goes wrong

| The app says | Do this |
|---|---|
| JeuVoice needs 8 GB of free space on ... Free some space and click Try again. | Delete files you do not need, then click **Try again**. |
| The engine could not be installed. Close other programs and click Try again. | Close other programs, also an antivirus scan if one runs. Click **Try again**. |
| The engine environment could not be made. Close other programs and click Try again. | Same as above. If it fails again, restart the computer. |
| Python could not be unpacked. Close other programs and click Try again. | Same as above. |
| The engine could not start. Close other programs and click Try again. | Click **Try again**. If it fails again, restart the computer. |
| The voice engine is from another JeuVoice version. Click Repair. | Open **Settings** and click **Repair**. Setup runs again (10 to 15 minutes). |
| The voice engine is not installed yet. | Open the **Voice** tab and let setup finish. |
| The voice engine is off. Open the Voice tab once to start it, then try again. | Open the **Voice** tab, then try again. |
| The voice engine is not ready. Restart the engine and try again. | Open **Settings** and click **Restart engine**. |
| The voice engine refused the connection. Restart the engine and try again. | Open **Settings** and click **Restart engine**. |
| The voice engine failed. Try again. | Click **Generate** again. If it fails again, restart the engine in **Settings**. |
| The voice model could not be loaded. | Check the Hugging Face token or the folder under **Voice model files** in **Settings**. |
| The model files could not be downloaded. | Check your internet connection and the Hugging Face token. |
| The render failed. Try again. | Click **Generate** again. |
| The voice engine does not run on this Mac yet. | Use the Voice tab on Windows. Transcribe and Train work on the Mac. |
| JeuVoice cannot read this file type: ... | Use a supported audio or video file. |
| The transcription failed. Try the file again. | Drop the file again. |
| The speech model could not be loaded. Download it again from the Settings tab. | Open **Settings** and download the speech model again. |
| Choose a preset first. / Paste a script first. | Pick a preset and paste a script, then click **Generate**. |
| Drag a region on the wave first. | Drag over the wave, then click **Retake region**. |
| Add your Kaggle key first. | Add the key on the Train tab. |
| Kaggle is busy. Wait a minute and try again. | Wait one minute, then click **Start on Kaggle** again. |
| Kaggle is still preparing the dataset. Wait a minute and start again. | Wait one minute, then click **Start on Kaggle** again. |
| Kaggle could not be reached ... Check your connection and try again. | Check your internet connection. |
| Your graphics card has ... GB. Training needs at least 12 GB. | Train on Kaggle. |
| No supported graphics card found. Speech will be slow. | JeuVoice found no NVIDIA graphics card. Transcribe works, but it is slow. Update the NVIDIA driver if the PC has an NVIDIA card. |
| No supported graphics card found. The voice engine needs an NVIDIA graphics card. | Use the Voice tab on a PC with an NVIDIA card. |

If a run fails on Kaggle, the Train tab shows the last 30 log lines. Click **Download bundle only** to save the data and the notebook. Send them to the team.

## Where JeuVoice keeps files

| What | Where |
|---|---|
| Your projects, presets and training projects | `Documents/JeuVoice` |
| The voice engine and the model files (about 8 GB) | Windows: `%APPDATA%\Jeu\JeuVoice`. macOS: `~/Library/Application Support/com.Jeu.JeuVoice` |
| The Hugging Face token and the Kaggle key | The system keychain |

Updates and the uninstaller never delete `Documents/JeuVoice`. The uninstaller asks before it deletes the voice engine and the models.
