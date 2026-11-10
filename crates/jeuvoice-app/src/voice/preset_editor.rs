//! Make clone and seed presets.
//!
//! Clone: drop a 3-10 s clip. The app checks the length, runs Whisper on it and shows
//! the text in a field the user must check. **Try** renders the fixed test sentence with
//! the clip. Seed: type voice attributes, **Try 3 voices** renders three seeds, the user
//! plays each, picks one, names it and saves.
//!
//! All engine calls block: run them on a worker, [`PresetEditor::poll`] on the UI thread.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use jeuvoice_core::audio;
use jeuvoice_core::preset::{AdapterRef, ModelRef, Preset, PresetKind, list_presets, save_preset};
use jeuvoice_engine_client::{EngineApi, JobState, Preset as WirePreset, RenderRequest};
use jeuvoice_whisper::{TranscribeOptions, Transcriber};

use crate::settings::default_terms;

/// The fixed test sentence every try renders.
pub const TRY_SENTENCE: &str = "Bugün size kısa bir hikâye anlatacağım.";

/// One try render.
#[derive(Clone, Debug)]
pub struct TryVoice {
    pub seed: u64,
    pub job: String,
    pub file: Option<PathBuf>,
    pub done: bool,
}

/// Make clone and seed presets.
pub struct PresetEditor {
    transcriber: Arc<dyn Transcriber>,
    engine: Arc<dyn EngineApi>,
    /// Base seed: the clone try and the first of the three seed tries.
    pub base_seed: u64,
    pub kind: PresetKind,
    pub clip: Option<PathBuf>,
    /// Why the clip cannot be used.
    pub clip_error: Option<String>,
    /// The clip's transcript, for the user to check.
    pub clip_text: String,
    pub name: String,
    pub instruct: String,
    pub tries: Vec<TryVoice>,
    /// Which try the user picked.
    pub picked: Option<usize>,
    pub status: Option<String>,
}

impl PresetEditor {
    pub fn new(transcriber: Arc<dyn Transcriber>, engine: Arc<dyn EngineApi>) -> Self {
        Self {
            transcriber,
            engine,
            base_seed: 1,
            kind: PresetKind::Clone,
            clip: None,
            clip_error: None,
            clip_text: String::new(),
            name: String::new(),
            instruct: String::new(),
            tries: Vec::new(),
            picked: None,
            status: None,
        }
    }

    /// Check a dropped clip: 3-10 s, then Whisper text for the user to check.
    /// Blocks (decode + Whisper): call on a worker.
    pub fn check_clip(&mut self, path: &Path, language: &str) {
        self.clip = None;
        self.clip_error = None;
        self.clip_text.clear();
        let audio = match audio::decode(path) {
            Ok(a) => a,
            Err(e) => {
                self.clip_error = Some(e.to_string());
                return;
            }
        };
        let secs = audio.samples.len() as f64 / f64::from(audio.rate.max(1));
        if !(3.0..=10.0).contains(&secs) {
            self.clip_error = Some("Use a clip between 3 and 10 seconds.".into());
            return;
        }
        let opts = TranscribeOptions { language: language.into(), use_gpu: false };
        match self.transcriber.transcribe(&audio::to_whisper(&audio), &opts, &mut |_| {}) {
            Ok(t) => {
                self.clip = Some(path.to_path_buf());
                self.clip_text = t.text();
            }
            Err(e) => self.clip_error = Some(e.to_string()),
        }
    }

    /// Render the fixed test sentence with the checked clip. Blocks: call on a worker.
    pub fn try_clone(&mut self, out_dir: &Path) {
        let Some(clip) = self.clip.clone() else {
            self.status = Some("Drop a voice clip first.".into());
            return;
        };
        let mut preset = WirePreset::default();
        preset.ref_audio = Some(clip.to_string_lossy().into_owned());
        if !self.clip_text.trim().is_empty() {
            preset.ref_text = Some(self.clip_text.clone());
        }
        self.start_try(out_dir, self.base_seed, preset, None);
    }

    /// Render the fixed test sentence with three seeds and the typed attributes.
    /// Blocks: call on a worker.
    pub fn try_three(&mut self, out_dir: &Path) {
        self.tries.clear();
        self.picked = None;
        for k in 0..3 {
            let mut preset = WirePreset::default();
            let instruct = self.instruct.trim().to_string();
            if !instruct.is_empty() {
                preset.instruct = Some(instruct);
            }
            self.start_try(out_dir, self.base_seed + k as u64, preset, None);
        }
    }

    fn start_try(&mut self, out_dir: &Path, seed: u64, preset: WirePreset, paragraphs: Option<Vec<usize>>) {
        let req = RenderRequest {
            script: TRY_SENTENCE.into(),
            paragraphs,
            preset,
            terms: default_terms(),
            base_seed: seed,
            part_overrides: HashMap::new(),
            out_dir: out_dir.join(format!("try-{seed}")),
        };
        match self.engine.render(&req) {
            Ok(job) => self.tries.push(TryVoice { seed, job, file: None, done: false }),
            Err(e) => self.status = Some(e.to_string()),
        }
    }

    /// Advance try renders. Call on the UI thread.
    pub fn poll(&mut self) {
        for t in &mut self.tries {
            if t.done {
                continue;
            }
            let Ok(status) = self.engine.job(&t.job) else { continue };
            match status.state {
                JobState::Done => {
                    t.done = true;
                    t.file = status.render_result().and_then(|r| r.parts.into_iter().next()).map(|p| PathBuf::from(p.file));
                }
                JobState::Failed => {
                    t.done = true;
                    self.status = Some(status.error.unwrap_or_else(|| "The try failed. Try again.".into()));
                }
                _ => {}
            }
        }
    }

    /// Save the preset: the picked seed (or the base seed), the clip for clones.
    pub fn save(&mut self, presets_dir: &Path) -> Result<Preset, String> {
        let name = self.name.trim().to_string();
        if name.is_empty() {
            return Err("Give the preset a name first.".into());
        }
        let seed = self.picked.and_then(|i| self.tries.get(i)).map(|t| t.seed).unwrap_or(self.base_seed);
        let (ref_audio, ref_wav) = match self.kind {
            PresetKind::Clone => {
                let Some(clip) = self.clip.clone() else { return Err("Drop a voice clip first.".into()) };
                if self.clip_text.trim().is_empty() {
                    return Err("Check the clip text first.".into());
                }
                (Some("ref.wav".to_string()), Some(clip))
            }
            PresetKind::Seed => (None, None),
        };
        let instruct = match self.kind {
            PresetKind::Seed => {
                let s = self.instruct.trim().to_string();
                (!s.is_empty()).then_some(s)
            }
            PresetKind::Clone => None,
        };
        let preset = Preset {
            name: name.clone(),
            kind: self.kind.clone(),
            ref_audio,
            ref_text: if self.kind == PresetKind::Clone { Some(self.clip_text.clone()) } else { None },
            seed,
            instruct,
            model: ModelRef {
                base: "k2-fsa/OmniVoice".into(),
                adapter: AdapterRef::None,
            },
            guidance: 2.5,
            pause: 0.1,
            para_pause: 0.3,
            num_step: 32,
            speed: 1.0,
        };
        save_preset(presets_dir, &preset, ref_wav.as_deref()).map_err(|e| format!("The preset could not be saved: {e}"))?;
        self.status = Some(format!("Saved preset {name}."));
        Ok(preset)
    }
}

/// Read back what the editor wrote, for tests.
pub fn saved_preset(presets_dir: &Path, name: &str) -> Option<Preset> {
    list_presets(presets_dir).into_iter().find(|p| p.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use jeuvoice_core::transcript::Word;
    use jeuvoice_engine_client::{FakeEngineApi, JobId, JobStatus};
    use jeuvoice_whisper::FakeTranscriber;

    fn wav(dir: &Path, name: &str, secs: u64) -> PathBuf {
        let path = dir.join(name);
        let spec = hound::WavSpec { channels: 1, sample_rate: 16_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for k in 0..secs * 16_000 {
            let s = (k as f32 * 440.0 * std::f32::consts::TAU / 16_000.0).sin() * 0.3;
            w.write_sample((s * i16::MAX as f32) as i16).unwrap();
        }
        w.finalize().unwrap();
        path
    }

    fn heard(words: &str) -> jeuvoice_core::transcript::Transcript {
        jeuvoice_core::transcript::Transcript {
            language: "tr".into(),
            words: words
                .split_whitespace()
                .enumerate()
                .map(|(i, w)| Word { text: w.into(), start: i as f64, end: i as f64 + 0.5, prob: 1.0 })
                .collect(),
        }
    }

    /// Records the seeds and clip paths of the renders it sees.
    struct Spy {
        inner: FakeEngineApi,
        seen: Mutex<Vec<(u64, Option<String>)>>,
    }

    impl EngineApi for Spy {
        fn health(&self) -> Result<jeuvoice_engine_client::Health, jeuvoice_engine_client::EngineError> {
            self.inner.health()
        }
        fn load(&self, m: &str, a: Option<&str>, d: &str) -> Result<JobId, jeuvoice_engine_client::EngineError> {
            self.inner.load(m, a, d)
        }
        fn unload(&self) -> Result<(), jeuvoice_engine_client::EngineError> {
            self.inner.unload()
        }
        fn render(&self, req: &RenderRequest) -> Result<JobId, jeuvoice_engine_client::EngineError> {
            self.seen.lock().unwrap().push((req.base_seed, req.preset.ref_audio.clone()));
            self.inner.render(req)
        }
        fn job(&self, id: &JobId) -> Result<JobStatus, jeuvoice_engine_client::EngineError> {
            self.inner.job(id)
        }
        fn cancel(&self, id: &JobId) -> Result<(), jeuvoice_engine_client::EngineError> {
            self.inner.cancel(id)
        }
        fn fetch_adapter(&self, r: &str, s: Option<&str>, t: Option<&str>) -> Result<PathBuf, jeuvoice_engine_client::EngineError> {
            self.inner.fetch_adapter(r, s, t)
        }
    }

    fn editor(spy: Arc<Spy>) -> PresetEditor {
        PresetEditor::new(Arc::new(FakeTranscriber(heard("selam dünya"))), spy as Arc<dyn EngineApi>)
    }

    #[test]
    fn short_clip_gives_the_length_message() {
        let tmp = tempfile::tempdir().unwrap();
        let spy = Arc::new(Spy { inner: FakeEngineApi::new(), seen: Mutex::new(vec![]) });
        let mut ed = editor(spy);
        ed.check_clip(&wav(tmp.path(), "kisa.wav", 2), "tr");
        assert_eq!(ed.clip_error.as_deref(), Some("Use a clip between 3 and 10 seconds."));
        assert_eq!(ed.clip, None);
    }

    #[test]
    fn checked_clip_fills_the_text_field() {
        let tmp = tempfile::tempdir().unwrap();
        let spy = Arc::new(Spy { inner: FakeEngineApi::new(), seen: Mutex::new(vec![]) });
        let mut ed = editor(spy);
        let clip = wav(tmp.path(), "ses.wav", 5);
        ed.check_clip(&clip, "tr");
        assert_eq!(ed.clip_error, None);
        assert_eq!(ed.clip, Some(clip));
        assert_eq!(ed.clip_text, "selam dünya");
    }

    #[test]
    fn try_three_sends_three_seeds() {
        let tmp = tempfile::tempdir().unwrap();
        let spy = Arc::new(Spy { inner: FakeEngineApi::new(), seen: Mutex::new(vec![]) });
        let mut ed = editor(spy.clone());
        ed.instruct = "warm".into();
        ed.try_three(tmp.path());
        ed.poll();
        assert_eq!(ed.tries.len(), 3);
        assert!(ed.tries.iter().all(|t| t.done && t.file.as_ref().is_some_and(|f| f.exists())));
        let seen = spy.seen.lock().unwrap().clone();
        assert_eq!(seen.iter().map(|(s, _)| *s).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert!(seen.iter().all(|(_, r)| r.is_none()), "seed tries use no clip");
    }

    #[test]
    fn try_clone_renders_with_the_clip() {
        let tmp = tempfile::tempdir().unwrap();
        let spy = Arc::new(Spy { inner: FakeEngineApi::new(), seen: Mutex::new(vec![]) });
        let mut ed = editor(spy.clone());
        let clip = wav(tmp.path(), "ses.wav", 5);
        ed.check_clip(&clip, "tr");
        ed.try_clone(tmp.path());
        ed.poll();
        assert_eq!(ed.tries.len(), 1);
        let seen = spy.seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].1.as_deref(), Some(clip.to_string_lossy().as_ref()));
    }

    #[test]
    fn save_writes_a_preset_that_lists_back() {
        let tmp = tempfile::tempdir().unwrap();
        let presets = tmp.path().join("presets");
        let spy = Arc::new(Spy { inner: FakeEngineApi::new(), seen: Mutex::new(vec![]) });
        let mut ed = editor(spy);
        let clip = wav(tmp.path(), "ses.wav", 5);
        ed.check_clip(&clip, "tr");
        ed.name = "benim".into();
        let saved = ed.save(&presets).unwrap();
        assert_eq!(saved.seed, 1);
        assert_eq!(saved.ref_audio.as_deref(), Some("ref.wav"));
        let back = saved_preset(&presets, "benim").expect("lists back");
        assert_eq!(back, saved);
        assert!(presets.join("benim").join("ref.wav").exists());

        // A seed preset keeps the picked try's seed and the typed attributes.
        let mut ed = editor(Arc::new(Spy { inner: FakeEngineApi::new(), seen: Mutex::new(vec![]) }));
        ed.kind = PresetKind::Seed;
        ed.instruct = "young".into();
        ed.name = "tohum".into();
        ed.try_three(tmp.path());
        ed.poll();
        ed.picked = Some(2);
        let saved = ed.save(&presets).unwrap();
        assert_eq!((saved.seed, saved.instruct.as_deref()), (3, Some("young")));
        assert_eq!(saved.ref_audio, None);
    }
}
