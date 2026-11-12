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
use jeuvoice_core::preset::{ModelRef, Preset, PresetKind, list_presets, save_preset};
use jeuvoice_engine_client::{EngineApi, JobState, Preset as WirePreset, RenderRequest};
use jeuvoice_whisper::{TranscribeOptions, Transcriber};

use crate::settings::{Settings, TokenStore, default_terms};
use crate::voices::traits::VoiceTraits;

/// The fixed test sentence every try renders.
pub const TRY_SENTENCE: &str = "Bugün size kısa bir hikâye anlatacağım.";

/// One try render.
#[derive(Clone, Debug)]
pub struct TryVoice {
    pub seed: u64,
    pub job: String,
    pub file: Option<PathBuf>,
    pub done: bool,
    /// Job reads that failed in a row. Many mean the engine stopped.
    pub read_errors: u32,
}

/// After this many failed job reads in a row, a try ends with an error.
const MAX_READ_ERRORS: u32 = 20;

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
    /// The attributes for a described voice (OmniVoice voice design).
    pub traits: VoiceTraits,
    pub guidance: f32,
    pub speed: f32,
    /// Semitones to move the voice after the render: negative is lower.
    pub pitch_shift: f32,
    /// The token sampling temperature: 0 is the likeliest sound.
    pub variety: f32,
    /// Model steps: more is slower and cleaner.
    pub num_step: u32,
    /// Silence between sentences and between paragraphs, in seconds.
    pub pause: f32,
    pub para_pause: f32,
    /// Use the base model without the tr2 adapter. It follows the voice attributes
    /// better, but speaks Turkish less well.
    pub use_base_model: bool,
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
            traits: VoiceTraits::default(),
            guidance: 2.5,
            speed: 1.0,
            pitch_shift: 0.0,
            variety: 0.0,
            num_step: 32,
            pause: 0.1,
            para_pause: 0.3,
            use_base_model: false,
            tries: Vec::new(),
            picked: None,
            status: None,
        }
    }

    /// Use `engine` for the next tries (the real engine, once it runs).
    pub fn set_engine(&mut self, engine: Arc<dyn EngineApi>) {
        self.engine = engine;
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
                self.clip_error = Some(crate::core_text::audio_error(&e));
                return;
            }
        };
        let secs = audio.samples.len() as f64 / f64::from(audio.rate.max(1));
        if !(3.0..=10.0).contains(&secs) {
            self.clip_error = Some(crate::i18n::tr!("Use a clip between 3 and 10 seconds.", "3 ile 10 saniye arasında bir klip kullanın.").into());
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
            self.status = Some(crate::i18n::tr!("Drop a voice clip first.", "Önce bir ses klibi bırakın.").into());
            return;
        };
        let mut preset = self.wire_preset();
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
        self.try_seeds(out_dir, self.base_seed);
    }

    /// Three more tries with the next three seeds. The tries so far stay.
    /// Blocks: call on a worker.
    pub fn try_more(&mut self, out_dir: &Path) {
        let next = self.tries.iter().map(|t| t.seed + 1).max().unwrap_or(self.base_seed);
        self.try_seeds(out_dir, next);
    }

    fn try_seeds(&mut self, out_dir: &Path, first: u64) {
        for k in 0..3 {
            let mut preset = self.wire_preset();
            preset.instruct = self.traits.instruct();
            self.start_try(out_dir, first + k, preset, None);
        }
    }

    /// The render settings every try shares.
    fn wire_preset(&self) -> WirePreset {
        WirePreset {
            guidance: self.guidance,
            speed: self.speed,
            pitch_shift: self.pitch_shift,
            variety: self.variety,
            num_step: self.num_step,
            pause: f64::from(self.pause),
            para_pause: f64::from(self.para_pause),
            ..WirePreset::default()
        }
    }

    /// Why `name` cannot be a new voice name, in plain words, or `None` when it can.
    /// Refuses an empty name, a taken name in any letter case (`ses_0007` counts
    /// as Sami), and names that Windows refuses as a folder name.
    pub fn name_problem(name: &str, presets_dir: &Path) -> Option<String> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Some(crate::i18n::tr!("Give the voice a name first.", "Önce sese bir ad verin.").into());
        }
        let bad_char = name.chars().any(|c| c.is_control() || r#"<>:"/\|?*"#.contains(c));
        let bad_end = name.ends_with('.') || name.ends_with(' ');
        let stem = trimmed.split('.').next().unwrap_or("").to_ascii_uppercase();
        let reserved = ["CON", "PRN", "AUX", "NUL"].contains(&stem.as_str())
            || ((stem.starts_with("COM") || stem.starts_with("LPT"))
                && stem.len() == 4
                && stem.as_bytes()[3].is_ascii_digit()
                && stem.as_bytes()[3] != b'0');
        if bad_char || bad_end || reserved {
            return Some(crate::i18n::tr!(
                r#"Use letters, digits and spaces in the name. Do not use < > : " / \ | ? * and do not end it with a dot."#,
                r#"Adda harf, rakam ve boşluk kullanın. < > : " / \ | ? * kullanmayın ve adı nokta ile bitirmeyin."#
            )
            .into());
        }
        let wanted = jeuvoice_core::preset::canonical_preset_name(trimmed).to_lowercase();
        if list_presets(presets_dir).iter().any(|p| p.name.to_lowercase() == wanted) {
            return Some(crate::i18n::trf!("A voice named {trimmed} exists. Pick another name.", "{trimmed} adında bir ses var. Başka bir ad seçin."));
        }
        None
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
            Ok(job) => self.tries.push(TryVoice { seed, job, file: None, done: false, read_errors: 0 }),
            Err(e) => self.status = Some(e.to_string()),
        }
    }

    /// Advance try renders. Call on the UI thread.
    pub fn poll(&mut self) {
        for t in &mut self.tries {
            if t.done {
                continue;
            }
            let Ok(status) = self.engine.job(&t.job) else {
                t.read_errors += 1;
                if t.read_errors >= MAX_READ_ERRORS {
                    t.done = true;
                    self.status = Some(crate::i18n::tr!(
                        "The voice engine stopped answering. Restart it in Settings.",
                        "Ses motoru cevap vermiyor. Ayarlar'dan yeniden başlatın."
                    )
                    .into());
                }
                continue;
            };
            t.read_errors = 0;
            match status.state {
                JobState::Done => {
                    t.done = true;
                    t.file = status.render_result().and_then(|r| r.parts.into_iter().next()).map(|p| PathBuf::from(p.file));
                }
                JobState::Failed => {
                    t.done = true;
                    self.status = Some(status.error.unwrap_or_else(|| crate::i18n::tr!("The try failed. Try again.", "Deneme başarısız oldu. Tekrar deneyin.").into()));
                }
                _ => {}
            }
        }
    }

    /// Save the preset: the picked seed (or the base seed), the clip for clones.
    pub fn save(&mut self, presets_dir: &Path, model: ModelRef) -> Result<Preset, String> {
        if let Some(problem) = Self::name_problem(&self.name, presets_dir) {
            return Err(problem);
        }
        let name = self.name.trim().to_string();
        let seed = self.picked.and_then(|i| self.tries.get(i)).map(|t| t.seed).unwrap_or(self.base_seed);
        let (ref_audio, ref_wav) = match self.kind {
            PresetKind::Clone => {
                let Some(clip) = self.clip.clone() else { return Err(crate::i18n::tr!("Drop a voice clip first.", "Önce bir ses klibi bırakın.").into()) };
                if self.clip_text.trim().is_empty() {
                    return Err(crate::i18n::tr!("Check the clip text first.", "Önce klip metnini denetleyin.").into());
                }
                (Some("ref.wav".to_string()), Some(clip))
            }
            PresetKind::Seed => (None, None),
        };
        let instruct = match self.kind {
            PresetKind::Seed => self.traits.instruct(),
            PresetKind::Clone => None,
        };
        let preset = Preset {
            name: name.clone(),
            kind: self.kind.clone(),
            ref_audio,
            ref_text: if self.kind == PresetKind::Clone { Some(self.clip_text.clone()) } else { None },
            seed,
            instruct,
            model,
            guidance: self.guidance,
            pause: self.pause,
            para_pause: self.para_pause,
            num_step: self.num_step,
            speed: self.speed,
            pitch_shift: self.pitch_shift,
            variety: self.variety,
        };
        save_preset(presets_dir, &preset, ref_wav.as_deref()).map_err(|e| crate::i18n::trf!("The preset could not be saved: {e}", "Hazır ayar kaydedilemedi: {e}"))?;
        self.status = Some(crate::i18n::trf!("Saved preset {name}.", "{name} hazır ayarı kaydedildi."));
        Ok(preset)
    }
}

/// Load `model` (base + adapter) in the engine and wait for it. Blocks: call on a worker.
pub(crate) fn load_model(api: &Arc<dyn EngineApi>, model: &ModelRef, settings: &Settings, store: &dyn TokenStore) -> Result<(), String> {
    let preset = Preset { model: model.clone(), ..jeuvoice_core::preset::default_preset() };
    let adapter = crate::voice::state::resolved_adapter(api, &preset, settings, store)?;
    let job = api.load(&model.base, adapter.as_deref(), "auto").map_err(|e| e.to_string())?;
    if crate::voice::state::wait_job(api, &job, std::time::Duration::from_secs(600)) {
        Ok(())
    } else {
        Err(crate::i18n::tr!("The voice model could not be loaded.", "Ses modeli yüklenemedi.").into())
    }
}

/// Read back what the editor wrote, for tests.
pub fn saved_preset(presets_dir: &Path, name: &str) -> Option<Preset> {
    list_presets(presets_dir).into_iter().find(|p| p.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jeuvoice_core::preset::AdapterRef;
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
        ed.traits.whisper = true;
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
        let saved = ed.save(&presets, sami_model()).unwrap();
        assert_eq!(saved.seed, 1);
        assert_eq!(saved.ref_audio.as_deref(), Some("ref.wav"));
        let back = saved_preset(&presets, "benim").expect("lists back");
        assert_eq!(back, saved);
        assert!(presets.join("benim").join("ref.wav").exists());

        // A seed preset keeps the picked try's seed and the typed attributes.
        let mut ed = editor(Arc::new(Spy { inner: FakeEngineApi::new(), seen: Mutex::new(vec![]) }));
        ed.kind = PresetKind::Seed;
        ed.traits.age = Some(crate::voices::traits::Age::Teenager);
        ed.name = "tohum".into();
        ed.try_three(tmp.path());
        ed.poll();
        ed.picked = Some(2);
        let saved = ed.save(&presets, sami_model()).unwrap();
        assert_eq!((saved.seed, saved.instruct.as_deref()), (3, Some("teenager")));
        assert_eq!(saved.ref_audio, None);
    }

    fn sami_model() -> ModelRef {
        jeuvoice_core::preset::default_preset().model
    }

    fn spy() -> Arc<Spy> {
        Arc::new(Spy { inner: FakeEngineApi::new(), seen: Mutex::new(vec![]) })
    }

    #[test]
    fn save_stores_the_given_model_and_settings() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ed = editor(spy());
        ed.kind = PresetKind::Seed;
        ed.guidance = 3.0;
        ed.speed = 1.1;
        ed.try_three(tmp.path());
        ed.poll();
        ed.picked = Some(1);
        ed.name = "Ece".into();
        let saved = ed.save(&tmp.path().join("presets"), sami_model()).unwrap();
        assert_eq!(saved.model, sami_model());
        assert_eq!((saved.guidance, saved.speed, saved.seed), (3.0, 1.1, ed.base_seed + 1));
    }

    #[test]
    fn described_voice_saves_its_instruct() {
        use crate::voices::traits::{Age, Gender};
        let tmp = tempfile::tempdir().unwrap();
        let mut ed = editor(spy());
        ed.kind = PresetKind::Seed;
        ed.traits.gender = Some(Gender::Female);
        ed.traits.age = Some(Age::Elderly);
        ed.name = "Nine".into();
        let saved = ed.save(&tmp.path().join("presets"), sami_model()).unwrap();
        assert_eq!(saved.instruct.as_deref(), Some("female, elderly"));
    }

    #[test]
    fn try_more_adds_the_next_seeds() {
        let tmp = tempfile::tempdir().unwrap();
        let spy = spy();
        let mut ed = editor(spy.clone());
        ed.try_three(tmp.path());
        ed.try_more(tmp.path());
        ed.poll();
        assert_eq!(ed.tries.iter().map(|t| t.seed).collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(spy.seen.lock().unwrap().len(), 6);
    }

    #[test]
    fn names_windows_refuses_are_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let presets = tmp.path().join("presets");
        for bad in ["a/b", "x:y", "CON", "com3", "Ece.", "", "   ", r"a\b", "lpt1"] {
            assert!(PresetEditor::name_problem(bad, &presets).is_some(), "{bad:?} must be refused");
        }
        assert!(PresetEditor::name_problem("Ece Nur", &presets).is_none());
        let mut ed = editor(spy());
        ed.kind = PresetKind::Seed;
        ed.name = "a/b".into();
        assert!(ed.save(&presets, sami_model()).is_err());
        assert!(list_presets(&presets).is_empty(), "nothing is written");
    }

    #[test]
    fn a_taken_name_is_an_error_in_any_case() {
        let tmp = tempfile::tempdir().unwrap();
        let presets = tmp.path().join("presets");
        jeuvoice_core::preset::save_preset(&presets, &jeuvoice_core::preset::default_preset(), None).unwrap();
        assert!(PresetEditor::name_problem("sami", &presets).is_some());
        assert!(PresetEditor::name_problem("ses_0007", &presets).is_some());
        assert!(PresetEditor::name_problem("Ece", &presets).is_none());
    }

    /// Records the load calls.
    struct LoadSpy {
        inner: FakeEngineApi,
        loads: Mutex<Vec<(String, Option<String>, String)>>,
    }

    impl EngineApi for LoadSpy {
        fn health(&self) -> Result<jeuvoice_engine_client::Health, jeuvoice_engine_client::EngineError> {
            self.inner.health()
        }
        fn load(&self, m: &str, a: Option<&str>, d: &str) -> Result<JobId, jeuvoice_engine_client::EngineError> {
            self.loads.lock().unwrap().push((m.into(), a.map(String::from), d.into()));
            self.inner.load(m, a, d)
        }
        fn unload(&self) -> Result<(), jeuvoice_engine_client::EngineError> {
            self.inner.unload()
        }
        fn render(&self, req: &RenderRequest) -> Result<JobId, jeuvoice_engine_client::EngineError> {
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

    #[test]
    fn load_model_loads_base_and_adapter() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("adapter");
        std::fs::create_dir_all(&dir).unwrap();
        let spy = Arc::new(LoadSpy { inner: FakeEngineApi::new(), loads: Mutex::new(vec![]) });
        let api: Arc<dyn EngineApi> = spy.clone();
        let model = ModelRef { base: "k2-fsa/OmniVoice".into(), adapter: AdapterRef::Local { dir: dir.to_string_lossy().into_owned() } };
        let store = crate::settings::MemStore::default();
        load_model(&api, &model, &crate::settings::Settings::default(), &store).unwrap();
        assert_eq!(
            spy.loads.lock().unwrap().clone(),
            vec![("k2-fsa/OmniVoice".to_string(), Some(dir.to_string_lossy().into_owned()), "auto".to_string())]
        );
    }
}
