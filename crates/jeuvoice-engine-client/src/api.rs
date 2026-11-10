//! Engine HTTP API: serde types mirroring `engine/jeuvoice_engine/server.py` + `jobs.py`.

use std::collections::HashMap;
use std::sync::Mutex;

use jeuvoice_core::terms::Terms;
use serde::{Deserialize, Serialize};

/// Plain errors the UI can show as is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// The engine did not start (or the first call failed).
    StartFailed,
    /// The engine process is gone.
    Stopped,
    /// The engine refused the token or the call.
    Refused,
    /// Anything else, in plain words (the server's `error` field, never a traceback).
    Failed(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StartFailed => write!(f, "The voice engine did not start. Open Settings and click Restart engine."),
            Self::Stopped => write!(f, "The voice engine stopped. Click Restart engine."),
            Self::Refused => write!(f, "The voice engine refused the connection. Restart the engine and try again."),
            Self::Failed(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for EngineError {}

pub type JobId = String;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Waiting,
    Running,
    Done,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Health {
    pub ok: bool,
    pub backend: String,
    pub device: String,
    pub vram_total_gb: Option<f64>,
    pub model_loaded: bool,
    pub version: String,
}

/// Engine render settings. Unknown fields (preset name, kind, model ref) pass through
/// untouched so the preset editor can grow without changing this crate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Preset {
    pub pause: f64,
    pub para_pause: f64,
    pub guidance: f32,
    pub num_step: u32,
    pub speed: f32,
    pub instruct: Option<String>,
    pub ref_audio: Option<String>,
    pub ref_text: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for Preset {
    fn default() -> Self {
        Self {
            pause: 0.1,
            para_pause: 0.3,
            guidance: 2.5,
            num_step: 32,
            speed: 1.0,
            instruct: None,
            ref_audio: None,
            ref_text: None,
            extra: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PartOverride {
    pub seed: Option<u64>,
    pub guidance: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RenderRequest {
    pub script: String,
    pub paragraphs: Option<Vec<usize>>,
    pub preset: Preset,
    pub terms: Terms,
    pub base_seed: u64,
    pub part_overrides: HashMap<String, PartOverride>,
    pub out_dir: std::path::PathBuf,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartResult {
    pub index: usize,
    pub paragraph: usize,
    pub language: String,
    pub text: String,
    pub seed: u64,
    pub guidance: f32,
    pub file: String,
    pub audio_s: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RenderResult {
    pub parts: Vec<PartResult>,
    pub pause: f64,
    pub para_pause: f64,
}

/// One cleaned or fetched training clip, as `/prep/*` jobs report it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrepClip {
    pub id: String,
    pub audio: String,
    pub duration_s: f32,
    pub band_khz: f32,
    /// Empty when the source has no text column: Whisper makes the text.
    pub text: String,
    /// Plain reason when the clip is left out (`None` keeps the clip).
    pub excluded: Option<String>,
}

/// What `/prep/hf` needs: which dataset, which columns, where to write the WAVs.
#[derive(Clone, Debug, PartialEq)]
pub struct HfPrep {
    pub repo: String,
    pub config: Option<String>,
    pub split: String,
    pub audio_col: String,
    pub text_col: Option<String>,
    pub limit: Option<u32>,
    pub token: Option<String>,
    pub out_dir: std::path::PathBuf,
}

/// The clips of a finished `/prep/clean` or `/prep/hf` job. `None` when the job
/// has no clip list (wrong job or engine too old).
pub fn parse_prep_clips(status: &JobStatus) -> Option<Vec<PrepClip>> {
    status.result.clone().and_then(|v| v.get("clips").cloned()).and_then(|v| serde_json::from_value(v).ok())
}

/// The manifest counts of a finished `/prep/manifest` job: `(train, dev)` rows.
pub fn parse_prep_manifest(status: &JobStatus) -> Option<(usize, usize)> {
    let v = status.result.clone()?;
    Some((v.get("train")?.as_u64()? as usize, v.get("dev")?.as_u64()? as usize))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JobStatus {
    pub state: JobState,
    pub progress: f64,
    pub message: String,
    /// The finished job's payload: render jobs carry a [`RenderResult`], load jobs a
    /// `{"model", "device"}` object. Kept generic so one shape never breaks the other.
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
}

impl JobStatus {
    /// The render payload, if this finished job was a render.
    pub fn render_result(&self) -> Option<RenderResult> {
        self.result.clone().and_then(|v| serde_json::from_value(v).ok())
    }
}

/// The engine API. All methods are blocking; call them on a worker.
pub trait EngineApi: Send + Sync {
    fn health(&self) -> Result<Health, EngineError>;
    fn load(&self, model: &str, adapter_dir: Option<&str>, device: &str) -> Result<JobId, EngineError>;
    fn unload(&self) -> Result<(), EngineError>;
    fn render(&self, req: &RenderRequest) -> Result<JobId, EngineError>;
    fn job(&self, id: &JobId) -> Result<JobStatus, EngineError>;
    fn cancel(&self, id: &JobId) -> Result<(), EngineError>;
    /// Download a Hugging Face adapter into the engine data folder.
    fn fetch_adapter(&self, repo: &str, subfolder: Option<&str>, token: Option<&str>) -> Result<std::path::PathBuf, EngineError>;
    /// Clean local clips (`/prep/clean`). Default: the engine has no data prep.
    fn prep_clean(&self, _clips: &[(String, String)], _out_dir: &std::path::Path) -> Result<JobId, EngineError> {
        Err(EngineError::Failed("This engine cannot prepare training data.".into()))
    }
    /// Fetch a Hugging Face dataset (`/prep/hf`). Default: the engine has no data prep.
    fn prep_hf(&self, _req: &HfPrep) -> Result<JobId, EngineError> {
        Err(EngineError::Failed("This engine cannot prepare training data.".into()))
    }
    /// Write `train.jsonl`/`dev.jsonl` (`/prep/manifest`). Default: no data prep.
    fn prep_manifest(
        &self,
        _clips: &[PrepClip],
        _weights: &[(String, u32)],
        _dev_ratio: f64,
        _out_dir: &std::path::Path,
    ) -> Result<JobId, EngineError> {
        Err(EngineError::Failed("This engine cannot prepare training data.".into()))
    }
}

/// HTTP engine client over `ureq`.
#[derive(Clone)]
pub struct HttpEngine {
    base: String,
    token: String,
}

impl HttpEngine {
    pub fn new(port: u16, token: impl Into<String>) -> Self {
        Self { base: format!("http://127.0.0.1:{port}"), token: token.into() }
    }

    fn get<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, EngineError> {
        let mut res = ureq::get(format!("{}{path}", self.base))
            .header("X-JeuVoice-Token", self.token.as_str())
            .call()
            .map_err(map_ureq)?;
        res.body_mut().read_json().map_err(|_| EngineError::Failed("The voice engine gave a bad answer.".into()))
    }

    fn post<B: Serialize, T: serde::de::DeserializeOwned>(&self, path: &str, body: &B) -> Result<T, EngineError> {
        let mut res = ureq::post(format!("{}{path}", self.base))
            .header("X-JeuVoice-Token", self.token.as_str())
            .header("Content-Type", "application/json")
            .send_json(body)
            .map_err(map_ureq)?;
        res.body_mut().read_json().map_err(|_| EngineError::Failed("The voice engine gave a bad answer.".into()))
    }

    /// Ask the engine process to stop (used by `EngineProcess::stop`).
    pub fn shutdown(&self) -> Result<(), EngineError> {
        #[derive(Deserialize)]
        struct Ok {
            #[allow(dead_code)]
            ok: bool,
        }
        let body = serde_json::json!({});
        // The server answers, then stops: a closed connection still counts as stopped.
        match self.post::<_, Ok>("/shutdown", &body) {
            Ok(_) | Err(EngineError::Stopped) => Ok(()),
            Err(e) => Err(e),
        }
    }
}

fn map_ureq(e: ureq::Error) -> EngineError {
    match e {
        ureq::Error::StatusCode(401) => EngineError::Refused,
        ureq::Error::StatusCode(_) => EngineError::Failed("The voice engine refused the call.".into()),
        // No connection: the process is gone.
        _ => EngineError::Stopped,
    }
}

#[derive(Deserialize)]
struct JobCreated {
    job: JobId,
}

impl EngineApi for HttpEngine {
    fn health(&self) -> Result<Health, EngineError> {
        self.get("/health")
    }

    fn load(&self, model: &str, adapter_dir: Option<&str>, device: &str) -> Result<JobId, EngineError> {
        let body = serde_json::json!({"model": model, "adapter_dir": adapter_dir, "device": device});
        self.post::<_, JobCreated>("/load", &body).map(|r| r.job)
    }

    fn unload(&self) -> Result<(), EngineError> {
        #[derive(Deserialize)]
        struct Ok {
            #[allow(dead_code)]
            ok: bool,
        }
        let body = serde_json::json!({});
        self.post::<_, Ok>("/unload", &body).map(|_| ())
    }

    fn render(&self, req: &RenderRequest) -> Result<JobId, EngineError> {
        self.post::<_, JobCreated>("/render", req).map(|r| r.job)
    }

    fn job(&self, id: &JobId) -> Result<JobStatus, EngineError> {
        let mut res = ureq::get(format!("{}/jobs/{id}", self.base))
            .header("X-JeuVoice-Token", self.token.as_str())
            .call()
            .map_err(map_ureq)?;
        res.body_mut()
            .read_json()
            .map_err(|_| EngineError::Failed("The voice engine gave a bad answer.".into()))
    }

    fn cancel(&self, id: &JobId) -> Result<(), EngineError> {
        #[derive(Deserialize)]
        struct Ok {
            #[allow(dead_code)]
            ok: bool,
        }
        let body = serde_json::json!({});
        self.post::<_, Ok>(&format!("/jobs/{id}/cancel"), &body).map(|_| ())
    }

    fn fetch_adapter(&self, repo: &str, subfolder: Option<&str>, token: Option<&str>) -> Result<std::path::PathBuf, EngineError> {
        #[derive(Deserialize)]
        struct Reply {
            ok: bool,
            path: Option<String>,
            error: Option<String>,
        }
        let body = serde_json::json!({"repo": repo, "subfolder": subfolder, "token": token});
        let reply: Reply = self.post("/fetch_adapter", &body)?;
        if reply.ok {
            reply.path.map(std::path::PathBuf::from).ok_or_else(|| EngineError::Failed("The voice engine gave a bad answer.".into()))
        } else {
            Err(EngineError::Failed(reply.error.unwrap_or_else(|| "The model files could not be downloaded.".into())))
        }
    }

    fn prep_clean(&self, clips: &[(String, String)], out_dir: &std::path::Path) -> Result<JobId, EngineError> {
        let body = serde_json::json!({
            "clips": clips.iter().map(|(id, audio)| serde_json::json!({"id": id, "audio": audio})).collect::<Vec<_>>(),
            "out_dir": out_dir,
        });
        self.post::<_, JobCreated>("/prep/clean", &body).map(|r| r.job)
    }

    fn prep_hf(&self, req: &HfPrep) -> Result<JobId, EngineError> {
        let body = serde_json::json!({
            "repo": req.repo,
            "config": req.config,
            "split": req.split,
            "audio_col": req.audio_col,
            "text_col": req.text_col,
            "limit": req.limit,
            "token": req.token,
            "out_dir": req.out_dir,
        });
        self.post::<_, JobCreated>("/prep/hf", &body).map(|r| r.job)
    }

    fn prep_manifest(
        &self,
        clips: &[PrepClip],
        weights: &[(String, u32)],
        dev_ratio: f64,
        out_dir: &std::path::Path,
    ) -> Result<JobId, EngineError> {
        let body = serde_json::json!({
            "clips": clips,
            "sources": weights.iter().map(|(name, weight)| serde_json::json!({"name": name, "weight": weight})).collect::<Vec<_>>(),
            "dev_ratio": dev_ratio,
            "out_dir": out_dir,
        });
        self.post::<_, JobCreated>("/prep/manifest", &body).map(|r| r.job)
    }
}

/// In-memory engine for app tests: renders are silent WAVs, done at once.
pub struct FakeEngineApi {
    jobs: Mutex<Vec<JobStatus>>,
}

impl FakeEngineApi {
    pub fn new() -> Self {
        Self { jobs: Mutex::new(Vec::new()) }
    }

    /// Push an already-done job carrying `result`.
    fn done_job(&self, result: serde_json::Value) -> JobId {
        let mut jobs = self.jobs.lock().unwrap();
        let id = format!("job-{}", jobs.len() + 1);
        jobs.push(JobStatus { state: JobState::Done, progress: 1.0, message: String::new(), result: Some(result), error: None });
        id
    }
}

impl Default for FakeEngineApi {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineApi for FakeEngineApi {
    fn health(&self) -> Result<Health, EngineError> {
        Ok(Health {
            ok: true,
            backend: "fake".into(),
            device: "none".into(),
            vram_total_gb: None,
            model_loaded: false,
            version: "test".into(),
        })
    }

    fn load(&self, _model: &str, _adapter_dir: Option<&str>, _device: &str) -> Result<JobId, EngineError> {
        let id = format!("job-{}", self.jobs.lock().unwrap().len() + 1);
        self.jobs.lock().unwrap().push(JobStatus {
            state: JobState::Done,
            progress: 1.0,
            message: String::new(),
            result: None,
            error: None,
        });
        Ok(id)
    }

    fn unload(&self) -> Result<(), EngineError> {
        Ok(())
    }

    fn render(&self, req: &RenderRequest) -> Result<JobId, EngineError> {
        let paragraphs: Vec<&str> =
            req.script.split("\n\n").map(str::trim).filter(|p| !p.is_empty()).collect();
        let wanted = req.paragraphs.clone();
        let mut parts = Vec::new();
        let mut index = 0;
        // Duration varies with the seed, like real renders vary with their seeds.
        let secs = 1.0 + (req.base_seed % 3) as f32 * 0.5;
        for (para, text) in paragraphs.iter().enumerate() {
            if let Some(w) = &wanted
                && !w.contains(&para)
            {
                continue;
            }
            index += 1;
            let samples = vec![0.0f32; (24_000.0 * secs) as usize];
            let file = req.out_dir.join(format!("part_{index}.wav"));
            std::fs::create_dir_all(&req.out_dir).map_err(|_| EngineError::Failed("A file could not be saved.".into()))?;
            write_wav(&file, &samples).map_err(|_| EngineError::Failed("A file could not be saved.".into()))?;
            parts.push(PartResult {
                index,
                paragraph: para,
                language: "tr".into(),
                text: (*text).to_string(),
                seed: req.base_seed,
                guidance: req.preset.guidance,
                file: file.to_string_lossy().into_owned(),
                audio_s: secs,
            });
        }
        let id = format!("job-{}", self.jobs.lock().unwrap().len() + 1);
        let result = RenderResult { parts, pause: req.preset.pause, para_pause: req.preset.para_pause };
        self.jobs.lock().unwrap().push(JobStatus {
            state: JobState::Done,
            progress: 1.0,
            message: String::new(),
            result: Some(serde_json::to_value(result).expect("a render result serializes")),
            error: None,
        });
        Ok(id)
    }

    fn job(&self, id: &JobId) -> Result<JobStatus, EngineError> {
        let n: usize = id.strip_prefix("job-").and_then(|n| n.parse().ok()).unwrap_or(0);
        self.jobs.lock().unwrap().get(n.wrapping_sub(1)).cloned().ok_or(EngineError::Failed("No such job.".into()))
    }

    fn cancel(&self, _id: &JobId) -> Result<(), EngineError> {
        Ok(())
    }

    fn fetch_adapter(&self, repo: &str, subfolder: Option<&str>, _token: Option<&str>) -> Result<std::path::PathBuf, EngineError> {
        let dir = std::env::temp_dir().join(format!("fake-adapter-{}", repo.replace('/', "__")));
        let dir = match subfolder {
            Some(s) => dir.join(s),
            None => dir,
        };
        std::fs::create_dir_all(&dir).map_err(|_| EngineError::Failed("A file could not be saved.".into()))?;
        Ok(dir)
    }

    /// Cleaned at once: readable WAVs keep their duration with band 18 kHz,
    /// unreadable ones are excluded with the engine's plain reason.
    fn prep_clean(&self, clips: &[(String, String)], out_dir: &std::path::Path) -> Result<JobId, EngineError> {
        std::fs::create_dir_all(out_dir).map_err(|_| EngineError::Failed("A file could not be saved.".into()))?;
        let cleaned: Vec<PrepClip> = clips
            .iter()
            .map(|(id, audio)| match hound::WavReader::open(audio) {
                Ok(r) => {
                    let spec = r.spec();
                    let secs = r.duration() as f32 / spec.sample_rate as f32;
                    PrepClip {
                        id: id.clone(),
                        audio: audio.clone(),
                        duration_s: secs,
                        band_khz: 18.0,
                        text: String::new(),
                        excluded: None,
                    }
                }
                Err(_) => PrepClip {
                    id: id.clone(),
                    audio: String::new(),
                    duration_s: 0.0,
                    band_khz: 0.0,
                    text: String::new(),
                    excluded: Some("The clip could not be read.".into()),
                },
            })
            .collect();
        Ok(self.done_job(serde_json::json!({"clips": cleaned})))
    }

    /// Fetched at once: writes `limit` (default 2) short tone WAVs. Clips carry
    /// text only when the form named a text column — otherwise Whisper runs.
    fn prep_hf(&self, req: &HfPrep) -> Result<JobId, EngineError> {
        std::fs::create_dir_all(&req.out_dir).map_err(|_| EngineError::Failed("A file could not be saved.".into()))?;
        let n = req.limit.unwrap_or(2).max(1) as usize;
        let text = req.text_col.as_ref().map(|_| "örnek metin".to_string()).unwrap_or_default();
        let mut clips = Vec::with_capacity(n);
        for i in 0..n {
            let id = format!("hf-{:05}", i + 1);
            let file = req.out_dir.join(format!("{id}.wav"));
            write_tone(&file, 1.0).map_err(|_| EngineError::Failed("A file could not be saved.".into()))?;
            clips.push(PrepClip {
                id,
                audio: file.to_string_lossy().into_owned(),
                duration_s: 1.0,
                band_khz: 18.0,
                text: text.clone(),
                excluded: None,
            });
        }
        Ok(self.done_job(serde_json::json!({"clips": clips})))
    }

    fn prep_manifest(
        &self,
        clips: &[PrepClip],
        _weights: &[(String, u32)],
        _dev_ratio: f64,
        out_dir: &std::path::Path,
    ) -> Result<JobId, EngineError> {
        std::fs::create_dir_all(out_dir).map_err(|_| EngineError::Failed("A file could not be saved.".into()))?;
        // All rows to train, like a dev_ratio of 0: the point is the files exist.
        let mut train = String::new();
        for c in clips {
            let row = serde_json::json!({
                "id": c.id,
                "audio_path": c.audio,
                "text": c.text,
                "language_id": "tr",
            });
            train.push_str(&serde_json::to_string(&row).map_err(|_| EngineError::Failed("A file could not be saved.".into()))?);
            train.push('\n');
        }
        std::fs::write(out_dir.join("train.jsonl"), train).map_err(|_| EngineError::Failed("A file could not be saved.".into()))?;
        std::fs::write(out_dir.join("dev.jsonl"), "").map_err(|_| EngineError::Failed("A file could not be saved.".into()))?;
        Ok(self.done_job(serde_json::json!({"train": clips.len(), "dev": 0})))
    }
}

fn write_tone(path: &std::path::Path, secs: f32) -> std::io::Result<()> {
    let samples: Vec<f32> =
        (0..(24_000.0 * secs) as usize).map(|k| (k as f32 * 440.0 * std::f32::consts::TAU / 24_000.0).sin() * 0.3).collect();
    write_wav(path, &samples)
}

fn write_wav(path: &std::path::Path, samples: &[f32]) -> std::io::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).map_err(std::io::Error::other)?;
    for s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).map_err(std::io::Error::other)?;
    }
    w.finalize().map_err(std::io::Error::other)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_engine_renders_silent_parts() {
        let api = FakeEngineApi::new();
        let dir = tempfile::tempdir().unwrap();
        let req = RenderRequest {
            script: "Birinci paragraf.\n\nİkinci paragraf.".into(),
            paragraphs: None,
            preset: Preset::default(),
            terms: Terms::default(),
            base_seed: 1,
            part_overrides: HashMap::new(),
            out_dir: dir.path().to_path_buf(),
        };
        let id = api.render(&req).unwrap();
        let status = api.job(&id).unwrap();
        assert_eq!(status.state, JobState::Done);
        let parts = status.render_result().unwrap().parts;
        assert_eq!(parts.len(), 2);
        assert!(std::path::Path::new(&parts[0].file).exists());
    }

    #[test]
    fn preset_keeps_unknown_fields() {
        let mut extra = HashMap::new();
        extra.insert("name".into(), serde_json::json!("ses_0007"));
        let p = Preset { extra, ..Preset::default() };
        let back: Preset = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(back.extra["name"], "ses_0007");
        assert_eq!(back.guidance, 2.5);
    }

    #[test]
    fn prep_results_parse_from_job_status() {
        let clips = vec![PrepClip {
            id: "a".into(),
            audio: "a.wav".into(),
            duration_s: 1.0,
            band_khz: 18.0,
            text: "merhaba".into(),
            excluded: None,
        }];
        let status = JobStatus {
            state: JobState::Done,
            progress: 1.0,
            message: String::new(),
            result: Some(serde_json::json!({"clips": clips})),
            error: None,
        };
        assert_eq!(parse_prep_clips(&status).unwrap()[0].text, "merhaba");
        let manifest = JobStatus { result: Some(serde_json::json!({"train": 8, "dev": 2})), ..status };
        assert_eq!(parse_prep_manifest(&manifest), Some((8, 2)));
    }
}
