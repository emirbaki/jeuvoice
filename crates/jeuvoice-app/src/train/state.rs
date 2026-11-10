//! Train tab, step 1 (data): sources, clips, review and the training project file.
//!
//! Local clips go decode → Whisper ([`SpeechService`] `Samples` jobs) →
//! `/prep/clean`. A Hugging Face source goes `/prep/hf` → Whisper only for clips
//! with empty text → `/prep/clean`. Each stage runs on a worker; results come
//! back through a channel that `poll` drains on the UI thread. The engine client
//! comes from the shared [`EngineService`](crate::services::engine::EngineService);
//! tests inject a fake with [`TrainData::set_engine`].
//!
//! Data lives in a training project folder
//! (`Documents/JeuVoice/Training/<name>/`) with a `train_project.json` holding
//! sources, clips, settings and the run (later tasks).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};

use jeuvoice_core::audio::{self, to_whisper};
use jeuvoice_core::queue::Update;
use jeuvoice_core::train::{Clip, Source, SourceKind, TrainSettings, minutes_by_source};
use jeuvoice_engine_client::{EngineApi, HfPrep, PrepClip, parse_prep_clips};
use jeuvoice_kaggle::{KaggleAuth, KeyStore};
use rune::{TextEdit, Waker};

use crate::services::player::{PlayerOwner, SharedPlayer};
use crate::services::speech::{JobTicket, SpeechJob, SpeechService};
use crate::settings::{HF_TOKEN_KEY, Settings, TokenStore};
use crate::train::run::{KaggleClient, RunEvent, RunRecord, RunState};
use crate::transcribe_tab::MEDIA;

/// Minutes with one decimal, or none past ten. Normalizes -0.0 to 0.0.
pub(crate) fn fmt_minutes(min: f32) -> String {
    let min = min + 0.0;
    if min >= 9.95 { format!("{min:.0}") } else { format!("{min:.1}") }
}

/// What a source row shows while its clips travel the pipeline.
#[derive(Clone, Debug, PartialEq)]
pub enum SourceStatus {
    Working(String),
    Ready,
    Failed(String),
}

/// One training clip with its measured band. The review list shows the band and
/// the excluded reason; `clip.excluded` being `Some` leaves the clip out.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TrainClip {
    #[serde(flatten)]
    pub clip: Clip,
    pub band_khz: f32,
}

/// Where the run trains. Only Kaggle runs in this plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrainTarget {
    ThisPc,
    #[default]
    Kaggle,
    RunPod,
}

/// What `train_project.json` holds.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TrainProject {
    pub name: String,
    pub sources: Vec<Source>,
    pub clips: Vec<TrainClip>,
    pub settings: TrainSettings,
    #[serde(default)]
    pub target: TrainTarget,
}

/// News from the decode, fetch and clean workers.
enum TrainEvent {
    Decoded { source: String, id: String, samples: Arc<Vec<f32>> },
    DecodeFailed { source: String, id: String, reason: String },
    HfDone { source: String, clips: Vec<PrepClip> },
    HfFailed { source: String, reason: String },
    Cleaned { source: String, clips: Vec<PrepClip> },
    CleanFailed { source: String, reason: String },
}

/// `Documents/JeuVoice/Training`, or a temp folder without Documents.
pub fn training_root() -> PathBuf {
    directories::UserDirs::new()
        .and_then(|d| d.document_dir().map(|p| p.join("JeuVoice").join("Training")))
        .unwrap_or_else(std::env::temp_dir)
}

fn default_settings() -> TrainSettings {
    TrainSettings {
        steps: 6000,
        lora_r: 32,
        lora_alpha: 64,
        batch_tokens: 2048,
        save_steps: 1000,
        learning_rate: 0.0001,
        start_from: jeuvoice_core::train::StartFrom::Base,
    }
}

pub struct TrainData {
    pub project: TrainProject,
    /// The Hugging Face form is open.
    pub show_hf: bool,
    pub repo_field: TextEdit,
    pub split_field: TextEdit,
    pub audio_field: TextEdit,
    pub text_field: TextEdit,
    pub limit_field: TextEdit,
    /// The source whose clips the review list shows.
    pub review: Option<String>,
    /// The clip whose text is open for editing.
    pub editing: Option<String>,
    pub edit_field: TextEdit,
    /// The Kaggle key form is open.
    pub show_kaggle_form: bool,
    pub kaggle_user_field: TextEdit,
    pub kaggle_key_field: TextEdit,
    /// The keychain user name, if the user saved a key.
    pub kaggle_user: Option<String>,
    /// The run the tab follows, if one started.
    pub run: Option<RunRecord>,
    /// The latest polled run state.
    pub run_status: Option<jeuvoice_kaggle::KernelStatus>,
    /// The last 30 log lines.
    pub run_log: Vec<String>,
    /// `(step, loss)` points from `train.log` (or the live tail).
    pub losses: Vec<(u32, f32)>,
    /// Checkpoint folders under `runs/<name>/`.
    pub checkpoints: Vec<String>,
    /// Bytes of the dataset upload so far.
    pub upload: Option<(u64, u64)>,
    /// Bytes of the output download so far.
    pub download: Option<(u64, u64)>,
    /// The built bundle (`runs/<name>/<id>.zip`).
    pub bundle_path: Option<PathBuf>,
    /// The "Start on Kaggle" confirm dialog is open.
    pub confirm_start: bool,
    /// The full notebook text is shown.
    pub show_script: bool,
    /// A bundle export waits for the worker.
    pub export_pending: bool,
    /// How often the run worker polls Kaggle (tests shrink this).
    pub run_poll_interval: std::time::Duration,
    /// Settings editors (applied with Apply).
    pub steps_field: TextEdit,
    pub lora_r_field: TextEdit,
    pub lora_alpha_field: TextEdit,
    pub batch_field: TextEdit,
    pub save_steps_field: TextEdit,
    /// The last thing to tell the user.
    pub notice: Option<String>,
    pub(crate) engine_api: Option<Arc<dyn EngineApi>>,
    pub(crate) client: Option<Arc<dyn KaggleClient>>,
    dir: PathBuf,
    token_store: Arc<dyn TokenStore>,
    kaggle_store: Arc<dyn KeyStore>,
    run_events_tx: mpsc::Sender<RunEvent>,
    run_events: mpsc::Receiver<RunEvent>,
    statuses: HashMap<String, SourceStatus>,
    /// Clip ids waiting for Whisper text.
    pending: HashSet<String>,
    /// Sources with a clean job out.
    cleaning: HashSet<String>,
    /// Whether the poll loop runs now (set on Start, cleared at the end).
    polling: bool,
    tickets: HashMap<JobTicket, (String, String)>,
    speech: SpeechService,
    player: SharedPlayer,
    events_tx: mpsc::Sender<TrainEvent>,
    events: mpsc::Receiver<TrainEvent>,
    waker: Waker,
}

impl TrainData {
    pub fn new(
        name: &str,
        dir: PathBuf,
        transcriber: Arc<dyn jeuvoice_whisper::Transcriber>,
        ensure_model: Option<crate::services::speech::EnsureModel>,
        token_store: Arc<dyn TokenStore>,
        kaggle_store: Arc<dyn KeyStore>,
        sound: bool,
    ) -> Self {
        let (events_tx, events) = mpsc::channel();
        let (run_events_tx, run_events) = mpsc::channel();
        let mut s = Self {
            project: TrainProject { name: name.into(), sources: Vec::new(), clips: Vec::new(), settings: default_settings(), target: TrainTarget::default() },
            show_hf: false,
            repo_field: TextEdit::new(""),
            split_field: TextEdit::new("train"),
            audio_field: TextEdit::new("audio"),
            text_field: TextEdit::new(""),
            limit_field: TextEdit::new(""),
            review: None,
            editing: None,
            edit_field: TextEdit::new(""),
            show_kaggle_form: false,
            kaggle_user_field: TextEdit::new(""),
            kaggle_key_field: TextEdit::new(""),
            kaggle_user: None,
            run: None,
            run_status: None,
            run_log: Vec::new(),
            losses: Vec::new(),
            checkpoints: Vec::new(),
            upload: None,
            download: None,
            bundle_path: None,
            confirm_start: false,
            show_script: false,
            export_pending: false,
            run_poll_interval: crate::train::run::POLL_INTERVAL,
            steps_field: TextEdit::new(""),
            lora_r_field: TextEdit::new(""),
            lora_alpha_field: TextEdit::new(""),
            batch_field: TextEdit::new(""),
            save_steps_field: TextEdit::new(""),
            notice: None,
            engine_api: None,
            client: None,
            dir,
            token_store,
            kaggle_store,
            run_events_tx,
            run_events,
            statuses: HashMap::new(),
            pending: HashSet::new(),
            cleaning: HashSet::new(),
            polling: false,
            tickets: HashMap::new(),
            speech: SpeechService::new(transcriber, ensure_model, None),
            player: SharedPlayer::new(sound),
            events_tx,
            events,
            waker: Arc::new(|| {}),
        };
        // Reopening the project folder restores everything.
        if let Ok(saved) = std::fs::read_to_string(s.project_file()).and_then(|t| serde_json::from_str::<TrainProject>(&t).map_err(|e| std::io::Error::other(e))) {
            s.project = saved;
            for src in &s.project.sources {
                s.statuses.insert(src.name.clone(), SourceStatus::Ready);
            }
        }
        s.refresh_kaggle_user();
        s.sync_settings_fields();
        // A run from the last session keeps going: resume its polling in setup.
        s.run = crate::train::run::discover_run(&s.dir);
        s
    }

    /// Point the tab at a Kaggle client (tests inject the fake).
    /// Later Train steps call the client the tab builds from the keychain key.
    #[allow(dead_code)]
    pub fn set_client(&mut self, client: Arc<dyn KaggleClient>) {
        self.client = Some(client);
    }

    fn project_file(&self) -> PathBuf {
        self.dir.join("train_project.json")
    }

    /// Point the tab at an engine (tests only; the app shares the Voice engine).
    /// Later Train steps call the engine through the shared client.
    #[allow(dead_code)]
    pub fn set_engine(&mut self, api: Arc<dyn EngineApi>) {
        self.engine_api = Some(api);
    }

    pub(crate) fn setup(&mut self, waker: Waker) {
        self.speech.setup(waker.clone());
        self.waker = waker;
        // The keychain key becomes the run client; a live run keeps polling.
        self.kaggle_client();
        self.resume_if_active();
    }

    // -- queries for the view and the tests --

    pub fn sources(&self) -> &[Source] {
        &self.project.sources
    }

    /// All clips, kept and left out alike. Later Train steps read the mix here.
    #[allow(dead_code)]
    pub fn clips(&self) -> &[TrainClip] {
        &self.project.clips
    }

    pub fn clips_of(&self, source: &str) -> Vec<&TrainClip> {
        self.project.clips.iter().filter(|c| c.clip.source == source).collect()
    }

    pub fn status_of(&self, source: &str) -> SourceStatus {
        self.statuses.get(source).cloned().unwrap_or(SourceStatus::Ready)
    }

    /// All sources idle with nothing waiting: the tab is done for now.
    /// The later Train steps poll this before they build the bundle.
    #[allow(dead_code)]
    pub fn ready(&self) -> bool {
        self.pending.is_empty()
            && self.cleaning.is_empty()
            && self.statuses.values().all(|s| matches!(s, SourceStatus::Ready))
    }

    pub fn excluded_count(&self) -> usize {
        self.project.clips.iter().filter(|c| c.clip.excluded.is_some()).count()
    }

    /// Raw and weighted minutes per source (excluded clips do not count).
    /// The source rows and the later Train steps read the mix here.
    #[allow(dead_code)]
    pub fn minutes(&self) -> Vec<(String, f32, f32)> {
        let clips: Vec<Clip> = self.project.clips.iter().map(|c| c.clip.clone()).collect();
        minutes_by_source(&clips, &self.project.sources)
    }

    /// The chips line: "N clips · M min", "K left out", "Band: B kHz".
    pub fn chips(&self) -> String {
        let included: Vec<&TrainClip> = self.project.clips.iter().filter(|c| c.clip.excluded.is_none()).collect();
        let min: f32 = included.iter().map(|c| c.clip.duration_s).sum::<f32>() / 60.0;
        let mut chips = format!("{} clips · {} min", included.len(), fmt_minutes(min));
        let left_out = self.excluded_count();
        if left_out > 0 {
            chips += &format!(" · {left_out} left out");
        }
        if !included.is_empty() {
            let mut bands: Vec<f32> = included.iter().map(|c| c.band_khz).collect();
            bands.sort_by(f32::total_cmp);
            chips += &format!(" · Band: {:.0} kHz", bands[bands.len() / 2]);
        }
        chips
    }

    // -- actions --

    /// Drop files or folders. Each drop is one source; folders keep their name.
    pub fn add_files(&mut self, paths: &[PathBuf]) {
        let mut files: Vec<PathBuf> = Vec::new();
        for p in paths {
            if p.is_dir() {
                let mut found: Vec<PathBuf> = Vec::new();
                if let Ok(entries) = std::fs::read_dir(&p) {
                    for e in entries.flatten() {
                        let path = e.path();
                        if path.is_file() && has_media_ext(&path) {
                            found.push(path);
                        }
                    }
                }
                found.sort();
                files.extend(found);
            } else if has_media_ext(p) {
                files.push(p.clone());
            }
        }
        if files.is_empty() {
            self.notice = Some("No audio files in the drop. Drop WAV, MP3 or video files.".into());
            return;
        }
        let base = if files.iter().all(|f| f.parent() == files[0].parent()) {
            files[0].parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "Local clips".into())
        } else {
            "Local clips".into()
        };
        let name = unique_name(&self.project.sources, &base);
        let dir: Option<String> = files[0].parent().map(|p| p.to_string_lossy().into_owned());
        self.project.sources.push(Source {
            name: name.clone(),
            kind: SourceKind::LocalFiles { paths: files.iter().map(|f| f.to_string_lossy().into_owned()).collect() },
            weight: 1,
            license: None,
            note: dir.map(|d| format!("Added from this folder: {d}")),
        });
        self.statuses.insert(name.clone(), SourceStatus::Working("Reading the clips…".into()));
        for (i, path) in files.iter().enumerate() {
            let id = format!("{name}-{i:04}");
            self.project.clips.push(TrainClip {
                clip: Clip {
                    id: id.clone(),
                    audio: path.to_string_lossy().into_owned(),
                    text: String::new(),
                    duration_s: 0.0,
                    source: name.clone(),
                    excluded: None,
                },
                band_khz: 0.0,
            });
            self.pending.insert(id);
        }
        self.decode_on_worker(name, files);
        self.save();
    }

    /// Add a Hugging Face dataset source from the form. Hides the form when the
    /// fetch starts.
    pub fn add_hf(&mut self) {
        let repo = self.repo_field.text.trim().to_string();
        if repo.is_empty() {
            self.notice = Some("Name the dataset first: a link or user/name.".into());
            return;
        }
        let split = nonempty(&self.split_field.text, "train");
        let audio_col = nonempty(&self.audio_field.text, "audio");
        let text_col = self.text_field.text.trim().to_string();
        let text_col = (!text_col.is_empty()).then_some(text_col);
        let limit = self.limit_field.text.trim().to_string();
        let limit = if limit.is_empty() { None } else { limit.parse::<u32>().ok() };
        if !self.limit_field.text.trim().is_empty() && limit.is_none() {
            self.notice = Some("“Use at most N clips” takes a whole number, or stays empty.".into());
            return;
        }
        let short = repo.rsplit('/').next().unwrap_or(&repo).to_string();
        let name = unique_name(&self.project.sources, &short);
        let Some(api) = self.engine_api.clone() else {
            self.notice = Some("The voice engine is off. Open the Voice tab once to start it, then try again.".into());
            return;
        };
        self.project.sources.push(Source {
            name: name.clone(),
            kind: SourceKind::HfDataset {
                repo: repo.clone(),
                config: None,
                split: split.clone(),
                audio_col: audio_col.clone(),
                text_col: text_col.clone(),
            },
            weight: 1,
            license: None,
            note: Some(format!("Hugging Face dataset {repo}.")),
        });
        self.statuses.insert(name.clone(), SourceStatus::Working("Fetching the dataset…".into()));
        self.show_hf = false;
        self.notice = None;
        let (events, wake) = (self.events_tx.clone(), self.waker.clone());
        let out_dir = self.dir.join("hf").join(&name);
        let token = self.token_store.get(HF_TOKEN_KEY);
        std::thread::spawn(move || {
            let event = match api.prep_hf(&HfPrep {
                repo,
                config: None,
                split,
                audio_col,
                text_col,
                limit,
                token,
                out_dir,
            }) {
                Ok(job) => match wait_job(&api, &job) {
                    Ok(status) => match parse_prep_clips(&status) {
                        Some(clips) => TrainEvent::HfDone { source: name, clips },
                        None => TrainEvent::HfFailed { source: name, reason: "The dataset gave no clips.".into() },
                    },
                    Err(e) => TrainEvent::HfFailed { source: name, reason: e },
                },
                Err(e) => TrainEvent::HfFailed { source: name, reason: e.to_string() },
            };
            let _ = events.send(event);
            wake();
        });
        self.save();
    }

    pub fn set_weight(&mut self, source: &str, weight: u32) {
        let weight = weight.clamp(1, 5);
        if let Some(s) = self.project.sources.iter_mut().find(|s| s.name == source) {
            s.weight = weight;
            self.save();
        }
    }

    pub fn remove_source(&mut self, source: &str) {
        self.project.sources.retain(|s| s.name != source);
        self.project.clips.retain(|c| c.clip.source != source);
        self.statuses.remove(source);
        self.cleaning.remove(source);
        if self.review.as_deref() == Some(source) {
            self.review = None;
        }
        self.save();
    }

    /// Where the run trains (This PC is off below 12 GB, RunPod is coming later).
    pub fn set_target(&mut self, target: TrainTarget) {
        if self.project.target != target {
            self.project.target = target;
            self.save();
        }
    }

    /// What the run starts from: the base model or a checkpoint.
    pub fn set_start(&mut self, start: jeuvoice_core::train::StartFrom) {
        if self.project.settings.start_from != start {
            self.project.settings.start_from = start;
            self.save();
        }
    }

    /// Read the keychain user name again (after saving or forgetting a key).
    pub fn refresh_kaggle_user(&mut self) {
        self.kaggle_user = self.kaggle_store.load().map(|a| a.username);
    }

    /// Save the Kaggle form to the keychain. Empty user or key stays on the form.
    pub fn save_kaggle_key(&mut self) {
        let username = self.kaggle_user_field.text.trim().to_string();
        let key = self.kaggle_key_field.text.trim().to_string();
        if username.is_empty() || key.is_empty() {
            self.notice = Some("Name the Kaggle user and paste the key.".into());
            return;
        }
        match self.kaggle_store.save(&KaggleAuth { username, key }) {
            Ok(()) => {
                self.notice = Some("Saved the Kaggle key.".into());
                self.show_kaggle_form = false;
                self.kaggle_user_field = TextEdit::new("");
                self.kaggle_key_field = TextEdit::new("");
                self.refresh_kaggle_user();
            }
            Err(e) => self.notice = Some(e.to_string()),
        }
    }

    /// Read a `kaggle.json` file into the keychain.
    pub fn import_kaggle_key(&mut self, path: &std::path::Path) {
        match jeuvoice_kaggle::import_kaggle_json(path) {
            Ok(auth) => match self.kaggle_store.save(&auth) {
                Ok(()) => {
                    self.notice = Some("Saved the Kaggle key.".into());
                    self.show_kaggle_form = false;
                    self.refresh_kaggle_user();
                }
                Err(e) => self.notice = Some(e.to_string()),
            },
            Err(e) => self.notice = Some(e.to_string()),
        }
    }

    /// Ask for a `kaggle.json` file and read it into the keychain.
    pub fn open_kaggle_import(&mut self, tx: &mut rune::Tx) {
        let options = rune::FileDialogOptions {
            title: "Import kaggle.json".into(),
            file_name: Some("kaggle.json".into()),
            ..Default::default()
        }
        .filter("Kaggle key", &["json"]);
        tx.open_file(options, |s: &mut crate::state::App, result, _tx| match result {
            Ok(Some(path)) => s.train.import_kaggle_key(&path),
            Ok(None) => {}
            Err(e) => s.train.notice = Some(format!("The Open window could not open: {e}")),
        });
    }

    /// Forget the saved Kaggle key.
    pub fn forget_kaggle_key(&mut self) {
        match self.kaggle_store.delete() {
            Ok(()) => {
                self.notice = Some("Removed the Kaggle key.".into());
                self.refresh_kaggle_user();
            }
            Err(e) => self.notice = Some(e.to_string()),
        }
    }

    /// Copy the settings into the edit fields.
    fn sync_settings_fields(&mut self) {
        let s = &self.project.settings;
        self.steps_field = TextEdit::new(&s.steps.to_string());
        self.lora_r_field = TextEdit::new(&s.lora_r.to_string());
        self.lora_alpha_field = TextEdit::new(&s.lora_alpha.to_string());
        self.batch_field = TextEdit::new(&s.batch_tokens.to_string());
        self.save_steps_field = TextEdit::new(&s.save_steps.to_string());
    }

    /// Parse the edit fields back into the settings.
    pub fn apply_settings(&mut self) {
        let parse = |t: &TextEdit, name: &str| {
            t.text.trim().parse::<u32>().ok().filter(|&v| v > 0).ok_or_else(|| format!("{name} takes a whole number above 0."))
        };
        let settings = (|| {
            let steps = parse(&self.steps_field, "Steps")?;
            let lora_r = parse(&self.lora_r_field, "LoRA r")?;
            let lora_alpha = parse(&self.lora_alpha_field, "LoRA alpha")?;
            let batch_tokens = parse(&self.batch_field, "Batch tokens")?;
            let save_steps = parse(&self.save_steps_field, "Save steps")?;
            Ok::<_, String>((steps, lora_r, lora_alpha, batch_tokens, save_steps))
        })();
        match settings {
            Ok((steps, lora_r, lora_alpha, batch_tokens, save_steps)) => {
                let s = &mut self.project.settings;
                s.steps = steps;
                s.lora_r = lora_r;
                s.lora_alpha = lora_alpha;
                s.batch_tokens = batch_tokens;
                s.save_steps = save_steps;
                self.notice = None;
                self.save();
            }
            Err(e) => self.notice = Some(e),
        }
    }

    /// Clips the run trains on: kept ones with their cleaned audio and text.
    pub fn included_clips(&self) -> Vec<crate::train::run::ClipRef> {
        self.project
            .clips
            .iter()
            .filter(|c| c.clip.excluded.is_none() && !c.clip.audio.is_empty())
            .map(|c| crate::train::run::ClipRef { id: c.clip.id.clone(), audio: c.clip.audio.clone(), text: c.clip.text.clone() })
            .collect()
    }

    /// The next free run name: `run-1`, `run-2`, …
    pub fn next_run_name(&self) -> String {
        for n in 1.. {
            let name = format!("run-{n}");
            if !self.dir.join("runs").join(&name).exists() {
                return name;
            }
        }
        unreachable!()
    }

    /// Whether a run is going now (polling, uploading or downloading).
    pub fn run_active(&self) -> bool {
        matches!(self.run.as_ref().map(|r| r.state), Some(RunState::Queued | RunState::Running))
    }

    /// A client for the run: the injected one, or the keychain key.
    fn kaggle_client(&mut self) -> Option<Arc<dyn KaggleClient>> {
        if self.client.is_none()
            && let Some(auth) = self.kaggle_store.load()
        {
            self.client = Some(Arc::new(jeuvoice_kaggle::Kaggle::new(auth)));
        }
        self.client.clone()
    }

    /// Start the confirmed run on Kaggle: manifests, bundle, upload, push, poll.
    pub fn start_run(&mut self) {
        self.confirm_start = false;
        if self.run_active() {
            return;
        }
        if self.project.target != TrainTarget::Kaggle {
            self.notice = Some("Pick Kaggle under “Where to train” first.".into());
            return;
        }
        let Some(kaggle) = self.kaggle_client() else {
            self.notice = Some("Add your Kaggle key first.".into());
            self.show_kaggle_form = true;
            return;
        };
        let Some(engine) = self.engine_api.clone() else {
            self.notice = Some("The voice engine is off. Open the Voice tab once to start it.".into());
            return;
        };
        let clips = self.included_clips();
        if clips.is_empty() {
            self.notice = Some("Add clips first: there is nothing to train on.".into());
            return;
        }
        let run_name = self.next_run_name();
        let req = crate::train::run::StartRequest {
            project_dir: self.dir.clone(),
            project_name: self.project.name.clone(),
            run_name,
            clips,
            sources: self.project.sources.iter().map(|s| (s.name.clone(), s.weight)).collect(),
            settings: self.project.settings.clone(),
            train_template: jeuvoice_core::kaggle_bundle::train_template(),
            owner: self.kaggle_user.clone().unwrap_or_default(),
            engine,
            kaggle,
            poll_interval: self.run_poll_interval,
        };
        self.run_status = None;
        self.run_log.clear();
        self.losses.clear();
        self.checkpoints.clear();
        self.upload = None;
        self.download = None;
        let (events, waker) = (self.run_events_tx.clone(), self.waker.clone());
        self.polling = true;
        std::thread::spawn(move || crate::train::run::start_run(req, events, waker));
    }

    /// A Queued or Running `run.json` keeps polling after an app restart.
    /// Only the poll loop resumes: manifests, bundle and push already happened.
    pub fn resume_if_active(&mut self) {
        if self.polling {
            return;
        }
        let Some(record) = self.run.clone() else { return };
        if !matches!(record.state, RunState::Queued | RunState::Running) {
            return;
        }
        let Some(kaggle) = self.client.clone() else { return };
        let run_dir = self.dir.join("runs").join(&record.name);
        // Rebuild what the last session already found.
        self.bundle_path = std::fs::read_dir(&run_dir)
            .ok()
            .and_then(|entries| {
                entries.flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|e| e == "zip"))
            });
        self.checkpoints = finished_checkpoints(&run_dir);
        self.losses = std::fs::read_to_string(run_dir.join("train.log"))
            .map(|t| crate::train::run::parse_losses(&t))
            .unwrap_or_default();
        let kernel = record.kernel.clone();
        let interval = self.run_poll_interval;
        let (events, waker) = (self.run_events_tx.clone(), self.waker.clone());
        std::thread::spawn(move || {
            crate::train::run::poll_loop(&kaggle, interval, &run_dir, &kernel, &events, &waker);
        });
        self.polling = true;
    }

    /// "Download bundle only": save the zip and the notebook with a Save dialog.
    /// Builds the bundle first when the run never got that far.
    pub fn export_bundle(&mut self, tx: &mut rune::Tx) {
        if let Some(path) = self.bundle_path.clone().filter(|p| p.is_file()) {
            self.save_bundle_dialog(tx, path);
            return;
        }
        if self.run_active() || self.included_clips().is_empty() {
            self.notice = Some("There is no bundle yet. Start the run first.".into());
            return;
        }
        let Some(engine) = self.engine_api.clone() else {
            self.notice = Some("The voice engine is off. Open the Voice tab once to start it.".into());
            return;
        };
        self.export_pending = true;
        self.notice = Some("Building the bundle…".into());
        // A fixed scratch folder: never confused with a real run.
        let run_name = "bundle-only".to_string();
        let (events, waker) = (self.run_events_tx.clone(), self.waker.clone());
        let project_dir = self.dir.clone();
        let project_name = self.project.name.clone();
        let clips = self.included_clips();
        let sources: Vec<(String, u32)> = self.project.sources.iter().map(|s| (s.name.clone(), s.weight)).collect();
        let settings = self.project.settings.clone();
        std::thread::spawn(move || {
            match crate::train::run::build_bundle_flow(
                &project_dir,
                &run_name,
                &project_name,
                &clips,
                &sources,
                &settings,
                &jeuvoice_core::kaggle_bundle::train_template(),
                &engine,
            ) {
                Ok(built) => {
                    let _ = events.send(RunEvent::BundleReady(built.bundle));
                    waker();
                }
                Err(e) => {
                    let _ = events.send(RunEvent::RunFailed(e));
                    waker();
                }
            }
        });
    }

    fn save_bundle_dialog(&mut self, tx: &mut rune::Tx, path: PathBuf) {
        let stem = path.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "bundle".into());
        let options = rune::FileDialogOptions {
            title: "Save the Kaggle bundle".into(),
            directory: path.parent().map(Path::to_path_buf),
            file_name: Some(format!("{stem}.zip")),
            ..Default::default()
        }
        .filter("Zip", &["zip"]);
        tx.save_file(options, move |s: &mut crate::state::App, result, _tx| match result {
            Ok(Some(to)) => s.train.finish_bundle_export(to),
            Ok(None) => s.train.export_pending = false,
            Err(e) => s.train.notice = Some(format!("The Save window could not open: {e}")),
        });
    }

    fn finish_bundle_export(&mut self, to: PathBuf) {
        self.export_pending = false;
        let Some(from) = self.bundle_path.clone() else { return };
        let out = (|| {
            std::fs::copy(&from, &to).map_err(|e| format!("The bundle could not be saved: {e}"))?;
            let nb_to = to.with_extension("ipynb");
            let nb_from = from.parent().map(|d| d.join("notebook.ipynb"));
            if let Some(nb_from) = nb_from.filter(|p| p.is_file()) {
                std::fs::copy(&nb_from, &nb_to).map_err(|e| format!("The notebook could not be saved: {e}"))?;
            }
            Ok::<_, String>(nb_to)
        })();
        match out {
            Ok(nb_to) => {
                self.notice = Some(format!(
                    "Saved {} and {}.",
                    to.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                    nb_to.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
                ));
            }
            Err(e) => self.notice = Some(e),
        }
    }

    /// Drain run worker news. `true` when the view changed.
    fn poll_run(&mut self, tx: &mut rune::Tx) -> bool {
        let mut changed = false;
        while let Ok(e) = self.run_events.try_recv() {
            changed = true;
            match e {
                RunEvent::UploadProgress(done, total) => self.upload = Some((done, total)),
                RunEvent::Started(record) => {
                    self.run = Some(record.clone());
                    self.polling = true;
                    self.upload = None;
                    self.notice = None;
                    self.save_record(&record);
                }
                RunEvent::Status(status) => {
                    let state = match status.state {
                        jeuvoice_kaggle::KernelState::Queued => RunState::Queued,
                        jeuvoice_kaggle::KernelState::Running => RunState::Running,
                        jeuvoice_kaggle::KernelState::Complete
                        | jeuvoice_kaggle::KernelState::Error
                        | jeuvoice_kaggle::KernelState::Cancelled => continue,
                    };
                    self.run_status = Some(status);
                    if let Some(r) = self.run.as_mut() {
                        r.state = state;
                        let record = r.clone();
                        self.save_record(&record);
                    }
                }
                RunEvent::LogTail(tail) => {
                    self.run_log = tail.lines().map(str::to_string).collect();
                    self.losses = crate::train::run::parse_losses(&tail);
                }
                RunEvent::PollFailed(message) => {
                    // Transient: the loop keeps polling, the notice shows it.
                    self.notice = Some(message);
                }
                RunEvent::DownloadProgress(done, total) => self.download = Some((done, total)),
                RunEvent::Downloaded { checkpoints, losses } => {
                    self.checkpoints = checkpoints;
                    self.download = None;
                    self.polling = false;
                    if !losses.is_empty() {
                        self.losses = losses;
                    }
                    if let Some(r) = self.run.as_mut() {
                        r.state = RunState::Complete;
                        r.message = None;
                        let record = r.clone();
                        self.save_record(&record);
                    }
                }
                RunEvent::RunFailed(message) => {
                    self.polling = false;
                    if let Some(r) = self.run.as_mut() {
                        r.state = RunState::Error;
                        r.message = Some(message.clone());
                        let record = r.clone();
                        self.save_record(&record);
                    } else {
                        self.notice = Some(message);
                    }
                }
                RunEvent::RunCancelled => {
                    self.polling = false;
                    if let Some(r) = self.run.as_mut() {
                        r.state = RunState::Cancelled;
                        r.message = Some("The run was cancelled.".into());
                        let record = r.clone();
                        self.save_record(&record);
                    }
                }
                RunEvent::BundleReady(path) => {
                    self.bundle_path = Some(path.clone());
                    // "Download bundle only" before the run built one waits for this.
                    if self.export_pending {
                        self.export_pending = false;
                        self.save_bundle_dialog(tx, path);
                    }
                }
            }
        }
        changed
    }

    fn save_record(&self, record: &RunRecord) {
        let _ = crate::train::run::save_record(&self.dir.join("runs").join(&record.name), record);
    }

    /// Leave a clip out of the mix (`excluded`), or bring it back.
    pub fn set_excluded(&mut self, id: &str, excluded: bool) {
        if let Some(c) = self.project.clips.iter_mut().find(|c| c.clip.id == id) {
            c.clip.excluded = excluded.then(|| "Left out in Review.".to_string());
            self.save();
        }
    }

    pub fn set_text(&mut self, id: &str, text: &str) {
        if let Some(c) = self.project.clips.iter_mut().find(|c| c.clip.id == id) {
            let text = text.trim().to_string();
            if c.clip.text != text {
                c.clip.text = text;
                self.save();
            }
        }
    }

    /// Play or pause a review clip through the shared sound player.
    pub fn toggle_play(&mut self, id: &str) {
        let clip = self.project.clips.iter().find(|c| c.clip.id == id);
        let Some(path) = clip.map(|c| c.clip.audio.clone()) else { return };
        if self.player.is_playing_for(PlayerOwner::Train) {
            if let Some(pb) = self.player.player(PlayerOwner::Train) {
                pb.pause();
            }
            return;
        }
        match audio::decode(Path::new(&path)) {
            Ok(a) => {
                self.player.load(PlayerOwner::Train, a);
                if let Some(pb) = self.player.player(PlayerOwner::Train) {
                    pb.play();
                }
                if let Some(e) = self.player.take_error() {
                    self.notice = Some(e);
                }
            }
            Err(e) => self.notice = Some(e.to_string()),
        }
    }

    /// Drain worker news, speech results and the edit field. `true` when the
    /// view changed.
    pub(crate) fn poll(&mut self, settings: &Settings, tx: &mut rune::Tx) -> bool {
        let mut changed = self.poll_run(tx);
        for e in self.speech.drain_events() {
            changed = true;
            if let crate::services::speech::ServiceEvent::Notice(n) = e {
                self.notice = Some(n);
            }
        }
        while let Ok(e) = self.events.try_recv() {
            changed = true;
            self.take_event(e, settings);
        }
        for (ticket, u) in self.speech.poll() {
            changed = true;
            let Some((source, id)) = self.tickets.remove(&ticket) else { continue };
            match u {
                Update::Progress(_) => {
                    self.tickets.insert(ticket, (source, id));
                }
                Update::Done(t) => {
                    let text = t.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
                    self.pending.remove(&id);
                    self.set_text_quiet(&id, &text);
                    self.maybe_start_clean(&source);
                    self.save();
                }
                Update::Failed(e) => {
                    self.pending.remove(&id);
                    self.notice = Some(format!("Whisper could not hear a clip: {e}"));
                    self.maybe_start_clean(&source);
                    self.save();
                }
            }
        }
        // The open text field writes back as the user types.
        if let Some(id) = self.editing.clone()
            && self.project.clips.iter().find(|c| c.clip.id == id).is_some_and(|c| c.clip.text != self.edit_field.text)
        {
            self.set_text(&id, &self.edit_field.text.clone());
            changed = true;
        }
        changed
    }

    fn take_event(&mut self, e: TrainEvent, settings: &Settings) {
        match e {
            TrainEvent::Decoded { source, id, samples } => {
                let opts =
                    jeuvoice_whisper::TranscribeOptions { language: settings.whisper_language.clone(), use_gpu: settings.use_gpu };
                let ticket = self.speech.submit(SpeechJob::Samples { samples_16k: samples, tag: 0, opts });
                self.tickets.insert(ticket, (source, id));
                self.set_status_quiet();
            }
            TrainEvent::DecodeFailed { source, id, reason } => {
                self.pending.remove(&id);
                if let Some(c) = self.project.clips.iter_mut().find(|c| c.clip.id == id) {
                    c.clip.excluded = Some(reason);
                }
                self.maybe_start_clean(&source);
                self.save();
            }
            TrainEvent::HfDone { source, clips } => {
                let mut silent: Vec<(String, PathBuf)> = Vec::new();
                for p in clips {
                    let id = format!("{source}-{}", p.id);
                    self.project.clips.push(TrainClip {
                        clip: Clip {
                            id: id.clone(),
                            audio: p.audio.clone(),
                            text: p.text.clone(),
                            duration_s: 0.0,
                            source: source.clone(),
                            excluded: p.excluded.clone(),
                        },
                        band_khz: 0.0,
                    });
                    if p.excluded.is_none() && p.text.is_empty() {
                        self.pending.insert(id.clone());
                        silent.push((id, PathBuf::from(p.audio)));
                    }
                }
                if !silent.is_empty() {
                    let names: Vec<String> = silent.iter().map(|(id, _)| id.clone()).collect();
                    let paths: Vec<PathBuf> = silent.into_iter().map(|(_, p)| p).collect();
                    self.decode_ids_on_worker(source.clone(), names, paths);
                    if let Some(s) = self.statuses.get_mut(&source) {
                        *s = SourceStatus::Working("Hearing the clips…".into());
                    }
                }
                self.maybe_start_clean(&source);
                self.save();
            }
            TrainEvent::HfFailed { source, reason } => {
                self.statuses.insert(source, SourceStatus::Failed(reason));
                self.save();
            }
            TrainEvent::Cleaned { source, clips } => {
                self.cleaning.remove(&source);
                for p in clips {
                    if let Some(c) = self.project.clips.iter_mut().find(|c| c.clip.source == source && c.clip.id == p.id) {
                        if !p.audio.is_empty() {
                            c.clip.audio = p.audio;
                        }
                        c.clip.duration_s = p.duration_s;
                        c.band_khz = p.band_khz;
                        if p.excluded.is_some() {
                            c.clip.excluded = p.excluded;
                        }
                    }
                }
                self.statuses.insert(source, SourceStatus::Ready);
                self.save();
            }
            TrainEvent::CleanFailed { source, reason } => {
                self.cleaning.remove(&source);
                self.statuses.insert(source, SourceStatus::Failed(reason));
                self.save();
            }
        }
    }

    /// Start `/prep/clean` once every clip of the source has text (or Whisper
    /// gave up on it).
    fn maybe_start_clean(&mut self, source: &str) {
        if self.cleaning.contains(source) {
            return;
        }
        let waiting = self.project.clips.iter().any(|c| c.clip.source == source && self.pending.contains(&c.clip.id));
        if waiting {
            return;
        }
        if !self.project.clips.iter().any(|c| c.clip.source == source && c.clip.excluded.is_none()) {
            self.statuses.insert(source.into(), SourceStatus::Ready);
            return;
        }
        let Some(api) = self.engine_api.clone() else {
            self.statuses.insert(source.into(), SourceStatus::Failed("The voice engine is off.".into()));
            return;
        };
        let pairs: Vec<(String, String)> = self
            .project
            .clips
            .iter()
            .filter(|c| c.clip.source == source && c.clip.excluded.is_none())
            .map(|c| (c.clip.id.clone(), c.clip.audio.clone()))
            .collect();
        self.cleaning.insert(source.into());
        self.statuses.insert(source.into(), SourceStatus::Working("Cleaning the clips…".into()));
        let (events, wake) = (self.events_tx.clone(), self.waker.clone());
        let source = source.to_string();
        let out_dir = self.dir.join("clean").join(&source);
        std::thread::spawn(move || {
            let event = match api.prep_clean(&pairs, &out_dir) {
                Ok(job) => match wait_job(&api, &job) {
                    Ok(status) => match parse_prep_clips(&status) {
                        Some(clips) => TrainEvent::Cleaned { source, clips },
                        None => TrainEvent::CleanFailed { source, reason: "The cleaning gave no clips.".into() },
                    },
                    Err(e) => TrainEvent::CleanFailed { source, reason: e },
                },
                Err(e) => TrainEvent::CleanFailed { source, reason: e.to_string() },
            };
            let _ = events.send(event);
            wake();
        });
    }

    /// Decode dropped files on a worker, then hand the samples to Whisper.
    fn decode_on_worker(&self, source: String, files: Vec<PathBuf>) {
        let ids: Vec<String> =
            self.project.clips.iter().filter(|c| c.clip.source == source).map(|c| c.clip.id.clone()).collect();
        self.decode_ids_on_worker(source, ids, files);
    }

    fn decode_ids_on_worker(&self, source: String, ids: Vec<String>, files: Vec<PathBuf>) {
        let (events, wake) = (self.events_tx.clone(), self.waker.clone());
        std::thread::spawn(move || {
            for (id, path) in ids.into_iter().zip(files) {
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| id.clone());
                match audio::decode(&path) {
                    Ok(a) => {
                        let _ = events.send(TrainEvent::Decoded { source: source.clone(), id, samples: Arc::new(to_whisper(&a)) });
                    }
                    Err(_) => {
                        let _ = events.send(TrainEvent::DecodeFailed {
                            source: source.clone(),
                            id,
                            reason: format!("JeuVoice cannot read this file: {name}."),
                        });
                    }
                }
            }
            wake();
        });
    }

    fn set_status_quiet(&mut self) {
        for (source, id) in self.tickets.values() {
            if self.pending.contains(id)
                && let Some(s) = self.statuses.get_mut(source)
                && matches!(s, SourceStatus::Working(_))
            {
                *s = SourceStatus::Working("Hearing the clips…".into());
            }
        }
    }

    fn set_text_quiet(&mut self, id: &str, text: &str) {
        if let Some(c) = self.project.clips.iter_mut().find(|c| c.clip.id == id) {
            c.clip.text = text.to_string();
        }
    }

    fn save(&self) {
        let _ = std::fs::create_dir_all(&self.dir);
        if let Ok(text) = serde_json::to_string_pretty(&self.project) {
            let _ = std::fs::write(self.project_file(), text);
        }
    }
}

/// Checkpoint folders under a run dir, sorted.
fn finished_checkpoints(run_dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(run_dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("checkpoint-")))
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    out.sort();
    out
}

fn has_media_ext(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| MEDIA.contains(&e.to_lowercase().as_str()))
}

fn nonempty(text: &str, fallback: &str) -> String {
    let text = text.trim();
    if text.is_empty() { fallback.into() } else { text.into() }
}

fn unique_name(sources: &[Source], base: &str) -> String {
    if !sources.iter().any(|s| s.name == base) {
        return base.into();
    }
    for n in 2.. {
        let name = format!("{base} {n}");
        if !sources.iter().any(|s| s.name == name) {
            return name;
        }
    }
    unreachable!()
}

/// Wait for an engine prep job. `true` while it finishes; the error is plain.
fn wait_job(api: &Arc<dyn EngineApi>, job: &jeuvoice_engine_client::JobId) -> Result<jeuvoice_engine_client::JobStatus, String> {
    loop {
        match api.job(job) {
            Ok(s) => match s.state {
                jeuvoice_engine_client::JobState::Done => return Ok(s),
                jeuvoice_engine_client::JobState::Failed => {
                    return Err(s.error.unwrap_or_else(|| "Preparing the data failed. Try again.".into()));
                }
                jeuvoice_engine_client::JobState::Cancelled => return Err("Preparing the data stopped.".into()),
                _ => std::thread::sleep(std::time::Duration::from_millis(200)),
            },
            Err(e) => return Err(e.to_string()),
        }
    }
}
