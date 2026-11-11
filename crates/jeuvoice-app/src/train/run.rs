//! Train tab, step 4 (run): start the Kaggle run, follow it, collect the outputs.
//!
//! `Start on Kaggle` runs one worker end to end: manifests (`/prep/manifest`),
//! the bundle, the dataset upload, the kernel push, then the poll loop (one
//! `kernel_status` per [`POLL_INTERVAL`]). News comes back as [`RunEvent`]
//! which `poll` drains on the UI thread. A `run.json` with a Queued or Running
//! run resumes polling after an app restart.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use jeuvoice_kaggle::{DatasetRef, KaggleError, KernelRef, KernelSpec, KernelStatus, accelerator_t4_x2};

/// How often the worker asks Kaggle for the run state.
pub const POLL_INTERVAL: Duration = Duration::from_secs(60);

/// The Kaggle client behind a trait so headless tests use a fake.
pub trait KaggleClient: Send + Sync {
    fn upload_dataset(
        &self,
        owner_slug: &str,
        title: &str,
        files: &[PathBuf],
        private: bool,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<DatasetRef, KaggleError>;
    fn push_kernel(&self, spec: &KernelSpec) -> Result<KernelRef, KaggleError>;
    fn kernel_status(&self, k: &KernelRef) -> Result<KernelStatus, KaggleError>;
    fn kernel_log_tail(&self, k: &KernelRef, lines: usize) -> Result<String, KaggleError>;
    fn download_output(
        &self,
        k: &KernelRef,
        dest: &Path,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<Vec<PathBuf>, KaggleError>;
}

impl KaggleClient for jeuvoice_kaggle::Kaggle {
    fn upload_dataset(
        &self,
        owner_slug: &str,
        title: &str,
        files: &[PathBuf],
        private: bool,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<DatasetRef, KaggleError> {
        self.upload_dataset(owner_slug, title, files, private, progress)
    }
    fn push_kernel(&self, spec: &KernelSpec) -> Result<KernelRef, KaggleError> {
        self.push_kernel(spec)
    }
    fn kernel_status(&self, k: &KernelRef) -> Result<KernelStatus, KaggleError> {
        self.kernel_status(k)
    }
    fn kernel_log_tail(&self, k: &KernelRef, lines: usize) -> Result<String, KaggleError> {
        self.kernel_log_tail(k, lines)
    }
    fn download_output(
        &self,
        k: &KernelRef,
        dest: &Path,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<Vec<PathBuf>, KaggleError> {
        self.download_output(k, dest, progress)
    }
}

/// What a run is doing. Saved in `run.json`; Queued and Running resume polling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RunState {
    #[default]
    Queued,
    Running,
    Complete,
    Error,
    Cancelled,
}

/// `runs/<name>/run.json`: what went up and what the run is doing.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RunRecord {
    pub name: String,
    pub dataset: String,
    pub kernel: KernelRef,
    /// Unix seconds.
    pub started_at: u64,
    pub state: RunState,
    pub message: Option<String>,
}

/// News from the run worker.
pub enum RunEvent {
    UploadProgress(u64, u64),
    Started(RunRecord),
    /// A fresh status from the poll loop.
    Status(KernelStatus),
    /// The last 30 log lines (the live loss and the error box read these).
    LogTail(String),
    /// The poll loop failed once (it keeps polling; the notice shows it).
    PollFailed(String),
    DownloadProgress(u64, u64),
    /// Outputs are in `runs/<name>/`: checkpoints and `(step, loss)` points.
    Downloaded { checkpoints: Vec<String>, losses: Vec<(u32, f32)> },
    /// The run failed: status Error or a worker error.
    RunFailed(String),
    /// The run was cancelled on Kaggle.
    RunCancelled,
    /// The bundle is ready for "Download bundle only".
    BundleReady(PathBuf),
    /// A checkpoint render finished: kept for replay (at most two).
    CkDone { checkpoint: String, wav: PathBuf },
    /// A checkpoint render failed.
    CkFailed(String),
}

/// `(step, loss)` points from a `train.log` text. The trainer writes
/// `Running evaluation at step N...` and later `Eval Loss: X`; those pair up.
/// A generic `step N ... loss X` line also parses, for other trainers.
pub fn parse_losses(log: &str) -> Vec<(u32, f32)> {
    let mut out = Vec::new();
    let mut step: Option<u32> = None;
    for line in log.lines() {
        if let Some(s) = eval_step(line) {
            step = Some(s);
        } else if let Some(loss) = eval_loss(line) {
            if let Some(s) = step {
                out.push((s, loss));
            }
        } else if let Some((s, l)) = generic_point(line) {
            out.push((s, l));
        }
    }
    out.sort_by_key(|(s, _)| *s);
    out.dedup_by_key(|(s, _)| *s);
    out
}

/// `Running evaluation at step 250...` → `250`.
fn eval_step(line: &str) -> Option<u32> {
    let lower = line.to_lowercase();
    let i = lower.find("evaluation at step")?;
    lower[i..].split_whitespace().nth(3).and_then(|w| w.trim_matches(|c: char| !c.is_ascii_digit()).parse().ok())
}

/// `Eval Loss: 3.9556` → `3.9556`.
fn eval_loss(line: &str) -> Option<f32> {
    let lower = line.to_lowercase();
    let i = lower.find("eval loss")?;
    lower[i..].split(':').nth(1).and_then(|v| v.trim().split_whitespace().next()).and_then(|v| v.parse().ok())
}

fn generic_point(line: &str) -> Option<(u32, f32)> {
    let lower = line.to_lowercase();
    let (Some(si), Some(li)) = (lower.find("step"), lower.find("loss")) else { return None };
    let step: Option<u32> = lower[si..].split_whitespace().nth(1).and_then(|w| w.trim_matches(|c: char| !c.is_ascii_digit()).parse().ok());
    let loss: Option<f32> = lower[li..].split_whitespace().nth(1).and_then(|w| w.trim_matches(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-')).parse().ok());
    Some((step?, loss?))
}

/// "12 min", "2 h 05 min": how long the run has been going.
pub fn elapsed_text(started_at: u64, now: u64) -> String {
    let secs = now.saturating_sub(started_at);
    if secs < 3600 {
        crate::i18n::tr!("{} min", "{} dk", (secs / 60).max(1))
    } else {
        crate::i18n::tr!("{} h {:02} min", "{} sa {:02} dk", secs / 3600, (secs % 3600) / 60)
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// What the start worker needs. Built on the UI thread, run on the worker.
pub struct StartRequest {
    pub project_dir: PathBuf,
    pub project_name: String,
    pub run_name: String,
    pub clips: Vec<ClipRef>,
    pub sources: Vec<(String, u32)>,
    pub settings: jeuvoice_core::train::TrainSettings,
    pub train_template: serde_json::Value,
    pub owner: String,
    pub engine: Arc<dyn jeuvoice_engine_client::EngineApi>,
    pub kaggle: Arc<dyn KaggleClient>,
    pub poll_interval: Duration,
}

/// One included clip: its id, cleaned audio and text.
pub struct ClipRef {
    pub id: String,
    pub audio: String,
    pub text: String,
}

/// The bundle and notebook, built by either worker.
pub struct BuiltBundle {
    pub bundle: PathBuf,
    pub notebook_text: String,
    pub run_id: String,
}

/// Manifests, staged WAVs, the zip and the notebook text. Shared by the start
/// worker and "Download bundle only".
pub fn build_bundle_flow(
    project_dir: &Path,
    run_name: &str,
    project_name: &str,
    clips: &[ClipRef],
    sources: &[(String, u32)],
    settings: &jeuvoice_core::train::TrainSettings,
    train_template: &serde_json::Value,
    engine: &Arc<dyn jeuvoice_engine_client::EngineApi>,
) -> Result<BuiltBundle, String> {
    let run_dir = project_dir.join("runs").join(run_name);
    let out_dir = run_dir.join("manifests");
    let prep: Vec<jeuvoice_engine_client::PrepClip> = clips
        .iter()
        .map(|c| jeuvoice_engine_client::PrepClip {
            id: c.id.clone(),
            audio: c.audio.clone(),
            duration_s: 0.0,
            band_khz: 0.0,
            text: c.text.clone(),
            excluded: None,
        })
        .collect();
    let job = engine.prep_manifest(&prep, sources, 0.02, &out_dir).map_err(|e| e.to_string())?;
    wait_engine_job(engine, &job)?;
    let audio_src = run_dir.join("audio_src");
    stage_clips(clips, &audio_src)?;
    let cfg = jeuvoice_core::kaggle_bundle::train_config(settings, train_template);
    let data = jeuvoice_core::kaggle_bundle::data_config("tokens");
    let run_id = slug(project_name, run_name);
    let bundle_marker = format!("bundle_{run_id}.txt");
    let bundle = jeuvoice_core::kaggle_bundle::build_bundle(&run_dir, &run_id, &audio_src, &out_dir, &cfg, &data)?;
    let notebook_text = jeuvoice_core::kaggle_bundle::notebook(&run_id, &bundle_marker);
    std::fs::write(run_dir.join("notebook.ipynb"), &notebook_text)
        .map_err(|e| crate::i18n::trf!("The notebook could not be saved: {e}", "Not defteri kaydedilemedi: {e}"))?;
    Ok(BuiltBundle { bundle, notebook_text, run_id })
}

/// The whole start flow on one worker: manifests, bundle, upload, push, then
/// the poll loop. Answers with [`RunEvent`]s.
pub fn start_run(req: StartRequest, events: std::sync::mpsc::Sender<RunEvent>, waker: rune::Waker) {
    let say = |e: RunEvent| {
        let _ = events.send(e);
        waker();
    };
    let run_dir = req.project_dir.join("runs").join(&req.run_name);
    // 1-2. Manifests, staged WAVs, the zip and the notebook.
    let built = match build_bundle_flow(
        &req.project_dir,
        &req.run_name,
        &req.project_name,
        &req.clips,
        &req.sources,
        &req.settings,
        &req.train_template,
        &req.engine,
    ) {
        Ok(b) => b,
        Err(e) => {
            say(RunEvent::RunFailed(e));
            return;
        }
    };
    say(RunEvent::BundleReady(built.bundle.clone()));
    // 3. Upload the bundle as a private dataset. Its title carries "data":
    // Kaggle refuses a kernel whose title another dataset already uses.
    let title = format!("{} {}", req.project_name, req.run_name);
    let dataset_title = format!("{title} data");
    let say_up = |done: u64, total: u64| {
        let _ = events.send(RunEvent::UploadProgress(done, total));
        waker();
    };
    let mut say_up = say_up;
    let dataset = match req.kaggle.upload_dataset(&req.owner, &dataset_title, &[built.bundle], true, &mut say_up) {
        Ok(d) => d,
        Err(e) => {
            say(RunEvent::RunFailed(e.to_string()));
            return;
        }
    };
    // 4. Push the notebook; the run starts.
    let spec = KernelSpec {
        slug: format!("{}/{}", req.owner, built.run_id),
        title,
        notebook_json: built.notebook_text,
        dataset_sources: vec![dataset.id()],
        enable_gpu: true,
        enable_internet: true,
        accelerator: accelerator_t4_x2().into(),
        private: true,
    };
    let kernel = match req.kaggle.push_kernel(&spec) {
        Ok(k) => k,
        Err(e) => {
            say(RunEvent::RunFailed(e.to_string()));
            return;
        }
    };
    let record = RunRecord {
        name: req.run_name.clone(),
        dataset: dataset.id(),
        kernel: kernel.clone(),
        started_at: now_secs(),
        state: RunState::Queued,
        message: None,
    };
    if let Err(e) = save_record(&run_dir, &record) {
        say(RunEvent::RunFailed(crate::i18n::trf!("The run could not be saved: {e}", "Çalıştırma kaydedilemedi: {e}")));
        return;
    }
    say(RunEvent::Started(record));
    poll_loop(&req.kaggle, req.poll_interval, &run_dir, &kernel, &events, &waker);
}

/// Follow the run to its end. On Complete the outputs land in `runs/<name>/`.
pub fn poll_loop(
    kaggle: &Arc<dyn KaggleClient>,
    poll_interval: Duration,
    run_dir: &Path,
    kernel: &KernelRef,
    events: &std::sync::mpsc::Sender<RunEvent>,
    waker: &rune::Waker,
) {
    loop {
        std::thread::sleep(poll_interval);
        let status = match kaggle.kernel_status(kernel) {
            Ok(s) => s,
            Err(e) => {
                let _ = events.send(RunEvent::PollFailed(e.to_string()));
                waker();
                continue;
            }
        };
        let _ = events.send(RunEvent::Status(status.clone()));
        waker();
        if let Ok(tail) = kaggle.kernel_log_tail(kernel, 30) {
            let _ = events.send(RunEvent::LogTail(tail));
            waker();
        }
        match status.state {
            jeuvoice_kaggle::KernelState::Queued | jeuvoice_kaggle::KernelState::Running => {}
            jeuvoice_kaggle::KernelState::Complete => {
                let event = match collect_outputs(kaggle, run_dir, kernel, events, waker) {
                    Ok((checkpoints, losses)) => RunEvent::Downloaded { checkpoints, losses },
                    Err(e) => RunEvent::RunFailed(e),
                };
                let _ = events.send(event);
                waker();
                return;
            }
            jeuvoice_kaggle::KernelState::Error => {
                let _ = events.send(RunEvent::RunFailed(status.message.unwrap_or_else(|| crate::i18n::tr!("The run failed.", "Çalıştırma başarısız oldu.").into())));
                waker();
                return;
            }
            jeuvoice_kaggle::KernelState::Cancelled => {
                let _ = events.send(RunEvent::RunCancelled);
                waker();
                return;
            }
        }
    }
}

/// Download the outputs, list the checkpoints and parse the loss curve.
/// Outputs nest under `result/`; the search walks the whole run folder.
fn collect_outputs(
    kaggle: &Arc<dyn KaggleClient>,
    run_dir: &Path,
    kernel: &KernelRef,
    events: &std::sync::mpsc::Sender<RunEvent>,
    waker: &rune::Waker,
) -> Result<(Vec<String>, Vec<(u32, f32)>), String> {
    let say_dl = |done: u64, total: u64| {
        let _ = events.send(RunEvent::DownloadProgress(done, total));
        waker();
    };
    let mut say_dl = say_dl;
    let files = kaggle.download_output(kernel, run_dir, &mut say_dl).map_err(|e| e.to_string())?;
    let _ = files;
    let checkpoints = find_checkpoints(run_dir);
    let log = find_file(run_dir, "train.log").and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
    Ok((checkpoints, parse_losses(&log)))
}

/// Checkpoint folder names anywhere under the run dir, sorted.
pub fn find_checkpoints(run_dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    collect_dirs(run_dir, &mut out);
    out.sort();
    out
}

fn collect_dirs(dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("checkpoint-")) {
                out.push(p.file_name().unwrap().to_string_lossy().into_owned());
            }
            collect_dirs(&p, out);
        }
    }
}

/// The first file with `name` anywhere under the run dir.
pub fn find_file(run_dir: &Path, name: &str) -> Option<PathBuf> {
    let mut stack = vec![run_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == name) {
                return Some(p);
            }
        }
    }
    None
}

/// The folder of a checkpoint by name, anywhere under the run dir.
pub fn ck_dir(run_dir: &Path, ck: &str) -> Option<PathBuf> {
    let mut stack = vec![run_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.file_name().is_some_and(|n| n == ck) {
                    return Some(p);
                }
                stack.push(p);
            }
        }
    }
    None
}

/// Copy each included clip to `<audio_src>/<id>.<ext>`.
fn stage_clips(clips: &[ClipRef], audio_src: &Path) -> Result<(), String> {
    std::fs::create_dir_all(audio_src).map_err(|e| crate::i18n::trf!("The bundle folder could not be made: {e}", "Paket klasörü oluşturulamadı: {e}"))?;
    for c in clips {
        let ext = Path::new(&c.audio).extension().and_then(|e| e.to_str()).unwrap_or("wav");
        let to = audio_src.join(format!("{}.{ext}", c.id));
        std::fs::copy(&c.audio, &to).map_err(|e| crate::i18n::tr!("The clip {} could not be packed: {e}", "{} klibi paketlenemedi: {e}", c.id))?;
    }
    Ok(())
}

fn wait_engine_job(engine: &Arc<dyn jeuvoice_engine_client::EngineApi>, job: &jeuvoice_engine_client::JobId) -> Result<(), String> {
    loop {
        match engine.job(job) {
            Ok(s) => match s.state {
                jeuvoice_engine_client::JobState::Done => return Ok(()),
                jeuvoice_engine_client::JobState::Failed => {
                    return Err(s.error.unwrap_or_else(|| crate::i18n::tr!("Preparing the data failed. Try again.", "Veri hazırlanamadı. Tekrar deneyin.").into()));
                }
                jeuvoice_engine_client::JobState::Cancelled => return Err(crate::i18n::tr!("Preparing the data stopped.", "Veri hazırlama durdu.").into()),
                _ => std::thread::sleep(Duration::from_millis(500)),
            },
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// `runs/<name>/run.json`.
pub fn record_path(project_dir: &Path, run_name: &str) -> PathBuf {
    project_dir.join("runs").join(run_name).join("run.json")
}

pub fn save_record(run_dir: &Path, record: &RunRecord) -> Result<(), String> {
    std::fs::create_dir_all(run_dir).map_err(|e| crate::i18n::trf!("The run folder could not be made: {e}", "Çalıştırma klasörü oluşturulamadı: {e}"))?;
    let text = serde_json::to_string_pretty(record).map_err(|e| crate::i18n::trf!("The run could not be saved: {e}", "Çalıştırma kaydedilemedi: {e}"))?;
    std::fs::write(run_dir.join("run.json"), text).map_err(|e| crate::i18n::trf!("The run could not be saved: {e}", "Çalıştırma kaydedilemedi: {e}"))
}

pub fn load_record(project_dir: &Path, run_name: &str) -> Option<RunRecord> {
    std::fs::read_to_string(record_path(project_dir, run_name))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
}

/// The latest run of the project (by start time), if any.
pub fn discover_run(project_dir: &Path) -> Option<RunRecord> {
    let runs = project_dir.join("runs");
    let entries = std::fs::read_dir(&runs).ok()?;
    let mut best: Option<RunRecord> = None;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(r) = load_record(project_dir, &name)
            && best.as_ref().is_none_or(|b| r.started_at >= b.started_at)
        {
            best = Some(r);
        }
    }
    best
}

/// A URL-safe run id: `my-training-run-1`.
pub fn slug(project_name: &str, run_name: &str) -> String {
    let mut out = String::new();
    for c in format!("{project_name}-{run_name}").to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

/// "230 MB of audio": the included clips' bytes on disk.
pub fn upload_estimate(clips: &[ClipRef]) -> String {
    let bytes: u64 = clips.iter().filter_map(|c| std::fs::metadata(&c.audio).ok()).map(|m| m.len()).sum();
    if bytes >= 1_000_000 {
        format!("{} MB", bytes / 1_000_000)
    } else {
        format!("{} KB", bytes.max(1) / 1_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn losses_come_from_step_loss_lines() {
        let log = "starting up\nstep 100 loss 2.50 grad ok\nstep 200 loss 1.75 grad ok\nno numbers here\n";
        assert_eq!(parse_losses(log), vec![(100, 2.5), (200, 1.75)]);
        assert!(parse_losses("nothing to see").is_empty());
    }

    #[test]
    fn losses_come_from_eval_lines() {
        // Verbatim shape of the omnivoice trainer log.
        let log = "11/07/2026 11:57:58 - INFO - omnivoice.training.trainer - Running evaluation at step 250...\n\
                   11/07/2026 11:58:01 - INFO - omnivoice.training.trainer - Eval Loss: 3.9556\n\
                   11/07/2026 12:04:46 - INFO - omnivoice.training.trainer - Running evaluation at step 500...\n\
                   11/07/2026 12:04:49 - INFO - omnivoice.training.trainer - Eval Loss: 4.5110\n";
        assert_eq!(parse_losses(log), vec![(250, 3.9556), (500, 4.511)]);
    }

    #[test]
    fn elapsed_reads_plain() {
        assert_eq!(elapsed_text(0, 60), "1 min");
        assert_eq!(elapsed_text(0, 12 * 60), "12 min");
        assert_eq!(elapsed_text(0, 2 * 3600 + 5 * 60), "2 h 05 min");
    }

    #[test]
    fn slugs_are_url_safe() {
        assert_eq!(slug("My Training", "run-1"), "my-training-run-1");
        assert_eq!(slug("Ses 0007!", "run-2"), "ses-0007-run-2");
    }
}
