//! One Whisper worker for the whole app.
//!
//! The Transcribe tab submits `SpeechJob::File`, the Voice tab (Plan 2) submits
//! `SpeechJob::Samples`. Jobs run one at a time, in order, on a worker thread.
//! `set_force_cpu` takes effect before the next job and drops the loaded model,
//! like the GPU switch does.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use jeuvoice_core::audio::{self, to_whisper};
use jeuvoice_core::queue::{Job, Update, spawn_worker};
use jeuvoice_core::transcript::Transcript;
use jeuvoice_whisper::{TranscribeOptions, Transcriber};
use rune::Waker;

use crate::cache;

/// Makes sure the speech model file is on disk before a job runs. It reports (bytes done,
/// bytes total), with a total of 0 when the size is unknown. The error is plain words.
pub type EnsureModel = Arc<dyn Fn(&mut dyn FnMut(u64, u64)) -> Result<(), String> + Send + Sync>;

/// One transcription input.
#[derive(Clone)]
pub enum SpeechJob {
    /// A dropped file (the Transcribe tab).
    File { path: PathBuf, opts: TranscribeOptions },
    /// Decoded samples with a tag that tells Voice paragraphs apart (Plan 2).
    /// The tag travels with the job; the caller maps the ticket back to its paragraph.
    #[allow(dead_code)]
    Samples {
        samples_16k: Arc<Vec<f32>>,
        #[allow(dead_code)]
        tag: u64,
        opts: TranscribeOptions,
    },
}

/// News from a running job that belongs to no queue row.
pub enum ServiceEvent {
    /// Speech model download progress in percent, `None` when the size is unknown.
    Download(Option<u8>),
    /// A line for the user: a file saved, or why it was not.
    Notice(String),
}

/// A submitted job. Maps back to a queue row in the tab that submitted it.
pub type JobTicket = usize;

pub struct SpeechService {
    transcriber: Arc<dyn Transcriber>,
    ensure_model: Option<EnsureModel>,
    cache_dir: Option<PathBuf>,
    events_tx: mpsc::Sender<ServiceEvent>,
    events: mpsc::Receiver<ServiceEvent>,
    waker: Waker,
    jobs: Option<mpsc::Sender<(usize, Job)>>,
    updates_tx: mpsc::Sender<(usize, Update)>,
    updates: mpsc::Receiver<(usize, Update)>,
    next: usize,
    force_cpu: bool,
    unload_pending: bool,
}

impl SpeechService {
    pub fn new(transcriber: Arc<dyn Transcriber>, ensure_model: Option<EnsureModel>, cache_dir: Option<PathBuf>) -> Self {
        let (updates_tx, updates) = mpsc::channel();
        let (events_tx, events) = mpsc::channel();
        Self {
            transcriber,
            ensure_model,
            cache_dir,
            events_tx,
            events,
            waker: Arc::new(|| {}),
            jobs: None,
            updates_tx,
            updates,
            next: 0,
            force_cpu: false,
            unload_pending: false,
        }
    }

    /// Starts the worker with the window's waker. No job runs before this.
    pub fn setup(&mut self, waker: Waker) {
        let (jobs_tx, jobs_rx) = mpsc::channel();
        let wake = waker.clone();
        spawn_worker(jobs_rx, self.updates_tx.clone(), move || wake());
        self.jobs = Some(jobs_tx);
        self.waker = waker;
    }

    pub fn ready(&self) -> bool {
        self.jobs.is_some()
    }

    /// Queue one job. The CPU flag applies to this and later jobs until it changes.
    pub fn submit(&mut self, job: SpeechJob) -> JobTicket {
        let ticket = self.next;
        self.next += 1;
        let mut job = job;
        if self.force_cpu {
            match &mut job {
                SpeechJob::File { opts, .. } | SpeechJob::Samples { opts, .. } => opts.use_gpu = false,
            }
        }
        let core_job = self.build(job);
        // `ready` is checked by the caller: the channel only closes when the app does.
        let _ = self.jobs.as_ref().expect("SpeechService::setup ran").send((ticket, core_job));
        ticket
    }

    /// All worker updates since the last call.
    pub fn poll(&mut self) -> Vec<(JobTicket, Update)> {
        self.updates.try_iter().collect()
    }

    /// Download progress and notices since the last call.
    pub fn drain_events(&mut self) -> Vec<ServiceEvent> {
        self.events.try_iter().collect()
    }

    /// Force the CPU from the next job on. Drops the loaded model, like the GPU switch.
    pub fn set_force_cpu(&mut self, on: bool) {
        if self.force_cpu != on {
            self.force_cpu = on;
            self.unload_pending = true;
        }
    }

    /// Drop the loaded model before the next job (the GPU choice changed).
    pub fn drop_loaded_model(&mut self) {
        self.unload_pending = true;
    }

    /// Whether the loaded model must go, once nothing runs. Clears the flag.
    pub fn take_unload_pending(&mut self) -> bool {
        std::mem::take(&mut self.unload_pending)
    }

    fn build(&self, job: SpeechJob) -> Job {
        let transcriber = self.transcriber.clone();
        let ensure_model = self.ensure_model.clone();
        let cache_dir = self.cache_dir.clone();
        let (events_tx, waker) = (self.events_tx.clone(), self.waker.clone());
        let say = move |e: ServiceEvent| {
            let _ = events_tx.send(e);
            waker();
        };
        let (path, samples, opts) = match job {
            SpeechJob::File { path, opts } => (Some(path), None, opts),
            SpeechJob::Samples { samples_16k, tag: _, opts } => (None, Some(samples_16k), opts),
        };
        Box::new(move |progress| {
            if let Some(ensure) = ensure_model {
                let mut last = None;
                ensure(&mut |done, total| {
                    let pct = (total > 0).then(|| (done.saturating_mul(100) / total).min(100) as u8);
                    if last != Some(pct) {
                        last = Some(pct);
                        say(ServiceEvent::Download(pct));
                    }
                })?;
            }
            match (path, samples) {
                (Some(path), _) => {
                    let a = audio::decode(&path).map_err(|e| e.to_string())?;
                    let t = transcribe_samples(&transcriber, &to_whisper(&a), &opts, progress)?;
                    if let Some(dir) = cache_dir
                        && let Err(e) = cache::save(&dir, &path, &opts.language, &t)
                    {
                        say(ServiceEvent::Notice(e));
                    }
                    Ok(t)
                }
                (_, Some(samples_16k)) => transcribe_samples(&transcriber, &samples_16k, &opts, progress),
                (None, None) => unreachable!("a speech job is a file or samples"),
            }
        })
    }
}

fn transcribe_samples(
    transcriber: &Arc<dyn Transcriber>,
    samples_16k: &[f32],
    opts: &TranscribeOptions,
    progress: &mut dyn FnMut(f32),
) -> Result<Transcript, String> {
    transcriber.transcribe(samples_16k, opts, progress).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jeuvoice_whisper::FakeTranscriber;

    /// A samples job (the Voice path) runs to Done without a file or a model.
    #[test]
    fn samples_job_runs_to_done() {
        let mut s = SpeechService::new(Arc::new(FakeTranscriber(Transcript::default())), None, None);
        s.setup(Arc::new(|| {}));
        let ticket = s.submit(SpeechJob::Samples {
            samples_16k: Arc::new(vec![0.0; 160]),
            tag: 7,
            opts: TranscribeOptions { language: "tr".into(), use_gpu: true },
        });
        let start = std::time::Instant::now();
        loop {
            for (t, u) in s.poll() {
                assert_eq!(t, ticket);
                if matches!(u, Update::Done(_)) {
                    return;
                }
            }
            assert!(start.elapsed() < std::time::Duration::from_secs(10), "the job finishes");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    #[test]
    fn force_cpu_unloads_once_per_change() {
        let mut s = SpeechService::new(Arc::new(FakeTranscriber(Transcript::default())), None, None);
        assert!(!s.take_unload_pending());
        s.set_force_cpu(true);
        assert!(s.take_unload_pending());
        assert!(!s.take_unload_pending());
        s.set_force_cpu(true);
        assert!(!s.take_unload_pending(), "no change, no unload");
        s.set_force_cpu(false);
        assert!(s.take_unload_pending());
    }
}
