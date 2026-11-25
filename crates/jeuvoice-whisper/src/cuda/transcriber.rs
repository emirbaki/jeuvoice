//! `CudaWhisper`: the `Transcriber` on the Rust kernels.
//!
//! The GPU objects belong to one thread: `Blas` is not `Send`, and the CUDA context is
//! bound to the thread that uses it. So `CudaWhisper` starts one worker thread. The worker
//! loads the model, then waits for jobs. `transcribe` sends the samples to the worker and
//! gets back progress messages and the result over a channel. The caller thread calls the
//! progress callback, so the callback needs no `Send`. This makes `CudaWhisper`
//! `Send + Sync` (deferred minor M6 of the Phase 1a review).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use jeuvoice_core::transcript::Transcript;

use super::model::GpuWhisper;
use crate::decode::vocab::Vocab;
use crate::decode::window::{DecodeParams, Transcription, transcribe_tokens};
use crate::ggml::GgmlModel;
use crate::{TranscribeOptions, Transcriber, WhisperError, tokens_to_words};

/// One transcription for the worker.
struct Job {
    samples: Vec<f32>,
    language: String,
    reply: Sender<Msg>,
}

/// A message from the worker to the caller of `transcribe`.
enum Msg {
    Progress(f32),
    /// The transcript and the number of windows that fell back to a higher temperature.
    Done(Result<(Transcript, usize), WhisperError>),
}

pub struct CudaWhisper {
    /// `None` only while `drop` runs.
    jobs: Mutex<Option<Sender<Job>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    /// The fallback windows of the last transcription (watch item W2 of the plan).
    last_fallbacks: AtomicUsize,
}

impl CudaWhisper {
    /// Starts the worker thread and loads the model from `model_path` on GPU 0. Every
    /// failure (no file, a model that is not large-v3, no GPU, no cuBLAS, too little video
    /// memory) gives `WhisperError::ModelLoad`. The app then uses whisper-rs.
    pub fn load(model_path: &Path) -> Result<CudaWhisper, WhisperError> {
        let path = model_path.to_path_buf();
        let (jobs_tx, jobs_rx) = channel::<Job>();
        let (ready_tx, ready_rx) = channel::<Result<(), WhisperError>>();
        let worker = std::thread::Builder::new()
            .name("jeuvoice-cuda-whisper".into())
            .spawn(move || worker(&path, &ready_tx, &jobs_rx))
            .map_err(|e| WhisperError::ModelLoad(format!("could not start the GPU worker thread: {e}")))?;
        let ready = ready_rx.recv().unwrap_or_else(|_| Err(WhisperError::ModelLoad("the GPU worker stopped while it loaded the model".into())));
        if let Err(e) = ready {
            let _ = worker.join();
            return Err(e);
        }
        Ok(CudaWhisper { jobs: Mutex::new(Some(jobs_tx)), worker: Mutex::new(Some(worker)), last_fallbacks: AtomicUsize::new(0) })
    }

    /// The windows of the last transcription that fell back to a temperature above 0. For
    /// the measurement tool.
    #[doc(hidden)]
    pub fn last_fallback_windows(&self) -> usize {
        self.last_fallbacks.load(Ordering::Relaxed)
    }
}

/// Any error of the load becomes `ModelLoad`, with its detail kept for the log.
fn as_load_error(e: WhisperError) -> WhisperError {
    match e {
        WhisperError::ModelLoad(s) | WhisperError::Transcribe(s) | WhisperError::Download(s) | WhisperError::Io(s) => WhisperError::ModelLoad(s),
    }
}

/// The worker thread: loads the model, reports the result, then runs the jobs in order
/// until `CudaWhisper` drops its sender.
fn worker(path: &Path, ready: &Sender<Result<(), WhisperError>>, jobs: &Receiver<Job>) {
    let loaded = catch_unwind(AssertUnwindSafe(|| -> Result<(Vocab, GpuWhisper), WhisperError> {
        let model = GgmlModel::open(path)?;
        let vocab = Vocab::from_ggml(&model)?;
        let gw = GpuWhisper::load(&model)?;
        Ok((vocab, gw))
    }));
    let (vocab, mut gw) = match loaded {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            let _ = ready.send(Err(as_load_error(e)));
            return;
        }
        Err(_) => {
            let _ = ready.send(Err(WhisperError::ModelLoad("the GPU worker failed while it loaded the model".into())));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    while let Ok(job) = jobs.recv() {
        let reply = job.reply.clone();
        let r = catch_unwind(AssertUnwindSafe(|| run_job(&vocab, &mut gw, &job))).unwrap_or_else(|_| {
            Err(WhisperError::Transcribe("the GPU worker failed during the transcription".into()))
        });
        // The caller may be gone. The next job still runs.
        let _ = reply.send(Msg::Done(r));
    }
}

fn run_job(vocab: &Vocab, gw: &mut GpuWhisper, job: &Job) -> Result<(Transcript, usize), WhisperError> {
    gw.set_audio(&job.samples)?;
    let params = DecodeParams { language: job.language.clone(), ..DecodeParams::default() };
    let reply = &job.reply;
    let t = transcribe_tokens(gw, vocab, &job.samples, &params, &mut |p| {
        let _ = reply.send(Msg::Progress(p));
    })?;
    let fallbacks = t.windows.iter().filter(|w| w.attempts > 1).count();
    Ok((to_transcript(vocab, &t), fallbacks))
}

/// The words of a transcription, with the token rule of `WhisperRs::transcribe`: skip the
/// tokens at and above EOT, start at `t_dtw` when it is set (else `t0`), end at
/// `max(t1, t_dtw)` (else `t1`), times in centiseconds.
fn to_transcript(vocab: &Vocab, t: &Transcription) -> Transcript {
    let mut tokens = Vec::new();
    for tok in t.segments.iter().flat_map(|s| &s.tokens) {
        if tok.id >= vocab.eot {
            continue;
        }
        let (start, end) = if tok.t_dtw >= 0 { (tok.t_dtw, tok.t1.max(tok.t_dtw)) } else { (tok.t0, tok.t1) };
        tokens.push((vocab.token_text_lossy(tok.id), start as f64 / 100.0, end as f64 / 100.0, tok.p));
    }
    Transcript { language: t.language.clone(), words: tokens_to_words(&tokens) }
}

impl Transcriber for CudaWhisper {
    fn transcribe(&self, samples_16k: &[f32], opts: &TranscribeOptions, progress: &mut dyn FnMut(f32)) -> Result<Transcript, WhisperError> {
        // The same progress rule as WhisperRs: 0.0 first, 1.0 at the end, and after a panic
        // of the callback no more calls and an error at the end.
        let mut panicked = false;
        let mut call = |v: f32| {
            if !panicked && catch_unwind(AssertUnwindSafe(|| progress(v))).is_err() {
                panicked = true;
            }
        };
        call(0.0);
        let (reply_tx, reply_rx) = channel();
        {
            let jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
            let jobs = jobs.as_ref().expect("the sender exists until drop");
            let job = Job { samples: samples_16k.to_vec(), language: opts.language.clone(), reply: reply_tx };
            jobs.send(job).map_err(|_| WhisperError::Transcribe("the GPU worker stopped".into()))?;
        }
        let result = loop {
            match reply_rx.recv() {
                Ok(Msg::Progress(p)) => call(p.clamp(0.0, 1.0)),
                Ok(Msg::Done(r)) => break r,
                Err(_) => break Err(WhisperError::Transcribe("the GPU worker stopped".into())),
            }
        };
        let (t, fallbacks) = result?;
        self.last_fallbacks.store(fallbacks, Ordering::Relaxed);
        call(1.0);
        if panicked {
            return Err(WhisperError::Transcribe("the progress callback panicked".into()));
        }
        Ok(t)
    }
}

impl Drop for CudaWhisper {
    fn drop(&mut self) {
        // Without a sender, the worker leaves its loop and frees the GPU objects on its own
        // thread.
        self.jobs.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(h) = self.worker.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = h.join();
        }
    }
}
