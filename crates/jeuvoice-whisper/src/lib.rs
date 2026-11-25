//! Whisper transcription for JeuVoice: a `Transcriber` trait over whisper.cpp
//! (through `whisper-rs`) that returns words with times.
#[cfg(feature = "cuda-kernels")]
pub mod cuda;
#[doc(hidden)]
pub mod compare;
pub mod decode;
pub mod ggml;
pub mod mel_cpu;
mod model;
mod words;

pub use model::{download_model, model_path};
pub use words::tokens_to_words;

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use jeuvoice_core::transcript::Transcript;
use whisper_rs::{
    DtwMode, DtwModelPreset, DtwParameters, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters,
};

/// Errors with plain words that a teammate can act on.
#[derive(Debug)]
pub enum WhisperError {
    /// The model file could not be loaded.
    ModelLoad(String),
    /// Whisper failed while it read the audio.
    Transcribe(String),
    /// The model download failed.
    Download(String),
    /// A file or folder could not be read or written.
    Io(String),
}

impl std::fmt::Display for WhisperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelLoad(_) => write!(
                f,
                "The speech model could not be loaded. Download it again from the Settings tab."
            ),
            Self::Transcribe(_) => {
                write!(f, "The transcription failed. Try the file again.")
            }
            Self::Download(_) => write!(
                f,
                "The speech model download failed. Check the internet connection and try again from the Settings tab."
            ),
            Self::Io(_) => {
                write!(f, "A file could not be saved. Check the free disk space.")
            }
        }
    }
}

impl std::error::Error for WhisperError {}

#[derive(Clone, Debug)]
pub struct TranscribeOptions {
    /// A language code such as "tr", or "auto" to detect the language.
    pub language: String,
    /// Not read by `WhisperRs::transcribe`: GPU use is chosen at `WhisperRs::load`.
    pub use_gpu: bool,
}

pub trait Transcriber: Send + Sync {
    /// `progress` gets values from 0.0 to 1.0. If `progress` panics, it is not called again
    /// and `transcribe` returns `WhisperError::Transcribe` after whisper finishes.
    fn transcribe(
        &self,
        samples_16k: &[f32],
        opts: &TranscribeOptions,
        progress: &mut dyn FnMut(f32),
    ) -> Result<Transcript, WhisperError>;
}

/// Returns a fixed transcript. For tests of the app.
pub struct FakeTranscriber(pub Transcript);

impl Transcriber for FakeTranscriber {
    fn transcribe(
        &self,
        _samples_16k: &[f32],
        _opts: &TranscribeOptions,
        progress: &mut dyn FnMut(f32),
    ) -> Result<Transcript, WhisperError> {
        progress(0.0);
        progress(1.0);
        Ok(self.0.clone())
    }
}

pub struct WhisperRs {
    ctx: WhisperContext,
    /// False when the GPU was asked for but the model loaded on the CPU.
    pub on_gpu: bool,
}

impl WhisperRs {
    /// Load the model. If the GPU load fails, load on the CPU and set `on_gpu` to false.
    pub fn load(model: &Path, use_gpu: bool) -> Result<Self, WhisperError> {
        let path = model.to_string_lossy();
        if use_gpu {
            if let Ok(ctx) = WhisperContext::new_with_params(&*path, context_params(true)) {
                return Ok(Self { ctx, on_gpu: true });
            }
        }
        let ctx = WhisperContext::new_with_params(&*path, context_params(false))
            .map_err(|e| WhisperError::ModelLoad(e.to_string()))?;
        Ok(Self { ctx, on_gpu: false })
    }
}

fn context_params(use_gpu: bool) -> WhisperContextParameters<'static> {
    let mut p = WhisperContextParameters::new();
    p.use_gpu(use_gpu);
    p.dtw_parameters(DtwParameters {
        mode: DtwMode::ModelPreset { model_preset: DtwModelPreset::LargeV3Turbo },
        ..Default::default()
    });
    p
}

/// One whisper.cpp token with all its data. For the developer tools that compare the GPU
/// path with whisper.cpp.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct RawToken {
    pub id: i32,
    pub tid: i32,
    pub p: f32,
    pub plog: f32,
    pub pt: f32,
    pub ptsum: f32,
    /// Centiseconds.
    pub t0: i64,
    pub t1: i64,
    /// Centiseconds, -1 when DTW did not set it.
    pub t_dtw: i64,
    pub text: String,
}

/// One whisper.cpp segment. For the developer tools.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct RawSegment {
    /// Centiseconds.
    pub t0: i64,
    pub t1: i64,
    pub no_speech_prob: f32,
    pub tokens: Vec<RawToken>,
}

impl WhisperRs {
    /// Runs whisper.cpp with the app's parameters and returns the detected or given language
    /// and the raw segments. For the developer tools.
    #[doc(hidden)]
    pub fn raw_segments(
        &self,
        samples_16k: &[f32],
        opts: &TranscribeOptions,
    ) -> Result<(String, Vec<RawSegment>), WhisperError> {
        let state = self.run_full(samples_16k, opts, &mut |_| {})?;
        let mut segments = Vec::new();
        for seg in state.as_iter() {
            let mut tokens = Vec::new();
            for i in 0..seg.n_tokens() {
                let Some(tok) = seg.get_token(i) else { continue };
                let d = tok.token_data();
                let text = tok.to_str_lossy().map(|t| t.into_owned()).unwrap_or_default();
                tokens.push(RawToken {
                    id: d.id,
                    tid: d.tid,
                    p: d.p,
                    plog: d.plog,
                    pt: d.pt,
                    ptsum: d.ptsum,
                    t0: d.t0,
                    t1: d.t1,
                    t_dtw: d.t_dtw,
                    text,
                });
            }
            segments.push(RawSegment {
                t0: seg.start_timestamp(),
                t1: seg.end_timestamp(),
                no_speech_prob: seg.no_speech_probability(),
                tokens,
            });
        }
        Ok((self.language_of(&state, opts), segments))
    }

    fn language_of(&self, state: &whisper_rs::WhisperState, opts: &TranscribeOptions) -> String {
        if opts.language == "auto" {
            whisper_rs::get_lang_str(state.full_lang_id_from_state()).unwrap_or("auto").to_string()
        } else {
            opts.language.clone()
        }
    }

    /// Runs `whisper_full` with the app's parameters.
    fn run_full(
        &self,
        samples_16k: &[f32],
        opts: &TranscribeOptions,
        progress: &mut dyn FnMut(f32),
    ) -> Result<whisper_rs::WhisperState, WhisperError> {
        let err = |e: whisper_rs::WhisperError| WhisperError::Transcribe(e.to_string());
        let mut state = self.ctx.create_state().map_err(err)?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(opts.language.as_str()));
        params.set_token_timestamps(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);

        // whisper-rs wants a 'static closure, but `progress` is borrowed. whisper.cpp calls
        // the callback only inside `state.full` below, on this thread, and `params` is not
        // used after that call, so the borrow outlives every use.
        // A panic must not cross the extern "C" trampoline (that aborts the process), so the
        // user closure runs in `catch_unwind`. After a panic it is never called again.
        let panicked = Cell::new(false);
        let mut call = |v: f32| {
            if panicked.get() {
                return;
            }
            if catch_unwind(AssertUnwindSafe(|| progress(v))).is_err() {
                panicked.set(true);
            }
        };
        call(0.0);
        let mut report = |p: i32| call((p as f32 / 100.0).clamp(0.0, 1.0));
        let report: Box<dyn FnMut(i32) + '_> = Box::new(&mut report);
        let report: Box<dyn FnMut(i32) + 'static> = unsafe { std::mem::transmute(report) };
        params.set_progress_callback_safe::<Option<Box<dyn FnMut(i32)>>, Box<dyn FnMut(i32)>>(Some(report));

        state.full(params, samples_16k).map_err(err)?;
        call(1.0);
        if panicked.get() {
            return Err(WhisperError::Transcribe("the progress callback panicked".to_string()));
        }
        Ok(state)
    }
}

impl Transcriber for WhisperRs {
    fn transcribe(
        &self,
        samples_16k: &[f32],
        opts: &TranscribeOptions,
        progress: &mut dyn FnMut(f32),
    ) -> Result<Transcript, WhisperError> {
        let state = self.run_full(samples_16k, opts, progress)?;
        let eot = self.ctx.token_eot();
        let mut tokens = Vec::new();
        for seg in state.as_iter() {
            for i in 0..seg.n_tokens() {
                let Some(tok) = seg.get_token(i) else { continue };
                let data = tok.token_data();
                if data.id >= eot {
                    continue;
                }
                let Ok(text) = tok.to_str_lossy() else { continue };
                // whisper.cpp token times are in centiseconds. `t_dtw` is one point per
                // token and is -1 when it is not set. Then t0 and t1 are used.
                let (start, end) = if data.t_dtw >= 0 {
                    (data.t_dtw, data.t1.max(data.t_dtw))
                } else {
                    (data.t0, data.t1)
                };
                tokens.push((
                    text.into_owned(),
                    start as f64 / 100.0,
                    end as f64 / 100.0,
                    tok.token_probability(),
                ));
            }
        }

        let language = self.language_of(&state, opts);
        Ok(Transcript { language, words: tokens_to_words(&tokens) })
    }
}
