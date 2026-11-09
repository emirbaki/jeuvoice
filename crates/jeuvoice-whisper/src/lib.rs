//! Whisper transcription for JeuVoice: a `Transcriber` trait over whisper.cpp
//! (through `whisper-rs`) that returns words with times.
mod model;
mod words;

pub use model::{download_model, model_path};
pub use words::tokens_to_words;

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
            Self::ModelLoad(e) => write!(
                f,
                "The speech model could not be loaded. Download it again from the Transcribe tab. Detail: {e}"
            ),
            Self::Transcribe(e) => {
                write!(f, "The transcription failed. Try the file again. Detail: {e}")
            }
            Self::Download(e) => write!(
                f,
                "The speech model download failed. Check the internet connection and try again. Detail: {e}"
            ),
            Self::Io(e) => {
                write!(f, "A file could not be saved. Check the free disk space. Detail: {e}")
            }
        }
    }
}

impl std::error::Error for WhisperError {}

pub struct TranscribeOptions {
    /// A language code such as "tr", or "auto" to detect the language.
    pub language: String,
    pub use_gpu: bool,
}

pub trait Transcriber: Send + Sync {
    /// `progress` gets values from 0.0 to 1.0.
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

impl Transcriber for WhisperRs {
    fn transcribe(
        &self,
        samples_16k: &[f32],
        opts: &TranscribeOptions,
        progress: &mut dyn FnMut(f32),
    ) -> Result<Transcript, WhisperError> {
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
        progress(0.0);
        let mut report = |p: i32| progress((p as f32 / 100.0).clamp(0.0, 1.0));
        let report: Box<dyn FnMut(i32) + '_> = Box::new(&mut report);
        let report: Box<dyn FnMut(i32) + 'static> = unsafe { std::mem::transmute(report) };
        params.set_progress_callback_safe::<Option<Box<dyn FnMut(i32)>>, Box<dyn FnMut(i32)>>(Some(report));

        state.full(params, samples_16k).map_err(err)?;
        progress(1.0);

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
                // whisper.cpp token times are in centiseconds.
                tokens.push((
                    text.into_owned(),
                    data.t0 as f64 / 100.0,
                    data.t1 as f64 / 100.0,
                    tok.token_probability(),
                ));
            }
        }

        let language = if opts.language == "auto" {
            whisper_rs::get_lang_str(state.full_lang_id_from_state())
                .unwrap_or("auto")
                .to_string()
        } else {
            opts.language.clone()
        };
        Ok(Transcript { language, words: tokens_to_words(&tokens) })
    }
}
