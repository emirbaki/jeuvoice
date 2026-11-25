//! `GpuWhisper`: the GPU mel, the encoder and the decoder together, as the `WhisperModel`
//! of the window loop (`decode::window`).

use jeuvoice_gpu::Gpu;
use jeuvoice_gpu::blas::Blas;

use super::decoder::Decoder;
use super::encoder::Encoder;
use super::mel::GpuMel;
use crate::WhisperError;
use crate::decode::window::WhisperModel;
use crate::ggml::GgmlModel;

/// The whole Whisper model on one GPU. It is not `Send`: `Blas` belongs to the thread that
/// made it. `CudaWhisper` (Task 11) runs it on one worker thread.
pub struct GpuWhisper {
    gpu: Gpu,
    blas: Blas,
    mel: GpuMel,
    encoder: Encoder,
    decoder: Decoder,
    has_audio: bool,
}

impl GpuWhisper {
    /// Opens GPU 0, loads cuBLAS, and uploads the mel filters, the encoder and the decoder
    /// of `model`.
    pub fn load(model: &GgmlModel) -> Result<GpuWhisper, WhisperError> {
        model.check_gpu_encoder()?;
        model.check_large_v3_turbo()?;
        let gpu = Gpu::open(0)?;
        let blas = Blas::load(&gpu)?;
        let mel = GpuMel::new(&gpu, &model.mel_filters.2)?;
        let encoder = Encoder::load(&gpu, model)?;
        let decoder = Decoder::load(&gpu, model)?;
        Ok(GpuWhisper { gpu, blas, mel, encoder, decoder, has_audio: false })
    }

    /// Computes the mel of the whole audio. Call it before the window loop.
    pub fn set_audio(&mut self, samples: &[f32]) -> Result<(), WhisperError> {
        self.mel.compute(&self.gpu, &self.blas, samples)?;
        self.has_audio = true;
        Ok(())
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The video memory of the encoder and the decoder (weights, activations, caches).
    /// The mel needs more for each file, and the CUDA context and cuBLAS need their own.
    pub fn model_bytes(&self) -> usize {
        self.encoder.weight_bytes() + self.encoder.activation_bytes() + self.decoder.weight_bytes() + self.decoder.activation_bytes()
    }
}

impl WhisperModel for GpuWhisper {
    fn encode(&mut self, seek: usize) -> Result<(), WhisperError> {
        if !self.has_audio {
            return Err(WhisperError::Transcribe("the GPU model has no audio: call set_audio first".into()));
        }
        let (arena, input) = self.encoder.input();
        self.mel.window_into(&self.gpu, seek, arena, input)?;
        let out = self.encoder.run(&self.gpu, &self.blas)?.clone();
        self.decoder.set_cross_kv(&self.gpu, &self.blas, self.encoder.arena(), &out)
    }

    fn decode_first(&mut self, tokens: &[u32]) -> Result<Vec<f32>, WhisperError> {
        self.decoder.decode_first(&self.gpu, &self.blas, tokens)
    }

    fn decode_next(&mut self, token: u32) -> Result<Vec<f32>, WhisperError> {
        self.decoder.decode_next(&self.gpu, &self.blas, token)
    }

    fn alignment(&mut self, tokens: &[u32], n_audio: usize) -> Result<Vec<f32>, WhisperError> {
        self.decoder.alignment(&self.gpu, &self.blas, tokens, n_audio)
    }
}
