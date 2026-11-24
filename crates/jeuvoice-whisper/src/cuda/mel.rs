//! The log-mel spectrogram on the GPU, the same math as `mel_cpu` (whisper.cpp).
//!
//! `compute` makes the mel of a whole file: the max for the clamp is over the whole file,
//! as in whisper.cpp. `window_into` then copies one normalized 3000-frame window into the
//! encoder input. The work runs in chunks of `CHUNK` frames, so a long file needs only the
//! `[n_len, 128]` result in memory (about 194 MB for one hour), not every frame.

use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::ops::{log10_max, log10_max_partials, mel_frames, mel_window, power_spectrum};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, Tensor};

use crate::WhisperError;
use crate::mel_cpu::{HOP, N_BINS, N_FFT, N_MEL, frame_count, hann, padded};

/// Frames for each chunk of the frame, spectrum and power steps.
const CHUNK: usize = 3000;

/// The mel of the last `compute` call.
struct Computed {
    arena: Arena,
    /// `[n_len, N_MEL]`: frames as rows, after log10.
    mel_t: Tensor,
    /// `[1]`: the largest value of `mel_t`.
    max: Tensor,
    n_len: usize,
}

pub struct GpuMel {
    /// The DFT matrix and the mel filters.
    consts: Arena,
    /// `[N_FFT, 2 · N_BINS]`: `cos(2πkj/N)` in columns `0..N_BINS`, `−sin(2πkj/N)` after.
    dft: Tensor,
    /// `[N_MEL, N_BINS]`, from the model file.
    filters: Tensor,
    hann: Vec<f32>,
    computed: Option<Computed>,
}

impl GpuMel {
    /// Uploads the DFT matrix and `filters` (`[128, 201]`, from the model file).
    pub fn new(gpu: &Gpu, filters: &[f32]) -> Result<GpuMel, WhisperError> {
        if filters.len() != N_MEL * N_BINS {
            return Err(WhisperError::ModelLoad(format!("the mel filters have {} values, not 128 x 201", filters.len())));
        }
        let mut plan = ArenaPlan::new();
        let t_dft = plan.add(DType::F32, &[N_FFT, 2 * N_BINS]);
        let t_filters = plan.add(DType::F32, &[N_MEL, N_BINS]);
        let consts = Arena::new(gpu, &plan)?;
        let mut d = vec![0.0f32; N_FFT * 2 * N_BINS];
        for j in 0..N_FFT {
            for k in 0..N_BINS {
                let a = 2.0 * std::f64::consts::PI * ((k * j) % N_FFT) as f64 / N_FFT as f64;
                d[j * 2 * N_BINS + k] = a.cos() as f32;
                d[j * 2 * N_BINS + N_BINS + k] = -a.sin() as f32;
            }
        }
        consts.upload(gpu, &t_dft, &d)?;
        consts.upload(gpu, &t_filters, filters)?;
        let hann = hann().iter().map(|&v| v as f32).collect();
        Ok(GpuMel { consts, dft: t_dft, filters: t_filters, hann, computed: None })
    }

    /// Computes the mel of all `samples` (16 kHz mono) and returns the frame count `n_len`.
    /// It replaces the result of the last call.
    pub fn compute(&mut self, gpu: &Gpu, blas: &Blas, samples: &[f32]) -> Result<usize, WhisperError> {
        self.computed = None; // free the old result before the new allocation
        let n_len = frame_count(samples.len());
        let p = padded(samples);
        let chunk = CHUNK.min(n_len);

        let mut plan = ArenaPlan::new();
        let t_pad = plan.add(DType::F32, &[p.len()]);
        let t_hann = plan.add(DType::F32, &[N_FFT]);
        let t_frames = plan.add(DType::F32, &[chunk, N_FFT]);
        let t_spec = plan.add(DType::F32, &[chunk, 2 * N_BINS]);
        let t_power = plan.add(DType::F32, &[chunk, N_BINS]);
        let t_mel = plan.add(DType::F32, &[n_len, N_MEL]);
        let t_part = plan.add(DType::F32, &[log10_max_partials(n_len * N_MEL)]);
        let t_max = plan.add(DType::F32, &[1]);
        let arena = Arena::new(gpu, &plan)?;
        arena.upload(gpu, &t_pad, &p)?;
        arena.upload(gpu, &t_hann, &self.hann)?;

        let (dft, filters) = (self.consts.slice::<f32>(&self.dft), self.consts.slice::<f32>(&self.filters));
        let (frames, spec, power, mel) =
            (arena.slice::<f32>(&t_frames), arena.slice::<f32>(&t_spec), arena.slice::<f32>(&t_power), arena.slice::<f32>(&t_mel));
        let mut first = 0;
        while first < n_len {
            let count = chunk.min(n_len - first);
            // 1. Frames times the Hann window.
            mel_frames(gpu, &arena, &t_pad, &t_hann, &t_frames, HOP, first, count)?;
            // 2. The DFT as a GEMM: spec [count, 402] = frames [count, 400] · dft [400, 402].
            blas.gemm_f32(gpu, &frames.range(0, count * N_FFT), Op::N, &dft, Op::N, &spec.range(0, count * 2 * N_BINS), count, 2 * N_BINS, N_FFT, 1.0, 0.0)?;
            // 3. The power of bins 0 to 200 (rows past `count` are not used).
            power_spectrum(gpu, &arena, &t_spec, &t_power)?;
            // 4. The mel projection: mel rows [count, 128] = power [count, 201] · filtersᵀ.
            blas.gemm_f32(gpu, &power.range(0, count * N_BINS), Op::N, &filters, Op::T, &mel.range(first * N_MEL, count * N_MEL), count, N_MEL, N_BINS, 1.0, 0.0)?;
            first += count;
        }
        // 5. log10 and the max over the whole file.
        log10_max(gpu, &arena, &t_mel, &t_part, &t_max)?;
        self.computed = Some(Computed { arena, mel_t: t_mel, max: t_max, n_len });
        Ok(n_len)
    }

    /// The frame count of the last `compute`, or 0.
    pub fn n_len(&self) -> usize {
        self.computed.as_ref().map_or(0, |c| c.n_len)
    }

    /// Writes the normalized window that starts at `start_frame` into `out` (`[128, w]` `F16`
    /// in `arena`). Frames past the end get the floor value, as the whisper.cpp padding does.
    ///
    /// # Panics
    /// If `compute` did not run first.
    pub fn window_into(&self, gpu: &Gpu, start_frame: usize, arena: &Arena, out: &Tensor) -> Result<(), WhisperError> {
        let c = self.computed.as_ref().expect("GpuMel::window_into needs compute first");
        mel_window(gpu, &c.arena, &c.mel_t, &c.max, start_frame, arena, out)?;
        Ok(())
    }
}
