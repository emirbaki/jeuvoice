//! The log-mel spectrogram on the CPU: a direct port of whisper.cpp `log_mel_spectrogram`
//! (whisper-rs-sys 0.15.0, `whisper.cpp/src/whisper.cpp`, line 3170). It is the test oracle
//! for the GPU mel (`cuda::mel`), so it favors clarity and f64 math over speed.
//!
//! The steps, as whisper.cpp does them:
//! 1. Pad: 200 samples of reflection at the start (`x[200], x[199], ..., x[1]`), then the
//!    samples, then 480,000 + 200 zeros (30 s of silence and half a frame).
//! 2. Frames of 400 samples every 160 samples: `n_len = (n + 480,000) / 160` frames.
//! 3. Each frame times a periodic Hann window, then the power spectrum of bins 0 to 200.
//! 4. Mel band `j` = `Σ_k power[k] · filters[j][k]`, then `log10(max(v, 1e-10))`.
//! 5. Over the whole file: `max`, then every value below `max - 8` becomes `max - 8`, then
//!    `(v + 4) / 4`.

/// Mel bands of large-v3 and large-v3-turbo.
pub const N_MEL: usize = 128;
/// Window size of one frame (25 ms at 16 kHz).
pub const N_FFT: usize = 400;
/// Bins 0 to 200 of the spectrum.
pub const N_BINS: usize = N_FFT / 2 + 1;
/// Step between frames (10 ms at 16 kHz).
pub const HOP: usize = 160;
/// Half a frame of padding at each end.
pub const PAD: usize = N_FFT / 2;
/// 30 s of zeros at the end.
pub const TAIL: usize = 16_000 * 30;

/// Number of mel frames whisper.cpp makes for `n` samples.
pub fn frame_count(n: usize) -> usize {
    (n + TAIL) / HOP
}

/// The padded signal: reflection at the start, zeros at the end. A sample that the
/// reflection needs but the input does not have (input shorter than 201 samples) is zero.
/// (whisper.cpp reads past its buffer there.)
pub fn padded(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    let mut p = vec![0.0f32; n + TAIL + 2 * PAD];
    for i in 0..PAD {
        // p[i] = samples[PAD - i]
        if let Some(&v) = samples.get(PAD - i) {
            p[i] = v;
        }
    }
    p[PAD..PAD + n].copy_from_slice(samples);
    p
}

/// The periodic Hann window of whisper.cpp: `0.5 · (1 − cos(2πi / N))`.
pub fn hann() -> Vec<f64> {
    (0..N_FFT).map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos())).collect()
}

/// The normalized log-mel spectrogram, `[N_MEL, n_len]` row-major, and `n_len`.
/// `filters` is the `[N_MEL, N_BINS]` filter bank from the model file.
pub fn log_mel(samples: &[f32], filters: &[f32]) -> (Vec<f32>, usize) {
    assert_eq!(filters.len(), N_MEL * N_BINS, "log_mel: filters must be [128, 201]");
    let n_len = frame_count(samples.len());
    let p = padded(samples);
    let w = hann();
    // cos and sin of 2π m / N for m in 0..N: bin k at sample j uses m = (k · j) mod N.
    let (cos, sin): (Vec<f64>, Vec<f64>) =
        (0..N_FFT).map(|m| (2.0 * std::f64::consts::PI * m as f64 / N_FFT as f64).sin_cos()).map(|(s, c)| (c, s)).unzip();

    // whisper.cpp computes frames only while they start inside the samples plus the start
    // padding. All later frames are silence: log10(1e-10) = -10.
    let n_signal = samples.len() + PAD;
    let n_computed = (n_signal / HOP + 1).min(n_len);
    let mut mel = vec![-10.0f64; N_MEL * n_len];
    let mut frame = vec![0.0f64; N_FFT];
    let mut power = vec![0.0f64; N_BINS];
    for i in 0..n_computed {
        let off = i * HOP;
        for j in 0..N_FFT {
            // whisper.cpp fills the part past `n_signal` with zeros. The padded buffer is
            // zero there already, except the end padding, which is zero too.
            frame[j] = if off + j < n_signal { w[j] * p[off + j] as f64 } else { 0.0 };
        }
        for (k, pk) in power.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (j, &f) in frame.iter().enumerate() {
                let m = (k * j) % N_FFT;
                re += f * cos[m];
                im -= f * sin[m];
            }
            *pk = re * re + im * im;
        }
        for b in 0..N_MEL {
            let row = &filters[b * N_BINS..(b + 1) * N_BINS];
            let sum: f64 = power.iter().zip(row).map(|(p, &f)| p * f as f64).sum();
            mel[b * n_len + i] = sum.max(1e-10).log10();
        }
    }

    let max = mel.iter().copied().fold(f64::MIN, f64::max);
    let floor = max - 8.0;
    let out = mel.iter().map(|&v| ((v.max(floor) + 4.0) / 4.0) as f32).collect();
    (out, n_len)
}
