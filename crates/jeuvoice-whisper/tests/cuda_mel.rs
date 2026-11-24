// The GPU mel against the CPU port (plan 1a, Task 9). Run with:
//   scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels
#![cfg(feature = "cuda-kernels")]

use jeuvoice_gpu::blas::Blas;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};
use jeuvoice_whisper::cuda::mel::GpuMel;
use jeuvoice_whisper::mel_cpu::{N_BINS, N_MEL, log_mel};

const WINDOW: usize = 3000;

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// A filter bank like Whisper's: overlapping triangles over the 201 bins.
fn triangle_filters() -> Vec<f32> {
    let mut f = vec![0.0f32; N_MEL * N_BINS];
    for b in 0..N_MEL {
        let center = 1.0 + b as f32 * (N_BINS as f32 - 3.0) / N_MEL as f32;
        for k in 0..N_BINS {
            f[b * N_BINS + k] = (1.0 - (k as f32 - center).abs() / 2.0).max(0.0) * 0.01;
        }
    }
    f
}

/// Computes the GPU mel of `samples` and returns the windows at `starts`, as f32.
fn gpu_windows(samples: &[f32], filters: &[f32], starts: &[usize]) -> (usize, Vec<Vec<f32>>) {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let mut mel = GpuMel::new(&gpu, filters).unwrap();
    let n_len = mel.compute(&gpu, &blas, samples).unwrap();
    let mut plan = ArenaPlan::new();
    let out = plan.add(DType::F16, &[N_MEL, WINDOW]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    let windows = starts
        .iter()
        .map(|&s| {
            mel.window_into(&gpu, s, &arena, &out).unwrap();
            arena.download::<half::f16>(&gpu, &out).unwrap().iter().map(|v| v.to_f32()).collect()
        })
        .collect();
    (n_len, windows)
}

/// The CPU window at `start`, with the floor value past the end (as the GPU does).
fn cpu_window(mel: &[f32], n_len: usize, start: usize) -> Vec<f32> {
    let floor = mel.iter().copied().fold(f32::MAX, f32::min);
    (0..N_MEL * WINDOW)
        .map(|i| {
            let (b, t) = (i / WINDOW, i % WINDOW);
            if start + t < n_len { mel[b * n_len + start + t] } else { floor }
        })
        .collect()
}

#[test]
fn gpu_mel_matches_cpu() {
    // 20 s of noise. The second window starts at frame 2000, so it holds the end of the
    // sound and the silence after it.
    let samples = rnd(16_000 * 20, 51, 0.3);
    let filters = triangle_filters();
    let (cpu, n_len_cpu) = log_mel(&samples, &filters);
    let (n_len, windows) = gpu_windows(&samples, &filters, &[0, 2000]);
    assert_eq!(n_len, n_len_cpu);
    for (w, start) in windows.iter().zip([0, 2000]) {
        assert_close(&format!("gpu mel window at {start}"), w, &cpu_window(&cpu, n_len, start), Tol { abs: 2e-2, rel: 2e-2 });
    }
}

#[test]
fn gpu_mel_short_and_empty_audio() {
    for n in [0usize, 1] {
        let (n_len, windows) = gpu_windows(&vec![0.0; n], &triangle_filters(), &[0]);
        assert_eq!(n_len, 3000);
        assert!(windows[0].iter().all(|&v| v == -1.5), "{n} samples: not all -1.5");
    }
}

#[test]
#[ignore = "needs the model file and the reference samples (tools/reference)"]
fn gpu_mel_matches_cpu_on_speech() {
    // my_voice2 is 33.6 s: longer than one window, so the max covers more than one window.
    let model = jeuvoice_whisper::ggml::GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/reference/whisper/my_voice2.f32");
    let bytes = std::fs::read(&path).unwrap();
    let samples: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let (cpu, n_len) = log_mel(&samples, &model.mel_filters.2);
    let starts = [0, 3000];
    let (_, windows) = gpu_windows(&samples, &model.mel_filters.2, &starts);
    for (w, start) in windows.iter().zip(starts) {
        assert_close(&format!("speech window at {start}"), w, &cpu_window(&cpu, n_len, start), Tol { abs: 2e-2, rel: 2e-2 });
    }
}

#[test]
#[ignore = "timing for lesson 7: run in release with --nocapture"]
fn mel_timing_and_f16_error() {
    // One 30 s window of speech: the GPU mel against the CPU port.
    let model = jeuvoice_whisper::ggml::GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/reference/whisper/my_voice2.f32");
    let bytes = std::fs::read(&path).unwrap();
    let all: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let samples = &all[..16_000 * 30];
    let filters = &model.mel_filters.2;

    let t = std::time::Instant::now();
    let (cpu, n_len) = log_mel(samples, filters);
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3;

    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let mut mel = GpuMel::new(&gpu, filters).unwrap();
    let mut plan = ArenaPlan::new();
    let out = plan.add(DType::F16, &[N_MEL, WINDOW]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    mel.compute(&gpu, &blas, samples).unwrap(); // warm-up: kernel load, cuBLAS init
    let runs = 10;
    let t = std::time::Instant::now();
    for _ in 0..runs {
        mel.compute(&gpu, &blas, samples).unwrap();
        mel.window_into(&gpu, 0, &arena, &out).unwrap();
        let _ = arena.download::<half::f16>(&gpu, &out).unwrap();
    }
    let gpu_ms = t.elapsed().as_secs_f64() * 1e3 / runs as f64;
    eprintln!("30 s window: CPU {cpu_ms:.1} ms, GPU {gpu_ms:.2} ms (with upload, allocation and download)");

    // The f16 question: round the samples to f16, as an f16 frame buffer would, and see
    // how far the normalized mel moves.
    let s16: Vec<f32> = samples.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
    let (cpu16, _) = log_mel(&s16, filters);
    let mut worst = 0.0f32;
    let mut over = 0usize;
    for (a, b) in cpu.iter().zip(&cpu16) {
        let d = (a - b).abs();
        worst = worst.max(d);
        if d > 0.02 {
            over += 1;
        }
    }
    eprintln!("f16 samples: max mel difference {worst:.4}, {over} of {} values over 0.02 (n_len {n_len})", cpu.len());
}
