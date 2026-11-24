// The GPU pieces of the log-mel spectrogram (plan 1a, Task 9). jeuvoice-whisper puts them
// together in `cuda::mel` and checks the whole result against its CPU port.

use half::f16;
use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{log10_max, log10_max_partials, mel_frames, mel_window, power_spectrum};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

#[test]
fn gemm_f32_matches_cpu() {
    // The DFT shape: frames [m, 400] times the DFT matrix [400, 402], and a Linear-style
    // B^T case like the mel projection power [m, 201] times filters^T (filters [128, 201]).
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    for (m, n, k, op_b) in [(37, 402, 400, Op::N), (37, 128, 201, Op::T)] {
        let a = rnd(m * k, 1, 1.0);
        let b = rnd(k * n, 2, 1.0);
        let mut plan = ArenaPlan::new();
        let ta = plan.add(DType::F32, &[m, k]);
        let b_shape = if op_b == Op::N { [k, n] } else { [n, k] };
        let tb = plan.add(DType::F32, &b_shape);
        let tc = plan.add(DType::F32, &[m, n]);
        let arena = Arena::new(&gpu, &plan).unwrap();
        arena.upload(&gpu, &ta, &a).unwrap();
        arena.upload(&gpu, &tb, &b).unwrap();
        blas.gemm_f32(&gpu, &arena.slice(&ta), Op::N, &arena.slice(&tb), op_b, &arena.slice(&tc), m, n, k, 1.0, 0.0).unwrap();
        let got = arena.download::<f32>(&gpu, &tc).unwrap();
        let want: Vec<f32> = (0..m * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                (0..k)
                    .map(|p| a[i * k + p] as f64 * if op_b == Op::N { b[p * n + j] } else { b[j * k + p] } as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        assert_close(&format!("gemm_f32 {m}x{n}x{k} {op_b:?}"), &got, &want, Tol { abs: 1e-4, rel: 1e-4 });
    }
}

#[test]
fn mel_frames_windows_the_signal() {
    // frames[f][j] = hann[j] * padded[(first + f) * hop + j] for f < count. Rows past
    // `count` keep their old values (the last chunk of a file is short).
    let gpu = Gpu::open(0).unwrap();
    let (rows, hop, size, first, count) = (7, 160, 400, 3, 5);
    let n_pad = (first + count - 1) * hop + size;
    let padded = rnd(n_pad, 3, 1.0);
    let hann = rnd(size, 4, 1.0);
    let mut plan = ArenaPlan::new();
    let tp = plan.add(DType::F32, &[n_pad]);
    let th = plan.add(DType::F32, &[size]);
    let tf = plan.add(DType::F32, &[rows, size]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &tp, &padded).unwrap();
    arena.upload(&gpu, &th, &hann).unwrap();
    arena.upload(&gpu, &tf, &vec![7.0f32; rows * size]).unwrap();
    mel_frames(&gpu, &arena, &tp, &th, &tf, hop, first, count).unwrap();
    let want: Vec<f32> = (0..rows * size)
        .map(|i| {
            let (f, j) = (i / size, i % size);
            if f < count { hann[j] * padded[(first + f) * hop + j] } else { 7.0 }
        })
        .collect();
    assert_eq!(arena.download::<f32>(&gpu, &tf).unwrap(), want);
}

#[test]
fn dev_slice_range_is_checked() {
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let t = plan.add(DType::F32, &[10, 4]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    let s = arena.slice::<f32>(&t);
    let r = s.range(8, 12);
    assert_eq!(r.len(), 12);
    assert_eq!(r.cu_deviceptr(), s.cu_deviceptr() + 8 * 4);
    assert!(std::panic::catch_unwind(|| s.range(30, 11)).is_err(), "a range past the end must panic");
}

#[test]
fn power_spectrum_adds_squares() {
    // spec [n, 2 * bins]: the real parts in columns 0..bins, the imaginary parts after.
    let gpu = Gpu::open(0).unwrap();
    let (n, bins) = (5, 201);
    let spec = rnd(n * 2 * bins, 5, 10.0);
    let mut plan = ArenaPlan::new();
    let ts = plan.add(DType::F32, &[n, 2 * bins]);
    let tp = plan.add(DType::F32, &[n, bins]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &ts, &spec).unwrap();
    power_spectrum(&gpu, &arena, &ts, &tp).unwrap();
    let want: Vec<f32> = (0..n * bins)
        .map(|i| {
            let (r, k) = (i / bins, i % bins);
            let (re, im) = (spec[r * 2 * bins + k], spec[r * 2 * bins + bins + k]);
            re * re + im * im
        })
        .collect();
    assert_close("power", &arena.download::<f32>(&gpu, &tp).unwrap(), &want, Tol { abs: 1e-4, rel: 1e-6 });
}

#[test]
fn log10_max_floors_and_finds_the_max() {
    // In place log10(max(v, 1e-10)), and the largest result. 70001 values: several blocks
    // and a short last block. Zeros become -10.
    let gpu = Gpu::open(0).unwrap();
    let n = 70_001;
    let mut x: Vec<f32> = rnd(n, 6, 1.0).iter().map(|v| v.abs() * 100.0).collect();
    x[123] = 0.0;
    x[69_999] = 5000.0;
    let mut plan = ArenaPlan::new();
    let tx = plan.add(DType::F32, &[n]);
    let tpart = plan.add(DType::F32, &[log10_max_partials(n)]);
    let tmax = plan.add(DType::F32, &[1]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &tx, &x).unwrap();
    log10_max(&gpu, &arena, &tx, &tpart, &tmax).unwrap();
    let want: Vec<f32> = x.iter().map(|&v| (v.max(1e-10) as f64).log10() as f32).collect();
    let got = arena.download::<f32>(&gpu, &tx).unwrap();
    assert_close("log10", &got, &want, Tol { abs: 1e-5, rel: 1e-5 });
    assert_eq!(got[123], -10.0);
    let max = arena.download::<f32>(&gpu, &tmax).unwrap()[0];
    assert!((max - 5000f32.log10()).abs() < 1e-5, "max {max}");
}

#[test]
fn mel_window_clamps_normalizes_and_transposes() {
    // mel_t [n_len, n_mel] (frames are rows). The window is [n_mel, w] (bands are rows),
    // with value ((max(v, max - 8)) + 4) / 4. Frames past n_len get 0.0, as whisper.cpp
    // writes them (it clears the encoder input and copies only the frames below n_len).
    let gpu = Gpu::open(0).unwrap();
    let (n_len, n_mel, w, start) = (50, 4, 20, 40);
    let mel_t: Vec<f32> = (0..n_len * n_mel).map(|i| -12.0 + (i % 17) as f32).collect();
    let max = 3.0f32;
    let mut plan = ArenaPlan::new();
    let tm = plan.add(DType::F32, &[n_len, n_mel]);
    let tmax = plan.add(DType::F32, &[1]);
    let tout = plan.add(DType::F16, &[n_mel, w]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &tm, &mel_t).unwrap();
    arena.upload(&gpu, &tmax, &[max]).unwrap();
    mel_window(&gpu, &arena, &tm, &tmax, start, &arena, &tout).unwrap();
    let floor = max - 8.0;
    let want: Vec<f32> = (0..n_mel * w)
        .map(|i| {
            let (b, t) = (i / w, i % w);
            if start + t >= n_len {
                return 0.0;
            }
            let v = mel_t[(start + t) * n_mel + b];
            f16::from_f32((v.max(floor) + 4.0) / 4.0).to_f32()
        })
        .collect();
    let got: Vec<f32> = arena.download::<f16>(&gpu, &tout).unwrap().iter().map(|v| v.to_f32()).collect();
    assert_eq!(got, want);
}
