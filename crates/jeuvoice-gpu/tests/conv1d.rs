use half::f16;
use jeuvoice_gpu::blas::Blas;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{ConvKind, conv1d};
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

fn r16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
}

fn h(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

/// One output of the kernel-3, padding-1 convolution, in f64.
fn cpu_at(x: &[f32], w: &[f32], c_in: usize, t_in: usize, stride: usize, co: usize, t: usize) -> f32 {
    let mut s = 0.0f64;
    for ci in 0..c_in {
        for k in 0..3 {
            let src = (t * stride + k) as isize - 1;
            if src >= 0 && (src as usize) < t_in {
                s += w[(co * c_in + ci) * 3 + k] as f64 * x[ci * t_in + src as usize] as f64;
            }
        }
    }
    s as f32
}

fn t_out(t_in: usize, stride: usize) -> usize {
    (t_in + 2 - 3) / stride + 1
}

/// Runs `conv1d` with `kind` and returns y `[c_out, t_out]` as f32.
fn run(gpu: &Gpu, blas: &Blas, x: &[f32], w: &[f32], c_in: usize, c_out: usize, t_in: usize, stride: usize, kind: ConvKind) -> Vec<f32> {
    let to = t_out(t_in, stride);
    let mut plan = ArenaPlan::new();
    let tx = plan.add(DType::F16, &[c_in, t_in]);
    let tw = plan.add(DType::F16, &[c_out, c_in, 3]);
    let ty = plan.add(DType::F16, &[c_out, to]);
    let ts = plan.add(DType::F16, &[c_in * 3, to]);
    let arena = Arena::new(gpu, &plan).unwrap();
    arena.upload(gpu, &tx, &h(x)).unwrap();
    arena.upload(gpu, &tw, &h(w)).unwrap();
    let scratch = if kind == ConvKind::Im2col { Some(&ts) } else { None };
    conv1d(gpu, &arena, blas, &tx, &tw, &ty, stride, scratch, kind).unwrap();
    arena.download::<f16>(gpu, &ty).unwrap().iter().map(|v| v.to_f32()).collect()
}

#[test]
fn conv1d_tiny_by_hand() {
    // x = [[1, 2, 3, 4, 5], [0, 0, 1, 0, 0]], one output channel.
    // Channel 0 weights [1, 0, -1] give x[t-1] - x[t+1] (zero padding at both ends).
    // Channel 1 weights [0, 2, 0] give 2 x[t].
    // Stride 1: [0-2, 1-3, 2-4, 3-5, 4-0] + [0, 0, 2, 0, 0] = [-2, -2, 0, -2, 4].
    // Stride 2 keeps t = 0, 2, 4: [-2, 0, 4].
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let w = [1.0, 0.0, -1.0, 0.0, 2.0, 0.0];
    for kind in [ConvKind::Direct, ConvKind::Im2col] {
        assert_eq!(run(&gpu, &blas, &x, &w, 2, 1, 5, 1, kind), vec![-2.0, -2.0, 0.0, -2.0, 4.0], "{kind:?} stride 1");
        assert_eq!(run(&gpu, &blas, &x, &w, 2, 1, 5, 2, kind), vec![-2.0, 0.0, 4.0], "{kind:?} stride 2");
    }
}

/// Compares 2000 random output positions with the CPU (a full CPU reference of conv2 is
/// 7.4 billion multiply-adds, too slow for a debug test).
fn check_shape(c_in: usize, c_out: usize, t_in: usize, stride: usize) {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    // Scale so each output is about 1: the sum has 3 c_in terms.
    let x = r16(&rnd(c_in * t_in, 41, 1.0));
    let w = r16(&rnd(c_out * c_in * 3, 42, 1.0 / (c_in as f32 * 3.0).sqrt()));
    let to = t_out(t_in, stride);
    let mut s = 7u32;
    let picks: Vec<(usize, usize)> = (0..2000)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let co = (s >> 8) as usize % c_out;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            // Include both edges: t = 0 and t = to - 1 use the zero padding.
            let t = match (s >> 8) % 10 {
                0 => 0,
                1 => to - 1,
                v => v as usize * 7919 % to,
            };
            (co, t)
        })
        .collect();
    let want: Vec<f32> = picks.iter().map(|&(co, t)| cpu_at(&x, &w, c_in, t_in, stride, co, t)).collect();
    for kind in [ConvKind::Direct, ConvKind::Im2col] {
        let y = run(&gpu, &blas, &x, &w, c_in, c_out, t_in, stride, kind);
        let got: Vec<f32> = picks.iter().map(|&(co, t)| y[co * to + t]).collect();
        assert_close(&format!("conv1d {kind:?} {c_in}->{c_out} t={t_in} s={stride}"), &got, &want, Tol { abs: 1e-2, rel: 1e-2 });
        assert!(y.iter().all(|v| v.is_finite()), "{kind:?}: inf or NaN in the output");
    }
}

#[test]
fn conv1d_whisper_conv1_shape() {
    check_shape(128, 1280, 3000, 1);
}

#[test]
fn conv1d_whisper_conv2_shape() {
    check_shape(1280, 1280, 3000, 2);
}
