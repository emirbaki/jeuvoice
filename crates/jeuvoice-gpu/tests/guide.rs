//! The GPU guided peak against a f64 reference of the engine's formula (plan Phase 2, E2).

use jeuvoice_gpu::ops::guide_peak;
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

/// The engine, in f64: three log_softmaxes, the mask set to -inf after the last one.
fn engine(c: &[f64], u: &[f64], g: f64, mask: usize) -> (f64, usize) {
    fn log_softmax(x: &[f64]) -> Vec<f64> {
        let top = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = x.iter().map(|v| (v - top).exp()).sum();
        let lse = top + sum.ln();
        x.iter().map(|v| v - lse).collect()
    }
    let lc = log_softmax(c);
    let lu = log_softmax(u);
    let z: Vec<f64> = lc.iter().zip(&lu).map(|(c, u)| c + g * (c - u)).collect();
    let mut lp = log_softmax(&z);
    lp[mask] = f64::NEG_INFINITY;
    let (at, &value) = lp.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    (value, at)
}

#[test]
fn the_guided_peak_matches_the_engine() {
    let gpu = Gpu::open(0).unwrap();
    let (poses, codebooks, vocab, width) = (3usize, 2usize, 8usize, 16usize);
    let mask = 7usize;
    let pairs = poses * codebooks;
    // Two planes with a different shape each, and the mask id is the maximum in one pair.
    let mut c = vec![0.0f32; poses * width];
    let mut u = vec![0.0f32; poses * width];
    for p in 0..poses {
        for v in 0..width {
            c[p * width + v] = ((p * 7 + v * 3) % 11) as f32 * 0.4 - 1.5;
            u[p * width + v] = ((p * 5 + v * 2) % 7) as f32 * 0.3 - 0.9;
        }
    }
    // Position 1, codebook 0: the mask logit wins before masking, which must lower the
    // confidence but not change the token.
    c[1 * width + 0 + mask] = 9.0;

    let mut plan = ArenaPlan::new();
    let tc = plan.add(DType::F32, &[poses, width]);
    let tu = plan.add(DType::F32, &[poses, width]);
    let tconf = plan.add(DType::F32, &[pairs]);
    let tpick = plan.add(DType::I32, &[pairs]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &tc, &c).unwrap();
    arena.upload(&gpu, &tu, &u).unwrap();
    guide_peak(&gpu, &arena, &tc, &tu, &tconf, &tpick, pairs, width, vocab, codebooks, mask, 2.5).unwrap();
    let conf = arena.download::<f32>(&gpu, &tconf).unwrap();
    let pick = arena.download::<i32>(&gpu, &tpick).unwrap();

    for pair in 0..pairs {
        let pos = pair / codebooks;
        let book = pair % codebooks;
        let base = pos * width + book * vocab;
        let cf: Vec<f64> = c[base..base + vocab].iter().map(|&v| v as f64).collect();
        let uf: Vec<f64> = u[base..base + vocab].iter().map(|&v| v as f64).collect();
        let (want, want_at) = engine(&cf, &uf, 2.5, mask);
        assert_eq!(pick[pair] as usize, want_at, "pair {pair}");
        let tolerance = 2e-3 * want.abs().max(1.0);
        assert!((conf[pair] as f64 - want).abs() <= tolerance, "pair {pair}: {} against {want}", conf[pair]);
    }
}
