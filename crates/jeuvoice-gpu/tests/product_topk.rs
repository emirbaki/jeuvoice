//! The GPU top-103 filter for product sampling. The kernel writes confidence, token ids and
//! values; the host keeps the Gumbel stream and picks the final token.

use jeuvoice_gpu::ops::product_topk;
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

#[test]
fn the_product_topk_matches_a_cpu_reference() {
    let gpu = Gpu::open(0).unwrap();
    let (codebooks, target, vocab, k) = (2usize, 3usize, 16usize, 2usize);
    let width = codebooks * vocab;
    let pairs = codebooks * target;
    let mask_id = vocab - 1;
    let guidance = 2.5f32;

    let c: Vec<f32> = (0..target * width).map(|i| (i % 17) as f32 * 0.25 - 2.0).collect();
    let u: Vec<f32> = (0..target * width).map(|i| (i % 13) as f32 * 0.125 - 1.0).collect();

    let mut plan = ArenaPlan::new();
    let tc = plan.add(DType::F32, &[target, width]);
    let tu = plan.add(DType::F32, &[target, width]);
    let logits = Arena::new(&gpu, &plan).unwrap();
    logits.upload(&gpu, &tc, &c).unwrap();
    logits.upload(&gpu, &tu, &u).unwrap();

    let mut out_plan = ArenaPlan::new();
    let conf = out_plan.add(DType::F32, &[pairs]);
    let ids = out_plan.add(DType::I32, &[pairs, k]);
    let vals = out_plan.add(DType::F32, &[pairs, k]);
    let out = Arena::new(&gpu, &out_plan).unwrap();
    product_topk(&gpu, &logits, &tc, &tu, &out, &conf, &ids, &vals, pairs, width, vocab, codebooks, mask_id, guidance, k).unwrap();

    let got_conf = out.download::<f32>(&gpu, &conf).unwrap();
    let got_ids = out.download::<i32>(&gpu, &ids).unwrap();
    let got_vals = out.download::<f32>(&gpu, &vals).unwrap();

    for pair in 0..pairs {
        let position = pair / codebooks;
        let codebook = pair % codebooks;
        let cb = (position * width + codebook * vocab)..(position * width + (codebook + 1) * vocab);
        let y: Vec<f32> = (0..vocab).map(|v| (1.0 + guidance) * c[cb.start + v] - guidance * u[cb.start + v]).collect();
        let top_all = y.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = top_all + y.iter().map(|v| (v - top_all).exp()).sum::<f32>().ln();
        let best = y.iter().enumerate().filter(|(v, _)| *v != mask_id).map(|(_, &v)| v).fold(f32::NEG_INFINITY, f32::max);
        assert!((got_conf[pair] - (best - lse)).abs() < 1e-3, "pair {pair}: confidence");

        let mut ranked: Vec<(f32, usize)> = y.iter().enumerate().filter(|(v, _)| *v != mask_id).map(|(v, &value)| (value, v)).collect();
        ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for j in 0..k {
            assert_eq!(got_ids[pair * k + j], ranked[j].1 as i32, "pair {pair}, slot {j}: id");
            assert!((got_vals[pair * k + j] - ranked[j].0).abs() < 1e-3, "pair {pair}, slot {j}: value");
        }
    }
}
