// The Whisper decoder on Rust kernels against PyTorch (plan 1b, Task 10). The decoder gets
// the PyTorch encoder output, so the test checks the decoder alone.
//   scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels --test cuda_decoder -- --ignored --nocapture --test-threads 1
#![cfg(feature = "cuda-kernels")]

use std::path::PathBuf;
use std::time::Instant;

use half::f16;
use jeuvoice_gpu::blas::Blas;
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};
use jeuvoice_whisper::cuda::decoder::Decoder;
use jeuvoice_whisper::ggml::GgmlModel;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;

struct Reference {
    steps: Vec<u32>,
    dtw_tokens: Vec<u32>,
    enc_out: Vec<f32>,
    /// `[steps.len() - 2][51866]`
    logits32: Vec<f32>,
    logits16: Vec<f32>,
    /// `[6][dtw_tokens.len()][1500]`
    align32: Vec<f32>,
    align16: Vec<f32>,
}

fn reference(clip: &str) -> Reference {
    let p: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "..", "target", "reference", "whisper", &format!("{clip}.decoder.safetensors")].iter().collect();
    let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}. Make it with tools/reference/README.md", p.display()));
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let f = |k: &str| {
        let v = st.tensor(k).unwrap();
        assert_eq!(v.dtype(), Dtype::F32, "{k}");
        v.data().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f32>>()
    };
    let i = |k: &str| {
        let v = st.tensor(k).unwrap();
        assert_eq!(v.dtype(), Dtype::I64, "{k}");
        v.data().chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32).collect::<Vec<u32>>()
    };
    Reference {
        steps: i("steps"),
        dtw_tokens: i("dtw_tokens"),
        enc_out: f("enc_out"),
        logits32: f("logits32"),
        logits16: f("logits16"),
        align32: f("align32"),
        align16: f("align16"),
    }
}

const N_VOCAB: usize = 51866;

/// The arg max of each row, the share of rows where `a` and `b` agree, the largest and the
/// mean absolute difference.
fn stats(a: &[f32], b: &[f32], row: usize) -> (f64, f32, f64) {
    let argmax = |r: &[f32]| r.iter().enumerate().fold((0, f32::NEG_INFINITY), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0;
    let rows = a.len() / row;
    let agree = (0..rows).filter(|&r| argmax(&a[r * row..(r + 1) * row]) == argmax(&b[r * row..(r + 1) * row])).count();
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let mean = a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).sum::<f64>() / a.len() as f64;
    (agree as f64 / rows as f64, max, mean)
}

struct Setup {
    gpu: Gpu,
    blas: Blas,
    dec: Decoder,
    enc: Arena,
    enc_t: jeuvoice_gpu::Tensor,
}

/// Loads the decoder and gives it the PyTorch encoder output (rounded to f16).
fn setup(r: &Reference) -> Setup {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let model = GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap();
    let mut dec = Decoder::load(&gpu, &model).unwrap();
    let mut plan = ArenaPlan::new();
    let t = plan.add(DType::F16, &[1500, 1280]);
    let enc = Arena::new(&gpu, &plan).unwrap();
    enc.upload(&gpu, &t, &r.enc_out.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).unwrap();
    dec.set_cross_kv(&gpu, &blas, &enc, &t).unwrap();
    Setup { gpu, blas, dec, enc, enc_t: t }
}

/// The logits after each token from the third on: one `decode_first` of the prompt, then
/// one `decode_next` for each later token.
fn step_logits(s: &mut Setup, steps: &[u32]) -> Vec<f32> {
    let mut out = s.dec.decode_first(&s.gpu, &s.blas, &steps[..3]).unwrap();
    for &t in &steps[3..] {
        out.extend(s.dec.decode_next(&s.gpu, &s.blas, t).unwrap());
    }
    out
}

#[test]
#[ignore = "needs the model file"]
fn weights_plan_and_a_missing_tensor() {
    let gpu = Gpu::open(0).unwrap();
    let mut model = GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap();
    let dec = Decoder::load(&gpu, &model).unwrap();
    let (w, a) = (dec.weight_bytes() >> 20, dec.activation_bytes() >> 20);
    eprintln!("decoder: weights {w} MB, activations {a} MB");
    // 133 MB of token table and 4 blocks of about 46 MB.
    assert!((300..340).contains(&w), "weights {w} MB");
    drop(dec);
    model.tensors.retain(|t| t.name != "decoder.blocks.2.cross_attn.value.bias");
    let e = Decoder::load(&gpu, &model).err().expect("a missing tensor must fail");
    assert!(format!("{e:?}").contains("decoder.blocks.2.cross_attn.value.bias"), "{e:?}");
}

#[test]
#[ignore = "needs the model file and the decoder reference (tools/reference)"]
fn step_logits_match_pytorch() {
    for clip in ["ses_001_0007", "my_voice"] {
        let r = reference(clip);
        let mut s = setup(&r);
        let got = step_logits(&mut s, &r.steps);
        assert_eq!(got.len(), r.logits32.len());
        let ours = stats(&got, &r.logits32, N_VOCAB);
        let torch16 = stats(&r.logits16, &r.logits32, N_VOCAB);
        eprintln!("{clip}: ours vs fp32: agree {:.4}, max {:.4}, mean {:.5}", ours.0, ours.1, ours.2);
        eprintln!("{clip}: torch fp16 vs fp32: agree {:.4}, max {:.4}, mean {:.5}", torch16.0, torch16.1, torch16.2);
        // Measured on 2026-11-25 against fp32 PyTorch, next to fp16 PyTorch on the same input:
        //
        // | clip | ours: agree, max, mean | PyTorch fp16: agree, max, mean |
        // |---|---|---|
        // | ses_001_0007 | 1.0, 0.0318, 0.00334 | 1.0, 0.0296, 0.00337 |
        // | my_voice | 1.0, 0.0560, 0.00448 | 1.0, 0.1234, 0.00432 |
        //
        // The limits allow 2 times the fp16 PyTorch error.
        assert!(ours.0 >= 0.99, "{clip}: the arg max agrees on only {:.4} of the rows", ours.0);
        assert!(ours.1 <= 2.0 * torch16.1.max(0.05), "{clip}: max difference {} against fp16 PyTorch {}", ours.1, torch16.1);
        assert!(ours.2 <= 2.0 * torch16.2.max(1e-3), "{clip}: mean difference {} against fp16 PyTorch {}", ours.2, torch16.2);
    }
}

#[test]
#[ignore = "needs the model file and the decoder reference (tools/reference)"]
fn one_batch_gives_the_logits_of_the_steps() {
    // The KV cache test: decode_first of the first k tokens gives the logits of step k - 3.
    let r = reference("ses_001_0007");
    let mut s = setup(&r);
    let steps = step_logits(&mut s, &r.steps);
    for k in [4, 20, r.steps.len()] {
        let batch = s.dec.decode_first(&s.gpu, &s.blas, &r.steps[..k]).unwrap();
        let row = &steps[(k - 3) * N_VOCAB..(k - 2) * N_VOCAB];
        let (agree, max, _) = stats(&batch, row, N_VOCAB);
        eprintln!("batch of {k}: agree {agree}, max {max:.4}");
        assert_eq!(agree, 1.0, "batch of {k}: another arg max");
        // Measured 2026-11-25: 0.0074 to 0.0081 (other GEMM shapes, so other f32 sums).
        assert!(max < 0.05, "batch of {k}: max difference {max}");
    }
}

#[test]
#[ignore = "needs the model file and the decoder reference (tools/reference)"]
fn alignment_heads_match_pytorch() {
    for clip in ["ses_001_0007", "my_voice"] {
        let r = reference(clip);
        let mut s = setup(&r);
        let got = s.dec.alignment(&s.gpu, &s.blas, &r.dtw_tokens, 1500).unwrap();
        assert_eq!(got.len(), r.align32.len());
        let ours = stats(&got, &r.align32, 1500);
        let torch16 = stats(&r.align16, &r.align32, 1500);
        eprintln!("{clip}: align ours vs fp32: agree {:.4}, max {:.5}, mean {:.6}", ours.0, ours.1, ours.2);
        eprintln!("{clip}: align torch fp16 vs fp32: agree {:.4}, max {:.5}, mean {:.6}", torch16.0, torch16.1, torch16.2);
        // Measured 2026-11-25: max 0.00156 and 0.00316 (PyTorch fp16: 0.00307 and 0.00301),
        // mean about 1e-6 for both. The limits allow 2 times the fp16 PyTorch error.
        assert!(ours.1 <= 2.0 * torch16.1.max(1e-3), "{clip}: max difference {} against fp16 PyTorch {}", ours.1, torch16.1);
        assert!(ours.2 <= 2.0 * torch16.2.max(1e-5), "{clip}: mean difference {} against fp16 PyTorch {}", ours.2, torch16.2);
        // Fewer audio positions give the first columns of the same rows.
        let part = s.dec.alignment(&s.gpu, &s.blas, &r.dtw_tokens, 400).unwrap();
        for (h, row) in (0..6).flat_map(|h| (0..r.dtw_tokens.len()).map(move |t| (h, t))) {
            let a = &part[(h * r.dtw_tokens.len() + row) * 400..][..400];
            let b = &got[(h * r.dtw_tokens.len() + row) * 1500..][..400];
            assert_eq!(a, b, "{clip}: head slot {h}, row {row}");
        }
    }
}

#[test]
#[ignore = "timing: run in release with --nocapture"]
fn decoder_timing() {
    let r = reference("my_voice");
    let mut s = setup(&r);
    let cross: Vec<f64> = (0..10)
        .map(|_| {
            let t0 = Instant::now();
            s.dec.set_cross_kv(&s.gpu, &s.blas, &s.enc, &s.enc_t).unwrap();
            s.gpu.stream().synchronize().unwrap();
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    // The steps: the prompt, then 200 single tokens (the logits download syncs each one).
    s.dec.decode_first(&s.gpu, &s.blas, &r.steps[..3]).unwrap();
    let mut step: Vec<f64> = (0..200)
        .map(|i| {
            let t0 = Instant::now();
            s.dec.decode_next(&s.gpu, &s.blas, r.steps[3 + i % (r.steps.len() - 3)]).unwrap();
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    step.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut cross = cross;
    cross.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("cross K/V per window: median {:.2} ms", cross[cross.len() / 2]);
    eprintln!("decoder step: median {:.3} ms, p90 {:.3} ms", step[100], step[180]);
}

#[test]
#[ignore = "needs the model file, the test clips and the whisper-rs fixtures (Task 0)"]
fn whole_model_smoke_on_one_clip() {
    // GpuWhisper and the window loop together on the 8.4 s clip. Task 11 checks all clips
    // against the full criteria. Here: most tokens equal whisper-rs, and the memory.
    use jeuvoice_whisper::cuda::model::GpuWhisper;
    use jeuvoice_whisper::decode::vocab::Vocab;
    use jeuvoice_whisper::decode::window::{DecodeParams, transcribe_tokens};
    let clip = std::path::Path::new(r"D:\MLProjelerim\tts_lab\data\ref\sesim\ses_001_0007.wav");
    let samples = jeuvoice_core::audio::to_whisper(&jeuvoice_core::audio::decode(clip).unwrap());
    let model = GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap();
    let vocab = Vocab::from_ggml(&model).unwrap();
    let probe = Gpu::open(0).unwrap();
    let free0 = probe.free_vram().unwrap();
    let mut gw = GpuWhisper::load(&model).unwrap();
    gw.set_audio(&samples).unwrap();
    let t0 = Instant::now();
    let t = transcribe_tokens(&mut gw, &vocab, &samples, &DecodeParams::default(), &mut |_| {}).unwrap();
    let secs = t0.elapsed().as_secs_f64();
    let used = (free0 - gw.gpu().free_vram().unwrap()) >> 20;
    eprintln!("model {} MB, VRAM used by the load and the run {used} MB, run {secs:.2} s", gw.model_bytes() >> 20);

    let ours: Vec<u32> = t.segments.iter().flat_map(|s| s.tokens.iter().map(|t| t.id)).collect();
    let p: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "..", "target", "reference", "whisper", "ses_001_0007.rs.json"].iter().collect();
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let theirs: Vec<u32> = v["segments"].as_array().unwrap().iter().flat_map(|s| s["tokens"].as_array().unwrap().iter().map(|t| t["id"].as_u64().unwrap() as u32)).collect::<Vec<_>>();
    let same = ours.iter().zip(&theirs).filter(|(a, b)| a == b).count();
    eprintln!("tokens: ours {}, whisper-rs {}, same position and id {same}", ours.len(), theirs.len());
    let text: String = t.segments.iter().map(|s| String::from_utf8_lossy(&s.text).into_owned()).collect();
    eprintln!("text: {text}");
    assert!(same as f64 >= 0.9 * theirs.len() as f64, "only {same} of {} tokens equal", theirs.len());
}
