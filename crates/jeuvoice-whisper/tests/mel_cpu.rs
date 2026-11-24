use jeuvoice_whisper::mel_cpu::{N_MEL, frame_count, log_mel};

/// A small 128 x 201 filter bank for the tests: band j is one bin wide at bin j + 10.
fn simple_filters() -> Vec<f32> {
    let mut f = vec![0.0f32; N_MEL * 201];
    for j in 0..N_MEL {
        f[j * 201 + j + 10] = 1.0;
    }
    f
}

#[test]
fn frame_count_follows_whisper_cpp() {
    // whisper.cpp: n_len = (n + 480,000 + 400 - 400) / 160 (30 s of zero padding).
    assert_eq!(frame_count(0), 3000);
    assert_eq!(frame_count(1), 3000);
    assert_eq!(frame_count(160), 3001);
    assert_eq!(frame_count(16_000 * 10), 4000);
}

#[test]
fn mel_short_and_empty_audio() {
    // No sound: every band is log10(1e-10) = -10. The max is -10, the clamp floor is -18,
    // and (v + 4) / 4 = -1.5 everywhere.
    for n in [0usize, 1] {
        let samples = vec![0.0f32; n];
        let (mel, n_len) = log_mel(&samples, &simple_filters());
        assert_eq!(n_len, 3000);
        assert_eq!(mel.len(), N_MEL * 3000);
        assert!(mel.iter().all(|&v| v == -1.5), "{n} samples: not all -1.5");
    }
}

#[test]
fn a_sine_lands_in_its_band() {
    // A sine at bin 50 (50 * 16000 / 400 = 2000 Hz). With the one-bin filters, band 40
    // (bin 50) is the loudest band in a middle frame.
    let n = 16_000;
    let samples: Vec<f32> = (0..n).map(|i| (2.0 * std::f32::consts::PI * 2000.0 * i as f32 / 16_000.0).sin() * 0.5).collect();
    let (mel, n_len) = log_mel(&samples, &simple_filters());
    let frame = 50;
    let col: Vec<f32> = (0..N_MEL).map(|j| mel[j * n_len + frame]).collect();
    let loudest = col.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    assert_eq!(loudest, 40, "the 2000 Hz sine must land in the band of bin 50");
    assert!(mel.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "needs the reference files from tools/reference (see its README)"]
fn mel_cpu_matches_reference() {
    // Clips of 30 s or less: whisper.cpp and Hugging Face see the same window. The edges
    // differ: HF pads the end with a reflection, whisper.cpp with zeros.
    let model = jeuvoice_whisper::ggml::GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/reference/whisper");
    for clip in ["my_voice", "ses_001_0007"] {
        let r = jeuvoice_gpu_free_reference(&root.join(format!("{clip}.safetensors")));
        let (samples_shape, samples) = &r["samples"];
        assert_eq!(samples_shape.len(), 1);
        let (mel, n_len) = log_mel(samples, &model.mel_filters.2);
        let (_, want) = &r["mel"];
        let got: Vec<f32> = (0..N_MEL).flat_map(|j| mel[j * n_len..j * n_len + 3000].to_vec()).collect();
        let mut worst = 0.0f32;
        let mut sum = 0.0f64;
        for (g, w) in got.iter().zip(want) {
            worst = worst.max((g - w).abs());
            sum += (g - w).abs() as f64;
        }
        eprintln!("{clip}: max abs diff {worst:.4}, mean abs diff {:.5}", sum / got.len() as f64);
        assert!(sum / (got.len() as f64) < 0.01, "{clip}: mean difference too large");
    }
}

/// Reads an f32 safetensors file without the GPU crate (this test runs in the plain build).
fn jeuvoice_gpu_free_reference(path: &std::path::Path) -> std::collections::HashMap<String, (Vec<usize>, Vec<f32>)> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    st.tensors()
        .into_iter()
        .map(|(name, v)| {
            let vals = v.data().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            (name, (v.shape().to_vec(), vals))
        })
        .collect()
}
