//! Prototype: capture the real OmniVoice forward pass and replay it with new inputs.
//!
//! Run it alone (the weights fill 1.4 GB of the 4 GB card):
//!   scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test graph_forward -- --ignored --nocapture --test-threads 1
#![cfg(feature = "cuda-kernels")]

use std::collections::HashMap;

use jeuvoice_gpu::bench::time_on_stream;
use jeuvoice_gpu::blas::Blas;
use jeuvoice_gpu::{Arena, Gpu, GpuError, Graph, Tensor};
use jeuvoice_omnivoice::config::{OmniVoiceConfig, default_cache_root, model_dir};
use jeuvoice_omnivoice::forward::forward_compute;
use jeuvoice_omnivoice::paths::{adapter_dir, data_models_dir};
use jeuvoice_omnivoice::weights::OmniVoiceWeights;

fn reference(path: &str) -> HashMap<String, Vec<f32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut out = HashMap::new();
    for name in tensors.names() {
        let t = tensors.tensor(name).unwrap();
        let v: Vec<f32> = match t.dtype() {
            safetensors::Dtype::F32 => t.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            safetensors::Dtype::I32 => t.data().chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect(),
            safetensors::Dtype::U8 => t.data().iter().map(|&b| b as f32).collect(),
            other => panic!("{name}: {other:?}"),
        };
        out.insert(name.to_string(), v);
    }
    out
}

fn no_taps(_: &str, _: &Arena, _: &Tensor) {}

#[test]
#[ignore = "timing: needs the GPU, the checkpoint, the adapter and the s1 taps fixture"]
fn a_captured_forward_matches_and_runs_faster() {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let dir = model_dir(&default_cache_root(), std::env::var("JEUVOICE_OMNIVOICE_MODEL").ok().as_deref()).unwrap();
    let cfg = OmniVoiceConfig::from_file(&dir.join("config.json")).unwrap();
    let appdata = std::env::var("APPDATA").ok();
    let models = data_models_dir(None, appdata.as_deref()).unwrap();
    let w = OmniVoiceWeights::load(&gpu, &dir, adapter_dir(&models).as_deref(), &cfg).unwrap();

    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/reference/omnivoice");
    let taps = reference(base.join("s1.taps.safetensors").to_str().unwrap());
    let t = taps["ids0"].len() / cfg.num_audio_codebook;
    let ids: Vec<i32> = taps["ids0"].iter().map(|&v| v as i32).collect();
    let mask: Vec<i32> = taps["audio_mask0"].iter().map(|&v| v as i32).collect();
    let id_tensor = w.act.ids.prefix_as(&[cfg.num_audio_codebook, t]);
    let mask_tensor = w.act.mask.prefix_as(&[1, t]);
    let logits = w.act.logits.prefix_as(&[t, cfg.audio_head_rows()]);
    let pass = || -> Result<(), GpuError> {
        w.arena.upload(&gpu, &id_tensor, &ids)?;
        w.arena.upload(&gpu, &mask_tensor, &mask)?;
        forward_compute(&w, &cfg, &gpu, &blas, 1, t, None, &mut no_taps).map_err(|e| GpuError::Driver(e.to_string()))
    };

    // Warm up the kernels and the cuBLAS heuristics before capture.
    pass().unwrap();
    gpu.stream().synchronize().unwrap();
    pass().unwrap();
    gpu.stream().synchronize().unwrap();
    let direct_logits = w.arena.download::<f32>(&gpu, &logits).unwrap();

    let graph = Graph::capture(&gpu, || {
        forward_compute(&w, &cfg, &gpu, &blas, 1, t, None, &mut no_taps).map_err(|e| GpuError::Driver(e.to_string()))
    })
    .unwrap();
    w.arena.upload(&gpu, &id_tensor, &ids).unwrap();
    w.arena.upload(&gpu, &mask_tensor, &mask).unwrap();
    graph.launch(&gpu).unwrap();
    gpu.stream().synchronize().unwrap();
    let replay_logits = w.arena.download::<f32>(&gpu, &logits).unwrap();
    let worst = direct_logits.iter().zip(&replay_logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("capture correctness: max abs {worst:.6}");
    assert!(worst < 1e-4, "the replay changed the logits by {worst}");

    let direct = time_on_stream(&gpu, 1, 6, || pass()).unwrap();
    let replay = time_on_stream(&gpu, 1, 6, || {
        w.arena.upload(&gpu, &id_tensor, &ids)?;
        w.arena.upload(&gpu, &mask_tensor, &mask)?;
        graph.launch(&gpu)
    })
    .unwrap();
    eprintln!("forward at t={t}: direct {:.1} us, graph replay {:.1} us", direct.us, replay.us);
}
