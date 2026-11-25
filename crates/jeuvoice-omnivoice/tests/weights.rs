//! Loading the OmniVoice weights and merging the LoRA adapter (plan Phase 2, Task 6).
//! Needs the GPU and the files on this machine: the checkpoint in the Hugging Face cache and
//! the adapter in the app data folder.
//!
//! Run the two tests **one at a time, by name**: together they hold more than the 4 GB of the
//! card (`the_adapter_changes_the_base_weights` keeps the base and the merged arena at once,
//! 2.7 GB), so a single process cannot hold both.
//!
//!   scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test weights -- --ignored --nocapture --test-threads 1 the_weights_load_and_fit_the_plan
#![cfg(feature = "cuda-kernels")]

use jeuvoice_gpu::{DType, Gpu};
use jeuvoice_omnivoice::config::{OmniVoiceConfig, default_cache_root, model_dir};
use jeuvoice_omnivoice::paths::{adapter_dir, data_models_dir};
use jeuvoice_omnivoice::weights::{BATCH, OmniVoiceWeights, T_MAX};

fn model() -> std::path::PathBuf {
    model_dir(&default_cache_root(), std::env::var("JEUVOICE_OMNIVOICE_MODEL").ok().as_deref()).unwrap()
}

fn adapter() -> Option<std::path::PathBuf> {
    let appdata = std::env::var("APPDATA").ok();
    let models = data_models_dir(std::env::var("JEUVOICE_DATA_DIR").ok().as_deref(), appdata.as_deref()).ok()?;
    adapter_dir(&models)
}

#[test]
#[ignore = "needs the GPU, the OmniVoice checkpoint and the adapter"]
fn the_weights_load_and_fit_the_plan() {
    let gpu = Gpu::open(0).unwrap();
    let dir = model();
    let cfg = OmniVoiceConfig::from_file(&dir.join("config.json")).unwrap();
    let adapter = adapter();
    eprintln!("adapter: {}", adapter.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "none".into()));

    let w = OmniVoiceWeights::load(&gpu, &dir, adapter.as_deref(), &cfg).unwrap();
    eprintln!(
        "loaded {} tensors in {:.1} s: weights {:.0} MB, activations {:.0} MB",
        w.len(),
        w.load_seconds,
        w.weight_bytes as f64 / 1e6,
        w.activation_bytes as f64 / 1e6
    );

    // The tensors the forward pass needs, with the shapes of the checkpoint.
    let d = cfg.hidden_size;
    assert_eq!(w.get("llm.embed_tokens.weight").shape(), &[cfg.vocab_size, d]);
    assert_eq!(w.get("audio_embeddings.weight").shape(), &[cfg.audio_head_rows(), d]);
    assert_eq!(w.get("audio_heads.weight").shape(), &[cfg.audio_head_rows(), d]);
    assert_eq!(w.get("llm.norm.weight").shape(), &[d]);
    assert_eq!(w.get("llm.layers.27.mlp.down_proj.weight").shape(), &[d, cfg.intermediate_size]);

    // The dtype policy: norms f32, matrices f16.
    assert_eq!(w.get("llm.norm.weight").dtype(), DType::F32, "the final norm stays f32");
    assert_eq!(w.get("llm.layers.0.input_layernorm.weight").dtype(), DType::F32);
    assert_eq!(w.get("llm.layers.0.self_attn.q_norm.weight").dtype(), DType::F32);
    assert_eq!(w.get("llm.layers.0.self_attn.qkv_proj.weight").dtype(), DType::F16, "matrices are f16");
    assert_eq!(w.get("llm.embed_tokens.weight").dtype(), DType::F16);

    // The fused projections hold q | k | v and gate | up side by side (P3), and the parts are
    // not in the arena a second time.
    assert_eq!(w.get("llm.layers.0.self_attn.qkv_proj.weight").shape(), &[cfg.q_width() + 2 * cfg.kv_width(), d]);
    assert_eq!(w.get("llm.layers.0.mlp.gate_up_proj.weight").shape(), &[2 * cfg.intermediate_size, d]);
    assert!(!w.has("llm.layers.0.self_attn.q_proj.weight"), "the separate q projection must not stay");
    assert!(!w.has("llm.layers.0.mlp.up_proj.weight"), "the separate up projection must not stay");

    // The activations cover the longest sequence. The residual stream and the logits are 2-D,
    // the layout the GEMMs use: one row per position.
    assert_eq!(w.act.x.shape(), &[BATCH * T_MAX, d]);
    assert_eq!(w.act.logits.shape(), &[BATCH * T_MAX, cfg.audio_head_rows()]);
    assert_eq!(w.act.inv_freq.numel(), cfg.head_dim / 2);

    // The arena is inside a 4 GB card, with the context and the audio tokenizer on the CPU.
    let total = (w.weight_bytes + w.activation_bytes) as f64 / 1e6;
    eprintln!("arena total: {total:.0} MB");
    assert!(total < 1700.0, "the arena takes {total:.0} MB, too much for a 4 GB card");
}

/// The adapter must change the weights: with the LoRA merged, a projection differs from the
/// base one.
#[test]
#[ignore = "needs the GPU, the OmniVoice checkpoint and the adapter"]
fn the_adapter_changes_the_base_weights() {
    let gpu = Gpu::open(0).unwrap();
    let dir = model();
    let cfg = OmniVoiceConfig::from_file(&dir.join("config.json")).unwrap();
    let Some(adapter) = adapter() else {
        eprintln!("no adapter on this machine: skipped");
        return;
    };
    let base = OmniVoiceWeights::load(&gpu, &dir, None, &cfg).unwrap();
    let merged = OmniVoiceWeights::load(&gpu, &dir, Some(&adapter), &cfg).unwrap();
    for name in ["llm.layers.0.self_attn.qkv_proj.weight", "audio_embeddings.weight"] {
        let a = base.arena.download::<half::f16>(&gpu, base.get(name)).unwrap();
        let b = merged.arena.download::<half::f16>(&gpu, merged.get(name)).unwrap();
        let different = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        eprintln!("{name}: {different} of {} values differ", a.len());
        assert!(different > a.len() / 2, "{name} did not change, the adapter did not apply");
    }
}

/// The fused matrices are the parts of the checkpoint, side by side and in order: q | k | v
/// and gate | up. The parts are not uploaded separately, so the load must not lose or reorder
/// them (P3).
#[test]
#[ignore = "needs the GPU and the OmniVoice checkpoint"]
fn the_fused_weights_are_the_parts_in_order() {
    let gpu = Gpu::open(0).unwrap();
    let dir = model();
    let cfg = OmniVoiceConfig::from_file(&dir.join("config.json")).unwrap();
    let index = jeuvoice_omnivoice::index::TensorIndex::open(&dir.join("model.safetensors")).unwrap();
    // Base only: with an adapter the parts would no longer equal the file.
    let w = OmniVoiceWeights::load(&gpu, &dir, None, &cfg).unwrap();

    let groups: [(&str, &[&str]); 2] = [
        (
            "llm.layers.0.self_attn.qkv_proj.weight",
            &["llm.layers.0.self_attn.q_proj.weight", "llm.layers.0.self_attn.k_proj.weight", "llm.layers.0.self_attn.v_proj.weight"],
        ),
        ("llm.layers.3.mlp.gate_up_proj.weight", &["llm.layers.3.mlp.gate_proj.weight", "llm.layers.3.mlp.up_proj.weight"]),
    ];
    for (fused_name, parts) in groups {
        let t = w.get(fused_name);
        let cols = t.shape()[1];
        assert_eq!(cols, cfg.hidden_size, "{fused_name}: the row must be the hidden size");
        let want_rows: usize = parts.iter().map(|p| index.entry(p).unwrap().shape[0]).sum();
        assert_eq!(t.shape()[0], want_rows, "{fused_name}: the row count must be the sum of the parts");
        let got: Vec<half::f16> = w.arena.download(&gpu, t).unwrap();
        let mut row = 0;
        for part in parts {
            assert!(!w.has(part), "{part} is still uploaded: the fused matrix must replace it");
            let values = index.read_f32(part).unwrap();
            let rows = values.len() / cols;
            for i in 0..values.len() {
                let at = (row + i / cols) * cols + i % cols;
                assert_eq!(got[at], half::f16::from_f32(values[i]), "{fused_name}: {part} value {i} is wrong");
            }
            row += rows;
        }
        eprintln!("{fused_name}: {} rows in the order {:?}", t.shape()[0], parts);
    }
}
