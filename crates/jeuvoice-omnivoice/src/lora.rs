//! The LoRA merge of the OmniVoice adapter: `W' = W + scale · B · A` on the host, in f32.
//!
//! The adapter file names its tensors `base_model.model.llm.layers.N.self_attn.q_proj.lora_A.weight`
//! and `...lora_B.weight`, and the base checkpoint names the same weight
//! `llm.layers.N.self_attn.q_proj.weight`. `base_name` maps one to the other.
//!
//! `audio_embeddings` and `audio_heads` come from `modules_to_save`: the adapter holds the
//! whole trained table, so it replaces the base tensor instead of merging into it.

/// The prefix of every key in a PEFT adapter file.
pub const ADAPTER_PREFIX: &str = "base_model.model.";
/// The suffix of the A half of a pair.
pub const LORA_A: &str = ".lora_A.weight";
/// The suffix of the B half of a pair.
pub const LORA_B: &str = ".lora_B.weight";

/// The base tensor name of a LoRA half, or `None` for a key that is not a LoRA half.
///
/// `base_model.model.llm.layers.3.mlp.down_proj.lora_A.weight` gives
/// `llm.layers.3.mlp.down_proj.weight`.
pub fn base_name(adapter_key: &str) -> Option<String> {
    let rest = adapter_key.strip_prefix(ADAPTER_PREFIX)?;
    let module = rest.strip_suffix(LORA_A).or_else(|| rest.strip_suffix(LORA_B))?;
    Some(format!("{module}.weight"))
}

/// The base tensor name of a `modules_to_save` key, or `None`. An
/// `audio_embeddings` or `audio_heads` key replaces the base table with the same name.
pub fn replaced_name(adapter_key: &str) -> Option<String> {
    let rest = adapter_key.strip_prefix(ADAPTER_PREFIX)?;
    let module = rest.strip_suffix(".weight")?;
    (module == "audio_embeddings" || module == "audio_heads").then(|| rest.to_string())
}

/// The name of the other half of a pair: `lora_A` gives `lora_B` and back.
pub fn other_half(adapter_key: &str) -> Option<String> {
    if let Some(module) = adapter_key.strip_suffix(LORA_A) {
        Some(format!("{module}{LORA_B}"))
    } else {
        adapter_key.strip_suffix(LORA_B).map(|module| format!("{module}{LORA_A}"))
    }
}

/// `w += scale · B · A`, where `w` is `[out_dim, in_dim]` row major, `a` is `[r, in_dim]` and
/// `b` is `[out_dim, r]`. Both in the row-major order of the safetensors file.
///
/// # Panics
/// If a slice has the wrong length: that is a broken adapter, not user input.
pub fn merge_into(w: &mut [f32], a: &[f32], b: &[f32], out_dim: usize, in_dim: usize, r: usize, scale: f32) {
    assert_eq!(w.len(), out_dim * in_dim, "merge_into: w is not [{out_dim}, {in_dim}]");
    assert_eq!(a.len(), r * in_dim, "merge_into: a is not [{r}, {in_dim}]");
    assert_eq!(b.len(), out_dim * r, "merge_into: b is not [{out_dim}, {r}]");
    for o in 0..out_dim {
        let wrow = &mut w[o * in_dim..(o + 1) * in_dim];
        for k in 0..r {
            let s = b[o * r + k] * scale;
            if s == 0.0 {
                continue;
            }
            let arow = &a[k * in_dim..(k + 1) * in_dim];
            for i in 0..in_dim {
                wrow[i] += s * arow[i];
            }
        }
    }
}

/// The LoRA scale of an adapter: `alpha / r`.
pub fn scale_of(alpha: f32, r: usize) -> f32 {
    assert!(r > 0, "scale_of: r must be positive");
    alpha / r as f32
}
