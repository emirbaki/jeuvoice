//! The OmniVoice weights and the activations of one pass, in one arena, with the LoRA
//! adapter merged on the host (plan Phase 2, Task 6).
//!
//! The checkpoint stores every weight as f32. The arena takes the matrices as **f16** (the
//! model is 612.6M parameters, so f16 is 1.23 GB instead of 2.45 GB and a 4 GB card still
//! fits), and the norms as **f32** (hard rule 5). The LoRA merge runs in f32 before the
//! conversion, so the rounding happens once.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, Tensor};

use crate::config::OmniVoiceConfig;
use crate::index::TensorIndex;
use crate::lora::{LORA_A, merge_into, other_half, replaced_name, scale_of};
use crate::{Result, TtsError};

/// The most positions one pass can hold. A render of 8.4 s uses 486; the prompt and the target
/// of a long take stay under 1024.
pub const T_MAX: usize = 1024;
/// The two rows of a pass: the conditioned and the unconditioned input of the guidance.
pub const BATCH: usize = 2;

/// The activations of one pass. The residual stream `x` is f32, as in the Whisper path: the
/// f16 residual lost a token on a near tie there, and it costs little at this size.
pub struct Activations {
    pub x: Tensor,
    /// The LayerNorm output, the f16 input of the projections.
    pub h: Tensor,
    /// `[BATCH, T, q_width]`: the queries before RoPE, then the rotated queries.
    pub q: Tensor,
    /// `[BATCH, T, kv_width]`.
    pub k: Tensor,
    pub v: Tensor,
    /// `[BATCH, T, q_width]`: the attention output before the out projection.
    pub att: Tensor,
    /// `[BATCH, T, d_mlp]`: the gate, then `silu(gate)·up` in place.
    pub gate: Tensor,
    /// `[BATCH, T, d_mlp]`.
    pub up: Tensor,
    /// `[BATCH, T, hidden]`: a projection result, the residual addend, and the MLP output.
    pub proj: Tensor,
    /// `[BATCH, T, audio_head_rows]` f32: the logits of every codebook.
    pub logits: Tensor,
    /// `[T]` i32: the sequence positions for RoPE.
    pub positions: Tensor,
    /// `[head_dim / 2]` f32: the RoPE inverse frequencies.
    pub inv_freq: Tensor,
}

impl Activations {
    fn plan(plan: &mut ArenaPlan, cfg: &OmniVoiceConfig) -> Activations {
        let d = cfg.hidden_size;
        let (q, kv, mlp) = (cfg.q_width(), cfg.kv_width(), cfg.intermediate_size);
        let t = T_MAX;
        let b = BATCH;
        Activations {
            x: plan.add(DType::F32, &[b, t, d]),
            h: plan.add(DType::F16, &[b, t, d]),
            q: plan.add(DType::F16, &[b, t, q]),
            k: plan.add(DType::F16, &[b, t, kv]),
            v: plan.add(DType::F16, &[b, t, kv]),
            att: plan.add(DType::F16, &[b, t, q]),
            gate: plan.add(DType::F16, &[b, t, mlp]),
            up: plan.add(DType::F16, &[b, t, mlp]),
            proj: plan.add(DType::F16, &[b, t, q]),
            logits: plan.add(DType::F32, &[b, t, cfg.audio_head_rows()]),
            positions: plan.add(DType::I32, &[t]),
            inv_freq: plan.add(DType::F32, &[cfg.head_dim / 2]),
        }
    }
}

/// Every weight of the checkpoint by its file name, in the arena, plus the activations.
pub struct OmniVoiceWeights {
    pub arena: Arena,
    pub act: Activations,
    pub weight_bytes: usize,
    pub activation_bytes: usize,
    /// The seconds the load took, the LoRA merge and the uploads included.
    pub load_seconds: f64,
    tensors: HashMap<String, Tensor>,
}

/// True when this weight stays in f32: the norms (the LayerNorm and the per-head q/k norm).
fn stays_f32(name: &str) -> bool {
    name.contains("norm")
}

impl OmniVoiceWeights {
    /// Reads the checkpoint, merges the adapter when there is one, and uploads everything into
    /// one arena with the activations of one pass.
    pub fn load(gpu: &Gpu, model_dir: &Path, adapter: Option<&Path>, cfg: &OmniVoiceConfig) -> Result<OmniVoiceWeights> {
        cfg.check()?;
        let started = Instant::now();
        let base = TensorIndex::open(&model_dir.join("model.safetensors"))?;
        let adapter_idx = match adapter {
            Some(dir) => Some(TensorIndex::open(&dir.join("adapter_model.safetensors"))?),
            None => None,
        };

        // The plan: every float tensor of the checkpoint, and the activations after them.
        let mut plan = ArenaPlan::new();
        let mut planned: Vec<(String, Tensor, bool)> = Vec::new();
        for name in base.names() {
            let e = base.get(name).expect("the name came from the index");
            if e.dtype != crate::index::SafeDtype::F32 && e.dtype != crate::index::SafeDtype::F16 {
                continue; // codebook_layer_offsets is i64 and is derived from the config
            }
            let f32_keep = stays_f32(name);
            let dtype = if f32_keep { DType::F32 } else { DType::F16 };
            planned.push((name.to_string(), plan.add(dtype, &e.shape), f32_keep));
        }
        let weight_bytes = plan.total_bytes();
        let act = Activations::plan(&mut plan, cfg);
        let activation_bytes = plan.total_bytes() - weight_bytes;
        let arena = Arena::new(gpu, &plan)?;

        // The scale of the adapter: alpha / r, from its config.
        let scale = match adapter {
            Some(dir) => adapter_scale(dir)?,
            None => 1.0,
        };

        let mut tensors = HashMap::new();
        for (name, t, f32_keep) in planned {
            let mut values = base.read_f32(&name)?;
            if let Some(a) = &adapter_idx {
                let a_key = format!("base_model.model.{name}");
                if replaced_name(&a_key).is_some() {
                    // A modules_to_save table replaces the base one.
                    if let Some(e) = a.get(&a_key) {
                        if e.numel() != values.len() {
                            return Err(TtsError::ModelLoad(format!(
                                "the adapter's {name} has {} values, the base has {}",
                                e.numel(),
                                values.len()
                            )));
                        }
                        values = a.read_f32(&a_key)?;
                    }
                } else if let Some(lora_a) = a_key.strip_suffix(".weight").map(|m| format!("{m}{LORA_A}")) {
                    if a.get(&lora_a).is_some() {
                        merge_pair(a, &lora_a, &name, &mut values, scale)?;
                    }
                }
            }
            match f32_keep {
                true => arena.upload(gpu, &t, &values)?,
                false => {
                    let half: Vec<half::f16> = values.iter().map(|&v| half::f16::from_f32(v)).collect();
                    arena.upload(gpu, &t, &half)?;
                }
            }
            tensors.insert(name, t);
        }

        // The RoPE table and the positions, once.
        let inv = jeuvoice_gpu::ops::rope_inv_freq(cfg.rope_theta, cfg.head_dim);
        arena.upload(gpu, &act.inv_freq, &inv)?;
        let positions: Vec<i32> = (0..T_MAX as i32).collect();
        arena.upload(gpu, &act.positions, &positions)?;

        Ok(OmniVoiceWeights { arena, act, weight_bytes, activation_bytes, load_seconds: started.elapsed().as_secs_f64(), tensors })
    }

    /// The tensor of `name` (the file name, for example `llm.layers.0.mlp.up_proj.weight`).
    ///
    /// # Panics
    /// A missing name is a bug in the forward pass, not user input: `load` plans every tensor
    /// of the checkpoint.
    pub fn get(&self, name: &str) -> &Tensor {
        self.tensors.get(name).unwrap_or_else(|| panic!("the checkpoint has no tensor {name}"))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Merges the pair that starts at `a_key` into `values` (the f32 base weight named `name`).
fn merge_pair(adapter: &TensorIndex, a_key: &str, name: &str, values: &mut [f32], scale: f32) -> Result<()> {
    let b_key = other_half(a_key).ok_or_else(|| TtsError::ModelLoad(format!("{a_key} is not a LoRA half")))?;
    let a = adapter.read_f32(a_key)?;
    let b = adapter.read_f32(&b_key)?;
    let a_shape = adapter.entry(a_key)?.shape.clone();
    let b_shape = adapter.entry(&b_key)?.shape.clone();
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(TtsError::ModelLoad(format!("{a_key} or {b_key} is not a 2-D pair")));
    }
    let (r, in_dim) = (a_shape[0], a_shape[1]);
    if b_shape[1] != r {
        return Err(TtsError::ModelLoad(format!("{b_key} is {:?}, not [out, {r}]", b_shape)));
    }
    let out_dim = b_shape[0];
    if out_dim * in_dim != values.len() {
        return Err(TtsError::ModelLoad(format!(
            "{name}: the base is {} values, the LoRA pair is [{out_dim}, {in_dim}]",
            values.len()
        )));
    }
    merge_into(values, &a, &b, out_dim, in_dim, r, scale);
    Ok(())
}

/// `alpha / r` from `adapter_config.json`.
fn adapter_scale(dir: &Path) -> Result<f32> {
    let path = dir.join("adapter_config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| TtsError::ModelLoad(format!("could not read {}: {e}", path.display())))?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| TtsError::ModelLoad(format!("{} is not JSON: {e}", path.display())))?;
    let r = json.get("r").and_then(|v| v.as_u64()).ok_or_else(|| TtsError::ModelLoad(format!("{} has no r", path.display())))? as usize;
    let alpha = json
        .get("lora_alpha")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| TtsError::ModelLoad(format!("{} has no lora_alpha", path.display())))? as f32;
    Ok(scale_of(alpha, r))
}
