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
use crate::lora::{LORA_A, Shapes, check_adapter, merge_into, other_half, replaced_name, scale_of};
use crate::{Result, TtsError};

/// The most positions one pass can hold. A render of 8.4 s uses 486; the prompt and the target
/// of a long take stay under 1024.
pub const T_MAX: usize = 1024;
/// The two rows of a pass: the conditioned and the unconditioned input of the guidance.
pub const BATCH: usize = 2;

/// The activations of one pass, flattened over the batch: a row of the sequence is one row of
/// the tensors, so the GEMMs and the row kernels take slices with `Tensor::rows`.
pub struct Activations {
    /// `[batches·t, hidden]` f32: the residual stream.
    pub x: Tensor,
    /// `[batches·t, hidden]` f16: the LayerNorm output, the GEMM input.
    pub h: Tensor,
    /// `[batches·t, q_width + 2·kv_width]` f16: the fused attention projection, q | k | v
    /// side by side in one matrix (P3).
    pub qkv: Tensor,
    /// `[batches, heads_q, t, head_dim]` f16: the queries with the head layout of attention.
    pub q: Tensor,
    /// `[batches, heads_kv, t, head_dim]` f16.
    pub k: Tensor,
    pub v: Tensor,
    /// `[codebooks·t]` f32 and i32: the guided confidence and the picked token of every pair.
    pub conf: Tensor,
    pub pick: Tensor,
    /// `[batches·codebooks, t]` i32: the token ids of the pass, uploaded by the forward pass.
    pub ids: Tensor,
    /// `[batches, t]` i32: 1 where the position is an audio frame, else 0.
    pub mask: Tensor,
    /// `[batches]` i32: the real length of each row of a padded pass.
    pub valid: Tensor,
    /// `[batches, heads_q, t, head_dim]` f16: the attention output.
    pub att: Tensor,
    /// `[batches·t, q_width]` f16: the attention output in the row layout.
    pub att_row: Tensor,
    /// `[batches·t, d_mlp]` f16: `silu(gate)·up`, the input of the down projection.
    pub gate: Tensor,
    /// `[batches·t, 2·d_mlp]` f16: the fused MLP projection, gate | up side by side (P3).
    pub gate_up: Tensor,
    /// `[batches·t, audio_head_rows]` f32: the logits of every codebook.
    pub logits: Tensor,
    /// `[head_dim / 2]` f32: the RoPE inverse frequencies.
    pub inv_freq: Tensor,
 	/// `[batches·heads_q, t, t]` f16: the attention scores.
    pub scores: Tensor,
}

impl Activations {
    pub fn plan(plan: &mut jeuvoice_gpu::ArenaPlan, cfg: &OmniVoiceConfig) -> Activations {
        use jeuvoice_gpu::DType;
        let (d, q_w, kv_w, mlp) = (cfg.hidden_size, cfg.q_width(), cfg.kv_width(), cfg.intermediate_size);
        let (t, b, hq, hkv, hd) = (crate::weights::T_MAX, crate::weights::BATCH, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let rows = b * t;
        Activations {
            x: plan.add(DType::F32, &[rows, d]),
            h: plan.add(DType::F16, &[rows, d]),
            qkv: plan.add(DType::F16, &[rows, q_w + 2 * kv_w]),
            q: plan.add(DType::F16, &[b, hq, t, hd]),
            k: plan.add(DType::F16, &[b, hkv, t, hd]),
            v: plan.add(DType::F16, &[b, hkv, t, hd]),
            att: plan.add(DType::F16, &[b, hq, t, hd]),
            att_row: plan.add(DType::F16, &[rows, q_w]),
            gate: plan.add(DType::F16, &[rows, mlp]),
            gate_up: plan.add(DType::F16, &[rows, 2 * mlp]),
            logits: plan.add(DType::F32, &[rows, cfg.audio_head_rows()]),
            inv_freq: plan.add(DType::F32, &[hd / 2]),
            scores: plan.add(DType::F16, &[b * hq, t, t]),
            conf: plan.add(DType::F32, &[cfg.num_audio_codebook * t]),
            pick: plan.add(DType::I32, &[cfg.num_audio_codebook * t]),
            ids: plan.add(DType::I32, &[b * cfg.num_audio_codebook, t]),
            mask: plan.add(DType::I32, &[b, t]),
            valid: plan.add(DType::I32, &[b]),
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
        // The adapter must be complete before the arena is allocated: a missing pair or a
        // wrong replacement shape is a broken download, not a quiet base run.
        if let Some(idx) = &adapter_idx {
            let shapes = |index: &TensorIndex| -> Shapes {
                index.names().map(|n| (n.to_string(), index.get(n).expect("the name came from the index").shape.clone())).collect()
            };
            check_adapter(&shapes(&base), &shapes(idx))?;
        }

        // The plan: every float tensor of the checkpoint, and the activations after them. The
        // fused matrices (P3) replace their parts, so the parts are not planned.
        let mut plan = ArenaPlan::new();
        let mut planned: Vec<(String, Tensor, bool)> = Vec::new();
        let mut fused: HashMap<String, Vec<String>> = HashMap::new();
        for name in base.names() {
            let e = base.get(name).expect("the name came from the index");
            if e.dtype != crate::index::SafeDtype::F32 && e.dtype != crate::index::SafeDtype::F16 {
                continue; // codebook_layer_offsets is i64 and is derived from the config
            }
            if fused_part(name) {
                continue; // planned with the group that `fused_group` reports
            }
            match fused_group(name) {
                Some((fused_name, parts)) => {
                    let (rows, cols) = fused_shape(&base, &parts)?;
                    let t = plan.add(DType::F16, &[rows, cols]);
                    planned.push((fused_name.clone(), t, false));
                    fused.insert(fused_name, parts);
                }
                None => {
                    let f32_keep = stays_f32(name);
                    let dtype = if f32_keep { DType::F32 } else { DType::F16 };
                    planned.push((name.to_string(), plan.add(dtype, &e.shape), f32_keep));
                }
            }
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
            // A fused matrix reads its parts in order, each with its own LoRA merge.
            let values = match fused.remove(&name) {
                None => read_merged(&base, adapter_idx.as_ref(), &name, scale)?,
                Some(parts) => {
                    let mut all = Vec::with_capacity(t.numel());
                    for part in &parts {
                        all.extend(read_merged(&base, adapter_idx.as_ref(), part, scale)?);
                    }
                    all
                }
            };
            match f32_keep {
                true => arena.upload(gpu, &t, &values)?,
                false => {
                    let half: Vec<half::f16> = values.iter().map(|&v| half::f16::from_f32(v)).collect();
                    arena.upload(gpu, &t, &half)?;
                }
            }
            tensors.insert(name, t);
        }

        // The RoPE table, once. The positions follow the head-row layout of the pass, and
        // the forward pass uploads the prefix it uses.
        let inv = jeuvoice_gpu::ops::rope_inv_freq(cfg.rope_theta, cfg.head_dim);
        arena.upload(gpu, &act.inv_freq, &inv)?;

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

    /// True when the arena holds `name`. The fused matrices replace their parts (P3), so a
    /// part is absent.
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// The parts of the fused attention matrix, in row order: q | k | v.
const QKV: [&str; 3] = [".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"];
/// The parts of the fused MLP matrix, in row order: gate | up.
const GATE_UP: [&str; 2] = [".mlp.gate_proj", ".mlp.up_proj"];

/// The fused matrix that `name` starts: its new name and its parts in row order. `None` when
/// `name` does not start a group.
fn fused_group(name: &str) -> Option<(String, Vec<String>)> {
    let stem = name.strip_suffix(".weight")?;
    for (parts, fused) in [(QKV.as_slice(), ".self_attn.qkv_proj"), (GATE_UP.as_slice(), ".mlp.gate_up_proj")] {
        if let Some(layer) = stem.strip_suffix(parts[0]) {
            let parts = parts.iter().map(|p| format!("{layer}{p}.weight")).collect();
            return Some((format!("{layer}{fused}.weight"), parts));
        }
    }
    None
}

/// True for a part that a group of `fused_group` holds, but which does not start the group.
fn fused_part(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".weight") else { return false };
    [QKV[1], QKV[2], GATE_UP[1]].iter().any(|p| stem.ends_with(p))
}

/// The shape of a fused matrix: the rows of the parts one below the other, one common width.
fn fused_shape(index: &TensorIndex, parts: &[String]) -> Result<(usize, usize)> {
    let (mut rows, mut cols) = (0usize, None);
    for part in parts {
        let e = index.get(part).ok_or_else(|| {
            TtsError::ModelLoad(format!("the checkpoint has no {part}, so the fused matrix cannot be built"))
        })?;
        if e.shape.len() != 2 {
            return Err(TtsError::ModelLoad(format!("{part} is {:?}, not a matrix", e.shape)));
        }
        if let Some(c) = cols {
            if c != e.shape[1] {
                return Err(TtsError::ModelLoad(format!("{part} has {} columns, the fused matrix needs {c}", e.shape[1])));
            }
        } else {
            cols = Some(e.shape[1]);
        }
        rows += e.shape[0];
    }
    Ok((rows, cols.unwrap_or(0)))
}

/// The f32 values of one checkpoint tensor, with the adapter's LoRA merged or its replacement
/// table applied.
fn read_merged(base: &TensorIndex, adapter: Option<&TensorIndex>, name: &str, scale: f32) -> Result<Vec<f32>> {
    let mut values = base.read_f32(name)?;
    if let Some(a) = adapter {
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
                merge_pair(a, &lora_a, name, &mut values, scale)?;
            }
        }
    }
    Ok(values)
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
