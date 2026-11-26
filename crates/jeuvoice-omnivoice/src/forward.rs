//! The OmniVoice (Qwen3) forward pass: the mixed embedding, 28 blocks of bidirectional
//! attention and SwiGLU, the final RMSNorm and the audio head (plan Phase 2, Task 7).
//!
//! The layout of one row of the sequence is the row-major `[positions, hidden]` of the GEMMs.
//! The attention needs `[batches, heads, positions, width]`, so the projections go through
//! `to_heads` and back through `from_heads`; the copies are small next to the score matrices.

use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::ops::{
    BidirScratch, attention_bidir, embed_mixed, f16_to_f32, from_heads, rmsnorm_rows_warp, rmsnorm_to_f16, rope_rows, swiglu_strided, to_heads_at,
};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, Tensor};

use crate::config::OmniVoiceConfig;
use crate::{Result, TtsError};

/// One pass over `rows` sequences of `t` positions each, with the ids of `[rows·codebooks, t]`
/// and the audio mask of `[rows, t]`. `taps` gets the embedding and every block output.
///
/// The logits of `[rows·t, audio_head_rows]` stay in the arena: the caller reads the rows of
/// the positions it needs.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    w: &crate::weights::OmniVoiceWeights,
    cfg: &OmniVoiceConfig,
    gpu: &Gpu,
    blas: &Blas,
    ids: &[i32],
    audio_mask: &[i32],
    rows: usize,
    t: usize,
    valid: Option<&[i32]>,
    // When set, the final norm and the audio head process only this block of rows (the start
    // row and the count), and the logits are written compactly from row zero. The decode uses
    // it for the target frames of both rows; the tap path passes `None`.
    head: Option<(usize, usize)>,
    taps: &mut dyn FnMut(&str, &Arena, &Tensor),
) -> Result<()> {
    forward_inner(w, cfg, gpu, blas, ids, audio_mask, rows, t, valid, head, taps, false)
}

/// The same pass without the input uploads. Use this inside a CUDA graph capture. The caller
/// uploads the ids and the mask into the arena before every replay. The captured nodes then
/// read the fixed device addresses of the arena. `valid` must be `None`, because the padded
/// path allocates a scratch arena while it enqueues.
///
/// This function does not check the ids and the mask. The caller must check them with
/// `inputs::check` before the first enqueue, as `forward` does.
///
/// # Panics
/// On a padded pass, which cannot be captured.
#[allow(clippy::too_many_arguments)]
pub fn forward_compute(
    w: &crate::weights::OmniVoiceWeights,
    cfg: &OmniVoiceConfig,
    gpu: &Gpu,
    blas: &Blas,
    rows: usize,
    t: usize,
    head: Option<(usize, usize)>,
    taps: &mut dyn FnMut(&str, &Arena, &Tensor),
) -> Result<()> {
    let empty = [];
    forward_inner(w, cfg, gpu, blas, &empty, &empty, rows, t, None, head, taps, true)
}

#[allow(clippy::too_many_arguments)]
fn forward_inner(
    w: &crate::weights::OmniVoiceWeights,
    cfg: &OmniVoiceConfig,
    gpu: &Gpu,
    blas: &Blas,
    ids: &[i32],
    audio_mask: &[i32],
    rows: usize,
    t: usize,
    valid: Option<&[i32]>,
    head: Option<(usize, usize)>,
    taps: &mut dyn FnMut(&str, &Arena, &Tensor),
    skip_uploads: bool,
) -> Result<()> {
    let ar = &w.arena;
    let a = &w.act;
    if rows == 0 || t == 0 {
        return Ok(());
    }
    if rows > crate::weights::BATCH || t > crate::weights::T_MAX {
        return Err(TtsError::Render(format!(
            "a pass of {rows} rows and {t} positions does not fit the plan ({} rows, {} positions)",
            crate::weights::BATCH,
            crate::weights::T_MAX
        )));
    }
    let n = rows * t;
    let (d, q_w, kv_w, mlp) = (cfg.hidden_size, cfg.q_width(), cfg.kv_width(), cfg.intermediate_size);
    let (hq, hkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
    let eps = cfg.rms_norm_eps;

    // The ids, the mask and the lengths live in this arena, so the caller may keep its own
    // copies anywhere.
    let codebooks = cfg.num_audio_codebook;
    // The ids index the embedding tables, so they are checked before any upload.
    let limits = crate::inputs::Limits {
        codebooks,
        vocab: cfg.vocab_size,
        audio_vocab: cfg.audio_vocab_size,
    };
    if !skip_uploads {
        crate::inputs::check(ids, audio_mask, rows, t, limits, valid)?;
    }
    // The tensors are planned for the longest pass, so the slices are prefixes of them.
    let ids_all = a.ids.prefix_as(&[rows * codebooks, t]);
    let q_all = a.q.prefix_as(&[rows, hq, t, hd]);
    let k_all = a.k.prefix_as(&[rows, hkv, t, hd]);
    let v_all = a.v.prefix_as(&[rows, hkv, t, hd]);
    let att_all = a.att.prefix_as(&[rows, hq, t, hd]);
    let mask_all = a.mask.prefix_as(&[rows, t]);
    if !skip_uploads {
        ar.upload(gpu, &ids_all, ids)?;
        ar.upload(gpu, &mask_all, audio_mask)?;
    }
    let padded = valid.is_some();
    // Only the padded reference path needs the score plane. Production uses flash attention and
    // keeps this 67 MB tensor out of the model arena. The test path allocates it once per pass.
    let padded_scratch = if padded {
        let mut plan = ArenaPlan::new();
        let scores = plan.add(DType::F16, &[rows * hq, t, t]);
        Some((Arena::new(gpu, &plan)?, scores))
    } else {
        None
    };
    if let Some(v) = valid {
        // The tensor holds one value per planned row, so a short pass pads with `t`.
        let mut lengths = vec![t as i32; crate::weights::BATCH];
        lengths[..rows].copy_from_slice(v);
        ar.upload(gpu, &a.valid, &lengths)?;
    }

    // The position of a head row is its token position, and the RoPE kernels derive it from the
    // row, so the pass uploads no position table (P4).

    // 1. The embedding: audio frames sum 8 codebook rows, text positions read the token.
    //    `embed_mixed` writes one row at a time, so the rows are separate calls.
    for r in 0..rows {
        // Scalar kernels read the ids and the mask, so the rows need no alignment.
        let ids_r = ids_all.rows_unaligned(r * codebooks, codebooks);
        let mask_r = mask_all.rows_unaligned(r, 1);
        embed_mixed(gpu, ar, &ids_r, &mask_r, w.get("llm.embed_tokens.weight"), w.get("audio_embeddings.weight"), &a.h.rows(r * t, t))?;
    }
    // The residual stream starts as the embedding, in f32.
    widen(gpu, ar, &a.h.rows(0, n), &a.x.rows(0, n))?;
    taps("embed", ar, &a.x.rows(0, n));

    for layer in 0..cfg.num_hidden_layers {
        let p = format!("llm.layers.{layer}");
        // 2. input_layernorm, then the fused attention projection: q | k | v in one matrix.
        rmsnorm_to_f16(gpu, ar, &a.x.rows(0, n), w.get(&format!("{p}.input_layernorm.weight")), &a.h.rows(0, n), eps)?;
        let qkv_w = q_w + 2 * kv_w;
        gemm(gpu, blas, ar, &a.h.rows(0, n), w.get(&format!("{p}.self_attn.qkv_proj.weight")), &a.qkv.rows(0, n), qkv_w)?;

        // 3. The head layout of each block, q/k-norm per head, and RoPE.
        to_heads_at(gpu, ar, &a.qkv.rows(0, n), 0, &q_all, rows, t, hq, hd)?;
        to_heads_at(gpu, ar, &a.qkv.rows(0, n), q_w, &k_all, rows, t, hkv, hd)?;
        to_heads_at(gpu, ar, &a.qkv.rows(0, n), q_w + kv_w, &v_all, rows, t, hkv, hd)?;
        // The head rows are contiguous, so the row kernels see them as [rows·heads, head_dim].
        let q_flat = q_all.reshaped(&[rows * hq * t, hd]);
        let k_flat = k_all.reshaped(&[rows * hkv * t, hd]);
        rmsnorm_rows_warp(gpu, ar, &q_flat, w.get(&format!("{p}.self_attn.q_norm.weight")), eps)?;
        rmsnorm_rows_warp(gpu, ar, &k_flat, w.get(&format!("{p}.self_attn.k_norm.weight")), eps)?;
        rope_rows(gpu, ar, &q_flat, &a.inv_freq, t)?;
        rope_rows(gpu, ar, &k_flat, &a.inv_freq, t)?;

        // 4. The attention, then back to the row layout and the output projection. The
        //    projection accumulates into the f32 residual stream, so there is no f16 round
        //    trip and no second pass over the row.
        let valid_t = a.valid.rows_unaligned(0, rows);
        let scratch = BidirScratch { scores: padded_scratch.as_ref().map(|(_, tensor)| tensor) };
        let scratch_arena = padded_scratch.as_ref().map(|(arena, _)| arena).unwrap_or(ar);
        attention_bidir(gpu, ar, blas, &q_all, &k_all, &v_all, &att_all, scratch, scratch_arena, if padded { Some(&valid_t) } else { None })?;
        from_heads(gpu, ar, &att_all, &a.att_row.rows(0, n), rows, t, hq, hd)?;
        accumulate(gpu, blas, ar, &a.att_row.rows(0, n), w.get(&format!("{p}.self_attn.o_proj.weight")), &a.x.rows(0, n), d)?;

        // 5. post_attention_layernorm and the SwiGLU MLP, with gate and up in one matrix.
        rmsnorm_to_f16(gpu, ar, &a.x.rows(0, n), w.get(&format!("{p}.post_attention_layernorm.weight")), &a.h.rows(0, n), eps)?;
        gemm(gpu, blas, ar, &a.h.rows(0, n), w.get(&format!("{p}.mlp.gate_up_proj.weight")), &a.gate_up.rows(0, n), 2 * mlp)?;
        swiglu_strided(gpu, ar, &a.gate_up.rows(0, n), 0, mlp, &a.gate.rows(0, n))?;
        accumulate(gpu, blas, ar, &a.gate.rows(0, n), w.get(&format!("{p}.mlp.down_proj.weight")), &a.x.rows(0, n), d)?;

        taps(&format!("hidden_{layer:02}"), ar, &a.x.rows(0, n));
    }

    // 6. The final norm and the audio head, over the whole pass or over one block of rows.
    let (head_start, head_rows) = head.unwrap_or((0, n));
    if head_start + head_rows > n || head_rows == 0 {
        return Err(TtsError::Render(format!(
            "the head block {head_start}..{} is outside the {n} rows of the pass",
            head_start + head_rows
        )));
    }
    let hidden = a.x.rows(head_start, head_rows);
    rmsnorm_to_f16(gpu, ar, &hidden, w.get("llm.norm.weight"), &a.h.rows(0, head_rows), eps)?;
    taps("post_norm", ar, &a.h.rows(0, head_rows));
    let (hs, wh, ls) = (ar.slice(&a.h.rows(0, head_rows)), ar.slice(w.get("audio_heads.weight")), ar.slice(&a.logits.rows(0, head_rows)));
    blas.gemm_f16_f32out(gpu, &hs, Op::N, &wh, Op::T, &ls, head_rows, cfg.audio_head_rows(), d, 1.0, 0.0)?;
    Ok(())
}

/// `y [rows, n_in] = x [rows, n_in] · Wᵀ` with `W` stored `[n_out, n_in]`, as a Linear layer.
fn gemm(gpu: &Gpu, blas: &Blas, ar: &Arena, x: &Tensor, w: &Tensor, y: &Tensor, n_out: usize) -> Result<()> {
    let rows = x.shape()[0];
    let n_in = x.shape()[1];
    blas.gemm_f16(gpu, &ar.slice(x), Op::N, &ar.slice(w), Op::T, &ar.slice(y), rows, n_out, n_in, 1.0, 0.0)?;
    Ok(())
}

/// `x [rows, n_in] += x [rows, n_in] · Wᵀ` in f32: a projection accumulated into the residual
/// stream, with `W` stored `[n_out, n_in]` (`n_out == n_in` here).
fn accumulate(gpu: &Gpu, blas: &Blas, ar: &Arena, x: &Tensor, w: &Tensor, residual: &Tensor, n_out: usize) -> Result<()> {
    let rows = x.shape()[0];
    let n_in = x.shape()[1];
    blas.gemm_f16_f32out(gpu, &ar.slice(x), Op::N, &ar.slice(w), Op::T, &ar.slice(residual), rows, n_out, n_in, 1.0, 1.0)?;
    Ok(())
}

/// The wide copy of the embedding into the f32 residual stream.
fn widen(gpu: &Gpu, ar: &Arena, src: &Tensor, dst: &Tensor) -> Result<()> {
    f16_to_f32(gpu, ar, dst, src)?;
    Ok(())
}
