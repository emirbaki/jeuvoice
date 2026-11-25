//! The OmniVoice (Qwen3) forward pass: the mixed embedding, 28 blocks of bidirectional
//! attention and SwiGLU, the final RMSNorm and the audio head (plan Phase 2, Task 7).
//!
//! The layout of one row of the sequence is the row-major `[positions, hidden]` of the GEMMs.
//! The attention needs `[batches, heads, positions, width]`, so the projections go through
//! `to_heads` and back through `from_heads`; the copies are small next to the score matrices.

use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::ops::{
    BidirScratch, RopeKind, add_into_f32, attention_bidir, embed_mixed, f16_to_f32, from_heads, rmsnorm_rows, rmsnorm_to_f16, rope_with, swiglu, to_heads,
};
use jeuvoice_gpu::{Arena, Gpu, Tensor};

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
    taps: &mut dyn FnMut(&str, &Arena, &Tensor),
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
    if ids.len() != rows * codebooks * t || audio_mask.len() != rows * t {
        return Err(TtsError::Render(format!(
            "the pass needs {} ids and {} mask values, not {} and {}",
            rows * codebooks * t,
            rows * t,
            ids.len(),
            audio_mask.len()
        )));
    }
    // The tensors are planned for the longest pass, so the slices are prefixes of them.
    let ids_all = a.ids.prefix_as(&[rows * codebooks, t]);
    let q_all = a.q.prefix_as(&[rows, hq, t, hd]);
    let k_all = a.k.prefix_as(&[rows, hkv, t, hd]);
    let v_all = a.v.prefix_as(&[rows, hkv, t, hd]);
    let att_all = a.att.prefix_as(&[rows, hq, t, hd]);
    let scratch_all = BidirScratch {
        kx: &a.kx.prefix_as(&[rows, hq, t, hd]),
        vx: &a.vx.prefix_as(&[rows, hq, t, hd]),
        scores: &a.scores.prefix_as(&[rows * hq, t, t]),
    };
    let mask_all = a.mask.prefix_as(&[rows, t]);
    let pos_all = a.positions.prefix_as(&[rows * hq * t]);
    let pos_kv_all = a.positions_kv.prefix_as(&[rows * hkv * t]);
    ar.upload(gpu, &ids_all, ids)?;
    ar.upload(gpu, &mask_all, audio_mask)?;
    let padded = valid.is_some();
    if let Some(v) = valid {
        // The tensor holds one value per planned row, so a short pass pads with `t`.
        let mut lengths = vec![t as i32; crate::weights::BATCH];
        lengths[..rows].copy_from_slice(v);
        ar.upload(gpu, &a.valid, &lengths)?;
    }

    // The positions of every head row: the token position, the same for all heads of a frame.
    let positions = |heads: usize| {
        let mut pos = vec![0i32; rows * heads * t];
        for r in 0..rows {
            for h in 0..heads {
                for j in 0..t {
                    pos[(r * heads + h) * t + j] = j as i32;
                }
            }
        }
        pos
    };
    ar.upload(gpu, &pos_all, &positions(hq))?;
    ar.upload(gpu, &pos_kv_all, &positions(hkv))?;

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
        // 2. input_layernorm, then the three projections.
        rmsnorm_to_f16(gpu, ar, &a.x.rows(0, n), w.get(&format!("{p}.input_layernorm.weight")), &a.h.rows(0, n), eps)?;
        gemm(gpu, blas, ar, &a.h.rows(0, n), w.get(&format!("{p}.self_attn.q_proj.weight")), &a.q_row.rows(0, n), q_w)?;
        gemm(gpu, blas, ar, &a.h.rows(0, n), w.get(&format!("{p}.self_attn.k_proj.weight")), &a.k_row.rows(0, n), kv_w)?;
        gemm(gpu, blas, ar, &a.h.rows(0, n), w.get(&format!("{p}.self_attn.v_proj.weight")), &a.v_row.rows(0, n), kv_w)?;

        // 3. The head layout, q/k-norm per head, and RoPE.
        to_heads(gpu, ar, &a.q_row.rows(0, n), &q_all, rows, t, hq, hd)?;
        to_heads(gpu, ar, &a.k_row.rows(0, n), &k_all, rows, t, hkv, hd)?;
        to_heads(gpu, ar, &a.v_row.rows(0, n), &v_all, rows, t, hkv, hd)?;
        // The head rows are contiguous, so the row kernels see them as [rows·heads, head_dim].
        let q_flat = q_all.reshaped(&[rows * hq * t, hd]);
        let k_flat = k_all.reshaped(&[rows * hkv * t, hd]);
        rmsnorm_rows(gpu, ar, &q_flat, w.get(&format!("{p}.self_attn.q_norm.weight")), eps)?;
        rmsnorm_rows(gpu, ar, &k_flat, w.get(&format!("{p}.self_attn.k_norm.weight")), eps)?;
        rope_with(gpu, ar, &q_flat, &pos_all, &a.inv_freq, RopeKind::HalfSplit)?;
        rope_with(gpu, ar, &k_flat, &pos_kv_all, &a.inv_freq, RopeKind::HalfSplit)?;

        // 4. The attention, then back to the row layout and the output projection.
        let valid_t = a.valid.rows_unaligned(0, rows);
        let scratch = BidirScratch { kx: scratch_all.kx, vx: scratch_all.vx, scores: scratch_all.scores };
        attention_bidir(gpu, ar, blas, &q_all, &k_all, &v_all, &att_all, scratch, if padded { Some(&valid_t) } else { None })?;
        from_heads(gpu, ar, &att_all, &a.att_row.rows(0, n), rows, t, hq, hd)?;
        gemm(gpu, blas, ar, &a.att_row.rows(0, n), w.get(&format!("{p}.self_attn.o_proj.weight")), &a.proj.rows(0, n), d)?;
        add_into_f32(gpu, ar, &a.x.rows(0, n), &a.proj.rows(0, n))?;

        // 5. post_attention_layernorm and the SwiGLU MLP.
        rmsnorm_to_f16(gpu, ar, &a.x.rows(0, n), w.get(&format!("{p}.post_attention_layernorm.weight")), &a.h.rows(0, n), eps)?;
        gemm(gpu, blas, ar, &a.h.rows(0, n), w.get(&format!("{p}.mlp.gate_proj.weight")), &a.gate.rows(0, n), mlp)?;
        gemm(gpu, blas, ar, &a.h.rows(0, n), w.get(&format!("{p}.mlp.up_proj.weight")), &a.up.rows(0, n), mlp)?;
        swiglu(gpu, ar, &a.gate.rows(0, n), &a.up.rows(0, n), &a.gate.rows(0, n))?;
        gemm(gpu, blas, ar, &a.gate.rows(0, n), w.get(&format!("{p}.mlp.down_proj.weight")), &a.proj.rows(0, n), d)?;
        add_into_f32(gpu, ar, &a.x.rows(0, n), &a.proj.rows(0, n))?;

        taps(&format!("hidden_{layer:02}"), ar, &a.x.rows(0, n));
    }

    // 6. The final norm and the audio head.
    rmsnorm_to_f16(gpu, ar, &a.x.rows(0, n), w.get("llm.norm.weight"), &a.h.rows(0, n), eps)?;
    taps("post_norm", ar, &a.h.rows(0, n));
    let (hs, wh, ls) = (ar.slice(&a.h.rows(0, n)), ar.slice(w.get("audio_heads.weight")), ar.slice(&a.logits.rows(0, n)));
    blas.gemm_f16_f32out(gpu, &hs, Op::N, &wh, Op::T, &ls, n, cfg.audio_head_rows(), d, 1.0, 0.0)?;
    Ok(())
}

/// `y [rows, n_in] = x [rows, n_in] · Wᵀ` with `W` stored `[n_out, n_in]`, as a Linear layer.
fn gemm(gpu: &Gpu, blas: &Blas, ar: &Arena, x: &Tensor, w: &Tensor, y: &Tensor, n_out: usize) -> Result<()> {
    let rows = x.shape()[0];
    let n_in = x.shape()[1];
    blas.gemm_f16(gpu, &ar.slice(x), Op::N, &ar.slice(w), Op::T, &ar.slice(y), rows, n_out, n_in, 1.0, 0.0)?;
    Ok(())
}

/// The wide copy of the embedding into the f32 residual stream.
fn widen(gpu: &Gpu, ar: &Arena, src: &Tensor, dst: &Tensor) -> Result<()> {
    f16_to_f32(gpu, ar, dst, src)?;
    Ok(())
}
