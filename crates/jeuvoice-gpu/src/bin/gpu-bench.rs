//! The kernel benchmark table. Build and run (release):
//!   scripts\cargo.cmd oxide build -- -p jeuvoice-gpu --bin gpu-bench --release
//!   target\release\gpu-bench.exe [name filter]

use half::f16;
use jeuvoice_gpu::bench::{Timing, pct_of_memory_limit, pct_of_tensor_limit, time_on_stream};
use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::ops::{
    AttnKind, ConvKind, DecodeAttnKind, ProbsOut, attention_decode_with, decode_split_scratch, GemmKind, LayerNormKind, SoftmaxKind, Width, attention, bias_gelu_with, conv1d, embed_one, embed_with, gemm_lesson, layernorm_with, rmsnorm, rmsnorm_naive,
    BidirScratch, RopeKind, SwigluKind, attention_bidir_grouped, attention_flash128, embed_audio_sum, product_topk, rmsnorm_rows, rmsnorm_rows_naive, rmsnorm_rows_warp, rope_with, softmax_rows_with, swiglu_strided, swiglu_with,
};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, GpuError};

/// One row of the table.
struct Row {
    name: String,
    size: String,
    t: Timing,
    /// "GB/s" or "TFLOP/s" value and unit.
    rate: (f64, &'static str),
    pct: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let gpu = Gpu::open(0)?;
    let info = gpu.info().clone();
    println!("{} | cc {}.{} | {} SMs | {:.1} GB VRAM", info.name, info.cc.0, info.cc.1, info.sm_count, info.total_vram as f64 / (1u64 << 30) as f64);
    println!("limits: {:.0} GB/s memory, {:.1} TFLOP/s fp16 tensor (fp32 accumulate)\n", info.mem_bandwidth_gbs, info.fp16_tensor_tflops);

    let mut rows = Vec::new();
    // GEMM rows: x [m, k] times W^T with W [n, k], like a Linear layer.
    for (m, n, k) in [(1500, 1280, 1280), (1500, 5120, 1280), (4096, 1024, 1024)] {
        let name = "gemm_f16".to_string();
        if name.contains(&filter) {
            rows.push(bench_gemm(&gpu, "gemm_f16", m, n, k)?);
        }
    }

    // The production shapes of one OmniVoice pass: 2 rows of 486 positions (972 rows).
    for (name, m, n, k) in [
        ("gemm_q", 972usize, 2048usize, 1024usize),
        ("gemm_kv", 972, 1024, 1024),
        ("gemm_o", 972, 1024, 2048),
        ("gemm_gate_up", 972, 3072, 1024),
        ("gemm_down", 972, 1024, 3072),
        ("gemm_head_f32out", 972, 8200, 1024),
    ] {
        if name.contains(&filter) {
            rows.push(if name.ends_with("f32out") {
                bench_gemm_f32out(&gpu, name, m, n, k)?
            } else {
                bench_gemm(&gpu, name, m, n, k)?
            });
        }
    }

    // Logits rows: the decoder output times the token table, f32 out. One row is one decoder
    // step: it reads the whole 133 MB table, so it shows the memory limit of a step.
    if "gemm_f16_f32out".contains(&filter) {
        for m in [1, 7] {
            rows.push(bench_gemm_f32out(&gpu, "gemm_f16_f32out", m, 51866, 1280)?);
        }
    }

    // RMSNorm rows: [4096, 1024], about the token count of one OmniVoice pass.
    for (name, dtype, naive) in [
        ("rmsnorm_naive_f32", DType::F32, true),
        ("rmsnorm_f32", DType::F32, false),
        ("rmsnorm_f16", DType::F16, false),
    ] {
        if name.contains(&filter) {
            rows.push(bench_rmsnorm(&gpu, name, dtype, naive, 4096, 1024)?);
        }
    }

    // Element-wise rows: the MLP hidden layer of the encoder, [1500, 5120].
    for (name, width) in [("bias_gelu_x1", Width::One), ("bias_gelu_x2", Width::Two), ("bias_gelu_x2_b32", Width::TwoPacked)] {
        if name.contains(&filter) {
            rows.push(bench_bias_gelu(&gpu, name, width, 1500, 5120)?);
        }
    }

    // LayerNorm rows: one encoder activation, [1500, 1280] out of 32 blocks.
    for (name, dtype, kind) in [
        ("layernorm_3pass_f16", DType::F16, LayerNormKind::ThreePass),
        ("layernorm_f16_regs", DType::F16, LayerNormKind::Registers),
        ("layernorm_f16", DType::F16, LayerNormKind::RegistersPairs),
        ("layernorm_f32", DType::F32, LayerNormKind::RegistersPairs),
        ("layernorm_f16_x4", DType::F16, LayerNormKind::RegistersPairs),
    ] {
        if name.contains(&filter) {
            // The x4 row is 4 times the rows of one window: the launch cost fades there.
            let n_rows = if name.ends_with("x4") { 6000 } else { 1500 };
            rows.push(bench_layernorm(&gpu, name, dtype, kind, n_rows, 1280)?);
        }
    }

    // Softmax rows: the attention scores of one encoder layer, [20 heads * 1500, 1500].
    for (name, kind) in [
        ("softmax_3pass_f16", SoftmaxKind::ThreePass),
        ("softmax_online_f16", SoftmaxKind::Online),
        ("softmax_regs_f16", SoftmaxKind::Registers),
    ] {
        if name.contains(&filter) {
            rows.push(bench_softmax(&gpu, name, kind, 20 * 1500, 1500)?);
        }
    }

    // GEMM lesson rows: the same shape as the cuBLAS row above, for lesson 4.
    for (name, kind) in [("gemm_naive_f16", GemmKind::Naive), ("gemm_tiled_f16", GemmKind::Tiled), ("gemm_regtile_f16", GemmKind::RegTile), ("gemm_tc_f16", GemmKind::TensorCore), ("gemm_tc_vec_f16", GemmKind::TensorCoreVec), ("gemm_tc_pre_f16", GemmKind::TensorCorePrefetch)] {
        if name.contains(&filter) {
            rows.push(bench_gemm_lesson(&gpu, name, kind, 1500, 1280, 1280)?);
        }
    }

    // Attention rows: the encoder's attention, 20 heads of 1500 rows.
    for (name, kind) in [("attention_naive_f16", AttnKind::Naive), ("attention_flash_f16", AttnKind::Flash), ("attention_flash_tc_f16", AttnKind::FlashTc)] {
        if name.contains(&filter) {
            rows.push(bench_attention(&gpu, name, kind, 20, 1500)?);
        }
    }

    // conv1d rows: the two convolutions of the Whisper encoder stem.
    for (name, kind) in [("conv1d_direct_f16", ConvKind::Direct), ("conv1d_im2col_f16", ConvKind::Im2col)] {
        if name.contains(&filter) {
            rows.push(bench_conv1d(&gpu, name, kind, 128, 1280, 3000, 1)?);
            rows.push(bench_conv1d(&gpu, name, kind, 1280, 1280, 3000, 2)?);
        }
    }

    // Embedding rows: one decoder step (1 row) and a long prompt or DTW pass (224 rows),
    // with the real table of 51866 x 1280. Each call uploads the ids first, as the decoder
    // does. The ids_upload row is that upload alone.
    for (name, width) in [("embed_x1", Width::One), ("embed_x2", Width::Two)] {
        if name.contains(&filter) {
            for n in [1, 224] {
                rows.push(bench_embed(&gpu, name, width, n)?);
            }
        }
    }
    // Decoder attention rows (20 heads of 64): one step against 448 cached keys, one step
    // of cross-attention on the 1500 encoder positions, a causal prompt of 224 rows, and the
    // DTW pass (224 rows of cross-attention with the softmax of 2 heads written out).
    for (name, kind) in [("attn_dec_naive", DecodeAttnKind::Naive), ("attn_dec_split", DecodeAttnKind::Split)] {
        if name.contains(&filter) {
            for (nq, nk, causal, probs) in [(1, 448, false, false), (1, 1500, false, false), (224, 224, true, false), (224, 1500, false, true)] {
                rows.push(bench_attn_decode(&gpu, name, kind, nq, nk, causal, probs)?);
            }
        }
    }

    if "embed_one".contains(&filter) {
        rows.push(bench_embed_one(&gpu)?);
    }
    if "ids_upload".contains(&filter) {
        rows.push(bench_ids_upload(&gpu)?);
    }

    // RoPE and the per-head RMSNorm rows of the Qwen3 backbone (Phase 2, lesson 10).
    // 8 MB in and 8 MB out: big enough that the 4 MB L2 does not hold the tensor.
    for (name, kind) in [
        ("rope_f16", RopeKind::HalfSplitShared),
        ("rope_f16_naive", RopeKind::HalfSplit),
        ("rope_f16_interleaved", RopeKind::Interleaved),
    ] {
        if name.contains(&filter) {
            rows.push(bench_rope(&gpu, name, kind, 2000, 16, 128)?);
        }
    }
    for (name, kind) in [("rmsnorm_rows_f16", "block"), ("rmsnorm_rows_warp", "warp"), ("rmsnorm_rows_naive", "naive")] {
        if name.contains(&filter) {
            rows.push(bench_rmsnorm_rows(&gpu, name, kind, 32_000, 128)?);
        }
    }
    // The derived-position RoPE of P4: the same 8 MB tensor as the rows above (32 000 head rows
    // are 16 heads of 2000 positions), with the position from the row and no table.
    if "rope_half_f16_rows".contains(&filter) {
        rows.push(bench_rope_rows(&gpu, "rope_half_f16_rows", 32_000, 128, 2000)?);
    }
    // The bidirectional GQA attention of one OmniVoice pass: 2 rows, 16 heads of 128, 486 positions.
    for (name, valid) in [
        ("attention_bidir_f16", None),
        ("attention_bidir_padded_f16", Some([486i32, 214i32])),
    ] {
        if name.contains(&filter) {
            rows.push(bench_attention_bidir(&gpu, name, 16, 486, 128, valid)?);
        }
    }
    // The guided peak of one decode step: 428 positions of 8 codebooks of 1025 tokens.
    if "guide_peak_f32".contains(&filter) {
        rows.push(bench_guide_peak(&gpu, "guide_peak_f32", 428 * 8, 8200, 1025, 8)?);
    }
    // Product top-k for one decode step: 1712 pairs, the top 103 tokens of 1025.
    if "product_topk_f32".contains(&filter) {
        rows.push(bench_product_topk(&gpu, "product_topk_f32", 214, 8, 1025)?);
    }
    // The 8-codebook audio embedding of one pass: [8, 486] ids into [8200, 1024].
    if "embed_audio_sum_f16".contains(&filter) {
        rows.push(bench_embed_audio(&gpu, "embed_audio_sum_f16", 486, 1024)?);
    }
    // SwiGLU on the MLP buffer of one pass: [486, 3072], read gate and up, write out.
    for (name, kind) in [("swiglu_f16", SwigluKind::Two), ("swiglu_f16_x1", SwigluKind::One)] {
        if name.contains(&filter) {
            rows.push(bench_swiglu(&gpu, name, kind, 486, 3072)?);
            // The production pass has two rows of 486 positions.
            rows.push(bench_swiglu(&gpu, name, kind, 972, 3072)?);
        }
    }
    // The fused gate/up row of P3: the same 486 x 3072 elements, read from one [rows, 6144]
    // tensor, so the three tensors are 6144, 6144 and 3072 values per row.
    if "swiglu_strided_f16".contains(&filter) {
        for rows_n in [486usize, 214] {
            rows.push(bench_swiglu_strided(&gpu, "swiglu_strided_f16", rows_n, 3072)?);
        }
    }

    // P3: the fused projections against the separate ones, at the two real pass shapes. One
    // conditioned pass is 486 rows, the unconditioned pass 214.
    for (name, m, n, k) in [
        ("gemm_p3_qkv_486", 486usize, 4096usize, 1024usize),
        ("gemm_p3_q_486", 486, 2048, 1024),
        ("gemm_p3_kv_486", 486, 1024, 1024),
        ("gemm_p3_qkv_214", 214, 4096, 1024),
        ("gemm_p3_q_214", 214, 2048, 1024),
        ("gemm_p3_kv_214", 214, 1024, 1024),
        ("gemm_p3_gate_up_486", 486, 6144, 1024),
        ("gemm_p3_gate_486", 486, 3072, 1024),
        ("gemm_p3_gate_up_214", 214, 6144, 1024),
        ("gemm_p3_gate_214", 214, 3072, 1024),
    ] {
        if name.contains(&filter) {
            rows.push(bench_gemm(&gpu, name, m, n, k)?);
        }
    }

    // The d128 flash attention of P5 against the grouped path at the two production shapes:
    // one conditioned pass of 486 positions and one unconditioned pass of 214.
    if "attn_p5".contains(&filter) {
        for t_len in [486usize, 214] {
            for (name, flash) in [("attn_p5_grouped", false), ("attn_p5_flash128", true)] {
                let row = format!("{name}_{t_len}");
                if row.contains(&filter) || filter.is_empty() {
                    rows.push(bench_attention_p5(&gpu, &row, t_len, flash)?);
                }
            }
        }
    }

    println!("{:<20} {:>16} {:>10} {:>14} {:>8}", "kernel", "size", "us", "rate", "% limit");
    for r in &rows {
        println!("{:<20} {:>16} {:>10.1} {:>9.1} {:<4} {:>7.0}%", r.name, r.size, r.t.us, r.rate.0, r.rate.1, r.pct);
    }
    Ok(())
}

fn bench_bias_gelu(gpu: &Gpu, name: &str, width: Width, rows: usize, d: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[rows, d]);
    let b = plan.add(DType::F16, &[d]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &x, &vec![f16::from_f32(0.25); rows * d])?;
    arena.upload(gpu, &b, &vec![f16::from_f32(0.0); d])?;
    let t = time_on_stream(gpu, 3, 50, || bias_gelu_with(gpu, &arena, &x, &b, width))?;
    // In place: read x once, write x once.
    let bytes = (2 * rows * d * 2) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

fn bench_embed(gpu: &Gpu, name: &str, width: Width, n: usize) -> Result<Row, GpuError> {
    let (vocab, d, n_ctx) = (51866, 1280, 448);
    let mut plan = ArenaPlan::new();
    let tok = plan.add(DType::F16, &[vocab, d]);
    let pos = plan.add(DType::F32, &[n_ctx, d]);
    let ids_t = plan.add(DType::I32, &[n_ctx]);
    let out = plan.add(DType::F16, &[n_ctx, d]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &tok, &vec![f16::from_f32(0.25); vocab * d])?;
    arena.upload(gpu, &pos, &vec![0.5f32; n_ctx * d])?;
    // Ids spread over the table, so the rows come from all over the 133 MB.
    let ids: Vec<u32> = (0..n as u32).map(|i| (i * 7919) % vocab as u32).collect();
    let t = time_on_stream(gpu, 3, 50, || embed_with(gpu, &arena, &ids, &ids_t, &tok, &pos, 0, &out, width))?;
    // Read one table row (f16) and one position row (f32), write one row (f16).
    let bytes = (n * d * (2 + 4 + 2)) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{n}x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

fn bench_embed_one(gpu: &Gpu) -> Result<Row, GpuError> {
    let (vocab, d, n_ctx) = (51866, 1280, 448);
    let mut plan = ArenaPlan::new();
    let tok = plan.add(DType::F16, &[vocab, d]);
    let pos = plan.add(DType::F32, &[n_ctx, d]);
    let out = plan.add(DType::F16, &[1, d]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &tok, &vec![f16::from_f32(0.25); vocab * d])?;
    arena.upload(gpu, &pos, &vec![0.5f32; n_ctx * d])?;
    let mut i = 0u32;
    let t = time_on_stream(gpu, 3, 50, || {
        i = (i + 7919) % vocab as u32;
        embed_one(gpu, &arena, i, &tok, &pos, 0, &out)
    })?;
    let bytes = (d * (2 + 4 + 2)) as u64;
    Ok(Row {
        name: "embed_one".into(),
        size: format!("1x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

fn bench_attn_decode(gpu: &Gpu, name: &str, kind: DecodeAttnKind, nq: usize, nk: usize, causal: bool, with_probs: bool) -> Result<Row, GpuError> {
    let (heads, w) = (20, 1280);
    let mut plan = ArenaPlan::new();
    let q = plan.add(DType::F16, &[nq, w]);
    let k = plan.add(DType::F16, &[nk, w]);
    let v = plan.add(DType::F16, &[nk, w]);
    let out = plan.add(DType::F16, &[nq, w]);
    let pt = plan.add(DType::F32, &[2, nq, nk]);
    let scratch = plan.add(DType::F32, &[decode_split_scratch(heads, nq, nk)]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &q, &vec![f16::from_f32(0.1); nq * w])?;
    arena.upload(gpu, &k, &vec![f16::from_f32(0.1); nk * w])?;
    arena.upload(gpu, &v, &vec![f16::from_f32(0.1); nk * w])?;
    let probs = with_probs.then_some(ProbsOut { probs: &pt, heads_mask: (1 << 4) | (1 << 11), slot0: 0 });
    let t = time_on_stream(gpu, 3, 50, || attention_decode_with(gpu, &arena, &q, &k, &v, &out, heads, causal, probs, Some(&scratch), kind))?;
    // The least memory traffic: K and V once, q and out once, and the probs.
    let mut bytes = (2 * nk * w * 2 + 2 * nq * w * 2) as u64;
    if with_probs {
        bytes += (2 * nq * nk * 4) as u64;
    }
    Ok(Row {
        name: name.into(),
        size: format!("{nq}x{nk}{}", if causal { " c" } else if with_probs { " p" } else { "" }),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

fn bench_ids_upload(gpu: &Gpu) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let ids_t = plan.add(DType::I32, &[448]);
    let arena = Arena::new(gpu, &plan)?;
    let host = vec![0i32; 448];
    let t = time_on_stream(gpu, 3, 50, || arena.upload(gpu, &ids_t, &host))?;
    let bytes = (448 * 4) as u64;
    Ok(Row {
        name: "ids_upload".into(),
        size: "448 i32".into(),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

fn bench_rmsnorm(gpu: &Gpu, name: &str, dtype: DType, naive: bool, rows: usize, d: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let x = plan.add(dtype, &[rows, d]);
    let w = plan.add(dtype, &[d]);
    let y = plan.add(dtype, &[rows, d]);
    let arena = Arena::new(gpu, &plan)?;
    match dtype {
        DType::F16 => {
            arena.upload(gpu, &x, &vec![f16::from_f32(0.5); rows * d])?;
            arena.upload(gpu, &w, &vec![f16::from_f32(1.0); d])?;
        }
        _ => {
            arena.upload(gpu, &x, &vec![0.5f32; rows * d])?;
            arena.upload(gpu, &w, &vec![1.0f32; d])?;
        }
    }
    let t = time_on_stream(gpu, 3, 50, || {
        if naive {
            rmsnorm_naive(gpu, &arena, &x, &w, &y, 1e-6)
        } else {
            rmsnorm(gpu, &arena, &x, &w, &y, 1e-6)
        }
    })?;
    // Bytes moved: read x once, write y once. w is small and stays in the cache.
    let bytes = (2 * rows * d * dtype.size()) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

/// LayerNorm moves x in and y out. gamma and beta are one row and stay in the cache.
fn bench_layernorm(gpu: &Gpu, name: &str, dtype: DType, kind: LayerNormKind, rows: usize, d: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let x = plan.add(dtype, &[rows, d]);
    let g = plan.add(dtype, &[d]);
    let b = plan.add(dtype, &[d]);
    let y = plan.add(dtype, &[rows, d]);
    let arena = Arena::new(gpu, &plan)?;
    match dtype {
        DType::F16 => {
            arena.upload(gpu, &x, &vec![f16::from_f32(0.5); rows * d])?;
            arena.upload(gpu, &g, &vec![f16::from_f32(1.0); d])?;
            arena.upload(gpu, &b, &vec![f16::from_f32(0.0); d])?;
        }
        _ => {
            arena.upload(gpu, &x, &vec![0.5f32; rows * d])?;
            arena.upload(gpu, &g, &vec![1.0f32; d])?;
            arena.upload(gpu, &b, &vec![0.0f32; d])?;
        }
    }
    let t = time_on_stream(gpu, 3, 50, || layernorm_with(gpu, &arena, &x, &g, &b, &y, 1e-5, kind))?;
    // Bytes moved: read x once, write y once.
    let bytes = (2 * rows * d * dtype.size()) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

/// Softmax works in place: read x, write x.
fn bench_softmax(gpu: &Gpu, name: &str, kind: SoftmaxKind, rows: usize, n: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[rows, n]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &x, &vec![f16::from_f32(0.25); rows * n])?;
    let t = time_on_stream(gpu, 3, 50, || softmax_rows_with(gpu, &arena, &x, 0.125, None, kind))?;
    let bytes = (2 * rows * n * 2) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{n}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

/// A lesson GEMM (lesson 4) on the encoder's projection shape.
fn bench_gemm_lesson(gpu: &Gpu, name: &str, kind: GemmKind, m: usize, n: usize, k: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let a = plan.add(DType::F16, &[m, k]);
    let b = plan.add(DType::F16, &[k, n]);
    let c = plan.add(DType::F16, &[m, n]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &a, &vec![f16::from_f32(0.01); m * k])?;
    arena.upload(gpu, &b, &vec![f16::from_f32(0.02); k * n])?;
    let t = time_on_stream(gpu, 3, 10, || gemm_lesson(gpu, &arena, kind, &a, &b, &c))?;
    let flops = 2 * (m * n * k) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{m}x{n}x{k}"),
        t,
        rate: (flops as f64 / (t.us * 1e-6) / 1e12, "TF/s"),
        pct: pct_of_tensor_limit(flops, &t, gpu.info()),
    })
}

/// One attention pass over the encoder's shapes: 20 heads of 1500 x 64.
fn bench_attention(gpu: &Gpu, name: &str, kind: AttnKind, heads: usize, seq: usize) -> Result<Row, GpuError> {
    let blas = Blas::load(gpu)?;
    let mut plan = ArenaPlan::new();
    let q = plan.add(DType::F16, &[heads, seq, 64]);
    let k = plan.add(DType::F16, &[heads, seq, 64]);
    let v = plan.add(DType::F16, &[heads, seq, 64]);
    let out = plan.add(DType::F16, &[heads, seq, 64]);
    let scratch = plan.add(DType::F16, &[heads, seq, seq]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &q, &vec![f16::from_f32(0.01); heads * seq * 64])?;
    arena.upload(gpu, &k, &vec![f16::from_f32(0.01); heads * seq * 64])?;
    arena.upload(gpu, &v, &vec![f16::from_f32(0.01); heads * seq * 64])?;
    let t = time_on_stream(gpu, 3, 20, || attention(gpu, &arena, &blas, &q, &k, &v, &out, Some(&scratch), kind))?;
    // Two matrix multiplies: Q·K^T and probs·V.
    let flops = 4 * (heads * seq * seq * 64) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{heads}x{seq}x64"),
        t,
        rate: (flops as f64 / (t.us * 1e-6) / 1e12, "TF/s"),
        pct: pct_of_tensor_limit(flops, &t, gpu.info()),
    })
}

#[allow(clippy::too_many_arguments)]
fn bench_conv1d(gpu: &Gpu, name: &str, kind: ConvKind, c_in: usize, c_out: usize, t_in: usize, stride: usize) -> Result<Row, GpuError> {
    let blas = Blas::load(gpu)?;
    let t_out = (t_in - 1) / stride + 1;
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[c_in, t_in]);
    let w = plan.add(DType::F16, &[c_out, c_in, 3]);
    let y = plan.add(DType::F16, &[c_out, t_out]);
    let col = plan.add(DType::F16, &[c_in * 3, t_out]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &x, &vec![f16::from_f32(0.01); c_in * t_in])?;
    arena.upload(gpu, &w, &vec![f16::from_f32(0.02); c_out * c_in * 3])?;
    let scratch = if kind == ConvKind::Im2col { Some(&col) } else { None };
    let t = time_on_stream(gpu, 2, 10, || conv1d(gpu, &arena, &blas, &x, &w, &y, stride, scratch, kind))?;
    let flops = 2 * (c_out * t_out * c_in * 3) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{c_in}>{c_out} t{t_in}/{stride}"),
        t,
        rate: (flops as f64 / (t.us * 1e-6) / 1e12, "TF/s"),
        pct: pct_of_tensor_limit(flops, &t, gpu.info()),
    })
}

fn bench_gemm_f32out(gpu: &Gpu, name: &str, m: usize, n: usize, k: usize) -> Result<Row, GpuError> {
    let blas = Blas::load(gpu)?;
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[m, k]);
    let w = plan.add(DType::F16, &[n, k]);
    let y = plan.add(DType::F32, &[m, n]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &x, &vec![f16::from_f32(0.01); m * k])?;
    arena.upload(gpu, &w, &vec![f16::from_f32(0.02); n * k])?;
    let (xs, ws, ys) = (arena.slice(&x), arena.slice(&w), arena.slice(&y));
    let t = time_on_stream(gpu, 3, 20, || blas.gemm_f16_f32out(gpu, &xs, Op::N, &ws, Op::T, &ys, m, n, k, 1.0, 0.0))?;
    // A few rows: the table read sets the time, so the limit is the memory.
    let bytes = (n * k * 2 + m * k * 2 + m * n * 4) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{m}x{n}x{k}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

/// RoPE rotates in place: read and write every value once.
fn bench_rope(gpu: &Gpu, name: &str, kind: RopeKind, rows: usize, heads: usize, d: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[rows, heads, d]);
    let p = plan.add(DType::I32, &[rows]);
    let f = plan.add(DType::F32, &[d / 2]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &x, &vec![f16::from_f32(0.5); rows * heads * d])?;
    arena.upload(gpu, &p, &vec![3i32; rows])?;
    arena.upload(gpu, &f, &jeuvoice_gpu::ops::rope_inv_freq(1_000_000.0, d))?;
    let t = time_on_stream(gpu, 3, 50, || rope_with(gpu, &arena, &x, &p, &f, kind))?;
    let bytes = (2 * rows * heads * d * 2) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{heads}x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

/// RoPE with the position taken from the row: the same traffic as `bench_rope`, no table read.
fn bench_rope_rows(gpu: &Gpu, name: &str, rows: usize, d: usize, positions: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[rows, d]);
    let f = plan.add(DType::F32, &[d / 2]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &x, &vec![f16::from_f32(0.5); rows * d])?;
    arena.upload(gpu, &f, &jeuvoice_gpu::ops::rope_inv_freq(1_000_000.0, d))?;
    let t = time_on_stream(gpu, 3, 50, || jeuvoice_gpu::ops::rope_rows(gpu, &arena, &x, &f, positions))?;
    let bytes = (2 * rows * d * 2) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{d}, {positions} positions"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

/// As `bench_rope`, for the per-head RMSNorm.
fn bench_rmsnorm_rows(gpu: &Gpu, name: &str, kind: &str, rows: usize, d: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[rows, d]);
    let w = plan.add(DType::F32, &[d]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &x, &vec![f16::from_f32(0.5); rows * d])?;
    arena.upload(gpu, &w, &vec![1.0f32; d])?;
    let t = time_on_stream(gpu, 3, 50, || match kind {
        "warp" => rmsnorm_rows_warp(gpu, &arena, &x, &w, 1e-6),
        "naive" => rmsnorm_rows_naive(gpu, &arena, &x, &w, 1e-6),
        _ => rmsnorm_rows(gpu, &arena, &x, &w, 1e-6),
    })?;
    let bytes = (2 * rows * d * 2) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

/// The GQA attention of OmniVoice: q·kᵀ, the softmax, then probs·v, per batch and head.
fn bench_attention_bidir(gpu: &Gpu, name: &str, heads: usize, t_len: usize, d: usize, valid: Option<[i32; 2]>) -> Result<Row, GpuError> {
    let blas = Blas::load(gpu)?;
    let mut plan = ArenaPlan::new();
    let q = plan.add(DType::F16, &[2, heads, t_len, d]);
    let k = plan.add(DType::F16, &[2, heads / 2, t_len, d]);
    let v = plan.add(DType::F16, &[2, heads / 2, t_len, d]);
    let out = plan.add(DType::F16, &[2, heads, t_len, d]);
    let scores = plan.add(DType::F16, &[2 * heads, t_len, t_len]);
    let lens = valid.map(|_| plan.add(DType::I32, &[2]));
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &q, &vec![f16::from_f32(0.01); 2 * heads * t_len * d])?;
    arena.upload(gpu, &k, &vec![f16::from_f32(0.01); heads * t_len * d])?;
    arena.upload(gpu, &v, &vec![f16::from_f32(0.01); heads * t_len * d])?;
    if let (Some(v), Some(t)) = (valid, lens.as_ref()) {
        arena.upload(gpu, t, &v)?;
    }
    let t = time_on_stream(gpu, 3, 10, || attention_bidir_grouped(gpu, &arena, &arena, &blas, &q, &k, &v, &out, BidirScratch { scores: Some(&scores) }, lens.as_ref()))?;
    // Two products of q·kᵀ and probs·v per batch and head.
    let flops = 4 * (2 * heads * t_len * t_len * d) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("2x{heads}x{t_len}x{d}{}", if valid.is_some() { " padded" } else { "" }),
        t,
        rate: (flops as f64 / (t.us * 1e-6) / 1e12, "TF/s"),
        pct: pct_of_tensor_limit(flops, &t, gpu.info()),
    })
}

/// The d128 attention of one OmniVoice pass: the grouped path (cuBLAS plus a score plane in
/// DRAM) or the flash kernel, at `batches = 1`.
fn bench_attention_p5(gpu: &Gpu, name: &str, t_len: usize, flash: bool) -> Result<Row, GpuError> {
    let blas = Blas::load(gpu)?;
    let (heads, h_kv, d) = (16usize, 8usize, 128usize);
    let mut plan = ArenaPlan::new();
    let q = plan.add(DType::F16, &[1, heads, t_len, d]);
    let k = plan.add(DType::F16, &[1, h_kv, t_len, d]);
    let v = plan.add(DType::F16, &[1, h_kv, t_len, d]);
    let out = plan.add(DType::F16, &[1, heads, t_len, d]);
    let scores = plan.add(DType::F16, &[heads, t_len, t_len]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &q, &vec![f16::from_f32(0.01); heads * t_len * d])?;
    arena.upload(gpu, &k, &vec![f16::from_f32(0.01); h_kv * t_len * d])?;
    arena.upload(gpu, &v, &vec![f16::from_f32(0.01); h_kv * t_len * d])?;
    let t = time_on_stream(gpu, 3, 20, || {
        if flash {
            attention_flash128(gpu, &arena, &q, &k, &v, &out)
        } else {
            attention_bidir_grouped(gpu, &arena, &arena, &blas, &q, &k, &v, &out, BidirScratch { scores: Some(&scores) }, None)
        }
    })?;
    // Two products of q·kᵀ and probs·v per head.
    let flops = 4 * (heads * t_len * t_len * d) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("1x{heads}x{t_len}x{d}"),
        t,
        rate: (flops as f64 / (t.us * 1e-6) / 1e12, "TF/s"),
        pct: pct_of_tensor_limit(flops, &t, gpu.info()),
    })
}

/// The audio embedding reads eight table rows per frame and writes one row.
fn bench_embed_audio(gpu: &Gpu, name: &str, rows: usize, d: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let ids = plan.add(DType::I32, &[8, rows]);
    let table = plan.add(DType::F16, &[8 * 1025, d]);
    let out = plan.add(DType::F16, &[rows, d]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &ids, &vec![7i32; 8 * rows])?;
    arena.upload(gpu, &table, &vec![f16::from_f32(0.01); 8 * 1025 * d])?;
    let t = time_on_stream(gpu, 3, 50, || embed_audio_sum(gpu, &arena, &ids, &table, &out))?;
    // Eight reads and one write per output value.
    let bytes = (9 * rows * d * 2) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

/// SwiGLU reads gate and up and writes out: three tensors of the same size.
fn bench_swiglu(gpu: &Gpu, name: &str, kind: SwigluKind, rows: usize, d: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let gate = plan.add(DType::F16, &[rows, d]);
    let up = plan.add(DType::F16, &[rows, d]);
    let out = plan.add(DType::F16, &[rows, d]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &gate, &vec![f16::from_f32(0.5); rows * d])?;
    arena.upload(gpu, &up, &vec![f16::from_f32(0.25); rows * d])?;
    let t = time_on_stream(gpu, 3, 50, || swiglu_with(gpu, &arena, &gate, &up, &out, kind))?;
    let bytes = (3 * rows * d * 2) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

fn bench_swiglu_strided(gpu: &Gpu, name: &str, rows: usize, d: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let fused = plan.add(DType::F16, &[rows, 2 * d]);
    let out = plan.add(DType::F16, &[rows, d]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &fused, &vec![f16::from_f32(0.5); rows * 2 * d])?;
    let t = time_on_stream(gpu, 3, 50, || swiglu_strided(gpu, &arena, &fused, 0, d, &out))?;
    // Two reads of one wide row and one write of the narrow row: the same bytes as the
    // separate version, only the addresses differ.
    let bytes = (3 * rows * d * 2) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{d} of {rows}x{}", 2 * d),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

fn bench_gemm(gpu: &Gpu, name: &str, m: usize, n: usize, k: usize) -> Result<Row, GpuError> {
    let blas = Blas::load(gpu)?;
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[m, k]);
    let w = plan.add(DType::F16, &[n, k]);
    let y = plan.add(DType::F16, &[m, n]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &x, &vec![f16::from_f32(0.01); m * k])?;
    arena.upload(gpu, &w, &vec![f16::from_f32(0.02); n * k])?;
    let (xs, ws, ys) = (arena.slice(&x), arena.slice(&w), arena.slice(&y));
    let t = time_on_stream(gpu, 3, 20, || blas.gemm_f16(gpu, &xs, Op::N, &ws, Op::T, &ys, m, n, k, 1.0, 0.0))?;
    let flops = 2 * (m * n * k) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{m}x{n}x{k}"),
        t,
        rate: (flops as f64 / (t.us * 1e-6) / 1e12, "TF/s"),
        pct: pct_of_tensor_limit(flops, &t, gpu.info()),
    })
}

/// The guided peak of the decode: one block per (position, codebook) pair, two logit planes of
/// `vocab` values read per pair.
fn bench_guide_peak(gpu: &Gpu, name: &str, pairs: usize, width: usize, vocab: usize, codebooks: usize) -> Result<Row, GpuError> {
    let positions = pairs / codebooks;
    let mut plan = ArenaPlan::new();
    let c = plan.add(DType::F32, &[positions, width]);
    let u = plan.add(DType::F32, &[positions, width]);
    let conf = plan.add(DType::F32, &[pairs]);
    let pick = plan.add(DType::I32, &[pairs]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &c, &vec![0.01f32; positions * width])?;
    arena.upload(gpu, &u, &vec![0.02f32; positions * width])?;
    let t = time_on_stream(gpu, 3, 20, || {
        jeuvoice_gpu::ops::guide_peak(gpu, &arena, &c, &u, &conf, &pick, pairs, width, vocab, codebooks, 1024, 2.5)
    })?;
    let bytes = 2 * pairs * vocab * 4;
    Ok(Row {
        name: name.into(),
        size: format!("{pairs} pairs of {vocab}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: bytes as f64 / (t.us * 1e-6) / (gpu.info().mem_bandwidth_gbs * 1e9) * 100.0,
    })
}

/// Product sampling: the CPU keeps the Gumbel stream and reads only the top-k ids and values.
fn bench_product_topk(gpu: &Gpu, name: &str, target: usize, codebooks: usize, vocab: usize) -> Result<Row, GpuError> {
    let (pairs, width, k) = (target * codebooks, codebooks * vocab, (vocab as f64 * 0.1).ceil() as usize);
    let mut plan = ArenaPlan::new();
    let c = plan.add(DType::F32, &[target, width]);
    let u = plan.add(DType::F32, &[target, width]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &c, &vec![0.01f32; target * width])?;
    arena.upload(gpu, &u, &vec![0.02f32; target * width])?;

    let mut out_plan = ArenaPlan::new();
    let conf = out_plan.add(DType::F32, &[pairs]);
    let ids = out_plan.add(DType::I32, &[pairs, k]);
    let vals = out_plan.add(DType::F32, &[pairs, k]);
    let out = Arena::new(gpu, &out_plan)?;
    let t = time_on_stream(gpu, 3, 20, || {
        product_topk(gpu, &arena, &c, &u, &out, &conf, &ids, &vals, pairs, width, vocab, codebooks, 1024, 2.5, k)
    })?;
    let bytes = (2 * pairs * vocab + pairs * k * 2 + pairs) * 4;
    Ok(Row {
        name: name.into(),
        size: format!("{pairs} pairs, top {k} of {vocab}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: bytes as f64 / (t.us * 1e-6) / (gpu.info().mem_bandwidth_gbs * 1e9) * 100.0,
    })
}
