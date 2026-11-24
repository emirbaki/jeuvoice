//! The Whisper encoder on Rust kernels: one 30 s mel window `[128, 3000]` in, the encoder
//! output `[1500, 1280]` out. The math follows OpenAI Whisper:
//!
//! ```text
//! x = gelu(conv1(mel) + b1)                       [3000, 1280]
//! x = gelu(conv2(x) + b2)   (stride 2)            [1500, 1280]
//! x = x + positional_embedding
//! for each of the 32 blocks:
//!     x = x + out(attention(ln1(x)))              Q and V have a bias, K has none
//!     x = x + mlp2(gelu(mlp1(ln2(x))))
//! y = ln_post(x)
//! ```
//!
//! All activations are time first (`[time, channels]`), so each bias runs along the last
//! dimension. Attention uses `AttnKind::FlashTc` (tensor-core flash attention, lesson 5): on
//! the RTX 3050 Ti it takes 0.90 ms per layer, `AttnKind::Naive` (cuBLAS) 2.49 ms.

use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::ops::{
    AttnKind, Layout, add_into, add_rows, attention, bias_add, bias_gelu, conv1d_with_layout, layernorm, merge_heads,
    split_heads,
};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, Tensor};

use super::weights::EncoderWeights;
use crate::WhisperError;
use crate::ggml::GgmlModel;

/// LayerNorm epsilon of Whisper.
const LN_EPS: f32 = 1e-5;
/// Mel frames in one 30 s window.
const N_FRAMES: usize = 3000;

/// The activation tensors of one window.
struct Acts {
    /// `[n_mels, 3000]`, channels first: the input.
    mel: Tensor,
    /// `[n_mels · 3, 3000]`: im2col for conv1.
    col1: Tensor,
    /// `[3000, d]`: conv1 output.
    c1: Tensor,
    /// `[d · 3, n_ctx]`: im2col for conv2.
    col2: Tensor,
    /// `[n_ctx, d]`: the residual stream.
    x: Tensor,
    /// `[n_ctx, d]`: the LayerNorm output.
    h: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    /// `[heads, n_ctx, 64]` each.
    qh: Tensor,
    kh: Tensor,
    vh: Tensor,
    oh: Tensor,
    /// `[heads, n_ctx, n_ctx]`: the attention scores. Only `AttnKind::Naive` has it (86 MB).
    scores: Option<Tensor>,
    /// `[n_ctx, d]`: the merged heads.
    att: Tensor,
    /// `[n_ctx, d]`: the output of the attention projection and of mlp2.
    proj: Tensor,
    /// `[n_ctx, d_mlp]`
    mlp: Tensor,
    /// `[n_ctx, d]`: the encoder output.
    out: Tensor,
}

pub struct Encoder {
    arena: Arena,
    w: EncoderWeights,
    a: Acts,
    weight_bytes: usize,
    activation_bytes: usize,
    attn: AttnKind,
}

impl Encoder {
    /// Plans the weights and the activations in one arena, allocates it, and uploads the
    /// encoder weights of `model`. The attention is `AttnKind::FlashTc`.
    pub fn load(gpu: &Gpu, model: &GgmlModel) -> Result<Encoder, WhisperError> {
        Self::load_with_attention(gpu, model, AttnKind::FlashTc)
    }

    /// As `load`, with a chosen attention kernel. Only `AttnKind::Naive` plans the
    /// `[heads, 1500, 1500]` score tensor (86 MB).
    pub fn load_with_attention(gpu: &Gpu, model: &GgmlModel, attn: AttnKind) -> Result<Encoder, WhisperError> {
        model.check_gpu_encoder()?;
        let mut plan = ArenaPlan::new();
        let w = EncoderWeights::plan(&mut plan, model)?;
        let weight_bytes = plan.total_bytes();
        let (d, n_ctx, heads, d_mlp, n_mels) = (w.d, w.n_ctx, w.n_head, w.d_mlp, w.n_mels);
        if d != heads * 64 {
            return Err(WhisperError::ModelLoad(format!("the encoder width {d} is not {heads} heads of 64")));
        }
        let f = DType::F16;
        let a = Acts {
            mel: plan.add(f, &[n_mels, N_FRAMES]),
            col1: plan.add(f, &[n_mels * 3, N_FRAMES]),
            c1: plan.add(f, &[N_FRAMES, d]),
            col2: plan.add(f, &[d * 3, n_ctx]),
            x: plan.add(f, &[n_ctx, d]),
            h: plan.add(f, &[n_ctx, d]),
            q: plan.add(f, &[n_ctx, d]),
            k: plan.add(f, &[n_ctx, d]),
            v: plan.add(f, &[n_ctx, d]),
            qh: plan.add(f, &[heads, n_ctx, 64]),
            kh: plan.add(f, &[heads, n_ctx, 64]),
            vh: plan.add(f, &[heads, n_ctx, 64]),
            oh: plan.add(f, &[heads, n_ctx, 64]),
            scores: (attn == AttnKind::Naive).then(|| plan.add(f, &[heads, n_ctx, n_ctx])),
            att: plan.add(f, &[n_ctx, d]),
            proj: plan.add(f, &[n_ctx, d]),
            mlp: plan.add(f, &[n_ctx, d_mlp]),
            out: plan.add(f, &[n_ctx, d]),
        };
        let activation_bytes = plan.total_bytes() - weight_bytes;
        let arena = Arena::new(gpu, &plan)?;
        w.upload(gpu, &arena, model)?;
        Ok(Encoder { arena, w, a, weight_bytes, activation_bytes, attn })
    }

    /// The arena and the `[n_mels, 3000]` `F16` mel input tensor. `GpuMel::window_into`
    /// writes into it.
    pub fn input(&self) -> (&Arena, &Tensor) {
        (&self.arena, &self.a.mel)
    }

    /// Changes the attention kernel. For tests and lesson 5.
    ///
    /// # Panics
    /// If `kind` is `AttnKind::Naive` and the encoder has no score tensor: load it with
    /// `load_with_attention(.., AttnKind::Naive)` for that.
    pub fn set_attention(&mut self, kind: AttnKind) {
        assert!(kind != AttnKind::Naive || self.a.scores.is_some(), "set_attention: Naive needs an encoder loaded with AttnKind::Naive");
        self.attn = kind;
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    pub fn activation_bytes(&self) -> usize {
        self.activation_bytes
    }

    /// Encodes the current input. Returns the `[1500, 1280]` `F16` output in `arena()`.
    pub fn run(&mut self, gpu: &Gpu, blas: &Blas) -> Result<&Tensor, WhisperError> {
        self.run_with_taps(gpu, blas, &mut |_, _, _| {})
    }

    /// The arena of the encoder (weights, activations and the output).
    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// As `run`, and calls `taps` after each step with the names `conv1`, `conv2`, `embed`,
    /// `block_00` … `block_31` and `ln_post`. The tap sees the stream in order, so a download
    /// inside it reads the finished tensor.
    pub fn run_with_taps(&mut self, gpu: &Gpu, blas: &Blas, taps: &mut dyn FnMut(&str, &Arena, &Tensor)) -> Result<&Tensor, WhisperError> {
        let (ar, w, a, attn) = (&self.arena, &self.w, &self.a, self.attn);
        let (d, n_ctx, d_mlp) = (w.d, w.n_ctx, w.d_mlp);
        // y [rows, n_out] = x [rows, n_in] · Wᵀ, with W stored [n_out, n_in] (a Linear layer).
        let linear = |x: &Tensor, wt: &Tensor, y: &Tensor, rows: usize, n_in: usize, n_out: usize| {
            blas.gemm_f16(gpu, &ar.slice(x), Op::N, &ar.slice(wt), Op::T, &ar.slice(y), rows, n_out, n_in, 1.0, 0.0)
        };

        // The stem: two convolutions with GELU, then the positional embedding.
        conv1d_with_layout(gpu, ar, blas, &a.mel, Layout::ChannelsFirst, &w.conv1_w, &a.c1, Layout::TimeFirst, 1, &a.col1)?;
        bias_gelu(gpu, ar, &a.c1, &w.conv1_b)?;
        taps("conv1", ar, &a.c1);
        conv1d_with_layout(gpu, ar, blas, &a.c1, Layout::TimeFirst, &w.conv2_w, &a.x, Layout::TimeFirst, 2, &a.col2)?;
        bias_gelu(gpu, ar, &a.x, &w.conv2_b)?;
        taps("conv2", ar, &a.x);
        add_rows(gpu, ar, &a.x, &w.pos)?;
        taps("embed", ar, &a.x);

        for (i, b) in w.blocks.iter().enumerate() {
            // Self-attention.
            layernorm(gpu, ar, &a.x, &b.attn_ln_w, &b.attn_ln_b, &a.h, LN_EPS)?;
            linear(&a.h, &b.q_w, &a.q, n_ctx, d, d)?;
            bias_add(gpu, ar, &a.q, &b.q_b)?;
            linear(&a.h, &b.k_w, &a.k, n_ctx, d, d)?;
            linear(&a.h, &b.v_w, &a.v, n_ctx, d, d)?;
            bias_add(gpu, ar, &a.v, &b.v_b)?;
            split_heads(gpu, ar, &a.q, &a.qh)?;
            split_heads(gpu, ar, &a.k, &a.kh)?;
            split_heads(gpu, ar, &a.v, &a.vh)?;
            attention(gpu, ar, blas, &a.qh, &a.kh, &a.vh, &a.oh, a.scores.as_ref(), attn)?;
            merge_heads(gpu, ar, &a.oh, &a.att)?;
            linear(&a.att, &b.o_w, &a.proj, n_ctx, d, d)?;
            bias_add(gpu, ar, &a.proj, &b.o_b)?;
            add_into(gpu, ar, &a.x, &a.proj)?;
            // MLP.
            layernorm(gpu, ar, &a.x, &b.mlp_ln_w, &b.mlp_ln_b, &a.h, LN_EPS)?;
            linear(&a.h, &b.mlp1_w, &a.mlp, n_ctx, d, d_mlp)?;
            bias_gelu(gpu, ar, &a.mlp, &b.mlp1_b)?;
            linear(&a.mlp, &b.mlp2_w, &a.proj, n_ctx, d_mlp, d)?;
            bias_add(gpu, ar, &a.proj, &b.mlp2_b)?;
            add_into(gpu, ar, &a.x, &a.proj)?;
            taps(&format!("block_{i:02}"), ar, &a.x);
        }

        layernorm(gpu, ar, &a.x, &w.ln_post_w, &w.ln_post_b, &a.out, LN_EPS)?;
        taps("ln_post", ar, &a.out);
        Ok(&self.a.out)
    }
}
