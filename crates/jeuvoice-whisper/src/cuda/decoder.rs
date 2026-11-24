//! The Whisper decoder on Rust kernels. The math follows OpenAI Whisper:
//!
//! ```text
//! x = token_embedding[ids] + positional_embedding[pos]
//! for each of the 4 blocks:
//!     x = x + out(self_attention(ln1(x)))        causal, with the KV cache
//!     x = x + out(cross_attention(ln2(x), xa))   K and V from the encoder output xa
//!     x = x + mlp2(gelu(mlp1(ln3(x))))
//! logits = ln(x) · token_embeddingᵀ               f32, as whisper.cpp
//! ```
//!
//! Q and V have a bias, K has none. The tensors are `[rows, 1280]`: the attention reads the
//! heads in that layout (`ops::attention_decode`), so there is no head split.
//!
//! The KV cache: each block keeps the K and V rows of every earlier token (`[448, 1280]`
//! each). A step computes Q, K and V for its new rows only, writes K and V into the cache
//! at their positions, and reads the whole cache. The cross-attention K and V
//! (`[1500, 1280]` each) come from the encoder output once per window (`set_cross_kv`).
//!
//! The attention of many rows runs in groups of `GROUP` rows, so the scratch of the split
//! kernel stays small (4 MB, not 29 MB for a DTW pass of 227 rows).

use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::ops::{
    ProbsOut, add_into, attention_decode, bias_add, bias_gelu, decode_split_scratch, embed, embed_one, layernorm,
};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, Tensor};

use super::weights::DecoderWeights;
use crate::WhisperError;
use crate::decode::dtw::ALIGNMENT_HEADS;
use crate::ggml::GgmlModel;

/// LayerNorm epsilon of Whisper.
const LN_EPS: f32 = 1e-5;
/// The encoder positions that the cross-attention reads.
const N_AUDIO_CTX: usize = 1500;
/// The query rows of one attention call when many rows run (the prompt, the DTW pass).
const GROUP: usize = 32;

/// The activations, the caches and the scratch of the decoder.
struct Acts {
    /// `[n_ctx, d]` each: the residual stream, the LayerNorm output, Q, the attention output
    /// and the projection output.
    x: Tensor,
    h: Tensor,
    q: Tensor,
    att: Tensor,
    proj: Tensor,
    /// `[n_ctx, d_mlp]`
    mlp: Tensor,
    /// `[n_ctx, d]` per block: the self-attention KV cache.
    k_cache: Vec<Tensor>,
    v_cache: Vec<Tensor>,
    /// `[1500, d]` per block: the cross-attention K and V of the current window.
    ck: Vec<Tensor>,
    cv: Vec<Tensor>,
    /// `[n_ctx]` I32: the token ids of a batch.
    ids: Tensor,
    /// `[1, n_vocab]` F32: the logits of the last row.
    logits: Tensor,
    /// The partial results of `DecodeAttnKind::Split` for `GROUP` rows and 1500 keys.
    scratch: Tensor,
    /// `[6, GROUP, 1500]` F32: the alignment-head softmax of one group (the DTW pass).
    probs: Tensor,
}

pub struct Decoder {
    arena: Arena,
    w: DecoderWeights,
    a: Acts,
    /// The rows in the self-attention cache.
    past: usize,
    /// True after `set_cross_kv`.
    has_cross: bool,
    weight_bytes: usize,
    activation_bytes: usize,
}

/// What a pass gives back.
enum Mode<'a> {
    /// The logits of the last row.
    Logits,
    /// The alignment-head softmax of every row, for the first `n_audio` positions, into
    /// `out` (`[6][rows][n_audio]`). The pass stops after the cross-attention of the last
    /// alignment layer.
    Align { n_audio: usize, out: &'a mut [f32] },
}

impl Decoder {
    /// Plans the weights, the caches and the activations in one arena, allocates it, and
    /// uploads the decoder weights of `model`.
    pub fn load(gpu: &Gpu, model: &GgmlModel) -> Result<Decoder, WhisperError> {
        let mut plan = ArenaPlan::new();
        let w = DecoderWeights::plan(&mut plan, model)?;
        let weight_bytes = plan.total_bytes();
        let (d, n_ctx, heads, d_mlp, n_vocab) = (w.d, w.n_ctx, w.n_head, w.d_mlp, w.n_vocab);
        if d != heads * 64 {
            return Err(WhisperError::ModelLoad(format!("the decoder width {d} is not {heads} heads of 64")));
        }
        if w.blocks.len() <= ALIGNMENT_HEADS.iter().map(|&(l, _)| l).max().unwrap_or(0) {
            return Err(WhisperError::ModelLoad(format!("the decoder has {} blocks, too few for the large-v3-turbo alignment heads", w.blocks.len())));
        }
        let f = DType::F16;
        let n_layer = w.blocks.len();
        let a = Acts {
            x: plan.add(f, &[n_ctx, d]),
            h: plan.add(f, &[n_ctx, d]),
            q: plan.add(f, &[n_ctx, d]),
            att: plan.add(f, &[n_ctx, d]),
            proj: plan.add(f, &[n_ctx, d]),
            mlp: plan.add(f, &[n_ctx, d_mlp]),
            k_cache: (0..n_layer).map(|_| plan.add(f, &[n_ctx, d])).collect(),
            v_cache: (0..n_layer).map(|_| plan.add(f, &[n_ctx, d])).collect(),
            ck: (0..n_layer).map(|_| plan.add(f, &[N_AUDIO_CTX, d])).collect(),
            cv: (0..n_layer).map(|_| plan.add(f, &[N_AUDIO_CTX, d])).collect(),
            ids: plan.add(DType::I32, &[n_ctx]),
            logits: plan.add(DType::F32, &[1, n_vocab]),
            scratch: plan.add(DType::F32, &[decode_split_scratch(heads, GROUP, N_AUDIO_CTX.max(n_ctx))]),
            probs: plan.add(DType::F32, &[ALIGNMENT_HEADS.len(), GROUP, N_AUDIO_CTX]),
        };
        let activation_bytes = plan.total_bytes() - weight_bytes;
        let arena = Arena::new(gpu, &plan)?;
        w.upload(gpu, &arena, model)?;
        Ok(Decoder { arena, w, a, past: 0, has_cross: false, weight_bytes, activation_bytes })
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// The caches, the activations and the scratch.
    pub fn activation_bytes(&self) -> usize {
        self.activation_bytes
    }

    /// Computes the cross-attention K and V of each block from the encoder output `xa`
    /// (`[1500, 1280]` F16 in `xa_arena`). Call it once per window, after the encoder.
    pub fn set_cross_kv(&mut self, gpu: &Gpu, blas: &Blas, xa_arena: &Arena, xa: &Tensor) -> Result<(), WhisperError> {
        let d = self.w.d;
        assert_eq!(xa.dtype(), DType::F16, "set_cross_kv: the encoder output must be F16");
        assert_eq!(xa.shape(), &[N_AUDIO_CTX, d], "set_cross_kv: the encoder output must be [1500, {d}]");
        let ar = &self.arena;
        let xs = xa_arena.slice(xa);
        for (i, b) in self.w.blocks.iter().enumerate() {
            // The only reads across two arenas: cuBLAS takes device pointers, so the
            // one-arena rule of the ops does not apply here.
            blas.gemm_f16(gpu, &xs, Op::N, &ar.slice(&b.ck_w), Op::T, &ar.slice(&self.a.ck[i]), N_AUDIO_CTX, d, d, 1.0, 0.0)?;
            blas.gemm_f16(gpu, &xs, Op::N, &ar.slice(&b.cv_w), Op::T, &ar.slice(&self.a.cv[i]), N_AUDIO_CTX, d, d, 1.0, 0.0)?;
            bias_add(gpu, ar, &self.a.cv[i], &b.cv_b)?;
        }
        self.has_cross = true;
        Ok(())
    }

    /// Starts a new self-attention cache, decodes `tokens` in one batch, and returns the
    /// `n_vocab` logits of the last row.
    pub fn decode_first(&mut self, gpu: &Gpu, blas: &Blas, tokens: &[u32]) -> Result<Vec<f32>, WhisperError> {
        self.check_batch(tokens.len())?;
        self.past = 0;
        let n = tokens.len();
        embed(gpu, &self.arena, tokens, &self.a.ids, &self.w.tok, &self.w.pos, 0, &self.a.x.rows(0, n))?;
        self.pass(gpu, blas, n, Mode::Logits)?;
        self.past = n;
        self.logits_of_row(gpu, blas, n - 1)
    }

    /// Decodes one more token with the cache and returns its `n_vocab` logits.
    pub fn decode_next(&mut self, gpu: &Gpu, blas: &Blas, token: u32) -> Result<Vec<f32>, WhisperError> {
        if !self.has_cross {
            return Err(WhisperError::Transcribe("the decoder has no encoder output: call set_cross_kv first".into()));
        }
        if self.past >= self.w.n_ctx {
            return Err(WhisperError::Transcribe(format!("the decoder context is full ({} tokens)", self.w.n_ctx)));
        }
        let p = self.past;
        embed_one(gpu, &self.arena, token, &self.w.tok, &self.w.pos, p, &self.a.x.rows(0, 1))?;
        self.pass(gpu, blas, 1, Mode::Logits)?;
        self.past = p + 1;
        self.logits_of_row(gpu, blas, 0)
    }

    /// The DTW pass: starts a new cache, decodes `tokens` in one batch, and returns the
    /// cross-attention softmax of the `ALIGNMENT_HEADS` for the first `n_audio` encoder
    /// positions, as `[6][tokens][n_audio]`.
    pub fn alignment(&mut self, gpu: &Gpu, blas: &Blas, tokens: &[u32], n_audio: usize) -> Result<Vec<f32>, WhisperError> {
        self.check_batch(tokens.len())?;
        assert!(n_audio <= N_AUDIO_CTX, "alignment: {n_audio} audio positions, more than {N_AUDIO_CTX}");
        let n = tokens.len();
        self.past = 0;
        let mut out = vec![0.0f32; ALIGNMENT_HEADS.len() * n * n_audio];
        embed(gpu, &self.arena, tokens, &self.a.ids, &self.w.tok, &self.w.pos, 0, &self.a.x.rows(0, n))?;
        self.pass(gpu, blas, n, Mode::Align { n_audio, out: &mut out })?;
        self.past = n;
        Ok(out)
    }

    fn check_batch(&self, n: usize) -> Result<(), WhisperError> {
        if !self.has_cross {
            return Err(WhisperError::Transcribe("the decoder has no encoder output: call set_cross_kv first".into()));
        }
        if n == 0 || n > self.w.n_ctx {
            return Err(WhisperError::Transcribe(format!("the decoder takes 1 to {} tokens at a time, not {n}", self.w.n_ctx)));
        }
        Ok(())
    }

    /// The final LayerNorm of row `row` of `x` and its f32 logits, downloaded.
    fn logits_of_row(&self, gpu: &Gpu, blas: &Blas, row: usize) -> Result<Vec<f32>, WhisperError> {
        let (ar, w, a) = (&self.arena, &self.w, &self.a);
        let (x, h) = (a.x.rows(row, 1), a.h.rows(0, 1));
        layernorm(gpu, ar, &x, &w.ln_w, &w.ln_b, &h, LN_EPS)?;
        blas.gemm_f16_f32out(gpu, &ar.slice(&h), Op::N, &ar.slice(&w.tok), Op::T, &ar.slice(&a.logits), 1, w.n_vocab, w.d, 1.0, 0.0)?;
        Ok(ar.download::<f32>(gpu, &a.logits)?)
    }

    /// Runs the blocks on the `n` new rows of `x` at position `self.past`.
    fn pass(&self, gpu: &Gpu, blas: &Blas, n: usize, mut mode: Mode) -> Result<(), WhisperError> {
        let (ar, w, a) = (&self.arena, &self.w, &self.a);
        let (d, d_mlp, p) = (w.d, w.d_mlp, self.past);
        let linear = |x: &Tensor, wt: &Tensor, y: &Tensor, n_in: usize, n_out: usize| {
            blas.gemm_f16(gpu, &ar.slice(x), Op::N, &ar.slice(wt), Op::T, &ar.slice(y), n, n_out, n_in, 1.0, 0.0)
        };
        let (x, h, q, att, proj, mlp) = (a.x.rows(0, n), a.h.rows(0, n), a.q.rows(0, n), a.att.rows(0, n), a.proj.rows(0, n), a.mlp.rows(0, n));
        let last_align_layer = ALIGNMENT_HEADS.iter().map(|&(l, _)| l).max().unwrap_or(0);

        for (li, b) in w.blocks.iter().enumerate() {
            // Self-attention: the new K and V rows go into the cache at their positions.
            layernorm(gpu, ar, &x, &b.attn_ln_w, &b.attn_ln_b, &h, LN_EPS)?;
            linear(&h, &b.q_w, &q, d, d)?;
            bias_add(gpu, ar, &q, &b.q_b)?;
            let (kc, vc) = (a.k_cache[li].rows(p, n), a.v_cache[li].rows(p, n));
            linear(&h, &b.k_w, &kc, d, d)?;
            linear(&h, &b.v_w, &vc, d, d)?;
            bias_add(gpu, ar, &vc, &b.v_b)?;
            self.attend(gpu, &q, &a.k_cache[li], &a.v_cache[li], &att, p + n, true, None)?;
            linear(&att, &b.o_w, &proj, d, d)?;
            bias_add(gpu, ar, &proj, &b.o_b)?;
            add_into(gpu, ar, &x, &proj)?;

            // Cross-attention on the encoder output of this window.
            layernorm(gpu, ar, &x, &b.cross_ln_w, &b.cross_ln_b, &h, LN_EPS)?;
            linear(&h, &b.cq_w, &q, d, d)?;
            bias_add(gpu, ar, &q, &b.cq_b)?;
            let heads_here: Vec<(usize, usize)> = ALIGNMENT_HEADS.iter().enumerate().filter(|(_, ah)| ah.0 == li).map(|(slot, ah)| (slot, ah.1)).collect();
            let probs = match &mut mode {
                Mode::Align { n_audio, out } if !heads_here.is_empty() => Some((heads_here.as_slice(), *n_audio, &mut **out)),
                _ => None,
            };
            self.attend(gpu, &q, &a.ck[li], &a.cv[li], &att, N_AUDIO_CTX, false, probs)?;
            linear(&att, &b.co_w, &proj, d, d)?;
            bias_add(gpu, ar, &proj, &b.co_b)?;
            add_into(gpu, ar, &x, &proj)?;

            if matches!(mode, Mode::Align { .. }) && li == last_align_layer {
                // The DTW pass needs nothing after the last alignment head.
                return Ok(());
            }

            // MLP.
            layernorm(gpu, ar, &x, &b.mlp_ln_w, &b.mlp_ln_b, &h, LN_EPS)?;
            linear(&h, &b.mlp1_w, &mlp, d, d_mlp)?;
            bias_gelu(gpu, ar, &mlp, &b.mlp1_b)?;
            linear(&mlp, &b.mlp2_w, &proj, d_mlp, d)?;
            bias_add(gpu, ar, &proj, &b.mlp2_b)?;
            add_into(gpu, ar, &x, &proj)?;
        }
        Ok(())
    }

    /// Attention of the `q` rows against the first `nk` rows of `k` and `v`, in groups of
    /// `GROUP` rows. With `causal`, the rows are the last `q.rows` of the `nk` positions.
    /// `probs` names the alignment heads of this layer as `(slot, head)`, the audio
    /// positions to keep and the output `[6][rows][n_audio]`.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        gpu: &Gpu,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &Tensor,
        nk: usize,
        causal: bool,
        probs: Option<(&[(usize, usize)], usize, &mut [f32])>,
    ) -> Result<(), WhisperError> {
        let (ar, a) = (&self.arena, &self.a);
        let heads = self.w.n_head;
        let n = q.shape()[0];
        let mut probs = probs;
        let mut g0 = 0;
        while g0 < n {
            let gn = GROUP.min(n - g0);
            // A causal group sees the keys up to its own last row.
            let keys = if causal { nk - n + g0 + gn } else { nk };
            let (qg, og) = (q.rows(g0, gn), out.rows(g0, gn));
            let (kg, vg) = (k.rows(0, keys), v.rows(0, keys));
            let pt = a.probs.prefix_as(&[ALIGNMENT_HEADS.len(), gn, keys]);
            let po = probs.as_ref().map(|(hs, _, _)| ProbsOut {
                probs: &pt,
                heads_mask: hs.iter().fold(0u32, |m, &(_, hd)| m | (1 << hd)),
                slot0: hs[0].0 as u32,
            });
            attention_decode(gpu, ar, &qg, &kg, &vg, &og, heads, causal, po, Some(&a.scratch))?;
            if let Some((hs, n_audio, dst)) = probs.as_mut() {
                // The download waits for the attention. Copy the first n_audio positions of
                // the slots of this layer.
                let host = ar.download::<f32>(gpu, &pt)?;
                for &(slot, _) in hs.iter() {
                    for r in 0..gn {
                        let src = &host[(slot * gn + r) * keys..][..*n_audio];
                        dst[(slot * n + g0 + r) * *n_audio..][..*n_audio].copy_from_slice(src);
                    }
                }
            }
            g0 += gn;
        }
        Ok(())
    }
}
