//! The masked iterative decode of OmniVoice (plan Phase 2, Tasks 9 and 10).
//!
//! Every step runs one forward pass over the whole sequence: the conditioned row and the
//! unconditioned row. Classifier-free guidance mixes their logits, the model picks a token
//! for every masked position, and the `k` most confident positions are written. The engine
//! does the same in `_generate_iterative` (`omnivoice.py:1275`).


/// How the positions are chosen and the tokens sampled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// No noise: the plain confidence order and the argmax token. This is the mode the
    /// fixtures record, and the only one a token comparison can reach.
    Deterministic,
    /// The app's mode: Gumbel noise on the position score and on the token choice.
    Product,
}

/// The decode settings of the engine (`OmniVoiceGenerationConfig`).
#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub mode: Mode,
    /// The classifier-free guidance scale; 0 turns the guidance off.
    pub guidance: f32,
    pub num_step: usize,
    pub t_shift: f32,
    /// A score penalty per codebook, so the first codebook unmasks first.
    pub layer_penalty: f32,
    /// 0 for the deterministic mode, else the engine's 5.0.
    pub position_temperature: f32,
    /// 0 means greedy; the engine's product mode uses a top-k filter with a temperature.
    pub class_temperature: f32,
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            mode: Mode::Deterministic,
            guidance: 2.5,
            num_step: 32,
            t_shift: 0.1,
            layer_penalty: 5.0,
            position_temperature: 0.0,
            class_temperature: 0.0,
            seed: 42,
        }
    }
}

/// The times of the steps, as `_get_time_steps`: a linspace, then the shift curve. The engine
/// computes both in f32, so the numbers are f32 before they reach the schedule.
pub fn time_steps(num_step: usize, t_shift: f32) -> Vec<f32> {
    (0..=num_step)
        .map(|i| {
            let t = i as f32 / num_step as f32;
            t_shift * t / (1.0 + (t_shift - 1.0) * t)
        })
        .collect()
}

/// How many `(codebook, position)` pairs each step unmasks. The last step takes the rest.
///
/// Every pair of the target is a candidate, so the total is `codebooks * target_len`. The
/// steps do not unmask evenly: the shift curve gives the early steps fewer pairs, because a
/// wrong token early costs more.
pub fn schedule(target_len: usize, codebooks: usize, params: &Params) -> Vec<usize> {
    let total = target_len * codebooks;
    let times = time_steps(params.num_step, params.t_shift);
    let mut remaining = total;
    let mut out = Vec::with_capacity(params.num_step);
    for step in 0..params.num_step {
        let count = if step == params.num_step - 1 {
            remaining
        } else {
            let span = times[step + 1] as f64 - times[step] as f64;
            (total as f64 * span).ceil() as usize
        };
        let count = count.min(remaining);
        out.push(count);
        remaining -= count;
    }
    out
}

/// `[codebooks, len]` tokens: one row per codebook, the frame in place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokens {
    pub codebooks: usize,
    pub len: usize,
    pub ids: Vec<i32>,
}

/// The log-softmax peak of one row and its index, for the guided logits
/// `(1 + guidance)·c - guidance·u` with the mask id set to minus infinity.
///
/// The engine computes `log_softmax(c) + g·(log_softmax(c) - log_softmax(u))` and then one
/// more `log_softmax`. The two inner normalisers are the same constant for every value of the
/// row, and the outer `log_softmax` removes it, so the result is `log_softmax((1+g)c - g·u)`
/// exactly, without the two extra exponentials.
fn guided_peak(c: &[f32], u: &[f32], guidance: f32, mask_id: usize, y: &mut [f32]) -> (f32, usize) {
    // The mask id is part of the normaliser: the engine masks it *after* the log_softmax, so
    // its value moves every row's confidence, and with it the order of the positions.
    let mut top = f32::NEG_INFINITY;
    let mut best = f32::NEG_INFINITY;
    let mut best_at = 0usize;
    for v in 0..c.len() {
        let value = (1.0 + guidance) * c[v] - guidance * u[v];
        y[v] = value;
        if value > top {
            top = value;
        }
        if v != mask_id && value > best {
            best = value;
            best_at = v;
        }
    }
    let mut sum = 0.0f32;
    for &value in y.iter() {
        sum += (value - top).exp();
    }
    (best - (sum.ln() + top), best_at)
}

#[cfg(feature = "cuda-kernels")]
pub use gpu::decode;

#[cfg(feature = "cuda-kernels")]
mod gpu {
    use super::*;
    use crate::{Result, TtsError};
    use jeuvoice_gpu::blas::Blas;
    use jeuvoice_gpu::Gpu;

    use crate::config::OmniVoiceConfig;
    use crate::forward::forward;
    use crate::text::Layout;
    use crate::weights::OmniVoiceWeights;

    /// Runs the masked decode. `layout` is the step-0 sequence of the conditioned row: the
    /// style, the text and the prompt, with the target frames masked.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        w: &OmniVoiceWeights,
        cfg: &OmniVoiceConfig,
        gpu: &Gpu,
        blas: &Blas,
        layout: &Layout,
        prompt_len: usize,
        target_len: usize,
        params: &Params,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<Tokens> {
        if params.mode == Mode::Product {
            return Err(TtsError::Render(
                "the product mode needs the Gumbel noise of Task 10; the deterministic mode is ready".into(),
            ));
        }
        if params.num_step == 0 {
            return Err(TtsError::Render("the decode needs at least one step".into()));
        }
        let codebooks = cfg.num_audio_codebook;
        let vocab = cfg.audio_vocab_size;
        let mask_id = cfg.audio_mask_id as usize;
        let seq = layout.len;
        let prefix = seq
            .checked_sub(prompt_len + target_len)
            .ok_or_else(|| TtsError::Render("the layout is shorter than the prompt and the target".into()))?;
        if layout.ids.len() != codebooks * seq || layout.audio_mask.len() != seq {
            return Err(TtsError::Render("the layout does not match the codebooks".into()));
        }
        if seq > crate::weights::T_MAX {
            return Err(TtsError::Render(format!(
                "the sequence of {seq} positions is longer than the plan ({} positions)",
                crate::weights::T_MAX
            )));
        }

        let mut tokens = vec![mask_id as i32; codebooks * target_len];
        // The two rows of a step: the conditioned row is the whole layout, the unconditioned
        // row is the target frames alone. Both have `seq` positions in the padded batch.
        let mut ids = vec![mask_id as i32; 2 * codebooks * seq];
        let mut mask = vec![0i32; 2 * seq];
        mask[..seq].copy_from_slice(&layout.audio_mask);
        mask[seq..seq + target_len].fill(1);

        let counts = schedule(target_len, codebooks, params);
        let width = w.act.logits.shape()[1];
        let mut y = vec![0.0f32; vocab];
        let mut scores = vec![f32::NEG_INFINITY; codebooks * target_len];
        let mut picks = vec![0i32; codebooks * target_len];
        let target = prefix + prompt_len;

        for step in 0..params.num_step {
            // 1. The ids of the step: the layout with the tokens written in, and the target
            //    frames of the unconditioned row.
            ids[..codebooks * seq].copy_from_slice(&layout.ids);
            for c in 0..codebooks {
                ids[c * seq + target..c * seq + target + target_len]
                    .copy_from_slice(&tokens[c * target_len..(c + 1) * target_len]);
                let uncond = (codebooks + c) * seq;
                ids[uncond..uncond + target_len].copy_from_slice(&tokens[c * target_len..(c + 1) * target_len]);
            }
            let valid = [seq as i32, target_len as i32];
            forward(w, cfg, gpu, blas, &ids, &mask, 2, seq, Some(&valid), &mut |_, _, _| {})?;

            // 2. The logits of the target frames of both rows, to the host.
            let cond = download_rows(w, gpu, target, target_len)?;
            let uncond = download_rows(w, gpu, seq, target_len)?;

            // 3. The guided peak and its token, per codebook and frame.
            for c in 0..codebooks {
                for p in 0..target_len {
                    let at = p * width + c * vocab;
                    let (confidence, token) = guided_peak(
                        &cond[at..at + vocab],
                        &uncond[at..at + vocab],
                        params.guidance,
                        mask_id,
                        &mut y,
                    );
                    let index = c * target_len + p;
                    scores[index] = confidence - c as f32 * params.layer_penalty;
                    picks[index] = token as i32;
                }
            }

            // 4. The `k` most confident positions among the still masked ones.
            let mut candidates: Vec<usize> = (0..codebooks * target_len).filter(|&i| tokens[i] == mask_id as i32).collect();
            let k = counts[step].min(candidates.len());
            if k > 0 {
                candidates.select_nth_unstable_by(k - 1, |&a, &b| {
                    scores[b].total_cmp(&scores[a]).then(a.cmp(&b))
                });
                for &index in &candidates[..k] {
                    tokens[index] = picks[index];
                }
            }
            progress(step, params.num_step);
        }

        if tokens.iter().any(|&t| t == mask_id as i32) {
            return Err(TtsError::Render("the decode left masked frames: the schedule did not fill them".into()));
        }
        Ok(Tokens { codebooks, len: target_len, ids: tokens })
    }

    /// The logits of `rows` positions, `[rows, audio_head_rows]` f32, from `start`.
    fn download_rows(w: &OmniVoiceWeights, gpu: &Gpu, start: usize, rows: usize) -> Result<Vec<f32>> {
        let view = w.act.logits.rows_unaligned(start, rows);
        Ok(w.arena.download_unaligned::<f32>(gpu, &view)?)
    }
}