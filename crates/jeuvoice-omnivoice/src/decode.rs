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


/// A small seeded generator for the product mode: xorshift64* and the Gumbel transform. The
/// engine draws from torch's global Philox stream, which cannot be reproduced; the seed only
/// makes our product mode repeatable.
struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    /// A value in `(0, 1)`: 24 random bits, never exactly zero.
    fn uniform(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32 + 0.5) / (1u32 << 24) as f32
    }

    /// `-ln(-ln(u))` with the engine's 1e-10 clamp.
    fn gumbel(&mut self) -> f32 {
        let u = self.uniform().clamp(1e-10, 1.0 - 1e-10);
        -(-u.ln()).ln()
    }
}

fn sample_row_inner<F: FnMut(usize) -> f32>(
    c: &[f32],
    u: &[f32],
    guidance: f32,
    mask_id: usize,
    temperature: f32,
    y: &mut [f32],
    mut noise: F,
) -> (f32, i32) {
    let mut top = f32::NEG_INFINITY;
    for (v, slot) in y.iter_mut().enumerate() {
        *slot = (1.0 + guidance) * c[v] - guidance * u[v];
        if *slot > top {
            top = *slot;
        }
    }
    let mut sum = 0.0f32;
    for &value in y.iter() {
        sum += (value - top).exp();
    }
    let lse = sum.ln() + top;
    let mut best = f32::NEG_INFINITY;
    for (v, &value) in y.iter().enumerate() {
        if v != mask_id && value > best {
            best = value;
        }
    }
    let confidence = best - lse;
    if temperature <= 0.0 {
        let at = y.iter().enumerate().filter(|(v, _)| *v != mask_id).max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        return (confidence, at as i32);
    }
    // The engine keeps the top 10% of the 1025 tokens, then samples with the Gumbel trick.
    let k = ((y.len() as f64 * 0.1).ceil() as usize).max(1);
    let mut order: Vec<usize> = (0..y.len()).filter(|&v| v != mask_id).collect();
    order.select_nth_unstable_by(k - 1, |&a, &b| y[b].total_cmp(&y[a]));
    let keep = k.min(order.len());
    order[..keep].sort_unstable_by(|&a, &b| y[b].total_cmp(&y[a]).then(a.cmp(&b)));
    let mut best = f32::NEG_INFINITY;
    let mut at = 0i32;
    for (i, &v) in order[..keep].iter().enumerate() {
        let value = y[v] / temperature + noise(i);
        if value > best {
            best = value;
            at = v as i32;
        }
    }
    (confidence, at)
}

/// One row of the product mode: the token from the top-k filtered distribution with Gumbel
/// noise, and the confidence of the unfiltered distribution for the position score.
fn sample_row(c: &[f32], u: &[f32], guidance: f32, mask_id: usize, temperature: f32, y: &mut [f32], rng: &mut Rng) -> (f32, i32) {
    sample_row_inner(c, u, guidance, mask_id, temperature, y, |_| rng.gumbel())
}

/// `sample_row` with a fixed noise vector. Tests use it to check the Gumbel formula without the
/// engine's unreproducible stream. Noise has one value per kept token.
#[doc(hidden)]
pub fn sample_row_with_noise(c: &[f32], u: &[f32], guidance: f32, mask_id: usize, temperature: f32, y: &mut [f32], noise: &[f32]) -> (f32, i32) {
    sample_row_inner(c, u, guidance, mask_id, temperature, y, |i| noise[i])
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
        let mut rng = Rng(params.seed | 1);
        let mut scores = vec![f32::NEG_INFINITY; codebooks * target_len];
        let mut picks = vec![0i32; codebooks * target_len];
        let target = prefix + prompt_len;

        for step in 0..params.num_step {
            // A step with nothing to unmask needs no pass: the schedule is monotone, so the
            // only work left for it is the progress report.
            if counts[step] == 0 {
                progress(step, params.num_step);
                continue;
            }
            // 1. The conditioned row: the layout with the tokens written in. One row of `seq`.
            ids[..codebooks * seq].copy_from_slice(&layout.ids);
            for c in 0..codebooks {
                ids[c * seq + target..c * seq + target + target_len]
                    .copy_from_slice(&tokens[c * target_len..(c + 1) * target_len]);
            }
            mask[..seq].copy_from_slice(&layout.audio_mask);
            forward(w, cfg, gpu, blas, &ids[..codebooks * seq], &mask[..seq], 1, seq, None, None, &mut |_, _, _| {})?;

            // 2. The unconditioned row: the target frames alone, no padding to `seq`. It writes
            //    rows 0..target_len, so the conditioned target rows at `target..` stay intact.
            for c in 0..codebooks {
                ids[c * target_len..(c + 1) * target_len]
                    .copy_from_slice(&tokens[c * target_len..(c + 1) * target_len]);
            }
            mask[..target_len].fill(1);
            forward(w, cfg, gpu, blas, &ids[..codebooks * target_len], &mask[..target_len], 1, target_len, None, None, &mut |_, _, _| {})?;

            if params.mode == Mode::Product && params.class_temperature > 0.0 {
                // The top-k token sampling needs the full distribution, so this branch (which
                // the app never takes at variety 0) reads the planes and samples on the host.
                let cond_rows = download_rows(w, gpu, target, target_len)?;
                let uncond_rows = download_rows(w, gpu, 0, target_len)?;
                let (cond, uncond) = (&cond_rows[..], &uncond_rows[..]);
                for c in 0..codebooks {
                    for p in 0..target_len {
                        let index = c * target_len + p;
                        if tokens[index] != mask_id as i32 {
                            continue;
                        }
                        let at = p * width + c * vocab;
                        let (confidence, token) = sample_row(
                            &cond[at..at + vocab],
                            &uncond[at..at + vocab],
                            params.guidance,
                            mask_id,
                            params.class_temperature,
                            &mut y,
                            &mut rng,
                        );
                        scores[index] = confidence - c as f32 * params.layer_penalty;
                        picks[index] = token;
                    }
                }
            }
            // The guided peak of the kernel serves the deterministic mode and the variety-0
            // product mode: the argmax token and the unfiltered peak are the same.
            if params.mode == Mode::Deterministic || params.class_temperature <= 0.0 {
            // 2. The guided peak on the GPU: one block per (position, codebook) pair. The two
            //    logit planes stay on the device, so a step moves 13.4 KiB to the host.
            let pairs = codebooks * target_len;
            let c_plane = w.act.logits.rows_unaligned(target, target_len);
            let u_plane = w.act.logits.rows_unaligned(0, target_len);
            let conf = w.act.conf.prefix_as(&[pairs]);
            let pick = w.act.pick.prefix_as(&[pairs]);
            jeuvoice_gpu::ops::guide_peak(
                gpu,
                &w.arena,
                &c_plane,
                &u_plane,
                &conf,
                &pick,
                pairs,
                width,
                vocab,
                codebooks,
                mask_id,
                params.guidance,
            )?;
            let conf_values: Vec<f32> = w.arena.download::<f32>(gpu, &conf)?;
            let pick_values: Vec<i32> = w.arena.download::<i32>(gpu, &pick)?;
            for c in 0..codebooks {
                for p in 0..target_len {
                    let pair = p * codebooks + c;
                    let index = c * target_len + p;
                    scores[index] = conf_values[pair] - c as f32 * params.layer_penalty;
                    picks[index] = pick_values[pair];
                }
            }

            }
            // The position noise of the product mode, drawn once per masked pair in
            // codebook-major order, so one seed is repeatable. With temperature 0 the score is
            // untouched, and the product mode gives exactly the deterministic tokens.
            if params.mode == Mode::Product && params.position_temperature > 0.0 {
                for index in 0..codebooks * target_len {
                    if tokens[index] == mask_id as i32 {
                        scores[index] = scores[index] / params.position_temperature + rng.gumbel();
                    }
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
#[cfg(test)]
mod tests {
    use super::*;

    /// The engine's own formula, in f64: three log_softmaxes, the mask after the last one.
    fn engine_guide(c: &[f64], u: &[f64], g: f64, mask: usize) -> (f64, usize) {
        fn log_softmax(x: &[f64]) -> Vec<f64> {
            let top = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = x.iter().map(|v| (v - top).exp()).sum();
            let lse = top + sum.ln();
            x.iter().map(|v| v - lse).collect()
        }
        let lc = log_softmax(c);
        let lu = log_softmax(u);
        let z: Vec<f64> = lc.iter().zip(&lu).map(|(c, u)| c + g * (c - u)).collect();
        let mut lp = log_softmax(&z);
        lp[mask] = f64::NEG_INFINITY;
        let (at, &value) = lp.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        (value, at)
    }

    #[test]
    fn the_guidance_matches_the_engine_formula() {
        let cases: [(Vec<f64>, Vec<f64>, f64, usize); 4] = [
            (vec![1.0, 2.0, 3.0, 0.5], vec![0.0, 1.0, 2.0, 0.0], 0.0, 3),
            (vec![1.0, 2.0, 3.0, 0.5], vec![0.0, 1.0, 2.0, 0.0], 2.5, 3),
            // The mask id is the maximum before it is masked.
            (vec![-5.0, 20.0, 1.0, 2.0], vec![0.0, 19.0, 1.0, 1.0], 2.5, 1),
            // Large logits: the normaliser must be stable.
            (vec![1e4, 1e4 - 3.0, -1e4, 0.0], vec![1e4 - 1.0, 1e4 - 4.0, 0.0, -1e4], 2.5, 2),
        ];
        for (c, u, g, mask) in cases {
            let (cf, uf): (Vec<f32>, Vec<f32>) = (c.iter().map(|&v| v as f32).collect(), u.iter().map(|&v| v as f32).collect());
            let mut y = vec![0.0f32; c.len()];
            let (confidence, pick) = guided_peak(&cf, &uf, g as f32, mask, &mut y);
            let (want, want_pick) = engine_guide(&c, &u, g, mask);
            assert_eq!(pick, want_pick, "g = {g}, mask = {mask}");
            let tolerance = 1e-3 * want.abs().max(1.0);
            assert!((confidence as f64 - want).abs() <= tolerance, "g = {g}: {confidence} against {want}");
        }
    }

    #[test]
    fn the_mask_stays_in_the_normaliser() {
        // Raising only the mask logit moves every confidence down, so the score of a position
        // depends on it. The engine has the same behaviour, because it masks after the softmax.
        let c = [1.0f32, 2.0, 3.0, 0.0];
        let u = [0.0f32, 0.0, 0.0, 0.0];
        let mut y = vec![0.0f32; 4];
        let (low, _) = guided_peak(&c, &u, 2.5, 3, &mut y);
        let mut c_high = c;
        c_high[3] = 10.0;
        let (high, _) = guided_peak(&c_high, &u, 2.5, 3, &mut y);
        assert!(high < low - 1.0, "the mask logit must lower every confidence: {low} -> {high}");
    }
    #[test]
    fn the_product_sampling_stays_in_the_top_ten_percent() {
        // 20 tokens: the top 2 are far above the rest, so k = ceil(2.0) = 2.
        let mut c = vec![-5.0f32; 20];
        let u = vec![0.0f32; 20];
        c[3] = 10.0;
        c[11] = 9.0;
        let mut y = vec![0.0f32; 20];
        let mut rng = Rng(7);
        let mut seen = [0usize; 20];
        for _ in 0..2000 {
            let (_, token) = sample_row(&c, &u, 0.0, 19, 0.7, &mut y, &mut rng);
            seen[token as usize] += 1;
        }
        let outside = (0..20).any(|i| seen[i] > 0 && i != 3 && i != 11);
        assert!(!outside, "a token outside the top two was drawn: {seen:?}");
        assert!(seen[3] > 0 && seen[11] > 0, "both of the top two must appear: {seen:?}");
    }
    #[test]
    fn the_gumbel_stream_is_repeatable() {
        let mut a = Rng(42);
        let mut b = Rng(42);
        for draw in 0..8 {
            assert_eq!(a.gumbel(), b.gumbel(), "draw {draw}");
        }
        assert!(Rng(1).gumbel().is_finite());
    }
}
