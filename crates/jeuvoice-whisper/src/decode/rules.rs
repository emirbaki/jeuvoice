//! The token rules: a port of `whisper_process_logits` (whisper.cpp line 6142) for the
//! parameters of the app.
//!
//! The rules change one logits row before the sampler reads it. They make the text follow
//! the Whisper format: timestamps in pairs, timestamps that do not go back, and no special
//! tokens in the text.
//!
//! The app's parameters fix these whisper.cpp options: `suppress_blank = true`,
//! `max_initial_ts = 1.0`, `no_timestamps = false`, `tdrz_enable = false`,
//! `suppress_nst = false`, no `suppress_regex`, no grammar and no logits callback.

use super::sample::TokenData;
use super::vocab::{Vocab, N_LANGUAGES};

/// `max_initial_ts` of whisper.cpp: the first timestamp is at most 1.0 s.
const MAX_INITIAL_TS: f32 = 1.0;
/// One timestamp step: `WHISPER_CHUNK_SIZE / n_audio_ctx = 30 / 1500` seconds.
const TS_PRECISION: f32 = 30.0 / 1500.0;

/// The state of the decoder that the rules read.
pub struct RuleInput<'a> {
    /// The tokens sampled so far in this window and at this temperature (not the prompt).
    pub sampled: &'a [TokenData],
    /// The temperature. 0.0 means greedy.
    pub temperature: f32,
    /// True after the decoder sampled a timestamp in this window.
    pub has_ts: bool,
    /// The window advance so far, in centiseconds (`2 × (last timestamp − BEG)`).
    pub seek_delta: i32,
}

/// One logits row after the rules, with its log-softmax and softmax.
#[derive(Clone, Debug)]
pub struct Processed {
    pub logits: Vec<f32>,
    pub logprobs: Vec<f32>,
    pub probs: Vec<f32>,
}

/// Applies the rules to the raw logits `raw` (one row of `n_vocab` values) in the order of
/// whisper.cpp, then computes the logprobs and the probs.
///
/// # Panics
/// If `raw` does not have `n_vocab` values.
pub fn process_logits(vocab: &Vocab, raw: &[f32], input: &RuleInput) -> Processed {
    let n = vocab.n_vocab();
    assert_eq!(raw.len(), n, "process_logits: the row has {} logits, not n_vocab = {n}", raw.len());
    let (beg, eot) = (vocab.beg as usize, vocab.eot as usize);
    let is_initial = input.sampled.is_empty();
    let neg = f32::NEG_INFINITY;

    let mut logits = raw.to_vec();
    if input.temperature > 0.0 {
        for l in &mut logits {
            *l /= input.temperature;
        }
    }

    // Suppress blank: the first token cannot be EOT or " ".
    if is_initial {
        logits[eot] = neg;
        logits[vocab.space as usize] = neg;
    }

    // The special tokens never appear in the text.
    for id in [vocab.not, vocab.sot, vocab.nosp, vocab.solm, vocab.translate, vocab.transcribe, vocab.prev] {
        logits[id as usize] = neg;
    }
    for lang in 0..N_LANGUAGES {
        logits[vocab.lang_token(lang) as usize] = neg;
    }

    // Timestamps come in pairs, except just before EOT. With fewer than 2 sampled tokens,
    // whisper.cpp counts the penultimate token as a timestamp.
    let ts = |t: &TokenData| t.id as usize >= beg;
    let last_was_ts = input.sampled.last().is_some_and(ts);
    let penult_was_ts = input.sampled.len() < 2 || ts(&input.sampled[input.sampled.len() - 2]);
    if last_was_ts {
        if penult_was_ts {
            logits[beg..].fill(neg);
        } else {
            logits[..eot].fill(neg);
        }
    }

    // The first timestamp is at most MAX_INITIAL_TS.
    if is_initial {
        let tid0 = (MAX_INITIAL_TS / TS_PRECISION).round() as usize;
        let from = (beg + tid0 + 1).min(n);
        logits[from..].fill(neg);
    }

    // Timestamps do not go back.
    if input.has_ts {
        let tid0 = (input.seek_delta / 2).max(0) as usize;
        let to = (beg + tid0).min(n);
        logits[beg..to].fill(neg);
    }

    let mut logprobs = log_softmax(&logits);

    // If the timestamps together are more likely than any one other token, the next token
    // is a timestamp. This masks every id below BEG (EOT too). whisper.cpp does not compute
    // the softmax again after this mask.
    let ts_logprob = {
        let max = first_max(&logprobs[beg..]);
        let mut sum = 0.0f32;
        for &lp in &logprobs[beg..] {
            if lp > neg {
                sum += (lp - max).exp();
            }
        }
        if sum > 0.0 { sum.ln() + max } else { neg }
    };
    let max_text_logprob = first_max(&logprobs[..beg]);
    if ts_logprob > max_text_logprob {
        logits[..beg].fill(neg);
        logprobs[..beg].fill(neg);
    }

    // whisper_compute_probs: a masked logit gets 0.0, every other gets exp(logprob).
    let probs = logits.iter().zip(&logprobs).map(|(&l, &lp)| if l == neg { 0.0 } else { lp.exp() }).collect();
    Processed { logits, logprobs, probs }
}

/// `whisper_compute_logprobs`: the log-softmax in f32, with masked logits kept at −∞.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let neg = f32::NEG_INFINITY;
    let max = first_max(logits);
    let mut sum = 0.0f32;
    for &l in logits {
        if l > neg {
            sum += (l - max).exp();
        }
    }
    let lse = sum.ln() + max;
    logits.iter().map(|&l| if l > neg { l - lse } else { neg }).collect()
}

/// The largest value, as `*std::max_element` gives it (−∞ for an empty slice).
fn first_max(xs: &[f32]) -> f32 {
    xs.iter().copied().fold(f32::NEG_INFINITY, |m, x| if m < x { x } else { m })
}
