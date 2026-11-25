//! The sampler and the sequence score: ports of `whisper_sample_token` (whisper.cpp line
//! 6425) and `whisper_sequence_score` (line 6560).

use std::collections::BTreeMap;

use super::rules::Processed;
use super::vocab::Vocab;

/// One sampled token, as `whisper_token_data`. Times are in centiseconds, −1 when not set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenData {
    pub id: u32,
    /// The most likely timestamp token at this step (0 when no timestamp has a prob).
    pub tid: u32,
    /// The prob and the log prob of `id` after the rules.
    pub p: f32,
    pub plog: f32,
    /// The prob of `tid` over the sum of the timestamp probs.
    pub pt: f32,
    /// The sum of the timestamp probs.
    pub ptsum: f32,
    pub t0: i64,
    pub t1: i64,
    pub t_dtw: i64,
}

/// The random generator for the draws above temperature 0 (splitmix64).
///
/// whisper.cpp draws with `std::mt19937` and `std::discrete_distribution`. This generator
/// gives other draws, so a window that falls back to a temperature above 0 can give other
/// text than whisper.cpp (watch item W2 of the Phase 1b plan).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    /// A value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Samples one token from the probs of `p`. `best` takes the most likely token (the first
/// one on a tie). Else the token is a random draw with the probs as weights.
pub fn sample_token(vocab: &Vocab, p: &Processed, best: bool, rng: &mut Rng) -> TokenData {
    let beg = vocab.beg as usize;
    let mut t = TokenData { id: 0, tid: 0, p: 0.0, plog: 0.0, pt: 0.0, ptsum: 0.0, t0: -1, t1: -1, t_dtw: -1 };

    // whisper.cpp sums in double and skips probs of −∞ (a prob is never −∞, so it skips none).
    let (mut sum_ts, mut max_ts) = (0.0f64, 0.0f64);
    for (i, &pr) in p.probs.iter().enumerate().skip(beg) {
        sum_ts += pr as f64;
        if max_ts < pr as f64 {
            max_ts = pr as f64;
            t.tid = i as u32;
        }
    }
    t.pt = (max_ts / (sum_ts + 1e-10)) as f32;
    t.ptsum = sum_ts as f32;

    let id = if best {
        let mut id = 0;
        let mut pmax = 0.0f32;
        for (i, &pr) in p.probs.iter().enumerate() {
            if pmax < pr {
                pmax = pr;
                id = i;
            }
        }
        id
    } else {
        draw(&p.probs, rng)
    };
    t.id = id as u32;
    t.p = p.probs[id];
    t.plog = p.logprobs[id];

    if id >= beg {
        t.tid = t.id;
        t.pt = t.p;
    }
    t
}

/// A random index with the weights `probs`. With no weight above 0 it gives index 0.
fn draw(probs: &[f32], rng: &mut Rng) -> usize {
    let total: f64 = probs.iter().map(|&x| x as f64).sum();
    if total <= 0.0 {
        return 0;
    }
    let u = rng.next_f64() * total;
    let mut acc = 0.0f64;
    let mut last = 0;
    for (i, &pr) in probs.iter().enumerate() {
        if pr <= 0.0 {
            continue;
        }
        acc += pr as f64;
        last = i;
        if u < acc {
            return i;
        }
    }
    last
}

/// The score of a decoded sequence, as `whisper_sequence`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceScore {
    pub sum_logprobs: f64,
    pub avg_logprobs: f64,
    /// The entropy of the token ids of the last 32 tokens.
    pub entropy: f64,
    pub score: f64,
}

/// Scores the first `result_len` tokens. Gives `None` for `result_len == 0`, where
/// whisper.cpp leaves the score at its start values.
///
/// # Panics
/// If `result_len` is more than the number of tokens.
pub fn sequence_score(tokens: &[TokenData], result_len: usize, length_penalty: f32) -> Option<SequenceScore> {
    if result_len == 0 {
        return None;
    }
    let toks = &tokens[..result_len];
    let sum: f64 = toks.iter().map(|t| t.plog as f64).sum();
    let mut penalty = result_len as f64;
    if length_penalty > 0.0 {
        penalty = ((5.0 + penalty) / 6.0).powf(length_penalty as f64);
    }

    // whisper.cpp counts the ids in a std::map, so the sum runs in the order of the ids.
    let last = &toks[result_len.saturating_sub(32)..];
    let mut counts: BTreeMap<u32, u32> = BTreeMap::new();
    for t in last {
        *counts.entry(t.id).or_default() += 1;
    }
    let mut entropy = 0.0f64;
    for &c in counts.values() {
        let p = c as f64 / last.len() as f64;
        entropy -= p * p.ln();
    }

    Some(SequenceScore { sum_logprobs: sum, avg_logprobs: sum / result_len as f64, entropy, score: sum / penalty })
}
