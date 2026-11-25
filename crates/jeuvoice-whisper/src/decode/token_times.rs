//! The token-level times without DTW: a port of `get_signal_energy` (whisper.cpp line
//! 8390) and `whisper_exp_compute_token_level_timestamps` (line 8420).
//!
//! The decoder gives a timestamp only at the ends of a segment. These functions give each
//! token a start `t0` and an end `t1` (centiseconds) from three sources: the most likely
//! timestamp at each step (`tid`, `pt`, `ptsum`), the length of the token text, and the
//! signal energy. The app uses `t1` for the end of a word.
//!
//! The VAD step turns times into sample indexes relative to the segment start, but it reads
//! the energy of the whole file at those indexes. So for a segment after the start it reads
//! the energy at the start of the audio. The port keeps this, because the app compares
//! with whisper.cpp.

use super::sample::TokenData;
use super::vocab::Vocab;

/// `thold_pt` of the app's parameters.
pub const THOLD_PT: f32 = 0.01;
/// `thold_ptsum` of the app's parameters.
pub const THOLD_PTSUM: f32 = 0.01;
/// `WHISPER_SAMPLE_RATE`.
const SAMPLE_RATE: i64 = 16_000;

/// The mean absolute value of `signal` over a window of `2 × half_window + 1` samples. At
/// the ends the missing samples count as nothing, and the divisor stays the same.
pub fn signal_energy(signal: &[f32], half_window: usize) -> Vec<f32> {
    let n = signal.len() as i64;
    let hw = half_window as i64;
    let div = (2 * hw + 1) as f32;
    (0..n)
        .map(|i| {
            // The same order of the f32 sum as whisper.cpp.
            let mut sum = 0.0f32;
            for j in -hw..=hw {
                if i + j >= 0 && i + j < n {
                    sum += signal[(i + j) as usize].abs();
                }
            }
            sum / div
        })
        .collect()
}

/// `voice_length`: a cost that is high for text that takes longer to say. It counts bytes.
pub fn voice_length(text: &[u8]) -> f32 {
    let mut res = 0.0f32;
    for &c in text {
        res += match c {
            b' ' => 0.01,
            b',' => 2.0,
            b'.' | b'!' | b'?' => 3.0,
            b'0'..=b'9' => 3.0,
            _ => 1.0,
        };
    }
    res
}

/// The state that whisper.cpp keeps across segments (`t_beg`, `t_last`, `tid_last`). It
/// starts at 0 for each transcription.
#[derive(Clone, Copy, Debug, Default)]
pub struct TokenTimeState {
    pub t_beg: i64,
    pub t_last: i64,
    pub tid_last: i64,
}

fn timestamp_to_sample(t: i64, segment_t0: i64, n_samples: i64) -> i64 {
    // whisper.cpp casts the result to `int` before the clamp.
    let sample = ((t - segment_t0) * SAMPLE_RATE / 100) as i32 as i64;
    sample.min(n_samples - 1).max(0)
}

fn sample_to_timestamp(i_sample: i64, segment_t0: i64) -> i64 {
    100 * i_sample / SAMPLE_RATE + segment_t0
}

/// Sets `t0` and `t1` of the tokens of one segment (`seg_t0` to `seg_t1`). `energy` is the
/// `signal_energy` of the whole audio with a half window of 32. With no energy (no audio)
/// the times stay unset, as in whisper.cpp.
pub fn token_level_times(vocab: &Vocab, state: &mut TokenTimeState, energy: &[f32], seg_t0: i64, seg_t1: i64, tokens: &mut [TokenData]) {
    let n_samples = energy.len() as i64;
    if n_samples == 0 {
        return;
    }
    let n = tokens.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        tokens[0].t0 = seg_t0;
        tokens[0].t1 = seg_t1;
        return;
    }
    let beg = vocab.beg as i64;
    let mut vlen = vec![0.0f32; n];

    for j in 0..n {
        if j == 0 {
            if tokens[0].id == vocab.beg {
                tokens[0].t0 = seg_t0;
                tokens[0].t1 = seg_t0;
                tokens[1].t0 = seg_t0;
                state.t_beg = seg_t0;
                state.t_last = seg_t0;
                state.tid_last = beg;
            } else {
                tokens[0].t0 = state.t_last;
            }
        }
        let token = tokens[j];
        let tt = state.t_beg + 2 * (token.tid as i64 - beg);
        vlen[j] = voice_length(vocab.token_bytes(token.id));
        if token.pt > THOLD_PT && token.ptsum > THOLD_PTSUM && token.tid as i64 > state.tid_last && tt <= seg_t1 {
            if j > 0 {
                tokens[j - 1].t1 = tt;
            }
            tokens[j].t0 = tt;
            state.tid_last = token.tid as i64;
        }
    }

    tokens[n - 2].t1 = seg_t1;
    tokens[n - 1].t0 = seg_t1;
    tokens[n - 1].t1 = seg_t1;
    state.t_last = seg_t1;

    // Split the intervals of tokens with no time by the voice lengths.
    let (mut p0, mut p1) = (0usize, 0usize);
    loop {
        while p1 < n && tokens[p1].t1 < 0 {
            p1 += 1;
        }
        if p1 >= n {
            p1 -= 1;
        }
        if p1 > p0 {
            let psum: f64 = vlen[p0..=p1].iter().map(|&v| v as f64).sum();
            let dt = (tokens[p1].t1 - tokens[p0].t0) as f64;
            for j in p0 + 1..=p1 {
                // C++ converts the double to int64 by truncation.
                let ct = (tokens[j - 1].t0 as f64 + dt * vlen[j - 1] as f64 / psum) as i64;
                tokens[j - 1].t1 = ct;
                tokens[j].t0 = ct;
            }
        }
        p1 += 1;
        p0 = p1;
        if p1 >= n {
            break;
        }
    }

    // Fix the order (whisper.cpp: "just in case").
    for j in 0..n - 1 {
        if tokens[j].t1 < 0 {
            tokens[j + 1].t0 = tokens[j].t1;
        }
        if j > 0 && tokens[j - 1].t1 > tokens[j].t0 {
            tokens[j].t0 = tokens[j - 1].t1;
            tokens[j].t1 = tokens[j].t0.max(tokens[j].t1);
        }
    }

    // VAD: move the token ends to where the energy crosses half its local mean.
    let hw = SAMPLE_RATE / 8;
    let e = |k: i64| energy[k as usize];
    for j in 0..n {
        if tokens[j].id >= vocab.eot {
            continue;
        }
        let mut s0 = timestamp_to_sample(tokens[j].t0, seg_t0, n_samples);
        let mut s1 = timestamp_to_sample(tokens[j].t1, seg_t0, n_samples);
        let ss0 = (s0 - hw).max(0);
        let ss1 = (s1 + hw).min(n_samples);
        let ns = ss1 - ss0;
        let mut sum = 0.0f32;
        for k in ss0..ss1 {
            sum += e(k);
        }
        // `0.5 * sum / ns` is a double in whisper.cpp, stored in a float.
        let thold = (0.5 * sum as f64 / ns as f64) as f32;

        let mut k = s0;
        if e(k) > thold && j > 0 {
            while k > 0 && e(k) > thold {
                k -= 1;
            }
            tokens[j].t0 = sample_to_timestamp(k, seg_t0);
            if tokens[j].t0 < tokens[j - 1].t1 {
                tokens[j].t0 = tokens[j - 1].t1;
            } else {
                s0 = k;
            }
        } else {
            while e(k) < thold && k < s1 {
                k += 1;
            }
            s0 = k;
            tokens[j].t0 = sample_to_timestamp(k, seg_t0);
        }

        let mut k = s1;
        if e(k) > thold {
            while k < n_samples - 1 && e(k) > thold {
                k += 1;
            }
            tokens[j].t1 = sample_to_timestamp(k, seg_t0);
            if j < n - 1 && tokens[j].t1 > tokens[j + 1].t0 {
                tokens[j].t1 = tokens[j + 1].t0;
            } else {
                s1 = k;
            }
        } else {
            while e(k) < thold && k > s0 {
                k -= 1;
            }
            s1 = k;
            tokens[j].t1 = sample_to_timestamp(k, seg_t0);
        }
        // whisper.cpp keeps s0 and s1 after these lines but does not read them again.
        let _ = (s0, s1);
    }
}
