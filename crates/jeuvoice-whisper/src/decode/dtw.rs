//! DTW word times: a port of `whisper_exp_compute_token_level_timestamps_dtw` (whisper.cpp
//! line 8802), `median_filter` (line 8767) and `dtw_and_backtrace` (line 8677).
//!
//! Some cross-attention heads of the decoder (the "alignment heads") look at the audio
//! position of the token that they write. DTW (dynamic time warping) finds the cheapest
//! path through the `[token, audio]` cost matrix that moves forward in both. The audio
//! position where the path reaches a token is the start time of that token (`t_dtw`).

use super::sample::TokenData;
use super::vocab::Vocab;
use super::window::Segment;

/// The alignment heads of large-v3-turbo as `(decoder layer, head)`, in the order of
/// `g_aheads_large_v3_turbo` (whisper.cpp line 395).
pub const ALIGNMENT_HEADS: [(usize, usize); 6] = [(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)];
/// The width of the median filter over the audio positions.
pub const MEDFILT_WIDTH: usize = 7;
/// The number of prompt tokens before NOT in the DTW tokens: SOT and the language.
const SOT_SEQUENCE_LEN: usize = 2;

/// The tokens of the DTW pass: `[SOT, lang, NOT]`, the text tokens of `segments`, EOT.
pub fn dtw_tokens(vocab: &Vocab, lang_token: u32, segments: &[Segment]) -> Vec<u32> {
    let mut t = vec![vocab.sot, lang_token, vocab.not];
    t.extend(segments.iter().flat_map(|s| &s.tokens).filter(|t| vocab.is_text(t.id)).map(|t| t.id));
    t.push(vocab.eot);
    t
}

/// `ggml_norm` with eps 1e-9 over the tokens: for each head and audio position,
/// `(x − mean) / sqrt(var + eps)`. `w` is `[heads][tokens][audio]`.
pub fn normalize_over_tokens(w: &mut [f32], n_heads: usize, n_tokens: usize, n_audio: usize) {
    assert_eq!(w.len(), n_heads * n_tokens * n_audio, "normalize_over_tokens: the size is not heads × tokens × audio");
    let eps = 1e-9f32;
    let at = |h: usize, t: usize, a: usize| (h * n_tokens + t) * n_audio + a;
    for h in 0..n_heads {
        for a in 0..n_audio {
            // ggml sums in double and rounds the sum to float.
            let sum: f64 = (0..n_tokens).map(|t| w[at(h, t, a)] as f64).sum();
            let mean = sum as f32 / n_tokens as f32;
            let mut sum2 = 0.0f64;
            for t in 0..n_tokens {
                let y = w[at(h, t, a)] - mean;
                w[at(h, t, a)] = y;
                sum2 += (y * y) as f64;
            }
            let variance = (sum2 / n_tokens as f64) as f32;
            let scale = 1.0 / (variance + eps).sqrt();
            for t in 0..n_tokens {
                w[at(h, t, a)] *= scale;
            }
        }
    }
}

/// The median of seven values: a sorting network of 19 compare-exchanges (Knuth, TAOCP
/// 5.3.4). It needs no allocation and no comparison function, so the filter spends its time
/// on loads and compares. Lesson 9 measures it against `sort_by`.
#[inline]
fn median7(mut v: [f32; 7]) -> f32 {
    #[inline]
    fn ce(v: &mut [f32; 7], i: usize, j: usize) {
        if v[i] > v[j] {
            v.swap(i, j);
        }
    }
    ce(&mut v, 0, 6);
    ce(&mut v, 2, 3);
    ce(&mut v, 4, 5);
    ce(&mut v, 0, 2);
    ce(&mut v, 1, 4);
    ce(&mut v, 3, 6);
    ce(&mut v, 0, 1);
    ce(&mut v, 2, 5);
    ce(&mut v, 3, 4);
    ce(&mut v, 1, 2);
    ce(&mut v, 4, 6);
    ce(&mut v, 2, 3);
    ce(&mut v, 4, 5);
    ce(&mut v, 1, 2);
    ce(&mut v, 3, 4);
    ce(&mut v, 5, 6);
    v[3]
}

/// The median filter of whisper.cpp over the last dimension of `x` (`[rows][n]`), with
/// reflect padding: index −1 reads index 1, and index n reads n − 2.
///
/// # Panics
/// If `width` is even or not below `n` (whisper.cpp asserts the same).
pub fn median_filter(x: &mut [f32], rows: usize, n: usize, width: usize) {
    assert!(width % 2 == 1 && width < n, "median_filter: the width {width} must be odd and below {n}");
    assert_eq!(x.len(), rows * n, "median_filter: the size is not rows × n");
    if width == 7 {
        // The product path: a fixed window, the two padded ends handled on their own, so the
        // middle of the row has no branch at all.
        let n_i = n as i64;
        let half = 3usize;
        let mut out = vec![0.0f32; n];
        for r in 0..rows {
            let row = &mut x[r * n..(r + 1) * n];
            for (k, o) in out.iter_mut().enumerate() {
                let mut v = [0.0f32; 7];
                if k < half {
                    for (j, off) in (0..7).enumerate() {
                        let mut i = k as i64 + off as i64 - half as i64;
                        if i < 0 {
                            i = -i;
                        }
                        v[j] = row[i as usize];
                    }
                } else if k + half >= n {
                    for (j, off) in (0..7).enumerate() {
                        let mut i = k as i64 + off as i64 - half as i64;
                        if i >= n_i {
                            i = 2 * (n_i - 1) - i;
                        }
                        v[j] = row[i as usize];
                    }
                } else {
                    let base = k - half;
                    v.copy_from_slice(&row[base..base + 7]);
                }
                *o = median7(v);
            }
            row.copy_from_slice(&out);
        }
        return;
    }
    let half = (width / 2) as i64;
    let mut window = Vec::with_capacity(width);
    let mut out = vec![0.0f32; n];
    for r in 0..rows {
        let row = &mut x[r * n..(r + 1) * n];
        for (k, o) in out.iter_mut().enumerate() {
            window.clear();
            for off in -half..=half {
                let mut i = k as i64 + off;
                if i < 0 {
                    i = -i;
                } else if i >= n as i64 {
                    i = 2 * (n as i64 - 1) - i;
                }
                window.push(row[i as usize]);
            }
            window.sort_by(|a, b| a.total_cmp(b));
            *o = window[window.len() / 2];
        }
        row.copy_from_slice(&out);
    }
}

/// Steps 4 to 7 of the DTW: normalize over the tokens, the median filter, the mean over
/// the heads times −1, and the rows of SOT, the language and EOT left out. `w` is the
/// alignment-head softmax `[heads][tokens][audio]`. The result is `[tokens − 3][audio]`:
/// row 0 is NOT, then the text tokens.
pub fn cost_matrix(w: &[f32], n_heads: usize, n_tokens: usize, n_audio: usize) -> Vec<f32> {
    assert!(n_tokens > SOT_SEQUENCE_LEN + 1, "cost_matrix: {n_tokens} tokens is too few");
    let mut w = w.to_vec();
    normalize_over_tokens(&mut w, n_heads, n_tokens, n_audio);
    median_filter(&mut w, n_heads * n_tokens, n_audio, MEDFILT_WIDTH);
    let rows = n_tokens - SOT_SEQUENCE_LEN - 1;
    let mut out = vec![0.0f32; rows * n_audio];
    for r in 0..rows {
        let t = r + SOT_SEQUENCE_LEN;
        for a in 0..n_audio {
            // ggml_mean: a double sum rounded to float, divided by the count as float.
            let sum: f64 = (0..n_heads).map(|h| w[(h * n_tokens + t) * n_audio + a] as f64).sum();
            out[r * n_audio + a] = -(sum as f32 / n_heads as f32);
        }
    }
    out
}

/// DTW with backtrace on `x` (`[n][m]`, row-major). Gives the path from `(0, 0)` to
/// `(n − 1, m − 1)` as `(row, column)` pairs.
///
/// The step rule of whisper.cpp: the diagonal step when it is strictly the cheapest, else
/// the up step when it is strictly the cheapest, else the left step. So a tie goes left.
pub fn dtw_path(x: &[f32], n: usize, m: usize) -> Vec<(usize, usize)> {
    assert_eq!(x.len(), n * m, "dtw_path: the size is not n × m");
    let inf = f32::INFINITY;
    let w = m + 1;
    let mut cost = vec![inf; (n + 1) * w];
    let mut trace = vec![-1i8; (n + 1) * w];
    cost[0] = 0.0;
    for j in 1..=m {
        for i in 1..=n {
            let c0 = cost[(i - 1) * w + j - 1];
            let c1 = cost[(i - 1) * w + j];
            let c2 = cost[i * w + j - 1];
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            cost[i * w + j] = x[(i - 1) * m + j - 1] + c;
            trace[i * w + j] = t;
        }
    }
    for j in 0..=m {
        trace[j] = 2;
    }
    for i in 0..=n {
        trace[i * w] = 1;
    }
    let (mut i, mut j) = (n, m);
    let mut path = Vec::with_capacity(n + m);
    while i > 0 || j > 0 {
        // The rows and columns of a path from (n, m) never reach 0 before (0, 0): an
        // infinite cost is never strictly the smallest.
        path.push((i - 1, j - 1));
        match trace[i * w + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            2 => j -= 1,
            t => unreachable!("dtw_path: no step at ({i}, {j}): {t}"),
        }
    }
    path.reverse();
    path
}

/// Step 9: each time the row of the path changes, the next text token of `segments` gets
/// `t_dtw = 2 × column + seek`. Row 0 (NOT) gets no time. Tokens that are not text (id at
/// or above EOT) are skipped.
pub fn place_dtw_times(vocab: &Vocab, segments: &mut [Segment], path: &[(usize, usize)], seek: i64) {
    let mut text: Vec<&mut TokenData> = segments.iter_mut().flat_map(|s| s.tokens.iter_mut()).filter(|t| t.id < vocab.eot).collect();
    let mut next = 0;
    let mut last_row = 0;
    for &(row, col) in path {
        if row != last_row {
            last_row = row;
            if let Some(t) = text.get_mut(next) {
                t.t_dtw = 2 * col as i64 + seek;
            }
            next += 1;
        }
    }
}
