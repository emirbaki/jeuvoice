//! DTW word times: `whisper_exp_compute_token_level_timestamps_dtw`, `median_filter` and
//! `dtw_and_backtrace` of whisper.cpp.

mod common;

use jeuvoice_whisper::decode::dtw::{cost_matrix, dtw_path, median_filter, normalize_over_tokens, place_dtw_times};
use jeuvoice_whisper::decode::sample::TokenData;
use jeuvoice_whisper::decode::window::Segment;

#[test]
fn dtw_follows_a_diagonal() {
    let x = [0.0, 9.0, 9.0, 9.0, 0.0, 9.0, 9.0, 9.0, 0.0];
    assert_eq!(dtw_path(&x, 3, 3), vec![(0, 0), (1, 1), (2, 2)]);
}

#[test]
fn dtw_on_a_wide_matrix() {
    // Token 0 has audio 0 and 1, token 1 has audio 2 and 3.
    let x = [0.0, 0.0, 9.0, 9.0, 9.0, 9.0, 0.0, 0.0];
    assert_eq!(dtw_path(&x, 2, 4), vec![(0, 0), (0, 1), (1, 2), (1, 3)]);
}

#[test]
fn dtw_tie_goes_left() {
    // whisper.cpp takes the diagonal or the up step only when it is strictly the smallest.
    // With all costs equal at cell (2, 2) it takes the left step, so the path is not the
    // diagonal.
    let x = [0.0; 4];
    assert_eq!(dtw_path(&x, 2, 2), vec![(0, 0), (1, 0), (1, 1)]);
}

#[test]
fn median_filter_with_reflect_padding() {
    let mut x = vec![1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0];
    median_filter(&mut x, 1, 9, 3);
    // Ends: index -1 reflects to 1, and index 9 reflects to 7.
    assert_eq!(x, vec![9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0, 6.0]);
}

#[test]
fn median_filter_width_7_against_a_simple_reference() {
    let row: Vec<f32> = (0..12).map(|i| ((i * 37) % 11) as f32 - 3.0).collect();
    let n = row.len() as i64;
    let want: Vec<f32> = (0..n)
        .map(|k| {
            let mut w: Vec<f32> = (-3..=3)
                .map(|o| {
                    let i = k + o;
                    let i = if i < 0 { -i } else if i >= n { 2 * (n - 1) - i } else { i };
                    row[i as usize]
                })
                .collect();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            w[3]
        })
        .collect();
    // Two rows: each row is filtered alone.
    let mut x = [row.clone(), row].concat();
    median_filter(&mut x, 2, 12, 7);
    assert_eq!(&x[..12], &want[..]);
    assert_eq!(&x[12..], &want[..]);
}

#[test]
fn normalize_is_over_the_tokens() {
    // One head, 3 tokens, 2 audio positions: [h][token][audio].
    let mut w = vec![1.0, 10.0, 2.0, 10.0, 3.0, 10.0];
    normalize_over_tokens(&mut w, 1, 3, 2);
    let s = 1.0 / (2.0f32 / 3.0 + 1e-9).sqrt();
    let want = [-s, 0.0, 0.0, 0.0, s, 0.0];
    for (g, e) in w.iter().zip(want) {
        assert!((g - e).abs() < 1e-6, "{w:?}");
    }
}

#[test]
fn cost_matrix_drops_sot_lang_and_eot() {
    // 2 heads, 5 tokens [SOT, lang, NOT, text, EOT], 9 audio positions.
    let (h, t, a) = (2, 5, 9);
    let w: Vec<f32> = (0..h * t * a).map(|i| ((i * 7919) % 13) as f32 / 13.0).collect();
    let got = cost_matrix(&w, h, t, a);
    assert_eq!(got.len(), 2 * a, "rows NOT and text");

    let mut want = w.clone();
    normalize_over_tokens(&mut want, h, t, a);
    median_filter(&mut want, h * t, a, 7);
    for (row, tok) in [(0, 2), (1, 3)] {
        for k in 0..a {
            let mean = (want[tok * a + k] + want[(t + tok) * a + k]) / 2.0;
            assert!((got[row * a + k] + mean).abs() < 1e-6, "row {row} audio {k}");
        }
    }
}

fn tok(id: u32) -> TokenData {
    TokenData { id, tid: 0, p: 0.5, plog: -0.7, pt: 0.0, ptsum: 0.0, t0: -1, t1: -1, t_dtw: -1 }
}

fn seg(ids: &[u32]) -> Segment {
    Segment { t0: 0, t1: 0, text: vec![], no_speech_prob: 0.0, tokens: ids.iter().map(|&i| tok(i)).collect() }
}

#[test]
fn place_times_on_text_tokens_in_order() {
    let v = common::test_vocab();
    let ts = v.beg + 100;
    let mut segs = vec![seg(&[v.beg, 10, 11, ts]), seg(&[12, ts])];
    // Rows: 0 = NOT, 1 = token 10, 2 = token 11, 3 = token 12.
    let path = [(0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (3, 6)];
    place_dtw_times(&v, &mut segs, &path, 1000);
    let dtw: Vec<Vec<i64>> = segs.iter().map(|s| s.tokens.iter().map(|t| t.t_dtw).collect()).collect();
    assert_eq!(dtw, vec![vec![-1, 1004, 1008, -1], vec![1010, -1]]);
}
