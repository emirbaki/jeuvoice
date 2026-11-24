//! The token-level times without DTW: `get_signal_energy` and
//! `whisper_exp_compute_token_level_timestamps` of whisper.cpp.

mod common;

use jeuvoice_whisper::decode::sample::TokenData;
use jeuvoice_whisper::decode::token_times::{signal_energy, token_level_times, voice_length, TokenTimeState, THOLD_PT, THOLD_PTSUM};
use jeuvoice_whisper::decode::vocab::Vocab;
use jeuvoice_whisper::ggml::GgmlModel;

#[test]
fn signal_energy_is_the_mean_absolute_value_over_a_window() {
    let e = signal_energy(&[1.0, -2.0, 3.0], 1);
    // The window is 2 × 1 + 1 = 3 samples. Samples past the ends count as nothing, but the
    // sum is still divided by 3.
    assert_eq!(e, vec![3.0 / 3.0, 6.0 / 3.0, 5.0 / 3.0]);
    assert!(signal_energy(&[], 32).is_empty());
}

#[test]
fn voice_length_weights() {
    // A space 0.01, a comma 2, a full stop, ! and ? 3, a digit 3, any other byte 1.
    assert_eq!(voice_length(b" a,"), 0.01 + 1.0 + 2.0);
    assert_eq!(voice_length(b".!?7"), 12.0);
    // A two-byte UTF-8 letter counts as two bytes.
    assert_eq!(voice_length("ş".as_bytes()), 2.0);
}

fn tok(id: u32) -> TokenData {
    TokenData { id, tid: 0, p: 0.5, plog: -0.7, pt: 0.0, ptsum: 0.0, t0: -1, t1: -1, t_dtw: -1 }
}

#[test]
fn one_token_gets_the_segment_times() {
    let v = common::test_vocab();
    let mut toks = vec![tok(10)];
    token_level_times(&v, &mut TokenTimeState::default(), &[0.1; 1000], 100, 250, &mut toks);
    assert_eq!((toks[0].t0, toks[0].t1), (100, 250));
}

#[test]
fn no_signal_leaves_the_times_unset() {
    // whisper.cpp logs "no signal data available" and returns.
    let v = common::test_vocab();
    let mut toks = vec![tok(v.beg), tok(10), tok(v.beg + 50)];
    token_level_times(&v, &mut TokenTimeState::default(), &[], 0, 100, &mut toks);
    assert!(toks.iter().all(|t| t.t0 == -1 && t.t1 == -1));
}

#[test]
fn a_window_start_sets_t_beg_and_t_last() {
    let v = common::test_vocab();
    let mut state = TokenTimeState::default();
    let mut toks = vec![tok(v.beg), tok(10), tok(11), tok(v.beg + 100)];
    token_level_times(&v, &mut state, &[0.0; 40_000], 300, 500, &mut toks);
    assert_eq!(state.t_beg, 300);
    assert_eq!(state.t_last, 500);
    assert_eq!(toks[0].t0, 300);
    assert_eq!(toks[3].t1, 500);
}

fn real_vocab() -> Vocab {
    Vocab::from_ggml(&GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap()).unwrap()
}

#[test]
#[ignore = "needs the model file, the test clips and the whisper-rs fixtures (Task 0)"]
fn times_match_whisper_rs() {
    let v = real_vocab();
    for clip in common::CLIPS {
        let f = common::load(clip);
        let energy = signal_energy(&common::samples(&f), 32);
        let mut state = TokenTimeState::default();
        let (mut n, mut bad) = (0, Vec::new());
        for (si, s) in f.segments.iter().enumerate() {
            let mut toks: Vec<TokenData> = s.tokens.iter().map(|t| TokenData { t0: -1, t1: -1, ..*t }).collect();
            token_level_times(&v, &mut state, &energy, s.t0, s.t1, &mut toks);
            for (j, (got, want)) in toks.iter().zip(&s.tokens).enumerate() {
                n += 1;
                if (got.t0, got.t1) != (want.t0, want.t1) {
                    bad.push(format!("segment {si} token {j} (id {}): ({}, {}) vs whisper-rs ({}, {})", got.id, got.t0, got.t1, want.t0, want.t1));
                }
            }
        }
        assert!(bad.is_empty(), "{clip}: {} of {n} tokens differ:\n{}", bad.len(), bad.join("\n"));
        eprintln!("{clip}: all {n} token times equal");
    }
    assert!(THOLD_PT == 0.01 && THOLD_PTSUM == 0.01);
}
