//! The token rules (`whisper_process_logits`), the sampler (`whisper_sample_token`) and the
//! sequence score (`whisper_sequence_score`) of whisper.cpp, on made-up logits.

use jeuvoice_whisper::decode::rules::{process_logits, Processed, RuleInput};
use jeuvoice_whisper::decode::sample::{sample_token, sequence_score, Rng, TokenData};
use jeuvoice_whisper::decode::vocab::Vocab;

fn vocab() -> Vocab {
    let words = (0..50257)
        .map(|i| match i {
            220 => b" ".to_vec(),
            50256 => Vec::new(),
            _ => format!("w{i}").into_bytes(),
        })
        .collect();
    Vocab::new(51866, words).unwrap()
}

/// Raw logits: 0.0 everywhere, and `10.0` at text token 1000. With this row the text token
/// wins over the summed timestamp mass (ln 1501 is about 7.3).
fn text_row(v: &Vocab) -> Vec<f32> {
    let mut r = vec![0.0; v.n_vocab()];
    r[1000] = 10.0;
    r
}

fn tok(id: u32) -> TokenData {
    TokenData { id, tid: 0, p: 0.0, plog: 0.0, pt: 0.0, ptsum: 0.0, t0: -1, t1: -1, t_dtw: -1 }
}

fn run(v: &Vocab, raw: &[f32], sampled: &[u32], temperature: f32, has_ts: bool, seek_delta: i32) -> Processed {
    let sampled: Vec<TokenData> = sampled.iter().map(|&id| tok(id)).collect();
    process_logits(v, raw, &RuleInput { sampled: &sampled, temperature, has_ts, seek_delta })
}

const NEG: f32 = f32::NEG_INFINITY;

#[test]
fn temperature_divides_the_logits() {
    let v = vocab();
    let mut raw = text_row(&v);
    raw[5] = 3.0;
    let p = run(&v, &raw, &[1], 0.5, false, 0);
    assert_eq!(p.logits[5], 6.0);
    assert_eq!(p.logits[1000], 20.0);
    let p = run(&v, &raw, &[1], 0.0, false, 0);
    assert_eq!(p.logits[5], 3.0);
}

#[test]
fn suppress_blank_only_at_the_first_token() {
    let v = vocab();
    let p = run(&v, &text_row(&v), &[], 0.0, false, 0);
    assert_eq!(p.logits[v.eot as usize], NEG);
    assert_eq!(p.logits[v.space as usize], NEG);
    let p = run(&v, &text_row(&v), &[1000], 0.0, false, 0);
    assert!(p.logits[v.eot as usize].is_finite());
    assert!(p.logits[v.space as usize].is_finite());
}

#[test]
fn special_and_language_tokens_are_always_masked() {
    let v = vocab();
    for sampled in [&[][..], &[1000][..]] {
        let p = run(&v, &text_row(&v), sampled, 0.0, false, 0);
        for id in [v.not, v.sot, v.nosp, v.solm, v.translate, v.transcribe, v.prev] {
            assert_eq!(p.logits[id as usize], NEG, "token {id}");
            assert_eq!(p.probs[id as usize], 0.0, "token {id}");
        }
        for lang in 0..100 {
            assert_eq!(p.logits[v.lang_token(lang) as usize], NEG, "language {lang}");
        }
    }
}

#[test]
fn timestamps_come_in_pairs() {
    let v = vocab();
    let (ts, ts2) = (v.beg + 10, v.beg + 20);
    let all_ts_masked = |p: &Processed| (v.beg as usize..v.n_vocab()).all(|i| p.logits[i] == NEG);
    let all_text_masked = |p: &Processed| (0..v.eot as usize).all(|i| p.logits[i] == NEG);

    // One sampled token that is a timestamp: the "penultimate" counts as a timestamp.
    let p = run(&v, &text_row(&v), &[ts], 0.0, false, 0);
    assert!(all_ts_masked(&p) && !all_text_masked(&p));
    // Two timestamps: the next token must be text.
    let p = run(&v, &text_row(&v), &[ts, ts2], 0.0, false, 0);
    assert!(all_ts_masked(&p) && !all_text_masked(&p));
    // Text, then a timestamp: the next token must be a timestamp or EOT.
    // EOT gets a strong logit: with the text masked, a weak EOT loses to the timestamp mass,
    // and the mass rule then masks EOT too.
    let mut raw = text_row(&v);
    raw[v.eot as usize] = 10.0;
    let p = run(&v, &raw, &[1000, ts], 0.0, false, 0);
    assert!(all_text_masked(&p));
    assert!(p.logits[v.eot as usize].is_finite());
    assert!(p.logits[(v.beg + 30) as usize].is_finite());
    // A timestamp, then text: no pair rule.
    let p = run(&v, &text_row(&v), &[ts, 1000], 0.0, false, 0);
    assert!(!all_ts_masked(&p) && !all_text_masked(&p));
}

#[test]
fn initial_timestamp_is_at_most_one_second() {
    let v = vocab();
    // 1.0 s / 0.02 s = 50: BEG + 50 is allowed, BEG + 51 and up are not.
    let p = run(&v, &text_row(&v), &[], 0.0, false, 0);
    assert!(p.logits[(v.beg + 50) as usize].is_finite());
    assert!((v.beg as usize + 51..v.n_vocab()).all(|i| p.logits[i] == NEG));
    let p = run(&v, &text_row(&v), &[1000], 0.0, false, 0);
    assert!(p.logits[(v.beg + 51) as usize].is_finite());
}

#[test]
fn timestamps_increase() {
    let v = vocab();
    // seek_delta = 100 centiseconds: timestamps below BEG + 50 are masked.
    let p = run(&v, &text_row(&v), &[1000], 0.0, true, 100);
    assert!((v.beg as usize..v.beg as usize + 50).all(|i| p.logits[i] == NEG));
    assert!(p.logits[(v.beg + 50) as usize].is_finite());
    // Without has_ts there is no mask.
    let p = run(&v, &text_row(&v), &[1000], 0.0, false, 100);
    assert!(p.logits[v.beg as usize].is_finite());
}

#[test]
fn timestamp_mass_masks_all_tokens_below_beg_without_a_new_softmax() {
    let v = vocab();
    // All logits 0.0: the 1501 timestamps together are more likely than any one text token.
    let raw = vec![0.0; v.n_vocab()];
    let p = run(&v, &raw, &[1000], 0.0, false, 0);
    for i in 0..v.beg as usize {
        assert_eq!(p.logits[i], NEG, "logit {i}");
        assert_eq!(p.logprobs[i], NEG, "logprob {i}");
        assert_eq!(p.probs[i], 0.0, "prob {i}");
    }
    // The logprobs keep the softmax over the 51759 tokens that were left before this rule
    // (51866 less 7 special tokens and 100 languages). whisper.cpp does not normalize again.
    let want = -(51759f32).ln();
    for i in [v.beg, v.beg + 700, v.beg + 1500] {
        assert!((p.logprobs[i as usize] - want).abs() < 1e-4, "logprob {i}: {} vs {want}", p.logprobs[i as usize]);
        assert!((p.probs[i as usize] - want.exp()).abs() < 1e-9);
    }
    // With a strong text token there is no mask.
    let p = run(&v, &text_row(&v), &[1000], 0.0, false, 0);
    assert!(p.logits[1000].is_finite() && p.logits[v.eot as usize].is_finite());
}

#[test]
fn logprobs_and_probs_are_a_softmax() {
    let v = vocab();
    let p = run(&v, &text_row(&v), &[1000], 0.0, false, 0);
    let sum: f64 = p.probs.iter().map(|&x| x as f64).sum();
    assert!((sum - 1.0).abs() < 1e-3, "sum of probs {sum}");
    assert!((p.probs[1000] - p.logprobs[1000].exp()).abs() < 1e-7);
}

fn probs_with(v: &Vocab, pairs: &[(u32, f32)]) -> Processed {
    let mut probs = vec![0.0; v.n_vocab()];
    let mut logprobs = vec![NEG; v.n_vocab()];
    for &(id, p) in pairs {
        probs[id as usize] = p;
        logprobs[id as usize] = p.ln();
    }
    Processed { logits: logprobs.clone(), logprobs, probs }
}

#[test]
fn greedy_sample_of_a_text_token() {
    let v = vocab();
    let p = probs_with(&v, &[(7, 0.6), (v.beg + 3, 0.3), (v.beg + 5, 0.1)]);
    let t = sample_token(&v, &p, true, &mut Rng::new(0));
    assert_eq!(t.id, 7);
    assert_eq!(t.p, 0.6);
    assert_eq!(t.plog, 0.6f32.ln());
    assert_eq!(t.tid, v.beg + 3);
    assert!((t.pt - 0.75).abs() < 1e-6, "pt {}", t.pt);
    assert!((t.ptsum - 0.4).abs() < 1e-6, "ptsum {}", t.ptsum);
    assert_eq!((t.t0, t.t1, t.t_dtw), (-1, -1, -1));
}

#[test]
fn greedy_sample_of_a_timestamp() {
    let v = vocab();
    let p = probs_with(&v, &[(7, 0.2), (v.beg + 3, 0.7), (v.beg + 5, 0.1)]);
    let t = sample_token(&v, &p, true, &mut Rng::new(0));
    assert_eq!((t.id, t.tid), (v.beg + 3, v.beg + 3));
    assert_eq!(t.pt, t.p);
    assert_eq!(t.p, 0.7);
}

#[test]
fn no_timestamp_mass_gives_tid_zero() {
    // whisper.cpp starts `tid` at 0 and changes it only for a timestamp prob above 0.
    let v = vocab();
    let t = sample_token(&v, &probs_with(&v, &[(7, 1.0)]), true, &mut Rng::new(0));
    assert_eq!((t.id, t.tid, t.pt, t.ptsum), (7, 0, 0.0, 0.0));
}

#[test]
fn equal_probs_take_the_first_id() {
    let v = vocab();
    let t = sample_token(&v, &probs_with(&v, &[(9, 0.5), (4, 0.5)]), true, &mut Rng::new(0));
    assert_eq!(t.id, 4);
}

#[test]
fn random_sample_follows_the_probs() {
    let v = vocab();
    let one = probs_with(&v, &[(42, 1.0)]);
    let mut rng = Rng::new(0);
    for _ in 0..100 {
        assert_eq!(sample_token(&v, &one, false, &mut rng).id, 42);
    }
    let two = probs_with(&v, &[(3, 0.25), (8, 0.75)]);
    let draw = |seed| {
        let mut rng = Rng::new(seed);
        (0..4000).map(|_| sample_token(&v, &two, false, &mut rng).id).collect::<Vec<_>>()
    };
    let a = draw(0);
    let share = a.iter().filter(|&&id| id == 8).count() as f64 / a.len() as f64;
    assert!((share - 0.75).abs() < 0.03, "share of token 8: {share}");
    assert_eq!(a, draw(0), "the same seed must give the same draws");
    assert_ne!(a, draw(1));
}

fn plog_tokens(ids_plogs: &[(u32, f32)]) -> Vec<TokenData> {
    ids_plogs.iter().map(|&(id, plog)| TokenData { plog, ..tok(id) }).collect()
}

#[test]
fn sequence_score_of_a_hand_case() {
    let t = plog_tokens(&[(5, -0.1), (5, -0.2), (6, -0.3)]);
    let s = sequence_score(&t, 3, -1.0).unwrap();
    assert!((s.sum_logprobs - -0.6).abs() < 1e-6);
    assert!((s.avg_logprobs - -0.2).abs() < 1e-6);
    assert!((s.score - -0.2).abs() < 1e-6);
    let h = -((2.0f64 / 3.0) * (2.0f64 / 3.0).ln() + (1.0 / 3.0) * (1.0f64 / 3.0).ln());
    assert!((s.entropy - h).abs() < 1e-12, "entropy {} vs {h}", s.entropy);
    // Only the first result_len tokens count.
    let s = sequence_score(&t, 2, -1.0).unwrap();
    assert!((s.avg_logprobs - -0.15).abs() < 1e-6);
    assert_eq!(s.entropy, 0.0);
    // A length penalty above 0 uses ((5 + n) / 6) ^ penalty.
    let s = sequence_score(&t, 3, 1.0).unwrap();
    assert!((s.score - -0.6 / (8.0 / 6.0)).abs() < 1e-6);
    assert!(sequence_score(&t, 0, -1.0).is_none());
}

#[test]
fn entropy_uses_the_last_32_tokens() {
    // 8 tokens of id 1, then 32 tokens of id 2: the last 32 are all the same.
    let ids: Vec<(u32, f32)> = (0..40).map(|i| (if i < 8 { 1 } else { 2 }, -0.1)).collect();
    let s = sequence_score(&plog_tokens(&ids), 40, -1.0).unwrap();
    assert_eq!(s.entropy, 0.0);
    let s = sequence_score(&plog_tokens(&ids), 33, -1.0).unwrap();
    assert!(s.entropy > 0.0);
}
