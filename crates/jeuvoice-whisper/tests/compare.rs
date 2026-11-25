//! The measurement functions of plan 1b, Task 12: the word error rate and the word-start
//! match between two transcripts.

use jeuvoice_core::transcript::Word;
use jeuvoice_whisper::compare::{Op, align, compare_words, normalize};

fn w(text: &str, start: f64) -> Word {
    Word { text: text.into(), start, end: start + 0.2, prob: 0.9 }
}

#[test]
fn normalize_uses_the_turkish_lower_case_and_drops_punctuation() {
    assert_eq!(normalize("İstanbul,"), "istanbul");
    assert_eq!(normalize("IŞIK."), "ışık");
    assert_eq!(normalize("Dünya'nın"), "dünyanın");
    assert_eq!(normalize("2026"), "2026");
    assert_eq!(normalize("..."), "");
}

#[test]
fn align_finds_the_edits() {
    let r = ["a", "b", "c"].map(String::from);
    let h = ["a", "x", "c", "d"].map(String::from);
    assert_eq!(align(&r, &h), vec![Op::Match(0, 0), Op::Sub(1, 1), Op::Match(2, 2), Op::Ins(3)]);
    let h = ["b", "c"].map(String::from);
    assert_eq!(align(&r, &h), vec![Op::Del(0), Op::Match(1, 0), Op::Match(2, 1)]);
    assert_eq!(align(&[], &[]), vec![]);
}

#[test]
fn wer_and_start_differences() {
    let reference = [w("Dünyanın", 0.0), w("öbür", 0.5), w("ucunda", 1.0)];
    // One word changed and one extra word: WER 2 / 3. The two equal words have starts 0.02 s
    // and 0.04 s off, so the median of the start differences is the larger one (index 1 of 2).
    let ours = [w("dünyanın", 0.02), w("öbürü", 0.5), w("ucunda", 1.04), w("da", 1.3)];
    let c = compare_words(&reference, &ours);
    assert_eq!((c.substitutions, c.deletions, c.insertions), (1, 0, 1));
    assert!((c.wer - 2.0 / 3.0).abs() < 1e-12);
    assert_eq!(c.matched, 2);
    assert!((c.median_start_diff - 0.04).abs() < 1e-9, "{}", c.median_start_diff);
}

#[test]
fn equal_transcripts_have_no_error() {
    let a = [w("bir", 0.0), w("iki,", 0.3)];
    let b = [w("Bir", 0.0), w("iki", 0.3)];
    let c = compare_words(&a, &b);
    assert_eq!(c.wer, 0.0);
    assert_eq!(c.median_start_diff, 0.0);
}

#[test]
fn an_empty_reference_counts_every_word_as_an_error() {
    let c = compare_words(&[], &[w("bir", 0.0)]);
    assert_eq!(c.wer, 1.0);
    let c = compare_words(&[], &[]);
    assert_eq!(c.wer, 0.0);
}
