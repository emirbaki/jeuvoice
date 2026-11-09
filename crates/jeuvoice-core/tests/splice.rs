//! Splice tests: lengths, seamless seams, edge regions and cut points.

use std::collections::HashMap;

use jeuvoice_core::audio::Audio;
use jeuvoice_core::project::{Paragraph, Take, TakePart};
use jeuvoice_core::splice::{cut_points, paragraph_audio, plan_splice, sentence_spans, split_sentences, timeline};
use jeuvoice_core::transcript::Word;

const RATE: u32 = 24_000;

fn word(text: &str, start: f64, end: f64) -> Word {
    Word { text: text.into(), start, end, prob: 0.9 }
}

fn sine(secs: f64, freq: f64, amp: f32) -> Audio {
    let n = (secs * RATE as f64).round() as usize;
    let samples: Vec<f32> = (0..n).map(|i| amp * (2.0 * std::f32::consts::PI * freq as f32 * i as f32 / RATE as f32).sin()).collect();
    Audio { samples, rate: RATE }
}

/// Audio with tone where the words are and silence elsewhere, `total` seconds long.
fn speech_like(words: &[Word], total: f64, freq: f64) -> Audio {
    let n = (total * RATE as f64).round() as usize;
    let mut samples = vec![0.0f32; n];
    for (i, s) in samples.iter_mut().enumerate() {
        let t = i as f64 / RATE as f64;
        if words.iter().any(|w| t >= w.start && t < w.end) {
            *s = 0.4 * (2.0 * std::f32::consts::PI * freq as f32 * i as f32 / RATE as f32).sin();
        }
    }
    Audio { samples, rate: RATE }
}

fn take(id: u32, file: &str) -> Take {
    Take { id, seed: 1, guidance: 2.5, parts: vec![TakePart { index: 1, file: file.into(), language: "tr".into(), text: "".into(), audio_s: 0.0 }], file: file.into() }
}

fn paragraph(text: &str, words: Vec<Word>, file: &str) -> Paragraph {
    Paragraph { index: 0, text: text.into(), takes: vec![take(1, file)], base_take: 0, splices: vec![], words }
}

fn three_sentences() -> (Vec<Word>, &'static str) {
    let words = vec![
        word("Joel", 0.2, 0.6),
        word("giriyor.", 0.7, 1.5),
        word("Ellie", 2.2, 2.6),
        word("bekliyordu.", 2.7, 3.5),
        word("Yağmur", 4.2, 4.6),
        word("durdu.", 4.7, 5.5),
    ];
    (words, "Joel giriyor. Ellie bekliyordu. Yağmur durdu.")
}

#[test]
fn sentences_split_on_marks() {
    assert_eq!(split_sentences("Bir. İki mi? Evet!"), ["Bir.", "İki mi?", "Evet!"]);
    assert_eq!(split_sentences("Tek cümle"), ["Tek cümle"]);
    assert_eq!(split_sentences("  "), Vec::<&str>::new());
}

#[test]
fn sentence_spans_cover_three_sentences() {
    let (words, text) = three_sentences();
    let spans = sentence_spans(&words, text);
    assert_eq!(spans.len(), 3);
    assert_eq!(spans[0], (0.2, 1.5));
    assert_eq!(spans[1], (2.2, 3.5));
    assert_eq!(spans[2], (4.2, 5.5));
}

#[test]
fn splice_in_middle_keeps_length() {
    // Old: 3 sentences x 2 s. New take: the middle sentence takes 3 s.
    let (words, text) = three_sentences();
    let new_words = vec![
        word("Joel", 0.2, 0.6),
        word("giriyor.", 0.7, 1.5),
        word("Ellie", 2.2, 2.9),
        word("bekliyordu.", 3.0, 4.9),
        word("Yağmur", 5.2, 5.6),
        word("durdu.", 5.7, 6.5),
    ];
    let old = paragraph(text, words, "old.wav");
    let mut new_take = take(2, "new.wav");
    new_take.parts[0].text = text.into();
    let old_audio = speech_like(&three_sentences().0, 6.0, 220.0);
    let new_audio = speech_like(&new_words, 7.0, 220.0);
    let sp = plan_splice(&old, &old_audio, &new_take, &new_audio, &new_words, (2.5, 3.5));
    assert_eq!(sp.take, 2);
    // Cuts sit at the start of the silence gaps around the middle sentence.
    assert_eq!(sp.dst, (1.5, 3.5));
    assert_eq!(sp.src, (1.5, 4.9));

    let mut files = HashMap::new();
    files.insert("old.wav".to_string(), old_audio);
    files.insert("new.wav".to_string(), new_audio);
    let mut p = old.clone();
    p.takes.push(new_take);
    p.splices.push(sp);
    let out = paragraph_audio(&p, &|f| files[f].clone());
    // Total length = old - old span + new span.
    assert_eq!(out.samples.len(), 144_000 - 48_000 + 81_600);
}

#[test]
fn crossfade_has_no_jump_at_the_seams() {
    let (words, text) = three_sentences();
    let old = paragraph(text, words.clone(), "old.wav");
    let mut new_take = take(2, "new.wav");
    new_take.parts[0].text = text.into();
    let old_audio = sine(6.0, 220.0, 0.4);
    let new_audio = sine(6.0, 330.0, 0.4);
    let sp = plan_splice(&old, &old_audio, &new_take, &new_audio, &words, (2.5, 3.5));
    let mut files = HashMap::new();
    files.insert("old.wav".to_string(), old_audio);
    files.insert("new.wav".to_string(), new_audio);
    let mut p = old.clone();
    p.takes.push(new_take);
    p.splices.push(sp.clone());
    let out = paragraph_audio(&p, &|f| files[f].clone());
    for seam in [sp.dst.0, sp.dst.0 + (sp.src.1 - sp.src.0)] {
        let c = (seam * RATE as f64).round() as usize;
        let lo = c.saturating_sub(50);
        let hi = (c + 50).min(out.samples.len() - 1);
        let jump = out.samples[lo..=hi].windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(jump <= 0.05, "seam at {seam}s jumps {jump}");
    }
}

#[test]
fn region_at_first_and_last_sentence() {
    let (words, text) = three_sentences();
    let old = paragraph(text, words.clone(), "old.wav");
    let mut new_take = take(2, "new.wav");
    new_take.parts[0].text = text.into();
    let old_audio = speech_like(&words, 6.0, 220.0);
    let new_audio = speech_like(&words, 6.0, 220.0);
    let first = plan_splice(&old, &old_audio, &new_take, &new_audio, &words, (0.0, 0.5));
    assert_eq!(first.dst.0, 0.2, "starts at the first word, not at zero");
    assert_eq!(first.dst.1, 1.5, "ends at the first cut (gap start)");
    let last = plan_splice(&old, &old_audio, &new_take, &new_audio, &words, (5.0, 9.0));
    assert_eq!(last.dst.0, 3.5, "starts at the last cut (gap start)");
    assert_eq!(last.dst.1, 5.5, "ends at the last word, not past it");
}

#[test]
fn region_across_a_paragraph_boundary_gives_two_splices() {
    use jeuvoice_core::project::Project;
    use jeuvoice_core::terms::Terms;

    let (w1, t1) = (vec![word("Birinci", 0.1, 0.9), word("paragraf.", 1.0, 1.8)], "Birinci paragraf.");
    let (w2, t2) = (vec![word("İkinci", 0.1, 0.9), word("paragraf.", 1.0, 1.8)], "İkinci paragraf.");
    let mut p1 = paragraph(t1, w1.clone(), "p1.wav");
    p1.index = 0;
    let mut p2 = paragraph(t2, w2.clone(), "p2.wav");
    p2.index = 1;
    let project = Project {
        version: 1,
        name: "x".into(),
        script: format!("{t1}\n\n{t2}"),
        preset: "s".into(),
        terms: Terms::default(),
        base_seed: 1,
        paragraphs: vec![p1, p2],
    };
    let mut files = HashMap::new();
    files.insert("p1.wav".to_string(), sine(2.0, 220.0, 0.4));
    files.insert("p2.wav".to_string(), sine(2.0, 330.0, 0.4));
    let load = |f: &str| files[f].clone();
    let (_, starts) = timeline(&project, &load, 0.3);
    assert_eq!(starts.len(), 2);
    // A 1 s region over the boundary (para 2 starts at 2.3 s).
    let region = (starts[1].1 - 0.5, starts[1].1 + 0.5);
    let mut splices = Vec::new();
    for (para, &(_, start)) in project.paragraphs.iter().zip(starts.iter()) {
        let end = start + load(&para.takes[0].file).samples.len() as f64 / RATE as f64;
        if region.0 < end && region.1 > start {
            let local = ((region.0 - start).max(0.0), (region.1 - start).max(0.0));
            let audio = load(&para.takes[0].file);
            splices.push(plan_splice(para, &audio, &para.takes[0], &audio, &para.words, local));
        }
    }
    assert_eq!(splices.len(), 2, "one splice per paragraph");
}

#[test]
fn cut_points_pick_the_quiet_window() {
    // 1 s tone, 1 s silence, 1 s tone: the cut in the middle gap is inside the silence.
    let mut samples = sine(1.0, 220.0, 0.4).samples;
    samples.extend(std::iter::repeat_n(0.0, RATE as usize));
    samples.extend(sine(1.0, 220.0, 0.4).samples);
    let audio = Audio { samples, rate: RATE };
    let cuts = cut_points(&audio, &[(0.0, 1.0), (2.0, 3.0)]);
    assert_eq!(cuts.len(), 1);
    assert!((1.0..2.0).contains(&cuts[0]), "cut {} is in the silence", cuts[0]);
}
