use jeuvoice_core::subtitles::{to_srt, to_txt, to_vtt};
use jeuvoice_core::transcript::{Transcript, Word};

fn w(text: &str, start: f64, end: f64) -> Word {
    Word { text: text.into(), start, end, prob: 0.9 }
}

fn two_sentences() -> Transcript {
    Transcript {
        language: "tr".into(),
        words: vec![
            w("Merhaba.", 0.0, 0.6),
            w("Bugün", 1.5, 1.9),
            w("oyun", 1.9, 2.2),
            w("konuşuyoruz.", 2.2, 3.0),
        ],
    }
}

#[test]
fn srt_of_two_sentences() {
    let expected = "1\n00:00:00,000 --> 00:00:00,600\nMerhaba.\n\n2\n00:00:01,500 --> 00:00:03,000\nBugün oyun konuşuyoruz.\n\n";
    assert_eq!(to_srt(&two_sentences()), expected);
}

#[test]
fn vtt_has_header_and_dot_millis() {
    let v = to_vtt(&two_sentences());
    assert!(v.starts_with("WEBVTT\n\n"));
    assert!(v.contains("00:00:01.500 --> 00:00:03.000"));
    assert!(!v.contains(','));
}

#[test]
fn txt_is_one_line_per_cue() {
    assert_eq!(to_txt(&two_sentences()), "Merhaba.\nBugün oyun konuşuyoruz.\n");
    assert_eq!(to_txt(&Transcript::default()), "");
}

fn secs(t: &str) -> f64 {
    let t = t.replace(',', ".");
    let p: Vec<f64> = t.split(':').map(|x| x.parse().unwrap()).collect();
    p[0] * 3600.0 + p[1] * 60.0 + p[2]
}

#[test]
fn long_text_wraps_at_42_and_splits_cues() {
    let words = (0..30)
        .map(|i| w("abcçde", i as f64 * 0.3, i as f64 * 0.3 + 0.3))
        .collect();
    let t = Transcript { language: "tr".into(), words };
    let srt = to_srt(&t);
    let mut cues = 0;
    for block in srt.trim_end().split("\n\n") {
        cues += 1;
        let lines: Vec<&str> = block.lines().collect();
        let (a, b) = lines[1].split_once(" --> ").unwrap();
        assert!(secs(b) - secs(a) <= 6.0);
        let text = &lines[2..];
        assert!(text.len() <= 2, "too many lines: {block}");
        for l in text {
            assert!(l.chars().count() <= 42, "line too long: {l}");
        }
    }
    assert!(cues >= 3);
}

#[test]
fn gap_starts_new_cue_and_long_word_stands_alone() {
    let long = "a".repeat(50);
    let t = Transcript {
        language: "tr".into(),
        words: vec![w("bir", 0.0, 0.2), w("iki", 0.9, 1.1), w(&long, 1.1, 1.5), w("x", 1.5, 1.6)],
    };
    let txt = to_txt(&t);
    assert_eq!(txt, format!("bir
iki {long}
x
"));
    let srt = to_srt(&t);
    assert!(srt.contains(&format!("iki
{long}

")));
}

#[test]
fn empty_transcript_gives_empty_srt() {
    assert_eq!(to_srt(&Transcript::default()), "");
    assert_eq!(to_vtt(&Transcript::default()), "WEBVTT\n\n");
}
