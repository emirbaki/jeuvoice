use jeuvoice_core::transcript::{Transcript, Word};

fn w(text: &str, start: f64, end: f64) -> Word {
    Word { text: text.into(), start, end, prob: 0.9 }
}

#[test]
fn text_joins_words_with_space() {
    let t = Transcript { language: "tr".into(), words: vec![w("Merhaba.", 0.0, 0.6), w("Bugün", 1.0, 1.4)] };
    assert_eq!(t.text(), "Merhaba. Bugün");
}

#[test]
fn set_word_keeps_times() {
    let mut t = Transcript { language: "tr".into(), words: vec![w("Merhaba.", 0.0, 0.6)] };
    t.set_word(0, "Selam.");
    assert_eq!(t.words[0].text, "Selam.");
    assert_eq!(t.words[0].start, 0.0);
    assert_eq!(t.words[0].end, 0.6);
    assert_eq!(t.words[0].prob, 0.9);
}

#[test]
fn json_round_trip() {
    let t = Transcript { language: "tr".into(), words: vec![w("a", 0.0, 0.1)] };
    let s = serde_json::to_string(&t).unwrap();
    assert_eq!(serde_json::from_str::<Transcript>(&s).unwrap(), t);
}
