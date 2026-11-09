use jeuvoice_whisper::tokens_to_words;

#[test]
fn tokens_join_into_words() {
    let tokens = vec![
        (" Mer".to_string(), 0.0, 0.2, 0.9),
        ("haba".to_string(), 0.2, 0.5, 0.8),
        (" dünya".to_string(), 0.6, 1.0, 0.95),
    ];
    let words = tokens_to_words(&tokens);
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].text, "Merhaba");
    assert_eq!(words[0].start, 0.0);
    assert_eq!(words[0].end, 0.5);
    assert_eq!(words[0].prob, 0.8);
    assert_eq!(words[1].text, "dünya");
    assert_eq!(words[1].start, 0.6);
    assert_eq!(words[1].end, 1.0);
    assert_eq!(words[1].prob, 0.95);
}

#[test]
fn zero_length_words_get_a_positive_length() {
    let tokens = vec![
        (" O".to_string(), 4.6, 4.6, 0.9),
        (" bilgisayar".to_string(), 4.61, 5.2, 0.9),
        (" daha".to_string(), 7.7, 7.7, 0.9),
    ];
    let words = tokens_to_words(&tokens);
    for w in &words {
        assert!(w.end > w.start, "{w:?}");
    }
    assert_eq!(words[0].end, 4.61, "runs to the next close word");
    assert!(words[0].end <= words[1].start, "no overlap: {words:?}");
    assert!((words[2].end - 7.71).abs() < 1e-9);
}

#[test]
fn punctuation_attaches_to_previous_word() {
    let tokens = vec![
        (" merhaba".to_string(), 0.0, 0.5, 0.9),
        (" ,".to_string(), 0.5, 0.6, 0.8),
        (" dünya".to_string(), 0.7, 1.0, 0.9),
        (".".to_string(), 1.0, 1.1, 0.95),
    ];
    let words = tokens_to_words(&tokens);
    let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts, ["merhaba,", "dünya."]);
    assert_eq!(words[0].end, 0.6);
    assert_eq!(words[0].prob, 0.8);
    assert_eq!(words[1].end, 1.1);
}

#[test]
fn special_tokens_are_skipped() {
    let tokens = vec![
        ("[_BEG_]".to_string(), 0.0, 0.0, 1.0),
        (" Merhaba".to_string(), 0.0, 0.5, 0.9),
        ("[_TT_150]".to_string(), 0.5, 0.5, 1.0),
        (" dünya".to_string(), 0.6, 1.0, 0.9),
    ];
    let words = tokens_to_words(&tokens);
    let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts, ["Merhaba", "dünya"]);
}
