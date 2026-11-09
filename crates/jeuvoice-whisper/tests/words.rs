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
