use jeuvoice_core::transcript::Word;

/// Join whisper tokens `(text, start, end, prob)` into words. A token that starts with a
/// space starts a new word. Special tokens (`[_BEG_]`, `[_TT_150]`) are skipped.
/// Times are in seconds. The word prob is the lowest token prob in the word.
pub fn tokens_to_words(tokens: &[(String, f64, f64, f32)]) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut open = false;
    for (text, start, end, prob) in tokens {
        if text.starts_with("[_") {
            continue;
        }
        let piece = text.trim();
        if piece.is_empty() {
            // A bare space ends the current word.
            if !text.is_empty() {
                open = false;
            }
            continue;
        }
        if open && !text.starts_with(' ') {
            let w = words.last_mut().expect("open word exists");
            w.text.push_str(piece);
            w.end = *end;
            w.prob = w.prob.min(*prob);
        } else {
            words.push(Word { text: piece.to_string(), start: *start, end: *end, prob: *prob });
            open = true;
        }
    }
    words
}
