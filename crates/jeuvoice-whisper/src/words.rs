use jeuvoice_core::transcript::Word;

/// Shortest word length in seconds, used when whisper gives a zero-length span.
const MIN_WORD_SECS: f64 = 0.01;
/// A zero-length word runs up to the next word when the next word starts this close.
const MAX_FILL_SECS: f64 = 0.5;

/// Join whisper tokens `(text, start, end, prob)` into words. A token that starts with a
/// space starts a new word, unless it is only punctuation: that attaches to the previous
/// word. Special tokens (`[_BEG_]`, `[_TT_150]`) are skipped. Times are in seconds. The
/// word prob is the lowest token prob in the word. Every word has `end > start`, and a
/// word does not run past the start of the next word.
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
        let punct_only = piece.chars().all(|c| !c.is_alphanumeric());
        let joins = !words.is_empty() && (punct_only || (open && !text.starts_with(' ')));
        if joins {
            let w = words.last_mut().expect("a previous word exists");
            w.text.push_str(piece);
            w.end = w.end.max(*end);
            w.prob = w.prob.min(*prob);
        } else {
            words.push(Word { text: piece.to_string(), start: *start, end: *end, prob: *prob });
        }
        open = true;
    }
    fix_lengths(&mut words);
    words
}

fn fix_lengths(words: &mut [Word]) {
    for i in 0..words.len() {
        let next_start = words.get(i + 1).map(|n| n.start);
        let w = &mut words[i];
        if w.end <= w.start {
            w.end = match next_start {
                // Run to the next word when it is close, else keep the word short.
                Some(n) if n > w.start && n - w.start <= MAX_FILL_SECS => n,
                Some(n) if n > w.start => n.min(w.start + MIN_WORD_SECS),
                _ => w.start + MIN_WORD_SECS,
            };
        }
    }
}
