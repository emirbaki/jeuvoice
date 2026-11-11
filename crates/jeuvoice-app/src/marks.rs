//! Turn the script check into marks on the heard words, counts, and seek times. Pure functions.

use jeuvoice_core::script_diff::DiffOp;
use jeuvoice_core::transcript::Word;

use crate::waveform::Mark;

/// Words with a probability under this get a `LowProb` mark.
pub const LOW_PROB: f32 = 0.5;

/// The word index a diff op points at on the waveform. `Added` and `Changed` point at their
/// heard word. `Missing` points at the next heard word in diff order, or at `words_len` when
/// no heard word follows (the speaker stopped early). `Same` points nowhere.
pub fn target(diff: &[DiffOp], k: usize, words_len: usize) -> Option<usize> {
    match diff.get(k)? {
        DiffOp::Same { .. } => None,
        DiffOp::Added { heard } | DiffOp::Changed { heard, .. } => Some(*heard),
        DiffOp::Missing { .. } => Some(
            diff[k + 1..]
                .iter()
                .find_map(|op| match op {
                    DiffOp::Same { heard, .. } | DiffOp::Added { heard } | DiffOp::Changed { heard, .. } => Some(*heard),
                    DiffOp::Missing { .. } => None,
                })
                .unwrap_or(words_len),
        ),
    }
}

/// Marks for the waveform: one per difference, plus `LowProb` for every unsure word.
pub fn marks(diff: &[DiffOp], words: &[Word]) -> Vec<(usize, Mark)> {
    let mut out = Vec::new();
    for (k, op) in diff.iter().enumerate() {
        let mark = match op {
            DiffOp::Same { .. } => continue,
            DiffOp::Missing { .. } => Mark::Missing,
            DiffOp::Added { .. } => Mark::Added,
            DiffOp::Changed { .. } => Mark::Changed,
        };
        if let Some(i) = target(diff, k, words.len()) {
            out.push((i, mark));
        }
    }
    out.extend(words.iter().enumerate().filter(|(_, w)| w.prob < LOW_PROB).map(|(i, _)| (i, Mark::LowProb)));
    out
}

/// The time in seconds that a click on difference `k` moves the playhead to: the start of the
/// word its mark is on, or the end of the last word for a Missing mark after the last word.
pub fn seek_time(diff: &[DiffOp], k: usize, words: &[Word]) -> Option<f64> {
    let i = target(diff, k, words.len())?;
    match words.get(i) {
        Some(w) => Some(w.start),
        None => Some(words.last().map_or(0.0, |w| w.end)),
    }
}

/// "3 missing, 1 added, 2 changed".
pub fn counts_text(diff: &[DiffOp]) -> String {
    let (mut missing, mut added, mut changed) = (0, 0, 0);
    for op in diff {
        match op {
            DiffOp::Same { .. } => {}
            DiffOp::Missing { .. } => missing += 1,
            DiffOp::Added { .. } => added += 1,
            DiffOp::Changed { .. } => changed += 1,
        }
    }
    crate::i18n::trf!("{missing} missing, {added} added, {changed} changed", "{missing} eksik, {added} fazla, {changed} farklı")
}

#[cfg(test)]
mod tests {
    use super::*;
    use DiffOp::*;

    fn w(text: &str, start: f64, end: f64, prob: f32) -> Word {
        Word { text: text.into(), start, end, prob }
    }

    fn words() -> Vec<Word> {
        vec![w("Joel", 0.2, 0.6, 0.9), w("Elli", 0.8, 1.3, 0.3), w("uh", 1.5, 1.7, 0.9)]
    }

    #[test]
    fn each_difference_marks_a_heard_word() {
        let diff = [Same { script: 0, heard: 0 }, Missing { script: 1 }, Changed { script: 2, heard: 1 }, Added { heard: 2 }];
        let m = marks(&diff, &words());
        assert_eq!(m, vec![(1, Mark::Missing), (1, Mark::Changed), (2, Mark::Added), (1, Mark::LowProb)]);
    }

    #[test]
    fn missing_after_the_last_heard_word_marks_the_end() {
        let diff = [Same { script: 0, heard: 0 }, Same { script: 1, heard: 1 }, Same { script: 2, heard: 2 }, Missing { script: 3 }, Missing { script: 4 }];
        assert_eq!(target(&diff, 3, 3), Some(3));
        assert_eq!(target(&diff, 4, 3), Some(3));
        assert_eq!(seek_time(&diff, 3, &words()), Some(1.7));
    }

    #[test]
    fn a_run_of_missing_words_marks_the_next_heard_word() {
        let diff = [Missing { script: 0 }, Missing { script: 1 }, Same { script: 2, heard: 0 }];
        assert_eq!(target(&diff, 0, 3), Some(0));
        assert_eq!(target(&diff, 1, 3), Some(0));
        assert_eq!(seek_time(&diff, 0, &words()), Some(0.2));
        assert_eq!(target(&diff, 2, 3), None, "a Same op has no mark");
    }

    #[test]
    fn counts_name_each_kind() {
        let diff = [Missing { script: 0 }, Missing { script: 1 }, Added { heard: 0 }, Changed { script: 2, heard: 1 }];
        assert_eq!(counts_text(&diff), "2 missing, 1 added, 1 changed");
    }
}
