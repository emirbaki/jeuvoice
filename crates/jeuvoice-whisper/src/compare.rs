//! Compares two transcripts for the Phase 1 criteria: the word error rate (WER) and the
//! word-start difference. For the measurement tool (`examples/compare_whisper.rs`), not
//! for the app.

use jeuvoice_core::transcript::Word;

/// A word for the comparison: the Turkish lower case (`İ` to `i`, `I` to `ı`, then the
/// normal lower case), with every character that is not a letter or a digit removed.
pub fn normalize(word: &str) -> String {
    let mut s = String::new();
    for c in word.chars() {
        match c {
            'İ' => s.push('i'),
            'I' => s.push('ı'),
            c => s.extend(c.to_lowercase()),
        }
    }
    s.chars().filter(|c| c.is_alphanumeric()).collect()
}

/// One step of the alignment of `reference` and `hypothesis`, with the word indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Match(usize, usize),
    Sub(usize, usize),
    /// A reference word that the hypothesis does not have.
    Del(usize),
    /// A hypothesis word that the reference does not have.
    Ins(usize),
}

/// The cheapest edit path (Levenshtein, each edit costs 1) from `reference` to
/// `hypothesis`. On a tie it takes the diagonal step first, then a deletion.
pub fn align(reference: &[String], hypothesis: &[String]) -> Vec<Op> {
    let (n, m) = (reference.len(), hypothesis.len());
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for j in 0..=m {
        d[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = usize::from(reference[i - 1] != hypothesis[j - 1]);
            d[i][j] = (d[i - 1][j - 1] + sub).min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
        }
    }
    let mut ops = Vec::with_capacity(n.max(m));
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let same = reference[i - 1] == hypothesis[j - 1];
            if d[i][j] == d[i - 1][j - 1] + usize::from(!same) {
                ops.push(if same { Op::Match(i - 1, j - 1) } else { Op::Sub(i - 1, j - 1) });
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && d[i][j] == d[i - 1][j] + 1 {
            ops.push(Op::Del(i - 1));
            i -= 1;
        } else {
            ops.push(Op::Ins(j - 1));
            j -= 1;
        }
    }
    ops.reverse();
    ops
}

/// The result of `compare_words`.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    pub reference_words: usize,
    pub hypothesis_words: usize,
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
    /// `(substitutions + deletions + insertions) / reference words`.
    pub wer: f64,
    /// The words that the alignment matches.
    pub matched: usize,
    /// The median of `|start difference|` over the matched words, in seconds (the element
    /// at index `len / 2` of the sorted values).
    pub median_start_diff: f64,
}

/// Compares `hypothesis` with `reference` after `normalize`. Words that normalize to
/// nothing (only punctuation) are left out.
pub fn compare_words(reference: &[Word], hypothesis: &[Word]) -> Comparison {
    let keep = |ws: &[Word]| -> Vec<(String, f64)> { ws.iter().map(|w| (normalize(&w.text), w.start)).filter(|(t, _)| !t.is_empty()).collect() };
    let (r, h) = (keep(reference), keep(hypothesis));
    let rt: Vec<String> = r.iter().map(|(t, _)| t.clone()).collect();
    let ht: Vec<String> = h.iter().map(|(t, _)| t.clone()).collect();
    let ops = align(&rt, &ht);
    let (mut s, mut d, mut ins) = (0, 0, 0);
    let mut diffs = Vec::new();
    for op in &ops {
        match *op {
            Op::Match(i, j) => diffs.push((h[j].1 - r[i].1).abs()),
            Op::Sub(..) => s += 1,
            Op::Del(_) => d += 1,
            Op::Ins(_) => ins += 1,
        }
    }
    let errors = s + d + ins;
    let wer = if r.is_empty() { if errors == 0 { 0.0 } else { 1.0 } } else { errors as f64 / r.len() as f64 };
    diffs.sort_by(|a, b| a.total_cmp(b));
    let median_start_diff = diffs.get(diffs.len() / 2).copied().unwrap_or(0.0);
    Comparison {
        reference_words: r.len(),
        hypothesis_words: h.len(),
        substitutions: s,
        deletions: d,
        insertions: ins,
        wer,
        matched: diffs.len(),
        median_start_diff,
    }
}
