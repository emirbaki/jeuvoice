//! Paragraph audio from takes and splices, timelines and sentence cut points.
//!
//! A paragraph sounds like its base take with each splice's source span swapped in.
//! Every seam gets a [`CROSSFADE_S`] crossfade. Sentence spans come from the script
//! through [`script_diff`]; cut points sit in the quietest 20 ms window between two
//! sentences, so a splice never clips a word.

use std::collections::HashMap;

use crate::audio::{Audio, resample};
use crate::project::{Paragraph, Project, Splice, Take};
use crate::script_diff::script_diff;
use crate::transcript::{Transcript, Word};

/// Crossfade at both edges of a splice, in seconds.
pub const CROSSFADE_S: f64 = 0.03;

/// The paragraph's audio: the base take with the splices applied in order.
/// `load` reads a take file (`Take.file`).
pub fn paragraph_audio(p: &Paragraph, load: &impl Fn(&str) -> Audio) -> Audio {
    let empty = p.takes.is_empty();
    let base = if empty {
        return Audio { samples: Vec::new(), rate: 24_000 };
    } else {
        p.takes.get(p.base_take).unwrap_or(&p.takes[0])
    };
    let mut cur = load(&base.file);
    let by_id: HashMap<u32, &Take> = p.takes.iter().map(|t| (t.id, t)).collect();
    for sp in &p.splices {
        let Some(take) = by_id.get(&sp.take) else { continue };
        let mut ins = load(&take.file);
        if ins.rate != cur.rate {
            ins = resample(&ins, cur.rate);
        }
        cur = apply_splice(cur, ins, sp);
    }
    cur
}

fn seconds_to_samples(t: f64, rate: u32, len: usize) -> usize {
    ((t.max(0.0)) * f64::from(rate)).round() as usize % (len + 1)
}

/// Swap `sp.dst` of `cur` with `sp.src` of `ins`, then smooth both seams.
fn apply_splice(cur: Audio, ins: Audio, sp: &Splice) -> Audio {
    let rate = cur.rate;
    let (mut d0, mut d1) = (seconds_to_samples(sp.dst.0, rate, cur.samples.len()), seconds_to_samples(sp.dst.1, rate, cur.samples.len()));
    if d0 > d1 {
        std::mem::swap(&mut d0, &mut d1);
    }
    let (mut s0, mut s1) = (seconds_to_samples(sp.src.0, rate, ins.samples.len()), seconds_to_samples(sp.src.1, rate, ins.samples.len()));
    if s0 > s1 {
        std::mem::swap(&mut s0, &mut s1);
    }
    let mut out = Vec::with_capacity(cur.samples.len() - (d1 - d0) + (s1 - s0));
    out.extend_from_slice(&cur.samples[..d0]);
    out.extend_from_slice(&ins.samples[s0..s1]);
    out.extend_from_slice(&cur.samples[d1..]);
    let fade = (CROSSFADE_S * f64::from(rate)).round() as usize;
    blend_seam(&mut out, d0, fade);
    blend_seam(&mut out, d0 + (s1 - s0), fade);
    Audio { samples: out, rate }
}

/// Smooth the seam at `seam` (samples of `out` from stream A meet stream B there).
/// A raised blend over `fade` samples, in place so the length stays exact. Out-of-range
/// seams (empty sides) do nothing.
fn blend_seam(out: &mut [f32], seam: usize, fade: usize) {
    if fade < 2 || out.is_empty() || seam == 0 || seam >= out.len() {
        return;
    }
    let h = fade / 2;
    let start = seam.saturating_sub(h);
    let end = (seam + h).min(out.len());
    if end - start < 2 {
        return;
    }
    let window: Vec<f32> = out[start..end].to_vec();
    let mid = seam - start;
    let left_hold = window[mid.saturating_sub(1)];
    let right_hold = window[mid.min(window.len() - 1)];
    for (k, sample) in out[start..end].iter_mut().enumerate() {
        let t = k as f32 / (window.len() - 1) as f32;
        let a = if k < mid { window[k] } else { left_hold };
        let b = if k >= mid { window[k] } else { right_hold };
        *sample = a * (1.0 - t) + b * t;
    }
}

/// The project's timeline: paragraphs joined with `para_pause` seconds of silence.
/// Returns the audio and each paragraph's start time in seconds.
pub fn timeline(project: &Project, load: &impl Fn(&str) -> Audio, para_pause: f64) -> (Audio, Vec<(usize, f64)>) {
    let mut out = Vec::new();
    let mut starts = Vec::new();
    let mut rate = 24_000u32;
    let mut first = true;
    for p in &project.paragraphs {
        let mut a = paragraph_audio(p, load);
        if first {
            rate = a.rate;
            first = false;
        } else if a.rate != rate {
            a = resample(&a, rate);
        }
        starts.push((p.index, out.len() as f64 / f64::from(rate)));
        out.extend_from_slice(&a.samples);
        if p.index != project.paragraphs.last().map(|l| l.index).unwrap_or(p.index) {
            let gap = (para_pause.max(0.0) * f64::from(rate)).round() as usize;
            out.extend(std::iter::repeat_n(0.0, gap));
        }
    }
    (Audio { samples: out, rate }, starts)
}

/// Split `paragraph_text` on `.`, `?`, `!` (the mark stays with its sentence).
pub fn split_sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if matches!(bytes[i], b'.' | b'?' | b'!') {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j >= bytes.len() {
                // The end mark ends the text: it belongs to this sentence.
                out.push(text[start..].trim());
                return out;
            }
            // A mark inside the text ends a sentence only before whitespace.
            if j > i + 1 {
                out.push(text[start..j].trim());
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }
    if !text[start..].trim().is_empty() {
        out.push(text[start..].trim());
    }
    out
}

/// One span per script sentence, in word times. Sentences map to words through
/// [`script_diff`]: a span runs from the first to the last heard word of the sentence's
/// script words. A sentence with no heard words takes the gap around it.
pub fn sentence_spans(words: &[Word], paragraph_text: &str) -> Vec<(f64, f64)> {
    let sentences = split_sentences(paragraph_text);
    if sentences.is_empty() || words.is_empty() {
        return Vec::new();
    }
    // Script word index ranges, one per sentence.
    let mut ranges = Vec::new();
    let mut at = 0;
    for s in &sentences {
        let n = s.split_whitespace().count();
        ranges.push((at, at + n));
        at += n;
    }
    let t = Transcript { language: String::new(), words: words.to_vec() };
    let ops = script_diff(paragraph_text, &t);
    // Heard word per script word, from Same/Changed ops.
    let mut heard_of: HashMap<usize, usize> = HashMap::new();
    for op in &ops {
        match *op {
            crate::script_diff::DiffOp::Same { script, heard } | crate::script_diff::DiffOp::Changed { script, heard } => {
                heard_of.entry(script).or_insert(heard);
            }
            _ => {}
        }
    }
    let mut spans = Vec::new();
    let mut prev_end = words.first().map(|w| w.start).unwrap_or(0.0);
    // The script word count bounds the "next heard word" search below.
    let script_len = ranges.iter().map(|&(_, hi)| hi).max().unwrap_or(0);
    for (lo, hi) in ranges {
        let mut heard: Vec<usize> = (lo..hi).filter_map(|s| heard_of.get(&s).copied()).collect();
        heard.sort_unstable();
        heard.dedup();
        let span = match (heard.first(), heard.last()) {
            (Some(&a), Some(&b)) => (words[a].start, words[b].end),
            // No heard words: the gap between the neighbours.
            _ => {
                let next_start = (hi..script_len)
                    .filter_map(|s| heard_of.get(&s))
                    .map(|&h| words[h].start)
                    .next()
                    .unwrap_or_else(|| words.last().map(|w| w.end).unwrap_or(prev_end));
                (prev_end, next_start)
            }
        };
        prev_end = span.1.max(prev_end);
        spans.push(span);
    }
    spans
}

/// Cut points between spans: the start of the quietest 20 ms window inside each gap.
/// One cut per gap, in seconds.
pub fn cut_points(audio: &Audio, spans: &[(f64, f64)]) -> Vec<f64> {
    let rate = audio.rate.max(1) as f64;
    let win = ((0.02 * rate).round() as usize).max(1);
    let mut cuts = Vec::new();
    for pair in spans.windows(2) {
        let (g0, g1) = (pair[0].1, pair[1].0);
        let (mut a, b) = ((g0 * rate).round() as usize, ((g1 * rate).round() as usize).min(audio.samples.len()));
        a = a.min(audio.samples.len());
        if b <= a {
            cuts.push(g0);
            continue;
        }
        if b - a <= win {
            cuts.push((g0 + g1) / 2.0);
            continue;
        }
        let mut best = a;
        let mut best_rms = f64::MAX;
        for s in a..=(b - win) {
            let rms: f64 = audio.samples[s..s + win].iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / win as f64;
            if rms < best_rms {
                best_rms = rms;
                best = s;
            }
        }
        cuts.push(best as f64 / rate);
    }
    cuts
}

/// Plan the splice for `region` (seconds, in the old paragraph timeline): cover every
/// sentence that overlaps it. Source and destination use the cut points of the same
/// sentence indices, so no word is clipped.
pub fn plan_splice(
    old: &Paragraph,
    old_audio: &Audio,
    new_take: &Take,
    new_audio: &Audio,
    new_words: &[Word],
    region: (f64, f64),
) -> Splice {
    let (mut r0, mut r1) = region;
    if r0 > r1 {
        std::mem::swap(&mut r0, &mut r1);
    }
    let old_spans = sentence_spans(&old.words, &old.text);
    let new_text = new_take.parts.iter().map(|p| p.text.as_str()).collect::<Vec<_>>().join(" ");
    let new_text = if new_text.trim().is_empty() { old.text.clone() } else { new_text };
    let new_spans = sentence_spans(new_words, &new_text);
    if old_spans.is_empty() || new_spans.is_empty() {
        return Splice { take: new_take.id, src: (0.0, 0.0), dst: (r0, r1) };
    }
    // Sentences that overlap the region.
    let mut hit: Vec<usize> = old_spans
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let (s0, s1) = old_spans[*i];
            s0 < r1 && s1 > r0
        })
        .map(|(i, _)| i)
        .collect();
    if hit.is_empty() {
        // In a gap: take the sentence whose center is nearest the region center.
        let mid = (r0 + r1) / 2.0;
        let nearest = old_spans
            .iter()
            .enumerate()
            .min_by(|a, b| {
                let (a0, a1) = old_spans[a.0];
                let (b0, b1) = old_spans[b.0];
                ((a0 + a1) / 2.0 - mid).abs().partial_cmp(&((b0 + b1) / 2.0 - mid).abs()).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        hit.push(nearest);
    }
    let (first, last) = (*hit.first().unwrap(), *hit.last().unwrap());
    let dst = cut_span(&old_spans, &cut_points(old_audio, &old_spans), first, last);
    let (nfirst, nlast) = (first.min(new_spans.len() - 1), last.min(new_spans.len() - 1));
    let src = cut_span(&new_spans, &cut_points(new_audio, &new_spans), nfirst, nlast);
    Splice { take: new_take.id, src, dst }
}

/// Span of sentences `first..=last`, snapped to the cut points at its edges.
fn cut_span(spans: &[(f64, f64)], cuts: &[f64], first: usize, last: usize) -> (f64, f64) {
    let start = if first == 0 { spans[first].0 } else { cuts.get(first - 1).copied().unwrap_or(spans[first].0) };
    let end = if last + 1 >= spans.len() { spans[last].1 } else { cuts.get(last).copied().unwrap_or(spans[last].1) };
    (start.min(end), end.max(start))
}
