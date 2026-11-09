//! SRT, VTT and TXT export. All three share the same cue rules.
use crate::transcript::Transcript;

const MAX_LINE_CHARS: usize = 42;
const MAX_LINES: usize = 2;
const MAX_CUE_SECS: f64 = 6.0;
const GAP_SECS: f64 = 0.7;

struct Cue {
    start: f64,
    end: f64,
    lines: Vec<String>,
}

fn ends_sentence(text: &str) -> bool {
    text.ends_with(['.', '?', '!'])
}

fn build_cues(t: &Transcript) -> Vec<Cue> {
    let mut cues: Vec<Cue> = Vec::new();
    let mut cur: Option<Cue> = None;
    let mut prev_text = "";
    let mut prev_end = 0.0;

    for w in &t.words {
        let mut close = false;
        if let Some(c) = &cur {
            close = ends_sentence(prev_text)
                || w.start - prev_end >= GAP_SECS
                || w.end - c.start > MAX_CUE_SECS;
        }
        if close {
            cues.extend(cur.take());
        }
        if let Some(c) = &mut cur {
            let last = c.lines.last_mut().expect("cue has a line");
            if last.chars().count() + 1 + w.text.chars().count() <= MAX_LINE_CHARS {
                last.push(' ');
                last.push_str(&w.text);
            } else if c.lines.len() < MAX_LINES {
                c.lines.push(w.text.clone());
            } else {
                cues.extend(cur.take());
            }
        }
        match &mut cur {
            Some(c) => c.end = w.end,
            None => {
                cur = Some(Cue { start: w.start, end: w.end, lines: vec![w.text.clone()] });
            }
        }
        prev_text = &w.text;
        prev_end = w.end;
    }
    cues.extend(cur);
    cues
}

fn timestamp(secs: f64, sep: char) -> String {
    let ms = (secs.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{}{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        sep,
        ms % 1000
    )
}

pub fn to_srt(t: &Transcript) -> String {
    let mut out = String::new();
    for (i, c) in build_cues(t).iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            timestamp(c.start, ','),
            timestamp(c.end, ','),
            c.lines.join("\n")
        ));
    }
    out
}

pub fn to_vtt(t: &Transcript) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for c in build_cues(t) {
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            timestamp(c.start, '.'),
            timestamp(c.end, '.'),
            c.lines.join("\n")
        ));
    }
    out
}

/// One line per cue.
pub fn to_txt(t: &Transcript) -> String {
    build_cues(t).iter().map(|c| format!("{}\n", c.lines.join(" "))).collect()
}
