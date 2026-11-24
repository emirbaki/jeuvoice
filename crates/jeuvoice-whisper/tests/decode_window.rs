//! The window loop of `whisper_full_with_state`, with a scripted model in place of the GPU.

use std::collections::HashMap;

use jeuvoice_whisper::decode::rules::log_softmax;
use jeuvoice_whisper::decode::vocab::{lang_id, Vocab};
use jeuvoice_whisper::decode::window::{transcribe_tokens, DecodeParams, Transcription, WhisperModel};
use jeuvoice_whisper::WhisperError;

fn vocab() -> Vocab {
    let words = (0..50257)
        .map(|i| match i {
            220 => b" ".to_vec(),
            50256 => Vec::new(),
            _ => format!("w{i}").into_bytes(),
        })
        .collect();
    Vocab::new(51866, words).unwrap()
}

/// A logit of `STRONG` gives a prob of almost 1 (logprob about -5e-9), also after the
/// division by a temperature of 1.0.
const STRONG: f32 = 30.0;
/// A logit of `WEAK` gives a prob of about 0.3 (logprob about -1.2, below the -1.0 limit).
const WEAK: f32 = 10.0;

#[derive(Debug, PartialEq)]
enum Call {
    Encode(usize),
    First(Vec<u32>),
    Next(u32),
    Align(Vec<u32>, usize),
}

/// Gives the logits row for the next token of a script. The script of a window is chosen by
/// the `seek` of the last `encode` and the number of `decode_first` calls at that seek (the
/// temperature attempt).
struct Scripted {
    n_vocab: usize,
    eot: u32,
    seek: usize,
    attempt: HashMap<usize, usize>,
    generated: usize,
    calls: Vec<Call>,
    /// `(seek, attempt) -> (tokens, logit)`. A missing attempt uses the last one given.
    scripts: HashMap<(usize, usize), (Vec<u32>, f32)>,
    /// The row for the language detection decode (`decode_first(&[SOT])`).
    detect_row: Option<Vec<f32>>,
    sot: u32,
    /// `seek -> audio bounds` of the alignment blocks. A missing seek splits the audio
    /// into equal blocks.
    align_bounds: HashMap<usize, Vec<usize>>,
}

impl Scripted {
    fn new(v: &Vocab) -> Scripted {
        Scripted {
            n_vocab: v.n_vocab(),
            eot: v.eot,
            seek: 0,
            attempt: HashMap::new(),
            generated: 0,
            calls: Vec::new(),
            scripts: HashMap::new(),
            detect_row: None,
            sot: v.sot,
            align_bounds: HashMap::new(),
        }
    }

    fn script(mut self, seek: usize, attempt: usize, tokens: Vec<u32>, logit: f32) -> Scripted {
        self.scripts.insert((seek, attempt), (tokens, logit));
        self
    }

    fn row(&self) -> Vec<f32> {
        let a = self.attempt[&self.seek] - 1;
        let (tokens, logit) = (0..=a)
            .rev()
            .find_map(|k| self.scripts.get(&(self.seek, k)))
            .unwrap_or_else(|| panic!("no script for seek {}", self.seek));
        let id = tokens.get(self.generated).copied().unwrap_or(self.eot);
        let mut r = vec![0.0; self.n_vocab];
        r[id as usize] = *logit;
        r
    }
}

impl WhisperModel for Scripted {
    fn encode(&mut self, seek: usize) -> Result<(), WhisperError> {
        self.calls.push(Call::Encode(seek));
        self.seek = seek;
        Ok(())
    }

    fn decode_first(&mut self, tokens: &[u32]) -> Result<Vec<f32>, WhisperError> {
        self.calls.push(Call::First(tokens.to_vec()));
        if tokens == [self.sot] {
            return Ok(self.detect_row.clone().expect("a detect row"));
        }
        *self.attempt.entry(self.seek).or_default() += 1;
        self.generated = 0;
        Ok(self.row())
    }

    fn decode_next(&mut self, token: u32) -> Result<Vec<f32>, WhisperError> {
        self.calls.push(Call::Next(token));
        self.generated += 1;
        Ok(self.row())
    }

    /// Blocks: row 2 + k (NOT, then the text tokens) has 1.0 in block k of the audio, and
    /// 0.0 elsewhere. Rows SOT, lang and EOT are 0.0. The same for the 6 heads.
    fn alignment(&mut self, tokens: &[u32], n_audio: usize) -> Result<Vec<f32>, WhisperError> {
        self.calls.push(Call::Align(tokens.to_vec(), n_audio));
        let rows = tokens.len() - 3;
        let bounds = self.align_bounds.get(&self.seek).cloned().unwrap_or_else(|| (1..rows).map(|k| k * n_audio / rows).collect());
        assert_eq!(bounds.len(), rows - 1);
        let edges: Vec<usize> = std::iter::once(0).chain(bounds).chain(std::iter::once(n_audio)).collect();
        let n_tok = tokens.len();
        let mut w = vec![0.0f32; 6 * n_tok * n_audio];
        for h in 0..6 {
            for k in 0..rows {
                for a in edges[k]..edges[k + 1] {
                    w[(h * n_tok + 2 + k) * n_audio + a] = 1.0;
                }
            }
        }
        Ok(w)
    }
}

fn params(language: &str) -> DecodeParams {
    DecodeParams { language: language.into(), ..DecodeParams::default() }
}

fn run(m: &mut Scripted, n_samples: usize, language: &str) -> (Transcription, Vec<f32>) {
    let v = vocab();
    let mut progress = Vec::new();
    let samples = vec![0.0; n_samples];
    let t = transcribe_tokens(m, &v, &samples, &params(language), &mut |p| progress.push(p)).unwrap();
    (t, progress)
}

fn ids(s: &jeuvoice_whisper::decode::window::Segment) -> Vec<u32> {
    s.tokens.iter().map(|t| t.id).collect()
}

/// 45 s of audio: `seek_end = 1 + (720000 - 200) / 160 = 4499` centiseconds.
const N45: usize = 720_000;
/// 8 s of audio: `seek_end = 1 + (128000 - 200) / 160 = 799`.
const N8: usize = 128_000;

#[test]
fn three_windows_with_segments_past_prompt_and_progress() {
    let v = vocab();
    let ts = |k: u32| v.beg + k;
    let (w10, w11, w12, w13, w14) = (10, 11, 12, 13, 14);
    let mut m = Scripted::new(&v)
        // Window 1: two segments, 0 to 20 s and 20 to 28 s, then a pair of timestamps.
        .script(0, 0, vec![v.beg, w10, w11, ts(1000), ts(1000), w12, ts(1400), ts(1400), v.eot], STRONG)
        // Window 2 at 28 s: one segment of 10 s.
        .script(2800, 0, vec![v.beg, w13, ts(500), ts(500), v.eot], STRONG)
        // Window 3 at 38 s: the end of the audio ends the window at the timestamp.
        .script(3800, 0, vec![v.beg, w14, ts(345)], STRONG);
    let (t, progress) = run(&mut m, N45, "tr");

    assert_eq!(t.language, "tr");
    let seg: Vec<(i64, i64, Vec<u32>)> = t.segments.iter().map(|s| (s.t0, s.t1, ids(s))).collect();
    assert_eq!(
        seg,
        vec![
            (0, 2000, vec![v.beg, w10, w11, ts(1000)]),
            (2000, 2800, vec![w12, ts(1400)]),
            (2800, 3800, vec![v.beg, w13, ts(500)]),
            (3800, 4490, vec![v.beg, w14, ts(345)]),
        ]
    );
    assert_eq!(t.segments[0].text, b"w10w11");
    // The token times ran on each segment: the first token of the window starts at seek.
    assert_eq!(t.segments[0].tokens[0].t0, 0);
    assert_eq!(t.segments[2].tokens[0].t0, 2800);
    assert!(t.segments.iter().flat_map(|s| &s.tokens).all(|t| t.t0 >= 0 && t.t1 >= 0));
    let seeks: Vec<usize> = t.windows.iter().map(|w| w.seek).collect();
    assert_eq!(seeks, vec![0, 2800, 3800]);
    assert!(t.windows.iter().all(|w| w.attempts == 1 && w.temperature == 0.0));

    // The prompt of window 2 carries the tokens of window 1, and the prompt of window 3 the
    // tokens of windows 1 and 2 (3800 + 500 is below 4499, so the past is kept).
    let tr = v.lang_token(lang_id("tr").unwrap());
    let init = vec![v.sot, tr, v.transcribe];
    let firsts: Vec<&Vec<u32>> = m.calls.iter().filter_map(|c| if let Call::First(p) = c { Some(p) } else { None }).collect();
    assert_eq!(firsts[0], &init);
    let mut p2 = vec![v.prev, v.beg, w10, w11, ts(1000), ts(1000), w12, ts(1400), ts(1400)];
    p2.extend(&init);
    assert_eq!(firsts[1], &p2);
    let mut p3 = p2[..p2.len() - 3].to_vec();
    p3.extend([v.beg, w13, ts(500), ts(500)]);
    p3.extend(&init);
    assert_eq!(firsts[2], &p3);

    // Progress at the top of each loop: 100 × seek / seek_end in whole percent. Window 3
    // ends with text and one timestamp, so seek moves to seek_end (100 %).
    assert_eq!(progress, vec![0.0, 0.62, 0.84, 1.0]);
}

#[test]
fn dtw_gives_the_text_tokens_their_start_times() {
    let v = vocab();
    // Window 1: text tokens 10 and 11, seek_delta 200, so the DTW reads 100 audio positions.
    // NOT has audio 0..30, token 10 has 30..60, token 11 has 60..100.
    let mut m = Scripted::new(&v)
        .script(0, 0, vec![v.beg, 10, 11, v.beg + 100, v.beg + 100, v.eot], STRONG)
        .script(200, 0, vec![v.beg, 12, v.beg + 295], STRONG);
    m.align_bounds.insert(0, vec![30, 60]);
    let (t, _) = run(&mut m, N8, "tr");
    let tr = v.lang_token(lang_id("tr").unwrap());
    let aligns: Vec<&Call> = m.calls.iter().filter(|c| matches!(c, Call::Align(..))).collect();
    assert_eq!(aligns[0], &Call::Align(vec![v.sot, tr, v.not, 10, 11, v.eot], 100));
    let dtw: Vec<i64> = t.segments[0].tokens.iter().map(|t| t.t_dtw).collect();
    // 2 × audio index + seek. BEG and the timestamp get no DTW time.
    assert_eq!(dtw, vec![-1, 60, 120, -1]);
    // Window 2 at seek 200: n_frames = min(3000, seek_delta 590, 799 - 200) = 590, so 295
    // audio positions. Equal blocks: NOT has 0..147, token 12 starts at 147.
    assert_eq!(aligns[1], &Call::Align(vec![v.sot, tr, v.not, 12, v.eot], 295));
    assert_eq!(t.segments[1].tokens[1].t_dtw, 200 + 2 * 147);
}

#[test]
fn past_is_cleared_in_the_last_5_seconds() {
    let v = vocab();
    // 35 s: seek_end = 1 + (560000 - 200) / 160 = 3499. Window 2 starts at 3000, and
    // 3000 + 500 >= 3499, so its prompt has no past tokens.
    let mut m = Scripted::new(&v)
        .script(0, 0, vec![v.beg, 10, v.beg + 1500, v.beg + 1500, v.eot], STRONG)
        .script(3000, 0, vec![v.beg, 11, v.beg + 200], STRONG);
    let (t, _) = run(&mut m, 560_000, "tr");
    assert_eq!(t.windows.len(), 2);
    let tr = v.lang_token(lang_id("tr").unwrap());
    let firsts: Vec<&Vec<u32>> = m.calls.iter().filter_map(|c| if let Call::First(p) = c { Some(p) } else { None }).collect();
    assert_eq!(firsts[1], &vec![v.sot, tr, v.transcribe]);
}

#[test]
fn single_timestamp_ending_skips_the_window() {
    let v = vocab();
    // Text, then one timestamp, then EOT: whisper.cpp moves on by a full 30 s window.
    let mut m = Scripted::new(&v)
        .script(0, 0, vec![v.beg, 10, v.beg + 500, v.eot], STRONG)
        .script(3000, 0, vec![v.beg, 11, v.beg + 700], STRONG);
    let (t, _) = run(&mut m, N45, "tr");
    let seeks: Vec<usize> = t.windows.iter().map(|w| w.seek).collect();
    assert_eq!(seeks, vec![0, 3000]);
    assert_eq!((t.segments[0].t0, t.segments[0].t1), (0, 1000));
}

#[test]
fn low_logprob_falls_back_to_the_next_temperature() {
    let v = vocab();
    // Every token of this script has a logprob of about -1.2. (The second timestamp of a
    // pair has few open tokens, so it gets about -0.06 and lifts the mean above -1.0.)
    let script = vec![v.beg, 10, 11, v.beg + 395];
    let mut m = Scripted::new(&v).script(0, 0, script, WEAK).script(0, 1, vec![v.beg, 11, v.beg + 395], STRONG);
    let (t, _) = run(&mut m, N8, "tr");
    assert_eq!(t.windows.len(), 1);
    assert_eq!(t.windows[0].attempts, 2);
    assert!((t.windows[0].temperature - 0.2).abs() < 1e-6);
    assert_eq!(ids(&t.segments[0]), vec![v.beg, 11, v.beg + 395]);
}

#[test]
fn low_entropy_fails_every_temperature_and_keeps_the_last() {
    let v = vocab();
    // 40 times the same token: entropy of the last 32 tokens is far below 2.4.
    let mut script = vec![v.beg];
    script.extend(std::iter::repeat(10).take(40));
    script.push(v.beg + 395);
    let mut m = Scripted::new(&v).script(0, 0, script, STRONG);
    let (t, _) = run(&mut m, N8, "tr");
    let w = &t.windows[0];
    assert_eq!(w.attempts, 6, "temperatures 0.0, 0.2, 0.4, 0.6, 0.8, 1.0");
    assert!((w.temperature - 1.0).abs() < 1e-5);
    assert_eq!(t.segments.len(), 1);
    assert_eq!(t.segments[0].tokens.len(), 42);
}

#[test]
fn repetition_without_timestamps_fails_and_keeps_all_tokens() {
    let v = vocab();
    // No timestamp in 220 steps: each temperature fails. The last one is kept with all its
    // tokens, as one segment from 0 to seek + 3000.
    let script: Vec<u32> = std::iter::once(v.beg).chain((0..300).map(|i| 10 + (i % 2))).collect();
    let mut m = Scripted::new(&v).script(0, 0, script, STRONG).script(3000, 0, vec![v.beg, 11, v.beg + 700], STRONG);
    let (t, _) = run(&mut m, N45, "tr");
    assert_eq!(t.windows[0].attempts, 6);
    assert_eq!(t.segments[0].tokens.len(), 220);
    assert_eq!((t.segments[0].t0, t.segments[0].t1), (0, 3000));
}

#[test]
fn no_speech_window_after_language_detection() {
    let v = vocab();
    // The detection row: Turkish wins among the languages, and NOSP is very likely. That row
    // stays in row 0 of whisper.cpp's logits buffer, so the window reads its no-speech prob
    // from it. With a weak decode the window is no-speech: no segments.
    let mut detect = vec![0.0; v.n_vocab()];
    detect[v.lang_token(lang_id("tr").unwrap()) as usize] = 5.0;
    detect[v.lang_token(lang_id("de").unwrap()) as usize] = 4.0;
    detect[v.nosp as usize] = 30.0;
    let mut m = Scripted::new(&v).script(0, 0, vec![v.beg, 10, v.beg + 395], WEAK);
    m.detect_row = Some(detect);
    let (t, _) = run(&mut m, N8, "auto");
    assert_eq!(t.language, "tr");
    assert_eq!(&m.calls[..2], &[Call::Encode(0), Call::First(vec![v.sot])]);
    assert!(t.windows[0].no_speech_prob > 0.99);
    assert!(t.windows[0].is_no_speech);
    assert_eq!(t.windows[0].attempts, 1, "a likely no-speech window does not fall back");
    assert!(t.segments.is_empty());
}

#[test]
fn no_speech_prob_reads_the_last_one_token_row() {
    // W1: whisper.cpp reads row 0. In a new state with a given language it holds zeros, so
    // the first window gets 1 / n_vocab. The second window gets the softmax of the last
    // decode_next row of window 1.
    let v = vocab();
    let mut m = Scripted::new(&v)
        .script(0, 0, vec![v.beg, 10, v.beg + 1000, v.beg + 1000, v.eot], STRONG)
        .script(2000, 0, vec![v.beg, 11, v.beg + 1200], STRONG);
    let (t, _) = run(&mut m, N45, "tr");
    let first = t.windows[0].no_speech_prob;
    assert!((first - 1.0 / 51866.0).abs() < 1e-9, "{first}");
    // The last decode_next row of window 1 has EOT at STRONG and 0.0 elsewhere.
    let mut row = vec![0.0; v.n_vocab()];
    row[v.eot as usize] = STRONG;
    let want = log_softmax(&row)[v.nosp as usize].exp();
    assert_eq!(t.windows[1].no_speech_prob, want);
}

#[test]
fn short_input_gives_an_empty_transcript() {
    let v = vocab();
    // 1000 samples: seek_end = 1 + 800 / 160 = 6 centiseconds, below the 10 of whisper.cpp.
    let mut m = Scripted::new(&v);
    let (t, _) = run(&mut m, 1000, "tr");
    assert!(t.segments.is_empty() && t.windows.is_empty() && m.calls.is_empty());
    // With "auto", whisper.cpp detects the language first.
    let mut detect = vec![0.0; v.n_vocab()];
    detect[v.lang_token(lang_id("de").unwrap()) as usize] = 5.0;
    let mut m = Scripted::new(&v);
    m.detect_row = Some(detect);
    let (t, _) = run(&mut m, 1000, "auto");
    assert_eq!(t.language, "de");
    assert!(t.segments.is_empty());
    // No samples at all: whisper.cpp cannot detect a language.
    let mut m = Scripted::new(&v);
    assert!(transcribe_tokens(&mut m, &v, &[], &params("auto"), &mut |_| {}).is_err());
}

#[test]
fn unknown_language_is_an_error() {
    let v = vocab();
    let mut m = Scripted::new(&v);
    let e = transcribe_tokens(&mut m, &v, &vec![0.0; N8], &params("xx"), &mut |_| {}).unwrap_err();
    assert!(format!("{e:?}").contains("xx"), "{e:?}");
}
