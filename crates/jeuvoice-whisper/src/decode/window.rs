//! The window loop: a port of `whisper_full_with_state` (whisper.cpp line 6792) for the
//! parameters of the app (greedy, `best_of 1`, timestamps on).
//!
//! Whisper reads 30 s of audio at a time. The loop encodes the window at `seek`, decodes
//! tokens until EOT or the end of the audio, and moves `seek` to the last timestamp. A
//! window with a bad result is decoded again at a higher temperature.
//!
//! Times are in centiseconds ("frames" of 10 ms). One timestamp step is 2 frames (20 ms).

use super::rules::{log_softmax, process_logits, RuleInput};
use super::sample::{sample_token, sequence_score, Rng, TokenData};
use super::token_times::{signal_energy, token_level_times, TokenTimeState};
use super::vocab::{lang_code, lang_id, Vocab, N_LANGUAGES};
use crate::WhisperError;

/// The numeric part of Whisper, as the window loop sees it.
pub trait WhisperModel {
    /// Encodes the 30 s mel window that starts at frame `seek`.
    fn encode(&mut self, seek: usize) -> Result<(), WhisperError>;
    /// Starts a new self-attention cache, decodes `tokens`, and returns the `n_vocab`
    /// logits of the last row.
    fn decode_first(&mut self, tokens: &[u32]) -> Result<Vec<f32>, WhisperError>;
    /// Decodes one more token with the cache and returns its `n_vocab` logits.
    fn decode_next(&mut self, token: u32) -> Result<Vec<f32>, WhisperError>;
}

/// The decoding parameters. `Default` gives the values of the app (`WhisperRs`).
#[derive(Clone, Debug)]
pub struct DecodeParams {
    /// A language code such as "tr", or "auto".
    pub language: String,
    pub temperature: f32,
    pub temperature_inc: f32,
    pub entropy_thold: f64,
    pub logprob_thold: f64,
    pub no_speech_thold: f32,
    pub length_penalty: f32,
    /// Gives each token `t0` and `t1` (`token_level_times`). The app sets it.
    pub token_timestamps: bool,
}

impl Default for DecodeParams {
    fn default() -> Self {
        DecodeParams {
            language: "tr".into(),
            temperature: 0.0,
            temperature_inc: 0.2,
            entropy_thold: 2.4,
            logprob_thold: -1.0,
            no_speech_thold: 0.6,
            length_penalty: -1.0,
            token_timestamps: true,
        }
    }
}

/// One segment: the text between two timestamps.
#[derive(Clone, Debug)]
pub struct Segment {
    /// Centiseconds.
    pub t0: i64,
    pub t1: i64,
    /// The bytes of the text tokens (below EOT).
    pub text: Vec<u8>,
    pub no_speech_prob: f32,
    pub tokens: Vec<TokenData>,
}

/// What happened in one window. For the measurements (watch items W1 and W2).
#[derive(Clone, Debug)]
pub struct WindowInfo {
    pub seek: usize,
    /// The temperature of the kept result.
    pub temperature: f32,
    /// The number of temperatures tried (1 when there was no fallback).
    pub attempts: usize,
    pub no_speech_prob: f32,
    pub is_no_speech: bool,
}

#[derive(Clone, Debug)]
pub struct Transcription {
    pub language: String,
    pub segments: Vec<Segment>,
    pub windows: Vec<WindowInfo>,
}

/// `delta_min` of whisper.cpp: 100 ms.
const DELTA_MIN: i64 = 10;
/// `WHISPER_CHUNK_SIZE`: a window is 30 s = 3000 frames.
const WINDOW_FRAMES: i64 = 3000;
/// `n_text_ctx / 2`: the longest prompt (`n_max_text_ctx` is 16384, so this limit wins).
const MAX_PROMPT_CTX: usize = 448 / 2;
/// `n_text_ctx / 2 - 4`: the most tokens in one window.
const N_MAX_STEPS: usize = 448 / 2 - 4;
/// `WHISPER_HISTORY_CONDITIONING_TEMP_CUTOFF`: no past tokens in the prompt from here up.
const PAST_TEMP_CUTOFF: f32 = 0.5;

/// `mel.n_len_org` of whisper.cpp: the number of frames of the audio itself.
pub fn n_len_org(n_samples: usize) -> i64 {
    if n_samples == 0 {
        // whisper.cpp does not compute the mel for 0 samples, so the value stays 0.
        return 0;
    }
    // C integer division (it rounds toward zero, as Rust `/` does).
    1 + (n_samples as i64 + 200 - 400) / 160
}

/// The state of the one greedy decoder at one temperature.
struct Decoder {
    tokens: Vec<TokenData>,
    result_len: usize,
    seek_delta: i64,
    has_ts: bool,
    failed: bool,
    completed: bool,
}

/// Transcribes the 16 kHz `samples` with `model`, as `whisper_full` does. The model holds
/// the mel of the same samples.
pub fn transcribe_tokens(
    model: &mut dyn WhisperModel,
    vocab: &Vocab,
    samples: &[f32],
    params: &DecodeParams,
    progress: &mut dyn FnMut(f32),
) -> Result<Transcription, WhisperError> {
    let n_vocab = vocab.n_vocab();
    let check = |logits: &[f32]| {
        if logits.len() == n_vocab {
            Ok(())
        } else {
            Err(WhisperError::Transcribe(format!("the model gave {} logits, not n_vocab = {n_vocab}", logits.len())))
        }
    };
    let seek_end = n_len_org(samples.len());

    // W1: whisper.cpp reads the no-speech prob from row 0 of its logits buffer. Only a
    // 1-token decode fills that row. So it holds the logits of the last 1-token decode, or
    // zeros in a new state.
    let mut row0: Option<Vec<f32>> = None;

    let mut language = params.language.clone();
    if language.is_empty() || language == "auto" {
        if seek_end <= 0 {
            return Err(WhisperError::Transcribe("the audio is too short to detect the language".into()));
        }
        model.encode(0)?;
        let logits = model.decode_first(&[vocab.sot])?;
        check(&logits)?;
        language = lang_code(detect_language(vocab, &logits)).expect("a language index").to_string();
        row0 = Some(logits);
    }
    let lang = lang_id(&language).ok_or_else(|| WhisperError::Transcribe(format!("unknown language {language:?}")))?;

    // The token times keep a state across segments, and read the energy of the whole audio.
    let mut times = TokenTimes {
        on: params.token_timestamps,
        state: TokenTimeState::default(),
        energy: if params.token_timestamps { signal_energy(samples, 32) } else { Vec::new() },
    };

    let mut out = Transcription { language, segments: Vec::new(), windows: Vec::new() };
    if seek_end < DELTA_MIN {
        // whisper.cpp: "input is too short", and an empty result.
        return Ok(out);
    }

    let temperatures = temperatures(params);
    let prompt_init = [vocab.sot, vocab.lang_token(lang), vocab.transcribe];
    let mut prompt_past: Vec<u32> = Vec::new();
    // One generator for the whole transcription, as `mt19937(0)` at the state init.
    let mut rng = Rng::new(0);
    let mut seek: i64 = 0;

    loop {
        progress(((100 * seek) / seek_end) as f32 / 100.0);
        if seek + DELTA_MIN >= seek_end {
            break;
        }
        model.encode(seek as usize)?;

        // Less than 5 s left: the past tokens tend to make the decoder repeat itself.
        if seek > 0 && seek + 500 >= seek_end {
            prompt_past.clear();
        }

        let mut attempts = 0;
        let (dec, t_cur, prompt, no_speech_prob, avg_logprobs) = 'temps: {
            for (it, &t_cur) in temperatures.iter().enumerate() {
                attempts += 1;
                let mut prompt = Vec::new();
                if t_cur < PAST_TEMP_CUTOFF && !prompt_past.is_empty() {
                    prompt.push(vocab.prev);
                    let n_take = (MAX_PROMPT_CTX - 1).min(prompt_past.len());
                    prompt.extend_from_slice(&prompt_past[prompt_past.len() - n_take..]);
                }
                prompt.extend_from_slice(&prompt_init);

                let logits = model.decode_first(&prompt)?;
                check(&logits)?;
                let no_speech_prob = no_speech_prob(row0.as_deref(), n_vocab, vocab.nosp);

                let mut dec = Decoder {
                    tokens: Vec::new(),
                    result_len: 0,
                    seek_delta: WINDOW_FRAMES,
                    has_ts: false,
                    failed: false,
                    completed: false,
                };
                let mut processed = process_logits(
                    vocab,
                    &logits,
                    &RuleInput { sampled: &[], temperature: t_cur, has_ts: false, seek_delta: dec.seek_delta as i32 },
                );

                for i in 0..N_MAX_STEPS {
                    let tok = sample_token(vocab, &processed, t_cur < 1e-6, &mut rng);
                    dec.tokens.push(tok);
                    step(&mut dec, vocab, tok, i, seek, seek_end);
                    if dec.failed || dec.completed {
                        break;
                    }
                    let logits = model.decode_next(tok.id)?;
                    check(&logits)?;
                    processed = process_logits(
                        vocab,
                        &logits,
                        &RuleInput {
                            sampled: &dec.tokens,
                            temperature: t_cur,
                            has_ts: dec.has_ts,
                            seek_delta: dec.seek_delta as i32,
                        },
                    );
                    row0 = Some(logits);
                }

                // A decoder that failed in the loop keeps all its tokens and no score.
                let mut avg_logprobs = f64::NEG_INFINITY;
                if !dec.failed {
                    dec.tokens.truncate(dec.result_len);
                    if let Some(s) = sequence_score(&dec.tokens, dec.result_len, params.length_penalty) {
                        avg_logprobs = s.avg_logprobs;
                        if dec.result_len > 32 && s.entropy < params.entropy_thold {
                            dec.failed = true;
                        }
                    }
                }

                let last = it == temperatures.len() - 1;
                let bad = dec.failed || (avg_logprobs < params.logprob_thold && no_speech_prob < params.no_speech_thold);
                if last || !bad {
                    break 'temps (dec, t_cur, prompt, no_speech_prob, avg_logprobs);
                }
            }
            unreachable!("the last temperature always ends the loop");
        };

        let is_no_speech = no_speech_prob > params.no_speech_thold && avg_logprobs < params.logprob_thold;
        let tokens = &dec.tokens;

        // The rolling context for the next window.
        let mut past = Vec::new();
        if prompt.first() == Some(&vocab.prev) {
            past.extend_from_slice(&prompt[1..prompt.len() - prompt_init.len()]);
        }
        if !is_no_speech {
            past.extend(tokens[..dec.result_len].iter().map(|t| t.id));
        }
        prompt_past = past;

        if !tokens.is_empty() && !is_no_speech {
            make_segments(&mut out.segments, vocab, tokens, seek, dec.seek_delta, no_speech_prob, &mut times);
        }

        let mut seek_delta = dec.seek_delta;
        // whisper.cpp PR 2629: text and then one timestamp at the end skips the whole window.
        let n = tokens.len();
        if n > 1 && tokens[n - 2].id < vocab.beg && tokens[n - 1].id > vocab.beg {
            seek_delta = (seek_end - seek).min(WINDOW_FRAMES);
        }

        out.windows.push(WindowInfo { seek: seek as usize, temperature: t_cur, attempts, no_speech_prob, is_no_speech });
        seek += seek_delta;
    }
    Ok(out)
}

/// Updates the decoder after one sampled token (whisper.cpp line 7333 to 7396).
fn step(dec: &mut Decoder, vocab: &Vocab, tok: TokenData, i: usize, seek: i64, seek_end: i64) {
    // A timestamp moves the window end. BEG itself (0.00 s) does not count.
    if tok.id > vocab.beg {
        let seek_delta_new = 2 * (tok.id - vocab.beg) as i64;
        if dec.has_ts && dec.seek_delta > seek_delta_new && dec.result_len < i {
            // Time must not go back.
            dec.failed = true;
            return;
        }
        dec.seek_delta = seek_delta_new;
        dec.result_len = i + 1;
        dec.has_ts = true;
    }

    let at_audio_end = seek + dec.seek_delta + DELTA_MIN >= seek_end;
    if tok.id == vocab.eot || (dec.has_ts && at_audio_end) {
        if dec.result_len == 0 {
            if at_audio_end {
                dec.result_len = i + 1;
            } else {
                dec.failed = true;
                return;
            }
        }
        dec.completed = true;
        return;
    }

    // A repetition loop: no timestamp, or less than half a window, after all the steps.
    if i == N_MAX_STEPS - 1 && (dec.result_len == 0 || dec.seek_delta < WINDOW_FRAMES / 2) {
        dec.failed = true;
    }
}

/// The token-time step of the segments: on or off, its state and the signal energy.
struct TokenTimes {
    on: bool,
    state: TokenTimeState,
    energy: Vec<f32>,
}

impl TokenTimes {
    /// Pushes a segment and, when on, gives its tokens `t0` and `t1` at once, as
    /// whisper.cpp does after each segment.
    fn push(&mut self, out: &mut Vec<Segment>, vocab: &Vocab, mut seg: Segment) {
        if self.on {
            token_level_times(vocab, &mut self.state, &self.energy, seg.t0, seg.t1, &mut seg.tokens);
        }
        out.push(seg);
    }
}

/// Splits the tokens of a window into segments at the timestamp tokens (line 7605).
fn make_segments(out: &mut Vec<Segment>, vocab: &Vocab, tokens: &[TokenData], seek: i64, seek_delta: i64, nsp: f32, times: &mut TokenTimes) {
    let beg = vocab.beg;
    let time = |tid: u32| seek + 2 * (tid as i64 - beg as i64);
    let mut i0 = 0;
    let mut t0 = time(tokens[0].tid);
    let mut text: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let id = tokens[i].id;
        if id < vocab.eot {
            text.extend_from_slice(vocab.token_bytes(id));
        }
        if id > beg {
            let t1 = time(tokens[i].tid);
            if !text.is_empty() {
                let seg = Segment { t0, t1, text: std::mem::take(&mut text), no_speech_prob: nsp, tokens: tokens[i0..=i].to_vec() };
                times.push(out, vocab, seg);
            }
            text.clear();
            // Skip the rest of a run of timestamps.
            while i < tokens.len() && tokens[i].id > beg {
                i += 1;
            }
            i -= 1;
            t0 = t1;
            i0 = i + 1;
        }
        i += 1;
    }
    if !text.is_empty() {
        let seg = Segment { t0, t1: seek + seek_delta, text, no_speech_prob: nsp, tokens: tokens[i0..].to_vec() };
        times.push(out, vocab, seg);
    }
}

/// The temperatures of whisper.cpp, summed in f32: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0.
fn temperatures(params: &DecodeParams) -> Vec<f32> {
    if params.temperature_inc <= 0.0 {
        return vec![params.temperature];
    }
    let mut out = Vec::new();
    let mut t = params.temperature;
    while t < 1.0 + 1e-6 {
        out.push(t);
        t += params.temperature_inc;
    }
    out
}

/// The softmax of row 0 at the NOSP token. A missing row is a row of zeros.
fn no_speech_prob(row0: Option<&[f32]>, n_vocab: usize, nosp: u32) -> f32 {
    let zeros;
    let row = match row0 {
        Some(r) => r,
        None => {
            zeros = vec![0.0; n_vocab];
            &zeros
        }
    };
    let lp = log_softmax(row)[nosp as usize];
    if row[nosp as usize] == f32::NEG_INFINITY { 0.0 } else { lp.exp() }
}

/// The language with the largest logit (`whisper_lang_auto_detect_with_state`). On a tie
/// it takes the first code in alphabetical order, as the `g_lang` map lists them.
fn detect_language(vocab: &Vocab, logits: &[f32]) -> usize {
    let mut order: Vec<usize> = (0..N_LANGUAGES).collect();
    order.sort_by_key(|&i| lang_code(i).expect("a language index"));
    let mut best = order[0];
    for &i in &order[1..] {
        if logits[vocab.lang_token(i) as usize] > logits[vocab.lang_token(best) as usize] {
            best = i;
        }
    }
    best
}
