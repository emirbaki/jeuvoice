// CudaWhisper (plan 1b, Task 11): the Transcriber on the Rust kernels, against the
// whisper-rs fixtures of Task 0. Run the ignored tests one at a time (4 GB of VRAM):
//   scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels --release --test cuda_transcribe -- --ignored --nocapture --test-threads 1
#![cfg(feature = "cuda-kernels")]

mod common;

use std::path::Path;
use std::sync::Arc;

use jeuvoice_core::transcript::Word;
use jeuvoice_whisper::cuda::model::GpuWhisper;
use jeuvoice_whisper::cuda::transcriber::CudaWhisper;
use jeuvoice_whisper::decode::vocab::Vocab;
use jeuvoice_whisper::decode::window::{DecodeParams, transcribe_tokens};
use jeuvoice_whisper::ggml::GgmlModel;
use jeuvoice_whisper::{TranscribeOptions, Transcriber, WhisperError, tokens_to_words};

fn opts() -> TranscribeOptions {
    TranscribeOptions { language: "tr".into(), use_gpu: true }
}

#[test]
fn cuda_whisper_is_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<CudaWhisper>();
}

#[test]
fn a_missing_model_is_a_model_load_error() {
    let e = CudaWhisper::load(Path::new("no/such/model.bin")).err().expect("a missing file must fail");
    assert!(matches!(e, WhisperError::ModelLoad(_)), "{e:?}");
}

/// The words that whisper-rs gives for a fixture, with the rule of `WhisperRs::transcribe`.
fn fixture_words(f: &common::Fixture) -> Vec<Word> {
    let tokens: Vec<(String, f64, f64, f32)> = f
        .segments
        .iter()
        .flat_map(|s| &s.tokens)
        .filter(|t| t.id < 50257)
        .map(|t| {
            let (start, end) = if t.t_dtw >= 0 { (t.t_dtw, t.t1.max(t.t_dtw)) } else { (t.t0, t.t1) };
            (String::new(), start as f64 / 100.0, end as f64 / 100.0, t.p)
        })
        .collect();
    tokens_with_text(f, tokens)
}

fn tokens_with_text(f: &common::Fixture, mut tokens: Vec<(String, f64, f64, f32)>) -> Vec<Word> {
    let texts = text_of_fixture_tokens(f);
    for (t, text) in tokens.iter_mut().zip(texts) {
        t.0 = text;
    }
    tokens_to_words(&tokens)
}

/// The token texts of the fixture JSON, in order, for the text tokens only.
fn text_of_fixture_tokens(f: &common::Fixture) -> Vec<String> {
    let name = f.clip.file_stem().unwrap().to_string_lossy().into_owned();
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../target/reference/whisper/{name}.rs.json"));
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    v["segments"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|s| s["tokens"].as_array().unwrap().iter())
        .filter(|t| t["id"].as_u64().unwrap() < 50257)
        .map(|t| t["text"].as_str().unwrap().to_string())
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

#[test]
#[ignore = "needs the model file, the test clips and the whisper-rs fixtures (Task 0)"]
fn tokens_and_times_match_whisper_rs() {
    // The token ids, and for the tokens at the same position the t0, t1 and t_dtw times.
    // This is the first check of DTW on real audio (the open item of Task 6).
    let model = GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap();
    let vocab = Vocab::from_ggml(&model).unwrap();
    let mut gw = GpuWhisper::load(&model).unwrap();
    for clip in common::CLIPS {
        let f = common::load(clip);
        let samples = common::samples(&f);
        gw.set_audio(&samples).unwrap();
        let t = transcribe_tokens(&mut gw, &vocab, &samples, &DecodeParams::default(), &mut |_| {}).unwrap();
        let fallbacks = t.windows.iter().filter(|w| w.attempts > 1).count();
        let ours: Vec<_> = t.segments.iter().flat_map(|s| s.tokens.iter().copied()).collect();
        let theirs: Vec<_> = f.segments.iter().flat_map(|s| s.tokens.iter().copied()).collect();
        let same: Vec<_> = ours.iter().zip(&theirs).filter(|(a, b)| a.id == b.id).collect();
        for (i, (a, b)) in ours.iter().zip(&theirs).enumerate().filter(|(_, (a, b))| a.id != b.id) {
            eprintln!("{clip}: token {i}: ours {} {:?} (p {:.3}), whisper-rs {} {:?} (p {:.3})", a.id, vocab.token_text_lossy(a.id), a.p, b.id, vocab.token_text_lossy(b.id), b.p);
        }
        let times_equal = same.iter().filter(|(a, b)| a.t0 == b.t0 && a.t1 == b.t1).count();
        let dtw: Vec<f64> = same.iter().filter(|(a, b)| a.t_dtw >= 0 && b.t_dtw >= 0).map(|(a, b)| (a.t_dtw - b.t_dtw).abs() as f64).collect();
        let dtw_within_5 = dtw.iter().filter(|&&d| d <= 5.0).count();
        eprintln!(
            "{clip}: windows {} (fallbacks {fallbacks}), tokens ours {} / whisper-rs {}, same {}, t0/t1 equal {times_equal}, t_dtw |diff| median {:.1} cs, max {:.0} cs, within 5 cs {dtw_within_5}/{}",
            t.windows.len(),
            ours.len(),
            theirs.len(),
            same.len(),
            median(dtw.clone()),
            dtw.iter().cloned().fold(0.0, f64::max),
            dtw.len()
        );
        assert!(same.len() as f64 >= 0.98 * theirs.len() as f64, "{clip}: only {} of {} tokens equal", same.len(), theirs.len());
        assert!(median(dtw.clone()) <= 2.0, "{clip}: the DTW times differ by a median of {} cs", median(dtw));
    }
}

#[test]
#[ignore = "needs the model file, the test clips and the whisper-rs fixtures (Task 0)"]
fn words_match_whisper_rs() {
    let w = CudaWhisper::load(&jeuvoice_whisper::model_path()).unwrap();
    for clip in common::CLIPS {
        let f = common::load(clip);
        let samples = common::samples(&f);
        let mut last = -1.0f32;
        let t = w.transcribe(&samples, &opts(), &mut |p| last = p).unwrap();
        let want = fixture_words(&f);
        // A position zip: after one different word every later word sits one place off
        // (my_voice2 on 2026-11-25: 31 of 69 same by position, from one near tie). Task 12
        // aligns the words with Levenshtein for the WER.
        let n = t.words.len().min(want.len());
        let same_text = (0..n).filter(|&i| t.words[i].text == want[i].text).count();
        let starts: Vec<f64> = (0..n).filter(|&i| t.words[i].text == want[i].text).map(|i| (t.words[i].start - want[i].start).abs()).collect();
        eprintln!(
            "{clip}: words ours {} / whisper-rs {}, same text {same_text}, start |diff| median {:.3} s",
            t.words.len(),
            want.len(),
            median(starts.clone())
        );
        assert_eq!(t.language, "tr");
        assert_eq!(last, 1.0, "{clip}: the last progress value");
        assert!((t.words.len() as f64 - want.len() as f64).abs() <= 0.02 * want.len() as f64, "{clip}: word count {} against {}", t.words.len(), want.len());
        assert!(median(starts) <= 0.05, "{clip}: the word starts differ by more than 50 ms in the median");
        for p in t.words.windows(2) {
            assert!(p[1].start >= p[0].start, "{clip}: times must not go back: {p:?}");
        }
        assert!(t.words.iter().all(|w| w.end > w.start), "{clip}: a word with no length");
    }
}

#[test]
#[ignore = "needs the model file and a test clip"]
fn two_threads_one_after_the_other() {
    // Deferred minor M7: a transcription from a thread that did not load the model.
    let f = common::load("ses_001_0007");
    let samples = Arc::new(common::samples(&f));
    let w = Arc::new(CudaWhisper::load(&jeuvoice_whisper::model_path()).unwrap());
    let (w2, s2) = (Arc::clone(&w), Arc::clone(&samples));
    let first = std::thread::spawn(move || w2.transcribe(&s2, &opts(), &mut |_| {}).unwrap()).join().unwrap();
    let second = w.transcribe(&samples, &opts(), &mut |_| {}).unwrap();
    assert!(!first.words.is_empty());
    assert_eq!(first, second, "the same audio must give the same transcript on any thread");
}

#[test]
#[ignore = "needs the model file and a test clip"]
fn panicking_progress_returns_an_error() {
    let f = common::load("ses_001_0007");
    let samples = common::samples(&f);
    let w = CudaWhisper::load(&jeuvoice_whisper::model_path()).unwrap();
    let mut calls = 0;
    let r = w.transcribe(&samples, &opts(), &mut |_| {
        calls += 1;
        panic!("progress panic");
    });
    assert!(matches!(r, Err(WhisperError::Transcribe(_))), "{r:?}");
    assert_eq!(calls, 1, "after a panic the callback is not called again");
    // The worker still works after that.
    assert!(!w.transcribe(&samples, &opts(), &mut |_| {}).unwrap().words.is_empty());
}

/// What DTW adds to a whole transcription (lesson 9): the same clip with the DTW pass and
/// without it. The difference holds the GPU alignment pass and the CPU stages together.
#[test]
#[ignore = "needs the model file and the test clips"]
fn dtw_cost_in_a_transcription() {
    let model = GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap();
    let vocab = Vocab::from_ggml(&model).unwrap();
    let mut gw = GpuWhisper::load(&model).unwrap();
    let f = common::load(common::CLIPS[1]);
    let samples = common::samples(&f);
    gw.set_audio(&samples).unwrap();
    let mut times = Vec::new();
    for (label, dtw) in [("off", false), ("on", true), ("off", false), ("on", true)] {
        let params = DecodeParams { dtw, ..DecodeParams::default() };
        let t0 = std::time::Instant::now();
        let t = transcribe_tokens(&mut gw, &vocab, &samples, &params, &mut |_| {}).unwrap();
        let s = t0.elapsed().as_secs_f64();
        eprintln!("dtw {label}: {s:.3} s, {} windows", t.windows.len());
        times.push((label, s));
    }
    let off = times.iter().filter(|(l, _)| *l == "off").map(|(_, s)| *s).fold(f64::MAX, f64::min);
    let on = times.iter().filter(|(l, _)| *l == "on").map(|(_, s)| *s).fold(f64::MAX, f64::min);
    eprintln!("DTW adds {:.3} s to the best of each", on - off);
}

