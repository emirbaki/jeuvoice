//! End-to-end check with the real pieces: queue, the real Whisper model, SRT export.
//! Needs the downloaded model and the sample files on this machine.
//! Run: scripts\cargo.cmd test -p jeuvoice-app --release --test real_check -- --ignored --nocapture
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use jeuvoice_core::audio;
use jeuvoice_core::queue::{ItemState, Job, Queue, Update, spawn_worker};
use jeuvoice_core::script_diff::{DiffOp, script_diff};
use jeuvoice_core::subtitles::to_srt;
use jeuvoice_whisper::{TranscribeOptions, Transcriber, WhisperRs, model_path};

const REF: &str = r"D:\MLProjelerim\tts_lab\data\ref";

#[test]
#[ignore = "needs the downloaded model and real sample files"]
fn real_batch_with_one_bad_file() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let ref_dir = Path::new(REF);

    let video = root.join("my_voice2.mp4");
    let ogg = root.join("my_voice.ogg");
    let turkish_dir = root.join("Masaüstü");
    fs::create_dir_all(&turkish_dir).unwrap();
    let wav = turkish_dir.join("çekim 1.wav");
    let bad = root.join("bozuk.mp4");
    fs::copy(ref_dir.join("my_voice2.mp4"), &video).unwrap();
    fs::copy(ref_dir.join("my_voice.ogg"), &ogg).unwrap();
    fs::copy(ref_dir.join(r"sesim\ses_001_0007.wav"), &wav).unwrap();
    fs::write(&bad, b"this is not audio at all, just garbage bytes 0123456789").unwrap();

    let model = model_path();
    assert!(model.exists(), "model missing at {}", model.display());
    let whisper = Arc::new(WhisperRs::load(&model, true).expect("load model"));
    let on_gpu = whisper.on_gpu;

    let mut queue = Queue::default();
    queue.add(&[video.clone(), ogg.clone(), wav.clone(), bad.clone()]);

    // (audio seconds, wall seconds) per path.
    let timings: Arc<Mutex<HashMap<PathBuf, (f64, f64)>>> = Arc::default();
    let (jobs_tx, jobs_rx) = channel::<(usize, Job)>();
    let (upd_tx, upd_rx) = channel::<(usize, Update)>();
    let worker = spawn_worker(jobs_rx, upd_tx, || {});

    while let Some(i) = queue.take_next() {
        let path = queue.items[i].0.clone();
        let (w, timings) = (whisper.clone(), timings.clone());
        jobs_tx
            .send((
                i,
                Box::new(move |progress| {
                    let started = Instant::now();
                    let a = audio::decode(&path).map_err(|e| e.to_string())?;
                    let audio_secs = a.samples.len() as f64 / a.rate as f64;
                    let opts = TranscribeOptions { language: "tr".into(), use_gpu: true };
                    let t = w
                        .transcribe(&audio::to_whisper(&a), &opts, progress)
                        .map_err(|e| e.to_string())?;
                    let wall = started.elapsed().as_secs_f64();
                    timings.lock().unwrap().insert(path, (audio_secs, wall));
                    Ok(t)
                }),
            ))
            .unwrap();
    }
    drop(jobs_tx);
    worker.join().unwrap();
    for (i, u) in upd_rx.try_iter() {
        queue.apply(i, u);
    }

    eprintln!("on_gpu = {on_gpu}");
    let (mut done, mut failed) = (0, 0);
    let (mut total_audio, mut total_wall) = (0.0, 0.0);
    for (path, state) in &queue.items {
        match state {
            ItemState::Done(t) => {
                done += 1;
                assert!(!t.words.is_empty(), "empty transcript for {}", path.display());
                assert_eq!(t.language, "tr", "language for {}", path.display());
                for w in &t.words {
                    assert!(w.end > w.start, "zero-length word in {}: {w:?}", path.display());
                }
                let srt = to_srt(t);
                assert!(!srt.trim().is_empty());
                let srt_path = path.with_extension("srt");
                fs::write(&srt_path, &srt).unwrap();
                assert!(srt_path.exists());
                let (a, w) = timings.lock().unwrap()[path];
                total_audio += a;
                total_wall += w;
                eprintln!(
                    "DONE {} | audio {:.1} s | wall {:.1} s | speed {:.1} audio-min per wall-min | {} words | first word at {:.2} s",
                    path.file_name().unwrap().to_string_lossy(),
                    a,
                    w,
                    a / w,
                    t.words.len(),
                    t.words[0].start
                );
                eprintln!("  text: {}", t.text());
            }
            ItemState::Failed(e) => {
                failed += 1;
                eprintln!("FAILED {} | {e}", path.file_name().unwrap().to_string_lossy());
                assert_eq!(path, &bad);
                assert_eq!(
                    e,
                    "JeuVoice cannot read this file type: bozuk.mp4. Try a supported audio or video file and drop it again."
                );
            }
            other => panic!("unexpected state for {}: {other:?}", path.display()),
        }
    }
    assert_eq!((done, failed), (3, 1));
    eprintln!(
        "TOTAL audio {:.1} s | wall {:.1} s | speed {:.1} audio-min per wall-min (first file includes model warm-up)",
        total_audio,
        total_wall,
        total_audio / total_wall
    );

    // Script check against the take's own text.
    let script = fs::read_to_string(ref_dir.join(r"sesim\ses_001_0007.txt")).unwrap();
    let ItemState::Done(t) = &queue.items[2].1 else { unreachable!() };
    let ops = script_diff(&script, t);
    let count = |f: fn(&DiffOp) -> bool| ops.iter().filter(|o| f(o)).count();
    eprintln!(
        "SCRIPT CHECK same {} | changed {} | missing {} | added {}",
        count(|o| matches!(o, DiffOp::Same { .. })),
        count(|o| matches!(o, DiffOp::Changed { .. })),
        count(|o| matches!(o, DiffOp::Missing { .. })),
        count(|o| matches!(o, DiffOp::Added { .. })),
    );
}
