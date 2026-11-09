use std::path::Path;

use jeuvoice_core::audio::{decode, to_whisper};
use jeuvoice_whisper::{download_model, model_path, TranscribeOptions, Transcriber, WhisperRs};

#[test]
#[ignore = "needs the downloaded model and a real clip"]
fn transcribes_real_clip() {
    let model = model_path();
    assert!(model.exists(), "model missing at {}", model.display());
    let clip = Path::new(r"D:\MLProjelerim\tts_lab\data\ref\sesim\ses_001_0007.wav");
    let samples = to_whisper(&decode(clip).expect("decode clip"));

    let w = WhisperRs::load(&model, true).expect("load model");
    eprintln!("on_gpu = {}", w.on_gpu);
    let mut last = -1.0f32;
    let t = w
        .transcribe(
            &samples,
            &TranscribeOptions { language: "tr".into(), use_gpu: true },
            &mut |p| last = p,
        )
        .expect("transcribe");
    eprintln!("{} | {:?}", t.text(), t.words);

    assert!(t.words.len() >= 3);
    assert_eq!(t.language, "tr");
    assert_eq!(last, 1.0);
    for pair in t.words.windows(2) {
        assert!(pair[1].start >= pair[0].start, "times must not go back: {pair:?}");
    }
    for w in &t.words {
        assert!(w.end >= w.start);
    }
}

#[test]
#[ignore = "downloads about 1.6 GB on the first run"]
fn download_puts_model_in_place() {
    let mut last = (0u64, 0u64);
    let path = download_model(&mut |done, total| last = (done, total)).expect("download");
    assert_eq!(path, model_path());
    assert!(std::fs::metadata(&path).unwrap().len() > 1_000_000_000);
    assert_eq!(last.0, last.1);
    assert!(!path.with_extension("bin.part").exists());
}
