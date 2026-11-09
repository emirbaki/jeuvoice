use jeuvoice_core::audio::{AudioError, decode, to_whisper};
use std::path::Path;

fn write_wav(path: &Path, rate: u32, channels: u16, frames: usize, f: impl Fn(usize, u16) -> f32) {
    let spec = hound::WavSpec {
        channels,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..frames {
        for c in 0..channels {
            w.write_sample(f(i, c)).unwrap();
        }
    }
    w.finalize().unwrap();
}

#[test]
fn decodes_stereo_wav_to_mono() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.wav");
    write_wav(&p, 48000, 2, 48000, |_, c| if c == 0 { 0.5 } else { -0.5 });
    let a = decode(&p).unwrap();
    assert_eq!(a.rate, 48000);
    assert_eq!(a.samples.len(), 48000);
    assert!(a.samples.iter().all(|s| s.abs() < 1e-3));
}

#[test]
fn resamples_to_16k() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.wav");
    write_wav(&p, 48000, 1, 48000, |i, _| {
        (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.5
    });
    let a = decode(&p).unwrap();
    let out = to_whisper(&a);
    assert!((out.len() as i64 - 16000).abs() <= 16, "got {}", out.len());
}

#[test]
fn reads_paths_with_turkish_letters() {
    let dir = tempfile::tempdir().unwrap();
    let sub = dir.path().join("Masaüstü");
    std::fs::create_dir_all(&sub).unwrap();
    let p = sub.join("çekim 1.wav");
    write_wav(&p, 16000, 1, 1600, |_, _| 0.1);
    assert_eq!(decode(&p).unwrap().samples.len(), 1600);
}

#[test]
fn missing_file_is_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let err = decode(&dir.path().join("yok.wav")).unwrap_err();
    assert!(matches!(err, AudioError::NotFound(_)));
    assert!(err.to_string().starts_with("Cannot find the file"));
}

#[test]
fn decodes_ogg_opus() {
    // 0.5 s of a 440 Hz sine, mono, made with ffmpeg.
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sine-opus.ogg");
    let a = decode(&p).unwrap();
    assert_eq!(a.rate, 48000);
    assert_eq!(a.samples.len(), 24_000, "pre-skip and end trim applied once");
    let mid = &a.samples[6000..18000];
    let rms = (mid.iter().map(|s| s * s).sum::<f32>() / mid.len() as f32).sqrt();
    assert!((0.05..0.6).contains(&rms), "rms {rms}");
}
