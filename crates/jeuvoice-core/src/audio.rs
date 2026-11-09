//! Decode audio or video files to mono `f32` samples, and resample them.

use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};

use rubato::{FftFixedIn, Resampler};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Sample rate that Whisper expects.
pub const WHISPER_RATE: u32 = 16_000;

/// Mono audio.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub rate: u32,
}

#[derive(Debug)]
pub enum AudioError {
    NotFound(PathBuf),
    NoAudioTrack(PathBuf),
    Unsupported { path: PathBuf, detail: String },
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioError::NotFound(p) => write!(f, "Cannot find the file {}.", p.display()),
            AudioError::NoAudioTrack(p) => {
                write!(f, "The file {} has no sound track.", file_name(p))
            }
            AudioError::Unsupported { path, detail } => write!(
                f,
                "JeuVoice cannot read the sound in {} ({detail}).",
                file_name(path)
            ),
        }
    }
}

impl std::error::Error for AudioError {}

/// Decode the first audio track at its native rate, mixing channels to mono.
pub fn decode(path: &Path) -> Result<Audio, AudioError> {
    let unsupported = |detail: String| AudioError::Unsupported {
        path: path.to_path_buf(),
        detail,
    };
    let file = File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            AudioError::NotFound(path.to_path_buf())
        } else {
            unsupported(e.to_string())
        }
    })?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| unsupported(e.to_string()))?
        .format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::NoAudioTrack(path.to_path_buf()))?;
    let track_id = track.id;
    let params = track.codec_params.clone();
    let mut decoder = symphonia::default::get_codecs()
        .make(&params, &DecoderOptions::default())
        .map_err(|e| unsupported(e.to_string()))?;

    let mut rate = params.sample_rate.unwrap_or(0);
    let mut samples: Vec<f32> = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(unsupported(e.to_string())),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                rate = spec.rate;
                let channels = spec.channels.count().max(1);
                let buf = buf.get_or_insert_with(|| {
                    SampleBuffer::<f32>::new(decoded.capacity() as u64, spec)
                });
                buf.copy_interleaved_ref(decoded);
                for frame in buf.samples().chunks(channels) {
                    samples.push(frame.iter().sum::<f32>() / channels as f32);
                }
            }
            // A damaged packet is skipped; the rest of the file still plays.
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(unsupported(e.to_string())),
        }
    }

    if rate == 0 || samples.is_empty() {
        return Err(unsupported("no sound could be decoded".into()));
    }
    Ok(Audio { samples, rate })
}

/// Resample mono audio to `rate`.
pub fn resample(a: &Audio, rate: u32) -> Audio {
    if a.rate == rate || a.samples.is_empty() {
        return Audio { samples: a.samples.clone(), rate };
    }
    const CHUNK: usize = 1024;
    let mut rs = FftFixedIn::<f32>::new(a.rate as usize, rate as usize, CHUNK, 2, 1)
        .expect("valid resampler settings");
    let expected = (a.samples.len() as u64 * rate as u64 / a.rate as u64) as usize;
    let delay = rs.output_delay();
    let mut out: Vec<f32> = Vec::with_capacity(expected + delay + CHUNK);

    let mut pos = 0;
    while pos + rs.input_frames_next() <= a.samples.len() {
        let n = rs.input_frames_next();
        let r = rs.process(&[&a.samples[pos..pos + n]], None).expect("resample");
        out.extend_from_slice(&r[0]);
        pos += n;
    }
    let tail: Option<&[&[f32]]> = None;
    let rest = [&a.samples[pos..]];
    if let Ok(r) = rs.process_partial(Some(&rest), None) {
        out.extend_from_slice(&r[0]);
    }
    // Flush the filter so the end of the signal is not cut off.
    while out.len() < expected + delay {
        match rs.process_partial(tail, None) {
            Ok(r) if !r[0].is_empty() => out.extend_from_slice(&r[0]),
            _ => break,
        }
    }
    let samples: Vec<f32> = out.into_iter().skip(delay).take(expected).collect();
    Audio { samples, rate }
}

/// 16 kHz mono samples for Whisper.
pub fn to_whisper(a: &Audio) -> Vec<f32> {
    resample(a, WHISPER_RATE).samples
}
