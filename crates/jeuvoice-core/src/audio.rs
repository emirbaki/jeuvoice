//! Decode audio or video files to mono `f32` samples, and resample them.

use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};

use rubato::{FftFixedIn, Resampler};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, CODEC_TYPE_OPUS, DecoderOptions};
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

/// Why a file gave no sound. `Display` is plain words for the file list. The `detail` fields
/// keep the decoder's own text for logs (`Debug`) and never reach the user.
#[derive(Debug)]
pub enum AudioError {
    NotFound(PathBuf),
    /// The file could not be opened (no permission, or another program locks it).
    CannotOpen { path: PathBuf, detail: String },
    NoAudioTrack(PathBuf),
    /// No reader or decoder for this container or sound format.
    UnknownFormat { path: PathBuf, detail: String },
    /// The format is known but the data is broken.
    Damaged { path: PathBuf, detail: String },
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
            AudioError::CannotOpen { path, .. } => write!(
                f,
                "JeuVoice cannot open {}. Close other programs that use it and drop it again.",
                file_name(path)
            ),
            AudioError::NoAudioTrack(p) => {
                write!(f, "The file {} has no sound track.", file_name(p))
            }
            AudioError::UnknownFormat { path, .. } => write!(
                f,
                "JeuVoice cannot read this file type: {}. Try a supported audio or video file and drop it again.",
                file_name(path)
            ),
            AudioError::Damaged { path, .. } => write!(
                f,
                "The sound in {} is damaged. Export it again and drop it again.",
                file_name(path)
            ),
        }
    }
}

impl std::error::Error for AudioError {}

/// Decode the first audio track at its native rate, mixing channels to mono.
pub fn decode(path: &Path) -> Result<Audio, AudioError> {
    let unknown = |detail: String| AudioError::UnknownFormat { path: path.to_path_buf(), detail };
    let damaged = |detail: String| AudioError::Damaged { path: path.to_path_buf(), detail };
    // A symphonia error while reading: an unsupported feature means an unknown format, the
    // rest means broken data.
    let read_error = |e: SymError| match e {
        SymError::Unsupported(_) => unknown(e.to_string()),
        _ => damaged(e.to_string()),
    };
    let file = File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            AudioError::NotFound(path.to_path_buf())
        } else {
            AudioError::CannotOpen { path: path.to_path_buf(), detail: e.to_string() }
        }
    })?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    // Windows cameras and phones save `VIDEO.MKV` in upper case, and a path outside
    // UTF-8 would give no hint at all: normalise the extension before probing.
    if let Some(ext) = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()) {
        hint.with_extension(&ext);
    }
    let mut format = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions { enable_gapless: true, ..Default::default() },
            &MetadataOptions::default(),
        )
        // The probe fails when no reader knows the bytes: an unknown file type.
        .map_err(|e| unknown(e.to_string()))?
        .format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::NoAudioTrack(path.to_path_buf()))?;
    let track_id = track.id;
    let params = track.codec_params.clone();
    // symphonia reads Opus in Ogg and in Matroska/WebM but cannot decode it, so Opus has its
    // own decoder. Both containers hand over the `OpusHead` header as extra data.
    let mut opus = if params.codec == CODEC_TYPE_OPUS {
        Some(OpusDecoder::new(params.extra_data.as_deref().unwrap_or_default()).map_err(unknown)?)
    } else {
        None
    };
    let mut decoder = if opus.is_none() {
        Some(
            symphonia::default::get_codecs()
                .make(&params, &DecoderOptions::default())
                .map_err(|e| unknown(e.to_string()))?,
        )
    } else {
        None
    };

    let mut rate = params.sample_rate.unwrap_or(0);
    let mut samples: Vec<f32> = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(read_error(e)),
        };
        if packet.track_id() != track_id {
            continue;
        }
        if let Some(o) = opus.as_mut() {
            rate = OPUS_RATE;
            samples.extend(o.decode(&packet.data).map_err(damaged)?);
            continue;
        }
        let decoder = decoder.as_mut().expect("non-Opus tracks have a symphonia decoder");
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
            Err(e) => return Err(read_error(e)),
        }
    }

    // Opus ends at the stream length, which counts the pre-skip: in Ogg the last granule
    // position (RFC 7845 section 4), in Matroska/WebM the segment duration in its own time
    // base (whole milliseconds, so up to 1 ms of padding can stay).
    if let (Some(o), Some(end)) = (&opus, params.n_frames) {
        let end = match params.time_base {
            Some(tb) => {
                let n = u128::from(end) * u128::from(tb.numer) * u128::from(OPUS_RATE) / u128::from(tb.denom.max(1));
                u64::try_from(n).unwrap_or(u64::MAX)
            }
            None => end,
        };
        let keep = end.saturating_sub(o.pre_skip as u64) as usize;
        if keep > 0 {
            samples.truncate(keep);
        }
    }

    if rate == 0 || samples.is_empty() {
        return Err(damaged("no sound could be decoded".into()));
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

const OPUS_RATE: u32 = 48_000;

/// Ogg Opus packets to mono samples, with the pre-skip and output gain of RFC 7845.
struct OpusDecoder {
    decoder: opuscule::Decoder,
    channels: usize,
    pre_skip: usize,
    /// Decoded frames still to drop for the pre-skip.
    skip: usize,
    gain: f32,
    buffer: Vec<f32>,
}

impl OpusDecoder {
    /// From the `OpusHead` identification header (RFC 7845 section 5.1).
    fn new(head: &[u8]) -> Result<Self, String> {
        if head.len() < 19 || &head[..8] != b"OpusHead" {
            return Err("Opus track without an OpusHead header".into());
        }
        let channels = head[9] as usize;
        if head[18] != 0 || !(1..=2).contains(&channels) {
            return Err(format!("Opus with {channels} channels is not supported"));
        }
        let pre_skip = u16::from_le_bytes([head[10], head[11]]) as usize;
        let gain_db = i16::from_le_bytes([head[16], head[17]]) as f32 / 256.0;
        let layout = if channels == 1 { opuscule::Channels::Mono } else { opuscule::Channels::Stereo };
        Ok(Self {
            decoder: opuscule::Decoder::new(opuscule::SampleRate::Hz48000, layout),
            channels,
            pre_skip,
            skip: pre_skip,
            gain: 10f32.powf(gain_db / 20.0),
            // 120 ms, the longest packet.
            buffer: vec![0.0; 5760 * channels],
        })
    }

    fn decode(&mut self, packet: &[u8]) -> Result<Vec<f32>, String> {
        let frames = match self.decoder.decode(Some(packet), &mut self.buffer, false) {
            Ok(frames) => frames,
            // A damaged packet is skipped, as a player would.
            Err(opuscule::Error::InvalidPacket) => return Ok(Vec::new()),
            Err(e) => return Err(format!("Opus: {e}")),
        };
        let start = self.skip.min(frames);
        self.skip -= start;
        let ch = self.channels;
        Ok(self.buffer[start * ch..frames * ch]
            .chunks(ch)
            .map(|f| f.iter().sum::<f32>() / ch as f32 * self.gain)
            .collect())
    }
}
