//! Audio playback with cpal: play, pause, seek and a live position for the playhead.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use jeuvoice_core::audio::{Audio, resample};

const NO_DEVICE: &str = "No sound output found. Connect speakers or headphones.";

/// Pure position logic. `len` and `pos` count samples at the device rate `rate`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cursor {
    pub len: u64,
    pub rate: u32,
    pub pos: u64,
    pub playing: bool,
}

impl Cursor {
    pub fn new(len: u64, rate: u32) -> Self {
        Self { len, rate, pos: 0, playing: false }
    }

    /// Move to `seconds`, clamped to [0, len/rate].
    pub fn seek(&mut self, seconds: f64) {
        let s = if seconds.is_finite() { seconds.max(0.0) } else { 0.0 };
        self.pos = ((s * self.rate as f64).round() as u64).min(self.len);
    }

    /// Move forward by `frames`. At the end, stop at `len` and pause.
    pub fn advance(&mut self, frames: u64) {
        self.pos = self.pos.saturating_add(frames).min(self.len);
        if self.pos >= self.len {
            self.playing = false;
        }
    }

    pub fn position_seconds(&self) -> f64 {
        self.pos as f64 / self.rate as f64
    }
}

/// Plays one loaded clip on the default output device.
///
/// Not `Send`: the cpal stream must stay on the thread that created it (the UI thread).
pub struct Playback {
    _stream: cpal::Stream,
    rate: u32,
    samples: Arc<Mutex<Arc<Vec<f32>>>>,
    pos: Arc<AtomicU64>,
    playing: Arc<AtomicBool>,
}

impl Playback {
    pub fn new() -> Result<Self, String> {
        let device = cpal::default_host().default_output_device().ok_or(NO_DEVICE)?;
        let config = device.default_output_config().map_err(|_| NO_DEVICE.to_string())?;
        let rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let format = config.sample_format();
        let config: cpal::StreamConfig = config.into();

        let samples: Arc<Mutex<Arc<Vec<f32>>>> = Arc::new(Mutex::new(Arc::new(Vec::new())));
        let pos = Arc::new(AtomicU64::new(0));
        let playing = Arc::new(AtomicBool::new(false));

        let (s, p, pl) = (samples.clone(), pos.clone(), playing.clone());
        let err_fn = |e| eprintln!("sound output error: {e}");
        let stream = match format {
            cpal::SampleFormat::F32 => device.build_output_stream(
                &config,
                move |out: &mut [f32], _: &cpal::OutputCallbackInfo| fill(out, channels, &s, &p, &pl, rate),
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_output_stream(
                &config,
                move |out: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    let mut tmp = vec![0.0f32; out.len()];
                    fill(&mut tmp, channels, &s, &p, &pl, rate);
                    for (o, t) in out.iter_mut().zip(tmp) {
                        *o = (t.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    }
                },
                err_fn,
                None,
            ),
            _ => return Err("This sound output is not supported.".to_string()),
        }
        .map_err(|e| format!("Could not open the sound output: {e}"))?;
        stream
            .play()
            .map_err(|e| format!("Could not start the sound output: {e}"))?;

        Ok(Self { _stream: stream, rate, samples, pos, playing })
    }

    /// Load a clip (resampled once to the device rate). Stops playback and rewinds.
    pub fn load(&mut self, a: Arc<Audio>) {
        let data = if a.rate == self.rate { a.samples.clone() } else { resample(&a, self.rate).samples };
        self.playing.store(false, Ordering::SeqCst);
        self.pos.store(0, Ordering::SeqCst);
        *self.samples.lock().unwrap() = Arc::new(data);
    }

    pub fn play(&mut self) {
        let c = self.cursor();
        if c.pos >= c.len {
            self.pos.store(0, Ordering::SeqCst);
        }
        self.playing.store(c.len > 0, Ordering::SeqCst);
    }

    pub fn pause(&mut self) {
        self.playing.store(false, Ordering::SeqCst);
    }

    /// Seek to `t` seconds, clamped to the clip.
    pub fn seek(&mut self, t: f64) {
        let mut c = self.cursor();
        c.seek(t);
        self.pos.store(c.pos, Ordering::SeqCst);
    }

    /// Position in seconds.
    pub fn position(&self) -> f64 {
        self.cursor().position_seconds()
    }

    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::SeqCst)
    }

    fn cursor(&self) -> Cursor {
        let len = self.samples.lock().unwrap().len() as u64;
        Cursor {
            len,
            rate: self.rate,
            pos: self.pos.load(Ordering::SeqCst),
            playing: self.playing.load(Ordering::SeqCst),
        }
    }
}

/// Audio callback body: copy mono samples to every channel, or write silence.
fn fill(
    out: &mut [f32],
    channels: usize,
    samples: &Mutex<Arc<Vec<f32>>>,
    pos: &AtomicU64,
    playing: &AtomicBool,
    rate: u32,
) {
    out.fill(0.0);
    if !playing.load(Ordering::SeqCst) {
        return;
    }
    let Ok(guard) = samples.try_lock() else { return };
    let data = guard.clone();
    drop(guard);

    let start = pos.load(Ordering::SeqCst);
    let mut c = Cursor { len: data.len() as u64, rate, pos: start, playing: true };
    let frames = (out.len() / channels) as u64;
    let n = frames.min(c.len.saturating_sub(start)) as usize;
    for i in 0..n {
        let v = data[start as usize + i];
        out[i * channels..(i + 1) * channels].fill(v);
    }
    c.advance(frames);
    // A seek during this callback wins: only store if the position did not change.
    if pos.compare_exchange(start, c.pos, Ordering::SeqCst, Ordering::SeqCst).is_ok() && !c.playing {
        playing.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seek_clamps_to_length() {
        let mut c = Cursor::new(48_000, 48_000);
        c.seek(-3.0);
        assert_eq!(c.pos, 0);
        c.seek(0.5);
        assert_eq!(c.pos, 24_000);
        c.seek(99.0);
        assert_eq!(c.pos, 48_000);
        assert_eq!(c.position_seconds(), 1.0);
    }

    #[test]
    fn advance_stops_at_end_and_pauses() {
        let mut c = Cursor::new(1_000, 48_000);
        c.playing = true;
        c.advance(600);
        assert_eq!((c.pos, c.playing), (600, true));
        c.advance(600);
        assert_eq!((c.pos, c.playing), (1_000, false));
    }

    #[test]
    fn position_in_seconds_uses_the_source_rate() {
        // One second at 44.1 kHz, resampled to a 48 kHz device: still one second.
        let src = Audio { samples: vec![0.1; 44_100], rate: 44_100 };
        let out = resample(&src, 48_000);
        let mut c = Cursor::new(out.samples.len() as u64, 48_000);
        c.advance(24_000);
        assert!((c.position_seconds() - 0.5).abs() < 0.01);
        c.seek(1.0);
        assert!((c.pos as i64 - 48_000).abs() <= 2);
    }

    /// Manual check: needs a sound device. Run with `-- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn plays_a_file_for_two_seconds() {
        let path = std::path::Path::new("D:/MLProjelerim/tts_lab/data/ref/sesim/ses_001_0007.wav");
        let audio = Arc::new(jeuvoice_core::audio::decode(path).unwrap());
        let mut p = Playback::new().unwrap();
        p.load(audio);
        p.play();
        std::thread::sleep(std::time::Duration::from_secs(2));
        let pos = p.position();
        println!("position after 2 s: {pos:.3}");
        assert!(p.is_playing());
        assert!((pos - 2.0).abs() < 0.3, "position {pos}");
    }
}
