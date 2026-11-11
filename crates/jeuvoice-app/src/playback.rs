//! Audio playback with cpal: play, pause, seek and a live position for the playhead.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use jeuvoice_core::audio::{Audio, resample};

fn no_device() -> &'static str {
    crate::i18n::tr!("No sound output found. Connect speakers or headphones.", "Ses çıkışı bulunamadı. Hoparlör veya kulaklık bağlayın.")
}

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

    #[cfg_attr(not(test), allow(dead_code))] // the app reads the lock-free mirror instead
    pub fn position_seconds(&self) -> f64 {
        self.pos as f64 / self.rate as f64
    }
}

/// Everything the audio callback and the UI methods share. Always changed under one lock.
struct State {
    samples: Arc<Vec<f32>>,
    cursor: Cursor,
}

/// Lock-free copies of the state for cheap reads on the UI thread.
struct Mirror {
    pos: AtomicU64,
    playing: AtomicBool,
}

impl Mirror {
    fn publish(&self, c: &Cursor) {
        self.pos.store(c.pos, Ordering::SeqCst);
        self.playing.store(c.playing, Ordering::SeqCst);
    }
}

/// Plays one loaded clip on the default output device.
///
/// Not `Send`: the cpal stream must stay on the thread that created it (the UI thread).
pub struct Playback {
    _stream: cpal::Stream,
    rate: u32,
    state: Arc<Mutex<State>>,
    mirror: Arc<Mirror>,
}

impl Playback {
    pub fn new() -> Result<Self, String> {
        let device = cpal::default_host().default_output_device().ok_or(no_device())?;
        let config = device
            .default_output_config()
            .map_err(|e| crate::i18n::trf!("Could not read the sound output settings: {e}", "Ses çıkışı ayarları okunamadı: {e}"))?;
        let rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let format = config.sample_format();
        let config: cpal::StreamConfig = config.into();

        let state = Arc::new(Mutex::new(State {
            samples: Arc::new(Vec::new()),
            cursor: Cursor::new(0, rate),
        }));
        let mirror = Arc::new(Mirror { pos: AtomicU64::new(0), playing: AtomicBool::new(false) });

        let (st, mi) = (state.clone(), mirror.clone());
        let err_fn = |e| eprintln!("sound output error: {e}");
        let stream = match format {
            cpal::SampleFormat::F32 => device.build_output_stream(
                &config,
                move |out: &mut [f32], _: &cpal::OutputCallbackInfo| run_callback(&st, &mi, out, channels),
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => {
                // Scratch buffer allocated once; it only grows if the device asks for more.
                let mut scratch = vec![0.0f32; 16_384 * channels];
                device.build_output_stream(
                    &config,
                    move |out: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        if scratch.len() < out.len() {
                            scratch.resize(out.len(), 0.0);
                        }
                        let tmp = &mut scratch[..out.len()];
                        run_callback(&st, &mi, tmp, channels);
                        for (o, t) in out.iter_mut().zip(tmp.iter()) {
                            *o = (t.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        }
                    },
                    err_fn,
                    None,
                )
            }
            _ => return Err(crate::i18n::tr!("This sound output is not supported.", "Bu ses çıkışı desteklenmiyor.").to_string()),
        }
        .map_err(|e| crate::i18n::trf!("Could not open the sound output: {e}", "Ses çıkışı açılamadı: {e}"))?;
        stream
            .play()
            .map_err(|e| crate::i18n::trf!("Could not start the sound output: {e}", "Ses çıkışı başlatılamadı: {e}"))?;

        Ok(Self { _stream: stream, rate, state, mirror })
    }

    /// Load a clip (resampled once to the device rate). Stops playback and rewinds.
    pub fn load(&mut self, a: Arc<Audio>) {
        let data = if a.rate == self.rate { a.samples.clone() } else { resample(&a, self.rate).samples };
        let mut st = self.state.lock().unwrap();
        st.cursor = Cursor::new(data.len() as u64, self.rate);
        st.samples = Arc::new(data);
        self.mirror.publish(&st.cursor);
    }

    pub fn play(&mut self) {
        let mut st = self.state.lock().unwrap();
        if st.cursor.pos >= st.cursor.len {
            st.cursor.pos = 0;
        }
        st.cursor.playing = st.cursor.len > 0;
        self.mirror.publish(&st.cursor);
    }

    pub fn pause(&mut self) {
        let mut st = self.state.lock().unwrap();
        st.cursor.playing = false;
        self.mirror.publish(&st.cursor);
    }

    /// Seek to `t` seconds, clamped to the clip.
    pub fn seek(&mut self, t: f64) {
        let mut st = self.state.lock().unwrap();
        st.cursor.seek(t);
        self.mirror.publish(&st.cursor);
    }

    /// Position in seconds.
    pub fn position(&self) -> f64 {
        self.mirror.pos.load(Ordering::SeqCst) as f64 / self.rate as f64
    }

    /// The sound output's sample rate.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    pub fn is_playing(&self) -> bool {
        self.mirror.playing.load(Ordering::SeqCst)
    }
}

/// Take the lock without waiting; write silence if the UI holds it.
fn run_callback(state: &Mutex<State>, mirror: &Mirror, out: &mut [f32], channels: usize) {
    match state.try_lock() {
        Ok(mut st) => {
            fill(&mut st, out, channels);
            mirror.publish(&st.cursor);
        }
        Err(_) => out.fill(0.0),
    }
}

/// Pure audio callback body: copy mono samples to every channel, then silence.
fn fill(state: &mut State, out: &mut [f32], channels: usize) {
    out.fill(0.0);
    if !state.cursor.playing {
        return;
    }
    let start = state.cursor.pos as usize;
    let frames = out.len() / channels;
    let n = frames.min(state.samples.len().saturating_sub(start));
    for i in 0..n {
        out[i * channels..(i + 1) * channels].fill(state.samples[start + i]);
    }
    state.cursor.advance(frames as u64);
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

    fn state(samples: Vec<f32>) -> State {
        let mut cursor = Cursor::new(samples.len() as u64, 48_000);
        cursor.playing = true;
        State { samples: Arc::new(samples), cursor }
    }

    #[test]
    fn fill_copies_mono_to_all_channels_and_advances() {
        let mut st = state(vec![0.1, 0.2, 0.3, 0.4]);
        let mut out = [9.0f32; 6];
        fill(&mut st, &mut out, 2);
        assert_eq!(out, [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]);
        assert_eq!(st.cursor.pos, 3);
        assert!(st.cursor.playing);
    }

    #[test]
    fn fill_writes_silence_after_the_end_and_stops() {
        let mut st = state(vec![0.5, 0.6]);
        let mut out = [9.0f32; 8];
        fill(&mut st, &mut out, 2);
        assert_eq!(out, [0.5, 0.5, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(st.cursor.pos, 2);
        assert!(!st.cursor.playing);
    }

    #[test]
    fn seek_after_loading_a_shorter_clip_starts_from_the_new_clip() {
        let mut st = state(vec![0.1; 1000]);
        let mut out = [0.0f32; 4];
        fill(&mut st, &mut out, 1);
        // Load a shorter clip, then seek inside it.
        st.samples = Arc::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        st.cursor = Cursor::new(6, 48_000);
        st.cursor.seek(2.0 / 48_000.0);
        st.cursor.playing = true;
        fill(&mut st, &mut out, 1);
        assert_eq!(out, [0.3, 0.4, 0.5, 0.6]);
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
