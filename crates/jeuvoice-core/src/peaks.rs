//! Waveform peaks: the min and max of the samples in each short bucket, for drawing.

use crate::audio::Audio;

/// Buckets per second that the app stores for drawing the waveform.
pub const PEAKS_PER_SECOND: u32 = 200;

/// The min and max sample of each bucket. Bucket `k` covers `k / per_second` to `(k + 1) / per_second` seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct Peaks {
    pub per_second: u32,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

/// Compute the peaks of `a` at `per_second` buckets per second.
pub fn peaks(a: &Audio, per_second: u32) -> Peaks {
    let mut min = Vec::new();
    let mut max = Vec::new();
    if a.rate == 0 || per_second == 0 {
        return Peaks { per_second, min, max };
    }
    let rate = a.rate as u64;
    let ps = per_second as u64;
    let len = a.samples.len() as u64;
    // Sample `i` belongs to bucket `i * ps / rate`, so bucket `k` starts at sample `ceil(k * rate / ps)`.
    let start_of = |k: u64| (k * rate).div_ceil(ps).min(len) as usize;
    let count = (len * ps).div_ceil(rate);
    for k in 0..count {
        let slice = &a.samples[start_of(k)..start_of(k + 1)];
        if slice.is_empty() {
            min.push(0.0);
            max.push(0.0);
        } else {
            min.push(slice.iter().copied().fold(f32::INFINITY, f32::min));
            max.push(slice.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        }
    }
    Peaks { per_second, min, max }
}

impl Peaks {
    /// Fold the stored buckets into `buckets` equal columns covering `start_s` to `end_s` seconds.
    ///
    /// A column takes the lowest min and highest max of the buckets it covers.
    /// A column narrower than one bucket uses the bucket that holds its start.
    /// A column past the end of the audio is `(0.0, 0.0)`.
    pub fn range(&self, start_s: f64, end_s: f64, buckets: usize) -> Vec<(f32, f32)> {
        let len = self.min.len();
        let ps = self.per_second as f64;
        let width = (end_s - start_s) / buckets as f64;
        (0..buckets)
            .map(|i| {
                let t0 = start_s + width * i as f64;
                let t1 = t0 + width;
                let lo = (t0 * ps).floor().max(0.0) as usize;
                if lo >= len {
                    return (0.0, 0.0);
                }
                let mut hi = (t1 * ps).floor().max(0.0) as usize;
                if hi <= lo {
                    hi = lo + 1;
                }
                let hi = hi.min(len);
                let lows = self.min[lo..hi].iter().copied().fold(f32::INFINITY, f32::min);
                let highs = self.max[lo..hi].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                (lows, highs)
            })
            .collect()
    }
}
