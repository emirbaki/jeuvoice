use jeuvoice_core::audio::Audio;
use jeuvoice_core::peaks::peaks;

/// One second at 8 kHz alternating between +0.8 and -0.8.
fn square_wave() -> Audio {
    let samples = (0..8000).map(|i| if i % 2 == 0 { 0.8 } else { -0.8 }).collect();
    Audio { samples, rate: 8000 }
}

#[test]
fn peaks_of_a_square_wave() {
    let p = peaks(&square_wave(), 100);
    assert_eq!(p.per_second, 100);
    assert_eq!(p.min.len(), 100);
    assert_eq!(p.max.len(), 100);
    for (lo, hi) in p.min.iter().zip(&p.max) {
        assert_eq!((*lo, *hi), (-0.8, 0.8));
    }
}

#[test]
fn range_folds_buckets() {
    let p = peaks(&square_wave(), 100);
    let cols = p.range(0.0, 1.0, 10);
    assert_eq!(cols.len(), 10);
    for col in cols {
        assert_eq!(col, (-0.8, 0.8));
    }
}

#[test]
fn range_past_the_end_is_flat() {
    let p = peaks(&square_wave(), 100);
    let cols = p.range(0.9, 1.5, 6);
    assert_eq!(cols.len(), 6);
    // Columns from 1.0 s on lie past the audio.
    for col in &cols[1..] {
        assert_eq!(*col, (0.0, 0.0));
    }
    assert_eq!(cols[0], (-0.8, 0.8));
}

#[test]
fn range_narrower_than_a_bucket_uses_the_bucket_at_its_start() {
    // 100 Hz audio, 10 buckets per second, so each bucket is 10 samples
    // and holds the value equal to its index.
    let samples = (0..100).map(|i| (i / 10) as f32).collect();
    let p = peaks(&Audio { samples, rate: 100 }, 10);
    // 20 columns of 0.05 s each: two columns per bucket.
    let cols = p.range(0.0, 1.0, 20);
    for (i, col) in cols.iter().enumerate() {
        let bucket = (i / 2) as f32;
        assert_eq!(*col, (bucket, bucket), "column {i}");
    }
}

fn sine(freq: f32, rate: u32) -> Audio {
    let samples = (0..rate).map(|i| 0.5 * (std::f32::consts::TAU * freq * i as f32 / rate as f32).sin()).collect();
    Audio { samples, rate }
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len() as f32
}

#[test]
fn a_high_tone_has_a_higher_rate_than_a_low_tone() {
    let high = peaks(&sine(4000.0, 24_000), 200);
    let low = peaks(&sine(200.0, 24_000), 200);
    assert_eq!(high.zcr.len(), high.min.len());
    assert!(mean(&high.zcr) > 5.0 * mean(&low.zcr), "{} vs {}", mean(&high.zcr), mean(&low.zcr));
}

#[test]
fn silence_has_rate_zero() {
    let p = peaks(&Audio { samples: vec![0.0; 24_000], rate: 24_000 }, 200);
    assert!(p.zcr.iter().all(|&z| z == 0.0), "silence crosses nothing");
}

#[test]
fn clipped_audio_keeps_a_finite_rate() {
    let p = peaks(&square_wave(), 100);
    assert!(p.zcr.iter().all(|z| z.is_finite() && (0.0..=1.0).contains(z)), "{:?}", &p.zcr[..4]);
}

#[test]
fn range_zcr_averages_buckets() {
    let p = peaks(&square_wave(), 100);
    let cols = p.range_zcr(0.0, 1.0, 10);
    assert_eq!(cols.len(), 10);
    assert!(cols.iter().all(|&z| z > 0.9), "a square wave at Nyquist crosses at each sample: {cols:?}");
    assert_eq!(p.range_zcr(2.0, 3.0, 4), vec![0.0; 4], "past the end is 0");
}
