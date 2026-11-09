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
