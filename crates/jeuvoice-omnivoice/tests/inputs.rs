//! The input checks of a pass (plan Phase 2, Task 9). Pure: no GPU.

use jeuvoice_omnivoice::inputs::{Limits, check};

const LIMITS: Limits = Limits { codebooks: 2, vocab: 100, audio_vocab: 10 };

/// `rows = 1`, `t = 3`, codebles 2, the mask all text.
fn ids(mask: &[i32], value: i32) -> Vec<i32> {
    let mut ids = vec![value; 2 * 3];
    // The audio positions read the audio table, so keep their ids small there.
    for (p, &m) in mask.iter().enumerate() {
        if m == 1 {
            ids[p] = 5;
            ids[3 + p] = 5;
        }
    }
    ids
}

#[test]
fn a_valid_pass_passes() {
    let mask = [0, 1, 1];
    check(&ids(&mask, 7), &mask, 1, 3, LIMITS, Some(&[3])).unwrap();
    // The boundaries: id 0 and the last text id, a length of 0 and of t.
    check(&ids(&mask, 0), &mask, 1, 3, LIMITS, Some(&[0])).unwrap();
    let mut top = ids(&mask, 99);
    top[1] = 9; // an audio id, at the boundary of 0..10
    check(&top, &mask, 1, 3, LIMITS, Some(&[3])).unwrap();
}

#[test]
fn a_negative_text_id_is_an_error() {
    let mask = [0, 1, 1];
    let mut bad = ids(&mask, 7);
    bad[0] = -1;
    let e = check(&bad, &mask, 1, 3, LIMITS, None).unwrap_err().to_string();
    assert!(e.contains("text id -1") && e.contains("row 0") && e.contains("position 0"), "{e}");
}

#[test]
fn a_text_id_at_or_above_the_vocab_is_an_error() {
    let mask = [0, 1, 1];
    let mut bad = ids(&mask, 7);
    bad[0] = 100;
    assert!(check(&bad, &mask, 1, 3, LIMITS, None).is_err());
}

#[test]
fn a_negative_audio_id_is_an_error() {
    let mask = [0, 1, 1];
    let mut bad = ids(&mask, 7);
    bad[3 + 2] = -3; // codebook 1, position 2, an audio frame
    let e = check(&bad, &mask, 1, 3, LIMITS, None).unwrap_err().to_string();
    assert!(e.contains("audio id -3") && e.contains("position 2"), "{e}");
}

#[test]
fn an_audio_id_at_or_above_the_vocab_is_an_error() {
    let mask = [0, 1, 1];
    let mut bad = ids(&mask, 7);
    bad[3 + 1] = 10;
    assert!(check(&bad, &mask, 1, 3, LIMITS, None).is_err());
}

#[test]
fn a_mask_value_other_than_one_or_zero_is_an_error() {
    let mask = [0, 2, 1];
    let bad = ids(&mask, 7);
    let e = check(&bad, &mask, 1, 3, LIMITS, None).unwrap_err().to_string();
    assert!(e.contains("audio mask") && e.contains("position 1"), "{e}");
}

#[test]
fn a_valid_slice_of_the_wrong_length_is_an_error() {
    let mask = [0, 1, 1];
    assert!(check(&ids(&mask, 7), &mask, 1, 3, LIMITS, Some(&[3, 3])).is_err());
    assert!(check(&ids(&mask, 7), &mask, 1, 3, LIMITS, Some(&[])).is_err());
}

#[test]
fn a_valid_length_out_of_range_is_an_error() {
    let mask = [0, 1, 1];
    let ids = ids(&mask, 7);
    assert!(check(&ids, &mask, 1, 3, LIMITS, Some(&[-1])).is_err());
    assert!(check(&ids, &mask, 1, 3, LIMITS, Some(&[4])).is_err());
    assert!(check(&ids, &mask, 1, 3, LIMITS, Some(&[3])).is_ok());
}

#[test]
fn a_wrong_id_or_mask_length_is_an_error() {
    let mask = [0, 1, 1];
    assert!(check(&ids(&mask, 7)[..5], &mask, 1, 3, LIMITS, None).is_err());
    assert!(check(&ids(&mask, 7), &mask[..2], 1, 3, LIMITS, None).is_err());
}
