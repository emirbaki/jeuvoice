//! The frame estimate of the engine: `RuleDurationEstimator` (`utils/duration.py`). The model
//! turns the text into a frame count from per-character weights: one Latin letter is 1.0, a
//! space 0.2, a punctuation mark 0.5, a digit 3.5. The reference text and its frame count give
//! the speed of the voice, so the same text at another speed scales the count.

use unicode_general_category::{GeneralCategory, get_general_category};

use crate::duration_table::{RANGES, weight_of};

/// The weight of one character, as `_get_char_weight` (duration.py:145).
fn char_weight(c: char) -> f64 {
    let code = c as u32;
    if (0x41..=0x5A).contains(&code) || (0x61..=0x7A).contains(&code) {
        return weight_of("latin");
    }
    if code == 0x20 {
        return weight_of("space");
    }
    // The Arabic tatweel is a mark with no sound.
    if code == 0x0640 {
        return weight_of("mark");
    }
    use GeneralCategory::*;
    match get_general_category(c) {
        NonspacingMark | SpacingMark | EnclosingMark => return weight_of("mark"),
        ConnectorPunctuation | DashPunctuation | OpenPunctuation | ClosePunctuation | InitialPunctuation
        | FinalPunctuation | OtherPunctuation | MathSymbol | CurrencySymbol | ModifierSymbol | OtherSymbol => {
            return weight_of("punctuation");
        }
        SpaceSeparator | LineSeparator | ParagraphSeparator => return weight_of("space"),
        DecimalNumber | LetterNumber | OtherNumber => return weight_of("digit"),
        _ => {}
    }
    // The block table: the first range whose end is not below this code point.
    let index = RANGES.partition_point(|(end, _)| *end < code);
    if let Some((_, script)) = RANGES.get(index) {
        weight_of(script)
    } else if code > 0x20000 {
        weight_of("cjk")
    } else {
        weight_of("default")
    }
}

/// The sum of the weights of every character of `text`.
pub fn total_weight(text: &str) -> f64 {
    text.chars().map(char_weight).sum()
}

/// The estimated duration, in the same unit as `ref_frames` (frames of 25 per second).
///
/// The engine boosts a very short estimate with a cube root, so a one-word line does not
/// become five frames of audio.
pub fn estimate_duration(text: &str, ref_text: &str, ref_frames: usize) -> f64 {
    if ref_frames == 0 || ref_text.is_empty() {
        return 0.0;
    }
    let ref_weight = total_weight(ref_text);
    if ref_weight == 0.0 {
        return 0.0;
    }
    let speed_factor = ref_weight / ref_frames as f64;
    let estimate = total_weight(text) / speed_factor;
    if estimate < 50.0 {
        50.0 * (estimate / 50.0).powf(1.0 / 3.0)
    } else {
        estimate
    }
}

/// The frame count of `text`, as `_estimate_target_tokens` (`omnivoice.py:1198`).
///
/// Without a reference text the engine falls back to "Nice to meet you." and 25 frames.
/// `speed` above 1.0 makes the speech faster and the count shorter.
pub fn estimate_frames(text: &str, ref_text: Option<&str>, ref_frames: Option<usize>, speed: f32) -> usize {
    let (ref_text, ref_frames) = match (ref_text, ref_frames) {
        (Some(t), Some(n)) if !t.is_empty() => (t, n),
        _ => ("Nice to meet you.", 25),
    };
    let mut estimate = estimate_duration(text, ref_text, ref_frames);
    if speed > 0.0 && speed != 1.0 {
        estimate /= speed as f64;
    }
    (estimate.max(0.0) as usize).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_weights_follow_the_script() {
        assert_eq!(total_weight("ab"), 2.0);
        assert_eq!(total_weight("a b"), 2.2);
        assert_eq!(total_weight("a, b"), 2.7);
        assert_eq!(total_weight("7"), 3.5);
        // The Turkish letters are Latin: they all sit under the first block end (0x02AF).
        assert_eq!(total_weight("ğışçöüİ"), 7.0);
        // A combining mark is silent.
        assert_eq!(total_weight("a\u{0301}"), 1.0);
    }

    #[test]
    fn a_short_line_is_boosted_and_a_zero_frame_reference_gives_one() {
        assert!(estimate_duration("hi", "a much longer reference line", 100) < 50.0);
        assert_eq!(estimate_frames("hi", Some("reference"), Some(0), 1.0), 1);
        // Without a reference the engine falls back to "Nice to meet you." and 25 frames,\n        // and the cube-root boost turns 3.5 frames into 20.\n        assert_eq!(estimate_frames("hi", None, None, 1.0), 20);
    }

    #[test]
    fn speed_scales_the_count_down() {
        let base = estimate_frames("bu bir deneme cümlesidir", Some("kısa bir metin"), Some(40), 1.0);
        let fast = estimate_frames("bu bir deneme cümlesidir", Some("kısa bir metin"), Some(40), 2.0);
        assert!(fast < base);
    }
}
