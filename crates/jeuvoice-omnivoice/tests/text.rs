//! The OmniVoice text path (plan Phase 2, Task 11): the frame estimate, the packed sequence
//! and the Qwen3 tokenizer.
//!
//! `tests/data/text_vectors.json` holds what the engine produced for the three scripts:
//! the token ids, the step-0 sequence and the target frame count.

use std::path::PathBuf;

use jeuvoice_omnivoice::duration::estimate_frames;
use jeuvoice_omnivoice::text::{Bpe, build_layout};

fn vectors() -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/text_vectors.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("the vectors are valid JSON")
}

fn ints(value: &serde_json::Value) -> Vec<i32> {
    value.as_array().expect("an array").iter().map(|v| v.as_i64().expect("an integer") as i32).collect()
}

/// The same for a 2-D array, flattened in row-major order.
fn ints_2d(value: &serde_json::Value) -> Vec<i32> {
    value.as_array().expect("an array").iter().flat_map(ints).collect()
}

#[test]
fn frame_estimate_matches_the_engine() {
    let v = vectors();
    let ref_text = v["ref_text"].as_str().expect("the reference text");
    let speed = v["speed"].as_f64().expect("the speed") as f32;
    for script in v["scripts"].as_array().expect("the scripts") {
        let text = script["text"].as_str().expect("the text");
        let prompt_len = script["prompt_len"].as_u64().expect("the prompt length") as usize;
        let want = script["target_len"].as_u64().expect("the target length") as usize;
        let got = estimate_frames(text, Some(ref_text), Some(prompt_len), speed);
        assert_eq!(got, want, "{}: the frame estimate", script["id"]);
    }
}

#[test]
#[ignore = "needs the Qwen3 tokenizer.json of the checkpoint"]
fn the_layout_matches_the_engine() {
    let v = vectors();
    let dir = jeuvoice_omnivoice::config::model_dir(
        &jeuvoice_omnivoice::config::default_cache_root(),
        std::env::var("JEUVOICE_OMNIVOICE_MODEL").ok().as_deref(),
    )
    .expect("the model directory");
    let bpe = Bpe::load(&dir.join("tokenizer.json")).expect("the tokenizer");
    let ref_text = v["ref_text"].as_str().expect("the reference text");
    let lang = v["language"].as_str().expect("the language");
    for script in v["scripts"].as_array().expect("the scripts") {
        let text = script["text"].as_str().expect("the text");
        let prompt_len = script["prompt_len"].as_u64().expect("the prompt length") as usize;
        let target_len = script["target_len"].as_u64().expect("the target length") as usize;
        let prompt = ints_2d(&script["prompt_ids"]);
        let layout = build_layout(&bpe, 8, 1024, text, Some(ref_text), Some(lang), None, &prompt, prompt_len, target_len)
            .expect("the layout");
        if let Some(want) = script.get("ids0") {
            assert_eq!(layout.ids, ints_2d(want), "{}: the ids of the step-0 sequence", script["id"]);
            assert_eq!(layout.audio_mask, ints(&script["mask0"]), "{}: the audio mask", script["id"]);
        }
        // The style and the text blocks are the same on every script, so check them too.
        let style = jeuvoice_omnivoice::text::style_ids(&bpe, Some(lang), None, true).unwrap();
        assert_eq!(style, ints(&script["style_ids"]), "{}: the style ids", script["id"]);
        assert_eq!(layout.len, layout.ids.len() / 8, "{}: the row length", script["id"]);
    }
}

/// A tokenizer small enough to read: two letters, one merge, one added token.
fn synthetic() -> String {
    r#"{
      "model": {
        "vocab": {"a": 0, "b": 1, "ab": 2, "\u0120": 3, "\u00c4\u0141": 4, "\u00c4": 5, "\u0141": 6},
        "merges": [["a", "b"], ["\u00c4", "\u0141"]]
      },
      "added_tokens": [{"content": "<|x|>", "id": 7, "special": true}],
      "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
        {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
      ]},
      "normalizer": {"type": "NFC"}
    }"#
    .to_string()
}

#[test]
fn the_bpe_merges_the_byte_level_alphabet() {
    let dir = tempfile::tempdir().expect("a temp directory");
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, synthetic()).expect("the fixture");
    let bpe = Bpe::load(&path).expect("the tokenizer");

    // "ab" is one merge; "a b" keeps the space as `Ġ`; the added token splits the text.
    assert_eq!(bpe.encode("ab").unwrap(), vec![2]);
    assert_eq!(bpe.encode("a b").unwrap(), vec![0, 3, 1]);
    assert_eq!(bpe.encode("a<|x|>b").unwrap(), vec![0, 7, 1]);

    // The two bytes of "ğ" become `ÄŁ`, the byte-level alphabet of GPT-2.
    assert_eq!(bpe.encode("ğ").unwrap(), vec![4]);

    // NFC: the decomposed "ğ" (g and a combining breve) becomes the composed one.
    assert_eq!(bpe.encode("g\u{0306}").unwrap(), vec![4]);
}

/// The engine tokenizes only its own nonverbal tags on their own; any other bracket is
/// ordinary text. `tests/data/tag_vectors.json` holds the engine's ids for both kinds.
#[test]
#[ignore = "needs the Qwen3 tokenizer.json of the checkpoint"]
fn the_tags_follow_the_engine_pattern() {
    let dir = jeuvoice_omnivoice::config::model_dir(
        &jeuvoice_omnivoice::config::default_cache_root(),
        std::env::var("JEUVOICE_OMNIVOICE_MODEL").ok().as_deref(),
    )
    .expect("the model directory");
    let bpe = Bpe::load(&dir.join("tokenizer.json")).expect("the tokenizer");
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/tag_vectors.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let samples: serde_json::Value = serde_json::from_slice(&bytes).expect("valid JSON");
    for (text, want) in samples.as_object().expect("an object") {
        assert_eq!(bpe.encode(text).expect("the encoding"), ints(want), "{text:?}");
    }
}
#[test]
fn only_the_engine_tags_are_nonverbal() {
    use jeuvoice_omnivoice::text::is_nonverbal_tag;
    assert!(is_nonverbal_tag("[laughter]"));
    assert!(is_nonverbal_tag("[dissatisfaction-hnn]"));
    assert!(!is_nonverbal_tag("[foo]"), "a bracket the engine does not list is ordinary text");
    assert!(!is_nonverbal_tag("a[laughter]"), "a tag glued to text is not a whole tag");
}

#[test]
fn an_unknown_byte_is_an_error() {
    let dir = tempfile::tempdir().expect("a temp directory");
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, synthetic()).expect("the fixture");
    let bpe = Bpe::load(&path).expect("the tokenizer");
    // The fixture has no byte for "z", so the encoder must say so instead of dropping it.
    let e = bpe.encode("z").unwrap_err().to_string();
    assert!(e.contains("no token"), "{e}");
}