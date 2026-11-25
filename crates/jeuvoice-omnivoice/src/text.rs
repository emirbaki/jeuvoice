//! The text side of OmniVoice (plan Phase 2, Task 11): the Qwen3 byte-level BPE, the packed
//! prompt sequence and the language codes. The engine builds the same sequence in
//! `_prepare_inference_inputs` (`omnivoice.py:1198`).

use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

use fancy_regex::Regex;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use crate::{Result, TtsError};

/// The ids of the special tokens of the prompt layout (`added_tokens` of `tokenizer.json`).
pub const DENOISE: i32 = 151_669;
pub const LANG_START: i32 = 151_670;
pub const LANG_END: i32 = 151_671;
pub const INSTRUCT_START: i32 = 151_672;
pub const INSTRUCT_END: i32 = 151_673;
pub const TEXT_START: i32 = 151_674;
pub const TEXT_END: i32 = 151_675;

/// The Qwen3 tokenizer of the checkpoint: a byte-level BPE with a GPT-2 pre-split.
pub struct Bpe {
    vocab: HashMap<String, u32>,
    /// `"left right"` to rank: the 1-byte parts never contain a space, because the byte-level
    /// map turns a space into `Ġ`.
    ranks: HashMap<String, u32>,
    /// The added tokens, longest first, so the first match is the longest one.
    added: Vec<(String, u32)>,
    split: Regex,
    byte_chars: [char; 256],
}

/// The GPT-2 byte alphabet: the printable bytes map to themselves, the others to 256 and up.
fn byte_to_char() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut next = 0u32;
    for b in 0u32..256 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        if printable {
            map[b as usize] = char::from_u32(b).expect("a byte below 256 is a char");
        } else {
            map[b as usize] = char::from_u32(256 + next).expect("a private code point");
            next += 1;
        }
    }
    map
}

impl Bpe {
    /// Reads `tokenizer.json` of the checkpoint.
    pub fn load(path: &Path) -> Result<Bpe> {
        let bytes = std::fs::read(path).map_err(|e| TtsError::ModelLoad(format!("{}: {e}", path.display())))?;
        let json: Value = serde_json::from_slice(&bytes)
            .map_err(|e| TtsError::ModelLoad(format!("{}: {e}", path.display())))?;
        let vocab = json["model"]["vocab"]
            .as_object()
            .ok_or_else(|| TtsError::ModelLoad("tokenizer.json: no model.vocab".into()))?;
        let mut vocab_map = HashMap::with_capacity(vocab.len());
        for (part, id) in vocab {
            let id = id.as_u64().ok_or_else(|| TtsError::ModelLoad(format!("vocab: {part} has no id")))?;
            vocab_map.insert(part.clone(), id as u32);
        }
        let merges = json["model"]["merges"]
            .as_array()
            .ok_or_else(|| TtsError::ModelLoad("tokenizer.json: no model.merges".into()))?;
        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, pair) in merges.iter().enumerate() {
            let pair = pair
                .as_array()
                .ok_or_else(|| TtsError::ModelLoad(format!("merge {rank} is not a pair")))?;
            let (a, b) = (pair[0].as_str(), pair[1].as_str());
            if let (Some(a), Some(b)) = (a, b) {
                ranks.insert(format!("{a} {b}"), rank as u32);
            }
        }
        let mut added: Vec<(String, u32)> = json["added_tokens"]
            .as_array()
            .map(|list| {
                list.iter()
                    .filter_map(|t| Some((t["content"].as_str()?.to_string(), t["id"].as_u64()? as u32)))
                    .collect()
            })
            .unwrap_or_default();
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| b.0.cmp(&a.0)));
        let pattern = json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
            .as_str()
            .ok_or_else(|| TtsError::ModelLoad("tokenizer.json: no pre-split pattern".into()))?;
        let split = Regex::new(pattern).map_err(|e| TtsError::ModelLoad(format!("the pre-split pattern: {e}")))?;
        Ok(Bpe { vocab: vocab_map, ranks, added, split, byte_chars: byte_to_char() })
    }

    /// The ids of `text`, as the engine's tokenizer gives them.
    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let normalized: String = text.nfc().collect();
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < normalized.len() {
            match self.find_added(&normalized[start..]) {
                Some((at, len, id)) => {
                    if at > 0 {
                        self.encode_plain(&normalized[start..start + at], &mut out)?;
                    }
                    out.push(id as i32);
                    start += at + len;
                }
                None => {
                    self.encode_plain(&normalized[start..], &mut out)?;
                    break;
                }
            }
        }
        Ok(out)
    }

    /// The earliest added token, with its byte offset and length. A longer token wins a tie.
    fn find_added(&self, hay: &str) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (content, id) in &self.added {
            if let Some(at) = hay.find(content.as_str()) {
                let better = match best {
                    None => true,
                    Some((best_at, best_len, _)) => at < best_at || (at == best_at && content.len() > best_len),
                };
                if better {
                    best = Some((at, content.len(), *id));
                }
            }
        }
        best
    }

    /// The pre-split, the byte-level map and the merges of a segment without added tokens.
    fn encode_plain(&self, text: &str, out: &mut Vec<i32>) -> Result<()> {
        for found in self.split.find_iter(text) {
            let m = found.map_err(|e| TtsError::Render(format!("the pre-split of {text:?} failed: {e}")))?;
            let mapped: String = m.as_str().bytes().map(|b| self.byte_chars[b as usize]).collect();
            self.bpe_word(&mapped, out)?;
        }
        Ok(())
    }

    /// The merges of one pre-split word, lowest rank first, then the vocab ids.
    fn bpe_word(&self, word: &str, out: &mut Vec<i32>) -> Result<()> {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        while parts.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len() - 1 {
                let key = format!("{} {}", parts[i], parts[i + 1]);
                if let Some(&rank) = self.ranks.get(&key) {
                    if best.is_none_or(|(best_rank, _)| rank < best_rank) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, at)) = best else { break };
            let merged = format!("{}{}", parts[at], parts[at + 1]);
            parts.splice(at..at + 2, [merged]);
        }
        for part in parts {
            let id = self
                .vocab
                .get(&part)
                .ok_or_else(|| TtsError::Render(format!("the tokenizer has no token for {part:?}")))?;
            out.push(*id as i32);
        }
        Ok(())
    }
}

/// The packed sequence of one request: `[codebooks, len]` ids and the audio mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// `[codebooks, len]` i32, row major: one row per codebook.
    pub ids: Vec<i32>,
    /// `[len]` i32: 1 where the position is audio, else 0.
    pub audio_mask: Vec<i32>,
    /// The positions of the sequence.
    pub len: usize,
}

/// The language code of `input`, or `None` for "no language".
///
/// The engine warns and falls back to no language on a name it does not know. This function
/// returns an error instead: a typo must not change the voice.
pub fn resolve_language(input: Option<&str>) -> Result<Option<&'static str>> {
    let Some(input) = input else { return Ok(None) };
    let trimmed = input.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    crate::lang::code_of(trimmed)
        .map(Some)
        .ok_or_else(|| TtsError::Render(format!("the language '{trimmed}' is not one the model knows")))
}

/// `_combine_text` (`omnivoice.py:1698`): the reference transcript, then the text.
pub fn combine_text(ref_text: Option<&str>, text: &str) -> String {
    static SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new("[ \t]+").expect("a literal pattern"));
    static CJK: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new("(?<=[\u{4e00}-\u{9fff}])\\s+|\\s+(?=[\u{4e00}-\u{9fff}])").expect("a literal pattern")
    });
    let mut full = match ref_text {
        Some(r) if !r.is_empty() => format!("{} {}", r.trim(), text.trim()),
        _ => text.trim().to_string(),
    };
    full.retain(|c| c != '\r' && c != '\n');
    full = full.replace('\u{ff08}', "(").replace('\u{ff09}', ")");
    let collapsed = SPACES.replace_all(&full, " ");
    full = CJK.replace_all(&collapsed, "").into_owned();
    full
}

/// The engine's nonverbal tags, exactly: `_NONVERBAL_PATTERN` of `omnivoice.py:1651`. Any other
/// bracket is ordinary text and must go through the normal tokenizer path.
static NONVERBAL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\]").expect("a literal pattern")
});

/// True when `text` is exactly one of the engine's nonverbal tags.
pub fn is_nonverbal_tag(text: &str) -> bool {
    matches!(NONVERBAL.find(text), Ok(Some(m)) if m.start() == 0 && m.end() == text.len())
}
/// `_tokenize_with_nonverbal_tags` (`omnivoice.py:1658`): a bracketed tag is tokenized on its
/// own, so its ids do not depend on the language around it.
fn tokenize_with_nonverbal(bpe: &Bpe, text: &str) -> Result<Vec<i32>> {
    let mut out = Vec::new();
    let mut last = 0usize;
    for found in NONVERBAL.find_iter(text) {
        let tag = found.map_err(|e| TtsError::Render(format!("the tag split of {text:?} failed: {e}")))?;
        if tag.start() > last {
            out.extend(bpe.encode(&text[last..tag.start()])?);
        }
        out.extend(bpe.encode(tag.as_str())?);
        last = tag.end();
    }
    if last < text.len() {
        out.extend(bpe.encode(&text[last..])?);
    }
    Ok(out)
}

/// The style block of the prompt: `<|denoise|><|lang_start|>{lang}<|lang_end|>` and the
/// instruction block. `denoise` is present only when there is a prompt to clean the audio with.
pub fn style_ids(bpe: &Bpe, lang: Option<&str>, instruct: Option<&str>, has_prompt: bool) -> Result<Vec<i32>> {
    let mut text = String::new();
    if has_prompt {
        text.push_str("<|denoise|>");
    }
    text.push_str(&format!("<|lang_start|>{}<|lang_end|>", lang.unwrap_or("None")));
    text.push_str(&format!("<|instruct_start|>{}<|instruct_end|>", instruct.unwrap_or("None")));
    bpe.encode(&text)
}

/// Assembles the sequence from the token ids: the style, the wrapped text, the prompt frames
/// and the masked target frames.
#[allow(clippy::too_many_arguments)]
pub fn layout_from_ids(
    style: &[i32],
    text: &[i32],
    prompt: &[i32],
    prompt_len: usize,
    target_len: usize,
    codebooks: usize,
    mask_id: i32,
) -> Result<Layout> {
    if prompt.len() != codebooks * prompt_len {
        return Err(TtsError::Render(format!(
            "the prompt has {} values, not {codebooks} codebooks of {prompt_len}",
            prompt.len()
        )));
    }
    if target_len == 0 {
        return Err(TtsError::Render("the target has no frames".into()));
    }
    let prefix = style.len() + text.len();
    let len = prefix + prompt_len + target_len;
    let mut ids = vec![0i32; codebooks * len];
    for c in 0..codebooks {
        let row = &mut ids[c * len..(c + 1) * len];
        row[..style.len()].copy_from_slice(style);
        row[style.len()..prefix].copy_from_slice(text);
        row[prefix..prefix + prompt_len].copy_from_slice(&prompt[c * prompt_len..(c + 1) * prompt_len]);
        row[prefix + prompt_len..].fill(mask_id);
    }
    let mut audio_mask = vec![0i32; len];
    audio_mask[prefix..].fill(1);
    Ok(Layout { ids, audio_mask, len })
}

/// Encodes the text and assembles the sequence, as `_prepare_inference_inputs` does.
#[allow(clippy::too_many_arguments)]
pub fn build_layout(
    bpe: &Bpe,
    codebooks: usize,
    mask_id: i32,
    text: &str,
    ref_text: Option<&str>,
    lang: Option<&str>,
    instruct: Option<&str>,
    prompt: &[i32],
    prompt_len: usize,
    target_len: usize,
) -> Result<Layout> {
    let lang = resolve_language(lang)?;
    let style = style_ids(bpe, lang, instruct, prompt_len > 0)?;
    let full = combine_text(ref_text, text);
    let wrapped = format!("<|text_start|>{full}<|text_end|>");
    let text_ids = tokenize_with_nonverbal(bpe, &wrapped)?;
    layout_from_ids(&style, &text_ids, prompt, prompt_len, target_len, codebooks, mask_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_text_cleans_the_engine_way() {
        assert_eq!(combine_text(None, "  merhaba  dünya \n"), "merhaba dünya");
        assert_eq!(combine_text(Some("  kısa bir metin "), " ikinci cümle "), "kısa bir metin ikinci cümle");
        assert_eq!(combine_text(None, "a（b）c"), "a(b)c");
    }

    #[test]
    fn the_language_codes_follow_the_engine() {
        assert_eq!(resolve_language(Some("tr")).unwrap(), Some("tr"));
        assert_eq!(resolve_language(Some("Turkish")).unwrap(), Some("tr"));
        assert_eq!(resolve_language(Some(" TURKISH ")).unwrap(), Some("tr"));
        assert_eq!(resolve_language(Some("None")).unwrap(), None);
        assert_eq!(resolve_language(None).unwrap(), None);
        assert!(resolve_language(Some("klingon")).is_err());
    }

    #[test]
    fn the_layout_is_the_style_then_the_text_then_the_frames() {
        let style = vec![DENOISE, LANG_START];
        let text = vec![TEXT_START, 10, 11, TEXT_END];
        let prompt = (0..8).flat_map(|c| (0..3).map(move |j| c * 10 + j)).collect::<Vec<_>>();
        let layout = layout_from_ids(&style, &text, &prompt, 3, 2, 8, 1024).unwrap();
        assert_eq!(layout.len, 2 + 4 + 3 + 2);
        assert_eq!(layout.audio_mask, vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
        // Row c has its own prompt codes, and the target is the mask id on every row.
        for c in 0..8 {
            let row = &layout.ids[c * layout.len..(c + 1) * layout.len];
            assert_eq!(&row[..6], &[DENOISE, LANG_START, TEXT_START, 10, 11, TEXT_END]);
            assert_eq!(&row[6..9], &[c as i32 * 10, c as i32 * 10 + 1, c as i32 * 10 + 2]);
            assert_eq!(&row[9..], &[1024, 1024]);
        }
    }

    #[test]
    fn a_prompt_of_the_wrong_size_is_an_error() {
        assert!(layout_from_ids(&[], &[], &[1, 2, 3], 3, 1, 2, 1024).is_err());
        assert!(layout_from_ids(&[], &[], &[], 0, 0, 8, 1024).is_err());
    }
}
