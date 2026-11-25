//! The Whisper vocab and the special tokens, as `whisper_model_load` builds them.
//!
//! The ggml file stores the text tokens only (50257 words for large-v3). whisper.cpp gives
//! the ids from there up to `n_vocab` the names `[_EOT_]`, `[_SOT_]`, `[_LANG_xx]`,
//! `[_BEG_]`, `[_TT_k]` and so on. The ids of the special tokens move with the number of
//! languages.

use crate::ggml::GgmlModel;
use crate::WhisperError;

/// The number of languages in `LANGUAGES` (the `g_lang` table of whisper.cpp).
pub const N_LANGUAGES: usize = 100;

/// `(code, name)` by language index, as `g_lang` in whisper.cpp. The token of language `i`
/// is `SOT + 1 + i`.
const LANGUAGES: [(&str, &str); N_LANGUAGES] = [
    ("en", "english"), ("zh", "chinese"), ("de", "german"), ("es", "spanish"), ("ru", "russian"),
    ("ko", "korean"), ("fr", "french"), ("ja", "japanese"), ("pt", "portuguese"), ("tr", "turkish"),
    ("pl", "polish"), ("ca", "catalan"), ("nl", "dutch"), ("ar", "arabic"), ("sv", "swedish"),
    ("it", "italian"), ("id", "indonesian"), ("hi", "hindi"), ("fi", "finnish"), ("vi", "vietnamese"),
    ("he", "hebrew"), ("uk", "ukrainian"), ("el", "greek"), ("ms", "malay"), ("cs", "czech"),
    ("ro", "romanian"), ("da", "danish"), ("hu", "hungarian"), ("ta", "tamil"), ("no", "norwegian"),
    ("th", "thai"), ("ur", "urdu"), ("hr", "croatian"), ("bg", "bulgarian"), ("lt", "lithuanian"),
    ("la", "latin"), ("mi", "maori"), ("ml", "malayalam"), ("cy", "welsh"), ("sk", "slovak"),
    ("te", "telugu"), ("fa", "persian"), ("lv", "latvian"), ("bn", "bengali"), ("sr", "serbian"),
    ("az", "azerbaijani"), ("sl", "slovenian"), ("kn", "kannada"), ("et", "estonian"), ("mk", "macedonian"),
    ("br", "breton"), ("eu", "basque"), ("is", "icelandic"), ("hy", "armenian"), ("ne", "nepali"),
    ("mn", "mongolian"), ("bs", "bosnian"), ("kk", "kazakh"), ("sq", "albanian"), ("sw", "swahili"),
    ("gl", "galician"), ("mr", "marathi"), ("pa", "punjabi"), ("si", "sinhala"), ("km", "khmer"),
    ("sn", "shona"), ("yo", "yoruba"), ("so", "somali"), ("af", "afrikaans"), ("oc", "occitan"),
    ("ka", "georgian"), ("be", "belarusian"), ("tg", "tajik"), ("sd", "sindhi"), ("gu", "gujarati"),
    ("am", "amharic"), ("yi", "yiddish"), ("lo", "lao"), ("uz", "uzbek"), ("fo", "faroese"),
    ("ht", "haitian creole"), ("ps", "pashto"), ("tk", "turkmen"), ("nn", "nynorsk"), ("mt", "maltese"),
    ("sa", "sanskrit"), ("lb", "luxembourgish"), ("my", "myanmar"), ("bo", "tibetan"), ("tl", "tagalog"),
    ("mg", "malagasy"), ("as", "assamese"), ("tt", "tatar"), ("haw", "hawaiian"), ("ln", "lingala"),
    ("ha", "hausa"), ("ba", "bashkir"), ("jw", "javanese"), ("su", "sundanese"), ("yue", "cantonese"),
];

/// The index of a language code ("tr") or name ("turkish"), as `whisper_lang_id`.
pub fn lang_id(lang: &str) -> Option<usize> {
    LANGUAGES.iter().position(|(c, _)| *c == lang).or_else(|| LANGUAGES.iter().position(|(_, n)| *n == lang))
}

/// The code of the language with index `id`, as `whisper_lang_str`.
pub fn lang_code(id: usize) -> Option<&'static str> {
    LANGUAGES.get(id).map(|(c, _)| *c)
}

/// The token texts and the ids of the special tokens.
#[derive(Clone, Debug)]
pub struct Vocab {
    words: Vec<Vec<u8>>,
    pub eot: u32,
    pub sot: u32,
    pub translate: u32,
    pub transcribe: u32,
    pub solm: u32,
    pub prev: u32,
    pub nosp: u32,
    /// "No timestamps".
    pub not: u32,
    /// The first timestamp token (0.00 s). `beg + k` is `k × 0.02 s`.
    pub beg: u32,
    /// The token of the text `" "` (the `suppress_blank` rule).
    pub space: u32,
}

impl Vocab {
    /// Reads the vocab of a ggml model.
    pub fn from_ggml(model: &GgmlModel) -> Result<Vocab, WhisperError> {
        let n_vocab = usize::try_from(model.hparams.n_vocab).unwrap_or(0);
        Vocab::new(n_vocab, model.vocab.clone())
    }

    /// Builds the vocab from the `n_vocab` of the model and the words of the file.
    pub fn new(n_vocab: usize, file_words: Vec<Vec<u8>>) -> Result<Vocab, WhisperError> {
        let bad = |why: String| Err(WhisperError::ModelLoad(why));
        // whisper.cpp: `is_multilingual` is `n_vocab >= 51865`, and the languages are
        // `n_vocab - 51765 - 1`. The GPU path supports the 100 languages of large-v3 only.
        if n_vocab < 51865 {
            return bad(format!("the GPU path supports multilingual large-v3 models only, not a vocab of {n_vocab} tokens"));
        }
        let n_lang = n_vocab - 51765 - 1;
        if n_lang != N_LANGUAGES {
            return bad(format!("the GPU path supports the {N_LANGUAGES} languages of large-v3, not {n_lang}"));
        }
        if file_words.len() > 50257 {
            return bad(format!("the vocab has {} words in the file, but the model has text tokens 0 to 50256 only", file_words.len()));
        }
        // The ids of whisper.cpp (line 439), moved for a multilingual model (line 1622).
        let dt = n_lang as u32 - 98;
        let (eot, sot) = (50257, 50258);
        let mut v = Vocab {
            words: Vec::with_capacity(n_vocab),
            eot,
            sot,
            translate: 50357 + dt,
            transcribe: 50358 + dt,
            solm: 50359 + dt,
            prev: 50360 + dt,
            nosp: 50361 + dt,
            not: 50362 + dt,
            beg: 50363 + dt,
            space: 0,
        };
        // As `token_to_id` in whisper.cpp, a later word with the same text wins.
        let Some(space) = file_words.iter().rposition(|w| w == b" ") else {
            return bad("the vocab has no token for the text \" \"".to_string());
        };
        v.space = space as u32;
        let n_file = file_words.len();
        v.words = file_words;
        for i in n_file as u32..n_vocab as u32 {
            let name = v.extra_name(i);
            v.words.push(name.into_bytes());
        }
        Ok(v)
    }

    /// The name that whisper.cpp gives to an id past the words of the file (line 1647).
    fn extra_name(&self, i: u32) -> String {
        match i {
            _ if i > self.beg => format!("[_TT_{}]", i - self.beg),
            _ if i == self.eot => "[_EOT_]".into(),
            _ if i == self.sot => "[_SOT_]".into(),
            _ if i == self.translate => "[_TRANSLATE_]".into(),
            _ if i == self.transcribe => "[_TRANSCRIBE_]".into(),
            _ if i == self.solm => "[_SOLM_]".into(),
            _ if i == self.prev => "[_PREV_]".into(),
            _ if i == self.nosp => "[_NOSP_]".into(),
            _ if i == self.not => "[_NOT_]".into(),
            _ if i == self.beg => "[_BEG_]".into(),
            _ if i > self.sot && i <= self.sot + N_LANGUAGES as u32 => {
                format!("[_LANG_{}]", lang_code((i - self.sot - 1) as usize).expect("a language index"))
            }
            _ => format!("[_extra_token_{i}]"),
        }
    }

    /// The number of tokens (the length of a logits row).
    pub fn n_vocab(&self) -> usize {
        self.words.len()
    }

    /// The token of the language with index `lang`.
    pub fn lang_token(&self, lang: usize) -> u32 {
        assert!(lang < N_LANGUAGES, "lang_token: language index {lang} is not below {N_LANGUAGES}");
        self.sot + 1 + lang as u32
    }

    /// True for a timestamp token (`beg` and up).
    pub fn is_timestamp(&self, id: u32) -> bool {
        id >= self.beg
    }

    /// True for a text token (below `eot`).
    pub fn is_text(&self, id: u32) -> bool {
        id < self.eot
    }

    /// The raw bytes of a token.
    ///
    /// # Panics
    /// If `id` is not below `n_vocab`.
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        &self.words[id as usize]
    }

    /// The text of a token, with U+FFFD for bytes that are not UTF-8, as `to_str_lossy` in
    /// whisper-rs gives it. A character split over two tokens becomes U+FFFD in both.
    ///
    /// # Panics
    /// If `id` is not below `n_vocab`.
    pub fn token_text_lossy(&self, id: u32) -> String {
        String::from_utf8_lossy(self.token_bytes(id)).into_owned()
    }
}
