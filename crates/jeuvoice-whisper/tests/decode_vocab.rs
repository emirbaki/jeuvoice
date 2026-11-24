//! The vocab and the special tokens, as whisper.cpp builds them (`whisper_model_load`).

use jeuvoice_whisper::decode::vocab::{N_LANGUAGES, Vocab, lang_code, lang_id};
use jeuvoice_whisper::ggml::GgmlModel;

/// The 50257 words of the large-v3 file: id 220 is " " and id 50256 is empty, as in the real
/// file. The other words are made up.
fn file_words() -> Vec<Vec<u8>> {
    (0..50257)
        .map(|i| match i {
            220 => b" ".to_vec(),
            50256 => Vec::new(),
            _ => format!("w{i}").into_bytes(),
        })
        .collect()
}

#[test]
fn special_token_ids_of_large_v3() {
    let v = Vocab::new(51866, file_words()).unwrap();
    assert_eq!(v.n_vocab(), 51866);
    assert_eq!(v.eot, 50257);
    assert_eq!(v.sot, 50258);
    assert_eq!(v.lang_token(0), 50259);
    assert_eq!(v.lang_token(N_LANGUAGES - 1), 50358);
    assert_eq!(v.translate, 50359);
    assert_eq!(v.transcribe, 50360);
    assert_eq!(v.solm, 50361);
    assert_eq!(v.prev, 50362);
    assert_eq!(v.nosp, 50363);
    assert_eq!(v.not, 50364);
    assert_eq!(v.beg, 50365);
    assert_eq!(v.space, 220);
}

#[test]
fn names_of_the_extra_tokens() {
    let v = Vocab::new(51866, file_words()).unwrap();
    let t = |id| v.token_text_lossy(id);
    assert_eq!(t(220), " ");
    assert_eq!(t(50256), "");
    assert_eq!(t(50257), "[_EOT_]");
    assert_eq!(t(50258), "[_SOT_]");
    assert_eq!(t(50259), "[_LANG_en]");
    assert_eq!(t(50268), "[_LANG_tr]");
    assert_eq!(t(50358), "[_LANG_yue]");
    assert_eq!(t(50359), "[_TRANSLATE_]");
    assert_eq!(t(50360), "[_TRANSCRIBE_]");
    assert_eq!(t(50361), "[_SOLM_]");
    assert_eq!(t(50362), "[_PREV_]");
    assert_eq!(t(50363), "[_NOSP_]");
    assert_eq!(t(50364), "[_NOT_]");
    assert_eq!(t(50365), "[_BEG_]");
    assert_eq!(t(50366), "[_TT_1]");
    assert_eq!(t(51865), "[_TT_1500]");
}

#[test]
fn timestamp_and_text_classes() {
    let v = Vocab::new(51866, file_words()).unwrap();
    assert!(v.is_timestamp(50365) && v.is_timestamp(51865));
    assert!(!v.is_timestamp(50364));
    assert!(v.is_text(50256) && !v.is_text(50257));
}

#[test]
fn bad_utf8_is_lossy_as_in_whisper_rs() {
    let mut words = file_words();
    words[5] = vec![0xc3]; // the first half of a two-byte character
    let v = Vocab::new(51866, words).unwrap();
    assert_eq!(v.token_text_lossy(5), "\u{fffd}");
    assert_eq!(v.token_bytes(5), &[0xc3]);
}

#[test]
fn language_table() {
    assert_eq!(lang_id("en"), Some(0));
    assert_eq!(lang_id("tr"), Some(9));
    assert_eq!(lang_id("turkish"), Some(9));
    assert_eq!(lang_id("yue"), Some(99));
    assert_eq!(lang_id("xx"), None);
    assert_eq!(lang_code(9), Some("tr"));
    assert_eq!(lang_code(100), None);
    for i in 0..N_LANGUAGES {
        assert_eq!(lang_id(lang_code(i).unwrap()), Some(i), "language {i}");
    }
}

#[test]
fn english_only_and_broken_vocabs_give_errors() {
    let e = Vocab::new(51864, file_words()).unwrap_err().to_string();
    assert!(e.contains("speech model"), "{e}");
    let e = format!("{:?}", Vocab::new(51864, file_words()).unwrap_err());
    assert!(e.contains("multilingual"), "{e}");
    // More words in the file than n_vocab.
    assert!(Vocab::new(50000, file_words()).is_err());
}

#[test]
#[ignore = "needs the model file and the whisper-rs fixtures (Task 0)"]
fn texts_match_whisper_rs() {
    let model = GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap();
    let v = Vocab::from_ggml(&model).unwrap();
    // my_voice2 has a token with a character split over two tokens (U+FFFD in whisper-rs).
    let mut n = 0;
    for clip in ["my_voice", "my_voice2", "ses_001_0007"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../target/reference/whisper/{clip}.rs.json"));
        let json =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        // A small reader for the fixture lines: `{"id": N, ... "text": "..."}`.
        for line in json.lines().filter(|l| l.contains("\"id\": ")) {
            let id: u32 = line
                .split("\"id\": ")
                .nth(1)
                .unwrap()
                .split(',')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            let text = line
                .rsplit("\"text\": ")
                .next()
                .unwrap()
                .trim_end_matches(',')
                .trim_end_matches('}');
            let want: String = serde_like_unquote(text);
            assert_eq!(v.token_text_lossy(id), want, "{clip}: token {id}");
            n += 1;
        }
    }
    assert!(n > 300, "only {n} tokens in the fixture");
}

/// Unquotes the JSON strings that `dump_whisper_rs` writes (only `\"`, `\\` and `\uXXXX`).
fn serde_like_unquote(s: &str) -> String {
    let s = s
        .trim()
        .strip_prefix('"')
        .unwrap()
        .strip_suffix('"')
        .unwrap();
    let mut out = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next().unwrap() {
            'u' => {
                let hex: String = it.by_ref().take(4).collect();
                out.push(char::from_u32(u32::from_str_radix(&hex, 16).unwrap()).unwrap());
            }
            c => out.push(c),
        }
    }
    out
}
