use jeuvoice_core::script_diff::{DiffOp, script_diff};
use jeuvoice_core::transcript::{Transcript, Word};
use jeuvoice_core::turkish::{normalize, number_to_words, tr_lower};

fn heard(text: &str) -> Transcript {
    Transcript {
        language: "tr".into(),
        words: text
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| Word { text: w.into(), start: i as f64, end: i as f64 + 0.5, prob: 1.0 })
            .collect(),
    }
}

fn norm(s: &str) -> Vec<String> {
    normalize(s)
}

#[test]
fn turkish_lowercase() {
    assert_eq!(tr_lower("İSTANBUL IŞIK"), "istanbul ışık");
}

#[test]
fn numbers() {
    assert_eq!(number_to_words(0), "sıfır");
    assert_eq!(number_to_words(21), "yirmi bir");
    assert_eq!(number_to_words(100), "yüz");
    assert_eq!(number_to_words(1000), "bin");
    assert_eq!(number_to_words(2023), "iki bin yirmi üç");
    assert_eq!(number_to_words(1_000_000), "bir milyon");
    assert_eq!(number_to_words(1_250_000), "bir milyon iki yüz elli bin");
}

#[test]
fn number_forms() {
    assert_eq!(
        normalize("2.000 kişi, %50 indirim, 1,5 saat, 3. bölüm"),
        ["iki", "bin", "kişi", "yüzde", "elli", "indirim", "bir", "virgül", "beş", "saat", "üçüncü", "bölüm"]
    );
}

#[test]
fn ordinals() {
    assert_eq!(norm("21. gün"), ["yirmi", "birinci", "gün"]);
    assert_eq!(norm("4. kat"), ["dördüncü", "kat"]);
    assert_eq!(norm("10. kat"), ["onuncu", "kat"]);
    assert_eq!(norm("100. yıl"), ["yüzüncü", "yıl"]);
    // Next word starts uppercase: the dot ends a sentence, so the number stays cardinal.
    assert_eq!(norm("Yıl 1990. Sonra"), ["yıl", "bin", "dokuz", "yüz", "doksan", "sonra"]);
}

#[test]
fn plain_year_stays_cardinal() {
    assert_eq!(norm("2023 yılında çıktı."), ["iki", "bin", "yirmi", "üç", "yılında", "çıktı"]);
}

#[test]
fn number_then_dot_at_end_is_ordinal() {
    // Known limit: "Sayfa 3." at the end of a text is read as an ordinal ("üçüncü").
    assert_eq!(norm("Sayfa 3."), ["sayfa", "üçüncü"]);
}

#[test]
fn decimal_with_leading_zero() {
    assert_eq!(norm("1,05"), ["bir", "virgül", "sıfır", "beş"]);
}

#[test]
fn digits_match_words() {
    let ops = script_diff("2023 yılında çıktı.", &heard("iki bin yirmi üç yılında çıktı"));
    assert_eq!(
        ops,
        vec![
            DiffOp::Same { script: 0, heard: 0 },
            DiffOp::Same { script: 1, heard: 4 },
            DiffOp::Same { script: 2, heard: 5 },
        ]
    );
}

#[test]
fn heard_digits_match_script_words() {
    let ops = script_diff("iki bin yirmi üç yılında", &heard("2023 yılında"));
    assert!(ops.iter().all(|o| matches!(o, DiffOp::Same { .. })), "{ops:?}");
    assert_eq!(ops.len(), 5);
}

#[test]
fn skipped_and_added_words() {
    let ops = script_diff("Joel ve Ellie tünele giriyor", &heard("Joel Ellie hemen tünele giriyor"));
    assert!(ops.contains(&DiffOp::Missing { script: 1 }));
    assert!(ops.contains(&DiffOp::Added { heard: 2 }));
    assert_eq!(ops.len(), 6);
}

#[test]
fn changed_word() {
    let ops = script_diff("damara", &heard("damaraa"));
    assert_eq!(ops, vec![DiffOp::Changed { script: 0, heard: 0 }]);
}

#[test]
fn partly_wrong_number_is_one_op() {
    let ops = script_diff("2023 yılı", &heard("iki bin yirmi dört yılı"));
    assert_eq!(ops, vec![DiffOp::Changed { script: 0, heard: 0 }, DiffOp::Same { script: 1, heard: 4 }]);
}

#[test]
fn empty_inputs() {
    assert!(script_diff("", &Transcript::default()).is_empty());
    assert_eq!(script_diff("merhaba", &Transcript::default()), vec![DiffOp::Missing { script: 0 }]);
}
