//! Turkish text normalising for the script check: lowercase, no punctuation, numbers as words.
//! Pure functions only.

/// Lowercase with the Turkish dotted/dotless I rules: `I` -> `ı`, `İ` -> `i`.
pub fn tr_lower(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            'I' => out.push('ı'),
            'İ' => out.push('i'),
            _ => out.extend(c.to_lowercase()),
        }
    }
    out
}

const VOWELS: &str = "aeıioöuü";
const ONES: [&str; 10] = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"];
const TENS: [&str; 10] = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"];
const SCALES: [&str; 7] = ["", "bin", "milyon", "milyar", "trilyon", "katrilyon", "kentilyon"];

fn push_group(words: &mut Vec<&'static str>, g: u64) {
    let (h, t, o) = ((g / 100) as usize, ((g / 10) % 10) as usize, (g % 10) as usize);
    if h > 0 {
        if h > 1 {
            words.push(ONES[h]);
        }
        words.push("yüz");
    }
    if t > 0 {
        words.push(TENS[t]);
    }
    if o > 0 {
        words.push(ONES[o]);
    }
}

fn number_words(n: u64) -> Vec<&'static str> {
    if n == 0 {
        return vec!["sıfır"];
    }
    let mut groups = Vec::new();
    let mut rest = n;
    while rest > 0 {
        groups.push(rest % 1000);
        rest /= 1000;
    }
    let mut words = Vec::new();
    for (scale, &g) in groups.iter().enumerate().rev() {
        if g == 0 {
            continue;
        }
        // "bir" is dropped before "bin" (1000 = "bin"); it stays before "milyon" and larger.
        // "bir" before "yüz" is dropped inside `push_group`.
        if !(scale == 1 && g == 1) {
            push_group(&mut words, g);
        }
        if scale > 0 {
            words.push(SCALES[scale]);
        }
    }
    words
}

/// Turkish cardinal number as words, e.g. 2023 -> "iki bin yirmi üç".
pub fn number_to_words(n: u64) -> String {
    number_words(n).join(" ")
}

/// Ordinal form of one number word: "üç" -> "üçüncü", "yirmi" -> "yirminci".
fn ordinal_word(w: &str) -> String {
    let stem = if w == "dört" { "dörd" } else { w };
    let vowel = w.chars().rev().find(|c| VOWELS.contains(*c));
    let ends_in_vowel = w.chars().last().is_some_and(|c| VOWELS.contains(c));
    let s = match vowel {
        Some('a') | Some('ı') => 'ı',
        Some('e') | Some('i') => 'i',
        Some('o') | Some('u') => 'u',
        _ => 'ü',
    };
    if ends_in_vowel { format!("{stem}nc{s}") } else { format!("{stem}{s}nc{s}") }
}

/// Digits as words one by one (for numbers too big for u64).
fn digit_tokens(digits: &str) -> Vec<String> {
    digits
        .chars()
        .map(|c| if c == '0' { "sıfır".to_string() } else { ONES[c as usize - '0' as usize].to_string() })
        .collect()
}

fn digits_to_tokens(digits: &str, ordinal: bool) -> Vec<String> {
    let Ok(n) = digits.parse::<u64>() else { return digit_tokens(digits) };
    let mut words: Vec<String> = number_words(n).into_iter().map(String::from).collect();
    if ordinal && let Some(last) = words.last_mut() {
        *last = ordinal_word(last);
    }
    words
}

fn flush_letters(letters: &mut String, out: &mut Vec<String>) {
    if !letters.is_empty() {
        out.push(tr_lower(letters));
        letters.clear();
    }
}

fn is_group_at(chars: &[char], i: usize) -> bool {
    chars.get(i) == Some(&'.')
        && (1..=3).all(|k| chars.get(i + k).is_some_and(|d| d.is_ascii_digit()))
        && !chars.get(i + 4).is_some_and(|d| d.is_ascii_digit())
}

/// Normalise one whitespace-separated word. `dot_may_be_ordinal` tells whether a trailing "."
/// after a number may be read as an ordinal marker (next word starts lowercase, or none).
fn normalize_word(w: &str, dot_may_be_ordinal: bool) -> Vec<String> {
    let chars: Vec<char> = w.chars().collect();
    let mut out = Vec::new();
    let mut letters = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let percent = c == '%' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit());
        if !percent && !c.is_ascii_digit() {
            if c.is_alphanumeric() {
                letters.push(c);
            }
            i += 1;
            continue;
        }
        flush_letters(&mut letters, &mut out);
        if percent {
            out.push("yüzde".to_string());
            i += 1;
        }
        // Known limits: leading zeros are lost ("007" reads "yedi"). A trailing "50%" gives no "yüzde".
        // Integer part; "." between groups of exactly 3 digits is a thousands separator.
        let mut int = String::new();
        while i < chars.len() && chars[i].is_ascii_digit() {
            int.push(chars[i]);
            i += 1;
        }
        while is_group_at(&chars, i) {
            int.extend(&chars[i + 1..i + 4]);
            i += 4;
        }
        if chars.get(i) == Some(&',') && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit()) {
            // Decimal: "1,5" -> "bir virgül beş"; zeros after the comma are read one by one.
            let mut frac = String::new();
            i += 1;
            while i < chars.len() && chars[i].is_ascii_digit() {
                frac.push(chars[i]);
                i += 1;
            }
            out.extend(digits_to_tokens(&int, false));
            out.push("virgül".to_string());
            let zeros = frac.chars().take_while(|&c| c == '0').count();
            out.extend(std::iter::repeat_n("sıfır".to_string(), zeros));
            if zeros < frac.len() {
                out.extend(digits_to_tokens(&frac[zeros..], false));
            }
        } else {
            // A "." right after the number and ending the word is an ordinal marker.
            // Known limit: "Sayfa 3." at the end of a text is read as an ordinal too.
            let ordinal = !percent && chars.get(i) == Some(&'.') && i + 1 == chars.len() && dot_may_be_ordinal;
            if ordinal {
                i += 1;
            }
            out.extend(digits_to_tokens(&int, ordinal));
        }
    }
    flush_letters(&mut letters, &mut out);
    out
}

/// Normalised tokens with the index of the whitespace-separated word each came from.
pub fn normalize_indexed(text: &str) -> Vec<(String, usize)> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out = Vec::new();
    for (idx, w) in words.iter().enumerate() {
        let ordinal_ok = match words.get(idx + 1) {
            None => true,
            Some(next) => next.chars().find(|c| c.is_alphanumeric()).is_some_and(|c| c.is_lowercase()),
        };
        out.extend(normalize_word(w, ordinal_ok).into_iter().map(|t| (t, idx)));
    }
    out
}

/// Lowercase, drop punctuation, expand numbers to words (one token per word).
pub fn normalize(text: &str) -> Vec<String> {
    normalize_indexed(text).into_iter().map(|(t, _)| t).collect()
}
