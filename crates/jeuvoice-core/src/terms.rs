//! Voice script terms shared by the engine client, the project model and Settings.
//!
//! `en`: a sentence with one of these terms uses the English voice. `quote`: English
//! titles that keep their capitals in quotes. `respell`: `OLD=NEW` pairs applied to
//! the cleaned text (lowercase), for names the model misreads.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Terms {
    pub en: Vec<String>,
    pub quote: Vec<String>,
    pub respell: Vec<(String, String)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let t = Terms {
            en: vec!["tunnel".into()],
            quote: vec!["The Last of Us".into()],
            respell: vec![("damara".into(), "damara".into())],
        };
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(serde_json::from_str::<Terms>(&json).unwrap(), t);
    }

    #[test]
    fn missing_fields_give_empty_terms() {
        assert_eq!(serde_json::from_str::<Terms>("{}").unwrap(), Terms::default());
    }
}
