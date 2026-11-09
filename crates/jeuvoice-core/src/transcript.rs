//! Transcript model: words with times, edited by the user, saved to JSON.
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub prob: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct Transcript {
    pub language: String,
    pub words: Vec<Word>,
}

impl Transcript {
    /// All word texts joined with one space.
    pub fn text(&self) -> String {
        self.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")
    }

    /// Replace the text of word `i`; the times stay. Out-of-range `i` does nothing.
    pub fn set_word(&mut self, i: usize, text: &str) {
        if let Some(w) = self.words.get_mut(i) {
            w.text = text.to_string();
        }
    }
}
