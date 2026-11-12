//! Voice attributes for OmniVoice voice design, and the `instruct` text they make.
//!
//! OmniVoice accepts only fixed English words, comma + space separated, in the
//! order gender, age, pitch, style. Accents are left out: the app speaks Turkish.

use crate::i18n::tr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gender {
    Male,
    Female,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Age {
    Child,
    Teenager,
    YoungAdult,
    MiddleAged,
    Elderly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pitch {
    VeryLow,
    Low,
    Moderate,
    High,
    VeryHigh,
}

impl Gender {
    pub const ALL: [Gender; 2] = [Gender::Male, Gender::Female];

    fn word(self) -> &'static str {
        match self {
            Gender::Male => "male",
            Gender::Female => "female",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Gender::Male => tr!("Male", "Erkek"),
            Gender::Female => tr!("Female", "Kadın"),
        }
    }
}

impl Age {
    pub const ALL: [Age; 5] = [Age::Child, Age::Teenager, Age::YoungAdult, Age::MiddleAged, Age::Elderly];

    fn word(self) -> &'static str {
        match self {
            Age::Child => "child",
            Age::Teenager => "teenager",
            Age::YoungAdult => "young adult",
            Age::MiddleAged => "middle-aged",
            Age::Elderly => "elderly",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Age::Child => tr!("Child", "Çocuk"),
            Age::Teenager => tr!("Teenager", "Genç"),
            Age::YoungAdult => tr!("Young adult", "Genç yetişkin"),
            Age::MiddleAged => tr!("Middle-aged", "Orta yaş"),
            Age::Elderly => tr!("Elderly", "Yaşlı"),
        }
    }
}

impl Pitch {
    pub const ALL: [Pitch; 5] = [Pitch::VeryLow, Pitch::Low, Pitch::Moderate, Pitch::High, Pitch::VeryHigh];

    fn word(self) -> &'static str {
        match self {
            Pitch::VeryLow => "very low pitch",
            Pitch::Low => "low pitch",
            Pitch::Moderate => "moderate pitch",
            Pitch::High => "high pitch",
            Pitch::VeryHigh => "very high pitch",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Pitch::VeryLow => tr!("Very low", "Çok kalın"),
            Pitch::Low => tr!("Low", "Kalın"),
            Pitch::Moderate => tr!("Medium", "Orta"),
            Pitch::High => tr!("High", "İnce"),
            Pitch::VeryHigh => tr!("Very high", "Çok ince"),
        }
    }
}

/// The attributes the user picks. Each group can stay empty.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VoiceTraits {
    pub gender: Option<Gender>,
    pub age: Option<Age>,
    pub pitch: Option<Pitch>,
    pub whisper: bool,
}

impl VoiceTraits {
    /// The OmniVoice `instruct` text, or `None` when nothing is picked (auto voice).
    pub fn instruct(&self) -> Option<String> {
        let mut words: Vec<&str> = Vec::new();
        words.extend(self.gender.map(Gender::word));
        words.extend(self.age.map(Age::word));
        words.extend(self.pitch.map(Pitch::word));
        if self.whisper {
            words.push("whisper");
        }
        (!words.is_empty()).then(|| words.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OMNIVOICE_WORDS: [&str; 12] = [
        "male", "female", "child", "teenager", "young adult", "middle-aged", "elderly",
        "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch",
    ];

    #[test]
    fn all_traits_make_the_omnivoice_string() {
        let t = VoiceTraits { gender: Some(Gender::Female), age: Some(Age::YoungAdult), pitch: Some(Pitch::High), whisper: true };
        assert_eq!(t.instruct().as_deref(), Some("female, young adult, high pitch, whisper"));
    }

    #[test]
    fn no_traits_make_no_instruct() {
        assert_eq!(VoiceTraits::default().instruct(), None);
    }

    #[test]
    fn one_trait_alone() {
        let t = VoiceTraits { pitch: Some(Pitch::VeryLow), ..Default::default() };
        assert_eq!(t.instruct().as_deref(), Some("very low pitch"));
    }

    #[test]
    fn every_value_is_an_omnivoice_word() {
        let mut seen = Vec::new();
        for g in Gender::ALL {
            seen.push(VoiceTraits { gender: Some(g), ..Default::default() }.instruct().unwrap());
        }
        for a in Age::ALL {
            seen.push(VoiceTraits { age: Some(a), ..Default::default() }.instruct().unwrap());
        }
        for p in Pitch::ALL {
            seen.push(VoiceTraits { pitch: Some(p), ..Default::default() }.instruct().unwrap());
        }
        assert_eq!(seen, OMNIVOICE_WORDS.map(String::from).to_vec());
    }
}
