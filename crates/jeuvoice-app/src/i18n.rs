//! The UI language: English or Turkish.
//!
//! Each text sits at its call site with both languages: `tr!("Save", "Kaydet")`.
//! With format arguments, the macro returns a `String`:
//! `tr!("{n} files", "{n} dosya")`.
//!
//! The app keeps the language in one global value, so worker threads also make
//! their messages in the current language. Tests keep it per thread, so a
//! Turkish test does not change the language of a parallel English test.

#[cfg(not(test))]
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Lang {
    #[default]
    En,
    Tr,
}

impl Lang {
    /// `"tr"` gives Turkish. Every other value gives English.
    pub fn from_code(code: &str) -> Lang {
        if code == "tr" { Lang::Tr } else { Lang::En }
    }

    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Tr => "tr",
        }
    }
}

#[cfg(not(test))]
static LANG: AtomicU8 = AtomicU8::new(0);

#[cfg(test)]
thread_local! {
    static LANG: std::cell::Cell<Lang> = const { std::cell::Cell::new(Lang::En) };
}

/// The current UI language.
pub fn lang() -> Lang {
    #[cfg(not(test))]
    {
        if LANG.load(Ordering::Relaxed) == 1 { Lang::Tr } else { Lang::En }
    }
    #[cfg(test)]
    {
        LANG.with(|l| l.get())
    }
}

/// Set the UI language. The next frame shows the texts in it.
pub fn set_lang(lang: Lang) {
    #[cfg(not(test))]
    LANG.store(if lang == Lang::Tr { 1 } else { 0 }, Ordering::Relaxed);
    #[cfg(test)]
    LANG.with(|l| l.set(lang));
}

/// The text in the current language. Two literals give a `&'static str`, with
/// no formatting: use `trf!` for a text with `{name}` captures.
/// Format arguments give a `String`, made with `format!`.
macro_rules! tr {
    ($en:literal, $tr:literal $(,)?) => {
        match $crate::i18n::lang() {
            $crate::i18n::Lang::En => $en,
            $crate::i18n::Lang::Tr => $tr,
        }
    };
    ($en:literal, $tr:literal, $($arg:tt)+) => {
        match $crate::i18n::lang() {
            $crate::i18n::Lang::En => format!($en, $($arg)+),
            $crate::i18n::Lang::Tr => format!($tr, $($arg)+),
        }
    };
}
pub(crate) use tr;

/// Like `tr!` with format arguments, for texts with captured names only.
macro_rules! trf {
    ($en:literal, $tr:literal $(,)?) => {
        match $crate::i18n::lang() {
            $crate::i18n::Lang::En => format!($en),
            $crate::i18n::Lang::Tr => format!($tr),
        }
    };
}
pub(crate) use trf;

/// The app view in Turkish, for snapshot tests: it sets the language on the
/// thread that draws each frame.
#[cfg(test)]
pub fn turkish_view(s: &crate::state::App, cx: &mut rune::Cx<crate::state::App>) -> rune::El {
    set_lang(Lang::Tr);
    crate::view::view(s, cx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_ui_language_is_english() {
        assert_eq!(Lang::from_code("de"), Lang::En);
        assert_eq!(Lang::from_code(""), Lang::En);
        assert_eq!(Lang::from_code("tr"), Lang::Tr);
        assert_eq!(Lang::Tr.code(), "tr");
    }

    #[test]
    fn language_is_per_test_thread() {
        set_lang(Lang::Tr);
        assert_eq!(tr!("Save", "Kaydet"), "Kaydet");
        let other = std::thread::spawn(|| tr!("Save", "Kaydet")).join().unwrap();
        assert_eq!(other, "Save", "a new thread starts in English");
        let n = 3;
        assert_eq!(trf!("{n} files", "{n} dosya"), "3 dosya");
        set_lang(Lang::En);
        assert_eq!(tr!("{} files", "{} dosya", n), "3 files");
    }

    fn app() -> rune::Runtime<crate::state::App> {
        use std::sync::Arc;
        let deps = crate::state::Deps {
            transcriber: Arc::new(jeuvoice_whisper::FakeTranscriber(Default::default())),
            ensure_model: None,
            cache_dir: None,
            sound: false,
            model: None,
            settings: Default::default(),
            settings_path: None,
            token_store: Arc::new(crate::settings::MemStore::default()),
            setup_paths: None,
        };
        let env = rune::Env { system_dark: true, scale: 2.0 };
        let mut rt = rune::Runtime::new(crate::state::App::new(deps), crate::view::view, (1100.0, 720.0), env);
        rt.frame(0.0);
        rt
    }

    /// Tap the tab button, then let the page change finish.
    fn show(rt: &mut rune::Runtime<crate::state::App>, tab: &str) {
        let (x, y) = rt.find(tab).unwrap_or_else(|| panic!("{tab} is on screen"));
        rt.pointer_down(x, y);
        rt.pointer_up(x, y);
        for _ in 0..30 {
            rt.frame(1.0 / 60.0);
        }
    }

    fn text(rt: &rune::Runtime<crate::state::App>, key: &str) -> String {
        rt.elements()
            .into_iter()
            .find(|e| e.key.as_deref() == Some(key) && !e.exiting)
            .and_then(|e| e.text.or(e.label))
            .unwrap_or_else(|| panic!("{key} shows text"))
    }

    #[test]
    fn every_page_shows_turkish() {
        let mut rt = app();
        set_lang(Lang::Tr);
        show(&mut rt, "tab-transcribe");
        assert_eq!(text(&rt, "tab-transcribe"), "Yazıya dök");
        assert_eq!(text(&rt, "export-srt"), "SRT dışa aktar…");
        show(&mut rt, "tab-voice");
        assert_eq!(text(&rt, "voice-generate"), "Üret");
        show(&mut rt, "tab-train");
        assert_eq!(text(&rt, "train-recipe-open"), "tr2 nasıl yapıldı");
        show(&mut rt, "tab-settings");
        assert_eq!(text(&rt, "engine-status"), "Ses motoru: kapalı");
        assert_eq!(text(&rt, "use-gpu"), "GPU kullan");
        rt.state.setup.set_state_for_tests(crate::setup::SetupState::Running {
            step: crate::setup::SetupStep::Verify,
            progress: 0.95,
            detail: String::new(),
        });
        show(&mut rt, "tab-voice");
        assert_eq!(text(&rt, "setup-step-Verify"), "Motor denetimi");
    }

    #[test]
    fn an_english_page_stays_english() {
        let mut rt = app();
        show(&mut rt, "tab-settings");
        assert_eq!(text(&rt, "tab-settings"), "Settings");
        assert_eq!(text(&rt, "engine-status"), "Voice engine: off");
    }
}
