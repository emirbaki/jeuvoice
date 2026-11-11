//! The window: a tab bar and the page for the open tab.

use crate::state::{App, Tab};
use crate::{settings_tab, train, transcribe_tab, voice};
use rune::tok::{space, type_};
use crate::style;
use rune::{Cx, El, col, el, row, text};

fn tab_list() -> [(Tab, &'static str, &'static str); 4] {
    use crate::i18n::tr;
    [
        (Tab::Voice, "tab-voice", tr!("Voice", "Ses")),
        (Tab::Transcribe, "tab-transcribe", tr!("Transcribe", "Yazıya dök")),
        (Tab::Train, "tab-train", tr!("Train", "Eğitim")),
        (Tab::Settings, "tab-settings", tr!("Settings", "Ayarlar")),
    ]
}

pub fn view(s: &App, _: &mut Cx<App>) -> El {
    let p = style::palette();
    let mut tabs = row().gap(space::S1).padding(4.0).radius(999.0);
    for (tab, key, name) in tab_list() {
        let on = s.tab == tab;
        let mut button = el()
            .key(key)
            .role("tab")
            .label(name)
            .focusable(true)
            .px(space::S3)
            .py(6.0)
            .radius(999.0)
            .cursor(rune::Cursor::Pointer)
            .on_press(move |s: &mut App| s.tab = tab);
        if on {
            // One pill, matched across tabs, so it slides to the tab you pick.
            let pill = el()
                .key("tab-pill")
                .matched("tab-pill")
                .absolute(0.0, 0.0)
                .w_full()
                .h_full()
                .radius(999.0)
                .fill(style::white_fill(0.12))
                .border(1.0, p.glass_border);
            button = button.child(pill);
        }
        let ink = if on { p.text } else { p.text_secondary };
        tabs = tabs.child(button.child(text(name).font(type_::LABEL).color(ink)));
    }
    let tabs = style::glass_blur("glass-tabs", tabs);
    let (page_key, body) = match s.tab {
        Tab::Transcribe => ("page-transcribe", transcribe_tab::view(&s.transcribe)),
        // First start: Voice shows the setup page; Train waits for the engine.
        Tab::Voice if !s.setup.ready() => {
            ("page-voice", crate::setup::view::view(s.setup.state(), |s: &mut App, _| s.setup.retry()))
        }
        Tab::Train if !s.setup.ready() => ("page-train", crate::setup::view::waiting_hint()),
        Tab::Voice => ("page-voice", voice::view::view(s)),
        Tab::Settings => ("page-settings", settings_tab::view(s)),
        Tab::Train => ("page-train", train::view(s)),
    };
    let page = col()
        .key(page_key)
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .transition(style::page_exit())
        .child(col().w_full().grow(1.0).min_h(0.0).transition(style::page_transition()).child(body));
    let top = row().w_full().padding(space::S2).child(tabs);
    style::background(col().w_full().h_full().child(top).child(page))
}

#[cfg(test)]
mod tests {
    use crate::state::{App, Deps, Tab};
    use super::view;
    use jeuvoice_whisper::FakeTranscriber;
    use rune::{Env, Runtime};
    use std::sync::Arc;

    #[test]
    fn transcribe_is_the_first_tab_and_tabs_switch() {
        let deps = Deps { transcriber: Arc::new(FakeTranscriber(Default::default())), ensure_model: None, cache_dir: None, sound: false, model: None, settings: Default::default(), settings_path: None, token_store: Arc::new(crate::settings::MemStore::default()), setup_paths: None };
        let mut rt = Runtime::new(App::new(deps), view, (1100.0, 720.0), Env { system_dark: true, scale: 2.0 });
        rt.frame(0.0);
        assert_eq!(rt.state.tab, Tab::Transcribe);
        let (x, y) = rt.find("tab-settings").unwrap();
        rt.pointer_down(x, y); rt.pointer_up(x, y); rt.frame(1.0 / 60.0);
        assert_eq!(rt.state.tab, Tab::Settings);
    }

    /// The `[x, y, w, h]` layout target of the live element keyed `key`.
    fn rect(rt: &Runtime<App>, key: &str) -> [f32; 4] {
        rt.elements()
            .into_iter()
            .find(|e| e.key.as_deref() == Some(key) && !e.exiting)
            .unwrap_or_else(|| panic!("{key} is on screen"))
            .target
    }

    fn inside(a: [f32; 4], b: [f32; 4]) -> bool {
        a[0] >= b[0] - 0.5 && a[1] >= b[1] - 0.5 && a[0] + a[2] <= b[0] + b[2] + 0.5 && a[1] + a[3] <= b[1] + b[3] + 0.5
    }

    #[test]
    fn the_tab_pill_follows_the_active_tab() {
        let deps = Deps { transcriber: Arc::new(FakeTranscriber(Default::default())), ensure_model: None, cache_dir: None, sound: false, model: None, settings: Default::default(), settings_path: None, token_store: Arc::new(crate::settings::MemStore::default()), setup_paths: None };
        let mut rt = Runtime::new(App::new(deps), view, (1100.0, 720.0), Env { system_dark: true, scale: 2.0 });
        rt.frame(0.0);
        assert!(rt.find("glass-tabs").is_some(), "the tab bar is a blurred glass panel");
        assert!(inside(rect(&rt, "tab-pill"), rect(&rt, "tab-transcribe")));
        assert!(rt.find("page-transcribe").is_some());
        let (x, y) = rt.find("tab-train").unwrap();
        rt.pointer_down(x, y);
        rt.pointer_up(x, y);
        for _ in 0..60 {
            rt.frame(1.0 / 60.0);
        }
        assert!(inside(rect(&rt, "tab-pill"), rect(&rt, "tab-train")), "the pill moved to Train");
        assert!(rt.find("page-train").is_some());
        assert!(rt.find("page-transcribe").is_none(), "the old page left");
    }

    #[test]
    fn a_tab_change_keeps_the_blur_budget() {
        let deps = Deps { transcriber: Arc::new(FakeTranscriber(Default::default())), ensure_model: None, cache_dir: None, sound: false, model: None, settings: Default::default(), settings_path: None, token_store: Arc::new(crate::settings::MemStore::default()), setup_paths: None };
        let mut rt = Runtime::new(App::new(deps), view, (1100.0, 720.0), Env { system_dark: true, scale: 2.0 });
        rt.frame(0.0);
        let (x, y) = rt.find("tab-voice").unwrap();
        rt.pointer_down(x, y);
        rt.pointer_up(x, y);
        rt.frame(1.0 / 60.0);
        // An invisible element draws no blur: rune fades the blur with the opacity.
        let blurred = rt
            .elements()
            .iter()
            .filter(|e| e.opacity > 0.01 && e.key.as_deref().is_some_and(|k| k.starts_with("glass-")))
            .count();
        assert!(blurred <= 4, "{blurred} blurred panels during the page change, exiting ones included");
    }
}
