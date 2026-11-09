//! The window: a tab bar and the page for the open tab.

use crate::state::{App, Tab};
use rune::tok::{self, motion, radius, space, type_};
use rune::{Cx, El, col, el, row, text};

const TABS: [(Tab, &str, &str); 4] = [
    (Tab::Voice, "tab-voice", "Voice"),
    (Tab::Transcribe, "tab-transcribe", "Transcribe"),
    (Tab::Train, "tab-train", "Train"),
    (Tab::Settings, "tab-settings", "Settings"),
];

pub fn view(s: &App, _: &mut Cx<App>) -> El {
    let t = tok::theme(true);
    let mut bar = row().gap(space::S1).padding(space::S2).fill(t.card);
    for (tab, key, name) in TABS {
        let on = s.tab == tab;
        let button = el()
            .key(key)
            .role("tab")
            .label(name)
            .focusable(true)
            .padding(space::S2)
            .radius(radius::SM)
            .fill(if on { tok::accent(0, true) } else { t.control_track })
            .motion(motion::SNAPPY)
            .on_press(move |s: &mut App| s.tab = tab)
            .child(text(name).font(type_::TITLE).color(t.text));
        bar = bar.child(button);
    }
    let page = match s.tab {
        Tab::Transcribe => text("Transcribe").font(type_::TITLE).color(t.text),
        Tab::Settings => text("Settings").font(type_::TITLE).color(t.text),
        Tab::Voice | Tab::Train => text("Coming in a later version").font(type_::BODY).color(t.text_secondary),
    };
    let body = col().w_full().grow(1.0).center().child(page);
    col().w_full().h_full().fill(t.bg).child(bar).child(body)
}

#[cfg(test)]
mod tests {
    use crate::state::{App, Tab};
    use super::view;
    use rune::{Env, Runtime};

    #[test]
    fn transcribe_is_the_first_tab_and_tabs_switch() {
        let mut rt = Runtime::new(App::default(), view, (1100.0, 720.0), Env { system_dark: true, scale: 2.0 });
        rt.frame(0.0);
        assert_eq!(rt.state.tab, Tab::Transcribe);
        let (x, y) = rt.find("tab-settings").unwrap();
        rt.pointer_down(x, y); rt.pointer_up(x, y); rt.frame(1.0 / 60.0);
        assert_eq!(rt.state.tab, Tab::Settings);
    }
}
