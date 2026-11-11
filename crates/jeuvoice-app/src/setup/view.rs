//! The setup screen: the step list with ticks, a progress bar, the current
//! package, and a note that the Transcribe tab stays usable.

use rune::tok::{space, type_};
use rune::{El, col, el, row, text};

use crate::i18n::tr;
use crate::setup::{SetupState, SetupStep};
use crate::transcribe_tab::button;

const BAR_W: f32 = 320.0;
fn steps() -> [(SetupStep, &'static str); 6] {
    [
        (SetupStep::CheckDisk, tr!("Free space", "Boş alan")),
        (SetupStep::CheckGpu, tr!("Graphics card", "Ekran kartı")),
        (SetupStep::UnpackPython, "Python"),
        (SetupStep::MakeEnvironment, tr!("Engine environment", "Motor ortamı")),
        (SetupStep::InstallPackages, tr!("Engine packages", "Motor paketleri")),
        (SetupStep::Verify, tr!("Check the engine", "Motor denetimi")),
    ]
}

/// The full setup page. `retry` runs setup again after a failure.
pub fn view(state: &SetupState, retry: impl Fn(&mut crate::state::App, &mut rune::Tx) + 'static) -> El {
    let t = crate::style::theme();
    let mut page = col()
        .w(520.0)
        .gap(space::S2)
        .padding(space::S3)
        .child(text(tr!("Setting up the voice engine", "Ses motoru kuruluyor")).font(type_::TITLE).color(t.text));
    let (current, progress, detail) = match state {
        SetupState::NotNeeded | SetupState::Checking => (None, 0.0, tr!("Checking…", "Denetleniyor…").to_string()),
        SetupState::Running { step, progress, detail } => (Some(*step), *progress, detail.clone()),
        SetupState::Failed { message } => {
            page = page
                .child(text(message.as_str()).key("setup-error").font(type_::BODY).color(t.text).wrap())
                .child(button("setup-retry", tr!("Try again", "Tekrar dene"), true, retry));
            return wrap(page);
        }
        SetupState::Done => {
            page = page.child(text(tr!("Ready.", "Hazır.")).key("setup-done").font(type_::BODY).color(t.text));
            return wrap(page);
        }
    };
    let p = crate::style::palette();
    for (step, label) in steps() {
        let (tick, color) = match current {
            Some(c) if step < c => ("✓", p.green),
            Some(c) if step == c => ("…", p.accent),
            _ => ("•", t.text_secondary),
        };
        page = page.child(
            row()
                .gap(space::S2)
                .child(text(tick).font(type_::LABEL).color(color).w(14.0))
                .child(text(label).key(format!("setup-step-{step:?}")).font(type_::LABEL).color(t.text).wrap()),
        );
    }
    let done = BAR_W * progress.clamp(0.0, 1.0);
    page = page
        .child(
            el()
                .key("setup-bar")
                .role("progressbar")
                .label(tr!("Engine setup", "Motor kurulumu"))
                .w(BAR_W)
                .h(8.0)
                .radius(4.0)
                .fill(t.control_track)
                .child(el().w(done).h(8.0).radius(4.0).fill(p.accent)),
        )
        .child(text(detail.as_str()).key("setup-detail").font(type_::LABEL).color(t.text_secondary).wrap())
        .child(
            text(tr!("About 10–15 minutes. You can keep using the Transcribe tab.", "Yaklaşık 10–15 dakika. Bu sırada Yazıya dök sekmesini kullanabilirsiniz."))
                .font(type_::LABEL)
                .color(t.text_secondary)
                .wrap(),
        );
    wrap(page)
}

/// The page as one glass card in the middle of the window.
fn wrap(page: El) -> El {
    col().w_full().grow(1.0).min_h(0.0).center().child(crate::style::glass_blur("glass-setup", page))
}

/// What the Voice and Train tabs show while setup runs.
pub fn waiting_hint() -> El {
    col()
        .w_full()
        .grow(1.0)
        .center()
        .child(
            text(tr!("Setting up the voice engine…", "Ses motoru kuruluyor…"))
                .key("setup-waiting")
                .font(type_::BODY)
                .color(crate::style::theme().text_secondary),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_state_renders() {
        let _ = view(&SetupState::NotNeeded, |_, _| {});
        let _ = view(&SetupState::Checking, |_, _| {});
        let _ = view(&SetupState::Done, |_, _| {});
        let _ = view(&SetupState::Failed { message: "x".into() }, |_, _| {});
        let _ = view(&SetupState::Running { step: SetupStep::InstallPackages, progress: 0.5, detail: "torch".into() }, |_, _| {});
        let _ = waiting_hint();
    }
}
