//! The setup screen: the step list with ticks, a progress bar, the current
//! package, and a note that the Transcribe tab stays usable.

use rune::tok::{self, space, type_};
use rune::{El, col, el, row, text};

use crate::setup::{SetupState, SetupStep};
use crate::transcribe_tab::button;

const BAR_W: f32 = 320.0;
const STEPS: [(SetupStep, &str); 6] = [
    (SetupStep::CheckDisk, "Free space"),
    (SetupStep::CheckGpu, "Graphics card"),
    (SetupStep::UnpackPython, "Python"),
    (SetupStep::MakeEnvironment, "Engine environment"),
    (SetupStep::InstallPackages, "Engine packages"),
    (SetupStep::Verify, "Check the engine"),
];

/// The full setup page. `retry` runs setup again after a failure.
pub fn view(state: &SetupState, retry: impl Fn(&mut crate::state::App, &mut rune::Tx) + 'static) -> El {
    let t = tok::theme(true);
    let mut page = col()
        .w(520.0)
        .gap(space::S2)
        .padding(space::S3)
        .child(text("Setting up the voice engine").font(type_::TITLE).color(t.text));
    let (current, progress, detail) = match state {
        SetupState::NotNeeded | SetupState::Checking => (None, 0.0, "Checking…".to_string()),
        SetupState::Running { step, progress, detail } => (Some(*step), *progress, detail.clone()),
        SetupState::Failed { message } => {
            page = page
                .child(text(message.as_str()).key("setup-error").font(type_::BODY).color(t.text).wrap())
                .child(button("setup-retry", "Try again", true, retry));
            return wrap(page);
        }
        SetupState::Done => {
            page = page.child(text("Ready.").key("setup-done").font(type_::BODY).color(t.text));
            return wrap(page);
        }
    };
    for (step, label) in STEPS {
        let tick = match current {
            Some(c) if step < c => "✓",
            Some(c) if step == c => "…",
            _ => "•",
        };
        page = page.child(
            text(format!("{tick} {label}"))
                .key(format!("setup-step-{step:?}"))
                .font(type_::LABEL)
                .color(t.text)
                .wrap(),
        );
    }
    let done = BAR_W * progress.clamp(0.0, 1.0);
    page = page
        .child(
            el()
                .key("setup-bar")
                .role("progressbar")
                .label("Engine setup")
                .w(BAR_W)
                .h(8.0)
                .radius(4.0)
                .fill(t.control_track)
                .child(el().w(done).h(8.0).radius(4.0).fill(tok::accent(0, true))),
        )
        .child(text(detail.as_str()).key("setup-detail").font(type_::LABEL).color(t.text_secondary).wrap())
        .child(
            text("About 10–15 minutes. You can keep using the Transcribe tab.")
                .font(type_::LABEL)
                .color(t.text_secondary)
                .wrap(),
        );
    wrap(page)
}

fn wrap(page: El) -> El {
    row().w_full().grow(1.0).min_h(0.0).justify_center().child(page)
}

/// What the Voice and Train tabs show while setup runs.
pub fn waiting_hint() -> El {
    col()
        .w_full()
        .grow(1.0)
        .center()
        .child(
            text("Setting up the voice engine…")
                .key("setup-waiting")
                .font(type_::BODY)
                .color(tok::theme(true).text_secondary),
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
