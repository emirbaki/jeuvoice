//! The Settings tab: the Whisper language, the GPU switch and the speech model.

use rune::tok::{self, radius, space, type_};
use rune::{El, col, el, row, text};

use crate::state::App;
use crate::transcribe::TranscribeTab;
use crate::transcribe_tab::{button, red, section};

const LANGUAGES: [(&str, &str, &str); 3] = [("lang-tr", "tr", "Turkish"), ("lang-en", "en", "English"), ("lang-auto", "auto", "Detect")];
const BAR_W: f32 = 320.0;

pub fn view(s: &App) -> El {
    let t = tok::theme(true);
    let mut page = col()
        .w(560.0)
        .gap(space::S3)
        .padding(space::S3)
        .child(text("Settings").font(type_::TITLE).color(t.text))
        .child(section("Speech language"))
        .child(language_picker(&s.settings.whisper_language))
        .child(
            text("The language of the voice in your files. Detect finds the language of each file.")
                .font(type_::LABEL)
                .color(t.text_secondary)
                .wrap(),
        )
        .child(section("Graphics card"))
        .child(gpu_switch(s.settings.use_gpu))
        .child(section("Speech model"))
        .child(model_block(&s.transcribe));
    if let Some(e) = &s.settings_error {
        page = page.child(text(e.as_str()).key("settings-error").font(type_::LABEL).color(red()).wrap());
    }
    row().w_full().grow(1.0).min_h(0.0).justify_center().child(page)
}

fn language_picker(current: &str) -> El {
    let t = tok::theme(true);
    let mut r = row().gap(space::S1);
    for (key, code, name) in LANGUAGES {
        let on = current == code;
        r = r.child(
            el().key(key)
                .role("radio")
                .checked(on)
                .label(name)
                .focusable(true)
                .padding(space::S2)
                .radius(radius::SM)
                .fill(if on { tok::accent(0, true) } else { t.control_track })
                .on_press(move |s: &mut App| s.set_language(code))
                .child(text(name).font(type_::LABEL).color(t.text)),
        );
    }
    r
}

fn gpu_switch(on: bool) -> El {
    let t = tok::theme(true);
    let knob = el().w(18.0).h(18.0).radius(9.0).fill(t.text);
    let track = el()
        .key("use-gpu")
        .role("switch")
        .checked(on)
        .label("Use GPU")
        .focusable(true)
        .w(44.0)
        .h(24.0)
        .padding(3.0)
        .radius(12.0)
        .fill(if on { tok::accent(0, true) } else { t.control_track });
    let track = if on { track.justify_end() } else { track.justify_start() };
    row()
        .center_y()
        .gap(space::S2)
        .child(track.on_press(move |s: &mut App| s.set_use_gpu(!on)).child(knob))
        .child(text("Use GPU").font(type_::BODY).color(t.text))
}

/// What the status line says about the model.
pub fn model_status_text(s: &TranscribeTab) -> &'static str {
    match (s.model.ready, s.model.on_gpu) {
        (false, _) => "Not downloaded",
        (true, Some(true)) => "Ready (on GPU)",
        (true, Some(false)) => "Ready (on CPU)",
        (true, None) => "Ready",
    }
}

fn model_block(s: &TranscribeTab) -> El {
    let t = tok::theme(true);
    let mut block = col().gap(space::S2).child(text(model_status_text(s)).key("model-status").font(type_::BODY).color(t.text));
    if !s.model.ready {
        block = block.child(row().child(button("download-model", "Download", !s.busy(), |s, _| s.transcribe.start_model_download())));
    }
    if let Some(pct) = s.download {
        let done = pct.map_or(0.0, |p| BAR_W * f32::from(p) / 100.0);
        let label = pct.map_or("Downloading the speech model…".to_string(), |p| format!("Downloading the speech model… {p}%"));
        block = block
            .child(
                el().key("model-progress")
                    .role("progressbar")
                    .label(label.as_str())
                    .w(BAR_W)
                    .h(8.0)
                    .radius(4.0)
                    .fill(t.control_track)
                    .child(el().w(done).h(8.0).radius(4.0).fill(tok::accent(0, true))),
            )
            .child(text(label).key("model-progress-text").font(type_::LABEL).color(t.text_secondary));
    }
    if let Some(e) = &s.model_error {
        block = block.child(text(e.as_str()).key("model-error").font(type_::LABEL).color(red()).wrap());
    }
    block
}

#[cfg(test)]
#[path = "settings_tab_tests.rs"]
mod tests;
