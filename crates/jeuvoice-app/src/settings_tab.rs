//! The Settings tab: the Whisper language, the GPU switch, the speech model,
//! the Hugging Face token and the voice engine.

use rune::tok::{space, type_};
use rune::{El, col, el, row, text, text_input};

use crate::services::engine::EngineState;
use crate::state::App;
use crate::transcribe::TranscribeTab;
use crate::transcribe_tab::{button, red, section};

const LANGUAGES: [(&str, &str, &str); 3] = [("lang-tr", "tr", "Turkish"), ("lang-en", "en", "English"), ("lang-auto", "auto", "Detect")];
const BAR_W: f32 = 320.0;

/// A Settings tile: a glass card with a title and its blocks.
fn tile(key: &str, title: &str, body: El) -> El {
    crate::style::glass(
        col()
            .key(key)
            .w_full()
            .gap(space::S2)
            .padding(space::S3)
            .child(text(title).font(type_::TITLE).color(crate::style::palette().text))
            .child(body),
    )
}

pub fn view(s: &App) -> El {
    let t = crate::style::theme();
    let speech = col()
        .w_full()
        .gap(space::S2)
        .child(section("Speech language"))
        .child(language_picker(&s.settings.whisper_language))
        .child(
            text("The language of the voice in your files. Detect finds the language of each file.")
                .font(type_::LABEL)
                .color(t.text_secondary)
                .wrap(),
        )
        .child(section("Speech model"))
        .child(model_block(&s.transcribe));
    let gpu = col().w_full().gap(space::S2).child(gpu_switch(s.settings.use_gpu)).child(gpu_warning(&s.gpu));
    let engine = col()
        .w_full()
        .gap(space::S2)
        .child(engine_block(s))
        .child(section("Voice model files"))
        .child(adapter_block(s));
    let accounts = col()
        .w_full()
        .gap(space::S2)
        .child(section("Hugging Face token"))
        .child(token_block(s))
        .child(section("Kaggle profiles"))
        .child(profiles_block(s));
    let left = col()
        .grow(1.0)
        .min_w(0.0)
        .gap(space::S3)
        .child(tile("settings-tile-speech", "Speech", speech))
        .child(tile("settings-tile-gpu", "Graphics card", gpu));
    let right = col()
        .grow(1.0)
        .min_w(0.0)
        .gap(space::S3)
        .child(tile("settings-tile-engine", "Voice engine", engine))
        .child(tile("settings-tile-accounts", "Accounts", accounts));
    let mut page = col()
        .w_full()
        .max_w(1040.0)
        .gap(space::S3)
        .padding(space::S3)
        .child(text("Settings").font(type_::TITLE).color(t.text))
        .child(row().w_full().gap(space::S3).child(left).child(right));
    if let Some(e) = &s.settings_error {
        page = page.child(text(e.as_str()).key("settings-error").font(type_::LABEL).color(red()).wrap());
    }
    col()
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .scroll(rune::ScrollSpec { stick_bottom: false })
        .child(row().w_full().min_h(0.0).justify_center().child(page))
}

fn language_picker(current: &str) -> El {
    let t = crate::style::theme();
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
                .radius(10.0)
                .fill(if on { crate::style::palette().accent.alpha(0.22) } else { t.control_track })
                .on_press(move |s: &mut App| s.set_language(code))
                .child(text(name).font(type_::LABEL).color(t.text)),
        );
    }
    r
}

fn gpu_switch(on: bool) -> El {
    let t = crate::style::theme();
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
        .fill(if on { crate::style::palette().accent } else { t.control_track });
    let track = if on { track.justify_end() } else { track.justify_start() };
    row()
        .center_y()
        .gap(space::S2)
        .child(track.on_press(move |s: &mut App| s.set_use_gpu(!on)).child(knob))
        .child(text("Use GPU").font(type_::BODY).color(t.text))
}

/// The CPU mode warning, or an empty element when a card was found.
fn gpu_warning(gpu: &crate::services::gpu::GpuProbe) -> El {
    if *gpu != crate::services::gpu::GpuProbe::Missing {
        return el();
    }
    text(crate::services::gpu::NO_GPU_SETTINGS).key("gpu-warning").font(type_::BODY).color(red()).wrap()
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
    let t = crate::style::theme();
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
                    .child(el().w(done).h(8.0).radius(4.0).fill(crate::style::palette().accent)),
            )
            .child(text(label).key("model-progress-text").font(type_::LABEL).color(t.text_secondary));
    }
    if let Some(e) = &s.model_error {
        block = block.child(text(e.as_str()).key("model-error").font(type_::LABEL).color(red()).wrap());
    }
    block
}

/// The Hugging Face token: saved to the OS keychain, never to `settings.json`.
fn token_block(s: &App) -> El {
    let t = crate::style::theme();
    let field = text_input(&s.token_field, type_::BODY)
        .key("hf-token")
        .single_line()
        .w_full()
        .padding(space::S2)
        .radius(10.0)
        .fill(t.card)
        .placeholder("Paste the token, then Save", t.text_secondary)
        .on_edit(|s: &mut App| &mut s.token_field)
        .on_submit(|s: &mut App, _| s.save_hf_token());
    let mut block = col()
        .gap(space::S2)
        .child(field)
        .child(button("save-token", "Save token", true, |s, _| s.save_hf_token()))
        .child(
            text("Opens private voice models. Find it in your Hugging Face account settings.")
                .font(type_::LABEL)
                .color(t.text_secondary)
                .wrap(),
        );
    if let Some(m) = &s.files_status {
        block = block.child(text(m.as_str()).key("files-status").font(type_::LABEL).color(t.text).wrap());
    }
    block
}

/// Kaggle key profiles: named side-slots in the keychain. The Train tab runs
/// with the active one; secrets never land in `settings.json`.
fn profiles_block(s: &App) -> El {
    let t = crate::style::theme();
    let mut block = col().gap(space::S2);
    for name in &s.settings.kaggle_profiles {
        let on = s.settings.active_kaggle_profile.as_deref() == Some(name);
        let mut card = row()
            .w_full()
            .gap(space::S2)
            .padding(space::S2)
            .radius(10.0)
            .fill(if on { crate::style::palette().accent.alpha(0.18) } else { t.card })
            .child(text(name.as_str()).key(format!("profile-{name}")).font(type_::BODY).color(t.text).grow(1.0).min_w(0.0));
        if on {
            card = card.child(text("Active").font(type_::LABEL).color(t.text));
        } else {
            let select = name.clone();
            card = card.child(button(&format!("profile-select-{name}"), "Use", true, move |s: &mut App, _| {
                s.select_kaggle_profile(&select)
            }));
        }
        let delete = name.clone();
        card = card.child(button(&format!("profile-delete-{name}"), "Remove", true, move |s: &mut App, _| {
            s.delete_kaggle_profile(&delete)
        }));
        block = block.child(card);
    }
    block = block.child(profile_field("profile-name", "Profile name (work, home…)", &s.profile_name_field, |s| &mut s.profile_name_field));
    block = block.child(profile_field("profile-user", "Kaggle user name", &s.profile_user_field, |s| &mut s.profile_user_field));
    block = block.child(profile_field("profile-key", "Kaggle key", &s.profile_key_field, |s| &mut s.profile_key_field));
    block = block.child(button("profile-save", "Save profile", true, |s, _| s.add_kaggle_profile()));
    block = block.child(
        text("One key per Kaggle account. The Train tab uses the active profile.")
            .font(type_::LABEL)
            .color(t.text_secondary)
            .wrap(),
    );
    if let Some(m) = &s.files_status {
        block = block.child(text(m.as_str()).key("profiles-status").font(type_::LABEL).color(t.text).wrap());
    }
    block
}

fn profile_field(key: &str, placeholder: &str, value: &rune::TextEdit, on_edit: impl Fn(&mut App) -> &mut rune::TextEdit + 'static) -> El {
    let t = crate::style::theme();
    text_input(value, type_::BODY)
        .key(key)
        .single_line()
        .w_full()
        .padding(space::S2)
        .radius(10.0)
        .fill(t.card)
        .placeholder(placeholder, t.text_secondary)
        .on_edit(on_edit)
}

/// A local adapter folder that overrides every Hugging Face adapter (development).
fn adapter_block(s: &App) -> El {
    let t = crate::style::theme();
    let field = text_input(&s.adapter_field, type_::BODY)
        .key("adapter-dir")
        .single_line()
        .w_full()
        .padding(space::S2)
        .radius(10.0)
        .fill(t.card)
        .placeholder("A checkpoint folder, empty for the preset's own model", t.text_secondary)
        .on_edit(|s: &mut App| &mut s.adapter_field)
        .on_submit(|s: &mut App, _| s.save_adapter_dir());
    col().gap(space::S2).child(field).child(button("save-adapter", "Save folder", true, |s, _| s.save_adapter_dir()))
}

/// The voice engine state with a restart button.
fn engine_block(app: &App) -> El {
    let t = crate::style::theme();
    let state = app.transcribe.engine_state();
    let repair = matches!(&state, EngineState::Failed(e) if crate::version::is_version_mismatch(e));
    let (status, bad) = match state {
        EngineState::Off => ("Voice engine: off".to_string(), false),
        EngineState::Starting => ("Voice engine: starting…".to_string(), false),
        EngineState::Ready(h) => (format!("Voice engine: ready ({})", h.device), false),
        EngineState::Failed(e) => (format!("Voice engine: {e}"), true),
    };
    col()
        .gap(space::S2)
        .child(
            text(status)
                .key("engine-status")
                .font(type_::BODY)
                .color(if bad { red() } else { t.text })
                .wrap(),
        )
        .child(if repair {
            button("repair-engine", "Repair", true, |s, _| {
                s.setup.repair();
                s.tab = crate::state::Tab::Voice;
            })
        } else {
            button("restart-engine", "Restart engine", true, |s, _| s.transcribe.restart_engine())
        })
}

#[cfg(test)]
#[path = "settings_tab_tests.rs"]
mod tests;
