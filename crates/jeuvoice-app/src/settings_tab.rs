//! The Settings tab: the Whisper language, the GPU switch, the speech model,
//! the Hugging Face token and the voice engine.

use rune::tok::{space, type_};
use rune::{El, col, el, row, text, text_input};

use crate::i18n::{Lang, tr, trf};
use crate::services::engine::EngineState;
use crate::state::App;
use crate::transcribe::TranscribeTab;
use crate::transcribe_tab::{button, red, section};

/// The Whisper languages: key, code and name.
fn languages() -> [(&'static str, &'static str, &'static str); 3] {
    [
        ("lang-tr", "tr", tr!("Turkish", "Türkçe")),
        ("lang-en", "en", tr!("English", "İngilizce")),
        ("lang-auto", "auto", tr!("Detect", "Algıla")),
    ]
}

/// The UI languages. Each name is in its own language.
const UI_LANGUAGES: [(&str, Lang, &str); 2] = [("ui-lang-en", Lang::En, "English"), ("ui-lang-tr", Lang::Tr, "Türkçe")];
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
        .child(section(tr!("Speech language", "Konuşma dili")))
        .child(language_picker(&s.settings.whisper_language))
        .child(
            text(tr!(
                "The language of the voice in your files. Detect finds the language of each file.",
                "Dosyalarınızdaki sesin dili. Algıla, her dosyanın dilini bulur."
            ))
                .font(type_::LABEL)
                .color(t.text_secondary)
                .wrap(),
        )
        .child(section(tr!("Speech model", "Konuşma modeli")))
        .child(model_block(&s.transcribe));
    let gpu = col().w_full().gap(space::S2).child(gpu_switch(s.settings.use_gpu)).child(gpu_warning(&s.gpu));
    let engine = col()
        .w_full()
        .gap(space::S2)
        .child(engine_block(s))
        .child(section(tr!("Voice model files", "Ses modeli dosyaları")))
        .child(adapter_block(s));
    let accounts = col()
        .w_full()
        .gap(space::S2)
        .child(section(tr!("Hugging Face token", "Hugging Face tokeni")))
        .child(token_block(s))
        .child(section(tr!("Kaggle profiles", "Kaggle profilleri")))
        .child(profiles_block(s));
    let left = col()
        .grow(1.0)
        .min_w(0.0)
        .gap(space::S3)
        .child(tile("settings-tile-ui", tr!("Interface language", "Arayüz dili"), ui_language_picker(&s.settings.ui_language)))
        .child(tile("settings-tile-speech", tr!("Speech", "Konuşma"), speech))
        .child(tile("settings-tile-gpu", tr!("Graphics card", "Ekran kartı"), gpu));
    let right = col()
        .grow(1.0)
        .min_w(0.0)
        .gap(space::S3)
        .child(tile("settings-tile-engine", tr!("Voice engine", "Ses motoru"), engine))
        .child(tile("settings-tile-accounts", tr!("Accounts", "Hesaplar"), accounts));
    let mut page = col()
        .w_full()
        .max_w(1040.0)
        .gap(space::S3)
        .padding(space::S3)
        .child(text(tr!("Settings", "Ayarlar")).font(type_::TITLE).color(t.text))
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

/// A row of radio pills. `on_pick` runs with the index of the pressed pill.
fn radio_row(pills: &[(&'static str, &'static str, bool)], on_pick: impl Fn(&mut App, usize) + Clone + 'static) -> El {
    let t = crate::style::theme();
    let mut r = row().gap(space::S1);
    for (i, &(key, name, on)) in pills.iter().enumerate() {
        let pick = on_pick.clone();
        r = r.child(
            el().key(key)
                .role("radio")
                .checked(on)
                .label(name)
                .focusable(true)
                .padding(space::S2)
                .radius(10.0)
                .fill(if on { crate::style::palette().accent.alpha(0.22) } else { t.control_track })
                .on_press(move |s: &mut App| pick(s, i))
                .child(text(name).font(type_::LABEL).color(t.text)),
        );
    }
    r
}

/// The UI language: English or Turkish. The next frame shows the new texts.
fn ui_language_picker(current: &str) -> El {
    let current = Lang::from_code(current);
    let pills: Vec<_> = UI_LANGUAGES.iter().map(|&(key, lang, name)| (key, name, lang == current)).collect();
    radio_row(&pills, |s, i| s.set_ui_language(UI_LANGUAGES[i].1.code()))
}

fn language_picker(current: &str) -> El {
    let langs = languages();
    let pills: Vec<_> = langs.iter().map(|&(key, code, name)| (key, name, current == code)).collect();
    radio_row(&pills, move |s, i| s.set_language(langs[i].1))
}


fn gpu_switch(on: bool) -> El {
    let t = crate::style::theme();
    let knob = el().w(18.0).h(18.0).radius(9.0).fill(t.text);
    let track = el()
        .key("use-gpu")
        .role("switch")
        .checked(on)
        .label(tr!("Use GPU", "GPU kullan"))
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
        .child(text(tr!("Use GPU", "GPU kullan")).font(type_::BODY).color(t.text))
}

/// The CPU mode warning, or an empty element when a card was found.
fn gpu_warning(gpu: &crate::services::gpu::GpuProbe) -> El {
    if *gpu != crate::services::gpu::GpuProbe::Missing {
        return el();
    }
    text(crate::services::gpu::no_gpu_settings()).key("gpu-warning").font(type_::BODY).color(red()).wrap()
}

/// What the status line says about the model.
pub fn model_status_text(s: &TranscribeTab) -> &'static str {
    match (s.model.ready, s.model.on_gpu) {
        (false, _) => tr!("Not downloaded", "İndirilmedi"),
        (true, Some(true)) => tr!("Ready (on GPU)", "Hazır (GPU'da)"),
        (true, Some(false)) => tr!("Ready (on CPU)", "Hazır (CPU'da)"),
        (true, None) => tr!("Ready", "Hazır"),
    }
}

fn model_block(s: &TranscribeTab) -> El {
    let t = crate::style::theme();
    let mut block = col().gap(space::S2).child(text(model_status_text(s)).key("model-status").font(type_::BODY).color(t.text));
    if !s.model.ready {
        block = block.child(row().child(button("download-model", tr!("Download", "İndir"), !s.busy(), |s, _| s.transcribe.start_model_download())));
    }
    if let Some(pct) = s.download {
        let done = pct.map_or(0.0, |p| BAR_W * f32::from(p) / 100.0);
        let label = pct.map_or(tr!("Downloading the speech model…", "Konuşma modeli indiriliyor…").to_string(), |p| {
            trf!("Downloading the speech model… {p}%", "Konuşma modeli indiriliyor… %{p}")
        });
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
        .placeholder(tr!("Paste the token, then Save", "Tokeni yapıştırın, sonra kaydedin"), t.text_secondary)
        .on_edit(|s: &mut App| &mut s.token_field)
        .on_submit(|s: &mut App, _| s.save_hf_token());
    let mut block = col()
        .gap(space::S2)
        .child(field)
        .child(button("save-token", tr!("Save token", "Tokeni kaydet"), true, |s, _| s.save_hf_token()))
        .child(
            text(tr!(
                "Opens private voice models. Find it in your Hugging Face account settings.",
                "Özel ses modellerini açar. Hugging Face hesap ayarlarınızda bulunur."
            ))
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
            card = card.child(text(tr!("Active", "Etkin")).font(type_::LABEL).color(t.text));
        } else {
            let select = name.clone();
            card = card.child(button(&format!("profile-select-{name}"), tr!("Use", "Kullan"), true, move |s: &mut App, _| {
                s.select_kaggle_profile(&select)
            }));
        }
        let delete = name.clone();
        card = card.child(button(&format!("profile-delete-{name}"), tr!("Remove", "Kaldır"), true, move |s: &mut App, _| {
            s.delete_kaggle_profile(&delete)
        }));
        block = block.child(card);
    }
    block = block.child(profile_field(
        "profile-name",
        tr!("Profile name (work, home…)", "Profil adı (iş, ev…)"),
        &s.profile_name_field,
        |s| &mut s.profile_name_field,
    ));
    block = block.child(profile_field("profile-user", tr!("Kaggle user name", "Kaggle kullanıcı adı"), &s.profile_user_field, |s| {
        &mut s.profile_user_field
    }));
    block = block.child(profile_field("profile-key", tr!("Kaggle key", "Kaggle anahtarı"), &s.profile_key_field, |s| &mut s.profile_key_field));
    block = block.child(button("profile-save", tr!("Save profile", "Profili kaydet"), true, |s, _| s.add_kaggle_profile()));
    block = block.child(
        text(tr!(
            "One key per Kaggle account. The Train tab uses the active profile.",
            "Her Kaggle hesabı için bir anahtar. Eğitim sekmesi etkin profili kullanır."
        ))
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
        .placeholder(
            tr!("A checkpoint folder, empty for the preset's own model", "Bir kontrol noktası klasörü, hazır ayarın kendi modeli için boş"),
            t.text_secondary,
        )
        .on_edit(|s: &mut App| &mut s.adapter_field)
        .on_submit(|s: &mut App, _| s.save_adapter_dir());
    col().gap(space::S2).child(field).child(button("save-adapter", tr!("Save folder", "Klasörü kaydet"), true, |s, _| s.save_adapter_dir()))
}

/// The voice engine state with a restart button.
fn engine_block(app: &App) -> El {
    let t = crate::style::theme();
    let state = app.transcribe.engine_state();
    let repair = matches!(&state, EngineState::Failed(e) if crate::version::is_version_mismatch(e));
    let (status, bad) = match state {
        EngineState::Off => (tr!("Voice engine: off", "Ses motoru: kapalı").to_string(), false),
        EngineState::Starting => (tr!("Voice engine: starting…", "Ses motoru: başlıyor…").to_string(), false),
        EngineState::Ready(h) => (tr!("Voice engine: ready ({})", "Ses motoru: hazır ({})", h.device), false),
        EngineState::Failed(e) => (trf!("Voice engine: {e}", "Ses motoru: {e}"), true),
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
            button("repair-engine", tr!("Repair", "Onar"), true, |s, _| {
                s.setup.repair();
                s.tab = crate::state::Tab::Voice;
            })
        } else {
            button("restart-engine", tr!("Restart engine", "Motoru yeniden başlat"), true, |s, _| s.transcribe.restart_engine())
        })
}

#[cfg(test)]
#[path = "settings_tab_tests.rs"]
mod tests;
