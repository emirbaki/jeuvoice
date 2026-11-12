//! The Voices tab view: My voices on the left, the voice maker on the right.

use jeuvoice_core::preset::{BUILTIN_PRESET, PresetKind};
use rune::tok::{space, type_};
use rune::{Color, El, FileDialogOptions, ScrollSpec, Tx, col, el, row, text, text_input};

use crate::i18n::{tr, trf};
use crate::state::App;
use crate::style::{self, ButtonKind};
use crate::transcribe_tab::{MEDIA, section};
use crate::voices::state::{VoicesMode, VoicesStatus};
use crate::voices::traits::{Age, Gender, Pitch};

const LIST_W: f32 = 240.0;

pub fn view(s: &App) -> El {
    if !s.setup.ready() {
        return crate::setup::view::waiting_hint();
    }
    row()
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .gap(space::S3)
        .padding(space::S3)
        .child(style::glass_blur("glass-voices-list", voice_list(s).w(LIST_W).h_full().padding(space::S3)))
        .child(style::glass_blur("glass-voices-maker", maker(s).w_full().h_full().padding(space::S3)).grow(1.0).min_w(0.0))
}

fn voice_list(s: &App) -> El {
    let p = style::palette();
    let mut list = col().w_full().gap(space::S1);
    for v in s.voices.voices() {
        let name = v.name.clone();
        let kind = match v.kind {
            PresetKind::Clone => tr!("clone", "klon"),
            PresetKind::Seed => tr!("described", "tarif"),
        };
        let on = s.voices.selected.as_deref() == Some(name.as_str());
        let listen_name = name.clone();
        let mut item = row()
            .key(format!("voices-item-{name}"))
            .w_full()
            .center_y()
            .gap(space::S1)
            .px(space::S2)
            .py(6.0)
            .radius(10.0)
            .fill(if on { p.accent.alpha(0.18) } else { style::white_fill(0.0) })
            .cursor(rune::Cursor::Pointer)
            .on_press(move |s: &mut App| s.voices.selected = Some(listen_name.clone()))
            .child(
                col()
                    .grow(1.0)
                    .min_w(0.0)
                    .child(text(name.as_str()).font(type_::BODY).color(p.text).truncate())
                    .child(text(kind).font(type_::LABEL).color(p.text_secondary)),
            );
        if name != BUILTIN_PRESET {
            let del = name.clone();
            item = item.child(style::button_kind(&format!("voices-delete-{name}"), "✕", ButtonKind::Quiet, true, move |s, _| {
                s.voices.ask_delete(&del)
            }));
        }
        list = list.child(item);
    }
    let mut panel = col()
        .gap(space::S2)
        .child(text(tr!("My voices", "Seslerim")).font(type_::TITLE).color(p.text))
        .child(col().w_full().grow(1.0).min_h(0.0).scroll(ScrollSpec { stick_bottom: false }).child(list))
        .child(style::button("voices-new", tr!("+ New voice", "+ Yeni ses"), true, |s, _| {
            s.voices.selected = None;
            s.voices.name_field = rune::TextEdit::new("");
        }));
    if let Some(name) = &s.voices.selected {
        let n = name.clone();
        let reason = s.voices.try_block_reason(s.voice.is_busy());
        panel = panel.child(style::button("voices-listen", tr!("Listen", "Dinle"), reason.is_none(), move |s, _| s.voices_listen(&n)));
    }
    if let Some(name) = &s.voices.confirm_delete {
        panel = panel.child(delete_question(name));
    }
    panel
}

fn delete_question(name: &str) -> El {
    let p = style::palette();
    style::glass(
        col()
            .w_full()
            .gap(space::S2)
            .padding(space::S2)
            .child(
                text(trf!("Delete the voice {name}? This cannot be undone.", "{name} sesi silinsin mi? Bu geri alınamaz.").as_str())
                    .font(type_::BODY)
                    .color(p.text)
                    .wrap(),
            )
            .child(
                row()
                    .gap(space::S2)
                    .child(style::button("voices-delete-yes", tr!("Delete", "Sil"), true, |s, _| s.voices_confirm_delete()))
                    .child(style::button_kind("voices-delete-no", tr!("Cancel", "Vazgeç"), ButtonKind::Quiet, true, |s, _| {
                        s.voices.cancel_delete()
                    })),
            ),
    )
}

fn maker(s: &App) -> El {
    let v = &s.voices;
    let p = style::palette();
    let mode_chip = |key: &str, label: &str, mode: VoicesMode| {
        chip(key, label, v.mode == mode, move |s: &mut App| s.voices.mode = mode)
    };
    let mut page = col()
        .gap(space::S2)
        .child(text(tr!("New voice", "Yeni ses")).font(type_::TITLE).color(p.text))
        .child(
            row()
                .gap(space::S1)
                .child(mode_chip("voices-mode-clone", tr!("Clone", "Klonla"), VoicesMode::Clone))
                .child(mode_chip("voices-mode-describe", tr!("Describe", "Tarif et"), VoicesMode::Describe)),
        );
    page = match v.mode {
        VoicesMode::Clone => page.child(clone_inputs(s)),
        VoicesMode::Describe => page.child(trait_chips(s)),
    };
    page = page.child(settings_block(s)).child(try_row(s)).child(tries(s));
    page = page.child(status_line(s));
    page.child(save_row(s))
}

fn chip(key: &str, label: &str, on: bool, press: impl Fn(&mut App) + 'static) -> El {
    let p = style::palette();
    el()
        .key(key)
        .role("button")
        .label(label)
        .focusable(true)
        .px(space::S2)
        .py(4.0)
        .radius(999.0)
        .border(1.0, if on { p.accent } else { p.glass_border })
        .fill(if on { p.accent.alpha(0.14) } else { style::white_fill(0.0) })
        .cursor(rune::Cursor::Pointer)
        .on_press(press)
        .child(text(label).font(type_::LABEL).color(if on { p.accent } else { p.text }))
}

fn trait_chips(s: &App) -> El {
    let t = &s.voices.editor.traits;
    let mut genders = row().gap(space::S1).flex_wrap();
    for g in Gender::ALL {
        let on = t.gender == Some(g);
        let key = format!("voices-chip-gender-{}", if g == Gender::Male { "male" } else { "female" });
        genders = genders.child(chip(&key, g.label(), on, move |s: &mut App| {
            let tr = &mut s.voices.editor.traits;
            tr.gender = if tr.gender == Some(g) { None } else { Some(g) };
        }));
    }
    let mut ages = row().gap(space::S1).flex_wrap();
    for (a, slug) in Age::ALL.into_iter().zip(["child", "teenager", "young-adult", "middle-aged", "elderly"]) {
        let on = t.age == Some(a);
        ages = ages.child(chip(&format!("voices-chip-age-{slug}"), a.label(), on, move |s: &mut App| {
            let tr = &mut s.voices.editor.traits;
            tr.age = if tr.age == Some(a) { None } else { Some(a) };
        }));
    }
    let mut pitches = row().gap(space::S1).flex_wrap();
    for (pi, slug) in Pitch::ALL.into_iter().zip(["very-low", "low", "moderate", "high", "very-high"]) {
        let on = t.pitch == Some(pi);
        pitches = pitches.child(chip(&format!("voices-chip-pitch-{slug}"), pi.label(), on, move |s: &mut App| {
            let tr = &mut s.voices.editor.traits;
            tr.pitch = if tr.pitch == Some(pi) { None } else { Some(pi) };
        }));
    }
    let whisper = chip("voices-chip-style-whisper", tr!("Whisper", "Fısıltı"), t.whisper, |s: &mut App| {
        s.voices.editor.traits.whisper = !s.voices.editor.traits.whisper
    });
    col()
        .gap(space::S2)
        .child(section(tr!("Gender", "Cinsiyet")))
        .child(genders)
        .child(section(tr!("Age", "Yaş")))
        .child(ages)
        .child(section(tr!("Pitch", "Ses perdesi")))
        .child(pitches)
        .child(row().child(whisper))
}

fn clone_inputs(s: &App) -> El {
    let p = style::palette();
    let label = match &s.voices.editor.clip {
        Some(clip) => clip.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        None => tr!("Drop a 3–10 second clip, or click to pick one", "3–10 saniyelik bir klip bırakın veya seçmek için tıklayın").into(),
    };
    let drop = col()
        .key("voices-drop")
        .w_full()
        .padding(space::S3)
        .radius(12.0)
        .border(1.0, p.text_secondary.alpha(0.5))
        .border_dash(6.0, 4.0)
        .cursor(rune::Cursor::Pointer)
        .child(text(label.as_str()).font(type_::BODY).color(p.text_secondary).wrap())
        .on_press_tx(|_s: &mut App, tx: &mut Tx| {
            let options = FileDialogOptions { title: tr!("Pick a voice clip", "Ses klibi seçin").into(), ..Default::default() }
                .filter(tr!("Audio or video", "Ses veya video"), &MEDIA);
            tx.open_file(options, |s: &mut App, result, _tx| {
                if let Ok(Some(path)) = result {
                    let lang = s.settings.whisper_language.clone();
                    s.voices.drop_clip(path, &lang);
                }
            });
        })
        .on_drop_files(|s: &mut App, files, _| {
            if let Some(path) = files.first().cloned() {
                let lang = s.settings.whisper_language.clone();
                s.voices.drop_clip(path, &lang);
            }
        })
        .accepts_extensions(&MEDIA);
    let field = text_input(&s.voices.clip_text_field, type_::BODY)
        .key("voices-clip-text")
        .w_full()
        .h(64.0)
        .padding(space::S2)
        .radius(10.0)
        .fill(style::white_fill(0.06))
        .placeholder(tr!("What the clip says. Whisper fills it in, you check it.", "Klipte ne söyleniyor. Whisper doldurur, siz kontrol edin."), p.text_secondary)
        .on_edit(|s: &mut App| &mut s.voices.clip_text_field);
    col().gap(space::S2).child(drop).child(field)
}

fn settings_block(s: &App) -> El {
    let p = style::palette();
    let ed = &s.voices.editor;
    let mut block = col().gap(space::S1).child(style::button_kind(
        "voices-settings",
        if s.voices.show_settings { tr!("Settings ▾", "Ayarlar ▾") } else { tr!("Settings ▸", "Ayarlar ▸") },
        ButtonKind::Quiet,
        true,
        |s, _| s.voices.show_settings = !s.voices.show_settings,
    ));
    if s.voices.show_settings {
        let stepper = |key: &str, label: &str, value: String, minus: fn(&mut App), plus: fn(&mut App)| {
            row()
                .gap(space::S2)
                .center_y()
                .child(text(label).font(type_::LABEL).color(p.text_secondary).w(110.0))
                .child(style::button(&format!("{key}-minus"), "−", true, move |s, _| minus(s)))
                .child(text(value.as_str()).key(key).font(type_::BODY).color(p.text).w(48.0))
                .child(style::button(&format!("{key}-plus"), "+", true, move |s, _| plus(s)))
        };
        block = block
            .child(stepper(
                "voices-seed",
                tr!("Base seed", "Başlangıç seed"),
                ed.base_seed.to_string(),
                |s| s.voices.editor.base_seed = s.voices.editor.base_seed.saturating_sub(1).max(1),
                |s| s.voices.editor.base_seed += 1,
            ))
            .child(stepper(
                "voices-guidance",
                tr!("Guidance", "Guidance"),
                format!("{:.1}", ed.guidance),
                |s| s.voices.editor.guidance = (s.voices.editor.guidance - 0.5).max(1.0),
                |s| s.voices.editor.guidance = (s.voices.editor.guidance + 0.5).min(5.0),
            ))
            .child(stepper(
                "voices-speed",
                tr!("Speed", "Hız"),
                format!("{:.1}", ed.speed),
                |s| s.voices.editor.speed = ((s.voices.editor.speed - 0.1) * 10.0).round() / 10.0,
                |s| s.voices.editor.speed = ((s.voices.editor.speed + 0.1) * 10.0).round() / 10.0,
            ));
    }
    block
}

fn try_row(s: &App) -> El {
    let p = style::palette();
    let reason = s.voices.try_block_reason(s.voice.is_busy());
    let (label, more) = match s.voices.mode {
        VoicesMode::Clone => (tr!("Try", "Dene"), false),
        VoicesMode::Describe => (tr!("Try 3 voices", "3 ses dene"), false),
    };
    let mut r = row()
        .gap(space::S2)
        .center_y()
        .child(style::button_kind("voices-try", label, ButtonKind::Main, reason.is_none(), move |s, _| s.voices_try(more)));
    if s.voices.mode == VoicesMode::Describe && !s.voices.editor.tries.is_empty() {
        r = r.child(style::button("voices-try-more", tr!("+ Try 3 more", "+ 3 ses daha"), reason.is_none(), |s, _| s.voices_try(true)));
    }
    if let Some(reason) = reason.filter(|_| s.voice.is_busy()) {
        r = r.child(text(reason).font(type_::LABEL).color(p.text_secondary).wrap().grow(1.0).min_w(0.0));
    }
    r
}

fn tries(s: &App) -> El {
    let p = style::palette();
    let mut list = col().w_full().gap(space::S1);
    for (i, t) in s.voices.editor.tries.iter().enumerate() {
        let picked = s.voices.editor.picked == Some(i);
        let seed = t.seed;
        let state = if t.done { trf!("seed {seed}", "seed {seed}") } else { trf!("seed {seed} · rendering…", "seed {seed} · üretiliyor…") };
        let border: Color = if picked { p.accent } else { p.glass_border };
        list = list.child(
            row()
                .key(format!("voices-take-{i}"))
                .w_full()
                .center_y()
                .gap(space::S2)
                .px(space::S2)
                .py(6.0)
                .radius(10.0)
                .fill(style::white_fill(0.04))
                .border(1.0, border)
                .cursor(rune::Cursor::Pointer)
                .on_press(move |s: &mut App| s.voices.editor.picked = Some(i))
                .child(style::button(&format!("voices-play-{i}"), "▶", t.done && t.file.is_some(), move |s, _| {
                    let (transcribe, voices) = (&mut s.transcribe, &mut s.voices);
                    voices.play_try(i, transcribe.player_mut());
                }))
                .child(text(state.as_str()).font(type_::LABEL).color(if picked { p.accent } else { p.text_secondary }).grow(1.0))
                .child(text(if picked { "✓" } else { "" }).font(type_::BODY).color(p.accent)),
        );
    }
    list
}

fn status_line(s: &App) -> El {
    let p = style::palette();
    let (line, color) = match &s.voices.status {
        VoicesStatus::Idle => (s.voices.notice.clone().unwrap_or_default(), p.text_secondary),
        VoicesStatus::StartingEngine => (tr!("Starting the voice engine…", "Ses motoru başlıyor…").into(), p.text_secondary),
        VoicesStatus::CheckingClip => (tr!("Checking the clip…", "Klip kontrol ediliyor…").into(), p.text_secondary),
        VoicesStatus::LoadingModel => (tr!("Loading the voice model…", "Ses modeli yükleniyor…").into(), p.text_secondary),
        VoicesStatus::Rendering => (tr!("Making the tries…", "Denemeler üretiliyor…").into(), p.text_secondary),
        VoicesStatus::Failed(e) => (e.clone(), p.red),
    };
    text(line.as_str()).key("voices-status").font(type_::LABEL).color(color).wrap()
}

fn save_row(s: &App) -> El {
    let p = style::palette();
    let can_save = !s.voices.editor.tries.is_empty() && !s.voices.is_busy();
    row()
        .w_full()
        .gap(space::S2)
        .center_y()
        .child(
            text_input(&s.voices.name_field, type_::BODY)
                .key("voices-name")
                .single_line()
                .grow(1.0)
                .min_w(0.0)
                .padding(space::S2)
                .radius(10.0)
                .fill(style::white_fill(0.06))
                .placeholder(tr!("Name of the voice", "Sesin adı"), p.text_secondary)
                .on_edit(|s: &mut App| &mut s.voices.name_field),
        )
        .child(style::button("voices-save", tr!("Save", "Kaydet"), can_save, |s, _| {
            let _ = s.voices_save();
        }))
}
