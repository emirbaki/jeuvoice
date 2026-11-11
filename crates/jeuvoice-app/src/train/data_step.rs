//! The Train tab data step: drop zone, Hugging Face form, source rows and review.

use jeuvoice_core::train::Source;
use rune::tok::{space, type_};
use rune::{El, col, el, row, text, text_input};

use crate::i18n::{tr, trf};
use crate::state::App;
use crate::train::state::{SourceStatus, TrainData, fmt_minutes};
use crate::transcribe_tab::{MEDIA, button, section};

/// Labeled single-line field bound to a `TextEdit` of the Train tab.
fn field(s_key: &str, label_text: &str, value: &rune::TextEdit, on_edit: impl Fn(&mut App) -> &mut rune::TextEdit + 'static) -> El {
    let t = crate::style::theme();
    col().gap(space::S1).child(text(label_text).font(type_::LABEL).color(t.text_secondary)).child(
        text_input(value, type_::BODY)
            .key(s_key)
            .single_line()
            .w_full()
            .padding(space::S2)
            .radius(10.0)
            .fill(t.card)
            .on_edit(on_edit),
    )
}

pub fn view(s: &TrainData) -> El {
    let t = crate::style::theme();
    let mut page = col()
        .w_full()
        .gap(space::S2)
        .child(text(s.chips().as_str()).key("train-chips").font(type_::BODY).color(t.text_secondary).wrap())
        .child(drop_zone());
    if s.show_hf { page = page.child(hf_form(s)); } else {
        page = page.child(button("train-hf-open", tr!("Add Hugging Face dataset…", "Hugging Face veri kümesi ekle…"), true, |s, _| s.train.show_hf = true));
    }
    page = page.child(sources_list(s));
    if let Some(name) = &s.review {
        page = page.child(review_panel(s, name));
    }
    if let Some(n) = &s.notice {
        page = page.child(text(n.as_str()).key("train-notice").font(type_::LABEL).color(t.text_secondary).wrap());
    }
    page
}

fn drop_zone() -> El {
    let t = crate::style::theme();
    col()
        .key("train-drop")
        .w_full()
        .padding(space::S3)
        .radius(10.0)
        .fill(t.card)
        .child(text(tr!("Drop clips or a folder here", "Klipleri veya bir klasörü buraya bırakın")).font(type_::TITLE).color(t.text))
        .child(text(tr!("WAV, MP3 or video — many at once", "WAV, MP3 veya video — birden çok dosya")).font(type_::LABEL).color(t.text_secondary))
        .on_drop_files(|s: &mut App, files, _| s.train.add_files(files))
        .accepts_extensions(&MEDIA)
}

fn hf_form(s: &TrainData) -> El {
    let mut form = col()
        .w_full()
        .gap(space::S2)
        .padding(space::S2)
        .radius(10.0)
        .fill(crate::style::theme().card)
        .child(section(tr!("Hugging Face dataset", "Hugging Face veri kümesi")))
        .child(field("train-hf-repo", tr!("Dataset (link or user/name)", "Veri kümesi (bağlantı veya kullanıcı/ad)"), &s.repo_field, |s| &mut s.train.repo_field))
        .child(field("train-hf-split", tr!("Split", "Bölüm"), &s.split_field, |s| &mut s.train.split_field))
        .child(field("train-hf-audio", tr!("Audio column", "Ses sütunu"), &s.audio_field, |s| &mut s.train.audio_field))
        .child(field(
            "train-hf-text",
            tr!("Text column (empty: Whisper makes the text)", "Metin sütunu (boş: metni Whisper yazar)"),
            &s.text_field,
            |s| &mut s.train.text_field,
        ))
        .child(field("train-hf-limit", tr!("Use at most N clips (empty: all)", "En fazla N klip kullan (boş: hepsi)"), &s.limit_field, |s| &mut s.train.limit_field));
    form = form.child(
        row()
            .gap(space::S2)
            .child(button("train-hf-add", tr!("Add dataset", "Veri kümesini ekle"), true, |s, _| s.train.add_hf()))
            .child(button("train-hf-cancel", tr!("Cancel", "İptal"), true, |s, _| s.train.show_hf = false)),
    );
    form
}

fn sources_list(s: &TrainData) -> El {
    let mut list = col().w_full().gap(space::S1);
    for (i, source) in s.sources().iter().enumerate() {
        list = list.child(source_row(s, i, source));
    }
    list
}

fn source_row(s: &TrainData, i: usize, source: &Source) -> El {
    let t = crate::style::theme();
    let clips = s.clips_of(&source.name);
    let kept: Vec<_> = clips.iter().filter(|c| c.clip.excluded.is_none()).collect();
    let min: f32 = kept.iter().map(|c| c.clip.duration_s).sum::<f32>() / 60.0;
    let weighted = min * source.weight as f32;
    let min_text = fmt_minutes(min);
    let w_text = fmt_minutes(weighted);
    let unpointed = matches!(&source.kind, jeuvoice_core::train::SourceKind::LocalFolder { dir } if dir.is_empty());
    let status = if unpointed {
        crate::train::recipe_page::local_prompt().to_string()
    } else {
        match s.status_of(&source.name) {
            SourceStatus::Working(m) => m,
            SourceStatus::Ready => tr!("{} clips · {min_text} min · ×{} = {w_text} min", "{} klip · {min_text} dk · ×{} = {w_text} dk", kept.len(), source.weight),
            SourceStatus::Failed(e) => e,
        }
    };
    let mut card = row()
        .w_full()
        .gap(space::S2)
        .padding(space::S2)
        .radius(10.0)
        .fill(t.card)
        .child(
            col()
                .gap(space::S1)
                .grow(1.0)
                .min_w(0.0)
                .child(text(source.name.as_str()).font(type_::BODY).color(t.text))
                .child(text(status.as_str()).key(format!("train-status-{i}")).font(type_::LABEL).color(t.text_secondary).wrap()),
        )
        .child(button(&format!("train-wminus-{i}"), "−", source.weight > 1, {
            let name = source.name.clone();
            move |s: &mut App, _| {
                let w = s.train.sources().iter().find(|x| x.name == name).map(|x| x.weight).unwrap_or(1);
                s.train.set_weight(&name, w - 1);
            }
        }))
        .child(button(&format!("train-wplus-{i}"), "+", source.weight < 5, {
            let name = source.name.clone();
            move |s: &mut App, _| {
                let w = s.train.sources().iter().find(|x| x.name == name).map(|x| x.weight).unwrap_or(1);
                s.train.set_weight(&name, w + 1);
            }
        }));
    if unpointed {
        let name = source.name.clone();
        card = card.child(button(&format!("train-pick-{i}"), tr!("Pick folder", "Klasör seç"), true, move |s: &mut App, tx| {
            s.train.pick_local_folder(&name, tx)
        }));
    }
    card = card.child(button(&format!("train-review-{i}"), tr!("Review", "İncele"), !clips.is_empty(), {
        let name = source.name.clone();
        move |s: &mut App, _| {
            s.train.review = Some(name.clone());
            s.train.editing = None;
        }
    }));
    card.child(button(&format!("train-remove-{i}"), tr!("Remove", "Kaldır"), true, {
        let name = source.name.clone();
        move |s: &mut App, _| s.train.remove_source(&name)
    }))
}

fn review_panel(s: &TrainData, name: &str) -> El {
    let t = crate::style::theme();
    let mut list = col().w_full().gap(space::S1);
    for c in s.clips_of(name) {
        let id = c.clip.id.clone();
        let mut card = col().w_full().gap(space::S1).padding(space::S2).radius(10.0).fill(t.card);
        let mut head = row().w_full().gap(space::S2);
        head = head.child(button(&format!("train-play-{id}"), "▶", c.clip.excluded.is_none(), {
            let id = id.clone();
            move |s: &mut App, _| s.train.toggle_play(&id)
        }));
        head = head.child(
            text(c.clip.id.as_str())
                .font(type_::LABEL)
                .color(t.text_secondary)
                .grow(1.0)
                .min_w(0.0)
                .truncate(),
        );
        head = head.child(text(format!("{:.0} kHz", c.band_khz).as_str()).font(type_::LABEL).color(t.text_secondary));
        let (label, exclude) = match &c.clip.excluded {
            Some(_) => (tr!("Include", "Dahil et"), false),
            None => (tr!("Exclude", "Dışarıda bırak"), true),
        };
        head = head.child(button(&format!("train-excl-{id}"), label, true, {
            let id = id.clone();
            move |s: &mut App, _| s.train.set_excluded(&id, exclude)
        }));
        card = card.child(head);
        if s.editing.as_deref() == Some(&id) {
            card = card.child(
                text_input(&s.edit_field, type_::BODY)
                    .key("train-edit")
                    .single_line()
                    .w_full()
                    .padding(space::S2)
                    .radius(10.0)
                    .fill(crate::style::theme().control_track)
                    .on_edit(|s: &mut App| &mut s.train.edit_field),
            );
            card = card.child(button("train-edit-done", tr!("Done", "Bitti"), true, |s, _| s.train.editing = None));
        } else {
            let shown = if c.clip.text.is_empty() { tr!("(no text yet — type it in)", "(henüz metin yok — yazın)") } else { c.clip.text.as_str() };
            card = card.child(
                el()
                    .key(format!("train-text-{id}"))
                    .role("button")
                    .label(tr!("Edit the text", "Metni düzenle"))
                    .focusable(true)
                    .child(text(shown).font(type_::BODY).color(t.text).wrap())
                    .on_press({
                        let id = id.clone();
                        let text = c.clip.text.clone();
                        move |s: &mut App| {
                            s.train.editing = Some(id.clone());
                            s.train.edit_field = rune::TextEdit::new(text.clone());
                        }
                    }),
            );
        }
        if let Some(reason) = &c.clip.excluded {
            card = card.child(text(reason.as_str()).font(type_::LABEL).color(t.text_secondary).wrap());
        }
        list = list.child(card);
    }
    col()
        .w_full()
        .gap(space::S2)
        .padding(space::S2)
        .radius(10.0)
        .fill(t.card)
        .child(
            row()
                .w_full()
                .gap(space::S2)
                .child(text(trf!("Review: {name}", "İnceleme: {name}").as_str()).font(type_::TITLE).color(t.text).grow(1.0).min_w(0.0))
                .child(button("train-review-close", tr!("Close", "Kapat"), true, |s, _| {
                    s.train.review = None;
                    s.train.editing = None;
                })),
        )
        .child(list)
}
