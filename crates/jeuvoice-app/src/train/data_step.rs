//! The Train tab data step: drop zone, Hugging Face form, source rows and review.

use jeuvoice_core::train::Source;
use rune::tok::{self, radius, space, type_};
use rune::{El, col, el, row, text, text_input};

use crate::state::App;
use crate::train::state::{SourceStatus, TrainData, fmt_minutes};
use crate::transcribe_tab::{MEDIA, button, section};

/// Labeled single-line field bound to a `TextEdit` of the Train tab.
fn field(s_key: &str, label_text: &str, value: &rune::TextEdit, on_edit: impl Fn(&mut App) -> &mut rune::TextEdit + 'static) -> El {
    let t = tok::theme(true);
    col().gap(space::S1).child(text(label_text).font(type_::LABEL).color(t.text_secondary)).child(
        text_input(value, type_::BODY)
            .key(s_key)
            .single_line()
            .w_full()
            .padding(space::S2)
            .radius(radius::SM)
            .fill(t.card)
            .on_edit(on_edit),
    )
}

pub fn view(s: &TrainData) -> El {
    let t = tok::theme(true);
    let mut page = col()
        .w_full()
        .gap(space::S2)
        .child(text("Training data").font(type_::TITLE).color(t.text))
        .child(text(s.chips().as_str()).key("train-chips").font(type_::BODY).color(t.text_secondary).wrap())
        .child(drop_zone());
    if s.show_hf { page = page.child(hf_form(s)); } else {
        page = page.child(button("train-hf-open", "Add Hugging Face dataset…", true, |s, _| s.train.show_hf = true));
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
    let t = tok::theme(true);
    col()
        .key("train-drop")
        .w_full()
        .padding(space::S3)
        .radius(radius::SM)
        .fill(t.card)
        .child(text("Drop clips or a folder here").font(type_::TITLE).color(t.text))
        .child(text("WAV, MP3 or video — many at once").font(type_::LABEL).color(t.text_secondary))
        .on_drop_files(|s: &mut App, files, _| s.train.add_files(files))
        .accepts_extensions(&MEDIA)
}

fn hf_form(s: &TrainData) -> El {
    let mut form = col()
        .w_full()
        .gap(space::S2)
        .padding(space::S2)
        .radius(radius::SM)
        .fill(tok::theme(true).card)
        .child(section("Hugging Face dataset"))
        .child(field("train-hf-repo", "Dataset (link or user/name)", &s.repo_field, |s| &mut s.train.repo_field))
        .child(field("train-hf-split", "Split", &s.split_field, |s| &mut s.train.split_field))
        .child(field("train-hf-audio", "Audio column", &s.audio_field, |s| &mut s.train.audio_field))
        .child(field(
            "train-hf-text",
            "Text column (empty: Whisper makes the text)",
            &s.text_field,
            |s| &mut s.train.text_field,
        ))
        .child(field("train-hf-limit", "Use at most N clips (empty: all)", &s.limit_field, |s| &mut s.train.limit_field));
    form = form.child(
        row()
            .gap(space::S2)
            .child(button("train-hf-add", "Add dataset", true, |s, _| s.train.add_hf()))
            .child(button("train-hf-cancel", "Cancel", true, |s, _| s.train.show_hf = false)),
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
    let t = tok::theme(true);
    let clips = s.clips_of(&source.name);
    let kept: Vec<_> = clips.iter().filter(|c| c.clip.excluded.is_none()).collect();
    let min: f32 = kept.iter().map(|c| c.clip.duration_s).sum::<f32>() / 60.0;
    let weighted = min * source.weight as f32;
    let min_text = fmt_minutes(min);
    let w_text = fmt_minutes(weighted);
    let unpointed = matches!(&source.kind, jeuvoice_core::train::SourceKind::LocalFolder { dir } if dir.is_empty());
    let status = if unpointed {
        crate::train::recipe_page::LOCAL_PROMPT.to_string()
    } else {
        match s.status_of(&source.name) {
            SourceStatus::Working(m) => m,
            SourceStatus::Ready => format!("{} clips · {min_text} min · ×{} = {w_text} min", kept.len(), source.weight),
            SourceStatus::Failed(e) => e,
        }
    };
    let mut card = row()
        .w_full()
        .gap(space::S2)
        .padding(space::S2)
        .radius(radius::SM)
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
        card = card.child(button(&format!("train-pick-{i}"), "Pick folder", true, move |s: &mut App, tx| {
            s.train.pick_local_folder(&name, tx)
        }));
    }
    card = card.child(button(&format!("train-review-{i}"), "Review", !clips.is_empty(), {
        let name = source.name.clone();
        move |s: &mut App, _| {
            s.train.review = Some(name.clone());
            s.train.editing = None;
        }
    }));
    card.child(button(&format!("train-remove-{i}"), "Remove", true, {
        let name = source.name.clone();
        move |s: &mut App, _| s.train.remove_source(&name)
    }))
}

fn review_panel(s: &TrainData, name: &str) -> El {
    let t = tok::theme(true);
    let mut list = col().w_full().gap(space::S1);
    for c in s.clips_of(name) {
        let id = c.clip.id.clone();
        let mut card = col().w_full().gap(space::S1).padding(space::S2).radius(radius::SM).fill(t.card);
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
            Some(_) => ("Include", false),
            None => ("Exclude", true),
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
                    .radius(radius::SM)
                    .fill(tok::theme(true).control_track)
                    .on_edit(|s: &mut App| &mut s.train.edit_field),
            );
            card = card.child(button("train-edit-done", "Done", true, |s, _| s.train.editing = None));
        } else {
            let shown = if c.clip.text.is_empty() { "(no text yet — type it in)" } else { c.clip.text.as_str() };
            card = card.child(
                el()
                    .key(format!("train-text-{id}"))
                    .role("button")
                    .label("Edit the text")
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
        .radius(radius::SM)
        .fill(t.card)
        .child(
            row()
                .w_full()
                .gap(space::S2)
                .child(text(format!("Review: {name}").as_str()).font(type_::TITLE).color(t.text).grow(1.0).min_w(0.0))
                .child(button("train-review-close", "Close", true, |s, _| {
                    s.train.review = None;
                    s.train.editing = None;
                })),
        )
        .child(list)
}
