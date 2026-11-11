//! The Transcribe tab (layout B): files on the left, the waveform with its words in the
//! middle with the play bar under it, and the script check on the right.

use jeuvoice_core::queue::ItemState;
use jeuvoice_core::script_diff::DiffOp;
use jeuvoice_core::transcript::Transcript;
use rune::tok::{space, type_};
use rune::{El, FileDialogOptions, ScrollSpec, Transition, Tx, col, el, row, text, text_input};

use crate::i18n::{tr, trf};
use crate::marks::{counts_text, marks};
use crate::state::App;
use crate::transcribe::{Format, TranscribeTab};
use crate::waveform::waveform;

/// Extensions the drop zone takes: audio and video files.
pub const MEDIA: [&str; 11] = ["wav", "mp3", "m4a", "aac", "ogg", "opus", "flac", "mp4", "mov", "mkv", "webm"];

const LEFT_W: f32 = 240.0;
const RIGHT_W: f32 = 290.0;

pub(crate) use crate::style::{green, red};

pub fn view(s: &TranscribeTab) -> El {
    row()
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .gap(space::S3)
        .padding(space::S3)
        .child(crate::style::glass_blur("glass-files", files_column(s).padding(space::S3)))
        .child(crate::style::glass_blur("glass-wave", main_column(s).padding(space::S3)).grow(1.0).min_w(0.0))
        .child(crate::style::glass_blur("glass-script", script_column(s).padding(space::S3)))
}

/// A gray glass button. `on` runs on press. A disabled button ignores presses.
pub(crate) fn button(key: &str, label: &str, enabled: bool, on: impl Fn(&mut App, &mut Tx) + 'static) -> El {
    crate::style::button(key, label, enabled, on)
}

pub(crate) fn section(title: &str) -> El {
    text(title).font(type_::SECTION).color(crate::style::theme().text_secondary)
}

fn file_name(p: &std::path::Path) -> String {
    p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned())
}

/// What a file row says about its file.
fn status(s: &TranscribeTab, state: &ItemState) -> String {
    match state {
        ItemState::Waiting => tr!("Waiting", "Bekliyor").into(),
        ItemState::Running(p) => match s.download {
            Some(Some(pct)) => trf!("Downloading the speech model… {pct}%", "Konuşma modeli indiriliyor… %{pct}"),
            Some(None) => tr!("Downloading the speech model…", "Konuşma modeli indiriliyor…").into(),
            None => tr!("Transcribing… {}%", "Yazıya dökülüyor… %{}", (p * 100.0).round() as u32),
        },
        ItemState::Done(_) => tr!("Done", "Bitti").into(),
        ItemState::Failed(e) => crate::core_text::queue_failure(e),
    }
}

fn files_column(s: &TranscribeTab) -> El {
    let t = crate::style::theme();
    let drop = col()
        .key("drop")
        .w_full()
        .h(110.0)
        .padding(space::S3)
        .gap(space::S1)
        .radius(12.0)
        .center()
        .fill(t.card)
        .hover(crate::style::hover_fill())
        .border(1.0, crate::style::palette().text_secondary.alpha(0.5))
        .border_dash(6.0, 4.0)
        .cursor(rune::Cursor::Pointer)
        .child(text(tr!("Drop audio or video files here", "Dosyaları buraya bırakın")).font(type_::TITLE).color(t.text).wrap())
        .child(text(tr!("Click to pick, or drop many", "Seçmek için tıklayın veya bırakın")).key("drop-hint").font(type_::LABEL).color(t.text_secondary).wrap())
        .on_press_tx(|_s: &mut App, tx: &mut Tx| {
            let options = FileDialogOptions {
                title: tr!("Open audio or video", "Ses veya video aç").into(),
                ..Default::default()
            }
            .filter(tr!("Audio or video", "Ses veya video"), &MEDIA);
            tx.open_file(options, |s: &mut App, result, _tx| match result {
                Ok(Some(path)) => s.transcribe.add_files(&[path]),
                Ok(None) => {}
                Err(e) => {
                    s.transcribe.notice = Some(trf!("The Open window could not open: {e}", "Aç penceresi açılamadı: {e}"))
                }
            });
        })
        .on_drop_files(|s: &mut App, files, _| s.transcribe.add_files(files))
        .accepts_extensions(&MEDIA);
    let mut list = col().w_full().gap(space::S1);
    for (i, (path, state)) in s.queue.items.iter().enumerate() {
        let on = s.selected == Some(i);
        let failed = matches!(state, ItemState::Failed(_));
        let row = col()
            .key(format!("file-{i}"))
            .w_full()
            .padding(space::S2)
            .gap(2.0)
            .radius(10.0)
            .fill(if on { crate::style::palette().accent.alpha(0.18) } else { t.card })
            .hover(if on { crate::style::palette().accent.alpha(0.28) } else { crate::style::hover_fill() })
            .cursor(rune::Cursor::Pointer)
            .on_press(move |s: &mut App| s.transcribe.select(i))
            .child(text(file_name(path)).font(type_::TITLE).color(t.text).truncate())
            .child(
                text(status(s, state))
                    .key(format!("status-{i}"))
                    .font(type_::LABEL)
                    .color(if failed { red() } else { t.text_secondary })
                    .wrap(),
            );
        list = list.child(row);
    }
    col()
        .w(LEFT_W)
        .h_full()
        .min_h(0.0)
        .gap(space::S2)
        .child(drop)
        .child(section(tr!("Files", "Dosyalar")))
        .child(col().w_full().grow(1.0).min_h(0.0).scroll(ScrollSpec { stick_bottom: false }).child(list))
}

fn main_column(s: &TranscribeTab) -> El {
    let t = crate::style::theme();
    let done = s.selected_done();
    let title = done.map_or_else(|| tr!("No file open", "Açık dosya yok").to_string(), |(p, _)| file_name(p));
    let has = done.is_some();
    let header = row()
        .w_full()
        .center_y()
        .gap(space::S1)
        .child(text(title).font(type_::TITLE).color(t.text).grow(1.0).truncate())
        .child(button("export-srt", tr!("Export SRT…", "SRT dışa aktar…"), has, |s, tx| s.transcribe.export(Format::Srt, tx)))
        .child(button("export-vtt", tr!("Export VTT…", "VTT dışa aktar…"), has, |s, tx| s.transcribe.export(Format::Vtt, tx)))
        .child(button("export-txt", tr!("Export TXT…", "TXT dışa aktar…"), has, |s, tx| s.transcribe.export(Format::Txt, tx)));

    let mut wave_card = col().w_full().padding(space::S2).radius(12.0).fill(t.card).gap(space::S2);
    wave_card = match (done, &s.loaded) {
        (Some((_, tr)), Some(l)) => wave_card.child(waveform(
            "wave",
            &s.view,
            &l.peaks,
            &tr.words,
            &marks(&s.diff, &tr.words),
            l.duration,
            |s: &mut App| &mut s.transcribe.view,
            |s: &mut App, i| s.transcribe.seek_word(i),
            |s: &mut App, i| s.transcribe.edit_word(i),
        )),
        (Some(_), None) => wave_card.child(placeholder(tr!("Loading the waveform…", "Dalga biçimi yükleniyor…"))),
        (None, _) => wave_card.child(placeholder(tr!("Drop files on the left. Pick a finished file to see its words here.", "Dosyaları sola bırakın. Sözcüklerini burada görmek için biten bir dosya seçin."))),
    };
    if s.editing.is_some() {
        wave_card = wave_card.child(word_editor(s));
    }

    let mut main = col().grow(1.0).min_w(0.0).h_full().gap(space::S2).child(header).child(wave_card);
    if s.loaded.is_some() {
        main = main.child(
            text(tr!("Click a word to play from it. Double-click a word to fix it.", "Oradan çalmak için bir sözcüğe tıklayın. Düzeltmek için çift tıklayın.")).font(type_::LABEL).color(t.text_secondary),
        );
    }
    if let Some(n) = &s.notice {
        main = main.child(text(crate::core_text::message(n)).key("notice").font(type_::LABEL).color(t.text_secondary).wrap());
    }
    // The word row under the wave shows only the visible time span. The full
    // text needs its own wrapped, scrollable panel.
    if let Some((_, tr)) = done {
        main = main.child(transcript_panel(tr, &s.view));
    } else {
        main = main.child(el().grow(1.0));
    }
    main.child(play_bar(s))
}

/// The whole transcript as wrapped, clickable words. Words outside the wave's
/// zoomed span are dimmed; the word under the playhead carries a sliding pill.
fn transcript_panel(tr: &Transcript, view: &crate::waveform::WaveView) -> El {
    let t = crate::style::theme();
    let spans: Vec<(f64, f64)> = tr.words.iter().map(|w| (w.start, w.end)).collect();
    let active = crate::waveform::word_at(&spans, view.playhead);
    let end = view.start + view.span;
    let mut words = row().w_full().gap(2.0).flex_wrap();
    for (i, w) in tr.words.iter().enumerate() {
        let in_view = w.end >= view.start && w.start <= end;
        let mut word = el()
            .key(format!("transcript-word-{i}"))
            .role("button")
            .label(w.text.as_str())
            .px(2.0)
            .py(1.0)
            .radius(4.0)
            .opacity(if in_view { 1.0 } else { 0.45 })
            .transition(Transition::opacity())
            .cursor(rune::Cursor::Pointer)
            .hover(crate::style::hover_fill())
            .on_press(move |s: &mut App| {
                s.transcribe.seek_word(i);
                s.transcribe.edit_word(i);
            });
        if active == Some(i) {
            word = word.child(
                el()
                    .key("transcript-active")
                    .matched("transcript-active")
                    .absolute(0.0, 0.0)
                    .w_full()
                    .h_full()
                    .radius(4.0)
                    .motion(rune::tok::motion::SNAPPY)
                    .fill(crate::style::palette().accent.alpha(0.22)),
            );
        }
        words = words.child(word.child(text(w.text.as_str()).font(type_::BODY).color(t.text)));
    }
    col()
        .key("transcript-panel")
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .padding(space::S3)
        .gap(space::S2)
        .radius(12.0)
        .fill(t.card)
        .child(section(tr!("Transcript", "Transkript")))
        .child(
            col()
                .key("transcript-viewport")
                .w_full()
                .grow(1.0)
                .min_h(0.0)
                .scroll(ScrollSpec { stick_bottom: false })
                .child(words),
        )
}

fn placeholder(msg: &str) -> El {
    col().w_full().h(122.0).center().child(text(msg).font(type_::BODY).color(crate::style::theme().text_secondary))
}

fn word_editor(s: &TranscribeTab) -> El {
    let t = crate::style::theme();
    let field = text_input(&s.word_field, type_::BODY)
        .key("word-edit")
        .single_line()
        .w(220.0)
        .padding(space::S1)
        .radius(10.0)
        .fill(t.control_track)
        .on_edit(|s: &mut App| &mut s.transcribe.word_field)
        .on_submit(|s: &mut App, _| s.transcribe.save_word());
    row()
        .gap(space::S2)
        .child(text(tr!("Fix the word:", "Sözcüğü düzeltin:")).font(type_::LABEL).color(t.text_secondary))
        .child(field)
        .child(button("word-save", tr!("Save", "Kaydet"), true, |s, _| s.transcribe.save_word()))
        .child(button("word-cancel", tr!("Cancel", "İptal"), true, |s, _| s.transcribe.cancel_edit()))
}

/// `m:ss.d`
fn clock(t: f64) -> String {
    let tenths = (t.max(0.0) * 10.0).round() as u64;
    format!("{}:{:02}.{}", tenths / 600, tenths / 10 % 60, tenths % 10)
}

fn play_bar(s: &TranscribeTab) -> El {
    let t = crate::style::theme();
    let duration = s.loaded.as_ref().map_or(0.0, |l| l.duration);
    let label = if s.is_playing() { tr!("Pause", "Duraklat") } else { tr!("Play", "Çal") };
    row()
        .w_full()
        .center_y()
        .gap(space::S2)
        .padding(space::S2)
        .radius(12.0)
        .fill(t.card)
        .child(button("play", label, s.loaded.is_some(), |s, _| s.transcribe.toggle_play()))
        .child(
            text(format!("{} / {}", clock(s.view.playhead), clock(duration)))
                .key("time")
                .font(type_::MONO_CAPTION)
                .color(t.text),
        )
}

fn script_column(s: &TranscribeTab) -> El {
    let t = crate::style::theme();
    let field = text_input(&s.script_field, type_::BODY)
        .key("script")
        .w_full()
        .h(180.0)
        .padding(space::S2)
        .radius(10.0)
        .fill(t.card)
        .border(1.0, t.card_border)
        .placeholder(tr!("Paste the script here", "Metni buraya yapıştırın"), t.text_secondary)
        .on_edit(|s: &mut App| &mut s.transcribe.script_field)
        .on_submit(|s: &mut App, _| s.transcribe.script_field.insert("\n"));
    let summary = if s.script.trim().is_empty() {
        tr!("Paste a script to check what the speaker said.", "Konuşmacının ne dediğini denetlemek için bir metin yapıştırın.").to_string()
    } else if s.selected_done().is_none() {
        tr!("Pick a finished file to check it.", "Denetlemek için biten bir dosya seçin.").to_string()
    } else {
        counts_text(&s.diff)
    };
    let mut column = col()
        .w(RIGHT_W)
        .h_full()
        .min_h(0.0)
        .gap(space::S2)
        .child(text(tr!("Script check", "Metin denetimi")).font(type_::TITLE).color(t.text))
        .child(field)
        .child(button("open-script", tr!("Open script…", "Metni aç…"), true, |s, tx| s.transcribe.open_script(tx)))
        .child(text(summary).key("script-counts").font(type_::BODY).color(t.text).wrap());
    if let Some((_, tr)) = s.selected_done() {
        let script_words: Vec<&str> = s.script.split_whitespace().collect();
        let sw = |i: usize| script_words.get(i).copied().unwrap_or("");
        let hw = |i: usize| tr.words.get(i).map_or("", |w| w.text.as_str());
        let mut list = col().w_full().gap(space::S1);
        let (mut n_missing, mut n_added, mut n_changed) = (0, 0, 0);
        for (k, op) in s.diff.iter().enumerate() {
            let (kind, word, color, pill_key) = match *op {
                DiffOp::Same { .. } => continue,
                DiffOp::Missing { script } => {
                    n_missing += 1;
                    (tr!("Missing", "Eksik"), sw(script).to_string(), red(), format!("diff-missing-{}", n_missing - 1))
                }
                DiffOp::Added { heard } => {
                    n_added += 1;
                    (tr!("Added", "Fazla"), hw(heard).to_string(), green(), format!("diff-added-{}", n_added - 1))
                }
                DiffOp::Changed { script, heard } => {
                    n_changed += 1;
                    (tr!("Changed", "Farklı"), format!("{} → {}", sw(script), hw(heard)), red(), format!("diff-changed-{}", n_changed - 1))
                }
            };
            let line = format!("{kind}: {word}");
            list = list.child(
                row()
                    .key(format!("diff-{k}"))
                    .role("button")
                    .label(line.as_str())
                    .w_full()
                    .center_y()
                    .gap(space::S2)
                    .px(space::S1)
                    .py(2.0)
                    .radius(10.0)
                    .cursor(rune::Cursor::Pointer)
                    .hover(crate::style::hover_fill())
                    .on_press(move |s: &mut App| s.transcribe.seek_diff(k))
                    .child(text(kind).font(type_::LABEL).color(t.text_secondary).w(64.0))
                    .child(crate::style::pill(&word, color).key(pill_key)),
            );
        }
        column = column.child(col().w_full().grow(1.0).min_h(0.0).scroll(ScrollSpec { stick_bottom: false }).child(list));
    }
    column
}

#[cfg(test)]
#[path = "transcribe_tab_tests.rs"]
mod tests;
