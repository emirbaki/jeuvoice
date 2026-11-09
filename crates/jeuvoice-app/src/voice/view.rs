//! The Voice tab: project bar, script or timeline, play bar and status.

use rune::tok::{self, radius, space, type_};
use rune::{Color, El, col, el, oklch, row, text, text_input};

use crate::marks::{counts_text, marks};
use crate::state::App;
use crate::transcribe_tab::{button, red, section};
use crate::voice::state::{VOICE_SUPPORTED, VoiceStatus, VoiceTimeline};
use crate::waveform::waveform;

const LEFT_W: f32 = 200.0;

fn green() -> Color {
    oklch(0.74, 0.17, 150.0)
}

pub fn view(s: &App) -> El {
    if !VOICE_SUPPORTED {
        return col()
            .w_full()
            .grow(1.0)
            .center()
            .child(text("The voice engine does not run on this Mac yet.").font(type_::BODY).color(tok::theme(true).text));
    }
    row()
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .gap(space::S3)
        .padding(space::S3)
        .child(side_column(s))
        .child(main_column(s))
}

fn side_column(s: &App) -> El {
    let t = tok::theme(true);
    let v = &s.voice;
    let mut list = col().w_full().gap(space::S1);
    for p in &v.presets {
        let on = p.name == v.preset_name;
        let name = p.name.clone();
        let label = name.clone();
        list = list.child(
            el().key(format!("preset-{name}"))
                .role("radio")
                .checked(on)
                .label(label.as_str())
                .focusable(true)
                .padding(space::S2)
                .radius(radius::SM)
                .fill(if on { tok::accent(0, true) } else { t.control_track })
                .on_press(move |s: &mut App| s.voice.preset_name = name.clone())
                .child(text(label).font(type_::LABEL).color(t.text)),
        );
    }
    col().w(LEFT_W).h_full().min_h(0.0).gap(space::S2).child(section("Voice")).child(list)
}

fn main_column(s: &App) -> El {
    let t = tok::theme(true);
    let v = &s.voice;
    let title = v.project.as_ref().map(|p| p.name.clone()).unwrap_or_else(|| "No project".into());
    let header = row()
        .w_full()
        .center_y()
        .gap(space::S1)
        .child(text(title).key("voice-project").font(type_::TITLE).color(t.text).grow(1.0).truncate())
        .child(button("voice-open", "Open…", true, |s, tx| s.voice.open_dialog(tx)))
        .child(button("voice-new", "New", true, |s, _| s.voice.new_project()))
        .child(button("voice-generate", "Generate", v.project.is_some() || !v.script_field.text.trim().is_empty(), |s, _| {
            let cpu = crate::services::engine::whisper_must_use_cpu(&s.transcribe.engine_state());
            let store = s.token_store();
            let settings = s.settings.clone();
            let (transcribe, voice) = (&mut s.transcribe, &mut s.voice);
            voice.generate(transcribe.engine_mut(), &settings, store, cpu)
        }))
        .child(button("voice-export-wav", "Export WAV", v.timeline.is_some(), |s, tx| s.voice.export_wav(tx)))
        .child(button("voice-export-srt", "Export SRT", v.timeline.is_some(), |s, tx| s.voice.export_srt(tx)));

    let mut main = col().grow(1.0).min_w(0.0).h_full().gap(space::S2).child(header).child(name_row(s));
    match (v.timeline.as_ref(), v.project.as_ref()) {
        (Some(tl), Some(_)) => {
            main = main.child(wave_card(s, tl));
        }
        _ => {
            main = main.child(script_field(s));
        }
    }
    main = main.child(text(status_text(&v.status)).key("voice-status").font(type_::BODY).color(status_color(&v.status)).wrap());
    if !v.diff.is_empty() {
        main = main.child(text(counts_text(&v.diff)).key("voice-counts").font(type_::BODY).color(t.text).wrap());
    }
    if let Some(n) = &v.notice {
        main = main.child(text(n.as_str()).key("voice-notice").font(type_::LABEL).color(t.text_secondary).wrap());
    }
    main.child(el().grow(1.0)).child(play_bar(s))
}

fn name_row(s: &App) -> El {
    let t = tok::theme(true);
    let field = text_input(&s.voice.name_field, type_::BODY)
        .key("voice-name")
        .single_line()
        .w(220.0)
        .padding(space::S1)
        .radius(radius::SM)
        .fill(t.card)
        .on_edit(|s: &mut App| &mut s.voice.name_field);
    row().gap(space::S2).child(text("Project:").font(type_::LABEL).color(t.text_secondary)).child(field)
}

fn script_field(s: &App) -> El {
    let t = tok::theme(true);
    text_input(&s.voice.script_field, type_::BODY)
        .key("voice-script")
        .w_full()
        .h(220.0)
        .padding(space::S2)
        .radius(radius::SM)
        .fill(t.card)
        .border(1.0, t.card_border)
        .placeholder("Paste the script here, one paragraph per block", t.text_secondary)
        .on_edit(|s: &mut App| &mut s.voice.script_field)
}

fn wave_card(s: &App, tl: &VoiceTimeline) -> El {
    let t = tok::theme(true);
    let card = col().w_full().padding(space::S2).radius(radius::MD).fill(t.card).gap(space::S2);
    card.child(waveform(
        "voice-wave",
        &s.voice.view,
        &tl.peaks,
        &tl.words,
        &marks(&s.voice.diff, &tl.words),
        tl.duration,
        |s: &mut App| &mut s.voice.view,
        |s: &mut App, i| s.voice.seek_word(i),
        |_, _| {},
    ))
}

fn status_text(status: &VoiceStatus) -> String {
    match status {
        VoiceStatus::Idle => "Pick a preset, paste a script and press Generate.".into(),
        VoiceStatus::StartingEngine => "Starting the voice engine… This takes about a minute the first time.".into(),
        VoiceStatus::Working(m) => m.clone(),
        VoiceStatus::Ready => "Done.".into(),
        VoiceStatus::Failed(e) => e.clone(),
    }
}

fn status_color(status: &VoiceStatus) -> Color {
    match status {
        VoiceStatus::Failed(_) => red(),
        VoiceStatus::Ready => green(),
        _ => tok::theme(true).text,
    }
}

fn play_bar(s: &App) -> El {
    let t = tok::theme(true);
    let duration = s.voice.timeline.as_ref().map_or(0.0, |l| l.duration);
    let label = if s.voice.is_playing(s.transcribe.player()) { "Pause" } else { "Play" };
    row()
        .w_full()
        .center_y()
        .gap(space::S2)
        .padding(space::S2)
        .radius(radius::MD)
        .fill(t.card)
        .child(button("voice-play", label, s.voice.timeline.is_some(), |s, _| {
            let t = &mut s.transcribe;
            let v = &mut s.voice;
            v.toggle_play(t.player_mut())
        }))
        .child(
            text(format!("{} / {}", clock(s.voice.view.playhead), clock(duration)))
                .key("voice-time")
                .font(type_::MONO_CAPTION)
                .color(t.text),
        )
}

/// `m:ss.d`
fn clock(t: f64) -> String {
    let tenths = (t.max(0.0) * 10.0).round() as u64;
    format!("{}:{:02}.{}", tenths / 600, tenths / 10 % 60, tenths % 10)
}
