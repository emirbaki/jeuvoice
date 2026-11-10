//! The Voice tab: project bar, script or timeline, takes, play bar and status.

use rune::tok::{self, radius, space, type_};
use rune::{Color, El, Key, KeyEvent, ScrollSpec, TextEdit, Tx, col, el, oklch, row, text, text_input};

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
        .on_key(|s: &mut App, ev: &KeyEvent, _tx: &mut Tx| {
            if voice_field_focused(s) {
                return false;
            }
            match &ev.key {
                Key::Command('z') if ev.mods.cmd && !ev.mods.shift => {
                    s.voice.undo();
                    true
                }
                Key::Command('y') if ev.mods.cmd => {
                    s.voice.redo();
                    true
                }
                Key::Command('Z') if ev.mods.cmd => {
                    s.voice.redo();
                    true
                }
                _ => false,
            }
        })
}

/// Text fields that keep keyboard shortcuts for themselves while focused.
fn voice_field_focused(s: &App) -> bool {
    let v = &s.voice;
    v.script_field.focused
        || v.name_field.focused
        || v.fix_word.as_ref().is_some_and(|f| f.field.focused)
        || v.adv_guidance.focused
        || v.adv_seed.focused
        || v.adv_pause.focused
        || v.adv_para_pause.focused
        || v.adv_en.focused
        || v.adv_quote.focused
        || v.adv_respell.focused
        || v.adv_para_guidance.focused
        || v.adv_para_seed.focused
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
    let mut side = col().w(LEFT_W).h_full().min_h(0.0).gap(space::S2).child(section("Voice")).child(list);
    side = side.child(section("Takes")).child(takes_panel(s));
    side.child(
        col()
            .w_full()
            .grow(1.0)
            .min_h(0.0)
            .scroll(ScrollSpec { stick_bottom: false })
            .child(takes_list(s)),
    )
}

/// Takes of the panel paragraph: play/select, use, and three more.
fn takes_panel(s: &App) -> El {
    col()
        .w_full()
        .gap(space::S1)
        .child(
            row()
                .w_full()
                .center_y()
                .gap(space::S2)
                .child(button("voice-retake", "Retake region", true, |s, _| {
                    let store = s.token_store();
                    let settings = s.settings.clone();
                    let cpu = crate::services::engine::whisper_must_use_cpu(&s.transcribe.engine_state());
                    let (engine, voice) = (&mut s.transcribe, &mut s.voice);
                    voice.retake(engine.engine_mut(), &settings, store, cpu)
                }))
                .child(button("voice-try-more", "+ Try 3 more", true, |s, _| {
                    let store = s.token_store();
                    let settings = s.settings.clone();
                    let cpu = crate::services::engine::whisper_must_use_cpu(&s.transcribe.engine_state());
                    let (engine, voice) = (&mut s.transcribe, &mut s.voice);
                    voice.try_more(engine.engine_mut(), &settings, store, cpu)
                })),
        )
        .child(button("voice-use-take", "Use this take", true, |s, _| {
            s.voice.use_take()
        }))
        .child(
            row().w_full().center_y().gap(space::S2).child(button(
                "voice-undo",
                "Undo",
                s.voice.history.can_undo(),
                |s, _| s.voice.undo(),
            ))
            .child(button("voice-redo", "Redo", s.voice.history.can_redo(), |s, _| s.voice.redo())),
        )
}

/// One take row: tap to play and pick it for "Use this take".
fn takes_list(s: &App) -> El {
    let t = tok::theme(true);
    let mut list = col().w_full().gap(space::S1);
    let takes: Vec<(usize, u32)> = s
        .voice
        .panel_para()
        .and_then(|p| {
            s.voice.project.as_ref()?.paragraphs.iter().find(|q| q.index == p).map(|q| (p, q.takes.iter().map(|t| t.id).collect::<Vec<_>>()))
        })
        .map(|(p, ids)| ids.into_iter().map(|id| (p, id)).collect())
        .unwrap_or_default();
    if takes.is_empty() {
        return list.child(text("No takes yet.").font(type_::LABEL).color(t.text_secondary));
    }
    for (para, id) in takes {
        let on = s.voice.picked_take == Some(id);
        list = list.child(
            el().key(format!("take-{id}"))
                .role("button")
                .label(format!("Take {id}").as_str())
                .w_full()
                .px(space::S2)
                .py(space::S1)
                .radius(radius::SM)
                .fill(if on { tok::accent(0, true).alpha(0.25) } else { t.card })
                .on_press(move |s: &mut App| {
                    let (transcribe, voice) = (&mut s.transcribe, &mut s.voice);
                    voice.play_take(transcribe.player_mut(), para, id)
                })
                .child(text(format!("▶ Take {id}")).font(type_::BODY).color(t.text)),
        );
    }
    list
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
    if v.term_suggest.is_some() {
        main = main.child(button("voice-add-en-term", "Add to English terms?", true, |s, _| s.voice.add_en_term()));
    }
    if v.fix_word.is_some() {
        main = main.child(fix_panel(s));
    }
    main = main.child(advanced_block(s));
    if let Some(n) = &v.notice {
        main = main.child(text(n.as_str()).key("voice-notice").font(type_::LABEL).color(t.text_secondary).wrap());
    }
    main.child(el().grow(1.0)).child(play_bar(s))
}

/// Fix one word: the field says it as it should sound.
fn fix_panel(s: &App) -> El {
    let t = tok::theme(true);
    let fix = s.voice.fix_word.as_ref().expect("fix panel shows for a word");
    let heard =
        s.voice.timeline.as_ref().and_then(|tl| tl.words.get(fix.index)).map(|w| w.text.clone()).unwrap_or_default();
    let field = text_input(&fix.field, type_::BODY)
        .key("voice-say-as")
        .single_line()
        .w(220.0)
        .padding(space::S1)
        .radius(radius::SM)
        .fill(t.control_track)
        .on_edit(|s: &mut App| &mut s.voice.fix_word.as_mut().expect("fix panel shows").field);
    col()
        .gap(space::S1)
        .child(text(format!("Say it as (heard “{heard}”):")).font(type_::LABEL).color(t.text_secondary).wrap())
        .child(
            row()
                .gap(space::S2)
                .child(field)
                .child(button("voice-retake-word", "Retake word", true, |s, _| {
                    let store = s.token_store();
                    let settings = s.settings.clone();
                    let cpu = crate::services::engine::whisper_must_use_cpu(&s.transcribe.engine_state());
                    let (engine, voice) = (&mut s.transcribe, &mut s.voice);
                    voice.retake_word(engine.engine_mut(), &settings, store, cpu)
                }))
                .child(button("voice-fix-cancel", "Cancel", true, |s, _| s.voice.cancel_fix())),
        )
}

/// Advanced render settings, closed by default.
fn advanced_block(s: &App) -> El {
    let mut block = col().gap(space::S2);
    block = block.child(button("voice-advanced", "Advanced", true, |s, _| s.voice.advanced_open = !s.voice.advanced_open));
    if !s.voice.advanced_open {
        return block;
    }
    let para = s.voice.panel_para().map(|p| format!("Paragraph {p}")).unwrap_or_else(|| "Paragraph –".into());
    block
        .child(adv_text(s, "voice-guidance", "Guidance", |s| &s.voice.adv_guidance, |s| &mut s.voice.adv_guidance))
        .child(adv_text(s, "voice-seed", "Base seed", |s| &s.voice.adv_seed, |s| &mut s.voice.adv_seed))
        .child(adv_text(s, "voice-pause", "Pause", |s| &s.voice.adv_pause, |s| &mut s.voice.adv_pause))
        .child(adv_text(
            s,
            "voice-para-pause",
            "Paragraph pause",
            |s| &s.voice.adv_para_pause,
            |s| &mut s.voice.adv_para_pause,
        ))
        .child(adv_text(s, "voice-en", "English terms (,)", |s| &s.voice.adv_en, |s| &mut s.voice.adv_en))
        .child(adv_text(s, "voice-quote", "Quote terms (,)", |s| &s.voice.adv_quote, |s| &mut s.voice.adv_quote))
        .child(adv_text(s, "voice-respell", "Respell OLD=NEW (,)", |s| &s.voice.adv_respell, |s| &mut s.voice.adv_respell))
        .child(text(format!("{para} guidance/seed (empty = default)")).font(type_::LABEL).color(tok::theme(true).text_secondary))
        .child(adv_text(
            s,
            "voice-para-guidance",
            "Guidance",
            |s| &s.voice.adv_para_guidance,
            |s| &mut s.voice.adv_para_guidance,
        ))
        .child(adv_text(s, "voice-para-seed", "Seed", |s| &s.voice.adv_para_seed, |s| &mut s.voice.adv_para_seed))
        .child(button("voice-adv-apply", "Apply", true, |s, _| s.voice.apply_advanced()))
}

fn adv_text(
    s: &App,
    key: &str,
    label: &str,
    get: fn(&App) -> &TextEdit,
    set: fn(&mut App) -> &mut TextEdit,
) -> El {
    let t = tok::theme(true);
    row().gap(space::S2).child(text(label).font(type_::LABEL).color(t.text_secondary).w(140.0)).child(
        text_input(get(s), type_::BODY)
            .key(key)
            .single_line()
            .w(220.0)
            .padding(space::S1)
            .radius(radius::SM)
            .fill(t.control_track)
            .on_edit(move |s: &mut App| set(s)),
    )
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
        |s: &mut App, i| s.voice.select_word(i),
        |s: &mut App, i| s.voice.fix_word(i),
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
            let (transcribe, voice) = (&mut s.transcribe, &mut s.voice);
            voice.toggle_play(transcribe.player_mut())
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
