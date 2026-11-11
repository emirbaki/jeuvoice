//! The Voice tab: project bar, script or timeline, takes, play bar and status.

use rune::tok::{space, type_};
use rune::{Color, El, Key, KeyEvent, ScrollSpec, TextEdit, Tx, col, el, row, text, text_input};

use crate::i18n::{tr, trf};
use crate::marks::{counts_text, marks};
use crate::state::App;
use crate::transcribe_tab::{button, red, section};
use crate::voice::state::{VOICE_SUPPORTED, VoiceStatus, VoiceTimeline};
use crate::waveform::waveform;

const LEFT_W: f32 = 200.0;

use crate::style::green;

pub fn view(s: &App) -> El {
    if !VOICE_SUPPORTED {
        return col()
            .w_full()
            .grow(1.0)
            .center()
            .child(text(tr!("The voice engine does not run on this Mac yet.", "Ses motoru bu Mac'te henüz çalışmıyor.")).font(type_::BODY).color(crate::style::theme().text));
    }
    row()
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .gap(space::S3)
        .padding(space::S3)
        .child(crate::style::glass_blur("glass-voice-takes", side_column(s).padding(space::S3)))
        .child(
            col()
                .grow(1.0)
                .min_w(0.0)
                .h_full()
                .gap(space::S3)
                .child(crate::style::glass_blur("glass-voice-editor", main_column(s).w_full().padding(space::S3)).grow(1.0).min_h(0.0))
                .child(crate::style::glass_blur("glass-voice-play", play_bar(s))),
        )
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
    let t = crate::style::theme();
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
                .radius(10.0)
                .fill(if on { crate::style::palette().accent.alpha(0.22) } else { t.control_track })
                .hover(if on { crate::style::palette().accent.alpha(0.30) } else { crate::style::hover_fill() })
                .cursor(rune::Cursor::Pointer)
                .on_press(move |s: &mut App| s.voice.preset_name = name.clone())
                .child(text(label).font(type_::LABEL).color(t.text)),
        );
    }
    let mut side = col().w(LEFT_W).h_full().min_h(0.0).gap(space::S2).child(section(tr!("Voice", "Ses"))).child(list);
    side = side.child(section(tr!("Takes", "Çekimler"))).child(takes_panel(s));
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
                .flex_wrap()
                .child(button("voice-retake", tr!("Retake region", "Bölgeyi yeniden çek"), true, |s, _| {
                    let store = s.token_store();
                    let settings = s.settings.clone();
                    let cpu = crate::services::engine::whisper_must_use_cpu(&s.transcribe.engine_state());
                    let (engine, voice) = (&mut s.transcribe, &mut s.voice);
                    voice.retake(engine.engine_mut(), &settings, store, cpu)
                }))
                .child(button("voice-try-more", tr!("+ Try 3 more", "+ 3 tane daha dene"), true, |s, _| {
                    let store = s.token_store();
                    let settings = s.settings.clone();
                    let cpu = crate::services::engine::whisper_must_use_cpu(&s.transcribe.engine_state());
                    let (engine, voice) = (&mut s.transcribe, &mut s.voice);
                    voice.try_more(engine.engine_mut(), &settings, store, cpu)
                })),
        )
        .child(button("voice-use-take", tr!("Use this take", "Bu çekimi kullan"), true, |s, _| {
            s.voice.use_take()
        }))
        .child(
            row().w_full().center_y().gap(space::S2).child(button(
                "voice-undo",
                tr!("Undo", "Geri al"),
                s.voice.history.can_undo(),
                |s, _| s.voice.undo(),
            ))
            .child(button("voice-redo", tr!("Redo", "Yinele"), s.voice.history.can_redo(), |s, _| s.voice.redo())),
        )
}

/// One take row: tap to play and pick it for "Use this take".
fn takes_list(s: &App) -> El {
    let t = crate::style::theme();
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
        return list.child(text(tr!("No takes yet.", "Henüz çekim yok.")).font(type_::LABEL).color(t.text_secondary));
    }
    for (para, id) in takes {
        let on = s.voice.picked_take == Some(id);
        list = list.child(
            el().key(format!("take-{id}"))
                .role("button")
                .label(trf!("Take {id}", "Çekim {id}").as_str())
                .w_full()
                .px(space::S2)
                .py(space::S1)
                .radius(10.0)
                .fill(if on { crate::style::palette().accent.alpha(0.18) } else { t.card })
                .hover(if on { crate::style::palette().accent.alpha(0.28) } else { crate::style::hover_fill() })
                .cursor(rune::Cursor::Pointer)
                .on_press(move |s: &mut App| {
                    let (transcribe, voice) = (&mut s.transcribe, &mut s.voice);
                    voice.play_take(transcribe.player_mut(), para, id)
                })
                .child(text(trf!("▶ Take {id}", "▶ Çekim {id}")).font(type_::BODY).color(t.text)),
        );
    }
    list
}

fn main_column(s: &App) -> El {
    let t = crate::style::theme();
    let v = &s.voice;
    let title = v.project.as_ref().map(|p| p.name.clone()).unwrap_or_else(|| tr!("No project", "Proje yok").into());
    let header = row()
        .w_full()
        .center_y()
        .gap(space::S1)
        .child(text(title).key("voice-project").font(type_::TITLE).color(t.text).grow(1.0).truncate())
        .child(button("voice-open", tr!("Open…", "Aç…"), true, |s, tx| s.voice.open_dialog(tx)))
        .child(button("voice-new", tr!("New", "Yeni"), true, |s, _| s.voice.new_project()))
        .child(crate::style::button_kind("voice-generate", tr!("Generate", "Üret"), crate::style::ButtonKind::Main, v.project.is_some() || !v.script_field.text.trim().is_empty(), |s, _| {
            s.generate_voice()
        }))
        .child(button("voice-export-wav", tr!("Export WAV", "WAV dışa aktar"), v.timeline.is_some(), |s, tx| s.voice.export_wav(tx)))
        .child(button("voice-export-srt", tr!("Export SRT", "SRT dışa aktar"), v.timeline.is_some(), |s, tx| s.voice.export_srt(tx)));

    let mut main = col()
        .key("voice-main")
        .grow(1.0)
        .min_w(0.0)
        .h_full()
        .min_h(0.0)
        .gap(space::S2)
        .scroll(ScrollSpec { stick_bottom: false })
        .child(header)
        .child(name_row(s));
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
        main = main.child(button("voice-add-en-term", tr!("Add to English terms?", "İngilizce terimlere eklensin mi?"), true, |s, _| s.voice.add_en_term()));
    }
    if v.fix_word.is_some() {
        main = main.child(fix_panel(s));
    }
    main = main.child(advanced_block(s));
    if let Some(n) = &v.notice {
        main = main.child(text(crate::core_text::message(n)).key("voice-notice").font(type_::LABEL).color(t.text_secondary).wrap());
    }
    main
}

/// Fix one word: the field says it as it should sound.
fn fix_panel(s: &App) -> El {
    let t = crate::style::theme();
    let fix = s.voice.fix_word.as_ref().expect("fix panel shows for a word");
    let heard =
        s.voice.timeline.as_ref().and_then(|tl| tl.words.get(fix.index)).map(|w| w.text.clone()).unwrap_or_default();
    let field = text_input(&fix.field, type_::BODY)
        .key("voice-say-as")
        .single_line()
        .w(220.0)
        .padding(space::S1)
        .radius(10.0)
        .fill(t.control_track)
        .on_edit(|s: &mut App| &mut s.voice.fix_word.as_mut().expect("fix panel shows").field);
    col()
        .gap(space::S1)
        .child(text(trf!("Say it as (heard “{heard}”):", "Şöyle söylensin (duyulan: “{heard}”):")).font(type_::LABEL).color(t.text_secondary).wrap())
        .child(
            row()
                .gap(space::S2)
                .child(field)
                .child(button("voice-retake-word", tr!("Retake word", "Sözcüğü yeniden çek"), true, |s, _| {
                    let store = s.token_store();
                    let settings = s.settings.clone();
                    let cpu = crate::services::engine::whisper_must_use_cpu(&s.transcribe.engine_state());
                    let (engine, voice) = (&mut s.transcribe, &mut s.voice);
                    voice.retake_word(engine.engine_mut(), &settings, store, cpu)
                }))
                .child(button("voice-fix-cancel", tr!("Cancel", "İptal"), true, |s, _| s.voice.cancel_fix())),
        )
}

/// Advanced render settings, closed by default.
fn advanced_block(s: &App) -> El {
    let mut block = col().gap(space::S2);
    block = block.child(button("voice-advanced", tr!("Advanced", "Gelişmiş"), true, |s, _| s.voice.advanced_open = !s.voice.advanced_open));
    if !s.voice.advanced_open {
        return block;
    }
    let para = s.voice.panel_para().map(|p| trf!("Paragraph {p}", "Paragraf {p}")).unwrap_or_else(|| tr!("Paragraph –", "Paragraf –").into());
    block
        .child(adv_text(s, "voice-guidance", tr!("Guidance", "Yönlendirme"), |s| &s.voice.adv_guidance, |s| &mut s.voice.adv_guidance))
        .child(adv_text(s, "voice-seed", tr!("Base seed", "Temel tohum"), |s| &s.voice.adv_seed, |s| &mut s.voice.adv_seed))
        .child(adv_text(s, "voice-pause", tr!("Pause", "Duraklama"), |s| &s.voice.adv_pause, |s| &mut s.voice.adv_pause))
        .child(adv_text(
            s,
            "voice-para-pause",
            tr!("Paragraph pause", "Paragraf duraklaması"),
            |s| &s.voice.adv_para_pause,
            |s| &mut s.voice.adv_para_pause,
        ))
        .child(adv_text(s, "voice-en", tr!("English terms (,)", "İngilizce terimler (,)"), |s| &s.voice.adv_en, |s| &mut s.voice.adv_en))
        .child(adv_text(s, "voice-quote", tr!("Quote terms (,)", "Tırnaklı terimler (,)"), |s| &s.voice.adv_quote, |s| &mut s.voice.adv_quote))
        .child(adv_text(s, "voice-respell", tr!("Respell OLD=NEW (,)", "Okunuş ESKİ=YENİ (,)"), |s| &s.voice.adv_respell, |s| &mut s.voice.adv_respell))
        .child(text(trf!("{para} guidance/seed (empty = default)", "{para} yönlendirme/tohum (boş = varsayılan)")).font(type_::LABEL).color(crate::style::theme().text_secondary))
        .child(adv_text(
            s,
            "voice-para-guidance",
            tr!("Guidance", "Yönlendirme"),
            |s| &s.voice.adv_para_guidance,
            |s| &mut s.voice.adv_para_guidance,
        ))
        .child(adv_text(s, "voice-para-seed", tr!("Seed", "Tohum"), |s| &s.voice.adv_para_seed, |s| &mut s.voice.adv_para_seed))
        .child(button("voice-adv-apply", tr!("Apply", "Uygula"), true, |s, _| s.voice.apply_advanced()))
}

fn adv_text(
    s: &App,
    key: &str,
    label: &str,
    get: fn(&App) -> &TextEdit,
    set: fn(&mut App) -> &mut TextEdit,
) -> El {
    let t = crate::style::theme();
    row().gap(space::S2).child(text(label).font(type_::LABEL).color(t.text_secondary).w(140.0)).child(
        text_input(get(s), type_::BODY)
            .key(key)
            .single_line()
            .w(220.0)
            .padding(space::S1)
            .radius(10.0)
            .fill(t.control_track)
            .on_edit(move |s: &mut App| set(s)),
    )
}

fn name_row(s: &App) -> El {
    let t = crate::style::theme();
    let field = text_input(&s.voice.name_field, type_::BODY)
        .key("voice-name")
        .single_line()
        .w(220.0)
        .padding(space::S1)
        .radius(10.0)
        .fill(t.card)
        .on_edit(|s: &mut App| &mut s.voice.name_field);
    row().gap(space::S2).child(text(tr!("Project:", "Proje:")).font(type_::LABEL).color(t.text_secondary)).child(field)
}

fn script_field(s: &App) -> El {
    let t = crate::style::theme();
    text_input(&s.voice.script_field, type_::BODY)
        .key("voice-script")
        .w_full()
        .h(220.0)
        .padding(space::S2)
        .radius(10.0)
        .fill(t.card)
        .border(1.0, t.card_border)
        .placeholder(tr!("Paste the script here, one paragraph per block", "Metni buraya yapıştırın, her blokta bir paragraf"), t.text_secondary)
        .on_edit(|s: &mut App| &mut s.voice.script_field)
}

fn wave_card(s: &App, tl: &VoiceTimeline) -> El {
    let t = crate::style::theme();
    let card = col().w_full().padding(space::S2).radius(12.0).fill(t.card).gap(space::S2);
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
        VoiceStatus::Idle => tr!("Pick a preset, paste a script and press Generate.", "Bir hazır ayar seçin, metni yapıştırın ve Üret'e basın.").into(),
        VoiceStatus::StartingEngine => tr!("Starting the voice engine… This takes about a minute the first time.", "Ses motoru başlıyor… İlk seferde yaklaşık bir dakika sürer.").into(),
        VoiceStatus::Working(m) => m.clone(),
        VoiceStatus::Ready => tr!("Done.", "Bitti.").into(),
        VoiceStatus::Failed(e) => crate::core_text::message(e),
    }
}

fn status_color(status: &VoiceStatus) -> Color {
    match status {
        VoiceStatus::Failed(_) => red(),
        VoiceStatus::Ready => green(),
        _ => crate::style::theme().text,
    }
}

fn play_bar(s: &App) -> El {
    let t = crate::style::theme();
    let duration = s.voice.timeline.as_ref().map_or(0.0, |l| l.duration);
    let label = if s.voice.is_playing(s.transcribe.player()) { tr!("Pause", "Duraklat") } else { tr!("Play", "Çal") };
    row()
        .w_full()
        .center_y()
        .gap(space::S2)
        .padding(space::S2)
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
