//! The Train tab, step 4: the run script, start, progress and outputs.
//!
//! The settings read plain and edit through Apply. `View full script` shows
//! the notebook text read-only. `Start on Kaggle` asks for confirm, then the
//! worker from [`run`](crate::train::run) takes over. Checkpoints list with a
//! Test button each (Task 8 wires it).

use rune::tok::{space, type_};
use rune::{El, ScrollSpec, Shape, canvas, col, row, text, text_input};

use crate::i18n::{tr, trf};
use crate::state::App;
use crate::train::run::{elapsed_text, slug};
use crate::train::state::{TrainData, TrainTarget};
use crate::train::target_step::kaggle_estimate;
use crate::transcribe_tab::{button};

const CURVE_H: f32 = 120.0;

pub fn view(s: &TrainData) -> El {
    col()
        .w_full()
        .gap(space::S2)
        .child(settings_card(s))
        .child(script_card(s))
        .child(start_card(s))
        .child(status_card(s))
        .child(checkpoints_card(s))
}

fn settings_card(s: &TrainData) -> El {
    let t = crate::style::theme();
    let mut card = col().w_full().gap(space::S1).padding(space::S2).radius(10.0).fill(t.card);
    let settings = &s.project.settings;
    let rows = [
        (tr!("Steps", "Adım"), &s.steps_field, "train-set-steps"),
        ("LoRA r", &s.lora_r_field, "train-set-lora-r"),
        ("LoRA alpha", &s.lora_alpha_field, "train-set-lora-alpha"),
        (tr!("Batch tokens", "Toplu token"), &s.batch_field, "train-set-batch"),
        (tr!("Save steps", "Kayıt adımı"), &s.save_steps_field, "train-set-save"),
    ];
    for (label, field, key) in rows {
        card = card.child(
            row().w_full().gap(space::S2).center_y()
                .child(text(label).font(type_::LABEL).color(t.text_secondary).w(130.0))
                .child(
                    text_input(field, type_::BODY)
                        .key(key)
                        .single_line()
                        .grow(1.0)
                        .min_w(0.0)
                        .padding(space::S2)
                        .radius(10.0)
                        .fill(t.control_track)
                        .on_edit(field_editor(key)),
                ),
        );
    }
    let _ = settings;
    card.child(button("train-run-apply", tr!("Apply", "Uygula"), true, |s, _| s.train.apply_settings()))
}

fn field_editor(key: &str) -> impl Fn(&mut App) -> &mut rune::TextEdit + 'static {
    let key = key.to_string();
    move |s: &mut App| match key.as_str() {
        "train-set-steps" => &mut s.train.steps_field,
        "train-set-lora-r" => &mut s.train.lora_r_field,
        "train-set-lora-alpha" => &mut s.train.lora_alpha_field,
        "train-set-batch" => &mut s.train.batch_field,
        _ => &mut s.train.save_steps_field,
    }
}

fn script_card(s: &TrainData) -> El {
    let t = crate::style::theme();
    let mut card = col()
        .w_full()
        .gap(space::S1)
        .padding(space::S2)
        .radius(10.0)
        .fill(t.card)
        .child(button("train-run-script", if s.show_script { tr!("Hide script", "Betiği gizle") } else { tr!("View full script", "Betiğin tamamını göster") }, true, |s, _| {
            s.train.show_script = !s.train.show_script;
        }));
    if s.show_script {
        let script = preview_notebook(s);
        card = card.child(
            col()
                .w_full()
                .grow(1.0)
                .min_h(0.0)
                .scroll(ScrollSpec { stick_bottom: false })
                .child(text(script.as_str()).key("train-script-text").font(type_::BODY.mono()).color(t.text)),
        );
    }
    card
}

/// The notebook the run would push: the current run's, or the next name.
fn preview_notebook(s: &TrainData) -> String {
    let run_name = s.run.as_ref().map(|r| r.name.clone()).unwrap_or_else(|| s.next_run_name());
    let run_id = slug(&s.project.name, &run_name);
    jeuvoice_core::kaggle_bundle::notebook(&run_id, &format!("bundle_{run_id}.txt"))
}

fn start_card(s: &TrainData) -> El {
    let t = crate::style::theme();
    let can_start = s.project.target == TrainTarget::Kaggle && !s.run_active() && !s.included_clips().is_empty();
    let mut card = col().w_full().gap(space::S1).padding(space::S2).radius(10.0).fill(t.card);
    card = card.child(
        row().w_full().gap(space::S2)
            .child(crate::style::button_kind("train-run-start", tr!("Start on Kaggle", "Kaggle'da başlat"), crate::style::ButtonKind::Main, can_start, |s, _| s.train.confirm_start = true))
            .child(button("train-bundle-save", tr!("Download bundle only", "Yalnızca paketi indir"), !s.included_clips().is_empty(), |s, tx| {
                s.train.export_bundle(tx)
            })),
    );
    if s.confirm_start {
        let est = crate::train::run::upload_estimate(&s.included_clips());
        card = card.child(
            text(trf!("Upload {est} of audio to a private Kaggle dataset and start a notebook on your Kaggle account?", "{est} ses özel bir Kaggle veri kümesine yüklensin ve Kaggle hesabınızda bir not defteri başlatılsın mı?")
                .as_str())
            .key("train-confirm-text")
            .font(type_::BODY)
            .color(t.text)
            .wrap(),
        );
        card = card.child(
            row().gap(space::S2)
                .child(button("train-run-confirm", tr!("Start", "Başlat"), true, |s, _| s.train.start_run()))
                .child(button("train-run-cancel", tr!("Cancel", "İptal"), true, |s, _| s.train.confirm_start = false)),
        );
    }
    card
}

fn status_card(s: &TrainData) -> El {
    let t = crate::style::theme();
    let mut card = col().w_full().gap(space::S1).padding(space::S2).radius(10.0).fill(t.card);
    let Some(run) = &s.run else {
        return card.child(text(tr!("No run yet.", "Henüz çalıştırma yok.")).font(type_::BODY).color(t.text_secondary));
    };
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(run.started_at);
    let mut line = format!("{} · {}", state_name(&run.state), elapsed_text(run.started_at, now));
    line += &format!(" · {}", estimate(s, now));
    if let Some((step, loss)) = s.losses.last() {
        line += &trf!(" · loss {loss:.2} at step {step}", " · {step}. adımda kayıp {loss:.2}");
    }
    card = card.child(text(line.as_str()).key("train-run-status").font(type_::BODY).color(t.text).wrap());
    if let Some((done, total)) = s.upload {
        card = card.child(
            text(tr!("Uploading the dataset… {} of {} MB", "Veri kümesi yükleniyor… {} / {} MB", done / 1_000_000, total / 1_000_000).as_str())
                .key("train-upload")
                .font(type_::LABEL)
                .color(t.text_secondary),
        );
    }
    if let Some((done, total)) = s.download {
        card = card.child(
            text(tr!("Downloading the outputs… {} of {} MB", "Çıktılar indiriliyor… {} / {} MB", done / 1_000_000, total / 1_000_000).as_str())
                .key("train-download")
                .font(type_::LABEL)
                .color(t.text_secondary),
        );
    }
    if run.state == crate::train::run::RunState::Error {
        let mut err = run.message.clone().unwrap_or_else(|| tr!("The run failed.", "Çalıştırma başarısız oldu.").into());
        // The last 30 lines: the worker already asks the log tail for 30.
        let start = s.run_log.len().saturating_sub(30);
        for line in s.run_log.iter().skip(start).take(30) {
            err.push('\n');
            err.push_str(line);
        }
        card = card.child(text(crate::core_text::message(&err)).key("train-run-error").font(type_::LABEL).color(t.text_secondary).wrap());
    }
    if s.losses.len() >= 2 {
        card = card.child(loss_curve(&s.losses));
    }
    card
}

/// What is left: from the live loss point, else the static estimate.
/// The run state in plain words.
fn state_name(state: &crate::train::run::RunState) -> &'static str {
    use crate::train::run::RunState;
    match state {
        RunState::Queued => tr!("Queued", "Sırada"),
        RunState::Running => tr!("Running", "Çalışıyor"),
        RunState::Complete => tr!("Complete", "Tamamlandı"),
        RunState::Error => tr!("Error", "Hata"),
        RunState::Cancelled => tr!("Cancelled", "İptal edildi"),
    }
}

fn estimate(s: &TrainData, now: u64) -> String {
    let steps = s.project.settings.steps;
    if let (Some((step, _)), Some(run)) = (s.losses.last(), &s.run) {
        if *step > 0 && *step < steps {
            let elapsed = now.saturating_sub(run.started_at).max(1);
            let left = elapsed as f32 / *step as f32 * (steps - *step) as f32;
            let mins = (left / 60.0).round() as u64;
            if mins < 1 {
                return tr!("under a minute left", "bir dakikadan az kaldı").into();
            } else if mins < 60 {
                return trf!("about {mins} min left", "yaklaşık {mins} dk kaldı");
            } else {
                return tr!("about {} h {:02} min left", "yaklaşık {} sa {:02} dk kaldı", mins / 60, mins % 60);
            }
        }
    }
    kaggle_estimate(steps)
}

fn loss_curve(losses: &[(u32, f32)]) -> El {
    let t = crate::style::theme();
    let width = rune::memo::layout_size("train-loss-curve").map_or(600.0, |v| v[0]).max(1.0);
    let (s_min, s_max) = (losses[0].0 as f32, losses[losses.len() - 1].0 as f32);
    let mut l_min = f32::INFINITY;
    let mut l_max = f32::NEG_INFINITY;
    for (_, l) in losses {
        l_min = l_min.min(*l);
        l_max = l_max.max(*l);
    }
    if l_max - l_min < 1e-6 {
        l_max = l_min + 1.0;
    }
    let points: Vec<[f32; 2]> = losses
        .iter()
        .map(|(s, l)| {
            let x = (*s as f32 - s_min) / (s_max - s_min).max(1.0) * width;
            let y = CURVE_H - (l - l_min) / (l_max - l_min) * (CURVE_H - 8.0) - 4.0;
            [x, y]
        })
        .collect();
    col()
        .w_full()
        .gap(space::S1)
        .child(text(tr!("Loss", "Kayıp")).font(type_::LABEL).color(t.text_secondary))
        .child(canvas(Shape::new().stroke(points, 2.0, crate::style::palette().accent)).key("train-loss-curve").w_full().h(CURVE_H).fill(t.control_track))
}

fn checkpoints_card(s: &TrainData) -> El {
    let t = crate::style::theme();
    let mut card = col().w_full().gap(space::S1).padding(space::S2).radius(10.0).fill(t.card);
    if s.checkpoints.is_empty() {
        return card.child(text(tr!("No checkpoints yet.", "Henüz kontrol noktası yok.")).font(type_::BODY).color(t.text_secondary));
    }
    for ck in &s.checkpoints {
        let mut head = row()
            .w_full()
            .gap(space::S2)
            .child(text(ck.as_str()).key(format!("train-checkpoint-{ck}")).font(type_::BODY).color(t.text).grow(1.0).min_w(0.0));
        if s.ck_replays.iter().any(|(c, _)| c == ck) {
            let slot = s.ck_replays.iter().position(|(c, _)| c == ck).unwrap_or(0);
            let letter = ["A", "B"].get(slot).unwrap_or(&"•");
            let ck = ck.clone();
            head = head.child(button(&format!("train-replay-{ck}"), &format!("▶ {letter}"), true, move |s: &mut App, _| {
                crate::train::checkpoints::replay_checkpoint(s, &ck)
            }));
        } else if s.ck_working.as_deref() == Some(ck) {
            head = head.child(button(&format!("train-testing-{ck}"), tr!("Testing…", "Deneniyor…"), false, |_, _| {}));
        } else {
            let ck = ck.clone();
            head = head.child(button(&format!("train-test-{ck}"), tr!("Test", "Dene"), true, move |s: &mut App, _| {
                crate::train::checkpoints::test_checkpoint(s, &ck)
            }));
        }
        let ck = ck.clone();
        head = head.child(button(&format!("train-use-preset-{ck}"), tr!("Use as preset", "Hazır ayar yap"), true, move |s: &mut App, _| {
            crate::train::checkpoints::use_as_preset(s, &ck)
        }));
        card = card.child(head);
    }
    card
}
