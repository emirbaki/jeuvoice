//! The Train tab, step 3: where the run trains.
//!
//! This PC needs a 12 GB CUDA card (read from the engine `/health`). Kaggle is
//! free T4 x2, about an hour per 1,000 steps. RunPod says "Coming later".

use rune::tok::{self, space, type_};
use rune::{El, col, row, text, text_input};

use crate::services::engine::EngineState;
use crate::state::App;
use crate::train::state::{TrainData, TrainTarget};
use crate::transcribe_tab::{button, section};

/// What This PC offers, from the engine health.
#[derive(Clone, Debug, PartialEq)]
pub enum PcOffer {
    /// A big enough card: `Ready { name }`.
    Ready { name: String },
    /// A card too small, or no CUDA card: the plain reason.
    TooSmall { reason: String },
    /// The engine is off: start the Voice tab once.
    Unknown,
}

/// Read the graphics card from the engine health. Training needs 12 GB.
pub fn this_pc(engine: &EngineState) -> PcOffer {
    let EngineState::Ready(h) = engine else { return PcOffer::Unknown };
    let gb = h.vram_total_gb.unwrap_or(0.0);
    let name = match &h.vram_total_gb {
        Some(gb) => format!("{} on {} · {gb:.0} GB", h.backend, h.device, gb = gb),
        None => format!("{} on {}", h.backend, h.device),
    };
    if h.device == "cuda" && gb >= 12.0 {
        PcOffer::Ready { name }
    } else if h.device == "cuda" {
        PcOffer::TooSmall { reason: format!("Your graphics card has {gb:.0} GB. Training needs at least 12 GB.") }
    } else {
        PcOffer::TooSmall { reason: "Training needs a CUDA card with at least 12 GB.".into() }
    }
}

/// "about 6 hours for 6,000 steps": an hour per 1,000 steps.
pub fn kaggle_estimate(steps: u32) -> String {
    let hours = steps as f32 / 1000.0;
    let when = if hours < 1.0 {
        "under an hour".into()
    } else if (hours - 1.0).abs() < f32::EPSILON {
        "about 1 hour".into()
    } else if hours.fract() == 0.0 {
        format!("about {} hours", hours as u32)
    } else {
        format!("about {hours:.1} hours")
    };
    format!("{when} for {} steps", comma_steps(steps))
}

fn comma_steps(steps: u32) -> String {
    let digits: Vec<char> = steps.to_string().chars().collect();
    let mut out = String::with_capacity(digits.len() + 2);
    for (i, c) in digits.iter().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*c);
    }
    out
}

pub fn view(s: &TrainData, engine: &EngineState) -> El {
    col().w_full().gap(space::S2).child(section("Where to train")).child(pc_card(s, engine)).child(kaggle_card(s)).child(runpod_card())
}

fn pc_card(s: &TrainData, engine: &EngineState) -> El {
    let on = s.project.target == TrainTarget::ThisPc;
    let (sub, enabled) = match this_pc(engine) {
        PcOffer::Ready { name } => (name, true),
        PcOffer::TooSmall { reason } => (reason.clone(), false),
        PcOffer::Unknown => ("The voice engine is off. Open the Voice tab once to start it.".into(), false),
    };
    card(
        "This PC",
        &sub,
        "train-pc",
        "train-pc-status",
        "train-target-pc",
        on,
        enabled,
        TrainTarget::ThisPc,
    )
}

fn kaggle_card(s: &TrainData) -> El {
    let t = crate::style::theme();
    let on = s.project.target == TrainTarget::Kaggle;
    let est = kaggle_estimate(s.project.settings.steps);
    let sub = format!("T4 ×2 · free · {est}");
    let mut inner = col().w_full().gap(space::S1).child(text("Kaggle").font(type_::BODY).color(t.text));
    inner = inner.child(text(sub.as_str()).key("train-kaggle-est").font(type_::LABEL).color(t.text_secondary).wrap());
    match &s.kaggle_user {
        Some(user) => {
            inner = inner.child(
                text(format!("Signed in as {user}").as_str()).key("train-kaggle-user").font(type_::LABEL).color(t.text_secondary),
            );
            inner = inner.child(
                row().gap(space::S2)
                    .child(button("train-kaggle-change", "Change", true, |s, _| s.train.show_kaggle_form = true))
                    .child(button("train-kaggle-forget", "Forget", true, |s, _| s.train.forget_kaggle_key())),
            );
        }
        None => {
            inner = inner.child(button("train-kaggle-add", "Add Kaggle key", true, |s, _| s.train.show_kaggle_form = true));
        }
    }
    if s.show_kaggle_form {
        inner = inner.child(kaggle_form(s));
    }
    inner = inner.child(button("train-target-kaggle", if on { "Selected" } else { "Use" }, !on, |s, _| {
        s.train.set_target(TrainTarget::Kaggle)
    }));
    card_wrap(inner, "train-kaggle", on, true)
}

fn kaggle_form(s: &TrainData) -> El {
    let t = crate::style::theme();
    let mut form = col().w_full().gap(space::S2).padding(space::S2).radius(10.0).fill(t.control_track);
    form = form.child(
        text("Find the key under your Kaggle account avatar → Settings → API → Create New Token.")
            .font(type_::LABEL)
            .color(t.text_secondary)
            .wrap(),
    );
    form = form.child(
        text_input(&s.kaggle_user_field, type_::BODY)
            .key("train-kaggle-user-field")
            .single_line()
            .w_full()
            .padding(space::S2)
            .radius(10.0)
            .fill(t.card)
            .placeholder("Kaggle user name", t.text_secondary)
            .on_edit(|s: &mut App| &mut s.train.kaggle_user_field),
    );
    form = form.child(
        text_input(&s.kaggle_key_field, type_::BODY)
            .key("train-kaggle-key-field")
            .single_line()
            .w_full()
            .padding(space::S2)
            .radius(10.0)
            .fill(t.card)
            .placeholder("Kaggle key", t.text_secondary)
            .on_edit(|s: &mut App| &mut s.train.kaggle_key_field),
    );
    form = form.child(
        row()
            .gap(space::S2)
            .child(button("train-kaggle-save", "Save key", true, |s, _| s.train.save_kaggle_key()))
            .child(button("train-kaggle-import", "Import kaggle.json", true, |s, tx| s.train.open_kaggle_import(tx)))
            .child(button("train-kaggle-cancel", "Cancel", true, |s, _| s.train.show_kaggle_form = false)),
    );
    form
}

fn runpod_card() -> El {
    card_wrap(
        col().w_full().gap(space::S1)
            .child(text("RunPod").font(type_::BODY).color(crate::style::theme().text))
            .child(text("Coming later").font(type_::LABEL).color(crate::style::theme().text_secondary)),
        "train-runpod",
        false,
        false,
    )
}

/// A target card: text on the left, the Use button on the right.
fn card(
    title: &str,
    sub: &str,
    key: &str,
    status_key: &str,
    button_key: &str,
    on: bool,
    enabled: bool,
    target: TrainTarget,
) -> El {
    let t = crate::style::theme();
    row()
        .key(key)
        .w_full()
        .gap(space::S2)
        .padding(space::S2)
        .radius(10.0)
        .fill(if on { tok::accent(0, true) } else { t.card })
        .opacity(if enabled { 1.0 } else { 0.75 })
        .child(
            col()
                .gap(space::S1)
                .grow(1.0)
                .min_w(0.0)
                .child(text(title).font(type_::BODY).color(t.text))
                .child(text(sub).key(status_key).font(type_::LABEL).color(t.text_secondary).wrap()),
        )
        .child(button(button_key, if on { "Selected" } else { "Use" }, enabled && !on, move |s: &mut App, _| {
            s.train.set_target(target)
        }))
}

/// The card box for cards with their own buttons (Kaggle, RunPod).
fn card_wrap(inner: El, key: &str, on: bool, enabled: bool) -> El {
    let t = crate::style::theme();
    col()
        .key(key)
        .w_full()
        .gap(space::S2)
        .padding(space::S2)
        .radius(10.0)
        .fill(if on { tok::accent(0, true) } else { t.card })
        .opacity(if enabled { 1.0 } else { 0.75 })
        .child(inner)
}
