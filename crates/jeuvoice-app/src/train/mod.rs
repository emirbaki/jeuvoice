//! The Train tab: from clips to a Kaggle run, step by step.
//!
//! Step 1 (data) lives in [`data_step`]; steps 2 (model) and 3 (target) in
//! [`model_step`] and [`target_step`]; step 4 (run) comes in a later task.

pub mod bundle;
pub mod checkpoints;
pub mod data_step;
pub mod model_step;
pub mod recipe_page;
pub mod run;
pub mod run_step;
pub mod state;
pub mod target_step;

#[cfg(test)]
mod tests;

use rune::tok::{space, type_};
use rune::{El, col, row, text};

use crate::state::App;
use crate::train::state::training_root;

/// The whole Train tab: data, model, target and run steps.
pub fn view(s: &App) -> El {
    let t = rune::tok::theme(true);
    let adapters = model_step::known_adapters(
        &s.voice.presets,
        s.settings.local_adapter_dir.as_deref(),
        &training_root(),
    );
    let page = col()
        .w_full()
        .gap(space::S3)
        .child(
            row()
                .w_full()
                .gap(space::S2)
                .child(text("Training").font(type_::TITLE).color(t.text).grow(1.0).min_w(0.0))
                .child(crate::transcribe_tab::button("train-recipe-open", "How tr2 was made", true, |s, _| {
                    s.train.show_recipe = true
                })),
        )
        .child(data_step::view(&s.train))
        .child(model_step::view(&s.train, &adapters))
        .child(target_step::view(&s.train, &s.transcribe.engine_state()))
        .child(run_step::view(&s.train));
    let page = if s.train.show_recipe {
        page.child(recipe_page::view(&recipe_page::tr2_recipe()))
    } else {
        page
    };
    col()
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .padding(space::S3)
        .scroll(rune::ScrollSpec { stick_bottom: false })
        .child(page)
}
