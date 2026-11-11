//! The Train tab: from clips to a Kaggle run, step by step.
//!
//! Step 1 (data) lives in [`data_step`]; steps 2 (model) and 3 (target) in
//! [`model_step`] and [`target_step`]; step 4 (run) comes in a later task.

pub mod board;
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

/// The whole Train tab: a header, then the tile board or one open tile.
pub fn view(s: &App) -> El {
    let p = crate::style::palette();
    let adapters = model_step::known_adapters(
        &s.voice.presets,
        s.settings.local_adapter_dir.as_deref(),
        &training_root(),
    );
    let header = row()
        .w_full()
        .center_y()
        .gap(space::S2)
        .child(text("Training").font(type_::TITLE).color(p.text).grow(1.0).min_w(0.0))
        .child(crate::style::button_kind(
            "train-recipe-open",
            "How tr2 was made",
            crate::style::ButtonKind::Quiet,
            true,
            |s, _| s.train.show_recipe = true,
        ));
    let content = if s.train.show_recipe {
        crate::style::glass(
            col()
                .w_full()
                .grow(1.0)
                .min_h(0.0)
                .padding(space::S3)
                .scroll(rune::ScrollSpec { stick_bottom: false })
                .child(recipe_page::view(&recipe_page::tr2_recipe())),
        )
    } else {
        match s.train.open_tile {
            None => board::board(&s.train),
            Some(tile) => board::open_layout(s, tile, &adapters),
        }
    };
    col().w_full().grow(1.0).min_h(0.0).gap(space::S3).padding(space::S3).child(header).child(content)
}
