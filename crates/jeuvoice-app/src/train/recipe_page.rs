//! The tr2 recipe page: how tr2 was made, and "Train like tr2".
//!
//! The recipe (`assets/recipes/tr2.json`) names every source with its weight,
//! license and note, the base model and the run settings. `Copy recipe` puts
//! the JSON on the clipboard; `Train like tr2` fills the data step from it.

use jeuvoice_core::train::Recipe;
use rune::tok::{self, radius, space, type_};
use rune::{El, col, row, text};

use crate::transcribe_tab::{button, section};

/// The shipped tr2 recipe. Panics only when the asset itself breaks.
pub fn tr2_recipe() -> Recipe {
    serde_json::from_str(include_str!("../../../../assets/recipes/tr2.json")).expect("the shipped tr2 recipe parses")
}

/// Copy the recipe JSON to the OS clipboard.
pub fn copy_recipe() -> Result<(), String> {
    let text = include_str!("../../../../assets/recipes/tr2.json");
    arboard::Clipboard::new()
        .and_then(|mut cb| cb.set_text(text))
        .map_err(|_| "The clipboard could not be written.".to_string())
}

pub fn view(recipe: &Recipe) -> El {
    let t = tok::theme(true);
    let mut page = col().w_full().gap(space::S2);
    page = page.child(
        row()
            .w_full()
            .gap(space::S2)
            .child(text(format!("How {} was made", recipe.name).as_str()).font(type_::TITLE).color(t.text).grow(1.0).min_w(0.0))
            .child(button("train-recipe-close", "Close", true, |s, _| s.train.show_recipe = false)),
    );
    page = page.child(
        text(recipe.notes.as_str()).key("train-recipe-notes").font(type_::BODY).color(t.text).wrap(),
    );
    page = page.child(
        text(format!("Base model: {}", recipe.base_model).as_str())
            .key("train-recipe-base")
            .font(type_::LABEL)
            .color(t.text_secondary)
            .wrap(),
    );
    let st = &recipe.settings;
    page = page.child(
        text(
            format!(
                "Steps {} · LoRA r {} alpha {} · batch {} · save every {} · lr {}",
                st.steps, st.lora_r, st.lora_alpha, st.batch_tokens, st.save_steps, st.learning_rate
            )
            .as_str(),
        )
        .key("train-recipe-settings")
        .font(type_::LABEL)
        .color(t.text_secondary)
        .wrap(),
    );
    let mut list = col().w_full().gap(space::S1);
    for (i, src) in recipe.sources.iter().enumerate() {
        let mut card = col().w_full().gap(space::S1).padding(space::S2).radius(radius::SM).fill(t.card);
        card = card.child(
            text(format!("{} · ×{}", src.name, src.weight).as_str())
                .key(format!("train-recipe-source-{i}"))
                .font(type_::BODY)
                .color(t.text),
        );
        if let Some(license) = &src.license {
            card = card.child(
                text(format!("License: {license}").as_str())
                    .key(format!("train-recipe-license-{i}"))
                    .font(type_::LABEL)
                    .color(t.text_secondary),
            );
        }
        if let Some(note) = &src.note {
            card = card.child(text(note.as_str()).font(type_::LABEL).color(t.text_secondary).wrap());
        }
        list = list.child(card);
    }
    page = page.child(section("Sources")).child(list);
    page = page.child(
        row()
            .gap(space::S2)
            .child(button("train-recipe-copy", "Copy recipe", true, |s, _| {
                match copy_recipe() {
                    Ok(()) => s.train.notice = Some("Copied the recipe.".into()),
                    Err(e) => s.train.notice = Some(e),
                }
            }))
            .child(button("train-like-tr2", "Train like tr2", true, |s, _| s.train.train_like_tr2())),
    );
    col()
        .w_full()
        .gap(space::S2)
        .padding(space::S2)
        .radius(radius::SM)
        .fill(t.card)
        .child(page)
}

/// The source rows of unpointed local recipe sources say this.
pub const LOCAL_PROMPT: &str = "Add this folder from your own disk.";
