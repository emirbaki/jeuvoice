//! The Train tab: from clips to a Kaggle run, step by step.
//!
//! Step 1 (data) lives here: sources, clips, review and the training project
//! file. Steps 2–4 (model, target, run) come in later tasks.

pub mod bundle;
pub mod data_step;
pub mod state;

#[cfg(test)]
mod tests;
