//! Batch queue for transcription. One worker thread runs one job at a time,
//! because the GPU has little memory.
use crate::transcript::Transcript;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::{self, JoinHandle};

/// A transcription job. It gets a progress callback (0.0 to 1.0).
pub type Job = Box<dyn FnOnce(&mut dyn FnMut(f32)) -> Result<Transcript, String> + Send>;

#[derive(Clone, Debug, PartialEq)]
pub enum ItemState {
    Waiting,
    Running(f32),
    Done(Transcript),
    Failed(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Update {
    Progress(f32),
    Done(Transcript),
    Failed(String),
}

#[derive(Default)]
pub struct Queue {
    pub items: Vec<(PathBuf, ItemState)>,
}

impl Queue {
    /// Adds paths as waiting items. A path that is already in the queue is skipped.
    pub fn add(&mut self, paths: &[PathBuf]) {
        for p in paths {
            if !self.items.iter().any(|(q, _)| q == p) {
                self.items.push((p.clone(), ItemState::Waiting));
            }
        }
    }

    pub fn next_waiting(&self) -> Option<usize> {
        self.items
            .iter()
            .position(|(_, s)| matches!(s, ItemState::Waiting))
    }

    /// Finds the next waiting item, marks it as running and returns its index.
    pub fn take_next(&mut self) -> Option<usize> {
        let i = self.next_waiting()?;
        self.items[i].1 = ItemState::Running(0.0);
        Some(i)
    }

    /// Applies a worker update to an item. An index out of range does nothing.
    pub fn apply(&mut self, i: usize, update: Update) {
        if let Some((_, state)) = self.items.get_mut(i) {
            *state = match update {
                Update::Progress(p) => ItemState::Running(p),
                Update::Done(t) => ItemState::Done(t),
                Update::Failed(e) => ItemState::Failed(e),
            };
        }
    }
}

/// Starts the worker. It runs one job at a time and exits when `jobs` closes.
/// `wake` is called after each update is sent.
pub fn spawn_worker(
    jobs: Receiver<(usize, Job)>,
    updates: Sender<(usize, Update)>,
    wake: impl Fn() + Send + 'static,
) -> JoinHandle<()> {
    thread::spawn(move || {
        while let Ok((i, job)) = jobs.recv() {
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                job(&mut |p| {
                    let _ = updates.send((i, Update::Progress(p)));
                    wake();
                })
            }));
            let last = match outcome {
                Ok(Ok(t)) => Update::Done(t),
                Ok(Err(e)) => Update::Failed(e),
                Err(_) => Update::Failed(
                    "Transcription stopped because of an internal error.".to_string(),
                ),
            };
            let _ = updates.send((i, last));
            wake();
        }
    })
}
