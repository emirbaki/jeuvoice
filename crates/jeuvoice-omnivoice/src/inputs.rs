//! The checks on the user's token ids before a pass uploads them (plan Phase 2, Task 9).
//!
//! The kernels index the embedding tables with these ids, so a negative id becomes a huge
//! `usize` and misses the table. The check is pure, so the plain suite tests it without a GPU.

use crate::{Result, TtsError};

/// The sizes a pass needs to check an id.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// The number of audio codebooks, the first dimension of the id block.
    pub codebooks: usize,
    /// The rows of the text table.
    pub vocab: usize,
    /// The rows of the audio table for one codebook.
    pub audio_vocab: usize,
}

/// Checks the ids, the audio mask and the valid lengths of one pass.
///
/// The ids are `[rows · codebooks, t]`, the mask `[rows, t]` (1 for an audio frame), and the
/// optional `valid` holds the real length of every row. Returns `TtsError::Render` with the
/// row, the position and the bad value: a user can act on that message.
#[allow(clippy::too_many_arguments)]
pub fn check(ids: &[i32], mask: &[i32], rows: usize, t: usize, limits: Limits, valid: Option<&[i32]>) -> Result<()> {
    let need_ids = rows * limits.codebooks * t;
    let need_mask = rows * t;
    if ids.len() != need_ids || mask.len() != need_mask {
        return Err(TtsError::Render(format!(
            "the pass needs {need_ids} ids and {need_mask} mask values, not {} and {}",
            ids.len(),
            mask.len()
        )));
    }
    for r in 0..rows {
        for p in 0..t {
            let at = r * t + p;
            let audio = mask[at];
            if audio != 0 && audio != 1 {
                return Err(TtsError::Render(format!(
                    "the audio mask at row {r}, position {p} is {audio}, not 0 or 1"
                )));
            }
            for c in 0..limits.codebooks {
                let id = ids[(r * limits.codebooks + c) * t + p];
                let (limit, what) = if audio == 1 {
                    (limits.audio_vocab, "audio")
                } else {
                    (limits.vocab, "text")
                };
                if id < 0 || id as usize >= limit {
                    return Err(TtsError::Render(format!(
                        "the {what} id {id} at row {r}, position {p}, codebook {c} is outside 0..{limit}"
                    )));
                }
            }
        }
    }
    if let Some(lengths) = valid {
        if lengths.len() != rows {
            return Err(TtsError::Render(format!(
                "the valid lengths have {} values, not one per row ({rows})",
                lengths.len()
            )));
        }
        for (r, &len) in lengths.iter().enumerate() {
            if len < 0 || len as usize > t {
                return Err(TtsError::Render(format!(
                    "the valid length of row {r} is {len}, not in 0..={t}"
                )));
            }
        }
    }
    Ok(())
}
