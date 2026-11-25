//! The decode schedule (plan Phase 2, Task 9), against the engine's own counts.
//!
//! `tests/data/text_vectors.json` holds the schedule of the three scripts, computed by the
//! engine's `_get_time_steps` and the `ceil` loop of `_generate_iterative`.

use std::path::PathBuf;

use jeuvoice_omnivoice::decode::{Params, schedule};

fn vectors() -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/text_vectors.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("the vectors are valid JSON")
}

#[test]
fn the_schedule_matches_the_engine() {
    let v = vectors();
    let num_step = v["num_step"].as_u64().expect("the step count") as usize;
    let t_shift = v["t_shift"].as_f64().expect("the shift") as f32;
    let params = Params { num_step, t_shift, ..Params::default() };
    for script in v["scripts"].as_array().expect("the scripts") {
        let target_len = script["target_len"].as_u64().expect("the target length") as usize;
        let want: Vec<usize> = script["schedule"]
            .as_array()
            .expect("the schedule")
            .iter()
            .map(|v| v.as_u64().expect("a count") as usize)
            .collect();
        let got = schedule(target_len, 8, &params);
        assert_eq!(got, want, "{}: the schedule", script["id"]);
        assert_eq!(got.iter().sum::<usize>(), target_len * 8, "{}: every pair", script["id"]);
    }
}

#[test]
fn the_shift_gives_the_early_steps_fewer_pairs() {
    let params = Params { num_step: 32, t_shift: 0.1, ..Params::default() };
    let counts = schedule(214, 8, &params);
    assert!(counts[0] < counts[16], "the shifted curve unmask more later: {counts:?}");
    assert_eq!(counts.len(), 32);
}
