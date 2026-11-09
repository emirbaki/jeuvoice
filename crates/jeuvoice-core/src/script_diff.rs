//! Compare the written script with the heard words. Pure functions only.
//!
//! Both sides are normalised to tokens, diffed with `similar`, then mapped back to the original
//! words (script: whitespace-separated words; heard: `Transcript::words`). A number can become
//! several tokens; all tokens of one original word give one op. When one script word matches
//! several heard words (or the reverse), the op points at the first heard word of the match and
//! the other heard words get no op of their own.
use std::collections::HashMap;

use similar::{Algorithm, DiffOp as SimOp, capture_diff_slices};

use crate::transcript::Transcript;
use crate::turkish::{normalize_indexed, normalize_words};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffOp {
    Same { script: usize, heard: usize },
    Missing { script: usize },
    Added { heard: usize },
    Changed { script: usize, heard: usize },
}

/// One token-level result, already mapped to original word indices.
struct Event {
    equal: bool,
    script: Option<usize>,
    heard: Option<usize>,
}

fn find(parent: &mut [usize], x: usize) -> usize {
    let mut r = x;
    while parent[r] != r {
        r = parent[r];
    }
    let mut c = x;
    while parent[c] != r {
        let n = parent[c];
        parent[c] = r;
        c = n;
    }
    r
}

/// Token-level diff. A run of deletes followed right away by inserts pairs up 1-to-1 as
/// changes when both runs have equal length. Otherwise they stay deletes and inserts.
fn token_events(a: &[(String, usize)], b: &[(String, usize)]) -> Vec<Event> {
    let at: Vec<&str> = a.iter().map(|(s, _)| s.as_str()).collect();
    let bt: Vec<&str> = b.iter().map(|(s, _)| s.as_str()).collect();
    let ev = |equal: bool, i: Option<usize>, j: Option<usize>| Event {
        equal,
        script: i.map(|i| a[i].1),
        heard: j.map(|j| b[j].1),
    };
    let mut events = Vec::new();
    let mut dels: Vec<usize> = Vec::new();
    let flush = |events: &mut Vec<Event>, dels: &mut Vec<usize>, ins: &[usize]| {
        let n = if dels.len() == ins.len() { dels.len() } else { 0 };
        for k in 0..n {
            events.push(ev(false, Some(dels[k]), Some(ins[k])));
        }
        for &i in &dels[n..] {
            events.push(ev(false, Some(i), None));
        }
        for &j in &ins[n..] {
            events.push(ev(false, None, Some(j)));
        }
        dels.clear();
    };
    for op in capture_diff_slices(Algorithm::Myers, &at, &bt) {
        match op {
            SimOp::Equal { old_index, new_index, len } => {
                flush(&mut events, &mut dels, &[]);
                for k in 0..len {
                    events.push(ev(true, Some(old_index + k), Some(new_index + k)));
                }
            }
            SimOp::Delete { old_index, old_len, .. } => {
                flush(&mut events, &mut dels, &[]);
                dels.extend(old_index..old_index + old_len);
            }
            SimOp::Insert { new_index, new_len, .. } => {
                let ins: Vec<usize> = (new_index..new_index + new_len).collect();
                flush(&mut events, &mut dels, &ins);
            }
            SimOp::Replace { old_index, old_len, new_index, new_len } => {
                flush(&mut events, &mut dels, &[]);
                dels.extend(old_index..old_index + old_len);
                let ins: Vec<usize> = (new_index..new_index + new_len).collect();
                flush(&mut events, &mut dels, &ins);
            }
        }
    }
    flush(&mut events, &mut dels, &[]);
    events
}

/// Diff of the script against the heard words. Indices point to script words (split on
/// whitespace) and to `Transcript::words`.
pub fn script_diff(script: &str, t: &Transcript) -> Vec<DiffOp> {
    let a = normalize_indexed(script);
    // Heard words keep their `Transcript::words` index, and a number ending in "." reads the
    // next heard word to tell an ordinal from a sentence end.
    let heard: Vec<&str> = t.words.iter().map(|w| w.text.as_str()).collect();
    let b = normalize_words(&heard);
    let events = token_events(&a, &b);

    // Events that share an original word belong to one cluster, so each word gets one op.
    let ns = script.split_whitespace().count();
    let mut parent: Vec<usize> = (0..ns + t.words.len()).collect();
    for e in &events {
        if let (Some(s), Some(h)) = (e.script, e.heard) {
            let (rs, rh) = (find(&mut parent, s), find(&mut parent, ns + h));
            parent[rs] = rh;
        }
    }
    let mut order: Vec<usize> = Vec::new();
    let mut clusters: HashMap<usize, Vec<&Event>> = HashMap::new();
    for e in &events {
        let node = e.script.unwrap_or_else(|| ns + e.heard.unwrap_or(0));
        let root = find(&mut parent, node);
        clusters
            .entry(root)
            .or_insert_with(|| {
                order.push(root);
                Vec::new()
            })
            .push(e);
    }

    let mut ops = Vec::new();
    for root in order {
        let evs = &clusters[&root];
        let mut s: Vec<usize> = evs.iter().filter_map(|e| e.script).collect();
        let mut h: Vec<usize> = evs.iter().filter_map(|e| e.heard).collect();
        s.dedup();
        h.dedup();
        let all_equal = evs.iter().all(|e| e.equal);
        let heard_for = |k: usize| h[k.min(h.len().saturating_sub(1))];
        if h.is_empty() {
            ops.extend(s.iter().map(|&script| DiffOp::Missing { script }));
        } else if s.is_empty() {
            ops.extend(h.iter().map(|&heard| DiffOp::Added { heard }));
        } else if all_equal {
            ops.extend(s.iter().enumerate().map(|(k, &script)| DiffOp::Same { script, heard: heard_for(k) }));
        } else {
            ops.extend(s.iter().enumerate().map(|(k, &script)| DiffOp::Changed { script, heard: heard_for(k) }));
            // Heard words beyond a 1-to-1 pairing with several script words stay Added.
            if s.len() > 1 {
                ops.extend(h.iter().skip(s.len()).map(|&heard| DiffOp::Added { heard }));
            }
        }
    }
    ops
}
