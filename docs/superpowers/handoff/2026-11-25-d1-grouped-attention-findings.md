# D1 grouped attention: findings for the supervisor (2026-11-25)

The grouped (repeat-free) attention was implemented, measured and **reverted**. This note has
everything needed to investigate it without re-deriving anything.

## 1. Where the code is

Branch `rust-cuda-phase2`, working tree on top of `651e2bb` (audit A/B, C, D3, D4 are in the
tree, uncommitted). The D1 change is **not** in the tree: `attention_bidir` is back to
`repeat_heads` + the per-query-head batched GEMMs. The bench row
`attention_bidir_padded_f16` and the production GEMM rows stay.

## 2. What was changed (and how to re-apply it)

In `crates/jeuvoice-gpu/src/ops/attention.rs`, `attention_bidir`, with
`group = hq / h_kv` and `q_rows = group * t_len`:

```rust
// remove the two repeat_heads calls
let (qs, ks, vs, ss, os) = (slice(q), slice(k), slice(v), slice(scores), slice(out));
let batch = b * h_kv;
let row_d = t_len * d;
// QK: Q [group*t, d] * K^T [d, t] per key head
blas.gemm_f16_batched(gpu, &qs, Op::N, q_rows * d, &ks, Op::T, row_d, &ss,
                      q_rows * t_len, q_rows, t_len, d, batch, 1.0, 0.0)?;
// ... the existing mask and softmax are unchanged ...
// PV: probs [group*t, t] * V [t, d] per key head
blas.gemm_f16_batched(gpu, &ss, Op::N, q_rows * t_len, &vs, Op::N, row_d,
                      &os, q_rows * d, q_rows, d, t_len, batch, 1.0, 0.0)?;
```

No other file changed: the wrapper already takes per-batch strides, and the scores/out layouts
are unchanged in memory (a key head's block of `group*t` rows is exactly the `group` query
heads' blocks, in order).

## 3. Measurements

| Item | Before | Grouped |
|---|---|---|
| attention, `[2,16,486,128]`, no padding | 1217.7 us | **882.1 us** (-27.6%) |
| attention, production padding `valid=[486,214]` | 1638.9 us | **1087.7 us** (-33.6%) |
| forward pass, 2 x 486, warm | 107.9 ms | 106.0 ms (no clear gain at this size) |
| conditioned logits vs f32 | 98.9%, max 0.0564 | 98.8%, max 0.0545 (fine) |
| **unconditioned logits vs f32** | **100%, max 0.0384** | **23.8%, max 10.7558** |

The 10 existing attention tests passed with the grouped path, including the GQA and padding
tests.

## 4. What the evidence rules in and out

- **Ruled out: the grouped strides and the memory layout.** The conditioned row (`valid = 486 =
  t`) is numerically exact, and the two logit rows are written by the same code, so the QK/PV
  strides, the score plane and the output plane are right.
- **Ruled out: the padding mask alone.** The existing padding test passes, but it uses
  `heads_q == heads_kv` (group 1), where the grouped path is identical to the old one. The
  failing case is exactly `group = 2` **and** `valid < t`. No test covers that pair.
- **Ruled in: something specific to a padded row when the score plane is `group*t` rows.**

## 5. The suspect list, in order

1. **`attn_pad_mask_f16` on a `group*t` plane** (`kernels/mod.rs`, right after
   `repeat_heads_f16`):

   ```rust
   let row = (i / t) % t;
   let col = i % t;
   let bh = i / (t * t);
   let b = bh / heads;
   let valid_b = *valid.add(b) as usize;
   let keep = if row < valid_b { col < valid_b } else { col == row };
   ```

   The index math is written for a plane whose blocks are exactly `t*t`. With `group*t` rows
   per key head the flat block is `group*t*t`, and `i / (t*t)` steps twice per key head. That is
   still the global query-head index if the layout is per query head, so this may be correct —
   but it has never been exercised with `group > 1`, and it is the only `valid`-dependent step.

2. **The softmax over the grouped rows**: `softmax_rows` is unchanged and works on flat rows, so
   this is unlikely; verify anyway with a two-row case.

3. **The PV product for a pad row**: the diagonal rule makes a pad query row a one-hot, so its
   output must be `v[row]` of its own key head. With the grouped PV the `v` row is the same, so
   this should hold.

## 6. The test to write first

A GPU test at `group = 2` and `valid < t`, against the CPU f64 reference, with every head (q and
kv) holding different values:

- `b = 2`, `hq = 4`, `h_kv = 2`, `t = 7`, `d = 8`, `valid = [7, 3]`.
- Compare every row, including the pad rows (`row >= valid`) of batch 1.
- Then a second case with the mask built by hand: a zero score everywhere except the diagonal,
  and check `row == col` survives.

If that test passes with the grouped path, bisect inside the function: run the grouped QK only,
compare the score plane against the old path, then run the mask, then the softmax, then the PV.
The first stage that differs localizes the fault.

Reproduce the failure end to end with:

```
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test forward -- --ignored --nocapture --test-threads 1 both_rows
```

The unconditioned row prints 23.8% and max 10.7558 with the grouped path, 100% and 0.0384
without it.

## 7. Why it is worth the work

The bench says 550 us per layer and about 15 ms per pass; with E2 that is roughly 0.5 s per
32-step decode. The transformation itself is sound: one batched product per key head instead of
one per query head, no K/V copy, and the score plane is unchanged.

## 8. Resolution: the mask launch used the key-head batch count

The mask index math was correct. The bug was the reused `batch` variable.

The grouped GEMMs need `b * h_kv` batches. The score tensor and the padding mask still need
`b * hq` batches. The first grouped change assigned `batch = b * h_kv`, and the existing mask
launch reused that value. With `group = 2`, the launch covered only the first half of the score
tensor. The unconditioned row lives in the second batch, so its scores received no padding mask.

The old padding test used `hq == h_kv`. Both batch counts are equal in that test, so it could not
catch the bug.

The fixed code uses two explicit values:

```rust
let kv_batch = b * h_kv;
let score_batch = b * hq;
```

Both GEMMs use `kv_batch`. The mask launch uses `score_batch`.

### Regression test

`grouped_padding_matches_the_reference` in `crates/jeuvoice-gpu/tests/attention_bidir.rs` uses:

- `b = 2`, `hq = 4`, `h_kv = 2`, `t = 7`, `d = 8`
- `valid = [7, 3]`
- different values in every Q, K and V head
- a full CPU f64 comparison and explicit checks of every padded output row

The test passed with the old repeat-heads path. It failed with the original grouped change and
passed after the batch-count fix.

### Verification after the fix

| Check | Result |
|---|---|
| real `both_rows` test | cond 98.8%, max 0.0545; uncond **100%**, max **0.0389** |
| real hidden-state test | all taps pass, logits 1693/1712 (98.9%), max 0.0557 |
| real deterministic decode | all three scripts pass, two s1 runs are identical |
| warm forward passes | 101.1 to 101.4 ms after the first warm pass |
| cold / warm s1 decode | 4.2 s / **3.57 s** |
| plain tests | 452 passed |
| `jeuvoice-gpu` tests | 114 passed |
| whisper GPU tests | 70 passed |

Isolated benchmark runs after the fix:

| Attention `[2,16,486,128]` | Before | Fixed grouped |
|---|---|---|
| without padding | 1217.7 us | **881.7 us** (-27.6%) |
| with production padding | 1638.9 us | **1297.4 us** (-20.8%) |

The earlier 1087.7 us padded grouped number was not a valid full-mask measurement. It also ran
only half of the mask. The fixed path does more work and remains faster than the repeat-heads
path.

The `kx` and `vx` scratch fields remain in the current plan to keep this fix small. They are no
longer read by `attention_bidir`. Their removal is a separate arena and API cleanup.
