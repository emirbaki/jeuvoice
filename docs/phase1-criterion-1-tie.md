# Phase 1, criterion 1: the near tie

Status: **open**. Criterion 1 fails on one token, and the evidence says our token is the model
answer. Only the user can change a criterion.

This record holds every number and command behind that sentence. It was written on
2026-11-25 for the branch `rust-cuda-phase1b`.

## 1. What fails

Criterion 1 of the Phase 1 spec: word error rate against the whisper.cpp output, 2% or less
on each test file.

| Clip | Audio | Words ours / whisper.cpp | WER |
|---|---|---|---|
| my_voice.ogg | 27.0 s | 43 / 43 | 0.00% |
| my_voice2.mp4 | 33.6 s | 69 / 70 | **2.86%** (1 substitution, 1 deletion) |
| ses_001_0007.wav | 8.4 s | 20 / 20 | 0.00% |

The other criteria pass: word starts 0 ms (limit 50 ms), the 69 s set 2.813 s against the
whisper.cpp 3.954 s of the same run (limit 4.3 s), peak VRAM 1798 MiB against 2157 MiB.

## 2. The token

my_voice2 window 1, token index 80 of 189:

| Source | Chosen token | Probability |
|---|---|---|
| Whisper.cpp through `whisper-rs` (the reference) | `u` (id 84) | 0.5173 (after the token rules) |
| Whisper.cpp on the CPU (`GPU=0`), same fixture tool | `u` (id 84) | the same 189 tokens |
| CudaWhisper, as sampled | `uy` (id 7493) | 0.5090 (after the token rules) |
| **CudaWhisper, raw logits (teacher forced)** | `uy` (id 7493) | **0.5469** |
| &nbsp;&nbsp;same step, second candidate | `u` (id 84) | 0.4506 |
| **PyTorch f32 (transformers, teacher forced)** | `uy` (id 7493) | **0.5448** |
| &nbsp;&nbsp;same step, second candidate | `u` (id 84) | 0.4528 |

The raw distribution of CudaWhisper matches PyTorch f32 within **0.002** in probability, and
0.002 in the second candidate as well. whisper.cpp's two backends agree with each other on the
other candidate.

The two tokens after the tie:

| Index | Whisper.cpp | CudaWhisper |
|---|---|---|
| 80 | `u` | `uy` |
| 81 | ` ile` (15465) | `la` (875) |
| 82 | ` çalış` (18107) | ` çalış` (18107) |

So the two paths rejoin at token 82. 187 of the 189 tokens are identical.

## 3. The word effect

| | The words |
|---|---|
| whisper.cpp | `… kupkuru vücudu ile çalışma odasında …` |
| CudaWhisper | `… kupkuru vücuduyla çalışma odasında …` |

Two reference words (`vücudu`, `ile`) against one hypothesis word (`vücuduyla`): one
substitution and one deletion of 70 words is 2.86%.

## 4. Why this is not our arithmetic

Our decoder logits agree with the PyTorch f32 reference on the whole step list: arg max
agreement 1.0, the largest difference 0.041, the mean 0.0033 (my_voice), 0.028 and 0.0026
(ses_001_0007). The tie is the one place where a 0.002 difference in the distribution crosses
a 0.095 margin, and it does not cross it here: we stay on the model side.

whisper.cpp decides this tie from a less precise distribution. Its CUDA matmul path for f16
weights and f32 activations converts the activation to f16 and calls `cublasGemmEx` with a
**half alpha, a half destination and `CUBLAS_COMPUTE_16F`**, then converts the result to f32
(`whisper-rs-sys-0.15.0/whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`, the `use_fp16` path
starts at line 1255, the call is at lines 1326 to 1336). So whisper.cpp accumulates its
matmuls in fp16. Our matmuls (cuBLAS and the kernels) accumulate in fp32.

Its CPU backend uses ggml's f32 accumulation and still gives the same tokens as its GPU, so
the tie is not a coin flip between two error patterns of the same size: whisper.cpp's
implementations agree with each other, and the f32 model agrees with us. Our numbers are
0.002 away from PyTorch f32, which is far too close to explain the choice by rounding on our
side.

## 5. What we tried, and what it cost

The first guess was our fp16 residual stream. whisper.cpp keeps its activations in f32.

| Change | Where | Result |
|---|---|---|
| f32 residual stream, f32 LayerNorm input, f32 norm weights | 4 decoder blocks | the tie stays: `uy` 0.5113 to 0.5097; decoder logits against PyTorch f32: max 0.056 to 0.041, mean 0.0045 to 0.0033 |
| the same change in the encoder, plus f32 positional embedding | 32 encoder blocks | the tie stays: `uy` 0.5092; encoder ln_post against PyTorch f32: max 3.45 to 1.28, mean 0.00409 to 0.00171 |

Both changes are in the branch (`278f3af`). They cost about 12 MiB of VRAM and no measurable
time, and they moved both halves closer to the f32 reference. They did not move the tie,
because the tie was never ours to move.

## 6. The caveat

The PyTorch check uses the Hugging Face feature extractor, which normalizes each 30 s window,
where whisper.cpp normalizes the whole file. The two mel inputs are therefore not identical.
This is why the decoder reference tool refuses a clip longer than one window, and why the
check above used the first five segments of my_voice2 only
(`target/reference/whisper/my_voice2.win1.rs.json`).

The margin limits the risk: PyTorch puts 0.5448 against 0.4528, a gap of 0.09 in probability
(0.19 in logit space). A mel normalization difference moves logits by about 0.001, not 0.19.
The caveat does not explain the gap.

## 7. How to reproduce every number

```powershell
# The reference tokens, on the GPU and on the CPU (the two agree in all 189 tokens).
scripts\cargo.cmd run -p jeuvoice-whisper --release --example dump_whisper_rs -- D:\MLProjelerim\tts_lab\data\ref\my_voice2.mp4 target\reference\whisper\my_voice2.rs.json
$env:GPU = "0"; scripts\cargo.cmd run -p jeuvoice-whisper --release --example dump_whisper_rs -- D:\MLProjelerim\tts_lab\data\ref\my_voice2.mp4 target\reference\whisper\my_voice2.cpu.json

# Our tokens with the probabilities.
scripts\cargo.cmd oxide build -- --release -p jeuvoice-whisper --features cuda-kernels --example dump_cuda_tokens
target\release\examples\dump_cuda_tokens.exe D:\MLProjelerim\tts_lab\data\ref\my_voice2.mp4 target\reference\whisper\my_voice2.cuda.json

# Our raw top of the distribution at the tie.
scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels --release --test cuda_transcribe -- --ignored --nocapture --test-threads 1 the_two_candidates

# PyTorch f32 at the same step (needs engine\.venv and the first five segments only).
engine\.venv\Scripts\python.exe -I tools\reference\whisper_decoder_reference.py target\reference\whisper\my_voice2.f32 target\reference\whisper\my_voice2.win1.rs.json target\reference\whisper\my_voice2.win1.decoder.safetensors

# The whole criteria table.
powershell -ExecutionPolicy Bypass -File scripts\compare-whisper.ps1
```

The date of these measurements: 2026-11-25, RTX 3050 Ti Laptop (4 GB), CUDA 13.4.

## 8. The options, and why we did not take two of them

1. **Accept the criterion as met with one explained exception.** The user's call. Our output
   is closer to the model than the reference output at the only token that differs.
2. **Change the reference of criterion 1** to the PyTorch f32 transcription, or ignore near
   ties below a probability margin. This is a change to the criterion, so again the user's
   call. It needs an f32 reference transcription of all three clips (the reference tool
   handles one window per clip today).
3. **Reproduce whisper.cpp's arithmetic**: call cuBLAS with a half destination and
   `CUBLAS_COMPUTE_16F` like its `use_fp16` path. This would match the reference token by
   token in the likely case, but it makes every number in the model less accurate, and it
   would lower the agreement with the f32 reference that the rest of the branch just bought.
   Not recommended. The measured gain of the f32 residual stream would be thrown away.

## 9. Where the code is

| Part | File |
|---|---|
| the token rules that mask and renormalize | `crates/jeuvoice-whisper/src/decode/rules.rs` |
| the sampling and the sequence score | `crates/jeuvoice-whisper/src/decode/sample.rs` |
| the token dump tools | `crates/jeuvoice-whisper/examples/dump_whisper_rs.rs` (`GPU=0`), `examples/dump_cuda_tokens.rs` |
| the raw distribution at the tie | `crates/jeuvoice-whisper/tests/cuda_transcribe.rs`, `the_two_candidates_at_the_near_tie` |
| the f32 residual stream | `crates/jeuvoice-whisper/src/cuda/decoder.rs`, `encoder.rs`, `crates/jeuvoice-gpu/src/ops/mixed` ops |
| the criteria table | `scripts/compare-whisper.ps1`, `crates/jeuvoice-whisper/src/compare.rs` |
