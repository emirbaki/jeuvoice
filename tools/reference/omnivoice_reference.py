"""Write the OmniVoice main-model reference tokens and taps for the Rust Phase 2 work.

Developer tool only. It never ships, and the app never calls it.

    engine\\.venv\\Scripts\\python.exe -I tools\\reference\\omnivoice_reference.py ^
        --ref-audio D:\\MLProjelerim\\tts_lab\\data\\ref\\my_voice.ogg ^
        --ref-text "<the transcript of that clip>"

It writes, into `target/reference/omnivoice/` (not committed):

| File | Content |
|---|---|
| `<id>.det.npz` | the deterministic tokens `[8, T]` int32, the prompt frames `[8, T_ref]`, the text, the language, the target length |
| `<id>.product.npz` | the same, from the product mode (`position_temperature` 5.0, the app's settings) |
| `s1.taps.safetensors` | `embed` and `hidden_00` … `hidden_27` `[T, 1024]` f32 of the first pass, and `cond_logits_00` / `uncond_logits_00` `[8, T_target, 1025]` |

The deterministic mode sets `position_temperature = 0.0` and `class_temperature = 0.0`, so the
position choice is the plain score order. The engine's default adds Gumbel noise to the
position score, and that noise comes from `torch.rand_like`; the Rust side cannot reproduce it
without the same Philox stream. Criterion 1 compares the deterministic mode, criterion 2 the
product mode by frame count and token share.

`--bench N` renders script s1 N times through the engine's own path (`generate`), and prints
the load time and the time of the three stages: the prompt tokens, the iterative decode, and
the audio decode (which is Phase 3).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="target/reference/omnivoice")
    ap.add_argument("--scripts", default="tools/reference/omnivoice_scripts.json")
    ap.add_argument("--ref-audio", default=None, help="the voice prompt wav; default: the scripts file")
    ap.add_argument("--ref-text", default=None, help="its transcript; default: the scripts file")
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument("--adapter", default=None, help="the LoRA adapter folder, if the voice preset uses one")
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bench", type=int, default=0, help="render s1 this many times for the baseline")
    ap.add_argument("--skip-product", action="store_true")
    ap.add_argument("--skip-taps", action="store_true")
    return ap.parse_args()


def generation_config(args: argparse.Namespace, deterministic: bool) -> "object":
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    return OmniVoiceGenerationConfig(
        num_step=args.num_step,
        guidance_scale=args.guidance,
        t_shift=0.1,
        layer_penalty_factor=5.0,
        position_temperature=0.0 if deterministic else 5.0,
        class_temperature=0.0,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
    )


def run_tokens(model, prompt, script: dict, language: str, speed: float, cfg) -> tuple[np.ndarray, int, int]:
    """One script through `_preprocess_all` and `_generate_iterative`; gives the tokens only."""
    task = model._preprocess_all(
        text=[script["text"]],
        language=[language],
        ref_text=[None],
        ref_audio=[None],
        voice_clone_prompt=[prompt],
        instruct=[None],
        preprocess_prompt=cfg.preprocess_prompt,
        speed=[speed],
        duration=[None],
        normalize_text=False,
    )
    frame_rate = model.audio_tokenizer.config.frame_rate
    short_idx, long_idx = task.get_indices(cfg, frame_rate)
    if long_idx:
        raise SystemExit(f"{script['id']}: {task.target_lens[0]} frames is over the chunk threshold")
    short_task = task.slice_task(short_idx)
    with torch.no_grad():
        out = model._generate_iterative(short_task, cfg)
    return out[0].to("cpu").numpy().astype(np.int32), int(task.target_lens[0]), int(prompt.ref_audio_tokens.shape[1])


def prompt_frames(prompt) -> np.ndarray:
    t = prompt.ref_audio_tokens
    if isinstance(t, torch.Tensor):
        t = t.to("cpu").numpy()
    return np.asarray(t).astype(np.int32)


def save_npz(path: Path, script: dict, language: str, speed: float, tokens: np.ndarray, target_len: int, prompt_tokens: np.ndarray, ref_text: str) -> None:
    np.savez_compressed(
        path,
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        text=np.array(script["text"]),
        language=np.array(language),
        speed=np.array(speed, dtype=np.float32),
        target_len=np.array(target_len, dtype=np.int32),
        ref_text=np.array(ref_text),
        script_id=np.array(script["id"]),
    )
    print(f"  {path.name}: tokens {tokens.shape}, target {target_len} frames, prompt {prompt_tokens.shape[1]} frames")


def capture_taps(model, prompt, script: dict, language: str, speed: float, cfg, out_path: Path) -> None:
    """One pass of the deterministic decode, with the layer outputs of the first step saved.

    Besides the hidden states, it saves the input of that pass, so the Rust side can run the
    same ids through its own forward pass: `ids0` `[8, T]` (the conditioned row),
    `audio_mask0` `[T]` (true where the position is audio) and `positions0` `[T]`.
    """
    from safetensors.numpy import save_file

    taps: dict[str, np.ndarray] = {}
    handles = []
    llm = model.llm
    layers = getattr(getattr(llm, "model", None), "layers", None) or getattr(llm, "layers")
    marked = {"first": False}

    def layer_hook(index: int):
        def hook(_module, _inputs, output):
            if not marked["first"]:
                return None
            h = output[0] if isinstance(output, tuple) else output
            taps[f"hidden_{index:02d}"] = h[0].detach().float().to("cpu").numpy()
            return None

        return hook

    def embed_hook(_module, inputs):
        if marked["first"]:
            taps["embed"] = inputs[0][0].detach().float().to("cpu").numpy()

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(layer_hook(i)))
    handles.append(layers[0].register_forward_pre_hook(embed_hook))

    # The first step's raw logits, conditioned and unconditioned, before the guidance mix.
    original = model._predict_tokens_with_scoring

    def wrapped(c_logits, u_logits, gen_config):
        if marked["first"]:
            taps["cond_logits_00"] = c_logits[0].detach().float().to("cpu").numpy()
            taps["uncond_logits_00"] = u_logits[0].detach().float().to("cpu").numpy()
            marked["first"] = False
        else:
            marked["first"] = False
        return original(c_logits, u_logits, gen_config)

    model._predict_tokens_with_scoring = wrapped
    marked["first"] = True  # the next forward pass is step 0
    try:
        run_tokens(model, prompt, script, language, speed, cfg)
    finally:
        model._predict_tokens_with_scoring = original
        for h in handles:
            h.remove()

    save_file(taps, str(out_path))
    print(f"  {out_path.name}: {len(taps)} tensors, embed {taps['embed'].shape}")


def first_pass_inputs(model, prompt, script: dict, language: str, speed: float, cfg, out_path: Path) -> None:
    """The ids, the audio mask and the positions of the first pass of the deterministic run.

    They go into the same taps file, so the Rust forward pass reads exactly the input that
    made those hidden states.
    """
    from safetensors.numpy import save_file

    task = model._preprocess_all(
        text=[script["text"]],
        language=[language],
        ref_text=[None],
        ref_audio=[None],
        voice_clone_prompt=[prompt],
        instruct=[None],
        preprocess_prompt=cfg.preprocess_prompt,
        speed=[speed],
        duration=[None],
        normalize_text=False,
    )
    inp = model._prepare_inference_inputs(
        task.texts[0], task.target_lens[0], task.ref_texts[0], task.ref_audio_tokens[0], task.langs[0], task.instructs[0], cfg.denoise
    )
    ids = inp["input_ids"][0].to("cpu").numpy().astype(np.int32)
    mask = inp["audio_mask"][0].to("cpu").numpy().astype(np.uint8)
    positions = np.arange(ids.shape[1], dtype=np.int32)
    extra = {"ids0": ids, "audio_mask0": mask, "positions0": positions}
    # Merge into the taps file that already exists.
    from safetensors.numpy import load_file

    existing = load_file(str(out_path)) if out_path.is_file() else {}
    existing.update(extra)
    save_file(existing, str(out_path))
    print(f"  {out_path.name}: + ids0 {ids.shape}, audio_mask0 {mask.shape} ({int(mask.sum())} audio positions)")


def bench(model, prompt, script: dict, language: str, speed: float, cfg, n: int, ref_audio: str, ref_text: str) -> None:
    """The engine's own path, stage by stage. The audio decode is Phase 3, not Phase 2."""
    wall, iterative, decode = [], [], []
    frame_rate = model.audio_tokenizer.config.frame_rate
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            audio = model.generate(
                text=script["text"],
                voice_clone_prompt=prompt,
                instruct=None,
                language=language,
                num_step=cfg.num_step,
                speed=speed,
                duration=None,
                guidance_scale=cfg.guidance_scale,
                class_temperature=0.0,
            )[0]
        wall.append(time.perf_counter() - t0)

        # The same script again, with only the token stage timed: the difference to `wall` is
        # the audio decode and the post processing.
        task = model._preprocess_all(
            text=[script["text"]],
            language=[language],
            ref_text=[None],
            ref_audio=[None],
            voice_clone_prompt=[prompt],
            instruct=[None],
            preprocess_prompt=cfg.preprocess_prompt,
            speed=[speed],
            duration=[None],
            normalize_text=False,
        )
        idx, _ = task.get_indices(cfg, frame_rate)
        short_task = task.slice_task(idx)
        t1 = time.perf_counter()
        with torch.no_grad():
            tokens = model._generate_iterative(short_task, cfg)
        iterative.append(time.perf_counter() - t1)
        t2 = time.perf_counter()
        with torch.no_grad():
            model._decode_and_post_process(tokens[0], task.ref_rms[0], cfg)
        decode.append(time.perf_counter() - t2)

    audio_s = audio.shape[-1] / getattr(model, "sampling_rate", 24000)
    med = statistics.median
    print(f"engine path, {n} renders of {script['id']} ({audio_s:.2f} s of audio, {frame_rate} frames/s):")
    print(f"  whole render      {med(wall):.3f} s")
    print(f"  tokens only       {med(iterative):.3f} s   <- Phase 2")
    print(f"  audio decode      {med(decode):.3f} s   <- Phase 3")


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(args.scripts).read_text(encoding="utf-8"))
    language, speed = specs["language"], float(specs.get("speed", 1.0))
    ref_audio = args.ref_audio or specs["ref_audio"]
    ref_text = args.ref_text or specs["ref_text"]
    if not ref_audio or not ref_text:
        raise SystemExit("give --ref-audio and --ref-text, or put ref_audio and ref_text in the scripts file")

    from omnivoice import OmniVoice
    from omnivoice.utils.lora import load_lora_adapter

    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained(args.model, device_map="cuda", dtype=torch.float16)
    if args.adapter:
        model = load_lora_adapter(model, args.adapter)
    model.eval()
    # As the engine does: the 800 MB audio tokenizer does not fit beside the model on a 4 GB
    # card, and it must run in the same place as the engine's so the prompt frames match.
    model.audio_tokenizer = model.audio_tokenizer.to("cpu")
    torch.cuda.empty_cache()
    load_s = time.perf_counter() - t0
    print(f"model loaded in {load_s:.1f} s, device {model.device}")

    t0 = time.perf_counter()
    prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    print(f"prompt in {time.perf_counter() - t0:.1f} s: {prompt_frames(prompt).shape[1]} frames")

    det = generation_config(args, deterministic=True)
    prod = generation_config(args, deterministic=False)
    for script in specs["scripts"]:
        print(script["id"])
        tokens, target_len, _ = run_tokens(model, prompt, script, language, speed, det)
        save_npz(out / f"{script['id']}.det.npz", script, language, speed, tokens, target_len, prompt_frames(prompt), ref_text)
        if not args.skip_product:
            torch.manual_seed(args.seed)
            tokens = run_tokens(model, prompt, script, language, speed, prod)[0]
            save_npz(out / f"{script['id']}.product.npz", script, language, speed, tokens, target_len, prompt_frames(prompt), ref_text)

    if not args.skip_taps:
        taps_path = out / f"{specs['scripts'][0]['id']}.taps.safetensors"
        capture_taps(model, prompt, specs["scripts"][0], language, speed, det, taps_path)
        first_pass_inputs(model, prompt, specs["scripts"][0], language, speed, det, taps_path)

    if args.bench:
        bench(model, prompt, specs["scripts"][0], language, speed, prod, args.bench, ref_audio, ref_text)


if __name__ == "__main__":
    main()
