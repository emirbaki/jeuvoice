"""Recompute one OmniVoice block in numpy f32 and compare the stages with the reference taps.

Developer tool only. It reads the checkpoint, the adapter (merged), and the s1 taps, then runs
layer 0 by hand: LayerNorm, the projections, qk-norm, RoPE, the bidirectional attention, the
out projection, the residual, the MLP. Any stage that is off points at the bug of the Rust
forward pass.

    engine\\.venv\\Scripts\\python.exe -I tools\\reference\\omnivoice_layer0.py [layer]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[2]
TAPS = ROOT / "target/reference/omnivoice/f32/s1.taps.safetensors"


def base_tensor(model: Path, name: str) -> np.ndarray:
    with safe_open(str(model), framework="np") as f:
        return np.asarray(f.get_tensor(name), dtype=np.float32)


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    ms = (x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps) * w).astype(np.float32)


def rope(x: np.ndarray, theta: float) -> np.ndarray:
    """Half-split RoPE over [tokens, heads, head_dim]."""
    t, heads, d = x.shape
    half = d // 2
    inv = theta ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    pos = np.arange(t, dtype=np.float64)[:, None]
    ang = pos * inv[None, :]
    cos = np.cos(ang).astype(np.float32)[:, None, :]
    sin = np.sin(ang).astype(np.float32)[:, None, :]
    a = x[..., :half]
    b = x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = a * cos - b * sin
    out[..., half:] = a * sin + b * cos
    return out


def main() -> None:
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model_dir = Path.home() / ".cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots"
    model_file = next(model_dir.glob("*/model.safetensors"))
    adapter_file = Path.home() / "AppData/Roaming/Jeu/JeuVoice/data/models/emirbakid__jeuvoice-omnivoice-tr2/adapter_model.safetensors"

    taps = load_file(str(TAPS)) if TAPS.is_file() else load_file(str(TAPS).replace("/f32", ""))
    x = taps["embed"].astype(np.float32)  # the reference embedding, f32
    print(f"embed: {x.shape}")

    # LoRA merge for this layer: W + (alpha/r) * B @ A.
    scale = 64.0 / 32.0

    def weight(name: str) -> np.ndarray:
        w = base_tensor(model_file, name)
        a_key = f"base_model.model.{name[:-len('.weight')]}.lora_A.weight"
        b_key = f"base_model.model.{name[:-len('.weight')]}.lora_B.weight"
        with safe_open(str(adapter_file), framework="np") as f:
            if a_key in f.keys():
                a = np.asarray(f.get_tensor(a_key), dtype=np.float32)
                b = np.asarray(f.get_tensor(b_key), dtype=np.float32)
                w = w + scale * (b @ a)
        return w

    eps = 1e-6
    heads, kv_heads, head_dim = 16, 8, 128
    h = rmsnorm(x, base_tensor(model_file, f"llm.layers.{layer}.input_layernorm.weight"), eps)
    q = h @ weight(f"llm.layers.{layer}.self_attn.q_proj.weight").T
    k = h @ weight(f"llm.layers.{layer}.self_attn.k_proj.weight").T
    v = h @ weight(f"llm.layers.{layer}.self_attn.v_proj.weight").T
    t = x.shape[0]

    qh = rmsnorm(q.reshape(t, heads, head_dim), base_tensor(model_file, f"llm.layers.{layer}.self_attn.q_norm.weight"), eps)
    kh = rmsnorm(k.reshape(t, kv_heads, head_dim), base_tensor(model_file, f"llm.layers.{layer}.self_attn.k_norm.weight"), eps)
    theta = 1_000_000.0
    qh = rope(qh, theta)
    kh = rope(kh, theta)
    vh = v.reshape(t, kv_heads, head_dim)

    # The bidirectional attention: head h reads kv head h / 2.
    att = np.empty((t, heads, head_dim), dtype=np.float32)
    for hd in range(heads):
        kk = kh[:, hd // 2, :]
        scores = qh[:, hd, :] @ kk.T / np.sqrt(head_dim)
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=1, keepdims=True)
        att[:, hd, :] = p.astype(np.float32) @ vh[:, hd // 2, :]

    o = att.reshape(t, heads * head_dim) @ weight(f"llm.layers.{layer}.self_attn.o_proj.weight").T
    x1 = x + o
    h2 = rmsnorm(x1, base_tensor(model_file, f"llm.layers.{layer}.post_attention_layernorm.weight"), eps)
    gate = h2 @ weight(f"llm.layers.{layer}.mlp.gate_proj.weight").T
    up = h2 @ weight(f"llm.layers.{layer}.mlp.up_proj.weight").T
    mid = (gate / (1.0 + np.exp(-gate.astype(np.float64)))) * up
    down = mid.astype(np.float32) @ weight(f"llm.layers.{layer}.mlp.down_proj.weight").T
    x2 = x1 + down

    want = taps[f"hidden_{layer:02d}"]
    d = np.abs(x2 - want)
    print(f"block out        max {d.max():.4f} mean {d.mean():.6f}  (reference mean |v| {np.abs(want).mean():.3f})")


if __name__ == "__main__":
    main()
