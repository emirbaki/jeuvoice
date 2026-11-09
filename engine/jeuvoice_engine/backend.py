"""Backend interface, fake backend and part seeds. No torch import here."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .text import Part

SAMPLE_RATE = 24000


def seed_for(part_index: int, base_seed: int, has_prompt: bool) -> int:
    """Part seed, as in the CLI: seed + i (1-based) with a reference clip, else seed."""
    return base_seed + part_index if has_prompt else base_seed


class Backend(Protocol):
    @property
    def device(self) -> str: ...
    @property
    def loaded(self) -> bool: ...
    def load(self, model: str, adapter_dir: str | None, device: str) -> None: ...
    def unload(self) -> None: ...
    def make_prompt(self, ref_wav: str | None, ref_text: str | None) -> object | None: ...
    def render(
        self,
        part: Part,
        prompt: object | None,
        instruct: str | None,
        seed: int,
        guidance: float,
        num_step: int,
        speed: float,
        duration: float | None,
    ) -> np.ndarray: ...


class FakeBackend:
    """Deterministic backend for tests: a 220 Hz tone, 0.06 s per text char."""

    def __init__(self) -> None:
        self._device = "none"
        self._loaded = False

    @property
    def device(self) -> str:
        return self._device

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, model: str, adapter_dir: str | None, device: str) -> None:
        self._device = device
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def make_prompt(self, ref_wav: str | None, ref_text: str | None) -> object | None:
        return ("prompt", ref_wav, ref_text) if ref_wav or ref_text else None

    def render(self, part, prompt, instruct, seed, guidance, num_step, speed, duration=None):
        rng = np.random.default_rng(seed)
        amp = 0.3 + 0.1 * rng.random()
        n = max(1, int(0.06 * len(part.text) * SAMPLE_RATE))
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        return (amp * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
