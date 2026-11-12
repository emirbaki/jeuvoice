"""Pitch shift and variety: the render settings that shape a voice."""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from jeuvoice_engine.backend import SAMPLE_RATE, FakeBackend
from jeuvoice_engine.jobs import Job, JobStore, make_render_run, shift_pitch


def peak_hz(y: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.fft.rfftfreq(len(y), 1 / SAMPLE_RATE)[int(np.argmax(spectrum))])


class Spy(FakeBackend):
    """The fake backend that keeps the variety of each render."""

    def __init__(self) -> None:
        super().__init__()
        self.varieties: list[float] = []

    def render(self, part, prompt, instruct, seed, guidance, num_step, speed, duration=None, variety=0.0):
        self.varieties.append(variety)
        return super().render(part, prompt, instruct, seed, guidance, num_step, speed, duration)


def render(tmp_path: Path, backend, preset: dict) -> list[dict]:
    store = JobStore(backend, logging.getLogger("test"))
    req = {
        "script": "Kısa bir cümle burada duruyor.",
        "preset": {"pause": 0.1, "para_pause": 0.3, "guidance": 2.5, "num_step": 32, "speed": 1.0, **preset},
        "terms": {},
        "base_seed": 1,
        "out_dir": str(tmp_path / "out"),
    }
    return make_render_run(store, req)(Job("j1", "render"))["parts"]


def test_shift_up_one_octave_doubles_the_pitch():
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    out = shift_pitch(tone, 12.0)
    assert len(out) == len(tone)
    assert abs(peak_hz(out) - 440.0) < 10.0


def test_zero_shift_keeps_the_samples():
    tone = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    assert np.array_equal(shift_pitch(tone, 0.0), tone)


def test_render_applies_the_pitch_shift(tmp_path):
    parts = render(tmp_path, FakeBackend(), {"pitch_shift": -12.0})
    y, _ = sf.read(parts[0]["file"], dtype="float32")
    assert abs(peak_hz(y) - 110.0) < 10.0


def test_render_without_shift_keeps_the_tone(tmp_path):
    parts = render(tmp_path, FakeBackend(), {})
    y, _ = sf.read(parts[0]["file"], dtype="float32")
    assert abs(peak_hz(y) - 220.0) < 10.0


def test_render_passes_the_variety(tmp_path):
    spy = Spy()
    render(tmp_path, spy, {"variety": 0.7})
    assert spy.varieties == [0.7]


def test_render_without_variety_is_greedy(tmp_path):
    spy = Spy()
    render(tmp_path, spy, {})
    assert spy.varieties == [0.0]
