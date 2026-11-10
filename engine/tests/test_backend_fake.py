"""Fake backend tests (no GPU, no torch)."""


import numpy as np

from jeuvoice_engine.backend import FakeBackend, seed_for
from jeuvoice_engine.text import Part


def part(text="hello world", index=1):
    return Part(index=index, paragraph=0, language="tr", text=text)


def test_render_length_follows_text():
    b = FakeBackend()
    b.load("m", None, "cpu")
    wav = b.render(part("abcd"), None, None, seed=1, guidance=2.5, num_step=32, speed=1.0)
    assert len(wav) == int(0.06 * 4 * 24000)
    assert wav.dtype == np.float32


def test_seed_changes_samples():
    b = FakeBackend()
    a = b.render(part("hello world"), None, None, seed=1, guidance=2.5, num_step=32, speed=1.0)
    c = b.render(part("hello world"), None, None, seed=2, guidance=2.5, num_step=32, speed=1.0)
    assert not np.allclose(a, c)


def test_seed_for():
    assert seed_for(3, 1, True) == 4
    assert seed_for(3, 1, False) == 1


def test_load_unload_round_trip():
    b = FakeBackend()
    assert not b.loaded
    b.load("m", None, "cpu")
    assert b.loaded and b.device == "cpu"
    b.unload()
    assert not b.loaded
