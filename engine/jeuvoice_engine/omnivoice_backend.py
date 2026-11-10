"""OmniVoice backend: the real model, matching the CLI sample for sample."""

from __future__ import annotations

import numpy as np

from .backend import SAMPLE_RATE
from .text import Part


class OmniVoiceBackend:
    def __init__(self) -> None:
        self._model = None
        self._device = "none"

    @property
    def device(self) -> str:
        return self._device

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, model: str, adapter_dir: str | None, device: str) -> None:
        import torch

        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        from omnivoice import OmniVoice

        loaded = OmniVoice.from_pretrained(model, device_map=device, dtype=dtype)
        if adapter_dir:
            from omnivoice.utils.lora import load_lora_adapter

            loaded = load_lora_adapter(loaded, adapter_dir)
        # The 800 MB audio tokenizer runs out of memory beside the main model, so it runs on the CPU.
        loaded.audio_tokenizer = loaded.audio_tokenizer.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()
        self._model = loaded
        self._device = device

    def unload(self) -> None:
        self._model = None
        self._device = "none"
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def make_prompt(self, ref_wav: str | None, ref_text: str | None) -> object | None:
        assert self._model is not None, "load first"
        if not ref_wav and not ref_text:
            return None
        return self._model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=ref_text)

    def render(self, part: Part, prompt, instruct, seed, guidance, num_step, speed, duration=None) -> np.ndarray:
        import torch

        assert self._model is not None, "load first"
        torch.manual_seed(seed)
        wav = self._model.generate(
            text=part.text,
            voice_clone_prompt=prompt,
            instruct=instruct,
            language=part.language,
            num_step=num_step,
            speed=speed,
            duration=duration,
            guidance_scale=guidance,
        )[0]
        return np.asarray(wav, dtype=np.float32).squeeze()
