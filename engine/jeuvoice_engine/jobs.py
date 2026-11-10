"""Engine jobs: one worker thread runs load/render jobs one at a time, in order.

Errors in `error` are plain words. The traceback goes only to the log.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

from .backend import SAMPLE_RATE
from .text import Part, plan_parts

PLAIN_LOAD_ERROR = "The voice model could not be loaded. Try again."
PLAIN_RENDER_ERROR = "The render failed. Try again."


class Job:
    def __init__(self, job_id: str, kind: str):
        self.id = job_id
        self.kind = kind
        self.state = "waiting"
        self.progress = 0.0
        self.message = ""
        self.result: dict | None = None
        self.error: str | None = None
        self.cancelled = False


class JobStore:
    """FIFO jobs on one worker thread. `backend_factory` makes the backend."""

    def __init__(self, backend, log: logging.Logger):
        self.backend = backend
        self.log = log
        self.jobs: dict[str, Job] = {}
        self._ids = itertools.count(1)
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._rendering = False
        worker = threading.Thread(target=self._work, daemon=True)
        worker.start()

    def submit(self, kind: str, run) -> Job:
        job_id = f"job-{next(self._ids)}"
        job = Job(job_id, kind)
        self.jobs[job_id] = job

        def _run():
            try:
                job.state = "running"
                job.result = run(job)
                job.state = "cancelled" if job.cancelled else "done"
                if job.state == "done":
                    job.progress = 1.0
            except Exception as e:  # noqa: BLE001 - plain error to the client, traceback to the log
                self.log.error("job %s failed: %s\n%s", job_id, e, traceback.format_exc())
                job.state = "failed"
                job.error = PLAIN_LOAD_ERROR if kind == "load" else PLAIN_RENDER_ERROR

        self._queue.put((job, _run, getattr(run, "_is_render", False)))
        return job

    def _work(self):
        while True:
            job, run, is_render = self._queue.get()
            if is_render:
                with self._lock:
                    self._rendering = True
            try:
                run()
            finally:
                if is_render:
                    with self._lock:
                        self._rendering = False
                self._queue.task_done()

    def is_rendering(self) -> bool:
        with self._lock:
            return self._rendering or any(
                j.state in ("waiting", "running") and j.kind == "render" for j in self.jobs.values()
            )

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.state in ("done", "failed", "cancelled"):
            return False
        job.cancelled = True
        return True


def _mark_render(fn):
    fn._is_render = True  # type: ignore[attr-defined]
    return fn


def make_load_run(store: JobStore, model: str, adapter_dir: str | None, device: str):
    def _run(job: Job):
        job.message = "Loading the voice model…"
        store.backend.load(model, adapter_dir, resolve_device(device))
        return {"model": model, "device": store.backend.device}

    return _run


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def make_render_run(store: JobStore, req: dict):
    preset = req.get("preset") or {}
    terms = req.get("terms") or {}
    pause = float(preset.get("pause", 0.1))
    para_pause = float(preset.get("para_pause", 0.3))
    guidance = float(preset.get("guidance", 2.5))
    num_step = int(preset.get("num_step", 32))
    speed = float(preset.get("speed", 1.0))
    instruct = preset.get("instruct")
    base_seed = int(req.get("base_seed", 1))
    overrides = req.get("part_overrides") or {}
    wanted = req.get("paragraphs")
    out_dir = Path(req["out_dir"])
    script = req.get("script", "")

    @_mark_render
    def _run(job: Job):
        parts = plan_parts(
            script,
            list(terms.get("en", [])),
            list(terms.get("quote", [])),
            [tuple(x) for x in terms.get("respell", [])],
        )
        if wanted is not None:
            wanted_set = set(wanted)
            parts = [p for p in parts if p.paragraph in wanted_set]
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt = store.backend.make_prompt(preset.get("ref_audio"), preset.get("ref_text"))
        has_prompt = prompt is not None
        rendered: list[dict] = []
        total = len(parts)
        for n, part in enumerate(parts, 1):
            if job.cancelled:
                break
            job.message = f"Part {n} of {total}"
            ov = overrides.get(str(part.index), {})
            seed = int(ov.get("seed", _seed_for(part.index, base_seed, has_prompt)))
            part_guidance = float(ov.get("guidance", guidance))
            wav = store.backend.render(part, prompt, instruct, seed, part_guidance, num_step, speed, None)
            wav = np.asarray(wav, dtype=np.float32)
            file = out_dir / f"part_{part.index}.wav"
            sf.write(file, wav, SAMPLE_RATE)
            rendered.append({
                "index": part.index,
                "paragraph": part.paragraph,
                "language": part.language,
                "text": part.text,
                "seed": seed,
                "guidance": part_guidance,
                "file": str(file),
                "audio_s": round(len(wav) / SAMPLE_RATE, 2),
            })
            job.progress = n / total if total else 1.0
            time.sleep(0)  # let a cancel land between parts
        if job.cancelled:
            job.progress = len(rendered) / total if total else 0.0
            return {"parts": rendered, "pause": pause, "para_pause": para_pause}
        job.progress = 1.0
        return {"parts": rendered, "pause": pause, "para_pause": para_pause}

    return _run


def _seed_for(part_index: int, base_seed: int, has_prompt: bool) -> int:
    from .backend import seed_for

    return seed_for(part_index, base_seed, has_prompt)


def job_status(job: Job) -> dict:
    return {
        "state": job.state,
        "progress": job.progress,
        "message": job.message,
        "result": job.result,
        "error": job.error,
    }
