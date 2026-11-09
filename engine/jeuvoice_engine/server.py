"""Local HTTP engine: health, load/unload, render jobs, shutdown.

Endpoints (all JSON, all need header X-JeuVoice-Token):
  GET  /health          -> {"ok": true, "backend": ..., "device": ..., "vram_total_gb": ..., "model_loaded": ..., "version": ...}
  POST /load            {"model", "adapter_dir"|null, "device": "auto"|"cuda"|"mps"|"cpu"} -> {"job": id}
  POST /unload          -> {"ok": true} (409 while a render runs)
  POST /render          {script, paragraphs|null, preset, terms, base_seed, part_overrides, out_dir} -> {"job": id}
  GET  /jobs/<id>       -> {"state", "progress", "message", "result"|null, "error"|null}
  POST /jobs/<id>/cancel -> takes effect between parts
  POST /fetch_adapter   {"repo", "subfolder"|null, "token"|null} -> {"ok": true, "path": ...}
                       or {"ok": false, "error": plain words}
  POST /shutdown        -> stops the process.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__

TOKEN_HEADER = "X-JeuVoice-Token"
READY_PREFIX = "JEUVOICE_ENGINE_READY port="


def detect_device(state) -> str:
    """Live device: what the loaded backend reports, or the host default."""
    try:
        device = state.store.backend.device
    except Exception:
        device = "none"
    if device and device != "none":
        return device
    name = state.backend_name
    if name != "omnivoice":
        return "none"
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


def vram_total_gb(device: str) -> float | None:
    if device != "cuda":
        return None
    try:
        import torch

        return round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except Exception:
        return None


class State:
    def __init__(self, backend_name: str, data_dir: Path, log: logging.Logger):
        from .jobs import JobStore

        self.backend_name = backend_name
        self.data_dir = data_dir
        self.log = log
        if backend_name == "omnivoice":
            from .omnivoice_backend import OmniVoiceBackend

            backend: object = OmniVoiceBackend()
        else:
            from .backend import FakeBackend

            backend = FakeBackend()
        self.store = JobStore(backend, log)


def make_handler(state: State, token: str, server_ref: dict):
    from .jobs import job_status, make_load_run, make_render_run

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            state.log.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, obj: dict):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                return {}

        def do_GET(self):  # noqa: N802
            if not self._authorized():
                self._send(401, {"ok": False, "error": "Wrong token."})
                return
            if self.path == "/health":
                device = detect_device(state)
                self._send(200, {
                    "ok": True,
                    "backend": state.backend_name,
                    "device": device,
                    "vram_total_gb": vram_total_gb(device),
                    "model_loaded": state.store.backend.loaded,
                    "version": __version__,
                })
                return
            if self.path.startswith("/jobs/"):
                job = state.store.get(self.path[len("/jobs/"):])
                if job is None:
                    self._send(404, {"ok": False, "error": "No such job."})
                    return
                self._send(200, job_status(job))
                return
            self._send(404, {"ok": False, "error": "Not found."})

        def do_POST(self):  # noqa: N802
            req = self._body()
            if not self._authorized():
                self._send(401, {"ok": False, "error": "Wrong token."})
                return
            if self.path == "/shutdown":
                self._send(200, {"ok": True})
                server = server_ref.get("server")

                def _stop():
                    if server is not None:
                        server.shutdown()

                threading.Thread(target=_stop, daemon=True).start()
                return
            if self.path == "/load":
                job = state.store.submit(
                    "load",
                    make_load_run(
                        state.store,
                        str(req.get("model", "k2-fsa/OmniVoice")),
                        req.get("adapter_dir"),
                        str(req.get("device", "auto")),
                    ),
                )
                self._send(200, {"job": job.id})
                return
            if self.path == "/unload":
                if state.store.is_rendering():
                    self._send(409, {"ok": False, "error": "A render is running."})
                    return
                try:
                    state.store.backend.unload()
                except Exception as e:  # noqa: BLE001 - plain error, detail in the log
                    state.log.error("unload failed: %s", e)
                    self._send(500, {"ok": False, "error": "The voice model could not be unloaded."})
                    return
                self._send(200, {"ok": True})
                return
            if self.path == "/render":
                job = state.store.submit("render", make_render_run(state.store, req))
                self._send(200, {"job": job.id})
                return
            if self.path == "/fetch_adapter":
                from .fetch import FetchError, fetch_adapter

                try:
                    path = fetch_adapter(
                        str(req.get("repo", "")),
                        req.get("subfolder"),
                        req.get("token"),
                        state.data_dir,
                        state.log,
                    )
                except FetchError as e:
                    self._send(200, {"ok": False, "error": e.message})
                    return
                self._send(200, {"ok": True, "path": path})
                return
            if self.path.startswith("/jobs/") and self.path.endswith("/cancel"):
                job_id = self.path[len("/jobs/"):-len("/cancel")]
                if state.store.get(job_id) is None:
                    self._send(404, {"ok": False, "error": "No such job."})
                    return
                state.store.cancel(job_id)
                self._send(200, {"ok": True})
                return
            self._send(404, {"ok": False, "error": "Not found."})

        def _authorized(self) -> bool:
            return self.headers.get(TOKEN_HEADER) == token

    return Handler


def serve(port: int, token: str, data_dir: Path, backend: str) -> tuple[ThreadingHTTPServer, int]:
    data_dir.mkdir(parents=True, exist_ok=True)
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("jeuvoice_engine")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(logging.FileHandler(logs / "engine.log", encoding="utf-8"))
    state = State(backend_name=backend, data_dir=data_dir, log=log)
    server_ref: dict = {}
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state, token, server_ref))
    server_ref["server"] = server
    return server, server.server_address[1]
