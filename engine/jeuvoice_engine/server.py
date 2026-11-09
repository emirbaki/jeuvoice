"""Local HTTP engine: health, shutdown. JSON bodies, token on every request.

Endpoints:
  GET  /health   -> {"ok": true, "backend": ..., "device": ..., "vram_total_gb": ..., "model_loaded": ..., "version": ...}
  POST /shutdown -> stops the process.
Any request without header X-JeuVoice-Token: <token> gets 401.
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


def detect_device(backend: str) -> str:
    """Device string for /health without importing torch (fake backend has none)."""
    if backend != "omnivoice":
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
    def __init__(self, backend: str, data_dir: Path, log: logging.Logger):
        self.backend = backend
        self.data_dir = data_dir
        self.log = log
        self.model_loaded = False


def make_handler(state: State, token: str, server_ref: dict):
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

        def _authorized(self) -> bool:
            return self.headers.get(TOKEN_HEADER) == token

        def do_GET(self):  # noqa: N802
            if self.path != "/health":
                self._send(404, {"ok": False, "error": "Not found."})
                return
            if not self._authorized():
                self._send(401, {"ok": False, "error": "Wrong token."})
                return
            device = detect_device(state.backend)
            self._send(200, {
                "ok": True,
                "backend": state.backend,
                "device": device,
                "vram_total_gb": vram_total_gb(device),
                "model_loaded": state.model_loaded,
                "version": __version__,
            })

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            if not self._authorized():
                self._send(401, {"ok": False, "error": "Wrong token."})
                return
            if self.path != "/shutdown":
                self._send(404, {"ok": False, "error": "Not found."})
                return
            self._send(200, {"ok": True})
            server = server_ref.get("server")

            def _stop():
                if server is not None:
                    server.shutdown()

            threading.Thread(target=_stop, daemon=True).start()

    return Handler


def serve(port: int, token: str, data_dir: Path, backend: str) -> tuple[ThreadingHTTPServer, int]:
    data_dir.mkdir(parents=True, exist_ok=True)
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("jeuvoice_engine")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(logging.FileHandler(logs / "engine.log", encoding="utf-8"))
    state = State(backend=backend, data_dir=data_dir, log=log)
    server_ref: dict = {}
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state, token, server_ref))
    server_ref["server"] = server
    return server, server.server_address[1]
