"""Start the engine: `python -m jeuvoice_engine --port 0 --token <T> --data-dir <D>`."""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

from .server import READY_PREFIX, serve


def _exit_when_parent_closes() -> None:
    """The app holds our stdin pipe open. Its close (normal quit or a kill)
    means nobody can use the engine: exit without waiting for a shutdown call."""
    try:
        while sys.stdin.buffer.read(1):
            pass
    except Exception:
        pass
    os._exit(0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--token", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--backend", default="fake", choices=["omnivoice", "fake"])
    args = ap.parse_args(argv)

    server, actual = serve(args.port, args.token, Path(args.data_dir), args.backend)
    # Exactly one line on stdout, after the HTTP server listens.
    sys.stdout.write(f"{READY_PREFIX}{actual}\n")
    sys.stdout.flush()
    # The app starts us with a stdin pipe; a terminal run has a TTY and skips this.
    if not sys.stdin.isatty():
        threading.Thread(target=_exit_when_parent_closes, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
