"""Start the engine: `python -m jeuvoice_engine --port 0 --token <T> --data-dir <D>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .server import READY_PREFIX, serve


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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
