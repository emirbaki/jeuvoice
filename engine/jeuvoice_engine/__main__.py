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
    if sys.platform == "win32":
        _poll_stdin_pipe_windows()
    else:
        try:
            while sys.stdin.buffer.read(1):
                pass
        except Exception:
            pass
    os._exit(0)


def _poll_stdin_pipe_windows() -> None:
    """Return when the stdin pipe breaks. Never keep a read open on it.

    A blocking read on a synchronous pipe handle holds the handle. A DLL that
    starts its Fortran runtime (scipy's BLAS, loaded by the voice model) then
    waits for that handle while it holds the Windows loader lock, so the load
    and every new thread stop for good. `PeekNamedPipe` checks the pipe and
    returns at once.
    """
    import ctypes
    import msvcrt
    import time
    from ctypes import wintypes

    peek = ctypes.windll.kernel32.PeekNamedPipe
    peek.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    peek.restype = wintypes.BOOL
    try:
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
    except Exception:
        return
    avail = wintypes.DWORD(0)
    while True:
        if not peek(handle, None, 0, None, ctypes.byref(avail), None):
            return  # broken pipe: the app is gone
        if avail.value:
            # Data is there, so this read returns at once. The app sends none today.
            try:
                if not sys.stdin.buffer.read1(avail.value):
                    return
            except Exception:
                return
        time.sleep(0.5)


def start_parent_watch() -> None:
    """Exit when the app closes our stdin. A terminal run has a TTY and skips this."""
    if not sys.stdin.isatty():
        threading.Thread(target=_exit_when_parent_closes, daemon=True).start()


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
    start_parent_watch()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
