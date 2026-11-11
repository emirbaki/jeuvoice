"""The parent watch: the engine exits when the app closes its stdin, and the
watch never blocks a native import (the voice model load imports scipy's BLAS
DLL in a worker thread while the HTTP server starts request threads)."""

import subprocess
import sys
from pathlib import Path

ENGINE = str(Path(__file__).resolve().parents[1])

IMPORT_WHILE_WATCHING = f"""
import sys, threading
sys.path.insert(0, {ENGINE!r})
from jeuvoice_engine.__main__ import start_parent_watch
start_parent_watch()
done = threading.Event()

def work():
    import scipy.linalg  # noqa: F401 - loads the BLAS DLL and its Fortran runtime
    done.set()

threading.Thread(target=work, daemon=True).start()
while not done.wait(0.05):
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
print("imported", flush=True)
"""

WATCH_ONLY = f"""
import sys, time
sys.path.insert(0, {ENGINE!r})
from jeuvoice_engine.__main__ import start_parent_watch
start_parent_watch()
print("watching", flush=True)
time.sleep(60)
"""


def test_a_native_import_finishes_while_the_parent_watch_runs():
    p = subprocess.Popen([sys.executable, "-c", IMPORT_WHILE_WATCHING], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        # stdin stays open, as the app keeps it: only the import can end the script.
        p.wait(timeout=90)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        raise AssertionError("the import hung while the parent watch ran") from None
    assert p.stdout.read().decode().strip() == "imported"


def test_closing_stdin_stops_the_process():
    p = subprocess.Popen([sys.executable, "-c", WATCH_ONLY], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert p.stdout.readline().decode().strip() == "watching"
    p.stdin.close()
    try:
        assert p.wait(timeout=10) == 0
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        raise AssertionError("the engine kept running after the app closed its stdin") from None
