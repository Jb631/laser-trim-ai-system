"""Tk is called from the main thread only -- including by the garbage collector.

Two separate things are pinned here. They were found together and they are NOT
the same bug; the 2x2 experiment of the H5 report (summarised in
`docs/decisions/2026-09-ledger-decisions.md`) kept them apart.

1. THE HANG (fixed by the function-scoped `tk_root` fixture in conftest.py).
   The suite was believed to "take 2.5 hours". It did not: it HUNG at >100% CPU,
   which looks identical to slow. Files that each pass alone wedged the Tk event
   loop when they shared one pytest process -- with a session-scoped `tk_root`
   plus the per-test `V6App` roots, three live CTk roots were in the process at
   once, and `Misc.update()` inside a test's settle helper then never returned.
   A regression guard for that must not itself be able to hang the suite, so it
   runs the reproduction in a SUBPROCESS with a hard wall-clock limit.

2. THE WORKER-THREAD FINALIZER (fixed by `guard_tk_font_finalizer`).
   `tkinter.font.Font.__del__` issues `font delete <name>`. Finalizers run on
   WHICHEVER thread drops the last reference, so a CTkFont orphaned by a page
   rebuild can be finalised inside an ingest worker. Measured on this machine:
   that Tcl call blocks the worker for 1.07 s in `_tkinter`'s WaitForMainloop,
   then raises `RuntimeError: main thread is not in main loop`, which `__del__`
   swallows -- and the font is not deleted anyway. So the unguarded path costs a
   full second of a worker's time and achieves nothing. The guard skips the call
   off-thread: same leak, no stall, and no Tk call from a worker, which this
   project forbids outright.
"""
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_a_font_finalised_on_a_worker_thread_makes_no_tk_call(tk_root):
    import gc
    import threading
    import tkinter.font as tkfont

    from laser_trim_analyzer.utils.threads import guard_tk_font_finalizer

    guard_tk_font_finalizer()
    calls = []

    # Dropped on a WORKER thread: the finalizer must make no Tcl call at all.
    f = tkfont.Font(root=tk_root, family="Helvetica", size=11)
    f._call = lambda *a: calls.append(a)      # every Tcl call the finalizer makes lands here
    box = [f]
    del f
    worker = threading.Thread(target=lambda: (box.clear(), gc.collect()), name="work")
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), "the worker never came back out of the finalizer"
    assert calls == [], f"a worker thread called into Tk: {calls}"

    # Dropped on the MAIN thread: unchanged, the font is still deleted.
    g = tkfont.Font(root=tk_root, family="Helvetica", size=12)
    name = g.name
    g._call = lambda *a: calls.append(a)
    del g
    gc.collect()
    assert calls == [("font", "delete", name)], calls


def test_the_guard_can_be_called_twice(tk_root):
    """Every V6App calls it on construction; wrapping the wrapper would nest."""
    import tkinter.font as tkfont

    from laser_trim_analyzer.utils.threads import guard_tk_font_finalizer

    guard_tk_font_finalizer()
    installed = tkfont.Font.__del__
    guard_tk_font_finalizer()
    assert tkfont.Font.__del__ is installed


@pytest.mark.slow
def test_two_files_that_pass_alone_still_pass_in_one_process(tmp_path):
    """The hang reproduction, in a subprocess that cannot hang this suite.

    Both targets pass on their own, so only running them TOGETHER catches this
    -- which is exactly the shape of bug a hand-picked gate file list can never
    see. `timeout` does not exist on macOS, hence the limit is enforced here.
    """
    junit = tmp_path / "repro.xml"
    cmd = [
        sys.executable, "-B", "-m", "pytest",
        "tests/test_dashboard.py::test_mini_trend_chart_set_points_no_crash",
        "tests/test_failed_file_markers.py",
        "-p", "no:cacheprovider",
        "-o", "faulthandler_timeout=60",   # on a regression, the dump names the test
        f"--junitxml={junit}",
    ]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO / "src"))
    try:
        done = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                              text=True, timeout=120)
    except subprocess.TimeoutExpired as expired:
        out = expired.stdout or b""
        out = out.decode(errors="replace") if isinstance(out, bytes) else out
        pytest.fail("the two files HUNG when run in ONE process (120 s limit). "
                    "The faulthandler dump names the test:\n" + out[-4000:])

    assert junit.exists(), f"pytest wrote no junit report:\n{done.stdout[-4000:]}"
    # Our own subprocess's report, written to tmp_path -- not untrusted input.
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    assert suite is not None, "junit report has no <testsuite>"
    tests = int(suite.get("tests", 0))
    failures = int(suite.get("failures", 0))
    errors = int(suite.get("errors", 0))
    assert tests > 0, "the reproduction collected nothing"
    assert (failures, errors) == (0, 0), (
        f"{failures} failures / {errors} errors in the reproduction:\n{done.stdout[-4000:]}"
    )
