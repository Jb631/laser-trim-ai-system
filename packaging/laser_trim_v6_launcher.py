"""The packaged app's entry script -- what `LaserTrimAnalyzer.exe` runs.

PyInstaller builds the .exe from this file (`packaging/laser_trim_v6.spec`). It is the packaged
twin of `run_v6.bat`'s `python -m src --v6`: the same `main()`, with the V6 UI forced. Any other
argument passes through -- `LaserTrimAnalyzer.exe --check` is the self-check the build runs.

Three things happen before the app's own code is even imported, in this order:

1. `multiprocessing.freeze_support()` -- FIRST. In a packaged build a spawned child process is a
   second run of this same .exe. freeze_support() recognises that run, becomes the worker it was
   meant to be, and exits. Without it, a stray spawn would start the whole app again: a second
   window, which spawns again. (The packaged build starts no worker processes of its own --
   `core/ingest_worker.worker_count()` -- so this is the guard behind that rule.)

2. A windowed build has no console: `sys.stdout` and `sys.stderr` are None there, and anything
   that writes to them without asking raises. They are pointed at the null device.

3. `--v6` is added when it is not already there.

Nothing here is specific to Windows, so the tests run it as it is (tests/test_packaging_files.py,
tests/test_packaged_check.py).
"""
import multiprocessing
import os
import sys


def run() -> None:
    multiprocessing.freeze_support()
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))
    if "--v6" not in sys.argv[1:]:
        sys.argv.insert(1, "--v6")
    # Imported only now: this import sets up the app's logging (and, outside --check, creates
    # data/ beside the .exe) -- none of which a spawned child should ever do.
    from laser_trim_analyzer.__main__ import main
    main()


if __name__ == "__main__":
    run()
