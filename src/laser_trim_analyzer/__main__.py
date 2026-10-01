"""
Laser Trim Analyzer v3 - Entry Point

Run with: python -m laser_trim_analyzer
"""

import sys
import os
import logging
import multiprocessing
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path

from laser_trim_analyzer.config import get_app_directory

# Suppress scikit-learn parallel warnings (compatibility issue with joblib)
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

# Fix Tcl/Tk library path for uv-installed Python on macOS
# This must happen before any tkinter imports
if sys.platform == "darwin" and "TCL_LIBRARY" not in os.environ:
    python_base = Path(sys.executable).resolve().parent.parent
    tcl_path = python_base / "lib" / "tcl8.6"
    tk_path = python_base / "lib" / "tk8.6"
    if tcl_path.exists():
        os.environ["TCL_LIBRARY"] = str(tcl_path)
    if tk_path.exists():
        os.environ["TK_LIBRARY"] = str(tk_path)


# `--check` (laser_trim_analyzer.selfcheck): the app checks itself and exits. Known here, at
# import, because it decides where this module may log -- see _log_handlers.
_CHECK_ONLY = "--check" in sys.argv[1:]


def _log_handlers(check_only: bool = False) -> list:
    """Where the app logs: the console when there is one, and the persistent log file.

    The log file lives in data/ next to the database so it's easy to find.
    Anchored to the app directory, not the cwd: the launchers cd here first so
    this is the same folder in production, but a bare Path("data") also made
    `import laser_trim_analyzer.__main__` scribble a data/ dir into whatever
    directory the importer happened to be sitting in (the test suite, notably —
    test_spec3a_shell.py:232 and :250 import it).

    A WINDOWED packaged build has no console: PyInstaller's windowed bootloader
    leaves sys.stdout and sys.stderr as None. A StreamHandler built on None
    falls back to sys.stderr -- None as well -- so every record would raise
    inside logging and be dropped the slow way. No console, no console handler:
    the log file is then the whole record.

    `--check` logs nowhere of its own (`check_only`): the build runs
    `LaserTrimAnalyzer.exe --check` inside the very folder that is then handed
    over, and that folder must never contain a `data` folder -- so no log file
    and no data/ are made. The self-check prints what the loggers said itself.
    """
    if check_only:
        return [logging.NullHandler()]
    handlers = []
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    log_dir = get_app_directory() / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    # 5 MB per file, keep last 3 rotations (up to 20 MB total)
    handlers.append(RotatingFileHandler(
        log_dir / "laser_trim.log", maxBytes=5_000_000, backupCount=3,
        encoding="utf-8",
    ))
    return handlers


# Setup logging — console + persistent log file
logging.basicConfig(
    level=logging.WARNING if _CHECK_ONLY else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=_log_handlers(_CHECK_ONLY),
)

logger = logging.getLogger(__name__)


def main():
    """Entry point. Default = V5 LaserTrimApp; --v6 = V6App (Spec 3a+)."""
    # FIRST, before anything is read, opened or built. In a packaged build
    # (PyInstaller) a spawned child process is a second run of this same
    # executable; freeze_support() recognises that run, becomes the worker it
    # was meant to be, and exits -- so a stray spawn can never open a second
    # window. From source, and in the app's own first run, it does nothing.
    # (The packaged launcher calls it too, before this module is even imported;
    # here it guards the entry point whichever script reaches it.)
    multiprocessing.freeze_support()
    from laser_trim_analyzer import selfcheck
    if "--check" in sys.argv[1:]:
        # No window, no app database, nothing written beside the app but the
        # result: every step a packaged build can be incomplete in, by name.
        sys.exit(selfcheck.main(sys.argv[1:]))
    use_v6 = "--v6" in sys.argv
    packaged = getattr(sys, "frozen", False)

    # Environment self-check (work incident 2026-07-10: a different pydantic
    # at work silently changed validation behavior and killed a full day of
    # processing). Two seconds at launch; failures land in the log in plain
    # language BEFORE any file is touched. The check itself is
    # selfcheck.environment() -- the one `--check` runs first.
    try:
        logging.getLogger(__name__).info("Environment OK — %s", selfcheck.environment())
    except Exception:
        logging.getLogger(__name__).critical(
            "ENVIRONMENT SELF-CHECK FAILED — "
            + ("this packaged build is incomplete, or was built from library "
               "versions other than the tested set. It cannot be repaired here: "
               "it needs a new build (scripts\\build_exe.ps1, whose --check "
               "names what is missing)." if packaged else
               "library versions on this machine "
               "differ from the tested set. Reinstall with: pip install -r "
               "requirements-pinned.txt  (delete .venv and relaunch run_v6.bat "
               "to rebuild it pinned)."), exc_info=True)
    logger.info(f"Starting Laser Trim Analyzer (UI: {'V6' if use_v6 else 'V5'})...")
    # Which build this is, once, when it is a packaged one (build_info.txt
    # beside the .exe, written by scripts/build_exe.ps1). From source there is
    # no stamp and nothing is said.
    stamp = selfcheck.build_line(selfcheck.read_build_info())
    if stamp:
        logger.info(stamp)
    try:
        from laser_trim_analyzer.config import get_config
        from laser_trim_analyzer.database.manager import allow_default_database
        config = get_config()
        logger.info(f"Config loaded - Database: {config.database.path}")
        config.database.ensure_directory()
        # The app -- and only the app -- opens the configured database without
        # naming it: its pages and the processor reach it through
        # get_database(). Anything else that tries is refused (manager.py).
        allow_default_database()
        if use_v6:
            from laser_trim_analyzer.gui.v6.app import V6App
            app = V6App(config)
        else:
            from laser_trim_analyzer.app import LaserTrimApp
            app = LaserTrimApp(config)
        app.run()
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("This packaged build is missing a module: it needs a new build "
                     "(run LaserTrimAnalyzer.exe --check to see everything that is missing)."
                     if packaged else
                     "Make sure all dependencies are installed: pip install -e .")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
