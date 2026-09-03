"""
Laser Trim Analyzer v3 - Entry Point

Run with: python -m laser_trim_analyzer
"""

import sys
import os
import logging
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

# Setup logging — console + persistent log file
# Log file lives in data/ next to the database so it's easy to find.
# Anchored to the app directory, not the cwd: the launchers cd here first so
# this is the same folder in production, but a bare Path("data") also made
# `import laser_trim_analyzer.__main__` scribble a data/ dir into whatever
# directory the importer happened to be sitting in (the test suite, notably —
# test_spec3a_shell.py:232 and :250 import it).
log_dir = get_app_directory() / "data"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "laser_trim.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # 5 MB per file, keep last 3 rotations (up to 20 MB total)
        RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3,
            encoding="utf-8",
        ),
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Entry point. Default = V5 LaserTrimApp; --v6 = V6App (Spec 3a+)."""
    use_v6 = "--v6" in sys.argv

    # Environment self-check (work incident 2026-07-10: a different pydantic
    # at work silently changed validation behavior and killed a full day of
    # processing). Two seconds at launch; failures land in the log in plain
    # language BEFORE any file is touched.
    try:
        import pydantic, numpy, pandas, sqlalchemy, matplotlib, customtkinter
        from laser_trim_analyzer.core.models import TrackData, AnalysisStatus
        # The coercion hook rightly WARNS when it drops a NaN (2026-08-31, the
        # linearity-magnitude fix made that silencer loud). This probe feeds it
        # a deliberate NaN, so mute the models logger for just this line —
        # otherwise every launch opens with an alarming warning the self-check
        # itself caused, and real drop warnings lose their signal value.
        _models_logger = logging.getLogger("laser_trim_analyzer.core.models")
        _models_logger.disabled = True
        try:
            _t = TrackData(track_id="_env", travel_length=1.0, linearity_spec=0.01,
                           status=AnalysisStatus.PASS, linearity_error=float("nan"))
        finally:
            _models_logger.disabled = False
        assert _t.linearity_error is None, "NaN coercion inactive"
        logging.getLogger(__name__).info(
            "Environment OK — pydantic %s, numpy %s, pandas %s, sqlalchemy %s, "
            "matplotlib %s, customtkinter %s",
            pydantic.VERSION, numpy.__version__, pandas.__version__,
            sqlalchemy.__version__, matplotlib.__version__,
            getattr(customtkinter, "__version__", "?"))
    except Exception:
        logging.getLogger(__name__).critical(
            "ENVIRONMENT SELF-CHECK FAILED — library versions on this machine "
            "differ from the tested set. Reinstall with: pip install -r "
            "requirements-pinned.txt  (delete .venv and relaunch run_v6.bat "
            "to rebuild it pinned).", exc_info=True)
    logger.info(f"Starting Laser Trim Analyzer (UI: {'V6' if use_v6 else 'V5'})...")
    try:
        from laser_trim_analyzer.config import get_config
        config = get_config()
        logger.info(f"Config loaded - Database: {config.database.path}")
        config.database.ensure_directory()
        if use_v6:
            from laser_trim_analyzer.gui.v6.app import V6App
            app = V6App(config)
        else:
            from laser_trim_analyzer.app import LaserTrimApp
            app = LaserTrimApp(config)
        app.run()
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -e .")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
