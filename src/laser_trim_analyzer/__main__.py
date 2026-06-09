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
# Log file lives in data/ next to the database so it's easy to find
log_dir = Path("data")
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
