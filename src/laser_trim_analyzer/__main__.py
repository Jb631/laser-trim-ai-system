"""
Laser Trim Analyzer v3 - Entry Point

Run with: python -m laser_trim_analyzer
"""

import sys
import os
import logging
from pathlib import Path

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

# Setup logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for v3."""
    logger.info("Starting Laser Trim Analyzer v3...")

    try:
        # Import here to avoid circular imports
        from laser_trim_analyzer.app import LaserTrimApp
        from laser_trim_analyzer.config import get_config

        # Load configuration
        config = get_config()
        logger.info(f"Config loaded - Database: {config.database.path}")

        # Ensure database directory exists
        config.database.ensure_directory()

        # Create and run the application
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
