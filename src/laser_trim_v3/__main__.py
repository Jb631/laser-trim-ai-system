"""
Laser Trim Analyzer v3 - Entry Point

Run with: python -m laser_trim_v3
"""

import sys
import logging
from pathlib import Path

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
        from laser_trim_v3.app import LaserTrimApp
        from laser_trim_v3.config import get_config

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
