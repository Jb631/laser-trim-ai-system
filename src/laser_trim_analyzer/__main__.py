"""
Entry point for running the laser_trim_analyzer package as a module.

This allows running: python -m laser_trim_analyzer
"""

import sys
from pathlib import Path

# Add the src directory to the path so imports work
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

if __name__ == "__main__":
    # Import and run the main function from the src package
    try:
        from laser_trim_analyzer.gui.main_window import MainWindow
        from laser_trim_analyzer.core.config import get_config
        from laser_trim_analyzer.core.utils import setup_logging
        import logging
        
        # Initialize configuration and logging
        config = get_config()
        logger = setup_logging(config.log_directory, logging.INFO)
        logger.info("Laser Trim Analyzer - Starting GUI mode...")
        
        # Create and run the GUI application
        app = MainWindow(config)
        app.run()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 