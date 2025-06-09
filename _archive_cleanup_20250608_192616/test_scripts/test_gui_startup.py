#!/usr/bin/env python3
"""
Test GUI startup and page loading

This script tests the GUI startup process and identifies any initialization errors.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gui_startup_test.log')
    ]
)

logger = logging.getLogger(__name__)

def test_gui_startup():
    """Test GUI startup and report any errors."""
    logger.info("Starting GUI test...")
    
    try:
        # Import the main window
        from laser_trim_analyzer.gui.ctk_main_window import CTkMainWindow
        logger.info("Successfully imported CTkMainWindow")
        
        # Try to create the window
        logger.info("Creating main window...")
        app = CTkMainWindow()
        logger.info("Main window created successfully")
        
        # Check which pages loaded successfully
        logger.info("\nPage loading status:")
        for page_name, page in app.pages.items():
            page_type = type(page).__name__
            logger.info(f"  {page_name}: {page_type}")
            
        # Try to show the batch page
        logger.info("\nAttempting to show batch processing page...")
        app._show_page('batch')
        logger.info("Batch page shown successfully")
        
        # Run briefly then exit
        app.after(2000, app.quit)  # Quit after 2 seconds
        app.mainloop()
        
        logger.info("GUI test completed successfully")
        
    except Exception as e:
        logger.error(f"GUI startup failed: {e}", exc_info=True)
        return False
        
    return True

if __name__ == "__main__":
    success = test_gui_startup()
    sys.exit(0 if success else 1)