"""Run Laser Trim Analyzer in Development Mode"""
import os
import sys

# Set development environment
os.environ["LTA_ENV"] = "development"

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Run the application
from laser_trim_analyzer.gui.main_window import MainWindow
from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.core.utils import setup_logging
import logging

# Initialize configuration and logging
config = get_config()
logger = setup_logging(config.log_directory, logging.INFO)
logger.info("Laser Trim Analyzer - Starting GUI mode (DEVELOPMENT)...")

# Create and run the GUI application
app = MainWindow(config)
app.run()