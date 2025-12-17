"""
Main window bridge for Laser Trim Analyzer GUI
This provides backward compatibility while using CTkMainWindow
"""

import logging
from typing import Optional
from laser_trim_analyzer.core.config import Config, get_config


class MainWindow:
    """Bridge class that delegates to CTkMainWindow"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the main window bridge"""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        self.ctk_window = None
    
    def run(self):
        """Run the application"""
        # Import and create CTkMainWindow
        from laser_trim_analyzer.gui.ctk_main_window import CTkMainWindow
        
        # Create the actual window
        self.ctk_window = CTkMainWindow(self.config)
        
        # Set up close handler
        self.ctk_window.protocol("WM_DELETE_WINDOW", self.ctk_window.on_closing)
        
        # Run the application
        self.ctk_window.run()


def main():
    """Main entry point for GUI application"""
    config = get_config()
    app = MainWindow(config)
    app.run()


if __name__ == "__main__":
    main()