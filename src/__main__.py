# src/laser_trim_analyzer/__main__.py
"""
Main entry point for the Laser Trim Analyzer application.

This module handles both GUI and CLI modes.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
import asyncio
import signal
import os
import traceback
import atexit

import click
from rich.console import Console
from rich.logging import RichHandler

from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.core.utils import setup_logging
from laser_trim_analyzer.core.constants import APP_NAME, APP_AUTHOR

# Set up rich console for better output
console = Console()

# Global shutdown flag
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    print(f"\nReceived signal {signum}, initiating graceful shutdown...")
    _shutdown_requested = True
    sys.exit(0)

def cleanup_resources():
    """Cleanup resources on exit."""
    try:
        # Force close any remaining matplotlib figures
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass
    
    try:
        # Stop any asyncio loops
        if asyncio._get_running_loop():
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.stop()
    except:
        pass
    
    print("Application cleanup completed.")

@click.command()
@click.option('--debug', is_flag=True, help='Enable debug mode with verbose logging')
@click.version_option(version='2.0.0', prog_name='Laser Trim Analyzer')
def main(debug: bool):
    """Main entry point for the Laser Trim Analyzer."""
    global _shutdown_requested
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function
    atexit.register(cleanup_resources)
    
    try:
        # Initialize configuration and logging
        config = get_config()
        
        # Set up logging level
        log_level = logging.DEBUG if debug else logging.INFO
        
        # Override debug setting if flag provided
        if debug:
            config.debug = True
            console.print("[yellow]Debug mode enabled[/yellow]")
        
        # Setup logging with output directory and level
        logger = setup_logging(config.log_directory, log_level)
        logger.info("Laser Trim Analyzer - Starting GUI mode...")
        
        if debug:
            logger.debug("Debug logging enabled")
            logger.debug(f"Configuration: {config}")
        
        # Check for shutdown before starting GUI
        if _shutdown_requested:
            return
            
        # Import and start GUI (heavy imports after signal handlers are set)
        from laser_trim_analyzer.gui.main_window import MainWindow
        
        # Check for shutdown after imports
        if _shutdown_requested:
            return
            
        # Create and run the GUI application
        app = MainWindow(config)
        
        # Run with proper exception handling
        try:
            app.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
        except Exception as e:
            logger.error(f"Application error: {e}")
            logger.error(traceback.format_exc())
        finally:
            # Ensure cleanup
            cleanup_resources()
            
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Final cleanup
        cleanup_resources()


if __name__ == "__main__":
    main()
