# src/laser_trim_analyzer/__main__.py
"""
Main entry point for the Laser Trim Analyzer application.

This module handles both GUI and CLI modes.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler

from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.core.constants import APP_NAME, APP_AUTHOR

# Set up rich console for better output
console = Console()


def setup_logging(debug: bool = False) -> None:
    """Set up application logging with rich handler."""
    config = get_config()

    # Create log directory
    config.log_directory.mkdir(parents=True, exist_ok=True)

    # Set up formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler
    log_file = config.log_directory / f"laser_trim_analyzer.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)

    # Console handler with Rich
    console_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=debug
    )
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Reduce noise from some libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Reduce noise from analysis modules during normal operation only
    if not debug:
        logging.getLogger("laser_trim_analyzer.analysis").setLevel(logging.WARNING)
        logging.getLogger("laser_trim_analyzer.gui.widgets").setLevel(logging.WARNING)
        logging.getLogger("laser_trim_analyzer.database").setLevel(logging.WARNING)
    else:
        # In debug mode, ensure analysis modules are debug level
        logging.getLogger("laser_trim_analyzer.analysis").setLevel(logging.DEBUG)
        logging.getLogger("laser_trim_analyzer.core").setLevel(logging.DEBUG)


def run_gui(debug: bool = False) -> None:
    """Run the GUI application."""
    setup_logging(debug)

    try:
        # Import here to avoid circular imports and heavy imports when using CLI
        from laser_trim_analyzer.gui.main_window import MainWindow

        console.print(f"[bold blue]{APP_NAME}[/bold blue] - Starting GUI mode...")

        app = MainWindow()
        if hasattr(app, 'run'):
            app.run()
        else:
            app.root.mainloop()

    except ImportError as e:
        console.print(f"[red]Error: Could not import GUI components: {e}[/red]")
        console.print("Make sure GUI dependencies are installed: pip install customtkinter")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error starting GUI: {e}[/red]")
        logging.exception("GUI startup error")
        sys.exit(1)


def run_cli(args: list) -> None:
    """Run the CLI application."""
    try:
        # Import here to avoid circular imports
        from laser_trim_analyzer.cli import cli

        # Remove the first argument (script name) if present
        if args and args[0].endswith(('.py', '__main__')):
            args = args[1:]

        cli(args)

    except ImportError as e:
        console.print(f"[red]Error: Could not import CLI components: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error in CLI: {e}[/red]")
        logging.exception("CLI error")
        sys.exit(1)


def main() -> None:
    """Main entry point that determines whether to run GUI or CLI."""
    # Check for debug flag early
    debug = '--debug' in sys.argv or '-d' in sys.argv

    # If no arguments or common GUI indicators, run GUI
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and debug):
        run_gui(debug)
    else:
        # Otherwise, run CLI
        run_cli(sys.argv)


if __name__ == "__main__":
    main()
