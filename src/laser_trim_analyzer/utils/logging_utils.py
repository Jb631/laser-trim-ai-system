"""
Logging utilities for Laser Trim Analyzer.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
        name: str = "laser_trim_analyzer",
        log_dir: Optional[Path] = None,
        level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console_output: Enable console output
        file_output: Enable file output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_output and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log filename with date
        log_filename = f"laser_trim_{datetime.now().strftime('%Y%m%d')}.log"
        log_path = log_dir / log_filename

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_exception(logger: logging.Logger, exception: Exception, message: str = ""):
    """
    Log an exception with traceback.

    Args:
        logger: Logger instance
        exception: Exception to log
        message: Additional context message
    """
    if message:
        logger.error(f"{message}: {str(exception)}", exc_info=True)
    else:
        logger.error(f"Exception occurred: {str(exception)}", exc_info=True)