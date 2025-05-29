"""
Core processing modules for laser trim analysis.

This package contains the main data processing engine and configuration
management for the Laser Trim AI System.
"""

# Import main classes from data_processor.py
from .data_processor import (
    DataProcessor,
    SystemType,
    DataExtraction,
    UnitProperties,
    SigmaResults
)

# Import configuration classes
from .config import (
    Config,
    ConfigManager,
    SystemConfig,
    ProcessingConfig,
    SystemAConfig,
    SystemBConfig,
    CalibrationConfig,
    OutputConfig,
    create_configured_processor
)

# Import adapter class for GUI compatibility
try:
    from .data_processor_adapter import LaserTrimDataProcessor
except ImportError:
    # If adapter doesn't exist yet, create a simple alias
    LaserTrimDataProcessor = DataProcessor

# Import data loader if it exists
try:
    from .data_loader import DataLoader
except ImportError:
    DataLoader = None

# Version info
__version__ = '1.0.0'
__author__ = 'QA Team'

# Define what should be imported with "from core import *"
__all__ = [
    # Data processing classes
    'DataProcessor',
    'LaserTrimDataProcessor',
    'SystemType',
    'DataExtraction',
    'UnitProperties',
    'SigmaResults',

    # Configuration classes
    'Config',
    'ConfigManager',
    'SystemConfig',
    'ProcessingConfig',
    'SystemAConfig',
    'SystemBConfig',
    'CalibrationConfig',
    'OutputConfig',

    # Factory functions
    'create_configured_processor',

    # Data loader (if available)
    'DataLoader',

    # Version info
    '__version__',
    '__author__'
]


# Module-level docstring for help()
def get_module_info():
    """
    Get information about the core module.

    Returns:
        dict: Module information including version and available classes
    """
    return {
        'version': __version__,
        'author': __author__,
        'description': 'Core data processing engine for Laser Trim AI System',
        'main_classes': {
            'DataProcessor': 'Main data processing engine',
            'LaserTrimDataProcessor': 'GUI-compatible data processor',
            'SystemType': 'Enumeration for system types (A/B)',
            'Config': 'System configuration management',
            'ConfigManager': 'Configuration file handler'
        },
        'data_classes': {
            'DataExtraction': 'Container for extracted measurement data',
            'UnitProperties': 'Container for unit-specific properties',
            'SigmaResults': 'Container for sigma calculation results'
        }
    }


# Initialization message (optional, can be removed in production)
import logging

logger = logging.getLogger(__name__)
logger.debug(f"Core module initialized (version {__version__})")