"""
Core processing modules for laser trim analysis.

This package contains the main data processing engine and configuration
management for the Laser Trim AI System.
"""

from core.data_processor import (
    DataProcessor,
    SystemType,
    DataExtraction,
    UnitProperties,
    SigmaResults
)

from core.config import (
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

# Create adapter alias for GUI compatibility
LaserTrimDataProcessor = DataProcessor

__all__ = [
    'DataProcessor',
    'LaserTrimDataProcessor',
    'SystemType',
    'DataExtraction',
    'UnitProperties',
    'SigmaResults',
    'Config',
    'ConfigManager',
    'SystemConfig',
    'ProcessingConfig',
    'SystemAConfig',
    'SystemBConfig',
    'CalibrationConfig',
    'OutputConfig',
    'create_configured_processor'
]

# Version info
__version__ = '1.0.0'
__author__ = 'QA Team'