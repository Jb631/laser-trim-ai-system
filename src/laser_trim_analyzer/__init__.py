"""
Laser Trim Analyzer - Professional Edition

A comprehensive analysis tool for laser trim data with advanced validation,
machine learning capabilities, and database integration.
"""

__version__ = "2.0.0"
__author__ = "Laser Trim Analysis Team"
__email__ = "support@lasertrimanalyzer.com"

from .core.config import get_config, Config
from .core.constants import APP_NAME
from .core.processor import LaserTrimProcessor

__all__ = [
    'get_config',
    'Config',
    'APP_NAME',
    'LaserTrimProcessor',
] 