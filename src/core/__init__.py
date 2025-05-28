# Add content to src/core/__init__.py
@"
"""
Core processing modules for laser trim analysis.
"""

from .data_loader import DataLoader
from .data_processor import DataExtractor
from .sigma_calculator import SigmaCalculator
from .system_detector import SystemDetector
from .filter_utils import apply_matlab_filter

__all__ = [
    'DataLoader',
    'DataExtractor',
    'SigmaCalculator',
    'SystemDetector',
    'apply_matlab_filter'
]
"@ | Out-File -FilePath "src/core/__init__.py" -Encoding UTF8