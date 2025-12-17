# src/__init__.py
"""
Laser Trim Analyzer - Production QA Analysis Platform

A comprehensive potentiometer quality analysis system with ML integration.
"""

__version__ = "3.0.0"
__author__ = "JB"

# Re-export from the main package
from laser_trim_analyzer.config import Config, get_config
from laser_trim_analyzer.core.models import (
    AnalysisResult,
    TrackData,
    FileMetadata,
    AnalysisStatus,
)

__all__ = [
    "Config",
    "get_config",
    "AnalysisResult",
    "TrackData",
    "FileMetadata",
    "AnalysisStatus",
]
