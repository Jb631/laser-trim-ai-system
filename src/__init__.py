# src/laser_trim_analyzer/__init__.py
"""
Laser Trim Analyzer v2 - Modern QA Analysis Platform

A comprehensive potentiometer quality analysis system with AI integration.
"""

__version__ = "2.0.0"
__author__ = "JB"

from laser_trim_analyzer.core.config import Config, get_config
from laser_trim_analyzer.core.models import (
    AnalysisResult,
    TrackData,
    FileMetadata,
    SigmaAnalysis,
    LinearityAnalysis,
    ResistanceAnalysis
)

__all__ = [
    "Config",
    "get_config",
    "AnalysisResult",
    "TrackData",
    "FileMetadata",
    "SigmaAnalysis",
    "LinearityAnalysis",
    "ResistanceAnalysis",
]