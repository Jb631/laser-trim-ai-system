"""
System constants based on validated legacy calculations.

This module contains all the constants used throughout the system,
preserving the exact values from your validated legacy code.
"""

from typing import Dict, Any, Final, List

# Application info
APP_NAME: Final[str] = "Laser Trim AI System"
APP_VERSION: Final[str] = "1.0.0"

# System type identifiers
SYSTEM_A: Final[str] = "A"
SYSTEM_B: Final[str] = "B"

# Track identifiers for System A
TRACK_1: Final[str] = "TRK1"
TRACK_2: Final[str] = "TRK2"
SYSTEM_A_TRACKS: Final[List[str]] = [TRACK_1, TRACK_2]

# Sigma calculation parameters (validated from legacy)
FILTER_SAMPLING_FREQUENCY: Final[int] = 100  # Hz
FILTER_CUTOFF_FREQUENCY: Final[int] = 80     # Hz
MATLAB_GRADIENT_STEP: Final[int] = 3         # Step size for gradient calculation
DEFAULT_SIGMA_SCALING_FACTOR: Final[float] = 24.0

# Model-specific thresholds
MODEL_8340_1_THRESHOLD: Final[float] = 0.4  # Fixed threshold for 8340-1

# System A sheet patterns and column mappings
SYSTEM_A_CONFIG: Final[Dict[str, Any]] = {
    "track_patterns": {
        TRACK_1: {
            "untrimmed": "SEC1 TRK1 0",
            "trimmed": "SEC1 TRK1 TRM"
        },
        TRACK_2: {
            "untrimmed": "SEC1 TRK2 0",
            "trimmed": "SEC1 TRK2 TRM"
        }
    },
    # Column indices (0-based for Python)
    "columns": {
        "measured_volts": 3,  # Column D
        "index": 4,           # Column E
        "theory_volts": 5,    # Column F
        "error": 6,           # Column G
        "position": 7,        # Column H
        "upper_limit": 8,     # Column I
        "lower_limit": 9      # Column J
    },
    # Cell references for unit properties
    "cells": {
        "unit_length": "B26",
        "untrimmed_resistance": "B10",
        "trimmed_resistance": "B10"  # Same cell in trimmed sheet
    }
}

# System B sheet patterns and column mappings
SYSTEM_B_CONFIG: Final[Dict[str, Any]] = {
    "sheets": {
        "untrimmed": "test",
        "final": "Lin Error",
        "trim_pattern": "Trim "  # Sheets starting with "Trim "
    },
    # Column indices (0-based for Python)
    "columns": {
        "error": 3,           # Column D
        "upper_limit": 5,     # Column F
        "lower_limit": 6,     # Column G
        "position": 8         # Column I
    },
    # Cell references for unit properties
    "cells": {
        "unit_length": "K1",
        "untrimmed_resistance": "R1",
        "trimmed_resistance": "R1"
    }
}

# Data processing parameters
MIN_DATA_POINTS: Final[int] = 50  # Minimum points for valid analysis
END_POINT_FILTER: Final[int] = 7  # Points to remove from each end

# Quality thresholds
MAX_ACCEPTABLE_SIGMA: Final[float] = 1.0  # Maximum sigma before automatic flag
MIN_POSITION_RANGE: Final[float] = 0.1    # Minimum travel length

# File patterns
SUPPORTED_EXTENSIONS: Final[List[str]] = [".xlsx", ".xls"]
EXCLUDED_PATTERNS: Final[List[str]] = ["~$", "temp", "backup", ".tmp"]

# Risk categories
RISK_CATEGORIES = {
    "LOW": {"min": 0.0, "max": 0.3, "color": "#00FF00"},
    "MEDIUM": {"min": 0.3, "max": 0.7, "color": "#FFA500"},
    "HIGH": {"min": 0.7, "max": 1.0, "color": "#FF0000"}
}

# Reporting constants
EXCEL_DATETIME_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
PLOT_DPI: Final[int] = 150
CHART_STYLE: Final[str] = "seaborn-v0_8"

# Processing modes
PROCESSING_MODES = {
    "SPEED": {
        "parallel": True,
        "generate_plots": False,
        "max_workers": None  # Auto-detect
    },
    "DETAIL": {
        "parallel": False,
        "generate_plots": True,
        "max_workers": 1
    }
}

# Model patterns for identification
MODEL_PATTERNS = {
    "8340_SERIES": r"^8340",
    "8555_SERIES": r"^8555",
    "6845_SERIES": r"^6845",
    "7230_SERIES": r"^7230"
}

# Caching settings
ENABLE_CACHING: Final[bool] = True
CACHE_EXPIRY_HOURS: Final[int] = 24

# Logging settings
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
MAX_LOG_FILE_SIZE: Final[int] = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: Final[int] = 5