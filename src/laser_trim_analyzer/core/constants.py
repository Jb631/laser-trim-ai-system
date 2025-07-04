# src/laser_trim_analyzer/core/constants.py
"""
Application constants for the Laser Trim Analyzer.

These are values that don't change and aren't configurable.
"""

from enum import Enum
from typing import Final


# Application info
APP_NAME: Final[str] = "Laser Trim Analyzer"
APP_AUTHOR: Final[str] = "QA Team"
APP_ORGANIZATION: Final[str] = "Potentiometer Manufacturing Co."

# File patterns
EXCEL_EXTENSIONS: Final[tuple[str, ...]] = (".xlsx", ".xls")
BACKUP_PATTERN: Final[str] = "*_backup_*"
TEMP_FILE_PREFIX: Final[str] = "~$"

# Default output folder name
DEFAULT_OUTPUT_FOLDER: Final[str] = "LaserTrimResults"

# System identifiers
class SystemIdentifier(str, Enum):
    """System identification patterns."""
    SYSTEM_A_PATTERN = "SEC1 TRK"
    SYSTEM_B_PATTERN_1 = "test"
    SYSTEM_B_PATTERN_2 = "Lin Error"

# Track identifiers for System A
class TrackID(str, Enum):
    """Track identifiers."""
    TRACK_1 = "TRK1"
    TRACK_2 = "TRK2"
    DEFAULT = "default"

# Sheet name patterns
SYSTEM_A_SHEET_PATTERNS = {
    "untrimmed": [
        "{track} 0",
        "SEC1 {track} 0",
        "{track}_0"
    ],
    "trimmed": [
        "{track} TRM",
        "SEC1 {track} TRM",
        "{track} TRM1",
        "SEC1 {track} {n} TRM{n}"  # Numbered trim sheets
    ]
}

SYSTEM_B_SHEET_PATTERNS = {
    "untrimmed": "test",
    "trimmed": "Trim {n}",
    "final": "Lin Error"
}

# Column mappings (0-based for Python)
SYSTEM_A_COLUMNS = {
    "position": 7,      # Column H
    "error": 6,         # Column G
    "measured_volts": 3,  # Column D
    "theory_volts": 5,   # Column F
    "upper_limit": 8,    # Column I
    "lower_limit": 9     # Column J
}

SYSTEM_B_COLUMNS = {
    "measured_volts": 0,  # Column A - measured voltage
    "index": 1,          # Column B - index
    "theory_volts": 2,   # Column C - theoretical voltage
    "error": 3,          # Column D - error
    "position": 4,       # Column E - position
    "upper_limit": 5,    # Column F - upper limit
    "lower_limit": 6     # Column G - lower limit
}

# Cell references for unit properties
SYSTEM_A_CELLS = {
    "unit_length": "B26",
    "untrimmed_resistance": "B10",
    "trimmed_resistance": "B10"  # Same cell, different sheet
}

SYSTEM_B_CELLS = {
    "unit_length": "K1",
    "untrimmed_resistance": "R1",
    "trimmed_resistance": "R1"
}

# Model prefixes and patterns
MODEL_PATTERNS = {
    "8340": r"^8340(-\d+)?",
    "8555": r"^8555",
    "6845": r"^6845",
    "7800": r"^78\d{2}"
}

# Special model handling
SPECIAL_MODELS = {
    "8340-1": {
        "fixed_sigma_threshold": 0.4,
        "requires_calibration": True
    }
}

# Analysis constants - EXACT LOCKHEED MARTIN MATLAB SPECIFICATIONS
DEFAULT_SIGMA_SCALING_FACTOR: Final[float] = 24.0
MATLAB_GRADIENT_STEP: Final[int] = 3                    # z_step in calc_gradient.m
MATLAB_POSITION_MIN: Final[float] = 0.06                # ROI extraction: 0.06" start
MATLAB_POSITION_MAX: Final[float] = 0.54                # ROI extraction: 0.54" end  
REFERENCE_SIGMA_GRADIENT: Final[float] = 0.023
DEFAULT_NUM_ZONES: Final[int] = 5
MIN_ZONES: Final[int] = 1
MAX_ZONES: Final[int] = 20
END_POINT_FILTER_COUNT: Final[int] = 5  # Number of points to remove from each end

# Filtering parameters - LOCKHEED MARTIN COMPLIANCE MODE
# Based on original LM MATLAB specifications from read_laser_data_v12.m
# Original algorithm: my_filtfiltfd2.m recursive filter
# Formula: output[i] = output[i-1] + (fc/fs) * (input[i] - output[i-1])
FILTER_SAMPLING_FREQUENCY: Final[int] = 100
FILTER_CUTOFF_FREQUENCY_LM_ORIGINAL: Final[int] = 80   # Original LM spec from MATLAB
FILTER_CUTOFF_FREQUENCY_CORRECTED: Final[int] = 40    # Technically correct (respects Nyquist)
FILTER_CUTOFF_FREQUENCY: Final[int] = 40  # Current default (use LM_ORIGINAL if needed)

# Risk thresholds
HIGH_RISK_THRESHOLD: Final[float] = 0.7
MEDIUM_RISK_THRESHOLD: Final[float] = 0.3

# Plot settings
PLOT_FIGURE_SIZE: Final[tuple[int, int]] = (15, 10)
PLOT_DPI: Final[int] = 150
PLOT_GRID_ALPHA: Final[float] = 0.3
PLOT_COLORS = {
    "untrimmed": "blue",
    "trimmed": "green",
    "spec_limit": "red",
    "filtered": "--"
}

# Report filenames
HTML_REPORT_FILENAME: Final[str] = "analysis_report.html"
EXCEL_REPORT_FILENAME: Final[str] = "summary.xlsx"
QA_DASHBOARD_PREFIX: Final[str] = "QA_Dashboard_"

# Database constants
DB_VERSION: Final[int] = 2
DB_TIMEOUT: Final[float] = 30.0
BATCH_INSERT_SIZE: Final[int] = 1000

# GUI constants
MIN_WINDOW_WIDTH: Final[int] = 900
MIN_WINDOW_HEIGHT: Final[int] = 600
PROGRESS_UPDATE_INTERVAL: Final[int] = 100  # milliseconds

# Error messages
ERROR_MESSAGES = {
    "NO_FILES": "No Excel files found in the selected directory",
    "INVALID_SYSTEM": "Could not detect system type for file: {filename}",
    "NO_DATA": "No valid data found in file: {filename}",
    "CALCULATION_ERROR": "Error calculating {parameter}: {error}",
    "DB_CONNECTION": "Could not connect to database: {error}",
    "FILE_PERMISSION": "Permission denied accessing file: {filename}"
}

# Success messages
SUCCESS_MESSAGES = {
    "ANALYSIS_COMPLETE": "Analysis completed successfully",
    "DB_SAVED": "Results saved to database",
    "REPORT_GENERATED": "Report generated: {filename}",
    "CALIBRATION_DONE": "Calibration completed with factor: {factor:.2f}"
}