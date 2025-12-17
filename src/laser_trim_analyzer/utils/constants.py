"""
Constants for Laser Trim Analyzer v3.

Consolidated from v2's scattered constants.
"""

from typing import Final, Dict

# Application info
APP_NAME: Final[str] = "Laser Trim Analyzer"
APP_VERSION: Final[str] = "3.0.0"

# File patterns
EXCEL_EXTENSIONS: Final[tuple] = (".xlsx", ".xls")

# Column mappings for System A (0-based)
SYSTEM_A_COLUMNS: Final[Dict[str, int]] = {
    "position": 7,       # Column H
    "error": 6,          # Column G
    "measured_volts": 3, # Column D
    "theory_volts": 5,   # Column F
    "upper_limit": 8,    # Column I
    "lower_limit": 9     # Column J
}

# Column mappings for System B (0-based)
SYSTEM_B_COLUMNS: Final[Dict[str, int]] = {
    "measured_volts": 0, # Column A
    "index": 1,          # Column B
    "theory_volts": 2,   # Column C
    "error": 3,          # Column D
    "position": 4,       # Column E
    "upper_limit": 5,    # Column F
    "lower_limit": 6     # Column G
}

# Cell references for metadata
SYSTEM_A_CELLS: Final[Dict[str, str]] = {
    "unit_length": "B26",
    "untrimmed_resistance": "B10",
    "trimmed_resistance": "B10"
}

SYSTEM_B_CELLS: Final[Dict[str, str]] = {
    "unit_length": "K1",
    "untrimmed_resistance": "R1",
    "trimmed_resistance": "R1"
}

# Sheet name patterns
SYSTEM_A_UNTRIMMED_PATTERNS = ["{track} 0", "SEC1 {track} 0", "{track}_0"]
SYSTEM_A_TRIMMED_PATTERNS = ["{track} TRM", "SEC1 {track} TRM", "{track} TRM1"]
SYSTEM_B_UNTRIMMED_SHEET = "test"
SYSTEM_B_TRIMMED_PATTERN = "Trim {n}"

# System detection patterns
SYSTEM_A_IDENTIFIER = "SEC1 TRK"
SYSTEM_B_IDENTIFIERS = ["test", "Lin Error"]

# Analysis parameters
DEFAULT_SIGMA_SCALING_FACTOR: Final[float] = 24.0
MATLAB_GRADIENT_STEP: Final[int] = 1
END_POINT_FILTER_COUNT: Final[int] = 5
BUTTERWORTH_CUTOFF: Final[float] = 0.1
BUTTERWORTH_ORDER: Final[int] = 2

# Threshold formula coefficients
# threshold = (linearity_spec / scaling_factor) * travel_factor
# where travel_factor adjusts for travel length
DEFAULT_TRAVEL_FACTOR: Final[float] = 1.0

# Risk thresholds
HIGH_RISK_THRESHOLD: Final[float] = 0.7
MEDIUM_RISK_THRESHOLD: Final[float] = 0.4
