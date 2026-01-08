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
SYSTEM_A_UNTRIMMED_PATTERNS: Final[list] = ["{track} 0", "SEC1 {track} 0", "{track}_0"]
SYSTEM_A_TRIMMED_PATTERNS: Final[list] = ["{track} TRM", "SEC1 {track} TRM", "{track} TRM1"]
SYSTEM_B_UNTRIMMED_SHEET: Final[str] = "test"
SYSTEM_B_TRIMMED_PATTERN: Final[str] = "Trim {n}"

# System detection patterns
SYSTEM_A_IDENTIFIER: Final[str] = "SEC1 TRK"
SYSTEM_B_IDENTIFIERS: Final[list] = ["test", "Lin Error"]

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

# =============================================================================
# Final Test File Constants
# For post-assembly testing (separate from laser trim)
# =============================================================================

# Final Test Format 1 - Standard format (e.g., 1081313-sn108_3-16-2011_12-17 PM.xls)
# Sheet: Sheet1
# Column mapping verified against model 8340-1 files
FINAL_TEST_FORMAT1_COLUMNS: Final[Dict[str, int]] = {
    "measured": 0,           # Column A - Measured Volts (actual output)
    "index": 1,              # Column B - Index/sample number
    "theory": 2,             # Column C - Theory Volts (expected/ideal value)
    "error": 3,              # Column D - Voltage Error (pre-calculated)
    "electrical_angle": 4,   # Column E - Electrical Angle (X-axis for linearity)
                             #   Linear pots (8340-1): 0 to ~0.61 inches
                             #   Rotary pots (2475): -170° to +170°
    "upper_limit": 6,        # Column G - Upper Spec Limit
    "lower_limit": 7,        # Column H - Lower Spec Limit
}

# Final Test Format 2 - Rout_ files (e.g., Rout_1091701_sn1695a_vo.xls)
# Sheet: Data
FINAL_TEST_FORMAT2_COLUMNS: Final[Dict[str, int]] = {
    "measured": 0,         # Column A - Measured value
    "position": 1,         # Column B - Position
    "index": 2,            # Column C - Index
}

# Final Test metadata cell locations (Format 1)
FINAL_TEST_FORMAT1_METADATA: Final[Dict[str, str]] = {
    "model_cell": "L1",       # Model/shop number location
    "datetime_cell": "N1",    # Test datetime location
    "data_start_row": 2,      # First row of actual data (0-indexed)
}

# Final Test Data Table sheet - test results summary
FINAL_TEST_DATA_TABLE_ROWS: Final[Dict[str, int]] = {
    "resistance": 13,      # Row with resistance test results
    "linearity": 14,       # Row with linearity test results
    "electrical_angle": 15, # Row with electrical angle test results
    "hysteresis": 16,      # Row with hysteresis test results
    "phasing": 17,         # Row with phasing test results
}
FINAL_TEST_DATA_TABLE_COLUMNS: Final[Dict[str, int]] = {
    "test_name": 1,        # Column B - Test name
    "spec": 3,             # Column D - Specification
    "measured": 4,         # Column E - Measured value
    "result": 5,           # Column F - PASSED/FAILED
}

# Detection patterns for Final Test files
FINAL_TEST_SHEET_PATTERNS: Final[list] = [
    "Sheet1",              # Standard format primary sheet
    "Data Table",          # Test results summary
    "TEST RESULTS DATA",   # Alternative identifier
]
FINAL_TEST_ROUT_PREFIX: Final[str] = "Rout_"  # Prefix for Format 2 files

# Trim file detection (things that indicate it's NOT a final test)
TRIM_FILE_INDICATORS: Final[list] = [
    "SEC1 TRK",           # System A multi-track
    "TRK1",               # Track identifiers
    "TRK2",
    "Lin Error",          # System B linearity error sheet
    "_Trimmed",           # Trimmed file suffix
    "_Untrimmed",         # Untrimmed file suffix
]

# Matching parameters for Final Test to Trim linking
FINAL_TEST_MAX_DAYS_FROM_TRIM: Final[int] = 60  # Maximum days between trim and final test
