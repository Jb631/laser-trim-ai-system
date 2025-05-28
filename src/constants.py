"""
Constants module for Laser Trim AI System

This module contains all constant values used throughout the system.
"""

# System type identifiers
SYSTEM_A = "A"
SYSTEM_B = "B"

# Track identifiers for System A
TRACK_1 = "TRK1"
TRACK_2 = "TRK2"
SYSTEM_A_TRACKS = [TRACK_1, TRACK_2]

# Filter parameters (matching MATLAB implementation)
FILTER_CUTOFF_FREQUENCY = 80  # Hz
FILTER_SAMPLING_FREQUENCY = 100  # Hz

# Gradient calculation parameters
MATLAB_GRADIENT_STEP = 3  # Step size for gradient calculation

# Default values
DEFAULT_SIGMA_THRESHOLD = 0.5
DEFAULT_SIGMA_SCALING_FACTOR = 24.0

# Column mappings for System A (0-based indices for Python/pandas)
SYSTEM_A_COLUMN_MAP = {
    'measured_volts': 3,   # Column D
    'index': 4,            # Column E
    'theory_volts': 5,     # Column F
    'error': 6,            # Column G
    'position': 7,         # Column H
    'upper_limit': 8,      # Column I
    'lower_limit': 9       # Column J
}

# Column mappings for System B (0-based indices)
SYSTEM_B_COLUMN_MAP = {
    'error': 3,            # Column D
    'upper_limit': 5,      # Column F
    'lower_limit': 6,      # Column G
    'position': 8          # Column I
}

# Cell references for unit properties
UNIT_LENGTH_CELLS = {
    SYSTEM_A: "B26",       # Unit angle/length in System A
    SYSTEM_B: "K1"         # Unit angle/length in System B
}

# Cell references for resistance values
RESISTANCE_CELLS = {
    SYSTEM_A: {
        'untrimmed': "B10",
        'trimmed': "B10"   # Same cell but different sheet
    },
    SYSTEM_B: {
        'untrimmed': "R1",
        'trimmed': "R1"    # Same cell but different sheet
    }
}

# Sheet name patterns for System A
SYSTEM_A_SHEET_PATTERNS = {
    'untrimmed': {
        TRACK_1: ["SEC1 TRK1 0", "SEC1TRK10", "TRK1 0"],
        TRACK_2: ["SEC1 TRK2 0", "SEC1TRK20", "TRK2 0"]
    },
    'trimmed': {
        TRACK_1: ["TRM", "TRIM", "TRK1 TRM", "TRK1 TRIM"],
        TRACK_2: ["TRM", "TRIM", "TRK2 TRM", "TRK2 TRIM"]
    }
}

# Sheet names for System B
SYSTEM_B_SHEETS = {
    'untrimmed': "test",
    'final': "Lin Error"
}

# Validation limits
POSITION_MIN = 0.0
POSITION_MAX = 1.0
ERROR_MIN = -0.1
ERROR_MAX = 0.1

# File extensions
VALID_EXTENSIONS = ['.xls', '.xlsx']

# Model patterns
MODEL_8340_PREFIX = "8340"
MODEL_8555_PREFIX = "8555"
MODEL_6845_PREFIX = "6845"

# Risk thresholds
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.3
LOW_RISK_THRESHOLD = 0.1