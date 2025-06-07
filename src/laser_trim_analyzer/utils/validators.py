"""
Validation utilities for laser trim analyzer.

Provides comprehensive validation for files, data, and QA-specific requirements.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import time
import json

from laser_trim_analyzer.core.error_handlers import (
    ErrorCode, ErrorCategory, ErrorSeverity,
    error_handler, handle_errors
)

# Try to import secure logging, fall back to standard logging
try:
    from laser_trim_analyzer.utils.logging_utils import SecureLogger
    logger = SecureLogger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("SecureLogger not available, using standard logging")


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)

    def merge(self, other: 'ValidationResult'):
        """Merge another validation result."""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.metadata.update(other.metadata)


@handle_errors(
    category=ErrorCategory.VALIDATION,
    severity=ErrorSeverity.WARNING,
    reraise=False
)
def validate_excel_file(
        file_path: Union[str, Path],
        required_sheets: Optional[List[str]] = None,
        max_file_size_mb: float = 100.0
) -> ValidationResult:
    """
    Validate Excel file for laser trim analysis with comprehensive edge case handling.

    Args:
        file_path: Path to Excel file
        required_sheets: List of required sheet names
        max_file_size_mb: Maximum file size in MB

    Returns:
        ValidationResult with validation status and messages
    """
    start_time = time.time()
    logger.debug(f"Starting Excel file validation: {file_path}")
    
    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        metadata={}
    )
    
    # Validate inputs
    if file_path is None:
        error_msg = "File path cannot be None"
        result.add_error(error_msg)
        logger.error(f"File validation failed: {error_msg}")
        return result
    
    if isinstance(file_path, str) and not file_path.strip():
        error_msg = "File path cannot be empty"
        result.add_error(error_msg)
        logger.error(f"File validation failed: {error_msg}")
        return result
    
    if max_file_size_mb <= 0:
        error_msg = f"Invalid max file size: {max_file_size_mb}MB. Must be positive."
        result.add_error(error_msg)
        logger.error(f"File validation failed: {error_msg}")
        return result
    
    try:
        file_path = Path(file_path)
        logger.debug(f"Converted to Path object: {file_path}")
    except Exception as e:
        error_msg = f"Invalid file path: {str(e)}"
        result.add_error(error_msg)
        logger.error(f"File validation failed: {error_msg}", exc_info=True)
        return result

    # Check file exists
    if not file_path.exists():
        error_msg = f"File not found: {file_path}"
        result.add_error(error_msg)
        logger.error(f"File validation failed: {error_msg}")
        return result

    # Check file extension
    if file_path.suffix.lower() not in ['.xlsx', '.xls']:
        error_msg = f"Invalid file type: {file_path.suffix}. Expected .xlsx or .xls"
        result.add_error(error_msg)
        logger.error(f"File validation failed: {error_msg}")
        return result

    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    result.metadata['file_size_mb'] = file_size_mb
    logger.debug(f"File size: {file_size_mb:.2f} MB")

    if file_size_mb > max_file_size_mb:
        error_msg = f"File too large: {file_size_mb:.1f} MB (max: {max_file_size_mb} MB)"
        result.add_error(error_msg)
        logger.error(f"File validation failed: {error_msg}")
        return result

    if file_size_mb < 0.001:  # Less than 1 KB
        error_msg = "File appears to be empty"
        result.add_error(error_msg)
        logger.error(f"File validation failed: {error_msg}")
        return result

    # Try to read Excel file
    try:
        # Try openpyxl first for xlsx files
        try:
            logger.debug("Attempting to read file with openpyxl engine")
            excel_file = pd.ExcelFile(file_path, engine='openpyxl')
            logger.debug("Successfully read file with openpyxl")
        except Exception as openpyxl_error:
            # Try xlrd for xls files
            logger.debug(f"openpyxl failed: {str(openpyxl_error)}, trying xlrd")
            try:
                excel_file = pd.ExcelFile(file_path, engine='xlrd')
                logger.debug("Successfully read file with xlrd")
            except Exception as xlrd_error:
                # File might be corrupted or not a real Excel file
                error_msg = (f"Cannot read Excel file. It may be corrupted or not a valid Excel file. "
                           f"openpyxl error: {str(openpyxl_error)}, "
                           f"xlrd error: {str(xlrd_error)}")
                result.add_error(error_msg)
                logger.error(f"File validation failed: {error_msg}")
                return result
                
        sheet_names = excel_file.sheet_names
        result.metadata['sheet_names'] = sheet_names
        result.metadata['sheet_count'] = len(sheet_names)

        # Check if file has any sheets
        if not sheet_names:
            result.add_error("Excel file contains no sheets")
            return result

        # Check for required sheets
        if required_sheets:
            missing_sheets = set(required_sheets) - set(sheet_names)
            if missing_sheets:
                result.add_error(f"Missing required sheets: {', '.join(missing_sheets)}")

        # Check for empty sheets
        for sheet_name in sheet_names[:5]:  # Check first 5 sheets
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=5)
                if df.empty:
                    result.add_warning(f"Sheet '{sheet_name}' appears to be empty")
            except Exception as e:
                result.add_warning(f"Could not read sheet '{sheet_name}': {str(e)}")

        # Validate sheet naming patterns
        has_system_a = any('TRK' in sheet for sheet in sheet_names)
        has_system_b = any(sheet in ['test', 'Lin Error'] for sheet in sheet_names)

        if has_system_a and has_system_b:
            result.add_warning("File contains mixed system indicators (both System A and B)")
        elif not has_system_a and not has_system_b:
            result.add_warning("Could not detect system type from sheet names")

        result.metadata['detected_system'] = 'A' if has_system_a else 'B' if has_system_b else 'Unknown'

    except Exception as e:
        result.add_error(f"Failed to read Excel file: {str(e)}")
        return result

    # Check filename format
    filename = file_path.stem
    filename_parts = filename.split('_')

    if len(filename_parts) < 2:
        result.add_warning("Filename doesn't follow expected format: MODEL_SERIAL[_EXTRA]")
    else:
        result.metadata['model_from_filename'] = filename_parts[0]
        result.metadata['serial_from_filename'] = filename_parts[1]

    # Check for temporary file indicators
    if filename.startswith('~$'):
        error_msg = "File appears to be a temporary Excel file"
        result.add_error(error_msg)
        logger.warning(f"File validation warning: {error_msg}")

    # Log validation completion
    elapsed_time = time.time() - start_time
    logger.info(
        f"Excel file validation completed in {elapsed_time:.3f}s - "
        f"Valid: {result.is_valid}, "
        f"Errors: {len(result.errors)}, "
        f"Warnings: {len(result.warnings)}, "
        f"File: {file_path.name}"
    )
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Validation metadata: {json.dumps(result.metadata, indent=2, default=str)}")

    return result


def validate_analysis_data(
        data: Dict[str, Any],
        system_type: str = 'A'
) -> ValidationResult:
    """
    Validate analysis data structure and values.

    Args:
        data: Analysis data dictionary
        system_type: System type ('A' or 'B')

    Returns:
        ValidationResult
    """
    start_time = time.time()
    logger.debug(f"Starting analysis data validation for system type: {system_type}")
    
    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        metadata={'system_type': system_type}
    )

    # Check required fields
    required_fields = ['positions', 'errors']
    for field in required_fields:
        if field not in data or data[field] is None:
            error_msg = f"Missing required field: {field}"
            result.add_error(error_msg)
            logger.error(f"Data validation error: {error_msg}")
        elif not isinstance(data[field], (list, np.ndarray)):
            error_msg = f"Field '{field}' must be a list or array"
            result.add_error(error_msg)
            logger.error(f"Data validation error: {error_msg}")
        elif len(data[field]) == 0:
            error_msg = f"Field '{field}' is empty"
            result.add_error(error_msg)
            logger.error(f"Data validation error: {error_msg}")

    if not result.is_valid:
        logger.error("Analysis data validation failed due to missing/invalid required fields")
        return result

    # Validate data consistency
    positions = np.array(data['positions'])
    errors = np.array(data['errors'])

    if len(positions) != len(errors):
        result.add_error(f"Length mismatch: positions ({len(positions)}) vs errors ({len(errors)})")
        return result

    result.metadata['data_points'] = len(positions)

    # Check for sufficient data
    if len(positions) < 10:
        result.add_error(f"Insufficient data points: {len(positions)} (minimum: 10)")
    elif len(positions) < 50:
        result.add_warning(f"Low number of data points: {len(positions)}")

    # Validate position data
    if len(positions) > 0:
        # Check for monotonic increase
        if not np.all(np.diff(positions) >= 0):
            result.add_warning("Position data is not monotonically increasing")

        # Check position range
        pos_range = positions.max() - positions.min()
        result.metadata['position_range'] = pos_range

        if pos_range < 0.1:
            result.add_error(f"Position range too small: {pos_range:.3f}")
        elif pos_range > 1000:
            result.add_warning(f"Unusually large position range: {pos_range:.1f}")

        # Check for duplicate positions
        unique_positions = np.unique(positions)
        if len(unique_positions) < len(positions):
            n_duplicates = len(positions) - len(unique_positions)
            result.add_warning(f"Found {n_duplicates} duplicate position values")

    # Validate error data
    if len(errors) > 0:
        # Check for NaN or inf values
        n_nan = np.sum(np.isnan(errors))
        n_inf = np.sum(np.isinf(errors))

        if n_nan > 0:
            result.add_error(f"Found {n_nan} NaN values in error data")
        if n_inf > 0:
            result.add_error(f"Found {n_inf} infinite values in error data")

        # Check error magnitude
        valid_errors = errors[~(np.isnan(errors) | np.isinf(errors))]
        if len(valid_errors) > 0:
            max_error = np.max(np.abs(valid_errors))
            result.metadata['max_error'] = max_error

            if max_error > 1.0:
                result.add_warning(f"Large error values detected (max: {max_error:.3f})")
            elif max_error < 0.0001:
                result.add_warning(f"Very small error values detected (max: {max_error:.6f})")

    # Validate optional fields
    if 'upper_limits' in data and data['upper_limits'] is not None:
        upper_limits = np.array(data['upper_limits'])
        if len(upper_limits) != len(positions):
            result.add_warning(f"Upper limits length ({len(upper_limits)}) doesn't match positions")

    if 'lower_limits' in data and data['lower_limits'] is not None:
        lower_limits = np.array(data['lower_limits'])
        if len(lower_limits) != len(positions):
            result.add_warning(f"Lower limits length ({len(lower_limits)}) doesn't match positions")

    # System-specific validation
    if system_type == 'A':
        # System A specific checks
        if 'travel_length' in data:
            travel_length = data['travel_length']
            if travel_length < 50 or travel_length > 200:
                warning_msg = f"Unusual travel length for System A: {travel_length}"
                result.add_warning(warning_msg)
                logger.warning(f"Data validation warning: {warning_msg}")

    elif system_type == 'B':
        # System B specific checks
        if 'linearity_spec' in data:
            spec = data['linearity_spec']
            if spec < 0.001 or spec > 0.1:
                warning_msg = f"Unusual linearity spec for System B: {spec}"
                result.add_warning(warning_msg)
                logger.warning(f"Data validation warning: {warning_msg}")

    # Log validation completion
    elapsed_time = time.time() - start_time
    logger.info(
        f"Analysis data validation completed in {elapsed_time:.3f}s - "
        f"Valid: {result.is_valid}, "
        f"Errors: {len(result.errors)}, "
        f"Warnings: {len(result.warnings)}, "
        f"Data points: {result.metadata.get('data_points', 0)}"
    )

    return result


def validate_model_number(
        model: str,
        known_models: Optional[List[str]] = None
) -> ValidationResult:
    """
    Validate potentiometer model number.

    Args:
        model: Model number string
        known_models: List of known valid models

    Returns:
        ValidationResult
    """
    logger.debug(f"Validating model number: {model}")
    
    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        metadata={'model': model}
    )

    if not model or not isinstance(model, str):
        error_msg = "Model number must be a non-empty string"
        result.add_error(error_msg)
        logger.error(f"Model validation failed: {error_msg}")
        return result

    # Remove whitespace
    model = model.strip()
    logger.debug(f"Stripped model: '{model}'")

    # Check length
    if len(model) < 3:
        result.add_error(f"Model number too short: '{model}'")
    elif len(model) > 20:
        result.add_warning(f"Unusually long model number: '{model}'")

    # Common model patterns for potentiometers
    patterns = [
        r'^\d{4}$',  # 4 digits (e.g., 8340)
        r'^\d{4}-\d+$',  # 4 digits + suffix (e.g., 8340-1)
        r'^[A-Z]{2,3}\d{3,4}$',  # Letters + digits
        r'^\d{3,4}[A-Z]{1,2}$',  # Digits + letters
    ]

    # Check if matches any pattern
    matches_pattern = any(re.match(pattern, model) for pattern in patterns)
    if not matches_pattern:
        result.add_warning(f"Model '{model}' doesn't match common patterns")

    # Check against known models
    if known_models:
        if model not in known_models:
            # Check for close matches
            close_matches = [m for m in known_models if m.startswith(model[:3])]
            if close_matches:
                result.add_warning(
                    f"Model '{model}' not in known list. Similar: {', '.join(close_matches[:3])}"
                )
            else:
                result.add_warning(f"Model '{model}' not in known models list")

    # Extract base model for special handling
    base_model = model.split('-')[0] if '-' in model else model
    result.metadata['base_model'] = base_model

    # Special model checks
    if base_model == '8340':
        result.metadata['requires_carbon_screen'] = True
        logger.debug(f"Model {model} requires carbon screen")
        if '-1' in model:
            result.metadata['fixed_threshold'] = 0.4
            logger.debug(f"Model {model} has fixed threshold: 0.4")

    # Log validation result
    logger.info(
        f"Model validation completed - Valid: {result.is_valid}, "
        f"Model: '{model}', Base: '{base_model}', "
        f"Warnings: {len(result.warnings)}"
    )

    return result


def validate_resistance_values(
        untrimmed: Optional[float],
        trimmed: Optional[float],
        tolerance_percent: float = 20.0
) -> ValidationResult:
    """
    Validate resistance measurements.

    Args:
        untrimmed: Untrimmed resistance value
        trimmed: Trimmed resistance value
        tolerance_percent: Maximum acceptable change percentage

    Returns:
        ValidationResult
    """
    logger.debug(f"Validating resistance values - Untrimmed: {untrimmed}, Trimmed: {trimmed}")
    
    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        metadata={}
    )

    # Check for None values
    if untrimmed is None and trimmed is None:
        warning_msg = "No resistance values provided"
        result.add_warning(warning_msg)
        logger.warning(f"Resistance validation: {warning_msg}")
        return result

    # Validate untrimmed resistance
    if untrimmed is not None:
        result.metadata['untrimmed_resistance'] = untrimmed

        if untrimmed <= 0:
            result.add_error(f"Invalid untrimmed resistance: {untrimmed} Ω")
        elif untrimmed < 100:
            result.add_warning(f"Low untrimmed resistance: {untrimmed} Ω")
        elif untrimmed > 1_000_000:
            result.add_warning(f"High untrimmed resistance: {untrimmed} Ω")

    # Validate trimmed resistance
    if trimmed is not None:
        result.metadata['trimmed_resistance'] = trimmed

        if trimmed <= 0:
            result.add_error(f"Invalid trimmed resistance: {trimmed} Ω")
        elif trimmed < 100:
            result.add_warning(f"Low trimmed resistance: {trimmed} Ω")
        elif trimmed > 1_000_000:
            result.add_warning(f"High trimmed resistance: {trimmed} Ω")

    # Compare values if both available
    if untrimmed is not None and trimmed is not None and untrimmed > 0:
        change = trimmed - untrimmed
        change_percent = (change / untrimmed) * 100

        result.metadata['resistance_change'] = change
        result.metadata['resistance_change_percent'] = change_percent

        if abs(change_percent) > tolerance_percent:
            result.add_error(
                f"Resistance change ({change_percent:.1f}%) exceeds "
                f"tolerance ({tolerance_percent}%)"
            )
        elif abs(change_percent) > tolerance_percent * 0.8:
            result.add_warning(
                f"Resistance change ({change_percent:.1f}%) approaching "
                f"tolerance limit ({tolerance_percent}%)"
            )

        # Check for increase (unusual)
        if change_percent > 5:
            warning_msg = f"Resistance increased by {change_percent:.1f}% after trimming"
            result.add_warning(warning_msg)
            logger.warning(f"Resistance validation: {warning_msg}")

    # Log validation result
    logger.info(
        f"Resistance validation completed - Valid: {result.is_valid}, "
        f"Untrimmed: {untrimmed}, Trimmed: {trimmed}, "
        f"Change: {result.metadata.get('resistance_change_percent', 'N/A'):.1f}%"
    )

    return result


def validate_sigma_values(
        sigma_gradient: float,
        sigma_threshold: float,
        model: Optional[str] = None
) -> ValidationResult:
    """
    Validate sigma gradient values.

    Args:
        sigma_gradient: Calculated sigma gradient
        sigma_threshold: Sigma threshold
        model: Model number for specific validation

    Returns:
        ValidationResult
    """
    logger.debug(
        f"Validating sigma values - Gradient: {sigma_gradient}, "
        f"Threshold: {sigma_threshold}, Model: {model}"
    )
    
    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        metadata={
            'sigma_gradient': sigma_gradient,
            'sigma_threshold': sigma_threshold,
            'sigma_pass': sigma_gradient <= sigma_threshold
        }
    )

    # Validate gradient
    if sigma_gradient < 0:
        result.add_error(f"Negative sigma gradient: {sigma_gradient}")
    elif sigma_gradient == 0:
        result.add_warning("Zero sigma gradient - check if data is valid")
    elif sigma_gradient > 1.0:
        result.add_warning(f"Extremely high sigma gradient: {sigma_gradient}")

    # Validate threshold
    if sigma_threshold <= 0:
        result.add_error(f"Invalid sigma threshold: {sigma_threshold}")
    elif sigma_threshold > 1.0:
        result.add_warning(f"Unusually high sigma threshold: {sigma_threshold}")

    # Model-specific validation
    if model:
        if model.startswith('8340-1') and abs(sigma_threshold - 0.4) > 0.001:
            result.add_warning(f"Model 8340-1 should have threshold 0.4, got {sigma_threshold}")

    # Check margin
    if result.metadata['sigma_pass']:
        margin = sigma_threshold - sigma_gradient
        margin_percent = (margin / sigma_threshold) * 100
        result.metadata['margin_percent'] = margin_percent

        if margin_percent < 10:
            warning_msg = f"Low sigma margin: {margin_percent:.1f}%"
            result.add_warning(warning_msg)
            logger.warning(f"Sigma validation: {warning_msg}")

    # Log validation result
    logger.info(
        f"Sigma validation completed - Valid: {result.is_valid}, "
        f"Pass: {result.metadata['sigma_pass']}, "
        f"Gradient: {sigma_gradient:.4f}, Threshold: {sigma_threshold:.4f}, "
        f"Margin: {result.metadata.get('margin_percent', 'N/A'):.1f}%"
    )

    return result


# Composite validator for complete analysis
class AnalysisValidator:
    """Comprehensive validator for analysis results."""

    @staticmethod
    def validate_complete_analysis(
            file_path: Path,
            analysis_data: Dict[str, Any],
            model: str,
            system_type: str
    ) -> ValidationResult:
        """
        Perform complete validation of analysis data.

        Args:
            file_path: Path to source file
            analysis_data: Complete analysis data
            model: Model number
            system_type: System type

        Returns:
            Comprehensive ValidationResult
        """
        start_time = time.time()
        logger.info(f"Starting complete analysis validation for {file_path.name}")
        
        # Start with file validation
        result = validate_excel_file(file_path)

        if not result.is_valid:
            return result

        # Validate model
        model_result = validate_model_number(model)
        result.merge(model_result)

        # Validate analysis data
        data_result = validate_analysis_data(analysis_data, system_type)
        result.merge(data_result)

        # Validate specific measurements if available
        if 'untrimmed_resistance' in analysis_data or 'trimmed_resistance' in analysis_data:
            resistance_result = validate_resistance_values(
                analysis_data.get('untrimmed_resistance'),
                analysis_data.get('trimmed_resistance')
            )
            result.merge(resistance_result)

        # Validate sigma values if available
        if 'sigma_gradient' in analysis_data and 'sigma_threshold' in analysis_data:
            sigma_result = validate_sigma_values(
                analysis_data['sigma_gradient'],
                analysis_data['sigma_threshold'],
                model
            )
            result.merge(sigma_result)

        # Cross-validation checks
        if result.is_valid:
            result = AnalysisValidator._cross_validate(analysis_data, result)

        # Log complete validation results
        elapsed_time = time.time() - start_time
        logger.info(
            f"Complete analysis validation finished in {elapsed_time:.3f}s - "
            f"Valid: {result.is_valid}, "
            f"Total errors: {len(result.errors)}, "
            f"Total warnings: {len(result.warnings)}"
        )
        
        if not result.is_valid:
            logger.error(f"Validation errors: {', '.join(result.errors[:3])}")

        return result

    @staticmethod
    def _cross_validate(
            analysis_data: Dict[str, Any],
            result: ValidationResult
    ) -> ValidationResult:
        """Perform cross-validation of different metrics."""

        # Check consistency between pass/fail status and values
        if 'sigma_pass' in analysis_data and 'sigma_gradient' in analysis_data:
            expected_pass = analysis_data.get('sigma_gradient', 0) <= analysis_data.get('sigma_threshold', 0)
            if analysis_data['sigma_pass'] != expected_pass:
                result.add_error("Inconsistent sigma pass/fail status")

        # Check risk category consistency
        if 'risk_category' in analysis_data and 'failure_probability' in analysis_data:
            prob = analysis_data['failure_probability']
            category = analysis_data['risk_category']

            expected_category = 'High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low'
            if category != expected_category:
                result.add_warning(
                    f"Risk category '{category}' doesn't match "
                    f"failure probability {prob:.2f}"
                )

        return result


# Validation utility functions
def is_valid_serial_number(serial: str) -> bool:
    """
    Check if serial number follows expected format.

    Args:
        serial: Serial number string

    Returns:
        True if valid format
    """
    logger.debug(f"Checking serial number validity: {serial}")
    
    if not serial or not isinstance(serial, str):
        logger.debug(f"Invalid serial - empty or not string: {serial}")
        return False

    # Remove whitespace
    serial = serial.strip()

    # Check basic requirements
    if len(serial) < 3 or len(serial) > 20:
        logger.debug(f"Invalid serial length: {len(serial)}")
        return False

    # Common serial patterns
    patterns = [
        r'^[A-Z]\d{5,}$',  # Letter + digits (e.g., A12345)
        r'^[A-Z]{2}\d{4,}$',  # 2 letters + digits
        r'^\d{6,}$',  # All digits
        r'^[A-Z0-9]{6,}$',  # Alphanumeric
    ]

    is_valid = any(re.match(pattern, serial) for pattern in patterns)
    logger.debug(f"Serial '{serial}' validity: {is_valid}")
    return is_valid


def validate_date_range(
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp],
        max_days: int = 365
) -> Tuple[bool, Optional[str]]:
    """
    Validate date range for queries.

    Args:
        start_date: Start date
        end_date: End date
        max_days: Maximum allowed days in range

    Returns:
        Tuple of (is_valid, error_message)
    """
    logger.debug(f"Validating date range: {start_date} to {end_date}")
    
    if start_date is None and end_date is None:
        logger.debug("No date range specified - valid")
        return True, None

    if start_date and end_date:
        if start_date > end_date:
            error_msg = "Start date must be before end date"
            logger.error(f"Date range validation failed: {error_msg}")
            return False, error_msg

        days_diff = (end_date - start_date).days
        if days_diff > max_days:
            error_msg = f"Date range exceeds maximum of {max_days} days"
            logger.error(f"Date range validation failed: {error_msg}")
            return False, error_msg

    # Check for future dates
    now = pd.Timestamp.now()
    if start_date and start_date > now:
        error_msg = "Start date cannot be in the future"
        logger.error(f"Date range validation failed: {error_msg}")
        return False, error_msg
    if end_date and end_date > now:
        error_msg = "End date cannot be in the future"
        logger.error(f"Date range validation failed: {error_msg}")
        return False, error_msg

    logger.debug("Date range validation passed")
    return True, None


def validate_file_naming_convention(
        filename: str,
        expected_format: str = "MODEL_SERIAL[_EXTRA]"
) -> ValidationResult:
    """
    Validate file naming convention.

    Args:
        filename: Filename without extension
        expected_format: Expected naming format

    Returns:
        ValidationResult
    """
    logger.debug(f"Validating filename convention: {filename}")
    
    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        metadata={'filename': filename}
    )

    # Remove any extension
    filename = Path(filename).stem

    # Split by underscore
    parts = filename.split('_')

    if len(parts) < 2:
        error_msg = (f"Filename doesn't match format '{expected_format}'. "
                    f"Got: '{filename}'")
        result.add_error(error_msg)
        logger.error(f"Filename validation: {error_msg}")
        return result

    # Validate model part
    model = parts[0]
    model_result = validate_model_number(model)
    if not model_result.is_valid:
        result.add_error(f"Invalid model in filename: {model}")
    result.metadata['model'] = model

    # Validate serial part
    serial = parts[1]
    if not is_valid_serial_number(serial):
        result.add_warning(f"Serial number may be invalid: {serial}")
    result.metadata['serial'] = serial

    # Check for extra parts
    if len(parts) > 2:
        extra_parts = parts[2:]
        result.metadata['extra_parts'] = extra_parts

        # Common extra parts
        valid_extras = ['REWORK', 'RETEST', 'CAL', 'TEST', 'QC']
        for extra in extra_parts:
            if extra.upper() not in valid_extras and not extra.isdigit():
                warning_msg = f"Unexpected filename component: '{extra}'"
                result.add_warning(warning_msg)
                logger.debug(f"Filename validation: {warning_msg}")

    # Log validation result
    logger.info(
        f"Filename validation completed - Valid: {result.is_valid}, "
        f"Model: {result.metadata.get('model', 'N/A')}, "
        f"Serial: {result.metadata.get('serial', 'N/A')}"
    )

    return result


# Batch validation for multiple files
class BatchValidator:
    """Validator for batch processing operations."""

    @staticmethod
    def validate_batch(
            file_paths: List[Path],
            max_batch_size: int = 1000
    ) -> ValidationResult:
        """
        Validate a batch of files.

        Args:
            file_paths: List of file paths
            max_batch_size: Maximum allowed batch size

        Returns:
            ValidationResult for the batch
        """
        start_time = time.time()
        logger.info(f"Starting batch validation for {len(file_paths)} files")
        
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            metadata={
                'total_files': len(file_paths),
                'valid_files': 0,
                'invalid_files': []
            }
        )

        # Check batch size
        if len(file_paths) == 0:
            result.add_error("No files provided for batch processing")
            return result

        if len(file_paths) > max_batch_size:
            result.add_error(
                f"Batch size ({len(file_paths)}) exceeds maximum ({max_batch_size})"
            )
            return result

        # Check for duplicates
        unique_paths = set(file_paths)
        if len(unique_paths) < len(file_paths):
            n_duplicates = len(file_paths) - len(unique_paths)
            result.add_warning(f"Found {n_duplicates} duplicate files in batch")

        # Validate each file
        models_found = set()
        total_size_mb = 0

        for file_path in file_paths:
            file_result = validate_excel_file(file_path)

            if not file_result.is_valid:
                result.metadata['invalid_files'].append({
                    'file': str(file_path),
                    'errors': file_result.errors
                })
                result.add_warning(f"Invalid file: {file_path.name}")
            else:
                result.metadata['valid_files'] += 1
                total_size_mb += file_result.metadata.get('file_size_mb', 0)

                # Extract model from filename
                filename_result = validate_file_naming_convention(file_path.stem)
                if 'model' in filename_result.metadata:
                    models_found.add(filename_result.metadata['model'])

        result.metadata['total_size_mb'] = total_size_mb
        result.metadata['unique_models'] = list(models_found)
        result.metadata['model_count'] = len(models_found)

        # Batch-level validations
        if result.metadata['valid_files'] == 0:
            result.add_error("No valid files found in batch")
        elif result.metadata['valid_files'] < len(file_paths) * 0.5:
            result.add_warning(
                f"More than 50% of files are invalid "
                f"({result.metadata['valid_files']}/{len(file_paths)})"
            )

        # Check total size
        if total_size_mb > 1000:  # 1 GB
            result.add_warning(
                f"Large batch size: {total_size_mb:.1f} MB. "
                "Processing may take significant time."
            )

        # Check model diversity
        if len(models_found) > 20:
            result.add_warning(
                f"Batch contains {len(models_found)} different models. "
                "Consider processing by model for better organization."
            )

        # Log batch validation summary
        elapsed_time = time.time() - start_time
        logger.info(
            f"Batch validation completed in {elapsed_time:.3f}s - "
            f"Valid files: {result.metadata['valid_files']}/{len(file_paths)}, "
            f"Total size: {result.metadata['total_size_mb']:.1f} MB, "
            f"Unique models: {result.metadata['model_count']}"
        )
        
        if result.metadata['invalid_files']:
            logger.warning(
                f"Found {len(result.metadata['invalid_files'])} invalid files in batch"
            )

        return result


def validate_user_input(
    input_value: Any,
    input_type: str,
    constraints: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """
    Comprehensive validation for user inputs with edge case handling.
    
    Args:
        input_value: The value to validate
        input_type: Type of input ('number', 'text', 'date', 'file_path', 'email', etc.)
        constraints: Optional constraints like min, max, pattern, etc.
        
    Returns:
        ValidationResult
    """
    logger.debug(f"Validating user input - Type: {input_type}, Value: {repr(input_value)[:100]}")
    
    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        metadata={'input_type': input_type, 'original_value': input_value}
    )
    
    constraints = constraints or {}
    
    # Handle None/empty inputs
    if input_value is None:
        if constraints.get('required', False):
            error_msg = f"{input_type} is required but was not provided"
            result.add_error(error_msg)
            logger.error(f"User input validation: {error_msg}")
        return result
    
    # Type-specific validation
    if input_type == 'number':
        result = _validate_number_input(input_value, constraints, result)
        
    elif input_type == 'text':
        result = _validate_text_input(input_value, constraints, result)
        
    elif input_type == 'date':
        result = _validate_date_input(input_value, constraints, result)
        
    elif input_type == 'file_path':
        result = _validate_file_path_input(input_value, constraints, result)
        
    elif input_type == 'email':
        result = _validate_email_input(input_value, constraints, result)
        
    elif input_type == 'list':
        result = _validate_list_input(input_value, constraints, result)
        
    elif input_type == 'dict':
        result = _validate_dict_input(input_value, constraints, result)
        
    else:
        warning_msg = f"Unknown input type: {input_type}"
        result.add_warning(warning_msg)
        logger.warning(f"User input validation: {warning_msg}")
    
    # Log validation result
    if not result.is_valid:
        logger.warning(
            f"User input validation failed - Type: {input_type}, "
            f"Errors: {len(result.errors)}"
        )
    else:
        logger.debug(f"User input validation passed - Type: {input_type}")
    
    return result


def _validate_number_input(value: Any, constraints: Dict[str, Any], result: ValidationResult) -> ValidationResult:
    """Validate numeric input with edge cases."""
    try:
        # Convert to float
        if isinstance(value, str):
            # Handle common number formats
            cleaned_value = value.strip().replace(',', '')
            if cleaned_value.endswith('%'):
                num_value = float(cleaned_value[:-1]) / 100
                result.metadata['interpreted_as_percentage'] = True
            else:
                num_value = float(cleaned_value)
        else:
            num_value = float(value)
        
        result.metadata['numeric_value'] = num_value
        
        # Check for special values
        if np.isnan(num_value):
            result.add_error("Value is NaN (Not a Number)")
            return result
        
        if np.isinf(num_value):
            result.add_error("Value is infinite")
            return result
        
        # Range validation
        min_val = constraints.get('min')
        max_val = constraints.get('max')
        
        if min_val is not None and num_value < min_val:
            result.add_error(f"Value {num_value} is below minimum {min_val}")
        
        if max_val is not None and num_value > max_val:
            result.add_error(f"Value {num_value} is above maximum {max_val}")
        
        # Type constraints
        if constraints.get('integer_only', False):
            if not num_value.is_integer():
                result.add_error(f"Value {num_value} must be an integer")
        
        if constraints.get('positive_only', False) and num_value <= 0:
            result.add_error(f"Value {num_value} must be positive")
        
        # Precision warnings
        if constraints.get('precision'):
            decimal_places = len(str(num_value).split('.')[-1]) if '.' in str(num_value) else 0
            if decimal_places > constraints['precision']:
                result.add_warning(f"Value has {decimal_places} decimal places, expected {constraints['precision']}")
        
    except (ValueError, TypeError) as e:
        result.add_error(f"Invalid number format: {value} ({str(e)})")
    
    return result


def _validate_text_input(value: Any, constraints: Dict[str, Any], result: ValidationResult) -> ValidationResult:
    """Validate text input with edge cases."""
    if not isinstance(value, str):
        try:
            text_value = str(value)
            result.metadata['converted_to_string'] = True
        except Exception:
            result.add_error(f"Cannot convert {type(value).__name__} to string")
            return result
    else:
        text_value = value
    
    # Basic validation
    if constraints.get('required') and not text_value.strip():
        result.add_error("Text cannot be empty or only whitespace")
    
    # Length constraints
    min_len = constraints.get('min_length', 0)
    max_len = constraints.get('max_length')
    
    if len(text_value) < min_len:
        result.add_error(f"Text too short: {len(text_value)} chars (min: {min_len})")
    
    if max_len and len(text_value) > max_len:
        result.add_error(f"Text too long: {len(text_value)} chars (max: {max_len})")
    
    # Pattern matching
    if 'pattern' in constraints:
        pattern = constraints['pattern']
        if not re.match(pattern, text_value):
            result.add_error(f"Text doesn't match required pattern: {pattern}")
    
    # Forbidden characters
    if 'forbidden_chars' in constraints:
        forbidden = constraints['forbidden_chars']
        found_forbidden = [c for c in forbidden if c in text_value]
        if found_forbidden:
            result.add_error(f"Text contains forbidden characters: {found_forbidden}")
    
    # SQL injection prevention
    sql_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'EXEC', 'UNION']
    if any(keyword in text_value.upper() for keyword in sql_keywords):
        result.add_warning("Text contains SQL-like keywords")
    
    # Check for control characters
    if any(ord(c) < 32 for c in text_value if c not in '\n\r\t'):
        result.add_warning("Text contains control characters")
    
    result.metadata['sanitized_value'] = text_value.strip()
    
    return result


def _validate_date_input(value: Any, constraints: Dict[str, Any], result: ValidationResult) -> ValidationResult:
    """Validate date input with edge cases."""
    date_value = None
    
    # Try to parse date
    if isinstance(value, datetime):
        date_value = value
    elif isinstance(value, str):
        # Try common date formats
        formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S',
            '%Y%m%d', '%d-%b-%Y', '%d %B %Y'
        ]
        
        for fmt in formats:
            try:
                date_value = datetime.strptime(value.strip(), fmt)
                result.metadata['date_format'] = fmt
                break
            except ValueError:
                continue
        
        if not date_value:
            result.add_error(f"Cannot parse date: {value}")
            return result
    else:
        result.add_error(f"Invalid date type: {type(value).__name__}")
        return result
    
    result.metadata['parsed_date'] = date_value
    
    # Validate date range
    if 'min_date' in constraints:
        min_date = constraints['min_date']
        if isinstance(min_date, str):
            min_date = datetime.fromisoformat(min_date)
        if date_value < min_date:
            result.add_error(f"Date {date_value} is before minimum {min_date}")
    
    if 'max_date' in constraints:
        max_date = constraints['max_date']
        if isinstance(max_date, str):
            max_date = datetime.fromisoformat(max_date)
        if date_value > max_date:
            result.add_error(f"Date {date_value} is after maximum {max_date}")
    
    # Check for unrealistic dates
    if date_value.year < 1900:
        result.add_warning(f"Date year {date_value.year} seems unrealistic")
    
    if date_value > datetime.now() + pd.Timedelta(days=365):
        result.add_warning("Date is more than 1 year in the future")
    
    return result


def _validate_file_path_input(value: Any, constraints: Dict[str, Any], result: ValidationResult) -> ValidationResult:
    """Validate file path input with security checks."""
    if not isinstance(value, (str, Path)):
        result.add_error(f"Invalid file path type: {type(value).__name__}")
        return result
    
    try:
        path = Path(value)
        result.metadata['absolute_path'] = str(path.absolute())
        
        # Security checks
        # Check for path traversal attempts
        if '..' in str(path):
            result.add_error("Path contains '..' which could be a security risk")
            return result
        
        # Check for null bytes
        if '\0' in str(path):
            result.add_error("Path contains null bytes")
            return result
        
        # Validate based on constraints
        must_exist = constraints.get('must_exist', False)
        if must_exist and not path.exists():
            result.add_error(f"Path does not exist: {path}")
        
        file_type = constraints.get('file_type')
        if file_type == 'file' and path.exists() and not path.is_file():
            result.add_error(f"Path is not a file: {path}")
        elif file_type == 'directory' and path.exists() and not path.is_dir():
            result.add_error(f"Path is not a directory: {path}")
        
        # Extension validation
        allowed_extensions = constraints.get('allowed_extensions')
        if allowed_extensions and path.suffix.lower() not in allowed_extensions:
            result.add_error(f"File extension {path.suffix} not allowed. Allowed: {allowed_extensions}")
        
        # Check if path is in allowed directories
        allowed_dirs = constraints.get('allowed_directories')
        if allowed_dirs:
            in_allowed = any(
                path.absolute().is_relative_to(Path(allowed_dir).absolute())
                for allowed_dir in allowed_dirs
            )
            if not in_allowed:
                result.add_error("Path is not in allowed directories")
        
    except Exception as e:
        result.add_error(f"Invalid path: {str(e)}")
    
    return result


def _validate_email_input(value: Any, constraints: Dict[str, Any], result: ValidationResult) -> ValidationResult:
    """Validate email input."""
    if not isinstance(value, str):
        result.add_error(f"Email must be a string, got {type(value).__name__}")
        return result
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, value):
        result.add_error(f"Invalid email format: {value}")
    
    return result


def _validate_list_input(value: Any, constraints: Dict[str, Any], result: ValidationResult) -> ValidationResult:
    """Validate list input with edge cases."""
    if not isinstance(value, (list, tuple)):
        result.add_error(f"Expected list, got {type(value).__name__}")
        return result
    
    list_value = list(value)
    result.metadata['list_length'] = len(list_value)
    
    # Length constraints
    min_items = constraints.get('min_items', 0)
    max_items = constraints.get('max_items')
    
    if len(list_value) < min_items:
        result.add_error(f"List too short: {len(list_value)} items (min: {min_items})")
    
    if max_items and len(list_value) > max_items:
        result.add_error(f"List too long: {len(list_value)} items (max: {max_items})")
    
    # Check for duplicates
    if constraints.get('unique_items', False):
        # Handle unhashable items
        try:
            unique_items = set(list_value)
            if len(unique_items) < len(list_value):
                result.add_error("List contains duplicate items")
        except TypeError:
            # Items are not hashable, do manual check
            for i, item1 in enumerate(list_value):
                for j, item2 in enumerate(list_value[i+1:], i+1):
                    if item1 == item2:
                        result.add_error(f"Duplicate items at positions {i} and {j}")
                        break
    
    # Validate individual items
    item_type = constraints.get('item_type')
    if item_type:
        for i, item in enumerate(list_value):
            item_result = validate_user_input(
                item, 
                item_type, 
                constraints.get('item_constraints', {})
            )
            if not item_result.is_valid:
                result.add_error(f"Item {i}: {', '.join(item_result.errors)}")
    
    return result


def _validate_dict_input(value: Any, constraints: Dict[str, Any], result: ValidationResult) -> ValidationResult:
    """Validate dictionary input."""
    if not isinstance(value, dict):
        result.add_error(f"Expected dictionary, got {type(value).__name__}")
        return result
    
    # Required keys
    required_keys = constraints.get('required_keys', [])
    missing_keys = set(required_keys) - set(value.keys())
    if missing_keys:
        result.add_error(f"Missing required keys: {missing_keys}")
    
    # Forbidden keys
    forbidden_keys = constraints.get('forbidden_keys', [])
    found_forbidden = set(value.keys()) & set(forbidden_keys)
    if found_forbidden:
        result.add_error(f"Found forbidden keys: {found_forbidden}")
    
    return result


def validate_analysis_parameters(params: Dict[str, Any]) -> ValidationResult:
    """
    Validate analysis parameters for edge cases and common errors.
    """
    logger.debug(f"Validating analysis parameters: {list(params.keys())}")
    
    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        metadata={}
    )
    
    # Validate sigma threshold
    if 'sigma_threshold' in params:
        sigma_result = validate_user_input(
            params['sigma_threshold'],
            'number',
            {'min': 0.0001, 'max': 1.0, 'positive_only': True}
        )
        if not sigma_result.is_valid:
            result.add_error(f"Invalid sigma threshold: {', '.join(sigma_result.errors)}")
    
    # Validate filter frequency
    if 'filter_frequency' in params:
        freq_result = validate_user_input(
            params['filter_frequency'],
            'number',
            {'min': 0.1, 'max': 100.0, 'positive_only': True}
        )
        if not freq_result.is_valid:
            result.add_error(f"Invalid filter frequency: {', '.join(freq_result.errors)}")
    
    # Validate batch size
    if 'batch_size' in params:
        batch_result = validate_user_input(
            params['batch_size'],
            'number',
            {'min': 1, 'max': 1000, 'integer_only': True}
        )
        if not batch_result.is_valid:
            result.add_error(f"Invalid batch size: {', '.join(batch_result.errors)}")
    
    # Cross-parameter validation
    if params.get('start_date') and params.get('end_date'):
        if params['start_date'] > params['end_date']:
            error_msg = "Start date cannot be after end date"
            result.add_error(error_msg)
            logger.error(f"Parameter validation: {error_msg}")
    
    # Log validation summary
    logger.info(
        f"Analysis parameters validation completed - "
        f"Valid: {result.is_valid}, "
        f"Errors: {len(result.errors)}, "
        f"Parameters validated: {len(params)}"
    )
    
    return result