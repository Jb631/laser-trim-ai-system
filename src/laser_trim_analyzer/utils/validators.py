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

logger = logging.getLogger(__name__)


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


def validate_excel_file(
        file_path: Union[str, Path],
        required_sheets: Optional[List[str]] = None,
        max_file_size_mb: float = 100.0
) -> ValidationResult:
    """
    Validate Excel file for laser trim analysis.

    Args:
        file_path: Path to Excel file
        required_sheets: List of required sheet names
        max_file_size_mb: Maximum file size in MB

    Returns:
        ValidationResult with validation status and messages
    """
    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        metadata={}
    )

    file_path = Path(file_path)

    # Check file exists
    if not file_path.exists():
        result.add_error(f"File not found: {file_path}")
        return result

    # Check file extension
    if file_path.suffix.lower() not in ['.xlsx', '.xls']:
        result.add_error(f"Invalid file type: {file_path.suffix}. Expected .xlsx or .xls")
        return result

    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    result.metadata['file_size_mb'] = file_size_mb

    if file_size_mb > max_file_size_mb:
        result.add_error(f"File too large: {file_size_mb:.1f} MB (max: {max_file_size_mb} MB)")
        return result

    if file_size_mb < 0.001:  # Less than 1 KB
        result.add_error("File appears to be empty")
        return result

    # Try to read Excel file
    try:
        # Try openpyxl first for xlsx files
        try:
            excel_file = pd.ExcelFile(file_path, engine='openpyxl')
        except Exception as openpyxl_error:
            # Try xlrd for xls files
            try:
                excel_file = pd.ExcelFile(file_path, engine='xlrd')
            except Exception as xlrd_error:
                # File might be corrupted or not a real Excel file
                result.add_error(f"Cannot read Excel file. It may be corrupted or not a valid Excel file. "
                               f"openpyxl error: {str(openpyxl_error)}, "
                               f"xlrd error: {str(xlrd_error)}")
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
        result.add_error("File appears to be a temporary Excel file")

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
            result.add_error(f"Missing required field: {field}")
        elif not isinstance(data[field], (list, np.ndarray)):
            result.add_error(f"Field '{field}' must be a list or array")
        elif len(data[field]) == 0:
            result.add_error(f"Field '{field}' is empty")

    if not result.is_valid:
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
                result.add_warning(f"Unusual travel length for System A: {travel_length}")

    elif system_type == 'B':
        # System B specific checks
        if 'linearity_spec' in data:
            spec = data['linearity_spec']
            if spec < 0.001 or spec > 0.1:
                result.add_warning(f"Unusual linearity spec for System B: {spec}")

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
    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        metadata={'model': model}
    )

    if not model or not isinstance(model, str):
        result.add_error("Model number must be a non-empty string")
        return result

    # Remove whitespace
    model = model.strip()

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
        if '-1' in model:
            result.metadata['fixed_threshold'] = 0.4

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
    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        metadata={}
    )

    # Check for None values
    if untrimmed is None and trimmed is None:
        result.add_warning("No resistance values provided")
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
            result.add_warning(f"Resistance increased by {change_percent:.1f}% after trimming")

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
            result.add_warning(f"Low sigma margin: {margin_percent:.1f}%")

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
    if not serial or not isinstance(serial, str):
        return False

    # Remove whitespace
    serial = serial.strip()

    # Check basic requirements
    if len(serial) < 3 or len(serial) > 20:
        return False

    # Common serial patterns
    patterns = [
        r'^[A-Z]\d{5,}$',  # Letter + digits (e.g., A12345)
        r'^[A-Z]{2}\d{4,}$',  # 2 letters + digits
        r'^\d{6,}$',  # All digits
        r'^[A-Z0-9]{6,}$',  # Alphanumeric
    ]

    return any(re.match(pattern, serial) for pattern in patterns)


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
    if start_date is None and end_date is None:
        return True, None

    if start_date and end_date:
        if start_date > end_date:
            return False, "Start date must be before end date"

        days_diff = (end_date - start_date).days
        if days_diff > max_days:
            return False, f"Date range exceeds maximum of {max_days} days"

    # Check for future dates
    now = pd.Timestamp.now()
    if start_date and start_date > now:
        return False, "Start date cannot be in the future"
    if end_date and end_date > now:
        return False, "End date cannot be in the future"

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
        result.add_error(
            f"Filename doesn't match format '{expected_format}'. "
            f"Got: '{filename}'"
        )
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
                result.add_warning(f"Unexpected filename component: '{extra}'")

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

        return result