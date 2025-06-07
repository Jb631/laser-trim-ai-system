"""
Excel utility functions for the Laser Trim Analyzer.

Provides specialized functions for reading and parsing Excel files
used in potentiometer testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import re
import os
import psutil
import tempfile
from contextlib import contextmanager
import time

from laser_trim_analyzer.core.constants import (
    SYSTEM_A_COLUMNS, SYSTEM_B_COLUMNS,
    SYSTEM_A_CELLS, SYSTEM_B_CELLS,
    SystemIdentifier
)
from laser_trim_analyzer.core.models import SystemType
from laser_trim_analyzer.core.exceptions import (
    DataExtractionError, SheetNotFoundError, SystemDetectionError
)
from laser_trim_analyzer.core.error_handlers import (
    ErrorCode, ErrorCategory, ErrorSeverity,
    error_handler, handle_errors, validate_file_upload,
    check_system_resources
)

# Import memory safety if available
try:
    from laser_trim_analyzer.core.memory_safety import (
        get_memory_validator, memory_safe_array, memory_safe_string,
        SafeFileHandle, MemorySafetyConfig, memory_safe_context
    )
    HAS_MEMORY_SAFETY = True
except ImportError:
    HAS_MEMORY_SAFETY = False

# Import secure logging with fallback
try:
    from laser_trim_analyzer.core.secure_logging import (
        get_logger, LogLevel, logged_function
    )
    secure_logger = get_logger(__name__, log_level=LogLevel.DEBUG)
    logger = logging.getLogger(__name__)
except ImportError:
    # Fallback to standard logging
    logger = logging.getLogger(__name__)
    secure_logger = None
    
    # Create a simple logged_function decorator fallback
    def logged_function(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Maximum limits for memory safety
MAX_EXCEL_ROWS = 1_000_000
MAX_EXCEL_SIZE_MB = 100
MAX_ARRAY_SIZE = 10_000_000  # 10M elements


@handle_errors(
    category=ErrorCategory.FILE_IO,
    severity=ErrorSeverity.ERROR,
    max_retries=2
)
@logged_function(log_inputs=True, log_outputs=True, log_performance=True)
def read_excel_sheet(
        file_path: Path,
        sheet_name: Union[str, int],
        header: Optional[int] = None,
        skiprows: Optional[int] = None
) -> pd.DataFrame:
    """
    Read a specific sheet from an Excel file with comprehensive error handling.

    Args:
        file_path: Path to Excel file
        sheet_name: Name or index of sheet
        header: Row to use as header
        skiprows: Rows to skip at start

    Returns:
        DataFrame with sheet data

    Raises:
        SheetNotFoundError: If sheet not found
        DataExtractionError: If file cannot be read
    """
    file_path = Path(file_path)
    
    # Log file information
    if secure_logger:
        file_info = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0,
            'file_extension': file_path.suffix,
            'sheet_name': sheet_name,
            'header': header,
            'skiprows': skiprows
        }
        secure_logger.debug("Starting Excel file read operation", context=file_info)
    
    # Validate file before attempting to read
    is_valid, error_msg = validate_file_upload(
        file_path,
        max_size_mb=100,
        allowed_extensions=['.xls', '.xlsx', '.xlsm', '.xlsb']
    )
    if not is_valid:
        error_handler.handle_error(
            error=DataExtractionError(error_msg),
            category=ErrorCategory.FILE_IO,
            severity=ErrorSeverity.ERROR,
            code=ErrorCode.FILE_WRONG_FORMAT,
            user_message=error_msg
        )
        raise DataExtractionError(error_msg)
    
    # Check system resources
    resources = check_system_resources()
    if resources.get('memory', {}).get('status') == 'critical':
        error_handler.handle_error(
            error=MemoryError("Insufficient memory to process file"),
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.ERROR,
            code=ErrorCode.INSUFFICIENT_MEMORY,
            user_message="Not enough memory to process this file. Please close other applications.",
            recovery_suggestions=[
                "Close other applications",
                "Try processing a smaller file",
                "Restart the application"
            ]
        )
        raise MemoryError("Insufficient memory to process Excel file")
    
    # Check if file might be password protected or corrupted
    try:
        # Quick check by trying to open file
        with open(file_path, 'rb') as f:
            header_bytes = f.read(8)
            
        # Check for encrypted Excel file signature
        if header_bytes.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
            # This is an OLE file, check if it's encrypted
            try:
                import olefile
                if olefile.isOleFile(file_path):
                    ole = olefile.OleFileIO(file_path)
                    if ole.exists('EncryptionInfo') or ole.exists('EncryptedPackage'):
                        ole.close()
                        raise DataExtractionError("Excel file appears to be password protected")
                    ole.close()
            except ImportError:
                # Can't check for encryption without olefile
                pass
    except Exception as e:
        logger.warning(f"Could not perform pre-read checks: {e}")

    # Try to read with different engines
    errors = []
    
    # First, try to read with default pandas (auto-detects engine)
    try:
        # Use memory safety context if available
        if HAS_MEMORY_SAFETY:
            with memory_safe_context(f"read_excel_{file_path.name}", max_memory_mb=500):
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    header=header,
                    skiprows=skiprows
                )
                
                # Validate dataframe size
                if df.shape[0] > MAX_EXCEL_ROWS:
                    raise DataExtractionError(
                        f"Excel file has too many rows: {df.shape[0]} (max: {MAX_EXCEL_ROWS})"
                    )
                
                # Check memory usage
                memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                if memory_usage_mb > MAX_EXCEL_SIZE_MB:
                    raise MemoryError(
                        f"Excel data uses too much memory: {memory_usage_mb:.1f}MB (max: {MAX_EXCEL_SIZE_MB}MB)"
                    )
        else:
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=header,
                skiprows=skiprows
            )
        
        # Log successful read with details
        if secure_logger:
            secure_logger.info(
                f"Successfully read sheet '{sheet_name}' from {file_path.name}",
                context={
                    'sheet_dimensions': {'rows': df.shape[0], 'columns': df.shape[1]},
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                    'column_count': len(df.columns),
                    'non_null_counts': df.count().to_dict() if df.shape[0] < 1000 else 'large_dataset'
                }
            )
        logger.debug(f"Read sheet '{sheet_name}' from {file_path.name}: {df.shape}")
        return df

    except ValueError as e:
        if "Worksheet" in str(e) and "does not exist" in str(e):
            error_handler.handle_error(
                error=e,
                category=ErrorCategory.FILE_IO,
                severity=ErrorSeverity.ERROR,
                code=ErrorCode.EXCEL_SHEET_MISSING,
                user_message=f"Sheet '{sheet_name}' not found in {file_path.name}",
                additional_data={'file_path': str(file_path), 'sheet_name': sheet_name}
            )
            raise SheetNotFoundError(f"Sheet '{sheet_name}' not found in {file_path.name}")
        errors.append(("auto-detect", e))
    except MemoryError as e:
        error_handler.handle_error(
            error=e,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.ERROR,
            code=ErrorCode.INSUFFICIENT_MEMORY,
            user_message="File is too large to process. Try a smaller file or close other applications.",
            additional_data={'file_size_mb': file_path.stat().st_size / (1024*1024)}
        )
        raise
    except Exception as e:
        if "Password protected" in str(e) or "encrypted" in str(e).lower():
            error_handler.handle_error(
                error=e,
                category=ErrorCategory.FILE_IO,
                severity=ErrorSeverity.ERROR,
                code=ErrorCode.EXCEL_PASSWORD_PROTECTED,
                user_message="Excel file is password protected. Please remove the password and try again.",
                recovery_suggestions=[
                    "Open the file in Excel",
                    "Remove password protection",
                    "Save and try again"
                ]
            )
            raise DataExtractionError("Excel file is password protected")
        errors.append(("auto-detect", e))

    # If default failed, try openpyxl explicitly (for xlsx)
    if file_path.suffix.lower() in ['.xlsx', '.xlsm', '.xlsb']:
        try:
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=header,
                skiprows=skiprows,
                engine='openpyxl'
            )
            logger.debug(f"Read sheet '{sheet_name}' from {file_path.name} using openpyxl: {df.shape}")
            return df
        except ImportError:
            errors.append(("openpyxl", "openpyxl not installed"))
        except Exception as e:
            errors.append(("openpyxl", e))

    # Try xlrd for xls files
    if file_path.suffix.lower() == '.xls':
        try:
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=header,
                skiprows=skiprows,
                engine='xlrd'
            )
            logger.debug(f"Read sheet '{sheet_name}' from {file_path.name} using xlrd: {df.shape}")
            return df
        except ImportError:
            error_handler.handle_error(
                error=ImportError("xlrd not installed"),
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.ERROR,
                code=ErrorCode.CONFIG_MISSING,
                user_message="Cannot read .xls files - required package 'xlrd' is not installed.",
                recovery_suggestions=[
                    "Install xlrd: pip install xlrd",
                    "Convert the file to .xlsx format",
                    "Use a different file"
                ]
            )
            raise DataExtractionError("Cannot read .xls files - xlrd not installed")
        except Exception as e:
            errors.append(("xlrd", e))

    # If all attempts failed, provide detailed error information
    error_details = "\n".join([f"  - {engine}: {error}" for engine, error in errors])
    
    error_handler.handle_error(
        error=DataExtractionError(f"Failed to read Excel file after trying multiple engines"),
        category=ErrorCategory.FILE_IO,
        severity=ErrorSeverity.ERROR,
        code=ErrorCode.EXCEL_CORRUPT,
        user_message=f"Unable to read Excel file '{file_path.name}'. The file may be corrupted or in an unsupported format.",
        technical_details=error_details,
        recovery_suggestions=[
            "Try opening and re-saving the file in Excel",
            "Check if the file opens correctly in Excel",
            "Convert to a different Excel format (.xlsx recommended)",
            "Check if the file is corrupted"
        ],
        additional_data={
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'sheet_name': sheet_name,
            'errors': errors
        }
    )
    
    raise DataExtractionError(f"Failed to read Excel sheet '{sheet_name}' from {file_path.name}")


@logged_function(log_inputs=True, log_outputs=True, log_performance=True)
def extract_cell_value(
        file_path: Path,
        sheet_name: str,
        cell_ref: str,
        default: Any = None
) -> Any:
    """
    Extract value from a specific cell.

    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name
        cell_ref: Cell reference (e.g., 'B10')
        default: Default value if extraction fails

    Returns:
        Cell value or default
    """
    # Log operation start
    if secure_logger:
        secure_logger.debug(
            f"Extracting cell value from {cell_ref}",
            context={
                'file': file_path.name,
                'sheet': sheet_name,
                'cell': cell_ref
            }
        )
    
    try:
        # Convert cell reference to row/col indices
        col_match = re.match(r'^([A-Z]+)(\d+)$', cell_ref.upper())
        if not col_match:
            logger.warning(f"Invalid cell reference: {cell_ref}")
            return default

        col_letters, row_num = col_match.groups()

        # Convert column letters to index
        col_idx = 0
        for char in col_letters:
            col_idx = col_idx * 26 + (ord(char) - ord('A')) + 1
        col_idx -= 1  # 0-based index

        row_idx = int(row_num) - 1  # 0-based index

        # Read the entire sheet first for better compatibility
        try:
            # Try auto-detection first
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        except Exception:
            # Try with specific engines
            if file_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine='openpyxl')
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine='xlrd')

        # Check if the cell exists
        if row_idx >= len(df) or col_idx >= len(df.columns):
            logger.warning(f"Cell {cell_ref} out of bounds in sheet {sheet_name}")
            return default

        # Get the value
        value = df.iloc[row_idx, col_idx]

        # Handle NaN
        if pd.isna(value):
            return default

        # Handle string values that might be numbers
        if isinstance(value, str):
            # Remove any non-numeric characters except decimal point and minus
            cleaned = re.sub(r'[^\d.\-]', '', value)
            if cleaned:
                try:
                    value = float(cleaned)
                except ValueError:
                    pass

        # Log successful extraction
        if secure_logger:
            secure_logger.debug(
                f"Successfully extracted cell value",
                context={
                    'cell': cell_ref,
                    'sheet': sheet_name,
                    'value': str(value)[:100] if value is not None else 'None',
                    'value_type': type(value).__name__,
                    'original_type': type(df.iloc[row_idx, col_idx]).__name__ if row_idx < len(df) and col_idx < len(df.columns) else 'N/A'
                }
            )
        
        logger.debug(f"Extracted {cell_ref}={value} from {sheet_name}")
        return value

    except Exception as e:
        if secure_logger:
            secure_logger.warning(
                f"Failed to extract cell value",
                context={
                    'cell': cell_ref,
                    'sheet': sheet_name,
                    'error': str(e),
                    'default_value': str(default)
                }
            )
        logger.warning(f"Failed to extract cell {cell_ref} from {sheet_name}: {e}")
        return default


@logged_function(log_inputs=False, log_outputs=True, log_performance=True)
def find_data_columns(
        df: pd.DataFrame,
        system: str = 'A'
) -> Dict[str, int]:
    """
    Find data columns in DataFrame based on system type.

    Args:
        df: DataFrame to search
        system: System type ('A' or 'B')

    Returns:
        Dictionary mapping column names to indices
    """
    # Log operation start
    if secure_logger:
        secure_logger.debug(
            "Finding data columns in DataFrame",
            context={
                'system_type': system,
                'dataframe_shape': {'rows': df.shape[0], 'columns': df.shape[1]},
                'column_names': list(df.columns[:10]) if hasattr(df, 'columns') else 'no_headers'
            }
        )
    
    if system == 'A':
        columns = SYSTEM_A_COLUMNS.copy()
    else:
        columns = SYSTEM_B_COLUMNS.copy()

    # Validate columns exist
    max_col = max(columns.values())
    if df.shape[1] <= max_col:
        if secure_logger:
            secure_logger.warning(
                "Insufficient columns in DataFrame",
                context={
                    'actual_columns': df.shape[1],
                    'required_columns': max_col + 1,
                    'system': system
                }
            )
        logger.warning(f"DataFrame has only {df.shape[1]} columns, expected at least {max_col + 1}")
        return {}

    # Find first row with numeric data in position column
    pos_col = columns['position']
    data_start_row = 0

    for i in range(min(20, df.shape[0])):  # Check first 20 rows
        try:
            val = pd.to_numeric(df.iloc[i, pos_col], errors='coerce')
            if pd.notna(val):
                data_start_row = i
                break
        except:
            continue

    # Log data detection results
    if secure_logger:
        secure_logger.debug(
            "Data row detection complete",
            context={
                'data_start_row': data_start_row,
                'rows_scanned': min(20, df.shape[0]),
                'position_column': pos_col
            }
        )
    
    logger.debug(f"Found data starting at row {data_start_row}")

    # Validate we have numeric data
    if data_start_row >= df.shape[0] - 10:  # Need at least 10 data points
        logger.warning("Insufficient numeric data found")
        return {}

    return columns


@logged_function(log_inputs=True, log_outputs=True, log_performance=True)
def detect_system_type(file_path: Path) -> SystemType:
    """
    Detect system type from Excel file structure.

    Args:
        file_path: Path to Excel file

    Returns:
        Detected system type

    Raises:
        SystemDetectionError: If system cannot be detected
    """
    # Log detection start
    if secure_logger:
        secure_logger.start_performance_tracking('system_detection')
        secure_logger.info(
            "Starting system type detection",
            context={
                'file': file_path.name,
                'extension': file_path.suffix,
                'size_mb': file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0
            }
        )
    
    try:
        # Try different approaches to read the file
        excel_file = None
        
        # First try with pandas directly
        try:
            # For .xls files, try without specifying engine first
            if file_path.suffix.lower() == '.xls':
                try:
                    excel_file = pd.ExcelFile(file_path)
                except Exception:
                    # If that fails, try with xlrd explicitly
                    excel_file = pd.ExcelFile(file_path, engine='xlrd')
            else:
                # For .xlsx files, use openpyxl
                excel_file = pd.ExcelFile(file_path, engine='openpyxl')
                
        except Exception as read_error:
            # If all pandas attempts fail, try to detect based on filename only
            logger.warning(f"Cannot read Excel file structure: {read_error}")
            logger.info("Attempting to detect system type from filename pattern")
            
            filename = file_path.stem.upper()
            
            # Check filename patterns
            if any(filename.startswith(prefix) for prefix in ['8340', '834', '8506', '8852']):
                logger.info("Detected System B from filename pattern")
                return SystemType.SYSTEM_B
            elif any(filename.startswith(prefix) for prefix in ['68', '78', '85']):
                logger.info("Detected System A from filename pattern")
                return SystemType.SYSTEM_A
            else:
                # Default to System B for .xls files as they're more common
                logger.warning(f"Could not read file or detect from filename, defaulting to System B for .xls file")
                return SystemType.SYSTEM_B
                
        # If we successfully read the file, check sheets
        if excel_file is not None:
            sheet_names = excel_file.sheet_names
            
            # Log sheet information
            if secure_logger:
                secure_logger.debug(
                    "Analyzing sheet names for system type",
                    context={
                        'sheet_count': len(sheet_names),
                        'sheet_names': sheet_names[:10] if len(sheet_names) > 10 else sheet_names,
                        'truncated': len(sheet_names) > 10
                    }
                )
            logger.debug(f"Sheets in {file_path.name}: {sheet_names}")

            # Check for System A patterns
            system_a_indicators = [
                SystemIdentifier.SYSTEM_A_PATTERN.value,
                "TRK1", "TRK2",
                "SEC1", "SEC2"
            ]

            for sheet in sheet_names:
                if any(indicator in sheet for indicator in system_a_indicators):
                    if secure_logger:
                        detection_time = secure_logger.end_performance_tracking('system_detection')
                        secure_logger.info(
                            "System A detected from sheet pattern",
                            context={
                                'matching_sheet': sheet,
                                'matched_indicators': [ind for ind in system_a_indicators if ind in sheet]
                            },
                            performance={'detection_time_ms': detection_time * 1000}
                        )
                    logger.info(f"Detected System A from sheet: {sheet}")
                    return SystemType.SYSTEM_A

            # Check for System B patterns
            system_b_indicators = [
                SystemIdentifier.SYSTEM_B_PATTERN_1.value,
                SystemIdentifier.SYSTEM_B_PATTERN_2.value,
                "test", "Lin Error", "Trim"
            ]

            system_b_count = sum(
                1 for sheet in sheet_names
                if any(indicator in sheet for indicator in system_b_indicators)
            )

            if system_b_count >= 2:  # Need at least 2 matching sheets
                if secure_logger:
                    detection_time = secure_logger.end_performance_tracking('system_detection')
                    secure_logger.info(
                        "System B detected from sheet patterns",
                        context={
                            'matching_count': system_b_count,
                            'total_sheets': len(sheet_names)
                        },
                        performance={'detection_time_ms': detection_time * 1000}
                    )
                logger.info("Detected System B from sheet patterns")
                return SystemType.SYSTEM_B

        # Check filename patterns as fallback
        filename = file_path.stem.upper()

        if any(filename.startswith(prefix) for prefix in ['8340', '834', '8506', '8852']):
            if secure_logger:
                secure_logger.info(
                    "System B detected from filename pattern",
                    context={'filename': filename, 'method': 'filename_prefix'}
                )
            logger.info("Detected System B from filename pattern")
            return SystemType.SYSTEM_B
        elif any(filename.startswith(prefix) for prefix in ['68', '78', '85']):
            if secure_logger:
                secure_logger.info(
                    "System A detected from filename pattern",
                    context={'filename': filename, 'method': 'filename_prefix'}
                )
            logger.info("Detected System A from filename pattern")
            return SystemType.SYSTEM_A

        # Default based on file extension
        if file_path.suffix.lower() == '.xls':
            if secure_logger:
                detection_time = secure_logger.end_performance_tracking('system_detection')
                secure_logger.warning(
                    "Using default system type for .xls file",
                    context={'filename': filename, 'default': 'System B'},
                    performance={'detection_time_ms': detection_time * 1000}
                )
            logger.warning(f"Could not definitively detect system type for {file_path.name}, defaulting to System B for .xls file")
            return SystemType.SYSTEM_B
        else:
            if secure_logger:
                detection_time = secure_logger.end_performance_tracking('system_detection')
                secure_logger.warning(
                    "Using default system type",
                    context={'filename': filename, 'default': 'System A'},
                    performance={'detection_time_ms': detection_time * 1000}
                )
            logger.warning(f"Could not definitively detect system type for {file_path.name}, defaulting to System A")
            return SystemType.SYSTEM_A

    except Exception as e:
        if secure_logger:
            detection_time = secure_logger.end_performance_tracking('system_detection')
            secure_logger.error(
                "Error during system type detection",
                context={'error_type': type(e).__name__, 'error_message': str(e)},
                performance={'detection_time_ms': detection_time * 1000},
                error=e
            )
        logger.error(f"Error detecting system type: {e}")
        # For unreadable files, try filename detection
        filename = file_path.stem.upper()
        if any(filename.startswith(prefix) for prefix in ['8340', '834', '8506', '8852']):
            logger.info("Detected System B from filename pattern (fallback)")
            return SystemType.SYSTEM_B
        else:
            raise SystemDetectionError(f"Failed to detect system type: {str(e)}")


def find_trim_sheets(
        file_path: Path,
        track_id: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Find all trim-related sheets in Excel file.

    Args:
        file_path: Path to Excel file
        track_id: Specific track to find (for System A)

    Returns:
        Dictionary categorizing sheets by type
    """
    # Try to read file with auto-detection first
    try:
        excel_file = pd.ExcelFile(file_path)
    except Exception:
        # If auto-detect fails, try specific engines
        if file_path.suffix.lower() == '.xlsx':
            try:
                excel_file = pd.ExcelFile(file_path, engine='openpyxl')
            except Exception as e:
                logger.error(f"Failed to read XLSX file: {e}")
                return {'untrimmed': [], 'trimmed': [], 'final': [], 'other': []}
        elif file_path.suffix.lower() == '.xls':
            try:
                excel_file = pd.ExcelFile(file_path, engine='xlrd')
            except ImportError:
                logger.error("xlrd not installed. Install with: pip install xlrd")
                return {'untrimmed': [], 'trimmed': [], 'final': [], 'other': []}
            except Exception as e:
                logger.error(f"Failed to read XLS file: {e}")
                return {'untrimmed': [], 'trimmed': [], 'final': [], 'other': []}
        else:
            logger.error(f"Unknown file extension: {file_path.suffix}")
            return {'untrimmed': [], 'trimmed': [], 'final': [], 'other': []}
            
    sheet_names = excel_file.sheet_names

    sheets = {
        'untrimmed': [],
        'trimmed': [],
        'final': [],
        'other': []
    }

    for sheet in sheet_names:
        sheet_upper = sheet.upper()

        # Skip temporary sheets
        if sheet.startswith('~'):
            continue

        # Categorize sheets
        if track_id and track_id not in sheet:
            continue

        if ' 0' in sheet or '_0' in sheet or sheet.endswith('0'):
            sheets['untrimmed'].append(sheet)
        elif 'TRM' in sheet_upper or 'TRIM' in sheet_upper:
            sheets['trimmed'].append(sheet)
        elif 'LIN ERROR' in sheet_upper or 'FINAL' in sheet_upper:
            sheets['final'].append(sheet)
        elif 'TEST' in sheet_upper and 'TRM' not in sheet_upper:
            sheets['untrimmed'].append(sheet)
        else:
            sheets['other'].append(sheet)

    return sheets


@logged_function(log_inputs=False, log_outputs=True, log_performance=True)
def validate_data_integrity(
        positions: List[float],
        errors: List[float],
        min_points: int = 10
) -> Tuple[bool, List[str]]:
    """
    Validate data integrity for analysis.

    Args:
        positions: Position data
        errors: Error data
        min_points: Minimum required data points

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Log validation start
    if secure_logger:
        secure_logger.debug(
            "Starting data integrity validation",
            context={
                'position_count': len(positions),
                'error_count': len(errors),
                'min_points_required': min_points
            }
        )

    # Check data length
    if len(positions) != len(errors):
        issues.append(f"Length mismatch: {len(positions)} positions, {len(errors)} errors")

    if len(positions) < min_points:
        issues.append(f"Insufficient data points: {len(positions)} < {min_points}")

    # Check for NaN/inf values
    if any(pd.isna(positions)) or any(pd.isna(errors)):
        issues.append("Data contains NaN values")

    if any(np.isinf(positions)) or any(np.isinf(errors)):
        issues.append("Data contains infinite values")

    # Check for duplicate positions
    unique_positions = len(set(positions))
    if unique_positions < len(positions):
        issues.append(f"Duplicate positions found: {len(positions) - unique_positions} duplicates")

    # Check position range
    if positions:
        pos_range = max(positions) - min(positions)
        if pos_range < 0.1:  # Minimum travel
            issues.append(f"Position range too small: {pos_range}")

    # Check for monotonic positions
    if positions:
        diffs = np.diff(positions)
        if not (np.all(diffs >= 0) or np.all(diffs <= 0)):
            issues.append("Positions are not monotonic")

    is_valid = len(issues) == 0
    
    # Log validation results
    if secure_logger:
        secure_logger.info(
            "Data integrity validation complete",
            context={
                'is_valid': is_valid,
                'issue_count': len(issues),
                'issues': issues if not is_valid else None,
                'data_stats': {
                    'position_range': max(positions) - min(positions) if positions else 0,
                    'unique_positions': len(set(positions)) if positions else 0,
                    'has_nan': any(pd.isna(positions)) or any(pd.isna(errors)) if positions and errors else False,
                    'has_inf': any(np.isinf(positions)) or any(np.isinf(errors)) if positions and errors else False
                }
            }
        )
    
    return is_valid, issues


def extract_metadata_from_sheets(
        file_path: Path,
        system_type: SystemType
) -> Dict[str, Any]:
    """
    Extract metadata from Excel file sheets.

    Args:
        file_path: Path to Excel file
        system_type: System type

    Returns:
        Dictionary of extracted metadata
    """
    metadata = {}

    try:
        # Get appropriate cell mappings
        if system_type == SystemType.SYSTEM_A:
            cells = SYSTEM_A_CELLS
        else:
            cells = SYSTEM_B_CELLS

        # Find sheets
        sheets = find_trim_sheets(file_path)

        # Try to extract from first available sheet
        test_sheet = None
        if sheets['untrimmed']:
            test_sheet = sheets['untrimmed'][0]
        elif sheets['final']:
            test_sheet = sheets['final'][0]

        if test_sheet:
            # Extract metadata from known cells
            metadata['unit_length'] = extract_cell_value(
                file_path, test_sheet, cells.get('unit_length', 'B26')
            )

            # Try to extract additional metadata
            if system_type == SystemType.SYSTEM_A:
                metadata['test_date'] = extract_cell_value(
                    file_path, test_sheet, 'B2'
                )
                metadata['operator'] = extract_cell_value(
                    file_path, test_sheet, 'B3'
                )
            else:
                metadata['test_voltage'] = extract_cell_value(
                    file_path, test_sheet, 'B1'
                )

    except Exception as e:
        logger.warning(f"Failed to extract metadata: {e}")

    return metadata


def create_summary_dataframe(
        file_path: Path,
        results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create a summary DataFrame from analysis results.

    Args:
        file_path: Source Excel file
        results: Analysis results dictionary

    Returns:
        Summary DataFrame
    """
    summary_data = {
        'Filename': file_path.name,
        'Model': results.get('model', 'Unknown'),
        'Serial': results.get('serial', 'Unknown'),
        'System': results.get('system', 'Unknown'),
        'Overall Status': results.get('overall_status', 'Unknown'),
        'Processing Time': results.get('processing_time', 0),
    }

    # Add track-specific data
    tracks = results.get('tracks', {})
    if len(tracks) == 1:
        # Single track - add data directly
        track_data = next(iter(tracks.values()))
        summary_data.update({
            'Sigma Gradient': track_data.get('sigma_gradient'),
            'Sigma Threshold': track_data.get('sigma_threshold'),
            'Sigma Pass': track_data.get('sigma_pass'),
            'Risk Category': track_data.get('risk_category')
        })
    else:
        # Multi-track - add data for each track
        for track_id, track_data in tracks.items():
            summary_data[f'{track_id} Status'] = track_data.get('status')
            summary_data[f'{track_id} Sigma'] = track_data.get('sigma_gradient')
            summary_data[f'{track_id} Pass'] = track_data.get('sigma_pass')

    return pd.DataFrame([summary_data])


@logged_function(log_inputs=True, log_outputs=True, log_performance=True)
def validate_excel_file_integrity(file_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Validate Excel file integrity before processing.
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    metadata = {
        'file_size_mb': 0,
        'is_encrypted': False,
        'is_corrupted': False,
        'sheet_count': 0,
        'has_macros': False,
        'file_type': 'unknown'
    }
    
    # Log validation start
    if secure_logger:
        secure_logger.info(
            "Starting Excel file integrity validation",
            context={'file': file_path.name if file_path else 'None'}
        )
    
    try:
        # Basic file checks
        if not file_path.exists():
            return False, "File does not exist", metadata
        
        file_size = file_path.stat().st_size
        metadata['file_size_mb'] = file_size / (1024 * 1024)
        
        # Check file size limits
        if metadata['file_size_mb'] > 100:
            return False, f"File too large: {metadata['file_size_mb']:.1f}MB (max: 100MB)", metadata
        
        if metadata['file_size_mb'] == 0:
            return False, "File is empty", metadata
        
        # Check file extension
        suffix = file_path.suffix.lower()
        if suffix not in ['.xls', '.xlsx', '.xlsm', '.xlsb']:
            return False, f"Unsupported file type: {suffix}", metadata
        
        metadata['file_type'] = suffix
        
        # Try to detect if file is encrypted or corrupted
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
            
            # Log file header analysis
            if secure_logger:
                secure_logger.debug(
                    "Analyzing file header",
                    context={
                        'header_bytes': header.hex(),
                        'header_length': len(header)
                    }
                )
                
            # Check for ZIP signature (modern Excel)
            if header.startswith(b'PK'):
                metadata['file_type'] = 'xlsx/xlsm'
                # Try to open as ZIP to check integrity
                import zipfile
                try:
                    with zipfile.ZipFile(file_path, 'r') as zf:
                        # Check if it's a valid Excel ZIP structure
                        if '[Content_Types].xml' not in zf.namelist():
                            metadata['is_corrupted'] = True
                            return False, "File appears to be corrupted (invalid Excel structure)", metadata
                except zipfile.BadZipFile:
                    metadata['is_corrupted'] = True
                    return False, "File appears to be corrupted (invalid ZIP structure)", metadata
                    
            # Check for OLE signature (old Excel)
            elif header.startswith(b'\\xd0\\xcf\\x11\\xe0\\xa1\\xb1\\x1a\\xe1'):
                metadata['file_type'] = 'xls'
                # Check if encrypted using olefile if available
                try:
                    import olefile
                    if olefile.isOleFile(file_path):
                        with olefile.OleFileIO(file_path) as ole:
                            # Check for encryption
                            if ole.exists('EncryptionInfo') or ole.exists('EncryptedPackage'):
                                metadata['is_encrypted'] = True
                                return False, "File is password protected", metadata
                            
                            # Check for macros
                            if ole.exists('Macros') or ole.exists('VBA'):
                                metadata['has_macros'] = True
                except ImportError:
                    # olefile not available, skip advanced checks
                    pass
                except Exception as e:
                    logger.warning(f"Error checking OLE file: {e}")
            else:
                # Unknown file format
                metadata['is_corrupted'] = True
                return False, "File format not recognized as valid Excel", metadata
                
        except Exception as e:
            logger.error(f"Error reading file header: {e}")
            metadata['is_corrupted'] = True
            return False, "Cannot read file - it may be corrupted", metadata
        
        # Try to get sheet count
        try:
            if suffix == '.xlsx':
                excel_file = pd.ExcelFile(file_path, engine='openpyxl')
            elif suffix == '.xls':
                excel_file = pd.ExcelFile(file_path, engine='xlrd')
            else:
                excel_file = pd.ExcelFile(file_path)
            
            metadata['sheet_count'] = len(excel_file.sheet_names)
            
            if metadata['sheet_count'] == 0:
                return False, "No sheets found in Excel file", metadata
                
        except Exception as e:
            # Can't read sheet info, but might still be valid
            logger.warning(f"Cannot read sheet information: {e}")
            metadata['sheet_count'] = -1
        
        return True, None, metadata
        
    except Exception as e:
        logger.error(f"Unexpected error validating file: {e}")
        return False, f"Validation error: {str(e)}", metadata


@contextmanager
def safe_excel_read(file_path: Path, max_memory_mb: float = 500):
    """
    Context manager for safe Excel file reading with memory monitoring.
    
    Args:
        file_path: Path to Excel file
        max_memory_mb: Maximum memory usage allowed in MB
        
    Yields:
        ExcelFile object
    """
    # Check available memory before reading
    memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
    available_memory = psutil.virtual_memory().available / (1024 * 1024)
    
    # Log memory status
    if secure_logger:
        secure_logger.debug(
            "Memory check before Excel read",
            context={
                'file': file_path.name,
                'process_memory_mb': memory_before,
                'available_memory_mb': available_memory,
                'required_memory_mb': max_memory_mb
            }
        )
    
    if available_memory < max_memory_mb:
        raise MemoryError(f"Insufficient memory: {available_memory:.1f}MB available, {max_memory_mb:.1f}MB required")
    
    excel_file = None
    temp_file = None
    
    try:
        # For very large files, create a temporary copy to avoid locking
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 50:  # Large file
            logger.info(f"Creating temporary copy for large file ({file_size_mb:.1f}MB)")
            temp_file = tempfile.NamedTemporaryFile(suffix=file_path.suffix, delete=False)
            temp_file.close()
            
            # Copy file in chunks to avoid memory spike
            with open(file_path, 'rb') as src, open(temp_file.name, 'wb') as dst:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
            
            read_path = Path(temp_file.name)
        else:
            read_path = file_path
        
        # Open Excel file
        if read_path.suffix.lower() == '.xlsx':
            excel_file = pd.ExcelFile(read_path, engine='openpyxl')
        elif read_path.suffix.lower() == '.xls':
            excel_file = pd.ExcelFile(read_path, engine='xlrd')
        else:
            excel_file = pd.ExcelFile(read_path)
        
        # Monitor memory usage
        memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_used = memory_after - memory_before
        
        # Log memory usage
        if secure_logger:
            secure_logger.info(
                "Excel file opened successfully",
                context={
                    'file': file_path.name,
                    'memory_used_mb': memory_used,
                    'memory_limit_mb': max_memory_mb,
                    'within_limit': memory_used <= max_memory_mb,
                    'used_temp_file': temp_file is not None
                }
            )
        
        if memory_used > max_memory_mb:
            logger.warning(f"Memory usage exceeded limit: {memory_used:.1f}MB used")
        
        yield excel_file
        
    finally:
        # Clean up
        cleanup_success = True
        if excel_file is not None:
            excel_file.close()
        
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                cleanup_success = False
                logger.warning(f"Failed to delete temporary file: {e}")
        
        # Log cleanup
        if secure_logger:
            memory_final = psutil.Process().memory_info().rss / (1024 * 1024)
            secure_logger.debug(
                "Excel read context cleanup",
                context={
                    'cleanup_success': cleanup_success,
                    'memory_released_mb': memory_after - memory_final if 'memory_after' in locals() else 0
                }
            )


def batch_validate_files(file_paths: List[Path]) -> Dict[Path, Tuple[bool, Optional[str]]]:
    """
    Validate multiple Excel files for batch processing.
    
    Args:
        file_paths: List of file paths to validate
        
    Returns:
        Dictionary mapping file paths to validation results
    """
    results = {}
    
    for file_path in file_paths:
        is_valid, error_msg, metadata = validate_excel_file_integrity(file_path)
        
        if is_valid:
            # Additional checks for batch processing
            if metadata['file_size_mb'] > 50:
                results[file_path] = (False, "File too large for batch processing (max 50MB)")
            elif metadata['has_macros']:
                results[file_path] = (False, "Files with macros not supported in batch mode")
            else:
                results[file_path] = (True, None)
        else:
            results[file_path] = (is_valid, error_msg)
    
    return results