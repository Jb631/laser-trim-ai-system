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

from laser_trim_analyzer.core.constants import (
    SYSTEM_A_COLUMNS, SYSTEM_B_COLUMNS,
    SYSTEM_A_CELLS, SYSTEM_B_CELLS,
    SystemIdentifier
)
from laser_trim_analyzer.core.models import SystemType
from laser_trim_analyzer.core.exceptions import (
    DataExtractionError, SheetNotFoundError, SystemDetectionError
)

logger = logging.getLogger(__name__)


def read_excel_sheet(
        file_path: Path,
        sheet_name: Union[str, int],
        header: Optional[int] = None,
        skiprows: Optional[int] = None
) -> pd.DataFrame:
    """
    Read a specific sheet from an Excel file.

    Args:
        file_path: Path to Excel file
        sheet_name: Name or index of sheet
        header: Row to use as header
        skiprows: Rows to skip at start

    Returns:
        DataFrame with sheet data

    Raises:
        SheetNotFoundError: If sheet not found
    """
    file_path = Path(file_path)

    # First, try to read with default pandas (auto-detects engine)
    try:
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=header,
            skiprows=skiprows
        )
        logger.debug(f"Read sheet '{sheet_name}' from {file_path.name}: {df.shape}")
        return df

    except ValueError as e:
        if "Worksheet" in str(e) and "does not exist" in str(e):
            raise SheetNotFoundError(f"Sheet '{sheet_name}' not found in {file_path.name}")
        # Try with explicit engines if auto-detect failed
        pass
    except Exception as first_error:
        # Try with explicit engines if auto-detect failed
        pass

    # If default failed, try openpyxl explicitly (for xlsx)
    if file_path.suffix.lower() == '.xlsx':
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
        except Exception:
            pass

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
            logger.error("xlrd not installed. Install with: pip install xlrd")
            raise DataExtractionError("Cannot read .xls files - xlrd not installed")
        except Exception as xlrd_error:
            logger.error(f"xlrd failed: {xlrd_error}")
            # Continue to raise the original error

    # If all attempts failed, raise the original error
    logger.error(f"Failed to read sheet '{sheet_name}' from {file_path.name}")
    raise DataExtractionError(f"Failed to read Excel sheet '{sheet_name}' from {file_path.name}")


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

        logger.debug(f"Extracted {cell_ref}={value} from {sheet_name}")
        return value

    except Exception as e:
        logger.warning(f"Failed to extract cell {cell_ref} from {sheet_name}: {e}")
        return default


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
    if system == 'A':
        columns = SYSTEM_A_COLUMNS.copy()
    else:
        columns = SYSTEM_B_COLUMNS.copy()

    # Validate columns exist
    max_col = max(columns.values())
    if df.shape[1] <= max_col:
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

    logger.debug(f"Found data starting at row {data_start_row}")

    # Validate we have numeric data
    if data_start_row >= df.shape[0] - 10:  # Need at least 10 data points
        logger.warning("Insufficient numeric data found")
        return {}

    return columns


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
            logger.debug(f"Sheets in {file_path.name}: {sheet_names}")

            # Check for System A patterns
            system_a_indicators = [
                SystemIdentifier.SYSTEM_A_PATTERN.value,
                "TRK1", "TRK2",
                "SEC1", "SEC2"
            ]

            for sheet in sheet_names:
                if any(indicator in sheet for indicator in system_a_indicators):
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
                logger.info("Detected System B from sheet patterns")
                return SystemType.SYSTEM_B

        # Check filename patterns as fallback
        filename = file_path.stem.upper()

        if any(filename.startswith(prefix) for prefix in ['8340', '834', '8506', '8852']):
            logger.info("Detected System B from filename pattern")
            return SystemType.SYSTEM_B
        elif any(filename.startswith(prefix) for prefix in ['68', '78', '85']):
            logger.info("Detected System A from filename pattern")
            return SystemType.SYSTEM_A

        # Default based on file extension
        if file_path.suffix.lower() == '.xls':
            logger.warning(f"Could not definitively detect system type for {file_path.name}, defaulting to System B for .xls file")
            return SystemType.SYSTEM_B
        else:
            logger.warning(f"Could not definitively detect system type for {file_path.name}, defaulting to System A")
            return SystemType.SYSTEM_A

    except Exception as e:
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