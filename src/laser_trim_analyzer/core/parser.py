"""
Excel file parser for Laser Trim Analyzer v3.

Simplified from v2's excel_utils.py (~1000 lines -> ~300 lines).
Handles both System A and System B file formats.
"""

import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

import pandas as pd
import numpy as np

from laser_trim_analyzer.core.models import SystemType, FileMetadata
from laser_trim_analyzer.utils.constants import (
    SYSTEM_A_COLUMNS, SYSTEM_B_COLUMNS,
    SYSTEM_A_CELLS, SYSTEM_B_CELLS,
    SYSTEM_A_IDENTIFIER, SYSTEM_B_IDENTIFIERS,
    EXCEL_EXTENSIONS,
    FINAL_TEST_SHEET_PATTERNS, FINAL_TEST_ROUT_PREFIX,
    TRIM_FILE_INDICATORS,
)

logger = logging.getLogger(__name__)


class ExcelParser:
    """
    Parser for laser trim Excel files.

    Handles:
    - System detection (A vs B)
    - Data extraction (positions, errors, limits)
    - Metadata extraction (model, serial, resistance)
    - Multi-track support for System A
    """

    def __init__(self):
        pass  # No cache needed - file opened once per parse

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse an Excel file and extract all data.

        Args:
            file_path: Path to Excel file

        Returns:
            Dictionary with:
            - metadata: FileMetadata object
            - tracks: List of track data dictionaries
            - file_hash: SHA256 hash for deduplication
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in EXCEL_EXTENSIONS:
            raise ValueError(f"Not an Excel file: {file_path}")

        logger.info(f"Parsing file: {file_path.name}")

        # Calculate file hash for deduplication (separate read, that's fine)
        file_hash = self._calculate_hash(file_path)

        # Single file open for all Excel operations - prevents file handle leaks
        with pd.ExcelFile(file_path) as xl:
            sheet_names = xl.sheet_names

            # Detect system type from sheet names (no file read needed)
            system_type = self._detect_system_from_sheets(sheet_names)
            logger.debug(f"Detected system: {system_type.value}")

            # Check for multi-track from sheet names (no file read needed)
            has_multi_tracks = self._has_multiple_tracks_from_sheets(sheet_names, system_type)

            # Extract test date (needs xl for fallback to cell reading)
            test_date = self._extract_test_date(xl, file_path, system_type)

            # Extract track data (needs xl)
            tracks = self._extract_tracks(xl, file_path, system_type)

        # Build metadata after file is closed
        metadata = self._build_metadata(file_path, system_type, has_multi_tracks, test_date)

        return {
            "metadata": metadata,
            "tracks": tracks,
            "file_hash": file_hash,
        }

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _detect_system_from_sheets(self, sheet_names: List[str]) -> SystemType:
        """
        Detect whether file is System A or System B from sheet names.

        System A: Has sheets like "SEC1 TRK1 0", "SEC1 TRK1 TRM"
        System B: Has sheets like "test", "Trim 1", "Lin Error"
        """
        # Check for System A patterns
        for sheet in sheet_names:
            if SYSTEM_A_IDENTIFIER in sheet:
                return SystemType.A

        # Check for System B patterns
        for identifier in SYSTEM_B_IDENTIFIERS:
            if any(identifier.lower() in s.lower() for s in sheet_names):
                return SystemType.B

        logger.warning(f"Could not detect system type from sheets, defaulting to B")
        return SystemType.B

    def _build_metadata(
        self, file_path: Path, system_type: SystemType,
        has_multi_tracks: bool, test_date: Optional[datetime]
    ) -> FileMetadata:
        """Build file metadata from already-extracted values."""
        # Parse filename for model and serial
        model, serial = self._parse_filename(file_path.name)

        # Get file modification time as fallback
        file_stat = file_path.stat()
        file_mod_date = datetime.fromtimestamp(file_stat.st_mtime)

        # Use test_date as primary file_date, fall back to file modification date
        # This ensures the date shown in the app reflects when the test was done, not when the file was copied/modified
        file_date = test_date if test_date else file_mod_date

        return FileMetadata(
            filename=file_path.name,
            file_path=file_path,
            file_date=file_date,
            test_date=test_date,
            model=model,
            serial=serial,
            system=system_type,
            has_multi_tracks=has_multi_tracks,
        )

    def _parse_filename(self, filename: str) -> Tuple[str, str]:
        """
        Parse model and serial from filename.

        Common patterns:
        - "8340-1_SN12345.xls"
        - "8340-3_AB12345.xls"
        - "Model_Serial_Date.xlsx"
        - "12345_8340-1.xls"
        - "8877_5_deg_1_TEST DATA_12-2-2024.xls" (with degree designator)
        - "1844202_10_TA_Test Data_11-22-2024.xls"
        - "8444-shop0_12-7-2021 10-14-41 AM.xlsx" (shop serial, not model suffix)
        - "8340sn209.xls" (concatenated model+serial)

        Model numbers can have suffixes like 8340-1, 8340-3, etc.
        Handles degree designators (e.g., "5_deg", "10_deg") which should NOT be the serial.
        """
        name = Path(filename).stem

        # Split only on underscores and spaces, NOT hyphens (to preserve model suffixes like 8340-1)
        parts = re.split(r'[_\s]+', name)

        # Handle concatenated model+serial (e.g., "8340sn209" with no separator)
        if len(parts) == 1:
            concat_match = re.match(r'^(\d{4,})[sS][nN](\d+[a-zA-Z]?)$', parts[0])
            if concat_match:
                return concat_match.group(1), concat_match.group(2)

        if len(parts) >= 2:
            model = "Unknown"
            serial = "Unknown"

            # First pass: find the model (4+ digit number, possibly with hyphen suffix)
            for i, part in enumerate(parts):
                # Model pattern: starts with 4+ digits, may have hyphen suffix
                # Examples: 8340, 8340-1, 7063-A, 5409A, 7953-1A, 8711-S
                model_match = re.match(r'^(\d{4,}[A-Za-z]?)(?:-([A-Za-z0-9]+))?$', part)
                if model_match:
                    base, suffix = model_match.group(1), model_match.group(2)
                    if suffix is None:
                        # No hyphen suffix (e.g., "8340", "5409A")
                        model = part
                        break
                    elif self._is_valid_model_suffix(suffix):
                        # Valid suffix like "1", "1A", "CT", "S"
                        model = part
                        break
                    else:
                        # Invalid suffix (e.g., "shop0") - base is model, suffix is serial
                        model = base
                        serial = suffix
                        break

            # Second pass: find the serial (only if not already found from suffix split)
            if serial == "Unknown":
                # Skip parts that are: the model, degree designators, known keywords, dates
                skip_keywords = {'test', 'data', 'deg', 'ta', 'tb', 'trimmed', 'correct', 'scrap', 'cut', 'wiper', 'path', 'am', 'pm'}

                for i, part in enumerate(parts):
                    part_lower = part.lower()

                    # Skip the model itself (check both full part and base model)
                    if part == model:
                        continue

                    # Skip the original unsplit part that contained model-suffix
                    # (e.g., "8444-shop0" when model is "8444")
                    if '-' in part and part.startswith(model + '-'):
                        continue

                    # Skip known keywords
                    if part_lower in skip_keywords:
                        continue

                    # Skip if this is part of a degree designator (e.g., "5" followed by "deg")
                    # Check if next part is "deg"
                    if i + 1 < len(parts) and parts[i + 1].lower() == 'deg':
                        continue

                    # Skip the "deg" part itself
                    if part_lower == 'deg':
                        continue

                    # Skip date patterns (e.g., "12-2-2024", "11-22-2024")
                    if re.match(r'^\d{1,2}-\d{1,2}-\d{4}$', part):
                        continue

                    # Skip time patterns (e.g., "4-05", "9-08")
                    if re.match(r'^\d{1,2}-\d{2}$', part):
                        continue

                    # Serial pattern: letter prefix + digits (SN12345, AB12345, TA, TB)
                    if re.match(r'^[A-Z]{1,3}\d+', part, re.IGNORECASE):
                        serial = part
                        break

                    # Pure numeric serial (only if model is already found)
                    elif re.match(r'^\d+$', part) and model != "Unknown":
                        serial = part
                        break

                # Fallback: if serial still unknown, use first non-model, non-keyword part
                if serial == "Unknown" and len(parts) > 1:
                    for i, part in enumerate(parts):
                        part_lower = part.lower()
                        if part != model and part_lower not in skip_keywords:
                            if '-' in part and part.startswith(model + '-'):
                                continue
                            # Skip degree designator pattern
                            if i + 1 < len(parts) and parts[i + 1].lower() == 'deg':
                                continue
                            if part_lower == 'deg':
                                continue
                            # Skip date/time patterns
                            if re.match(r'^\d{1,2}-\d{1,2}(-\d{4})?$', part):
                                continue
                            serial = part
                            break

            return model, serial

        return name, "Unknown"

    @staticmethod
    def _is_valid_model_suffix(suffix: str) -> bool:
        """
        Validate whether a hyphenated suffix is a legitimate model variant.

        Valid suffixes (from production data):
        - Single digit: "1", "3", "4"
        - Digit + letter: "1A"
        - Single letter: "S", "A"
        - 2-letter code: "CT"
        - "outer" for backwards compatibility

        Invalid suffixes:
        - "shop0", "shop", "shoptest" - location/serial identifiers
        """
        # Reject "shop" prefixed suffixes
        if suffix.lower().startswith('shop'):
            return False

        # Valid: 1-3 characters of digits and/or letters
        if re.match(r'^[A-Za-z0-9]{1,3}$', suffix):
            return True

        # Allow "outer" for backwards compatibility (8736-outer)
        if suffix.lower() == 'outer':
            return True

        return False

    def _extract_test_date(
        self, xl: pd.ExcelFile, file_path: Path, system_type: SystemType
    ) -> Optional[datetime]:
        """
        Try to extract the test/trim date.

        Priority:
        1. Date from filename (most reliable for System B files: MODEL_SERIAL_TA_Test Data_M-D-YYYY_...)
        2. Date from Excel cell (for files that store it inside)
        """
        # First, try to extract from filename (most common for System B)
        filename_date = self._extract_date_from_filename(file_path.name)
        if filename_date:
            return filename_date

        # Fall back to searching Excel file (using already-open xl object)
        try:
            first_sheet = xl.sheet_names[0]
            df = pd.read_excel(xl, sheet_name=first_sheet, header=None, nrows=5)

            for row in range(min(5, len(df))):
                for col in range(min(5, len(df.columns))):
                    value = df.iloc[row, col]
                    if isinstance(value, datetime):
                        return value
                    if isinstance(value, str):
                        # Try to parse date string
                        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
                            try:
                                return datetime.strptime(value, fmt)
                            except ValueError:
                                continue

            del df  # Free memory immediately
            return None

        except Exception as e:
            logger.debug(f"Could not extract test date from Excel: {e}")
            return None

    def _extract_date_from_filename(self, filename: str) -> Optional[datetime]:
        """
        Extract date from filename.

        Handles patterns like:
        - 1844202_10_TA_Test Data_11-22-2024_9-08 AMTrimmed Correct.xls
        - 8887_14_TA_Test Data_5-5-2025_9-58 AMScrap_Cut in wiper path.xls
        """
        # Look for date pattern M-D-YYYY or MM-DD-YYYY
        date_match = re.search(r'(\d{1,2})-(\d{1,2})-(\d{4})', filename)
        if date_match:
            month, day, year = date_match.groups()
            try:
                return datetime.strptime(f'{month}-{day}-{year}', '%m-%d-%Y')
            except ValueError:
                pass

        # Also try YYYY-MM-DD format
        date_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', filename)
        if date_match:
            year, month, day = date_match.groups()
            try:
                return datetime.strptime(f'{year}-{month}-{day}', '%Y-%m-%d')
            except ValueError:
                pass

        return None

    def _has_multiple_tracks_from_sheets(self, sheet_names: List[str], system_type: SystemType) -> bool:
        """Check if file has multiple tracks from sheet names."""
        if system_type == SystemType.B:
            return False  # System B files are single-track

        # Count TRK sheets
        trk_sheets = [s for s in sheet_names if "TRK" in s.upper()]
        # Multiple tracks if we have TRK1 and TRK2
        return len(set(re.findall(r'TRK\d', ' '.join(trk_sheets), re.I))) > 1

    def _extract_tracks(
        self, xl: pd.ExcelFile, file_path: Path, system_type: SystemType
    ) -> List[Dict[str, Any]]:
        """Extract data for all tracks in the file."""
        if system_type == SystemType.A:
            return self._extract_system_a_tracks(xl, file_path)
        else:
            return self._extract_system_b_tracks(xl, file_path)

    def _extract_system_a_tracks(self, xl: pd.ExcelFile, file_path: Path) -> List[Dict[str, Any]]:
        """Extract tracks from System A file."""
        tracks = []

        # Find track sheets
        for track_id in ["TRK1", "TRK2"]:
            # Find untrimmed sheet
            untrimmed_sheet = None
            trimmed_sheet = None

            for sheet in xl.sheet_names:
                sheet_upper = sheet.upper()
                if track_id in sheet_upper:
                    if " 0" in sheet or "_0" in sheet:
                        untrimmed_sheet = sheet
                    elif "TRM" in sheet_upper:
                        trimmed_sheet = sheet

            # Use trimmed sheet if available, otherwise use untrimmed
            # (allows processing of files where trimming was aborted/incomplete)
            data_sheet = trimmed_sheet or untrimmed_sheet
            if data_sheet:
                track_data = self._extract_track_data(
                    xl, file_path, data_sheet, untrimmed_sheet if trimmed_sheet else None,
                    SystemType.A, track_id
                )
                if track_data:
                    tracks.append(track_data)

        return tracks

    def _extract_system_b_tracks(self, xl: pd.ExcelFile, file_path: Path) -> List[Dict[str, Any]]:
        """Extract tracks from System B file."""
        tracks = []

        # Find untrimmed and trimmed sheets
        untrimmed_sheet = None
        lin_error_sheet = None  # Preferred: final linearity error after all trims
        trim_sheets = []  # Fallback: intermediate trim sheets

        for sheet in xl.sheet_names:
            sheet_lower = sheet.lower()
            if sheet_lower == "test":
                untrimmed_sheet = sheet
            # "Lin Error" is the preferred final trimmed data sheet
            elif sheet_lower == "lin error":
                lin_error_sheet = sheet
            # "Trim 1", "Trim 2", etc. are intermediate trim passes (fallback only)
            elif sheet_lower.startswith("trim ") and sheet_lower[5:].isdigit():
                trim_sheets.append(sheet)

        # Priority: Lin Error > highest Trim N > Trim 1
        if lin_error_sheet:
            trimmed_sheet = lin_error_sheet
        elif trim_sheets:
            # Sort trim sheets and use the highest number (most recent trim pass)
            trim_sheets.sort(key=lambda s: int(s.lower().split()[-1]))
            trimmed_sheet = trim_sheets[-1]  # Use highest trim number
        else:
            trimmed_sheet = None

        if trimmed_sheet:
            track_data = self._extract_track_data(
                xl, file_path, trimmed_sheet, untrimmed_sheet,
                SystemType.B, "default"
            )
            if track_data:
                tracks.append(track_data)

        return tracks

    def _extract_track_data(
        self,
        xl: pd.ExcelFile,
        file_path: Path,
        trimmed_sheet: str,
        untrimmed_sheet: Optional[str],
        system_type: SystemType,
        track_id: str
    ) -> Optional[Dict[str, Any]]:
        """Extract data for a single track."""
        try:
            columns = SYSTEM_A_COLUMNS if system_type == SystemType.A else SYSTEM_B_COLUMNS
            cells = SYSTEM_A_CELLS if system_type == SystemType.A else SYSTEM_B_CELLS

            # Read trimmed data from already-open xl object
            df = pd.read_excel(xl, sheet_name=trimmed_sheet, header=None)

            # Find data start row (first row with numeric position data)
            data_start = self._find_data_start(df, columns["position"])

            if data_start is None:
                logger.warning(f"Could not find data start in {trimmed_sheet}")
                return None

            # Extract columns
            # Position column should stop at first NaN
            positions = self._get_column_data(df, columns["position"], data_start)

            if not positions:
                logger.warning(f"No position data in {trimmed_sheet}")
                return None

            # Error column may have leading NaN values (no error at those positions)
            # Use allow_nan=True to handle this, then trim to match positions length
            errors = self._get_column_data(df, columns["error"], data_start, allow_nan=True)

            # Trim errors to match positions length (or pad if needed)
            if len(errors) > len(positions):
                errors = errors[:len(positions)]
            elif len(errors) < len(positions):
                # Pad with zeros if error column is shorter
                errors = errors + [0.0] * (len(positions) - len(errors))

            if not errors:
                logger.warning(f"No error data in {trimmed_sheet}")
                return None

            # Extract limit columns - use special method that preserves NaN values
            # NaN means "no spec limit" at that position (unlimited)
            num_data_points = len(positions)
            upper_limits = self._get_limit_column_data(df, columns["upper_limit"], data_start, num_data_points)
            lower_limits = self._get_limit_column_data(df, columns["lower_limit"], data_start, num_data_points)

            # Extract resistance and unit length from DataFrame already in memory
            trimmed_resistance = self._get_cell_from_df(df, cells["trimmed_resistance"])
            unit_length = self._get_cell_from_df(df, cells["unit_length"])

            # Extract untrimmed data if available
            untrimmed_positions = None
            untrimmed_errors = None
            untrimmed_resistance = None

            if untrimmed_sheet:
                try:
                    df_untrim = pd.read_excel(xl, sheet_name=untrimmed_sheet, header=None)
                    untrim_start = self._find_data_start(df_untrim, columns["position"])
                    if untrim_start is not None:
                        untrimmed_positions = self._get_column_data(df_untrim, columns["position"], untrim_start)
                        untrimmed_errors = self._get_column_data(df_untrim, columns["error"], untrim_start)
                    untrimmed_resistance = self._get_cell_from_df(df_untrim, cells["untrimmed_resistance"])
                    del df_untrim  # Free memory immediately
                except Exception as e:
                    logger.debug(f"Could not read untrimmed sheet: {e}")

            del df  # Free memory immediately

            # Calculate travel length
            travel_length = max(positions) - min(positions) if positions else 0.0

            # Calculate linearity spec from limits
            # Note: Some positions may have NaN limits intentionally (no spec for that portion of travel)
            # We preserve NaN values - they mean "unlimited" at that position
            linearity_spec = self._calculate_linearity_spec(upper_limits, lower_limits)

            return {
                "track_id": track_id,
                "positions": positions,
                "errors": errors,
                "upper_limits": upper_limits,
                "lower_limits": lower_limits,
                "untrimmed_positions": untrimmed_positions,
                "untrimmed_errors": untrimmed_errors,
                "travel_length": travel_length,
                "linearity_spec": linearity_spec,
                "unit_length": unit_length,
                "untrimmed_resistance": untrimmed_resistance,
                "trimmed_resistance": trimmed_resistance,
            }

        except Exception as e:
            logger.error(f"Error extracting track data from {trimmed_sheet}: {e}")
            return None

    def _find_data_start(self, df: pd.DataFrame, position_col: int) -> Optional[int]:
        """Find the first row with numeric data."""
        for i in range(min(20, len(df))):
            try:
                value = df.iloc[i, position_col]
                if pd.notna(value) and isinstance(value, (int, float)):
                    return i
            except (IndexError, ValueError):
                continue
        return None

    def _get_column_data(self, df: pd.DataFrame, col_idx: int, start_row: int, allow_nan: bool = False) -> List[float]:
        """
        Extract numeric data from a column.

        Args:
            df: DataFrame to extract from
            col_idx: Column index
            start_row: Starting row
            allow_nan: If True, replace NaN with 0.0 instead of stopping. Used for error columns
                       that may have leading NaN values (no error data for first few positions)

        Returns:
            List of float values
        """
        data = []
        consecutive_nan = 0
        for i in range(start_row, len(df)):
            try:
                value = df.iloc[i, col_idx]
                if pd.notna(value):
                    data.append(float(value))
                    consecutive_nan = 0
                elif allow_nan:
                    # For error columns, NaN typically means 0 error (within spec, no trim needed)
                    data.append(0.0)
                    consecutive_nan += 1
                    # If we have too many consecutive NaN, we've hit the end of data
                    if consecutive_nan > 10:
                        # Remove the trailing NaN placeholders
                        data = data[:-consecutive_nan]
                        break
                else:
                    break  # Stop at first empty cell
            except (ValueError, TypeError):
                break
        return data

    def _get_limit_column_data(self, df: pd.DataFrame, col_idx: int, start_row: int, num_rows: int) -> List[Optional[float]]:
        """
        Extract limit data from a column, preserving NaN values.

        NaN in spec limit columns means "no specification" (unlimited) at that position.
        This is different from position/error columns where NaN means end of data.
        """
        data = []
        for i in range(start_row, start_row + num_rows):
            if i >= len(df):
                data.append(None)
                continue
            try:
                value = df.iloc[i, col_idx]
                if pd.notna(value):
                    data.append(float(value))
                else:
                    data.append(None)  # Preserve NaN as None (no limit at this position)
            except (ValueError, TypeError):
                data.append(None)
        return data

    def _get_cell_from_df(self, df: pd.DataFrame, cell_ref: str) -> Optional[float]:
        """
        Get cell value from DataFrame already in memory - no file read.

        Args:
            df: DataFrame to extract from
            cell_ref: Excel-style cell reference (e.g., "A1", "B5")

        Returns:
            Float value or None
        """
        try:
            match = re.match(r'^([A-Z]+)(\d+)$', cell_ref.upper())
            if not match:
                return None

            col_letters, row_num = match.groups()

            # Convert column letters to index
            col_idx = 0
            for char in col_letters:
                col_idx = col_idx * 26 + (ord(char) - ord('A')) + 1
            col_idx -= 1

            row_idx = int(row_num) - 1

            if row_idx < len(df) and col_idx < len(df.columns):
                value = df.iloc[row_idx, col_idx]
                if pd.notna(value):
                    return float(value)

            return None

        except Exception:
            return None

    def _calculate_linearity_spec(
        self, upper_limits: List[float], lower_limits: List[float]
    ) -> float:
        """Calculate linearity spec from limits."""
        if not upper_limits or not lower_limits:
            return 0.01  # Default

        # Filter valid values
        valid_upper = [u for u in upper_limits if u is not None and not np.isnan(u)]
        valid_lower = [l for l in lower_limits if l is not None and not np.isnan(l)]

        if valid_upper and valid_lower:
            # Spec is half the average band width
            avg_upper = np.mean(valid_upper)
            avg_lower = np.mean(valid_lower)
            return (avg_upper - avg_lower) / 2

        return 0.01


# =============================================================================
# File Type Detection - Distinguish Trim files from Final Test files
# =============================================================================

def detect_file_type(file_path: Path) -> str:
    """
    Detect whether a file is a 'trim' or 'final_test' file.

    Detection logic:
    1. Check filename for Rout_ prefix (Format 2 final test)
    2. Check filename for trim indicators (_Trimmed, etc.)
    3. Check sheet names for trim patterns (SEC1 TRK, Lin Error)
    4. Check sheet names for final test patterns (Data Table)

    Args:
        file_path: Path to Excel file

    Returns:
        'trim' or 'final_test'
    """
    file_path = Path(file_path)
    filename = file_path.name

    # Check for Rout_ prefix (Format 2 final test)
    if filename.startswith(FINAL_TEST_ROUT_PREFIX):
        logger.debug(f"Detected final_test (Rout_ prefix): {filename}")
        return "final_test"

    # Check filename for trim indicators
    for indicator in TRIM_FILE_INDICATORS:
        if indicator in filename:
            logger.debug(f"Detected trim (filename indicator '{indicator}'): {filename}")
            return "trim"

    # Need to check sheet names - use context manager to prevent file handle leaks
    try:
        with pd.ExcelFile(file_path) as xl:
            sheet_names = xl.sheet_names

            # Check for trim indicators in sheet names
            for sheet in sheet_names:
                for indicator in TRIM_FILE_INDICATORS:
                    if indicator in sheet:
                        logger.debug(f"Detected trim (sheet indicator '{indicator}' in '{sheet}'): {filename}")
                        return "trim"

            # Check for final test indicators in sheet names
            for sheet in sheet_names:
                for pattern in FINAL_TEST_SHEET_PATTERNS:
                    if pattern.lower() in sheet.lower():
                        # Additional check: "Sheet1" alone isn't enough
                        # Look for "Data Table" specifically for final test
                        if pattern == "Sheet1":
                            # Need more evidence - check if Data Table exists
                            if "Data Table" in sheet_names:
                                logger.debug(f"Detected final_test (has Data Table sheet): {filename}")
                                return "final_test"
                        else:
                            logger.debug(f"Detected final_test (sheet pattern '{pattern}'): {filename}")
                            return "final_test"

            # Final heuristic: if file has only "Sheet1" and no trim indicators,
            # check the content structure
            if "Sheet1" in sheet_names and len(sheet_names) <= 3:
                # Could be final test - check for specific cell patterns
                try:
                    df = pd.read_excel(xl, sheet_name="Sheet1", header=None, nrows=5)
                    # Final test format has specific header patterns
                    # Check if row 0 has model info around column L (11)
                    if df.shape[1] > 11:
                        cell_val = df.iloc[0, 11] if pd.notna(df.iloc[0, 11]) else ""
                        if isinstance(cell_val, str) and any(c.isdigit() for c in cell_val):
                            # Has model-like value in expected position
                            logger.debug(f"Detected final_test (content structure): {filename}")
                            del df  # Free memory
                            return "final_test"
                    del df  # Free memory
                except Exception:
                    pass

            # Default to trim (existing behavior)
            logger.debug(f"Defaulting to trim: {filename}")
            return "trim"

    except Exception as e:
        logger.warning(f"Error detecting file type for {filename}: {e}, defaulting to trim")
        return "trim"


def is_final_test_file(file_path: Path) -> bool:
    """
    Quick check if a file is a final test file.

    Args:
        file_path: Path to Excel file

    Returns:
        True if final test file, False otherwise
    """
    return detect_file_type(file_path) == "final_test"


def is_trim_file(file_path: Path) -> bool:
    """
    Quick check if a file is a trim file.

    Args:
        file_path: Path to Excel file

    Returns:
        True if trim file, False otherwise
    """
    return detect_file_type(file_path) == "trim"
