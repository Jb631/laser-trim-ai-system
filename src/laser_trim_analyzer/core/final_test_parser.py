"""
Final Test file parser for Laser Trim Analyzer v3.

Parses post-assembly final test files for comparison with laser trim results.
Supports multiple file formats detected automatically.
"""

import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

import pandas as pd
import numpy as np

from laser_trim_analyzer.utils.constants import (
    FINAL_TEST_FORMAT1_COLUMNS,
    FINAL_TEST_FORMAT2_COLUMNS,
    FINAL_TEST_FORMAT1_METADATA,
    FINAL_TEST_DATA_TABLE_ROWS,
    FINAL_TEST_DATA_TABLE_COLUMNS,
    FINAL_TEST_ROUT_PREFIX,
    EXCEL_EXTENSIONS,
)

logger = logging.getLogger(__name__)


class FinalTestParser:
    """
    Parser for Final Test Excel files.

    Handles:
    - Format detection (Format 1 vs Format 2)
    - Data extraction (positions, errors, electrical angles)
    - Metadata extraction (model, serial, test date)
    - Test results extraction (pass/fail for each test type)
    """

    def __init__(self):
        pass  # No cache needed - file opened once per parse

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a Final Test Excel file and extract all data.

        Args:
            file_path: Path to Final Test Excel file

        Returns:
            Dictionary with:
            - metadata: Dict with model, serial, test_date, etc.
            - tracks: List of track data dictionaries
            - test_results: Dict with pass/fail for each test type
            - file_hash: SHA256 hash for deduplication
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in EXCEL_EXTENSIONS:
            raise ValueError(f"Not an Excel file: {file_path}")

        logger.info(f"Parsing Final Test file: {file_path.name}")

        # Calculate file hash for deduplication (separate file read, that's fine)
        file_hash = self._calculate_hash(file_path)

        # Single file open for all Excel operations - prevents file handle leaks
        with pd.ExcelFile(file_path) as xl:
            # Detect format from sheet names
            format_type = self._detect_format_from_sheets(file_path.name, xl.sheet_names)
            logger.debug(f"Detected Final Test format: {format_type}")

            # Parse according to format - all methods now receive xl object
            if format_type == "format2":
                return self._parse_format2(xl, file_path, file_hash)
            elif format_type == "format3_multitrack":
                return self._parse_format3_multitrack(xl, file_path, file_hash)
            elif format_type == "format4_parameters":
                return self._parse_format4_parameters(xl, file_path, file_hash)
            elif format_type == "format_shop_test":
                return self._parse_format_shop_test(xl, file_path, file_hash)
            else:
                return self._parse_format1(xl, file_path, file_hash)

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _detect_format_from_sheets(self, filename: str, sheet_names: List[str]) -> str:
        """
        Detect which Final Test format the file uses from sheet names.

        Returns:
            'format1', 'format2', 'format3_multitrack', 'format4_parameters', or 'format_shop_test'
        """
        # Check for Rout_ prefix (Format 2)
        if filename.startswith(FINAL_TEST_ROUT_PREFIX):
            return "format2"

        # Format 2 has "Data" and "Charts" sheets
        if "Data" in sheet_names and "Charts" in sheet_names:
            return "format2"

        # Format 3: Multi-track with sheets named A, B, C (must check BEFORE Format 1)
        # These files have A, B, C sheets instead of Sheet1
        if "A" in sheet_names and "Data Table" in sheet_names and "Sheet1" not in sheet_names:
            return "format3_multitrack"

        # Format 4: Parameters sheet format
        if "Parameters" in sheet_names:
            return "format4_parameters"

        # Format 1 has "Sheet1" and/or "Data Table"
        if "Sheet1" in sheet_names or "Data Table" in sheet_names:
            return "format1"

        # Shop test format has "test" sheet with metadata in cols 0-1, data in cols 3-8
        if "test" in sheet_names:
            return "format_shop_test"

        return "format1"

    def _parse_format1(self, xl: pd.ExcelFile, file_path: Path, file_hash: str) -> Dict[str, Any]:
        """
        Parse Format 1 Final Test file (standard format).

        Sheet: Sheet1 - Main data
        Sheet: Data Table - Test results summary
        """
        filename = file_path.name

        # Extract metadata from filename
        metadata = self._extract_metadata_from_filename(filename)

        # Try to get additional metadata from file content
        df = None
        try:
            df = pd.read_excel(xl, sheet_name="Sheet1", header=None)

            # Extract model from cell (around column L/M, row 0)
            if df.shape[1] > 11:
                model_cell = df.iloc[0, 11] if pd.notna(df.iloc[0, 11]) else None
                if model_cell and isinstance(model_cell, str):
                    # Extract model number pattern
                    match = re.search(r'(\d{6,7})', str(model_cell))
                    if match and not metadata.get("model"):
                        metadata["model"] = match.group(1)

            # Extract test datetime from cell (around column N, row 0)
            if df.shape[1] > 13:
                datetime_cell = df.iloc[0, 13] if pd.notna(df.iloc[0, 13]) else None
                if datetime_cell:
                    if isinstance(datetime_cell, datetime):
                        metadata["test_date"] = datetime_cell
                    elif isinstance(datetime_cell, str):
                        try:
                            metadata["test_date"] = pd.to_datetime(datetime_cell)
                        except Exception:
                            pass

        except Exception as e:
            logger.warning(f"Error extracting metadata from content: {e}")
        finally:
            if df is not None:
                del df  # Free memory

        # Extract linearity data
        tracks = self._extract_format1_tracks(xl)

        # Extract test results from Data Table
        test_results = self._extract_test_results(xl)

        return {
            "metadata": metadata,
            "tracks": tracks,
            "test_results": test_results,
            "file_hash": file_hash,
            "format": "format1",
        }

    def _parse_format2(self, xl: pd.ExcelFile, file_path: Path, file_hash: str) -> Dict[str, Any]:
        """
        Parse Format 2 Final Test file (Rout_ prefix files).

        Sheet: Data - Main measurement data
        Sheet: Charts - Visualization data
        """
        filename = file_path.name

        # Extract metadata from filename
        # Format: Rout_1091701_sn1695a_vo.xls
        metadata = self._extract_metadata_from_filename(filename)

        # Try to get additional metadata from Data sheet header
        df = None
        try:
            df = pd.read_excel(xl, sheet_name="Data", header=None, nrows=2)

            # First row often has model/test info
            if df.shape[1] > 3:
                # Check for model number in various cells
                for col in range(min(10, df.shape[1])):
                    cell_val = df.iloc[0, col] if pd.notna(df.iloc[0, col]) else None
                    if cell_val:
                        match = re.search(r'(\d{6,7})', str(cell_val))
                        if match and not metadata.get("model"):
                            metadata["model"] = match.group(1)
                            break

        except Exception as e:
            logger.warning(f"Error extracting Format 2 metadata: {e}")
        finally:
            if df is not None:
                del df  # Free memory

        # Extract linearity data
        tracks = self._extract_format2_tracks(xl)

        # Format 2 may not have detailed test results
        test_results = {
            "linearity_pass": None,
            "resistance_pass": None,
            "electrical_angle_pass": None,
            "hysteresis_pass": None,
            "phasing_pass": None,
        }

        return {
            "metadata": metadata,
            "tracks": tracks,
            "test_results": test_results,
            "file_hash": file_hash,
            "format": "format2",
        }

    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        Extract model, serial, and date from filename.

        Patterns:
        - 1081313-sn108_3-16-2011_12-17 PM.xls
        - Rout_1091701_sn1695a_vo.xls
        - 1844202-sn1004a_7-27-2022_1-26 PM.xls
        """
        metadata = {
            "filename": filename,
            "model": None,
            "serial": None,
            "file_date": None,
        }

        # Remove extension
        base = Path(filename).stem

        # Strip common prefixes like "Copy of "
        if base.lower().startswith("copy of "):
            base = base[8:]  # Remove "Copy of "

        # Handle Rout_ prefix format
        if base.startswith("Rout_"):
            base = base[5:]  # Remove "Rout_"
            # Pattern: 1091701_sn1695a_vo
            parts = base.split("_")
            if len(parts) >= 2:
                metadata["model"] = parts[0]
                # Serial is after 'sn'
                for part in parts[1:]:
                    if part.lower().startswith("sn"):
                        metadata["serial"] = part[2:] if len(part) > 2 else part
                        break
            return metadata

        # Standard format: 1081313-sn108_3-16-2011_12-17 PM
        # Or: 1844202-sn1004a_7-27-2022_1-26 PM
        # Or: 8340-1-sn470_5-30-2025_1-52 PM (model with suffix)

        # Try to extract model number with optional suffix (e.g., 8340-1)
        # Pattern: digits followed by optional hyphen-digits, before the -sn serial pattern
        model_match = re.match(r'^(\d+(?:-\d+)?)', base)
        if model_match:
            metadata["model"] = model_match.group(1)

        # Try to extract serial number (snXXX pattern)
        # Handle both hyphen and underscore separators: -sn108 or _SN16
        sn_match = re.search(r'[-_]sn([a-zA-Z0-9]+)', base, re.IGNORECASE)
        if sn_match:
            metadata["serial"] = sn_match.group(1)

        # Try to extract date - multiple patterns:
        # Pattern 1: M-D-YYYY (e.g., 3-16-2011)
        date_match = re.search(r'_(\d{1,2})-(\d{1,2})-(\d{4})', base)
        if date_match:
            try:
                month = int(date_match.group(1))
                day = int(date_match.group(2))
                year = int(date_match.group(3))
                metadata["file_date"] = datetime(year, month, day)
            except ValueError:
                pass
        else:
            # Pattern 2: MMDDYY (e.g., 050225 = May 2, 2025)
            date_match2 = re.search(r'_(\d{6})(?:[_.]|$)', base)
            if date_match2:
                try:
                    date_str = date_match2.group(1)
                    month = int(date_str[0:2])
                    day = int(date_str[2:4])
                    year = int(date_str[4:6])
                    # Handle 2-digit year
                    if year < 50:
                        year += 2000
                    else:
                        year += 1900
                    metadata["file_date"] = datetime(year, month, day)
                except ValueError:
                    pass

        return metadata

    def _extract_format1_tracks(self, xl: pd.ExcelFile) -> List[Dict[str, Any]]:
        """
        Extract track data from Format 1 file.

        Sheet1 standard column layout (verified against model 8340-1):
        - Column A (0): Measured Volts - actual output voltage
        - Column B (1): Index - sample number
        - Column C (2): Theory Volts - expected/ideal value
        - Column D (3): Voltage Error - pre-calculated error (Measured - Theory)
        - Column E (4): Electrical Angle - X-axis for linearity curve
            - Linear pots (8340-1): 0 to ~0.61 inches
            - Rotary pots (2475): -170° to +170°
        - Column G (6): Upper Spec Limit
        - Column H (7): Lower Spec Limit

        Uses pre-calculated error from Column D when available.
        Falls back to calculating error if Column D is empty.
        """
        tracks = []
        cols = FINAL_TEST_FORMAT1_COLUMNS.copy()
        df = None

        try:
            df = pd.read_excel(xl, sheet_name="Sheet1", header=None)

            # Helper function to check if value is numeric (handles numpy types)
            def is_numeric(val):
                return pd.notna(val) and np.issubdtype(type(val), np.number)

            # Find data start row (skip any header rows)
            data_start = 0
            for i in range(min(10, len(df))):
                # Look for numeric data in column A (measured) and column B (index)
                if df.shape[1] > 1:
                    val_a = df.iloc[i, 0]
                    val_b = df.iloc[i, 1]
                    if is_numeric(val_a) and is_numeric(val_b):
                        data_start = i
                        break

            # Detect format variation: some files have Col E = error duplicate or empty
            # Need to find the actual position column
            position_col = cols["electrical_angle"]  # Default: Col E (4)

            if df.shape[1] > 5:
                # Check if Col E is empty (NaN), duplicates Col D, or is constant
                col_e_empty_count = 0
                similar_count = 0
                col_e_values = []
                for i in range(data_start, min(data_start + 10, len(df))):
                    col_d = df.iloc[i, cols["error"]]
                    col_e = df.iloc[i, cols["electrical_angle"]]

                    if not is_numeric(col_e):
                        col_e_empty_count += 1
                    else:
                        col_e_values.append(float(col_e))
                        if is_numeric(col_d):
                            if abs(float(col_d) - float(col_e)) < 0.0001:
                                similar_count += 1

                # Check if Col E is mostly constant (e.g., all zeros)
                # A valid position column should have varying values
                constant_count = 0
                if len(col_e_values) >= 5:
                    ref = col_e_values[0]
                    constant_count = sum(1 for v in col_e_values if abs(v - ref) < 0.0001)

                # Col E is unusable if mostly empty, duplicates error, or constant
                col_e_unusable = (col_e_empty_count >= 5 or similar_count >= 5
                                  or constant_count >= 7)

                if col_e_unusable:
                    # Col E is empty or duplicates error - need to find position elsewhere
                    # Try Col F (5) first, but only if it's not also a duplicate of error
                    col_f_valid = 0
                    col_f_duplicates_error = 0
                    for i in range(data_start, min(data_start + 10, len(df))):
                        if df.shape[1] > 5:
                            col_f = df.iloc[i, 5]
                            col_d = df.iloc[i, cols["error"]]
                            if is_numeric(col_f):
                                col_f_valid += 1
                                if is_numeric(col_d) and abs(float(col_f) - float(col_d)) < 0.0001:
                                    col_f_duplicates_error += 1

                    if col_f_valid >= 5 and col_f_duplicates_error < 3:
                        position_col = 5
                        logger.debug(f"Format B: Using Col F for position")
                    else:
                        # Search other columns for position-like data (increasing values)
                        for test_col in range(8, min(df.shape[1], 16)):
                            vals = []
                            for i in range(data_start, min(data_start + 10, len(df))):
                                if is_numeric(df.iloc[i, test_col]):
                                    vals.append(float(df.iloc[i, test_col]))

                            if len(vals) >= 5:
                                # Check if values are monotonically increasing
                                if all(vals[i] < vals[i+1] for i in range(len(vals)-1)):
                                    position_col = test_col
                                    logger.debug(f"Format C: Found position in Col {test_col}")
                                    break
                        else:
                            # Fall back to using index as position
                            position_col = cols["index"]  # Col B (1)
                            logger.debug(f"Format D: Using index column for position")

            # Extract data arrays
            electrical_angles = []  # X-axis (linear inches or rotary degrees)
            measured_values = []
            theory_values = []
            file_errors = []  # Pre-calculated errors from file
            upper_limits = []
            lower_limits = []

            for i in range(data_start, len(df)):
                row = df.iloc[i]

                # Get electrical angle/position - X-axis for linearity
                # Uses position_col determined by format detection above
                if df.shape[1] > position_col:
                    ea = row.iloc[position_col]
                    if is_numeric(ea):
                        electrical_angles.append(float(ea))
                    else:
                        continue  # Skip rows without valid electrical angle
                else:
                    continue

                # Get measured value (Column A)
                if df.shape[1] > cols["measured"]:
                    meas = row.iloc[cols["measured"]]
                    measured_values.append(float(meas) if pd.notna(meas) else 0.0)
                else:
                    measured_values.append(0.0)

                # Get theory value (Column C)
                if df.shape[1] > cols["theory"]:
                    theory = row.iloc[cols["theory"]]
                    theory_values.append(float(theory) if pd.notna(theory) else None)
                else:
                    theory_values.append(None)

                # Get pre-calculated error (Column D)
                if df.shape[1] > cols["error"]:
                    err = row.iloc[cols["error"]]
                    file_errors.append(float(err) if pd.notna(err) else None)
                else:
                    file_errors.append(None)

                # Get upper limit (Column G)
                if df.shape[1] > cols["upper_limit"]:
                    upper = row.iloc[cols["upper_limit"]]
                    upper_limits.append(float(upper) if pd.notna(upper) else None)
                else:
                    upper_limits.append(None)

                # Get lower limit (Column H)
                if df.shape[1] > cols["lower_limit"]:
                    lower = row.iloc[cols["lower_limit"]]
                    lower_limits.append(float(lower) if pd.notna(lower) else None)
                else:
                    lower_limits.append(None)

            if electrical_angles and measured_values:
                # Use pre-calculated errors from file if available
                valid_file_errors = [e for e in file_errors if e is not None]
                n_points = len(electrical_angles)

                if len(valid_file_errors) >= n_points * 0.9:
                    # Use file errors (replace None with 0)
                    errors = [e if e is not None else 0.0 for e in file_errors]
                    logger.debug("Using pre-calculated errors from file")
                else:
                    # Fall back to calculating errors from measured vs theory or ideal line
                    valid_theory = [t for t in theory_values if t is not None]

                    if len(valid_theory) >= n_points * 0.9:
                        # Calculate from measured - theory
                        measured_arr = np.array(measured_values)
                        theory_arr = np.array([t if t is not None else measured_values[i]
                                               for i, t in enumerate(theory_values)])
                        errors_raw = measured_arr - theory_arr

                        # Normalize by full scale
                        full_scale = measured_arr.max() - measured_arr.min()
                        if full_scale > 0:
                            errors = (errors_raw / full_scale).tolist()
                        else:
                            errors = errors_raw.tolist()
                        logger.debug("Calculated errors from measured vs theory")
                    else:
                        # Fall back to linear fit using electrical angle as X-axis
                        ea_arr = np.array(electrical_angles)
                        measured_arr = np.array(measured_values)

                        if n_points >= 2:
                            coeffs = np.polyfit(ea_arr, measured_arr, 1)
                            ideal_values = np.polyval(coeffs, ea_arr)
                            errors_raw = measured_arr - ideal_values

                            full_scale = measured_arr.max() - measured_arr.min()
                            if full_scale > 0:
                                errors = (errors_raw / full_scale).tolist()
                            else:
                                errors = errors_raw.tolist()
                        else:
                            errors = [0.0] * len(measured_values)
                        logger.debug("Calculated errors from linear fit")

                # Sort all arrays by electrical_angle (ascending) for proper chart display
                if electrical_angles and len(electrical_angles) > 1:
                    # Check if we need to sort
                    if electrical_angles[0] > electrical_angles[-1]:
                        # Create sorted indices
                        sorted_indices = np.argsort(electrical_angles)
                        electrical_angles = [electrical_angles[i] for i in sorted_indices]
                        errors = [errors[i] for i in sorted_indices]
                        measured_values = [measured_values[i] for i in sorted_indices]
                        theory_values = [theory_values[i] for i in sorted_indices]
                        upper_limits = [upper_limits[i] for i in sorted_indices] if upper_limits else []
                        lower_limits = [lower_limits[i] for i in sorted_indices] if lower_limits else []
                        logger.debug(f"Sorted data by electrical angle: {electrical_angles[0]:.2f} -> {electrical_angles[-1]:.2f}")

                # Calculate linearity metrics
                linearity_error = max(abs(e) for e in errors) if errors else 0.0
                linearity_spec = self._calculate_linearity_spec(upper_limits, lower_limits)
                linearity_pass = linearity_error <= linearity_spec if linearity_spec > 0 else True

                # Count fail points (comparing error to spec limits)
                fail_points = 0
                for i, err in enumerate(errors):
                    upper = upper_limits[i] if i < len(upper_limits) else None
                    lower = lower_limits[i] if i < len(lower_limits) else None
                    if upper is not None and err > upper:
                        fail_points += 1
                    elif lower is not None and err < lower:
                        fail_points += 1

                # Find electrical angle of max deviation
                max_err_idx = errors.index(max(errors, key=abs)) if errors else 0
                max_dev_angle = electrical_angles[max_err_idx] if max_err_idx < len(electrical_angles) else 0.0

                tracks.append({
                    "track_id": "default",
                    "electrical_angles": electrical_angles,  # X-axis (inches for linear, degrees for rotary)
                    "measured_values": measured_values,
                    "theory_values": theory_values,
                    "errors": errors,
                    "upper_limits": upper_limits,
                    "lower_limits": lower_limits,
                    "linearity_error": linearity_error,
                    "linearity_spec": linearity_spec,
                    "linearity_pass": linearity_pass,
                    "linearity_fail_points": fail_points,
                    "max_deviation": linearity_error,
                    "max_deviation_angle": max_dev_angle,
                })

        except Exception as e:
            logger.error(f"Error extracting Format 1 tracks: {e}")
        finally:
            if df is not None:
                del df  # Free memory

        return tracks

    def _extract_format2_tracks(self, xl: pd.ExcelFile) -> List[Dict[str, Any]]:
        """
        Extract track data from Format 2 file (Rout_ prefix files).

        Data sheet contains:
        - Measured (col 0) - actual output value
        - Position (col 1) - normalized position (0.0 to 1.0)
        - Index (col 2)

        Linearity error is CALCULATED from measured vs ideal line.
        """
        tracks = []
        cols = FINAL_TEST_FORMAT2_COLUMNS
        df = None

        try:
            df = pd.read_excel(xl, sheet_name="Data", header=None)

            # Helper function to check if value is numeric (handles numpy types)
            def is_numeric(val):
                return pd.notna(val) and np.issubdtype(type(val), np.number)

            # Auto-detect data start row
            # Check if row 0 has numeric data in position column
            data_start = 0
            if df.shape[1] > cols["position"]:
                val = df.iloc[0, cols["position"]]
                if not is_numeric(val):
                    # Row 0 is header, start from row 1
                    data_start = 1
                    logger.debug("Format 2: Header detected, starting at row 1")
                else:
                    logger.debug("Format 2: No header, starting at row 0")

            positions = []
            measured_values = []

            for i in range(data_start, len(df)):
                row = df.iloc[i]

                # Get position
                if df.shape[1] > cols["position"]:
                    pos = row.iloc[cols["position"]]
                    if is_numeric(pos):
                        positions.append(float(pos))
                    else:
                        continue
                else:
                    continue

                # Get measured value
                if df.shape[1] > cols["measured"]:
                    meas = row.iloc[cols["measured"]]
                    measured_values.append(float(meas) if pd.notna(meas) else 0.0)
                else:
                    measured_values.append(0.0)

            if positions and measured_values:
                # CALCULATE linearity error from measured vs ideal
                positions_arr = np.array(positions)
                measured_arr = np.array(measured_values)

                # Fit ideal line: measured = m * position + b
                if len(positions) >= 2:
                    coeffs = np.polyfit(positions_arr, measured_arr, 1)
                    ideal_values = np.polyval(coeffs, positions_arr)

                    # Error is deviation from ideal line
                    errors_raw = measured_arr - ideal_values

                    # Normalize error by full scale range
                    full_scale = measured_arr.max() - measured_arr.min()
                    if full_scale > 0:
                        errors = (errors_raw / full_scale).tolist()
                    else:
                        errors = errors_raw.tolist()
                else:
                    errors = [0.0] * len(measured_values)

                linearity_error = max(abs(e) for e in errors) if errors else 0.0

                # Find position of max deviation
                max_err_idx = errors.index(max(errors, key=abs)) if errors else 0
                max_dev_position = positions[max_err_idx] if max_err_idx < len(positions) else 0.0

                tracks.append({
                    "track_id": "default",
                    "positions": positions,
                    "measured_values": measured_values,
                    "errors": errors,
                    "electrical_angles": positions,  # Use positions as electrical_angles for Format 2
                    "upper_limits": [],
                    "lower_limits": [],
                    "linearity_error": linearity_error,
                    "linearity_spec": None,  # Not available in Format 2
                    "linearity_pass": None,  # Need spec to determine
                    "linearity_fail_points": 0,
                    "max_deviation": linearity_error,
                    "max_deviation_position": max_dev_position,
                })

        except Exception as e:
            logger.error(f"Error extracting Format 2 tracks: {e}")
        finally:
            if df is not None:
                del df  # Free memory

        return tracks

    def _parse_format3_multitrack(self, xl: pd.ExcelFile, file_path: Path, file_hash: str) -> Dict[str, Any]:
        """
        Parse Format 3 Final Test file (multi-track with sheets A, B, C).

        Files like 8639-30 have separate sheets for each track (A, B, C).
        Each sheet has the same structure as Format 1.
        """
        filename = file_path.name

        # Extract metadata from filename
        metadata = self._extract_metadata_from_filename(filename)

        # Extract tracks from sheets A, B, C, etc.
        tracks = []
        track_sheets = [s for s in xl.sheet_names if len(s) == 1 and s.isalpha()]

        for sheet_name in track_sheets:
            try:
                track = self._extract_single_track_from_sheet(xl, sheet_name, track_id=sheet_name)
                if track:
                    tracks.append(track)
            except Exception as e:
                logger.debug(f"Error extracting track from sheet {sheet_name}: {e}")

        # Extract test results from Data Table
        test_results = self._extract_test_results(xl)

        return {
            "metadata": metadata,
            "tracks": tracks,
            "test_results": test_results,
            "file_hash": file_hash,
            "format": "format3_multitrack",
        }

    def _extract_single_track_from_sheet(
        self, xl: pd.ExcelFile, sheet_name: str, track_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Extract a single track from a sheet (used by Format 3).

        The column structure is similar to Format 1:
        - Col 0: Measured value
        - Col 1: Index
        - Col 2: Theory value
        - Col 3: Calculated error
        - Col 4: Position/Index
        - Col 5: Error (from file)
        - Col 6: Upper limit
        - Col 7: Lower limit
        """
        df = None
        try:
            df = pd.read_excel(xl, sheet_name=sheet_name, header=None)

            def is_numeric(val):
                return pd.notna(val) and np.issubdtype(type(val), np.number)

            # Find data start row
            data_start = 0
            for i in range(min(5, len(df))):
                if df.shape[1] > 1:
                    val = df.iloc[i, 0]
                    if is_numeric(val):
                        data_start = i
                        break

            # Extract data - use col 4 for position (index-based), col 5 for error
            electrical_angles = []
            errors = []
            upper_limits = []
            lower_limits = []

            for i in range(data_start, len(df)):
                row = df.iloc[i]

                # Position from column 4
                if df.shape[1] > 4 and is_numeric(row.iloc[4]):
                    electrical_angles.append(float(row.iloc[4]))
                else:
                    continue

                # Error from column 5
                if df.shape[1] > 5 and is_numeric(row.iloc[5]):
                    errors.append(float(row.iloc[5]))
                else:
                    errors.append(0.0)

                # Upper limit from column 6
                if df.shape[1] > 6 and is_numeric(row.iloc[6]):
                    upper_limits.append(float(row.iloc[6]))
                else:
                    upper_limits.append(None)

                # Lower limit from column 7
                if df.shape[1] > 7 and is_numeric(row.iloc[7]):
                    lower_limits.append(float(row.iloc[7]))
                else:
                    lower_limits.append(None)

            if electrical_angles and errors:
                linearity_error = max(abs(e) for e in errors) if errors else 0.0
                linearity_spec = self._calculate_linearity_spec(upper_limits, lower_limits)

                # Count fail points
                fail_points = 0
                for i, err in enumerate(errors):
                    upper = upper_limits[i] if i < len(upper_limits) else None
                    lower = lower_limits[i] if i < len(lower_limits) else None
                    if upper is not None and err > upper:
                        fail_points += 1
                    elif lower is not None and err < lower:
                        fail_points += 1

                return {
                    "track_id": track_id,
                    "electrical_angles": electrical_angles,
                    "errors": errors,
                    "upper_limits": upper_limits,
                    "lower_limits": lower_limits,
                    "linearity_error": linearity_error,
                    "linearity_spec": linearity_spec,
                    "linearity_pass": fail_points == 0,
                    "linearity_fail_points": fail_points,
                    "max_deviation": linearity_error,
                }

        except Exception as e:
            logger.debug(f"Error in _extract_single_track_from_sheet: {e}")
        finally:
            if df is not None:
                del df  # Free memory

        return None

    def _parse_format4_parameters(self, xl: pd.ExcelFile, file_path: Path, file_hash: str) -> Dict[str, Any]:
        """
        Parse Format 4 Final Test file (Parameters sheet format).

        Files like 8407-52.xls have a Parameters sheet with embedded data.
        Column structure appears similar but data starts at different rows.
        """
        filename = file_path.name

        # Extract metadata from filename
        metadata = self._extract_metadata_from_filename(filename)

        tracks = []
        df = None

        try:
            df = pd.read_excel(xl, sheet_name="Parameters", header=None)

            def is_numeric(val):
                return pd.notna(val) and np.issubdtype(type(val), np.number)

            # Data appears to start at row 0, but first few rows have text in col 0
            # Look for rows where col 4 (position) has numeric increasing values
            data_start = 0
            for i in range(min(10, len(df))):
                if df.shape[1] > 4 and is_numeric(df.iloc[i, 4]):
                    data_start = i
                    break

            # Extract data
            electrical_angles = []
            errors = []
            upper_limits = []
            lower_limits = []

            for i in range(data_start, len(df)):
                row = df.iloc[i]

                # Position from column 4
                if df.shape[1] > 4 and is_numeric(row.iloc[4]):
                    electrical_angles.append(float(row.iloc[4]))
                else:
                    continue

                # Error from column 5
                if df.shape[1] > 5 and is_numeric(row.iloc[5]):
                    errors.append(float(row.iloc[5]))
                else:
                    errors.append(0.0)

                # Upper limit from column 6
                if df.shape[1] > 6 and is_numeric(row.iloc[6]):
                    upper_limits.append(float(row.iloc[6]))
                else:
                    upper_limits.append(None)

                # Lower limit from column 7
                if df.shape[1] > 7 and is_numeric(row.iloc[7]):
                    lower_limits.append(float(row.iloc[7]))
                else:
                    lower_limits.append(None)

            if electrical_angles and errors:
                linearity_error = max(abs(e) for e in errors) if errors else 0.0
                linearity_spec = self._calculate_linearity_spec(upper_limits, lower_limits)

                # Count fail points
                fail_points = 0
                for i, err in enumerate(errors):
                    upper = upper_limits[i] if i < len(upper_limits) else None
                    lower = lower_limits[i] if i < len(lower_limits) else None
                    if upper is not None and err > upper:
                        fail_points += 1
                    elif lower is not None and err < lower:
                        fail_points += 1

                tracks.append({
                    "track_id": "default",
                    "electrical_angles": electrical_angles,
                    "errors": errors,
                    "upper_limits": upper_limits,
                    "lower_limits": lower_limits,
                    "linearity_error": linearity_error,
                    "linearity_spec": linearity_spec,
                    "linearity_pass": fail_points == 0,
                    "linearity_fail_points": fail_points,
                    "max_deviation": linearity_error,
                })

        except Exception as e:
            logger.error(f"Error parsing Format 4: {e}")
        finally:
            if df is not None:
                del df  # Free memory

        # No detailed test results for this format
        test_results = {
            "linearity_pass": None,
            "resistance_pass": None,
            "electrical_angle_pass": None,
            "hysteresis_pass": None,
            "phasing_pass": None,
        }

        return {
            "metadata": metadata,
            "tracks": tracks,
            "test_results": test_results,
            "file_hash": file_hash,
            "format": "format4_parameters",
        }

    def _parse_format_shop_test(self, xl: pd.ExcelFile, file_path: Path, file_hash: str) -> Dict[str, Any]:
        """
        Parse Shop Test format files (have 'test' sheet instead of 'Sheet1').

        Column layout in 'test' sheet:
        - Col 0-1: Metadata (Model, Shop, Test V, etc.)
        - Col 3: Position
        - Col 4: Vtheo (theory voltage)
        - Col 5: hi tol (upper tolerance)
        - Col 6: lo tol (lower tolerance)
        - Col 7: Meas V (measured voltage)
        - Col 8: Error
        - Col 9-10: Optional (some files have Column2/3 with pass/fail)

        Row 0 is header, data starts at row 1.
        """
        filename = file_path.name

        # Extract metadata from filename
        metadata = self._extract_metadata_from_filename(filename)

        tracks = []
        df = None

        try:
            df = pd.read_excel(xl, sheet_name="test", header=None)

            def is_numeric(val):
                return pd.notna(val) and np.issubdtype(type(val), np.number)

            # Extract metadata from cells
            # Row 0, Col 1: Model number
            # Row 1, Col 1: Shop number
            if df.shape[0] > 0 and df.shape[1] > 1:
                model_cell = df.iloc[0, 1]
                if pd.notna(model_cell):
                    metadata["model"] = str(model_cell)

            if df.shape[0] > 1 and df.shape[1] > 1:
                shop_cell = df.iloc[1, 1]
                if pd.notna(shop_cell):
                    # Shop might be the serial or shop number
                    metadata["serial"] = str(shop_cell)

            # Data starts at row 1 (after header row 0)
            data_start = 1

            electrical_angles = []  # Position (col 3)
            measured_values = []    # Meas V (col 7)
            theory_values = []      # Vtheo (col 4)
            file_errors = []        # Error (col 8)
            upper_limits = []       # hi tol (col 5)
            lower_limits = []       # lo tol (col 6)

            for i in range(data_start, len(df)):
                row = df.iloc[i]

                # Position from column 3
                if df.shape[1] > 3 and is_numeric(row.iloc[3]):
                    electrical_angles.append(float(row.iloc[3]))
                else:
                    continue

                # Theory from column 4
                if df.shape[1] > 4 and is_numeric(row.iloc[4]):
                    theory_values.append(float(row.iloc[4]))
                else:
                    theory_values.append(None)

                # Upper limit from column 5
                if df.shape[1] > 5 and is_numeric(row.iloc[5]):
                    upper_limits.append(float(row.iloc[5]))
                else:
                    upper_limits.append(None)

                # Lower limit from column 6
                if df.shape[1] > 6 and is_numeric(row.iloc[6]):
                    lower_limits.append(float(row.iloc[6]))
                else:
                    lower_limits.append(None)

                # Measured from column 7
                if df.shape[1] > 7 and is_numeric(row.iloc[7]):
                    measured_values.append(float(row.iloc[7]))
                else:
                    measured_values.append(0.0)

                # Error from column 8
                if df.shape[1] > 8 and is_numeric(row.iloc[8]):
                    file_errors.append(float(row.iloc[8]))
                else:
                    file_errors.append(None)

            if electrical_angles and measured_values:
                n_points = len(electrical_angles)

                # Use file errors if available
                valid_file_errors = [e for e in file_errors if e is not None]
                if len(valid_file_errors) >= n_points * 0.9:
                    errors = [e if e is not None else 0.0 for e in file_errors]
                else:
                    # Calculate errors from measured vs theory
                    errors = []
                    for i in range(len(measured_values)):
                        meas = measured_values[i]
                        theory = theory_values[i] if i < len(theory_values) and theory_values[i] is not None else meas
                        errors.append(meas - theory)

                linearity_error = max(abs(e) for e in errors) if errors else 0.0
                linearity_spec = self._calculate_linearity_spec(upper_limits, lower_limits)
                linearity_pass = linearity_error <= linearity_spec if linearity_spec > 0 else True

                # Count fail points
                fail_points = 0
                for i, err in enumerate(errors):
                    upper = upper_limits[i] if i < len(upper_limits) else None
                    lower = lower_limits[i] if i < len(lower_limits) else None
                    if upper is not None and err > upper:
                        fail_points += 1
                    elif lower is not None and err < lower:
                        fail_points += 1

                # Find position of max deviation
                max_err_idx = errors.index(max(errors, key=abs)) if errors else 0
                max_dev_angle = electrical_angles[max_err_idx] if max_err_idx < len(electrical_angles) else 0.0

                tracks.append({
                    "track_id": "default",
                    "electrical_angles": electrical_angles,
                    "measured_values": measured_values,
                    "theory_values": theory_values,
                    "errors": errors,
                    "upper_limits": upper_limits,
                    "lower_limits": lower_limits,
                    "linearity_error": linearity_error,
                    "linearity_spec": linearity_spec,
                    "linearity_pass": linearity_pass,
                    "linearity_fail_points": fail_points,
                    "max_deviation": linearity_error,
                    "max_deviation_angle": max_dev_angle,
                })

        except Exception as e:
            logger.error(f"Error parsing Shop Test format: {e}")
        finally:
            if df is not None:
                del df  # Free memory

        # No detailed test results for this format
        test_results = {
            "linearity_pass": None,
            "resistance_pass": None,
            "electrical_angle_pass": None,
            "hysteresis_pass": None,
            "phasing_pass": None,
        }

        return {
            "metadata": metadata,
            "tracks": tracks,
            "test_results": test_results,
            "file_hash": file_hash,
            "format": "format_shop_test",
        }

    def _extract_test_results(self, xl: pd.ExcelFile) -> Dict[str, Any]:
        """
        Extract test results from Sheet1.

        In Format 1 files, test results are embedded in Sheet1:
        - Row 1, Col 10: "Linearity Test:", Col 11: "PASSED/FAILED"
        - Row 2, Col 10: "Electrical Angle...", Col 11: "PASSED/FAILED"

        Returns pass/fail status for each test type.
        """
        results = {
            "linearity_pass": None,
            "resistance_pass": None,
            "resistance_value": None,
            "resistance_tolerance": None,
            "electrical_angle_pass": None,
            "hysteresis_pass": None,
            "phasing_pass": None,
        }
        df = None

        try:
            df = pd.read_excel(xl, sheet_name="Sheet1", header=None)

            # Test results are in columns 10-11, rows 1-5
            # Col 10 = test name, Col 11 = PASSED/FAILED
            if df.shape[1] > 11 and df.shape[0] > 5:
                for row_idx in range(1, min(10, df.shape[0])):
                    test_name = df.iloc[row_idx, 10] if pd.notna(df.iloc[row_idx, 10]) else ""
                    result_val = df.iloc[row_idx, 11] if pd.notna(df.iloc[row_idx, 11]) else ""

                    test_name_str = str(test_name).lower()
                    result_str = str(result_val).upper()

                    # Check for PASSED/FAILED in result column
                    is_passed = result_str == "PASSED"
                    is_failed = result_str == "FAILED"

                    if "linearity" in test_name_str:
                        if is_passed or is_failed:
                            results["linearity_pass"] = is_passed
                    elif "electrical angle" in test_name_str:
                        if is_passed or is_failed:
                            results["electrical_angle_pass"] = is_passed
                    elif "resistance" in test_name_str:
                        if is_passed or is_failed:
                            results["resistance_pass"] = is_passed
                    elif "hysteresis" in test_name_str:
                        if is_passed or is_failed:
                            results["hysteresis_pass"] = is_passed
                    elif "phasing" in test_name_str:
                        if is_passed or is_failed:
                            results["phasing_pass"] = is_passed

        except Exception as e:
            logger.error(f"Error extracting test results: {e}")
        finally:
            if df is not None:
                del df  # Free memory

        return results

    def _calculate_linearity_spec(
        self, upper_limits: List[Optional[float]], lower_limits: List[Optional[float]]
    ) -> float:
        """Calculate linearity spec from limits."""
        valid_upper = [u for u in upper_limits if u is not None and not np.isnan(u)]
        valid_lower = [l for l in lower_limits if l is not None and not np.isnan(l)]

        if valid_upper and valid_lower:
            avg_upper = np.mean(valid_upper)
            avg_lower = np.mean(valid_lower)
            return (avg_upper - avg_lower) / 2

        return 0.01  # Default
