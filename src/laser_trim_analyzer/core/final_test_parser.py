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
        self._cache: Dict[str, Any] = {}

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

        # Calculate file hash for deduplication
        file_hash = self._calculate_hash(file_path)

        # Detect format and parse accordingly
        format_type = self._detect_format(file_path)
        logger.debug(f"Detected Final Test format: {format_type}")

        if format_type == "format2":
            return self._parse_format2(file_path, file_hash)
        else:
            return self._parse_format1(file_path, file_hash)

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _detect_format(self, file_path: Path) -> str:
        """
        Detect which Final Test format the file uses.

        Returns:
            'format1' or 'format2'
        """
        filename = file_path.name

        # Check for Rout_ prefix (Format 2)
        if filename.startswith(FINAL_TEST_ROUT_PREFIX):
            return "format2"

        # Check sheet names
        try:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names

            # Format 2 has "Data" and "Charts" sheets
            if "Data" in sheet_names and "Charts" in sheet_names:
                return "format2"

            # Format 1 has "Sheet1" and "Data Table"
            if "Sheet1" in sheet_names or "Data Table" in sheet_names:
                return "format1"

        except Exception as e:
            logger.warning(f"Error detecting format: {e}, defaulting to format1")

        return "format1"

    def _parse_format1(self, file_path: Path, file_hash: str) -> Dict[str, Any]:
        """
        Parse Format 1 Final Test file (standard format).

        Sheet: Sheet1 - Main data
        Sheet: Data Table - Test results summary
        """
        filename = file_path.name
        xl = pd.ExcelFile(file_path)

        # Extract metadata from filename
        metadata = self._extract_metadata_from_filename(filename)

        # Try to get additional metadata from file content
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

    def _parse_format2(self, file_path: Path, file_hash: str) -> Dict[str, Any]:
        """
        Parse Format 2 Final Test file (Rout_ prefix files).

        Sheet: Data - Main measurement data
        Sheet: Charts - Visualization data
        """
        filename = file_path.name
        xl = pd.ExcelFile(file_path)

        # Extract metadata from filename
        # Format: Rout_1091701_sn1695a_vo.xls
        metadata = self._extract_metadata_from_filename(filename)

        # Try to get additional metadata from Data sheet header
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
        sn_match = re.search(r'-sn([a-zA-Z0-9]+)', base, re.IGNORECASE)
        if sn_match:
            metadata["serial"] = sn_match.group(1)

        # Try to extract date (M-D-YYYY pattern)
        date_match = re.search(r'_(\d{1,2})-(\d{1,2})-(\d{4})', base)
        if date_match:
            try:
                month = int(date_match.group(1))
                day = int(date_match.group(2))
                year = int(date_match.group(3))
                metadata["file_date"] = datetime(year, month, day)
            except ValueError:
                pass

        return metadata

    def _extract_format1_tracks(self, xl: pd.ExcelFile) -> List[Dict[str, Any]]:
        """
        Extract track data from Format 1 file.

        Sheet1 contains (standard layout):
        - Measured value (col 0) - actual output voltage/resistance
        - Index (col 1)
        - Electrical angle (col 2)
        - Position (col 4)
        - Upper limit (col 6, typically ±0.025)
        - Lower limit (col 7)

        Alternative layout (detected automatically):
        - Position (col 0) - if col 0 contains increasing values starting near 0
        - Index (col 1)
        - Measured value (col 2)
        - Error (col 3 or 5)
        - Upper limit (col 6)
        - Lower limit (col 7)

        Linearity error is CALCULATED as deviation from ideal line.
        """
        tracks = []
        cols = FINAL_TEST_FORMAT1_COLUMNS.copy()  # Make a copy we can modify

        try:
            df = pd.read_excel(xl, sheet_name="Sheet1", header=None)

            # Detect column layout by checking if col 0 or col 4 contains positions
            # Position column should have values starting near 0 and increasing
            data_start = 0
            for i in range(min(10, len(df))):
                if pd.notna(df.iloc[i, 0]) and isinstance(df.iloc[i, 0], (int, float)):
                    data_start = i
                    break

            # Check first few values in col 0 vs col 4 to determine layout
            col0_vals = []
            col4_vals = []
            for i in range(data_start, min(data_start + 5, len(df))):
                if pd.notna(df.iloc[i, 0]) and isinstance(df.iloc[i, 0], (int, float)):
                    col0_vals.append(float(df.iloc[i, 0]))
                if df.shape[1] > 4 and pd.notna(df.iloc[i, 4]) and isinstance(df.iloc[i, 4], (int, float)):
                    col4_vals.append(float(df.iloc[i, 4]))

            # Position values should start near 0 and increase
            # If col 0 looks like position data, use alternative layout
            if col0_vals and len(col0_vals) >= 3:
                # Check if col 0 starts near 0 and increases
                if col0_vals[0] < 0.1 and col0_vals[-1] > col0_vals[0]:
                    # Also check that values are increasing monotonically
                    if all(col0_vals[i] < col0_vals[i+1] for i in range(len(col0_vals)-1)):
                        # Use alternative layout: position in col 0, measured in col 2
                        cols = {
                            "measured": 2,  # Column C - appears to be measured value
                            "index": 1,
                            "electrical_angle": 4,  # Column E - electrical angle
                            "error": 5,
                            "position": 0,  # Column A - position
                            "upper_limit": 6,
                            "lower_limit": 7,
                        }
                        logger.debug(f"Using alternative column layout (position in col 0)")

            # Find data rows (skip header, look for numeric data)
            for i in range(min(10, len(df))):
                if df.shape[1] > cols["position"]:
                    val = df.iloc[i, cols["position"]]
                    if pd.notna(val) and isinstance(val, (int, float)):
                        data_start = i
                        break

            # Extract data arrays
            positions = []
            measured_values = []
            electrical_angles = []
            upper_limits = []
            lower_limits = []

            for i in range(data_start, len(df)):
                row = df.iloc[i]

                # Get position
                if df.shape[1] > cols["position"]:
                    pos = row.iloc[cols["position"]]
                    if pd.notna(pos) and isinstance(pos, (int, float)):
                        positions.append(float(pos))
                    else:
                        continue  # Skip rows without valid position
                else:
                    continue

                # Get measured value (col 0) - NOT error
                if df.shape[1] > cols["measured"]:
                    meas = row.iloc[cols["measured"]]
                    measured_values.append(float(meas) if pd.notna(meas) else 0.0)
                else:
                    measured_values.append(0.0)

                # Get electrical angle
                if df.shape[1] > cols["electrical_angle"]:
                    angle = row.iloc[cols["electrical_angle"]]
                    electrical_angles.append(float(angle) if pd.notna(angle) else 0.0)
                else:
                    electrical_angles.append(0.0)

                # Get limits
                if df.shape[1] > cols["upper_limit"]:
                    upper = row.iloc[cols["upper_limit"]]
                    upper_limits.append(float(upper) if pd.notna(upper) else None)
                else:
                    upper_limits.append(None)

                if df.shape[1] > cols["lower_limit"]:
                    lower = row.iloc[cols["lower_limit"]]
                    lower_limits.append(float(lower) if pd.notna(lower) else None)
                else:
                    lower_limits.append(None)

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

                # Calculate linearity metrics
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
                max_dev_position = positions[max_err_idx] if max_err_idx < len(positions) else 0.0

                tracks.append({
                    "track_id": "default",
                    "positions": positions,
                    "measured_values": measured_values,
                    "errors": errors,
                    "electrical_angles": electrical_angles,
                    "upper_limits": upper_limits,
                    "lower_limits": lower_limits,
                    "linearity_error": linearity_error,
                    "linearity_spec": linearity_spec,
                    "linearity_pass": linearity_pass,
                    "linearity_fail_points": fail_points,
                    "max_deviation": linearity_error,
                    "max_deviation_position": max_dev_position,
                })

        except Exception as e:
            logger.error(f"Error extracting Format 1 tracks: {e}")

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

        try:
            df = pd.read_excel(xl, sheet_name="Data", header=None)

            # Find data start (skip header row)
            data_start = 1

            positions = []
            measured_values = []

            for i in range(data_start, len(df)):
                row = df.iloc[i]

                # Get position
                if df.shape[1] > cols["position"]:
                    pos = row.iloc[cols["position"]]
                    if pd.notna(pos) and isinstance(pos, (int, float)):
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
                    "electrical_angles": [],  # Not available in Format 2
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

        return tracks

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
