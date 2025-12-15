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

from laser_trim_v3.core.models import SystemType, FileMetadata
from laser_trim_v3.utils.constants import (
    SYSTEM_A_COLUMNS, SYSTEM_B_COLUMNS,
    SYSTEM_A_CELLS, SYSTEM_B_CELLS,
    SYSTEM_A_IDENTIFIER, SYSTEM_B_IDENTIFIERS,
    EXCEL_EXTENSIONS,
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
        self._cache: Dict[str, Any] = {}

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

        # Calculate file hash for deduplication
        file_hash = self._calculate_hash(file_path)

        # Detect system type
        system_type = self._detect_system(file_path)
        logger.debug(f"Detected system: {system_type.value}")

        # Extract metadata
        metadata = self._extract_metadata(file_path, system_type)

        # Extract track data
        tracks = self._extract_tracks(file_path, system_type)

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

    def _detect_system(self, file_path: Path) -> SystemType:
        """
        Detect whether file is System A or System B.

        System A: Has sheets like "SEC1 TRK1 0", "SEC1 TRK1 TRM"
        System B: Has sheets like "test", "Trim 1", "Lin Error"
        """
        try:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names

            # Check for System A patterns
            for sheet in sheet_names:
                if SYSTEM_A_IDENTIFIER in sheet:
                    return SystemType.A

            # Check for System B patterns
            for identifier in SYSTEM_B_IDENTIFIERS:
                if any(identifier.lower() in s.lower() for s in sheet_names):
                    return SystemType.B

            logger.warning(f"Could not detect system type for {file_path.name}, defaulting to B")
            return SystemType.B

        except Exception as e:
            logger.error(f"Error detecting system type: {e}")
            return SystemType.UNKNOWN

    def _extract_metadata(self, file_path: Path, system_type: SystemType) -> FileMetadata:
        """Extract file metadata."""
        # Parse filename for model and serial
        model, serial = self._parse_filename(file_path.name)

        # Get file dates
        file_stat = file_path.stat()
        file_date = datetime.fromtimestamp(file_stat.st_mtime)

        # Try to extract test date from Excel file
        test_date = self._extract_test_date(file_path, system_type)

        # Detect multi-track
        has_multi_tracks = self._has_multiple_tracks(file_path, system_type)

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
        - "Model_Serial_Date.xlsx"
        - "12345_8340-1.xls"
        """
        name = Path(filename).stem

        # Try pattern: Model_Serial or Serial_Model
        parts = re.split(r'[_\-\s]+', name)

        if len(parts) >= 2:
            # Look for model pattern (usually starts with numbers like 8340, 6845)
            model = "Unknown"
            serial = "Unknown"

            for part in parts:
                # Model typically starts with 4+ digits
                if re.match(r'^\d{4}', part):
                    if model == "Unknown":
                        model = part
                    else:
                        serial = part
                elif re.match(r'^[A-Z]{2}\d+', part, re.IGNORECASE):
                    # Serial often has letter prefix
                    serial = part

            if serial == "Unknown" and len(parts) > 1:
                # Use second part as serial if not found
                serial = parts[1] if parts[0] == model else parts[0]

            return model, serial

        return name, "Unknown"

    def _extract_test_date(self, file_path: Path, system_type: SystemType) -> Optional[datetime]:
        """Try to extract the test/trim date from the Excel file."""
        try:
            # Common date cell locations
            date_cells = ["A1", "B1", "A2", "B2"]

            xl = pd.ExcelFile(file_path)
            first_sheet = xl.sheet_names[0]

            df = pd.read_excel(file_path, sheet_name=first_sheet, header=None, nrows=5)

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

            return None

        except Exception as e:
            logger.debug(f"Could not extract test date: {e}")
            return None

    def _has_multiple_tracks(self, file_path: Path, system_type: SystemType) -> bool:
        """Check if file has multiple tracks."""
        if system_type == SystemType.B:
            return False  # System B files are single-track

        try:
            xl = pd.ExcelFile(file_path)
            # Count TRK sheets
            trk_sheets = [s for s in xl.sheet_names if "TRK" in s.upper()]
            # Multiple tracks if we have TRK1 and TRK2
            return len(set(re.findall(r'TRK\d', ' '.join(trk_sheets), re.I))) > 1
        except Exception:
            return False

    def _extract_tracks(self, file_path: Path, system_type: SystemType) -> List[Dict[str, Any]]:
        """Extract data for all tracks in the file."""
        tracks = []

        if system_type == SystemType.A:
            tracks = self._extract_system_a_tracks(file_path)
        else:
            tracks = self._extract_system_b_tracks(file_path)

        return tracks

    def _extract_system_a_tracks(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract tracks from System A file."""
        tracks = []
        xl = pd.ExcelFile(file_path)

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

            if trimmed_sheet:
                track_data = self._extract_track_data(
                    file_path, trimmed_sheet, untrimmed_sheet,
                    SystemType.A, track_id
                )
                if track_data:
                    tracks.append(track_data)

        return tracks

    def _extract_system_b_tracks(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract tracks from System B file."""
        tracks = []
        xl = pd.ExcelFile(file_path)

        # Find untrimmed and trimmed sheets
        untrimmed_sheet = None
        trimmed_sheet = None

        for sheet in xl.sheet_names:
            sheet_lower = sheet.lower()
            if sheet_lower == "test":
                untrimmed_sheet = sheet
            elif "trim" in sheet_lower:
                trimmed_sheet = sheet

        if trimmed_sheet:
            track_data = self._extract_track_data(
                file_path, trimmed_sheet, untrimmed_sheet,
                SystemType.B, "default"
            )
            if track_data:
                tracks.append(track_data)

        return tracks

    def _extract_track_data(
        self,
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

            # Read trimmed data
            df = pd.read_excel(file_path, sheet_name=trimmed_sheet, header=None)

            # Find data start row (first row with numeric position data)
            data_start = self._find_data_start(df, columns["position"])

            if data_start is None:
                logger.warning(f"Could not find data start in {trimmed_sheet}")
                return None

            # Extract columns
            positions = self._get_column_data(df, columns["position"], data_start)
            errors = self._get_column_data(df, columns["error"], data_start)
            upper_limits = self._get_column_data(df, columns["upper_limit"], data_start)
            lower_limits = self._get_column_data(df, columns["lower_limit"], data_start)

            if not positions or not errors:
                logger.warning(f"No position/error data in {trimmed_sheet}")
                return None

            # Extract untrimmed data if available
            untrimmed_positions = None
            untrimmed_errors = None
            if untrimmed_sheet:
                try:
                    df_untrim = pd.read_excel(file_path, sheet_name=untrimmed_sheet, header=None)
                    untrim_start = self._find_data_start(df_untrim, columns["position"])
                    if untrim_start is not None:
                        untrimmed_positions = self._get_column_data(df_untrim, columns["position"], untrim_start)
                        untrimmed_errors = self._get_column_data(df_untrim, columns["error"], untrim_start)
                except Exception as e:
                    logger.debug(f"Could not read untrimmed sheet: {e}")

            # Extract resistance values
            untrimmed_resistance = None
            trimmed_resistance = None

            if untrimmed_sheet:
                untrimmed_resistance = self._extract_cell_value(
                    file_path, untrimmed_sheet, cells["untrimmed_resistance"]
                )

            trimmed_resistance = self._extract_cell_value(
                file_path, trimmed_sheet, cells["trimmed_resistance"]
            )

            # Extract unit length
            unit_length = self._extract_cell_value(
                file_path, trimmed_sheet, cells["unit_length"]
            )

            # Calculate travel length
            travel_length = max(positions) - min(positions) if positions else 0.0

            # Calculate linearity spec from limits
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

    def _get_column_data(self, df: pd.DataFrame, col_idx: int, start_row: int) -> List[float]:
        """Extract numeric data from a column."""
        data = []
        for i in range(start_row, len(df)):
            try:
                value = df.iloc[i, col_idx]
                if pd.notna(value):
                    data.append(float(value))
                else:
                    break  # Stop at first empty cell
            except (ValueError, TypeError):
                break
        return data

    def _extract_cell_value(
        self, file_path: Path, sheet_name: str, cell_ref: str
    ) -> Optional[float]:
        """Extract a single cell value."""
        try:
            # Parse cell reference
            match = re.match(r'^([A-Z]+)(\d+)$', cell_ref.upper())
            if not match:
                return None

            col_letters, row_num = match.groups()

            # Convert column to index
            col_idx = 0
            for char in col_letters:
                col_idx = col_idx * 26 + (ord(char) - ord('A')) + 1
            col_idx -= 1

            row_idx = int(row_num) - 1

            # Read sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

            if row_idx < len(df) and col_idx < len(df.columns):
                value = df.iloc[row_idx, col_idx]
                if pd.notna(value):
                    return float(value)

            return None

        except Exception as e:
            logger.debug(f"Could not extract cell {cell_ref}: {e}")
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
