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

from laser_trim_analyzer.core.parser import _read_once, _workbook
from laser_trim_analyzer.utils.hashing import hash_bytes_for, shares_one_stat
from laser_trim_analyzer.core.analyzer import max_abs_measured
from laser_trim_analyzer.utils.constants import (
    FINAL_TEST_FORMAT1_COLUMNS,
    FINAL_TEST_FORMAT2_COLUMNS,
    FINAL_TEST_FORMAT1_METADATA,
    FINAL_TEST_DATA_TABLE_ROWS,
    FINAL_TEST_DATA_TABLE_COLUMNS,
    FINAL_TEST_ROUT_PREFIX,
    FINAL_TEST_IGNORE_CELLS,
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

    @shares_one_stat
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

        logger.debug(f"Parsing Final Test file: {file_path.name}")

        # Calculate file hash for deduplication (separate file read, that's fine)
        file_hash = self._calculate_hash(file_path)

        # Single file open for all Excel operations - prevents file handle leaks
        with _workbook(file_path) as xl:
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

    # ---- The graded window: which sweep rows the station actually judged ----
    #
    # A final-test sheet grades itself. Column I carries a per-point 0/1 flag
    # (1 = this point's error fell outside the row's G/H limits) and the
    # station writes it ONLY on the rows it graded; the lead-in and run-out
    # rows named by the "# of elements to ignore at start / at end" parameters
    # are left blank. The verdict cell next to "Linearity Test:" is the AND of
    # those flags.
    #
    # The app grades the sweep itself -- offset/slope corrected, per-point,
    # zero-tolerance -- and that remains the disposition. What changed
    # (2026-09-13) is WHICH rows it may grade: the ungraded lead-in of model
    # 8232-1 reads 0 V against a theory of -0.045 V, a phantom 0.047 error
    # against a +/-0.010 band, and grading it failed 98.5% of that model's
    # files at a station that passed them. So the window is read off the file
    # and the rows outside it are excluded from grading, exactly as the
    # station excludes them.

    @staticmethod
    def _station_verdict_from_cell(value) -> Optional[bool]:
        """PASSED/FAILED -> True/False; anything else (incl. 'Not Tested') -> None."""
        if value is None:
            return None
        text = str(value).strip().upper()
        if text == "PASSED":
            return True
        if text == "FAILED":
            return False
        return None

    @classmethod
    def _read_sheet_verdict(cls, df, max_rows: int = 10) -> Optional[bool]:
        """The sheet's own linearity verdict, from the label/value pair in
        columns K/L (10/11). Format 3 carries one PER TRACK SHEET, which is why
        this takes a frame rather than the workbook."""
        if df is None or df.shape[1] <= 11:
            return None
        for row_idx in range(min(max_rows, df.shape[0])):
            label = df.iloc[row_idx, 10]
            if pd.notna(label) and "linearity" in str(label).lower():
                return cls._station_verdict_from_cell(df.iloc[row_idx, 11])
        return None

    @staticmethod
    def _read_ignore_counts(df) -> Tuple[Optional[int], Optional[int]]:
        """(ignore_start, ignore_end) from the parameter block, or (None, None).

        The label wording differs between template vintages -- 8232-1 says
        "# of elements to ignore at start", 7458 says "...from start" -- so the
        match is on 'ignore' plus 'start'/'end', and the value cell is read as
        text because some templates store it as a string.
        """
        label_col = FINAL_TEST_IGNORE_CELLS["label_col"]
        value_col = FINAL_TEST_IGNORE_CELLS["value_col"]
        if df is None or df.shape[1] <= value_col:
            return (None, None)
        start = end = None
        for row_idx in range(df.shape[0]):
            label = df.iloc[row_idx, label_col]
            if not isinstance(label, str) or "ignor" not in label.lower():
                continue
            lowered = label.lower()
            try:
                value = int(float(str(df.iloc[row_idx, value_col]).strip()))
            except (TypeError, ValueError):
                continue
            if value < 0:
                continue
            if "start" in lowered:
                start = value
            elif "end" in lowered:
                end = value
        return (start, end)

    @staticmethod
    def _station_flag(value) -> Optional[int]:
        """One column-I cell as 0/1, or None for blank/non-binary.

        Non-binary is deliberately None rather than a guess: a template whose
        column I holds something else must not be read as a grading flag.
        """
        if value is None or not isinstance(value, (int, float, np.integer, np.floating)):
            return None
        if pd.isna(value):
            return None
        as_float = float(value)
        if as_float == 0.0:
            return 0
        if as_float == 1.0:
            return 1
        return None

    @staticmethod
    def _window_from_flags(flags: List[Optional[int]], errors=None
                           ) -> Optional[Tuple[int, int]]:
        """(first, last) index the station both FLAGGED and MEASURED.

        A populated flag is the station saying "I judged this row". Most
        templates leave the ignored lead-in/run-out blank and the span is
        exactly the graded block (8232-1: rows 5..49 of 57, matching its
        declared 6-and-6 ignore counts). But model 7458's template writes a
        literal 0 on rows it never measured -- rows whose error cell is empty --
        and taking those at face value would put unmeasured points back inside
        the graded window, where a zero-tolerance grade must count each one as
        a fail. That is the exact false-FAIL this whole change exists to stop,
        so a row with no error reading cannot bound the window.

        Only the ENDS are trimmed: a blank error INSIDE the span stays inside
        it and is graded (as a fail), because the station would have flagged it.
        """
        n = len(flags)
        def measured(i: int) -> bool:
            if errors is None or i >= len(errors):
                return True
            value = errors[i]
            if value is None:
                return False
            return not (isinstance(value, float) and np.isnan(value))
        populated = [i for i in range(n) if flags[i] is not None and measured(i)]
        if not populated:
            return None
        return (populated[0], populated[-1])

    @staticmethod
    def _window_from_ignore(n_points: int, ignore_start: Optional[int],
                            ignore_end: Optional[int]) -> Optional[Tuple[int, int]]:
        """(first, last) graded index implied by the ignore counts.

        Returns None when neither count is known or when they would leave
        nothing to grade -- an empty window is not a fallback, it is a reason
        to grade everything and say so.
        """
        if ignore_start is None and ignore_end is None:
            return None
        low = max(0, int(ignore_start or 0))
        high = n_points - 1 - max(0, int(ignore_end or 0))
        if low > high or n_points <= 0:
            return None
        return (low, high)

    @classmethod
    def _graded_window(cls, n_points: int, flags: List[Optional[int]],
                       ignore_start: Optional[int], ignore_end: Optional[int],
                       errors=None
                       ) -> Tuple[Optional[Tuple[int, int]], str]:
        """The window and WHERE it came from.

        Precedence: the flags the station actually wrote, then the declared
        ignore counts, then nothing ('all_rows' -- grade the whole sweep, which
        is what the app did before this existed).
        """
        window = cls._window_from_flags(flags, errors)
        if window is not None:
            return window, "flags"
        window = cls._window_from_ignore(n_points, ignore_start, ignore_end)
        if window is not None:
            return window, "ignore_cells"
        return None, "all_rows"

    @staticmethod
    def _grade_points(errors, upper_limits, lower_limits,
                      window: Optional[Tuple[int, int]]
                      ) -> Tuple[int, Optional[bool]]:
        """Raw per-point grading, restricted to the graded window.

        This is the parser's UNCORRECTED count -- the processor replaces it
        with the analyzer's offset/slope-corrected one, which is the verdict of
        record. It still matters: it is what a track carries when no spec is
        found, and what `core.track_repair` writes back.

        `None` verdict means nothing was gradeable (no limits, or no point
        inside the window carrying both a limit pair and a reading). A track
        the app could not judge must not be recorded as a pass.
        """
        if not errors or not upper_limits or not lower_limits:
            return 0, None
        low, high = (window if window is not None else (0, len(errors) - 1))
        fail_points = 0
        graded = 0
        for i, err in enumerate(errors):
            if i < low or i > high:
                continue
            upper = upper_limits[i] if i < len(upper_limits) else None
            lower = lower_limits[i] if i < len(lower_limits) else None
            if upper is None or lower is None:
                continue
            if (isinstance(upper, float) and np.isnan(upper)) or \
               (isinstance(lower, float) and np.isnan(lower)):
                continue
            graded += 1
            # Same rule as Analyzer._count_fail_points: an unmeasured point
            # inside the graded band counts as a fail, because a zero-tolerance
            # spec cannot show it was in spec.
            if err is None or (isinstance(err, float) and np.isnan(err)):
                fail_points += 1
                continue
            if err > upper or err < lower:
                fail_points += 1
        if graded == 0:
            return 0, None
        return fail_points, fail_points == 0

    @staticmethod
    def _max_deviation_angle(errors, positions) -> Optional[float]:
        """Position of the largest MEASURED deviation, or None."""
        best_idx = None
        best = None
        for i, err in enumerate(errors):
            if err is None or (isinstance(err, float) and (np.isnan(err) or np.isinf(err))):
                continue
            magnitude = abs(err)
            if best is None or magnitude > best:
                best, best_idx = magnitude, i
        if best_idx is None or best_idx >= len(positions):
            return None
        return positions[best_idx]

    @staticmethod
    def _station_fields(station_flags, window, window_source,
                        ignore_start, ignore_end) -> Dict[str, Any]:
        """The station's own grading, carried alongside the app's as REFERENCE.

        None everywhere the file does not say: a template with no flag column
        gets `station_linearity_pass` None, not a manufactured True.
        """
        populated = [f for f in station_flags if f is not None]
        return {
            "station_flags": list(station_flags),
            "station_fail_points": (sum(1 for f in populated if f == 1)
                                    if populated else None),
            "station_linearity_pass": (all(f == 0 for f in populated)
                                       if populated else None),
            "graded_window": window,
            "graded_window_source": window_source,
            "graded_start": window[0] if window else None,
            "graded_end": window[1] if window else None,
            "ignore_start": ignore_start,
            "ignore_end": ignore_end,
        }

    @staticmethod
    def _note_cell_flag_conflict(test_results: Dict[str, Any],
                                 tracks: List[Dict[str, Any]]) -> None:
        """Record when the sheet's verdict cell and its own flags disagree.

        Neither is overridden -- both are stored, and this says they differ.
        45 of the 607 local Format 1 sample files with a verdict cell are in
        this state, almost all of them old templates whose verdict formula
        covers a narrower range than the flag column (8322-10 reads PASSED
        with 42 flags set). The conflict is a property of the FILE, not a
        parsing failure, so it is data rather than a warning.
        """
        cell = test_results.get("station_linearity_pass")
        flag_values = [t.get("station_linearity_pass") for t in tracks
                       if t.get("station_linearity_pass") is not None]
        if cell is None or not flag_values:
            test_results["station_cell_flag_conflict"] = None
            return
        test_results["station_cell_flag_conflict"] = (cell != all(flag_values))

    @staticmethod
    def _no_station_grading() -> Dict[str, Any]:
        """Station fields for a template that states none.

        Format 2 (Rout_) and the shop-test sheets carry no verdict cell and no
        flag column, and Format 4's flag column could not be identified (see
        _parse_format4_parameters). They get explicit Nones so every track dict
        has the same shape for `save_final_test` -- a missing key and a
        deliberate "the file does not say" must not look different downstream.
        """
        return {
            "station_flags": None,
            "station_fail_points": None,
            "station_linearity_pass": None,
            "graded_window": None,
            "graded_window_source": "all_rows",
            "graded_start": None,
            "graded_end": None,
            "ignore_start": None,
            "ignore_end": None,
        }

    def _calculate_hash(self, file_path: Path) -> str:
        """SHA256 via the one cached read (see parser._read_once).

        This parser used to read the file once to hash it and again to
        parse it; on the work share each read is a network transfer.
        """
        return hash_bytes_for(file_path, _read_once(file_path))

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
            try:
                df = pd.read_excel(xl, sheet_name="Sheet1", header=None)
            except ValueError:
                # See _extract_format1_tracks for rationale.
                if not xl.sheet_names:
                    raise
                fallback_sheet = xl.sheet_names[0]
                logger.warning(
                    f"'Sheet1' not found while reading metadata in {file_path.name!r}, "
                    f"falling back to first sheet {fallback_sheet!r}"
                )
                df = pd.read_excel(xl, sheet_name=fallback_sheet, header=None)

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

            # Extract compensation from cell M4 (column 12, row 3)
            from laser_trim_analyzer.utils.constants import FINAL_TEST_FORMAT1_COMPENSATION
            comp_info = FINAL_TEST_FORMAT1_COMPENSATION
            if df.shape[0] > comp_info["row"] and df.shape[1] > comp_info["col"]:
                comp_val = df.iloc[comp_info["row"], comp_info["col"]]
                if pd.notna(comp_val):
                    try:
                        metadata["station_compensation"] = float(comp_val)
                    except (ValueError, TypeError):
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
        self._note_cell_flag_conflict(test_results, tracks)

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
            "station_linearity_pass": None,
            "station_cell_flag_conflict": None,
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
        - 8340-1 final 215_6-4-2025_7-38 PM.xls
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

        # Try to extract model number with optional suffix (e.g., 8340-1, 7280-1-CT, 8508-A)
        # Pattern: digits followed by optional numeric segments and optional letter suffix
        model_match = re.match(r'^(\d+(?:-\d+)*(?:-[A-Z]+)?)', base)
        if model_match:
            metadata["model"] = model_match.group(1)

        # Try to extract serial number (snXXX pattern)
        # Handle both hyphen and underscore separators: -sn108 or _SN16
        sn_match = re.search(r'[-_]sn([a-zA-Z0-9]+)', base, re.IGNORECASE)
        if sn_match:
            metadata["serial"] = sn_match.group(1)
        else:
            # Fallback: "model final [sn] serial" pattern.  Production stations
            # write filenames as any of:
            #   "8340-1 final 215_6-4-2025_..."     -> serial 215  (legacy)
            #   "8340-1 final SN 5140.xls"          -> serial 5140 (FY26 batches)
            #   "8340-1 final sn  116xls.xls"       -> serial 116  (rename artifact:
            #                                                       extra space + "xls"
            #                                                       embedded before ext)
            # The optional "sn " token and one-or-more whitespace handle all three.
            # \s+ is intentional (not \s) so a double-space doesn't break the match.
            final_match = re.search(
                r'\bfinal\s+(?:sn\s+)?(\d+)', base, re.IGNORECASE
            )
            if final_match:
                metadata["serial"] = final_match.group(1)
            else:
                # Fallback B: "model-shop<N>" traveler-ID pattern.
                # Some FY26 production stations identify the unit by shop traveler rather
                # than a serial number (e.g. "1844205-shop1_<date>.xlsx").
                # Per 2026-05-30 decision, the shop ID IS the unit identifier
                # at those stations, so we store it verbatim (lowercased) as
                # the serial.
                # NOTE: \d+ is greedy so it eats the full shop number; we use
                # a negative-lookahead (?!\d) instead of \b because the digits
                # are typically followed by '_' which is itself a word
                # character in Python regex (so \b would not fire there).
                shop_match = re.search(r'[-_](shop\d+)(?!\d)', base, re.IGNORECASE)
                if shop_match:
                    metadata["serial"] = shop_match.group(1).lower()

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
            try:
                df = pd.read_excel(xl, sheet_name="Sheet1", header=None)
            except ValueError:
                # Format detector routed this to Format 1 but the workbook's
                # main sheet isn't named 'Sheet1'.  Some FY26 station exports
                # use station-specific sheet names while keeping the same
                # column layout.  Fall back to the first sheet.
                if not xl.sheet_names:
                    raise
                fallback_sheet = xl.sheet_names[0]
                logger.warning(
                    f"'Sheet1' not found in {getattr(xl, 'io', '?')}, "
                    f"falling back to first sheet {fallback_sheet!r}"
                )
                df = pd.read_excel(xl, sheet_name=fallback_sheet, header=None)

            # Helper function to check if value is numeric (handles numpy types)
            def is_numeric(val):
                return pd.notna(val) and isinstance(val, (int, float, np.integer, np.floating))

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
            # The station's own per-point flags, aligned to the rows that
            # SURVIVE the filters below, never to the raw spreadsheet rows.
            # Some files carry a stray populated row hundreds of rows past the
            # sweep (7281-sn466b: one at row 344 of a 29-row sweep); tying the
            # flags to kept rows is what keeps such a row from moving the
            # window.
            station_flags = []

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

                # Get measured value (Column A) — required for a real data point.
                # Some files have trailing rows where electrical angle/theory are populated
                # but no measurement was taken (column A is empty). The pre-calculated error
                # column then holds (0 − theory) which can be a multi-volt phantom error
                # and dominates the linearity result. Skip such rows entirely.
                if df.shape[1] > cols["measured"]:
                    meas = row.iloc[cols["measured"]]
                    if pd.isna(meas):
                        electrical_angles.pop()
                        continue
                    measured_values.append(float(meas))
                else:
                    electrical_angles.pop()
                    continue

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

                # Get the station's own verdict flag (Column I)
                if df.shape[1] > cols["station_flag"]:
                    station_flags.append(self._station_flag(row.iloc[cols["station_flag"]]))
                else:
                    station_flags.append(None)

            if electrical_angles and measured_values:
                # Use pre-calculated errors from file if available
                valid_file_errors = [e for e in file_errors if e is not None]
                n_points = len(electrical_angles)

                # Sanity-flag the file's error column without changing values.
                # Some files (e.g. 8501, 8186, 8290, 6952b, 8605-2) put
                # something other than linearity error in column D — a constant
                # offset, an absolute voltage. Real linearity errors are small
                # and centered near zero (median |err| well under spec band).
                # If median |err| > 0.1 AND > 10× the spec band, log a SANITY
                # warning so the user knows the reported value is suspect.
                # Values are NOT replaced — production data integrity over
                # heuristic correction.
                if valid_file_errors:
                    abs_errs = sorted(abs(e) for e in valid_file_errors)
                    median_abs_err = abs_errs[len(abs_errs) // 2]
                    valid_uppers = [u for u in upper_limits if u is not None]
                    valid_lowers = [l for l in lower_limits if l is not None]
                    spec_estimate = None
                    if valid_uppers and valid_lowers:
                        spec_estimate = (
                            sum(valid_uppers) / len(valid_uppers)
                            - sum(valid_lowers) / len(valid_lowers)
                        ) / 2
                    if (median_abs_err > 0.1
                            and (spec_estimate is None
                                 or median_abs_err > 10 * spec_estimate)):
                        logger.warning(
                            f"SANITY: FT format1 column D may not be errors — "
                            f"median |file_error|={median_abs_err:.4f} is "
                            f"{'>10× spec' if spec_estimate else 'large'} "
                            f"(spec≈{spec_estimate}). Reported linearity_error "
                            f"is suspect; verify file column layout."
                        )

                if len(valid_file_errors) >= n_points * 0.9:
                    # Use the file's own error column AS IT IS. A blank cell
                    # stays None -- it is a point the station did not measure,
                    # and the 0.0 that used to be written there is the most
                    # flattering value a zero-tolerance metric can hold: dead
                    # centre of every band. Downstream, None is either excluded
                    # (outside the graded window) or counted as a fail point
                    # (inside it), which is what an unmeasured point deserves.
                    errors = list(file_errors)
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

                        # VOLTS, like the limits these are graded against.
                        # Dividing by full scale turned the error into a
                        # fraction of full scale while columns G/H stayed in
                        # volts, shrinking every error ~10x on a 10 V part:
                        # 7539-2 sn23, which the station failed on 149 points,
                        # was stored as a PASS with 0 fail points. The
                        # shop-test parser below recovers errors the same way
                        # and has never divided.
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

                            # Volts, for the same reason as the branch above:
                            # the residual from the ideal line is judged
                            # against the sheet's own volt limits.
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
                        # The flags travel with their rows. A window derived
                        # from an unsorted flag list would name indices in the
                        # other sweep direction.
                        station_flags = ([station_flags[i] for i in sorted_indices]
                                         if station_flags else [])
                        logger.debug(f"Sorted data by electrical angle: {electrical_angles[0]:.2f} -> {electrical_angles[-1]:.2f}")

                # The window the STATION graded, and where that came from.
                ignore_start, ignore_end = self._read_ignore_counts(df)
                window, window_source = self._graded_window(
                    len(errors), station_flags, ignore_start, ignore_end, errors)

                # Calculate linearity metrics. The magnitude skips unmeasured
                # points rather than letting one at index 0 poison max().
                linearity_error = max_abs_measured(errors)
                linearity_spec = self._calculate_linearity_spec(upper_limits, lower_limits)

                fail_points, linearity_pass = self._grade_points(
                    errors, upper_limits, lower_limits, window)

                # Find electrical angle of max deviation (measured points only)
                max_dev_angle = self._max_deviation_angle(errors, electrical_angles)

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
                    **self._station_fields(station_flags, window, window_source,
                                           ignore_start, ignore_end),
                })

        except Exception as e:
            logger.error(f"Error extracting Format 1 tracks: {e}", exc_info=True)
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
                return pd.notna(val) and isinstance(val, (int, float, np.integer, np.floating))

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

                # Get position — finite values only. `pd.notna` (and so
                # is_numeric) answers True for ±inf, and an infinity is not a
                # position: np.polyfit conditions its Vandermonde matrix by
                # dividing each column by that column's norm, an infinite
                # entry makes the norm inf, and inf/inf fills the column with
                # NaN. LAPACK then prints six lines of XERBLA to the console
                # and the fit raises. Drop the row instead.
                if df.shape[1] > cols["position"]:
                    pos = row.iloc[cols["position"]]
                    if not is_numeric(pos):
                        continue
                    pos_val = float(pos)
                    if not np.isfinite(pos_val):
                        continue
                else:
                    continue

                # Get measured value — skip the row entirely when the cell
                # is missing/NaN or non-finite. Substituting 0.0 fabricates a
                # measurement at the lowest possible voltage, which corrupts
                # the measured-vs-ideal-line fit and the resulting error
                # series; an infinity poisons the same fit the same way a
                # non-finite position does.
                if df.shape[1] > cols["measured"]:
                    meas = row.iloc[cols["measured"]]
                    if not pd.notna(meas):
                        continue
                    meas_val = float(meas)
                    if not np.isfinite(meas_val):
                        continue
                    positions.append(pos_val)
                    measured_values.append(meas_val)
                else:
                    continue

            if positions and measured_values:
                # CALCULATE linearity error from measured vs ideal
                positions_arr = np.array(positions)
                measured_arr = np.array(measured_values)

                # A line needs the x axis to actually move. When every
                # surviving position is the SAME value, np.polyfit divides
                # that Vandermonde column by its own norm — 0/0 — fills it
                # with NaN, and LAPACK refuses the matrix: six XERBLA lines
                # printed with C printf — straight to file descriptor 1, so
                # replacing sys.stdout does not intercept them — followed by
                # LinAlgError. Two real 8213-1 files carry 2,998 rows of
                # position 0.0 and did exactly that on every ingest.
                #
                # There is no honest result to fall back to. With no position
                # information there is no deviation-from-ideal to measure, and
                # Format 2 carries no spec limits to judge one against, so an
                # errors array of zeros would enter final_test_tracks as
                # linearity_error 0.0 — a perfect score for a file that
                # measured nothing, dragging down every FT average it joins.
                # Report no track, which is what the swallowed LinAlgError
                # already produced, and say why.
                if len(positions) >= 2 and positions_arr.min() == positions_arr.max():
                    logger.warning(
                        "FT format2: position column has no spread (%r on all "
                        "%d rows) — no ideal line can be fitted, so no track "
                        "is reported for this file.",
                        positions[0], len(positions)
                    )
                    return tracks

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
                    **self._no_station_grading(),
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
        # Format 3 has no file-level verdict cell: each track sheet carries
        # its own. Zero-tolerance linearity means the file passed only if
        # every sheet that stated a verdict passed.
        cells = [t.get("station_cell_pass") for t in tracks
                 if t.get("station_cell_pass") is not None]
        if cells:
            test_results["station_linearity_pass"] = all(cells)
            if test_results.get("linearity_pass") is None:
                test_results["linearity_pass"] = all(cells)
        self._note_cell_flag_conflict(test_results, tracks)

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
                return pd.notna(val) and isinstance(val, (int, float, np.integer, np.floating))

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
            # Column I again, aligned to the rows that survive the filters.
            station_flags = []

            for i in range(data_start, len(df)):
                row = df.iloc[i]

                # Position from column 4
                if df.shape[1] > 4 and is_numeric(row.iloc[4]):
                    pos_val = float(row.iloc[4])
                else:
                    continue

                # Error from column 5 — skip rows with no error reading rather
                # than fabricating 0.0 (which would silently pass a fail point).
                if df.shape[1] > 5 and is_numeric(row.iloc[5]):
                    err_val = float(row.iloc[5])
                else:
                    continue

                electrical_angles.append(pos_val)
                errors.append(err_val)
                station_flags.append(
                    self._station_flag(row.iloc[8]) if df.shape[1] > 8 else None)

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
                ignore_start, ignore_end = self._read_ignore_counts(df)
                window, window_source = self._graded_window(
                    len(errors), station_flags, ignore_start, ignore_end, errors)

                linearity_error = max_abs_measured(errors)
                linearity_spec = self._calculate_linearity_spec(upper_limits, lower_limits)

                fail_points, linearity_pass = self._grade_points(
                    errors, upper_limits, lower_limits, window)

                return {
                    "track_id": track_id,
                    "electrical_angles": electrical_angles,
                    "errors": errors,
                    "upper_limits": upper_limits,
                    "lower_limits": lower_limits,
                    "linearity_error": linearity_error,
                    "linearity_spec": linearity_spec,
                    "linearity_pass": linearity_pass,
                    "linearity_fail_points": fail_points,
                    "max_deviation": linearity_error,
                    # Format 3 keeps its verdict cell PER TRACK SHEET: A, B and
                    # C each carry their own "Linearity Test:" pair, so the
                    # file-level cell read by _extract_test_results does not
                    # exist for these workbooks.
                    "station_cell_pass": self._read_sheet_verdict(df),
                    **self._station_fields(station_flags, window, window_source,
                                           ignore_start, ignore_end),
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
                return pd.notna(val) and isinstance(val, (int, float, np.integer, np.floating))

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
                    pos_val = float(row.iloc[4])
                else:
                    continue

                # Error from column 5 — skip rows with no error reading rather
                # than fabricating 0.0 (which would silently pass a fail point).
                if df.shape[1] > 5 and is_numeric(row.iloc[5]):
                    err_val = float(row.iloc[5])
                else:
                    continue

                electrical_angles.append(pos_val)
                errors.append(err_val)

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
                    **self._no_station_grading(),
                })

        except Exception as e:
            logger.error(f"Error parsing Format 4: {e}")
        finally:
            if df is not None:
                del df  # Free memory

        # Format 4 states ONE verdict, in the top-right cell of the
        # Parameters sheet (row 0, col L). Read separately from the frame
        # above so a failure to build tracks still recovers it.
        #
        # UNVERIFIED, deliberately left alone (2026-09-13): which column holds
        # this template's per-point flag. The obvious candidate, col K, is
        # populated on 21 of 8434ct-1118D's 101 sweep rows and on 34 of
        # 8407-51's 71 -- neither count lines up with the sweep, so it is not
        # a per-point flag and reading it as one would invent a graded window.
        # All four local Format 4 files were checked. Flags stay NULL and the
        # window stays 'all_rows' until a file proves otherwise.
        station_cell = None
        try:
            head = pd.read_excel(xl, sheet_name="Parameters", header=None, nrows=1)
            if head.shape[1] > 11 and head.shape[0] > 0:
                station_cell = self._station_verdict_from_cell(head.iloc[0, 11])
        except Exception as e:
            logger.debug(f"Format 4 verdict cell unreadable: {e}")

        test_results = {
            "linearity_pass": station_cell,
            "station_linearity_pass": station_cell,
            "station_cell_flag_conflict": None,
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
                return pd.notna(val) and isinstance(val, (int, float, np.integer, np.floating))

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

                # Position (col 3) and Measured (col 7) are both required. Skip the
                # whole row if either is missing -- never fabricate a 0.0 measurement
                # (that desyncs the arrays and creates phantom multi-volt errors).
                if not (df.shape[1] > 3 and is_numeric(row.iloc[3])):
                    continue
                if not (df.shape[1] > 7 and is_numeric(row.iloc[7])):
                    continue

                # Append atomically so every array stays aligned to the same rows.
                electrical_angles.append(float(row.iloc[3]))
                theory_values.append(
                    float(row.iloc[4]) if df.shape[1] > 4 and is_numeric(row.iloc[4]) else None)
                upper_limits.append(
                    float(row.iloc[5]) if df.shape[1] > 5 and is_numeric(row.iloc[5]) else None)
                lower_limits.append(
                    float(row.iloc[6]) if df.shape[1] > 6 and is_numeric(row.iloc[6]) else None)
                measured_values.append(float(row.iloc[7]))
                file_errors.append(
                    float(row.iloc[8]) if df.shape[1] > 8 and is_numeric(row.iloc[8]) else None)

            if electrical_angles and measured_values:
                n_points = len(electrical_angles)

                # Use file errors when present; for any missing error compute it
                # from measured - theory rather than fabricating a 0.0.
                valid_file_errors = [e for e in file_errors if e is not None]
                use_file = len(valid_file_errors) >= n_points * 0.9
                errors = []
                for i in range(len(measured_values)):
                    if use_file and file_errors[i] is not None:
                        errors.append(file_errors[i])
                    else:
                        meas = measured_values[i]
                        theory = (theory_values[i]
                                  if i < len(theory_values) and theory_values[i] is not None
                                  else meas)
                        errors.append(meas - theory)

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

                # Zero-tolerance: linearity passes only if ALL points are within limits
                linearity_pass = fail_points == 0

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
                    **self._no_station_grading(),
                })

        except Exception as e:
            logger.error(f"Error parsing Shop Test format: {e}")
        finally:
            if df is not None:
                del df  # Free memory

        # No detailed test results for this format
        test_results = {
            "linearity_pass": None,
            "station_linearity_pass": None,
            "station_cell_flag_conflict": None,
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
            # The same cell, under a name that says whose verdict it is. The
            # processor overwrites `linearity_pass` with the app's own
            # corrected grade (that is the disposition); this one is kept as
            # the station's REFERENCE and is never overwritten.
            "station_linearity_pass": None,
            "station_cell_flag_conflict": None,
            "resistance_pass": None,
            "resistance_value": None,
            "resistance_tolerance": None,
            "electrical_angle_pass": None,
            "hysteresis_pass": None,
            "phasing_pass": None,
        }
        df = None

        try:
            try:
                df = pd.read_excel(xl, sheet_name="Sheet1", header=None)
            except ValueError:
                # _extract_test_results is called from both _parse_format1
                # and _parse_format3_multitrack.  Format 3 is defined as
                # Sheet1-absent with single-letter track sheets + 'Data
                # Table' (see _detect_format_from_sheets line 115), and its
                # test results are not in any single sheet -- so the
                # fallback would spam a misleading warning on every format3
                # file.  Detect that signature and return defaults without
                # warning; only fall back when the workbook actually looks
                # like a misrouted format1.
                names = xl.sheet_names or []
                looks_like_format3 = (
                    "Data Table" in names
                    and any(len(s) == 1 and s.isalpha() for s in names)
                    and "Sheet1" not in names   # mirror _detect_format_from_sheets line 115
                )
                if looks_like_format3:
                    return results
                if not names:
                    raise
                fallback_sheet = names[0]
                logger.warning(
                    f"'Sheet1' not found while reading test results, "
                    f"falling back to first sheet {fallback_sheet!r}"
                )
                df = pd.read_excel(xl, sheet_name=fallback_sheet, header=None)

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
                            results["station_linearity_pass"] = is_passed
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
            # abs() guards against inverted upper/lower limit columns.
            # A negative spec would invert downstream pass/fail logic and
            # produce sign-inverted error_to_spec ratios in ML features.
            return abs(avg_upper - avg_lower) / 2

        logger.warning(
            "FT linearity_spec defaulting to 0.01 — limit columns missing or all NaN."
        )
        return 0.01  # Default
