"""
Output Smoothness file parser for Laser Trim Analyzer.

Parses output smoothness test files (.xlsx/.xls) from the test station.
Filename pattern: model-snSerial[_Primary|_Redundant]_OS_date_time.xlsx
Sheets: Test Data, Report, Rev History
"""

import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

OS_EXTENSIONS = {'.xlsx', '.xls'}

# Filename pattern: model-snSerial[_element]_OS_date_time.ext
OS_FILENAME_PATTERN = re.compile(
    r'^(.+?)-sn(.+?)(?:_(Primary|Redundant))?_OS_'
    r'(\d{1,2}-\d{1,2}-\d{4})_'
    r'(\d{1,2}-\d{2}(?:-\d{2})?\s*(?:AM|PM))'
    r'\.(xlsx?)',
    re.IGNORECASE
)


def is_smoothness_file(filename: str) -> bool:
    """Check if a filename matches the output smoothness pattern."""
    return '_OS_' in filename and Path(filename).suffix.lower() in OS_EXTENSIONS


class SmoothnessParser:
    """Parser for Output Smoothness Excel files."""

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse an Output Smoothness Excel file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Parsing Output Smoothness file: {file_path.name}")

        file_hash = self._calculate_hash(file_path)
        metadata = self._parse_filename(file_path.name)
        metadata["filename"] = file_path.name
        metadata["file_path"] = str(file_path)

        tracks = []
        try:
            with pd.ExcelFile(file_path) as xl:
                sheet_names = xl.sheet_names

                if "Test Data" in sheet_names:
                    tracks = self._parse_test_data_sheet(xl)
                elif len(sheet_names) > 0:
                    tracks = self._parse_test_data_sheet(xl, sheet_name=sheet_names[0])

                if "Report" in sheet_names:
                    report_data = self._parse_report_sheet(xl)
                    metadata.update(report_data)

        except Exception as e:
            logger.error(f"Error parsing smoothness file {file_path.name}: {e}")
            raise

        return {
            "metadata": metadata,
            "tracks": tracks,
            "file_hash": file_hash,
        }

    def _parse_filename(self, filename: str) -> Dict[str, Any]:
        """Extract model, serial, element label, and date from filename."""
        result = {
            "model": "Unknown",
            "serial": "Unknown",
            "element_label": None,
            "file_date": None,
            "test_date": None,
        }

        match = OS_FILENAME_PATTERN.match(filename)
        if match:
            result["model"] = match.group(1)
            result["serial"] = match.group(2)
            result["element_label"] = match.group(3)

            try:
                date_str = match.group(4)
                result["file_date"] = datetime.strptime(date_str, "%m-%d-%Y")
            except ValueError:
                logger.warning(f"Could not parse date from filename: {filename}")

            try:
                time_str = match.group(5).strip()
                parts = time_str.replace(' ', '-').split('-')
                if len(parts) >= 4:
                    h, m, s, ampm = parts[0], parts[1], parts[2], parts[3]
                    time_parsed = datetime.strptime(f"{h}:{m}:{s} {ampm}", "%I:%M:%S %p")
                elif len(parts) == 3:
                    h, m, ampm = parts[0], parts[1], parts[2]
                    time_parsed = datetime.strptime(f"{h}:{m} {ampm}", "%I:%M %p")
                else:
                    time_parsed = None

                if time_parsed and result["file_date"]:
                    result["test_date"] = result["file_date"].replace(
                        hour=time_parsed.hour,
                        minute=time_parsed.minute,
                        second=time_parsed.second if len(parts) >= 4 else 0,
                    )
            except (ValueError, IndexError):
                logger.warning(f"Could not parse time from filename: {filename}")
        else:
            # Fallback: try to extract from _OS_ marker
            stem = Path(filename).stem
            os_idx = stem.find('_OS_')
            if os_idx > 0:
                prefix = stem[:os_idx]
                sn_match = re.match(r'^(.+?)-sn(.+?)(?:_(Primary|Redundant))?$', prefix, re.IGNORECASE)
                if sn_match:
                    result["model"] = sn_match.group(1)
                    result["serial"] = sn_match.group(2)
                    result["element_label"] = sn_match.group(3)

        return result

    def _parse_test_data_sheet(
        self, xl: pd.ExcelFile, sheet_name: str = "Test Data"
    ) -> List[Dict[str, Any]]:
        """Parse the Test Data sheet for time-series + smoothness summary.

        The Betatronix Output Smoothness file has a fixed shape:
          Row 1 header: ['Model Parameters', 'Model Number', '<model>', None,
                         'Time (s)', 'Filtered Volts (V)', None,
                         'Max. Deviation (V)', 'Max. Spec. Dev. (V)', 'Result']
          Row 2 contains the precomputed test results in cols H/I/J:
              H = Max. Deviation (V)   <- this IS the max smoothness value
              I = Max. Spec. Dev. (V)  <- the spec limit
              J = Result               <- "PASSED" / "FAILED"
          Rows 2..N have time-series data in cols E (Time) and F (Filtered Volts).

        We try this fixed layout first because it deterministically extracts the
        right values. If headers don't match (different vendor/format), we fall
        back to the older keyword-based detection so we don't lose data.
        """
        try:
            df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
        except Exception as e:
            logger.warning(f"Could not read sheet '{sheet_name}': {e}")
            return []

        if df.empty:
            return []

        # ---- Path 1: Betatronix fixed-layout fast path -------------------
        # Quick fingerprint: row 1 col E/F should say "Time" and "Filtered"
        try:
            r1_e = str(df.iloc[0, 4]).lower() if df.shape[1] > 4 else ""
            r1_f = str(df.iloc[0, 5]).lower() if df.shape[1] > 5 else ""
            r1_h = str(df.iloc[0, 7]).lower() if df.shape[1] > 7 else ""
            r1_i = str(df.iloc[0, 8]).lower() if df.shape[1] > 8 else ""
            is_betatronix = (
                "time" in r1_e
                and ("volt" in r1_f or "filtered" in r1_f)
                and "deviation" in r1_h
                and "spec" in r1_i
            )
        except Exception:
            is_betatronix = False

        if is_betatronix:
            track = self._parse_betatronix_layout(df)
            if track:
                return [track]
            # If the fast path fails for some reason, fall through to fallback.

        # ---- Path 2: Generic header search (legacy fallback) -------------
        # Used for non-Betatronix files (e.g. NI TDM exports). Same logic as
        # before but with an expanded keyword list so newer column names
        # ("time", "voltage", "deviation") are also matched.
        return self._parse_generic_layout(xl, sheet_name, df)

    def _parse_betatronix_layout(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Parse the fixed Betatronix smoothness layout from a header-less DataFrame.

        Returns one track dict, or None if values can't be extracted.
        Cols (0-indexed): 4=Time, 5=Filtered Volts, 7=Max Dev, 8=Spec, 9=Result.
        """
        try:
            # Row 2 (index 1) holds the precomputed results.
            max_dev = df.iloc[1, 7] if df.shape[0] > 1 and df.shape[1] > 7 else None
            spec = df.iloc[1, 8] if df.shape[0] > 1 and df.shape[1] > 8 else None
            result = df.iloc[1, 9] if df.shape[0] > 1 and df.shape[1] > 9 else None

            # Coerce numerics; a missing max_dev means the file is malformed.
            try:
                max_smoothness = float(max_dev) if pd.notna(max_dev) else None
            except (TypeError, ValueError):
                max_smoothness = None
            try:
                spec_value = float(spec) if pd.notna(spec) else None
            except (TypeError, ValueError):
                spec_value = None

            if max_smoothness is None:
                logger.warning("Betatronix layout matched but row-2 Max Deviation is missing/non-numeric")
                return None

            # Time-series for charting (cols E/F from row 2 onward).
            times = pd.to_numeric(df.iloc[1:, 4], errors='coerce')
            volts = pd.to_numeric(df.iloc[1:, 5], errors='coerce')
            mask = times.notna() & volts.notna()
            positions = times[mask].tolist()
            values = volts[mask].tolist()

            # Pass/fail: trust the file's own Result column when present.
            passes: Optional[bool]
            if isinstance(result, str):
                result_str = result.strip().lower()
                if result_str in ("pass", "passed", "ok"):
                    passes = True
                elif result_str in ("fail", "failed"):
                    passes = False
                else:
                    passes = None
            else:
                passes = (max_smoothness <= spec_value) if spec_value and spec_value > 0 else None

            avg_smoothness = (sum(abs(v) for v in values) / len(values)) if values else 0.0

            return {
                "track_id": "default",
                "positions": positions,
                "smoothness_values": values,
                "smoothness_spec": spec_value,
                "max_smoothness": max_smoothness,
                "avg_smoothness": avg_smoothness,
                "smoothness_pass": passes,
            }
        except Exception as e:
            logger.warning(f"Betatronix layout parse failed: {e}")
            return None

    def _parse_generic_layout(
        self, xl: pd.ExcelFile, sheet_name: str, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Legacy keyword-based parser for non-Betatronix files.

        Expanded keyword sets so common variations ("time", "voltage",
        "deviation", "ripple") are recognised.
        """
        tracks: List[Dict[str, Any]] = []
        position_keywords = ['position', 'angle', 'travel', 'degrees', 'inches', 'time']
        smooth_keywords = ['smooth', 'output', 'volt', 'voltage', 'deviation', 'ripple', 'filtered']

        # Find header row
        header_row = None
        for i in range(min(20, len(df))):
            row_values = [str(v).lower().strip() for v in df.iloc[i] if pd.notna(v)]
            if any(kw in val for val in row_values for kw in position_keywords):
                header_row = i
                break

        if header_row is None:
            # Last-resort: numeric columns assumption
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                positions = df[numeric_cols[0]].dropna().tolist()
                values = df[numeric_cols[1]].dropna().tolist()
                min_len = min(len(positions), len(values))
                if min_len > 0:
                    tracks.append({
                        "track_id": "default",
                        "positions": positions[:min_len],
                        "smoothness_values": values[:min_len],
                        "smoothness_spec": None,
                        "max_smoothness": max(values[:min_len]),
                        "avg_smoothness": sum(values[:min_len]) / min_len,
                        "smoothness_pass": None,
                    })
                else:
                    logger.warning(f"No usable numeric columns in sheet '{sheet_name}'")
            else:
                logger.warning(f"No header row and <2 numeric columns in sheet '{sheet_name}'")
            return tracks

        df_data = pd.read_excel(xl, sheet_name=sheet_name, header=header_row)
        df_data.columns = [str(c).strip() for c in df_data.columns]

        pos_col = None
        for col in df_data.columns:
            if any(kw in col.lower() for kw in position_keywords):
                pos_col = col
                break

        smooth_cols: List[str] = []
        spec_value = None
        for col in df_data.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in smooth_keywords):
                smooth_cols.append(col)
            if 'spec' in col_lower or 'limit' in col_lower:
                valid_vals = df_data[col].dropna()
                if len(valid_vals) > 0:
                    try:
                        spec_value = float(valid_vals.iloc[0])
                    except (ValueError, TypeError):
                        pass

        if not pos_col or not smooth_cols:
            logger.warning(
                f"Generic parser found no usable columns in sheet '{sheet_name}' "
                f"(pos_col={pos_col!r}, smooth_cols={smooth_cols})"
            )
            return tracks

        positions = pd.to_numeric(df_data[pos_col], errors='coerce').dropna().tolist()
        for sc in smooth_cols:
            values = pd.to_numeric(df_data[sc], errors='coerce').dropna().tolist()
            if not values:
                continue
            min_len = min(len(positions), len(values))
            pos = positions[:min_len]
            vals = values[:min_len]
            max_val = max(vals) if vals else 0
            avg_val = sum(vals) / len(vals) if vals else 0
            passes = max_val <= spec_value if spec_value and spec_value > 0 else None
            tracks.append({
                "track_id": "default",
                "positions": pos,
                "smoothness_values": vals,
                "smoothness_spec": spec_value,
                "max_smoothness": max_val,
                "avg_smoothness": avg_val,
                "smoothness_pass": passes,
            })
        return tracks

    def _parse_report_sheet(self, xl: pd.ExcelFile) -> Dict[str, Any]:
        """Parse the Report sheet for summary info."""
        result = {}
        try:
            df = pd.read_excel(xl, sheet_name="Report", header=None)
            if df.empty:
                return result

            for i in range(len(df)):
                for j in range(len(df.columns) - 1):
                    label = str(df.iloc[i, j]).strip().lower() if pd.notna(df.iloc[i, j]) else ""
                    value = df.iloc[i, j + 1] if j + 1 < len(df.columns) and pd.notna(df.iloc[i, j + 1]) else None
                    if value is None:
                        continue
                    if 'spec' in label and 'smooth' in label:
                        try:
                            result["smoothness_spec"] = float(value)
                        except (ValueError, TypeError):
                            pass
                    elif label in ('result', 'pass/fail', 'status'):
                        val_str = str(value).strip().lower()
                        result["overall_pass"] = val_str in ('pass', 'passed', 'ok', 'yes', 'true')

        except Exception as e:
            logger.warning(f"Could not parse Report sheet: {e}")

        return result

    @staticmethod
    def _calculate_hash(file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
