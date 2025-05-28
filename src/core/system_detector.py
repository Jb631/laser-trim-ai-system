"""
System detection module.

Automatically detects whether a file is from System A or System B
based on sheet names and structure.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.core.constants import (
    SYSTEM_A, SYSTEM_B,
    SYSTEM_A_CONFIG, SYSTEM_B_CONFIG,
    SUPPORTED_EXTENSIONS
)

logger = logging.getLogger(__name__)


class SystemDetector:
    """Detects measurement system type from Excel files."""

    def __init__(self):
        """Initialize system detector with known patterns."""
        self.system_a_patterns = [
            "SEC1 TRK1",
            "SEC1 TRK2",
            "TRK1",
            "TRK2"
        ]

        self.system_b_indicators = [
            "test",
            "Lin Error",
            "Trim"
        ]

    def detect_system(self, file_path: str) -> Optional[str]:
        """
        Detect which system (A or B) the file belongs to.

        Args:
            file_path: Path to Excel file

        Returns:
            System type (SYSTEM_A or SYSTEM_B) or None if cannot detect
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported file type: {file_path.suffix}")
            return None

        try:
            # Get sheet names
            sheet_names = self._get_sheet_names(file_path)

            if not sheet_names:
                logger.error(f"No sheets found in {file_path}")
                return None

            # Check for System A patterns
            system_a_score = self._check_system_a_patterns(sheet_names)

            # Check for System B patterns
            system_b_score = self._check_system_b_patterns(sheet_names)

            logger.debug(f"System detection scores - A: {system_a_score}, B: {system_b_score}")

            # Determine system based on scores
            if system_a_score > system_b_score:
                logger.info(f"Detected System A for {file_path.name}")
                return SYSTEM_A
            elif system_b_score > system_a_score:
                logger.info(f"Detected System B for {file_path.name}")
                return SYSTEM_B
            else:
                # Try filename patterns as tiebreaker
                return self._detect_from_filename(file_path)

        except Exception as e:
            logger.error(f"Error detecting system for {file_path}: {e}")
            return None

    def _get_sheet_names(self, file_path: Path) -> List[str]:
        """Get all sheet names from Excel file."""
        try:
            # Use ExcelFile for better performance
            with pd.ExcelFile(file_path) as xls:
                return xls.sheet_names
        except Exception as e:
            logger.error(f"Error reading sheet names: {e}")
            return []

    def _check_system_a_patterns(self, sheet_names: List[str]) -> int:
        """
        Check for System A patterns in sheet names.

        Returns score indicating likelihood of System A.
        """
        score = 0

        for sheet in sheet_names:
            sheet_upper = sheet.upper()

            # Check for track patterns
            for pattern in self.system_a_patterns:
                if pattern in sheet_upper:
                    score += 2

            # Check for specific System A sheet patterns
            if "SEC1" in sheet_upper and ("0" in sheet or "TRM" in sheet_upper):
                score += 3

        return score

    def _check_system_b_patterns(self, sheet_names: List[str]) -> int:
        """
        Check for System B patterns in sheet names.

        Returns score indicating likelihood of System B.
        """
        score = 0

        for sheet in sheet_names:
            sheet_lower = sheet.lower()

            # Exact match for "test" sheet
            if sheet_lower == "test":
                score += 5

            # Lin Error sheet is strong indicator
            if "lin error" in sheet_lower:
                score += 5

            # Trim sheets
            if sheet.startswith("Trim ") and sheet[5:].isdigit():
                score += 2

        return score

    def _detect_from_filename(self, file_path: Path) -> Optional[str]:
        """
        Try to detect system from filename patterns.

        This is a fallback when sheet analysis is inconclusive.
        """
        filename = file_path.stem.upper()

        # Common patterns in filenames
        if any(pattern in filename for pattern in ["8340", "834"]):
            # 8340 series typically System B
            logger.info(f"Detected System B from filename pattern: {file_path.name}")
            return SYSTEM_B
        elif any(pattern in filename for pattern in ["68", "78", "85"]):
            # These series typically System A
            logger.info(f"Detected System A from filename pattern: {file_path.name}")
            return SYSTEM_A

        logger.warning(f"Could not determine system for {file_path.name}")
        return None

    def get_sheet_mapping(self, file_path: str, system: str) -> Dict[str, Any]:
        """
        Get sheet mapping for the detected system.

        Args:
            file_path: Path to Excel file
            system: Detected system type

        Returns:
            Dictionary with sheet information
        """
        sheet_names = self._get_sheet_names(Path(file_path))

        if system == SYSTEM_A:
            return self._get_system_a_mapping(sheet_names)
        elif system == SYSTEM_B:
            return self._get_system_b_mapping(sheet_names)
        else:
            return {}

    def _get_system_a_mapping(self, sheet_names: List[str]) -> Dict[str, Any]:
        """Get sheet mapping for System A files."""
        mapping = {
            "tracks": {},
            "has_multiple_tracks": False
        }

        # Check for each track
        for track_id in ["TRK1", "TRK2"]:
            track_info = {
                "untrimmed": None,
                "trimmed": None,
                "all_trim_sheets": []
            }

            for sheet in sheet_names:
                sheet_upper = sheet.upper()

                # Untrimmed sheet (contains "0")
                if track_id in sheet_upper and " 0" in sheet:
                    track_info["untrimmed"] = sheet

                # Trimmed sheets
                elif track_id in sheet_upper and "TRM" in sheet_upper:
                    track_info["all_trim_sheets"].append(sheet)
                    # Use the last trim sheet as the final
                    track_info["trimmed"] = sheet

            # Only add track if we found data
            if track_info["untrimmed"]:
                mapping["tracks"][track_id] = track_info

        mapping["has_multiple_tracks"] = len(mapping["tracks"]) > 1

        return mapping

    def _get_system_b_mapping(self, sheet_names: List[str]) -> Dict[str, Any]:
        """Get sheet mapping for System B files."""
        mapping = {
            "untrimmed": None,
            "final": None,
            "trim_sheets": [],
            "has_multiple_tracks": False
        }

        for sheet in sheet_names:
            sheet_lower = sheet.lower()

            if sheet_lower == "test":
                mapping["untrimmed"] = sheet
            elif "lin error" in sheet_lower:
                mapping["final"] = sheet
            elif sheet.startswith("Trim "):
                mapping["trim_sheets"].append(sheet)

        # Sort trim sheets numerically
        mapping["trim_sheets"].sort(key=lambda x: int(x.split()[1]) if x.split()[1].isdigit() else 0)

        return mapping