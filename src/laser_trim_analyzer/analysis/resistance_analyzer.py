"""
Resistance analyzer for laser trim data.

Analyzes resistance changes during the trim process.
"""

import time
from typing import Any, Dict, List, Optionalimport pandas as pd

from laser_trim_analyzer.core.models import ResistanceAnalysis
from laser_trim_analyzer.analysis.base import BaseAnalyzer


class ResistanceAnalyzer(BaseAnalyzer):
    """
    Analyzer for resistance change metrics.

    Extracts and analyzes resistance values before and after trimming
    to assess the impact of the trim process.
    """

    def analyze(self, data: Dict[str, Any]) -> ResistanceAnalysis:
        """
        Analyze resistance changes from trim data.

        Args:
            data: Dictionary containing:
                - untrimmed_resistance: Optional[float] - Resistance before trim
                - trimmed_resistance: Optional[float] - Resistance after trim
                - file_path: Optional[str] - Excel file path for extraction
                - untrimmed_sheet: Optional[str] - Sheet name for untrimmed data
                - trimmed_sheet: Optional[str] - Sheet name for trimmed data
                - system_type: Optional[str] - System type (A or B)
                - model: Optional[str] - Model identifier

        Returns:
            ResistanceAnalysis model with calculated metrics
        """
        start_time = time.time()

        # Try to get resistance values directly from data
        untrimmed_resistance = data.get('untrimmed_resistance')
        trimmed_resistance = data.get('trimmed_resistance')

        # If not provided, try to extract from Excel file
        if (untrimmed_resistance is None or trimmed_resistance is None) and 'file_path' in data:
            extracted_values = self._extract_from_excel(data)
            if untrimmed_resistance is None:
                untrimmed_resistance = extracted_values.get('untrimmed')
            if trimmed_resistance is None:
                trimmed_resistance = extracted_values.get('trimmed')

        # Calculate resistance change metrics
        resistance_change = None
        resistance_change_percent = None

        if untrimmed_resistance is not None and trimmed_resistance is not None:
            resistance_change = trimmed_resistance - untrimmed_resistance

            if untrimmed_resistance > 0:
                resistance_change_percent = (resistance_change / untrimmed_resistance) * 100
            else:
                self.logger.warning("Untrimmed resistance is zero or negative")

        # Validate resistance values
        validation_messages = self._validate_resistance_values(
            untrimmed_resistance, trimmed_resistance, resistance_change_percent
        )

        if validation_messages:
            for msg in validation_messages:
                self.logger.warning(msg)

        # Create result
        result = ResistanceAnalysis(
            untrimmed_resistance=untrimmed_resistance,
            trimmed_resistance=trimmed_resistance,
            resistance_change=resistance_change,
            resistance_change_percent=resistance_change_percent
        )

        # Log summary
        processing_time = time.time() - start_time
        self.log_analysis_summary("Resistance", result, processing_time)

        return result

    def _extract_from_excel(self, data: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """
        Extract resistance values from Excel file.

        Args:
            data: Dictionary with file path and sheet information

        Returns:
            Dictionary with extracted resistance values
        """
        extracted = {'untrimmed': None, 'trimmed': None}

        try:
            file_path = data['file_path']
            system_type = data.get('system_type', 'A')
            model = data.get('model', '')

            # Determine cell locations based on system type
            if system_type == 'B':
                untrimmed_cell = 'R1'
                trimmed_cell = 'R1'
            else:  # System A
                # Special handling for 8340 models
                if model.startswith('8340'):
                    untrimmed_cell = 'B10'
                    trimmed_cell = 'B10'
                else:
                    untrimmed_cell = 'B10'
                    trimmed_cell = 'B10'

            # Extract from untrimmed sheet
            untrimmed_sheet = data.get('untrimmed_sheet')
            if untrimmed_sheet:
                extracted['untrimmed'] = self._read_cell_value(
                    file_path, untrimmed_sheet, untrimmed_cell
                )

            # Extract from trimmed sheet
            trimmed_sheet = data.get('trimmed_sheet')
            if trimmed_sheet:
                extracted['trimmed'] = self._read_cell_value(
                    file_path, trimmed_sheet, trimmed_cell
                )

        except Exception as e:
            self.logger.error(f"Error extracting resistance from Excel: {e}")

        return extracted

    def _read_cell_value(self, file_path: str, sheet_name: str,
                         cell_ref: str) -> Optional[float]:
        """
        Read a single cell value from Excel.

        Args:
            file_path: Path to Excel file
            sheet_name: Name of the sheet
            cell_ref: Cell reference (e.g., 'B10')

        Returns:
            Cell value as float, or None if not found
        """
        try:
            # Read the specific cell
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=None,
                usecols=cell_ref[0],  # Column letter
                nrows=int(cell_ref[1:])  # Row number
            )

            # Get the value
            row_idx = int(cell_ref[1:]) - 1
            col_idx = ord(cell_ref[0].upper()) - ord('A')

            if row_idx < len(df) and col_idx < len(df.columns):
                value = df.iloc[row_idx, col_idx]

                # Convert to float if possible
                if pd.notna(value):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        self.logger.warning(
                            f"Could not convert cell {cell_ref} value '{value}' to float"
                        )

        except Exception as e:
            self.logger.error(
                f"Error reading cell {cell_ref} from sheet {sheet_name}: {e}"
            )

        return None

    def _validate_resistance_values(self, untrimmed: Optional[float],
                                    trimmed: Optional[float],
                                    change_percent: Optional[float]) -> List[str]:
        """
        Validate resistance values for reasonableness.

        Args:
            untrimmed: Untrimmed resistance value
            trimmed: Trimmed resistance value
            change_percent: Percentage change

        Returns:
            List of validation warning messages
        """
        messages = []

        # Check for missing values
        if untrimmed is None:
            messages.append("Missing untrimmed resistance value")
        if trimmed is None:
            messages.append("Missing trimmed resistance value")

        # Check for negative values
        if untrimmed is not None and untrimmed <= 0:
            messages.append(f"Invalid untrimmed resistance: {untrimmed} (must be positive)")
        if trimmed is not None and trimmed <= 0:
            messages.append(f"Invalid trimmed resistance: {trimmed} (must be positive)")

        # Check for unreasonable change
        if change_percent is not None:
            if abs(change_percent) > 50:
                messages.append(
                    f"Unusually large resistance change: {change_percent:.1f}% "
                    "(typical range: -20% to +20%)"
                )

            # Check for unexpected direction
            if change_percent < -20:
                messages.append(
                    f"Large resistance decrease: {change_percent:.1f}% "
                    "(verify trim process)"
                )

        # Check for reasonable resistance range (assuming kOhm units)
        if untrimmed is not None and (untrimmed < 0.1 or untrimmed > 1000):
            messages.append(
                f"Untrimmed resistance {untrimmed} outside typical range (0.1-1000 kΩ)"
            )

        return messages