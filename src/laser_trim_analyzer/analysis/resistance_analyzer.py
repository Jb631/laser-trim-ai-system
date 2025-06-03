"""
Resistance analyzer for laser trim data.

Analyzes resistance changes during the trim process.
"""

import time
from typing import Any, Dict, List, Optional
import pandas as pd

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
                - discovered_sheets: Optional[Dict] - Sheet names found by processor
                - untrimmed_sheet: Optional[str] - Sheet name for untrimmed data (fallback)
                - trimmed_sheet: Optional[str] - Sheet name for trimmed data (fallback)
                - system_type: Optional[str] - System type (A or B)
                - model: Optional[str] - Model identifier

        Returns:
            ResistanceAnalysis model with calculated metrics
        """
        start_time = time.time()

        # Try to get resistance values directly from data
        untrimmed_resistance = data.get('untrimmed_resistance')
        trimmed_resistance = data.get('trimmed_resistance')

        # If not provided, try to extract from Excel file using discovered sheets
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
        Extract resistance values from Excel file using discovered sheet names.

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
                untrimmed_cell = 'B10'
                trimmed_cell = 'B10'

            # Get sheet names - prefer discovered sheets from processor
            discovered_sheets = data.get('discovered_sheets', {})
            
            # Try to get actual sheet names from discovered sheets first
            untrimmed_sheet = None
            trimmed_sheet = None
            
            if discovered_sheets:
                untrimmed_sheet = discovered_sheets.get('untrimmed')
                trimmed_sheet = discovered_sheets.get('trimmed')
                self.logger.debug(f"Using discovered sheets: untrimmed='{untrimmed_sheet}', trimmed='{trimmed_sheet}'")
            
            # Fallback to provided sheet names if discovered sheets not available
            if not untrimmed_sheet:
                untrimmed_sheet = data.get('untrimmed_sheet')
            if not trimmed_sheet:
                trimmed_sheet = data.get('trimmed_sheet')
            
            # If still no sheet names, try to find them using fuzzy matching
            if not untrimmed_sheet or not trimmed_sheet:
                matched_sheets = self._find_sheets_by_fuzzy_matching(file_path, system_type)
                if not untrimmed_sheet:
                    untrimmed_sheet = matched_sheets.get('untrimmed')
                if not trimmed_sheet:
                    trimmed_sheet = matched_sheets.get('trimmed')

            # Extract from untrimmed sheet
            if untrimmed_sheet:
                extracted['untrimmed'] = self._read_cell_value(
                    file_path, untrimmed_sheet, untrimmed_cell
                )
                if extracted['untrimmed'] is not None:
                    self.logger.debug(f"Extracted untrimmed resistance: {extracted['untrimmed']} from sheet '{untrimmed_sheet}'")
            else:
                self.logger.warning(f"No untrimmed sheet found for file: {file_path}")

            # Extract from trimmed sheet
            if trimmed_sheet:
                extracted['trimmed'] = self._read_cell_value(
                    file_path, trimmed_sheet, trimmed_cell
                )
                if extracted['trimmed'] is not None:
                    self.logger.debug(f"Extracted trimmed resistance: {extracted['trimmed']} from sheet '{trimmed_sheet}'")
            else:
                self.logger.warning(f"No trimmed sheet found for file: {file_path}")

        except Exception as e:
            self.logger.error(f"Error extracting resistance from Excel: {e}")

        return extracted

    def _find_sheets_by_fuzzy_matching(self, file_path: str, system_type: str) -> Dict[str, Optional[str]]:
        """
        Find sheet names using fuzzy matching when exact names aren't available.
        
        Args:
            file_path: Path to Excel file
            system_type: System type (A or B)
            
        Returns:
            Dictionary with matched sheet names
        """
        matched = {'untrimmed': None, 'trimmed': None}
        
        try:
            # Get all sheet names from the file
            excel_file = pd.ExcelFile(file_path)
            available_sheets = excel_file.sheet_names
            
            if system_type == 'A':
                # Look for System A patterns
                untrimmed_patterns = ['trk1 0', 'sec1 trk1 0', 'trk 0']
                trimmed_patterns = ['trk1 trm', 'sec1 trk1', 'trm', 'trim']
            else:
                # Look for System B patterns
                untrimmed_patterns = ['test']
                trimmed_patterns = ['trim', 'lin error']
            
            # Find untrimmed sheet
            matched['untrimmed'] = self._find_closest_sheet_match(
                untrimmed_patterns, available_sheets
            )
            
            # Find trimmed sheet
            matched['trimmed'] = self._find_closest_sheet_match(
                trimmed_patterns, available_sheets
            )
            
            self.logger.debug(f"Fuzzy matching results: {matched} from sheets: {available_sheets}")
            
        except Exception as e:
            self.logger.error(f"Error in fuzzy sheet matching: {e}")
        
        return matched

    def _find_closest_sheet_match(self, patterns: List[str], available_sheets: List[str]) -> Optional[str]:
        """
        Find the closest matching sheet name from a list of patterns.
        
        Args:
            patterns: List of patterns to search for
            available_sheets: List of available sheet names
            
        Returns:
            Best matching sheet name or None
        """
        best_match = None
        best_score = 0
        
        for pattern in patterns:
            pattern_lower = pattern.lower()
            
            for sheet in available_sheets:
                sheet_lower = sheet.lower()
                
                # Exact match gets highest score
                if pattern_lower == sheet_lower:
                    return sheet
                
                # Partial match scoring
                score = 0
                if pattern_lower in sheet_lower:
                    score = len(pattern_lower) / len(sheet_lower)
                elif sheet_lower in pattern_lower:
                    score = len(sheet_lower) / len(pattern_lower) * 0.8
                
                # Bonus for containing key terms
                if 'trk' in pattern_lower and 'trk' in sheet_lower:
                    score += 0.2
                if '0' in pattern_lower and '0' in sheet_lower:
                    score += 0.1
                if 'trm' in pattern_lower and 'trm' in sheet_lower:
                    score += 0.2
                if 'trim' in pattern_lower and 'trim' in sheet_lower:
                    score += 0.2
                
                if score > best_score:
                    best_score = score
                    best_match = sheet
        
        return best_match if best_score > 0.3 else None

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

        # Check for reasonable resistance range
        # Handle both Ohm and kOhm units - typical pot values are 1kΩ to 100kΩ
        if untrimmed is not None:
            if untrimmed > 1000:  # Likely in Ohms
                if untrimmed < 100 or untrimmed > 1000000:  # 100Ω to 1MΩ
                    messages.append(
                        f"Untrimmed resistance {untrimmed} Ω outside typical range (100-1,000,000 Ω)"
                    )
            else:  # Likely in kOhms
                if untrimmed < 0.1 or untrimmed > 1000:  # 0.1kΩ to 1000kΩ
                    messages.append(
                        f"Untrimmed resistance {untrimmed} kΩ outside typical range (0.1-1000 kΩ)"
                    )

        return messages