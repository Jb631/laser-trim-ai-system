"""
Data Extractors Module

Handles extraction of measurement data and unit properties from Excel files
for both System A and System B formats.
"""

from ..constants import (
    SYSTEM_A, SYSTEM_B,
    SYSTEM_A_COLUMN_MAP,
    SYSTEM_B_COLUMN_MAP,
    UNIT_LENGTH_CELLS,
    RESISTANCE_CELLS
)
from ..utils.excel_utils import ExcelReader


class DataExtractor:
    """
    Extracts data from laser trim Excel files based on system type.

    Handles:
    - Position, error, and limit data extraction
    - Unit property extraction (length, resistance)
    - Multi-track support for System A
    - Different column mappings for each system
    """

    def __init__(self, system_type: str, logger: Optional[logging.Logger] = None):
        """
        Initialize data extractor for specific system type.

        Args:
            system_type: Either SYSTEM_A or SYSTEM_B
            logger: Logger instance
        """
        if system_type not in [SYSTEM_A, SYSTEM_B]:
            raise ValueError(f"Invalid system type: {system_type}")

        self.system_type = system_type
        self.logger = logger or logging.getLogger(__name__)
        self.excel_reader = ExcelReader(logger=self.logger)

        # Set column mapping based on system type
        self.column_map = (SYSTEM_A_COLUMN_MAP if system_type == SYSTEM_A
                           else SYSTEM_B_COLUMN_MAP)

    def extract_measurement_data(self, file_path: str, sheet_name: str) -> Dict[str, Any]:
        """
        Extract measurement data from a specific sheet.

        Args:
            file_path: Path to Excel file
            sheet_name: Name of sheet to extract from

        Returns:
            Dictionary containing:
                - position: List of position values
                - error: List of error values
                - upper_limit: List of upper limit values
                - lower_limit: List of lower limit values
                - travel_length: Calculated travel length
                - raw_data: Original DataFrame
        """
        self.logger.debug(f"Extracting data from sheet: {sheet_name}")

        try:
            # Read the sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Find data start row (skip headers)
            start_row = self._find_data_start_row(df)

            # Extract columns based on system mapping
            data = {
                'position': self._extract_column(df, 'position', start_row),
                'error': self._extract_column(df, 'error', start_row),
                'upper_limit': self._extract_column(df, 'upper_limit', start_row),
                'lower_limit': self._extract_column(df, 'lower_limit', start_row)
            }

            # Clean data (remove NaN rows)
            data = self._clean_measurement_data(data)

            # Calculate travel length
            if data['position']:
                data['travel_length'] = max(data['position']) - min(data['position'])
            else:
                data['travel_length'] = 0.0

            # Store raw data for reference
            data['raw_data'] = df

            self.logger.info(f"Extracted {len(data['position'])} data points from {sheet_name}")

            return data

        except Exception as e:
            self.logger.error(f"Error extracting data from {sheet_name}: {str(e)}")
            raise

    def extract_unit_properties(self, file_path: str, untrimmed_sheet: str,
                                trimmed_sheet: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract unit properties (length, resistance) from sheets.

        Args:
            file_path: Path to Excel file
            untrimmed_sheet: Name of untrimmed data sheet
            trimmed_sheet: Name of trimmed data sheet (optional)

        Returns:
            Dictionary containing:
                - unit_length: Unit length/angle value
                - untrimmed_resistance: Resistance before trimming
                - trimmed_resistance: Resistance after trimming
        """
        properties = {
            'unit_length': None,
            'untrimmed_resistance': None,
            'trimmed_resistance': None
        }

        try:
            # Extract unit length (from untrimmed sheet)
            if untrimmed_sheet:
                unit_length_cell = UNIT_LENGTH_CELLS.get(self.system_type)
                if unit_length_cell:
                    properties['unit_length'] = self.excel_reader.read_cell(
                        file_path, untrimmed_sheet, unit_length_cell
                    )
                    self.logger.debug(f"Unit length from {unit_length_cell}: {properties['unit_length']}")

                # Extract untrimmed resistance
                resistance_cell = RESISTANCE_CELLS.get(self.system_type, {}).get('untrimmed')
                if resistance_cell:
                    properties['untrimmed_resistance'] = self.excel_reader.read_cell(
                        file_path, untrimmed_sheet, resistance_cell
                    )

            # Extract trimmed resistance (from trimmed sheet if available)
            if trimmed_sheet:
                resistance_cell = RESISTANCE_CELLS.get(self.system_type, {}).get('trimmed')
                if resistance_cell:
                    properties['trimmed_resistance'] = self.excel_reader.read_cell(
                        file_path, trimmed_sheet, resistance_cell
                    )

            # Validate extracted values
            properties = self._validate_unit_properties(properties)

            return properties

        except Exception as e:
            self.logger.error(f"Error extracting unit properties: {str(e)}")
            return properties

    def _find_data_start_row(self, df: pd.DataFrame) -> int:
        """
        Find the row where actual data starts (after headers).

        Args:
            df: DataFrame to search

        Returns:
            Row index where data starts
        """
        # For System A, look for first numeric value in position column
        if self.system_type == SYSTEM_A:
            position_col = self.column_map['position']
            for i in range(min(10, len(df))):
                try:
                    val = df.iloc[i, position_col]
                    if pd.notna(val) and isinstance(val, (int, float)):
                        return i
                except:
                    continue

        # For System B, typically starts at row 0 or 1
        else:
            # Check if first row has numeric data
            try:
                error_col = self.column_map['error']
                val = df.iloc[0, error_col]
                if pd.notna(val) and isinstance(val, (int, float)):
                    return 0
            except:
                pass

        # Default to row 1 if not found
        return 1

    def _extract_column(self, df: pd.DataFrame, column_name: str,
                        start_row: int = 0) -> List[Optional[float]]:
        """
        Extract a specific column from DataFrame.

        Args:
            df: DataFrame to extract from
            column_name: Name of column in column_map
            start_row: Row to start extraction from

        Returns:
            List of values (with None for missing data)
        """
        col_index = self.column_map.get(column_name)
        if col_index is None or col_index >= len(df.columns):
            self.logger.warning(f"Column {column_name} not found or out of range")
            return []

        try:
            # Extract column and convert to numeric
            values = pd.to_numeric(df.iloc[start_row:, col_index], errors='coerce')
            return values.tolist()
        except Exception as e:
            self.logger.error(f"Error extracting column {column_name}: {str(e)}")
            return []

    def _clean_measurement_data(self, data: Dict[str, List]) -> Dict[str, List]:
        """
        Clean measurement data by removing rows with NaN position or error.

        Args:
            data: Dictionary with measurement data

        Returns:
            Cleaned data dictionary
        """
        # Find valid indices (non-NaN position and error)
        valid_indices = []

        for i in range(len(data.get('position', []))):
            if (i < len(data['position']) and
                    i < len(data['error']) and
                    pd.notna(data['position'][i]) and
                    pd.notna(data['error'][i])):
                valid_indices.append(i)

        # Create cleaned data
        cleaned = {}
        for key in ['position', 'error', 'upper_limit', 'lower_limit']:
            if key in data and data[key]:
                cleaned[key] = [data[key][i] for i in valid_indices
                                if i < len(data[key])]
            else:
                cleaned[key] = []

        # Sort by position
        if cleaned['position']:
            sorted_indices = sorted(range(len(cleaned['position'])),
                                    key=lambda i: cleaned['position'][i])

            for key in cleaned:
                if cleaned[key]:
                    cleaned[key] = [cleaned[key][i] for i in sorted_indices]

        self.logger.debug(f"Cleaned data: {len(data.get('position', []))} -> {len(cleaned['position'])} points")

        return cleaned

    def _validate_unit_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean unit property values.

        Args:
            properties: Dictionary with unit properties

        Returns:
            Validated properties dictionary
        """
        # Validate unit length
        unit_length = properties.get('unit_length')
        if unit_length is not None:
            try:
                unit_length = float(unit_length)
                # Check reasonable range (typically 50-400 degrees)
                if not (50 <= unit_length <= 400):
                    self.logger.warning(f"Unit length {unit_length} outside typical range (50-400)")
                properties['unit_length'] = unit_length
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid unit length value: {unit_length}")
                properties['unit_length'] = None

        # Validate resistances
        for key in ['untrimmed_resistance', 'trimmed_resistance']:
            value = properties.get(key)
            if value is not None:
                try:
                    value = float(value)
                    # Check reasonable range (typically 100-10000 ohms)
                    if not (100 <= value <= 10000):
                        self.logger.warning(f"{key} {value} outside typical range (100-10000)")
                    properties[key] = value
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid {key} value: {value}")
                    properties[key] = None

        return properties

    def extract_additional_metrics(self, file_path: str, sheet_name: str) -> Dict[str, Any]:
        """
        Extract additional metrics that might be present in the Excel file.

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet to extract from

        Returns:
            Dictionary with additional metrics
        """
        metrics = {}

        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # System-specific additional metrics
            if self.system_type == SYSTEM_A:
                # Look for measured volts, theory volts, etc.
                if 'measured_volts' in self.column_map:
                    col_idx = self.column_map['measured_volts']
                    if col_idx < len(df.columns):
                        metrics['measured_volts'] = df.iloc[:, col_idx].tolist()

                if 'theory_volts' in self.column_map:
                    col_idx = self.column_map['theory_volts']
                    if col_idx < len(df.columns):
                        metrics['theory_volts'] = df.iloc[:, col_idx].tolist()

            # Add more metrics as needed

        except Exception as e:
            self.logger.warning(f"Could not extract additional metrics: {str(e)}")

        return metrics