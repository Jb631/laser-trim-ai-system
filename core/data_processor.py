"""
Core Data Processing Engine for AI-Powered Laser Trim Analysis System

This module provides the foundation for loading, processing, and analyzing
laser trim data from Excel files with exact MATLAB calculation compatibility.

Author: QA Team
Date: 2024
Version: 1.0.0
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import warnings
from database import DatabaseManager

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class SystemType(Enum):
    """Enumeration for different measurement systems."""
    SYSTEM_A = "A"
    SYSTEM_B = "B"
    UNKNOWN = "Unknown"


@dataclass
class DataExtraction:
    """Container for extracted measurement data."""
    position: np.ndarray
    error: np.ndarray
    upper_limit: Optional[np.ndarray] = None
    lower_limit: Optional[np.ndarray] = None
    measured_volts: Optional[np.ndarray] = None
    theory_volts: Optional[np.ndarray] = None
    sheet_name: str = ""
    track_id: Optional[str] = None


@dataclass
class UnitProperties:
    """Container for unit-specific properties."""
    unit_length: Optional[float] = None
    untrimmed_resistance: Optional[float] = None
    trimmed_resistance: Optional[float] = None
    travel_length: Optional[float] = None
    linearity_spec: Optional[float] = None


@dataclass
class SigmaResults:
    """Container for sigma calculation results."""
    sigma_gradient: float
    sigma_threshold: float
    sigma_pass: bool
    gradients: np.ndarray
    filtered_error: np.ndarray
    gradient_positions: np.ndarray


class DataProcessor:
    """
    Core data processing engine for laser trim analysis.

    This class handles:
    - Excel file loading (both .xls and .xlsx)
    - System detection (A/B)
    - Multi-track support (TRK1/TRK2)
    - Sigma gradient calculations (MATLAB-compatible)
    - Data validation and error handling
    """

    # System A column mappings (0-based for Python)
    SYSTEM_A_COLUMNS = {
        'measured_volts': 3,  # Column D
        'index': 4,  # Column E
        'theory_volts': 5,  # Column F
        'error': 6,  # Column G
        'position': 7,  # Column H
        'upper_limit': 8,  # Column I
        'lower_limit': 9  # Column J
    }

    # System B column mappings
    SYSTEM_B_COLUMNS = {
        'error': 3,  # Column D
        'upper_limit': 5,  # Column F
        'lower_limit': 6,  # Column G
        'position': 8  # Column I
    }

    # Filter parameters (matching MATLAB implementation)
    FILTER_SAMPLING_FREQ = 100
    FILTER_CUTOFF_FREQ = 80
    GRADIENT_STEP = 3  # z_step in MATLAB

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the data processor.

        Args:
            logger: Optional logger instance for debugging
        """
        self.logger = logger or self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up default logger configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a single Excel file containing laser trim data.

        Args:
            file_path: Path to the Excel file

        Returns:
            Dictionary containing all extracted data and analysis results

        Raises:
            ValueError: If file doesn't exist or has invalid format
            Exception: For other processing errors
        """
        file_path = Path(file_path)

        # Validate file exists
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        # Validate file extension
        if file_path.suffix.lower() not in ['.xls', '.xlsx']:
            raise ValueError(f"Invalid file format: {file_path.suffix}")

        self.logger.info(f"Processing file: {file_path.name}")

        try:
            # Detect system type
            system_type = self._detect_system(file_path)
            self.logger.info(f"Detected system type: {system_type.value}")

            # Extract data based on system type
            if system_type == SystemType.SYSTEM_A:
                results = self._process_system_a(file_path)
            elif system_type == SystemType.SYSTEM_B:
                results = self._process_system_b(file_path)
            else:
                raise ValueError(f"Unknown system type for file: {file_path.name}")

            # Add file metadata
            results['file_info'] = {
                'filename': file_path.name,
                'filepath': str(file_path),
                'system_type': system_type.value,
                'processed_timestamp': pd.Timestamp.now().isoformat()
            }

            return results

        except Exception as e:
            self.logger.error(f"Error processing file {file_path.name}: {str(e)}")
            raise

    def _detect_system(self, file_path: Path) -> SystemType:
        """
        Detect which system type (A or B) the file belongs to.

        Args:
            file_path: Path to the Excel file

        Returns:
            SystemType enum value
        """
        try:
            # Read all sheet names
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names

            # Check for System A patterns
            if any('SEC1 TRK' in sheet for sheet in sheet_names):
                return SystemType.SYSTEM_A

            # Check for System B patterns
            if 'test' in sheet_names and 'Lin Error' in sheet_names:
                return SystemType.SYSTEM_B

            # Check filename patterns as fallback
            filename = file_path.name.upper()
            if any(pattern in filename for pattern in ['8340', '834']):
                return SystemType.SYSTEM_B
            elif any(pattern in filename for pattern in ['68', '78', '85']):
                return SystemType.SYSTEM_A

            return SystemType.UNKNOWN

        except Exception as e:
            self.logger.warning(f"Error detecting system type: {str(e)}")
            return SystemType.UNKNOWN

    def _process_system_a(self, file_path: Path) -> Dict[str, Any]:
        """
        Process System A files with multi-track support.

        Args:
            file_path: Path to the Excel file

        Returns:
            Dictionary containing processed data for all tracks
        """
        excel_file = pd.ExcelFile(file_path)
        results = {'tracks': {}}

        # Find all track sheets
        track_sheets = self._find_system_a_sheets(excel_file.sheet_names)

        for track_id, sheets in track_sheets.items():
            self.logger.info(f"Processing track: {track_id}")

            # Extract untrimmed data
            if sheets['untrimmed']:
                untrimmed_data = self._extract_system_a_data(
                    excel_file, sheets['untrimmed'], track_id
                )

                # Extract unit properties
                unit_props = self._extract_unit_properties_system_a(
                    excel_file, sheets['untrimmed']
                )

                # Calculate sigma gradient
                sigma_results = self._calculate_sigma_gradient(
                    untrimmed_data.position,
                    untrimmed_data.error,
                    unit_props
                )

                # Store results
                results['tracks'][track_id] = {
                    'untrimmed_data': untrimmed_data,
                    'unit_properties': unit_props,
                    'sigma_results': sigma_results,
                    'sheets': sheets
                }

                # Process trimmed data if available
                if sheets['trimmed']:
                    trimmed_data = self._extract_system_a_data(
                        excel_file, sheets['trimmed'], track_id
                    )
                    results['tracks'][track_id]['trimmed_data'] = trimmed_data

        return results

    def _process_system_b(self, file_path: Path) -> Dict[str, Any]:
        """
        Process System B files.

        Args:
            file_path: Path to the Excel file

        Returns:
            Dictionary containing processed data
        """
        excel_file = pd.ExcelFile(file_path)
        results = {'tracks': {}}

        # System B typically has single track
        track_id = 'default'

        # Find sheets
        if 'test' in excel_file.sheet_names:
            untrimmed_data = self._extract_system_b_data(excel_file, 'test')

            # Extract unit properties
            unit_props = self._extract_unit_properties_system_b(excel_file, 'test')

            # Calculate sigma gradient
            sigma_results = self._calculate_sigma_gradient(
                untrimmed_data.position,
                untrimmed_data.error,
                unit_props
            )

            results['tracks'][track_id] = {
                'untrimmed_data': untrimmed_data,
                'unit_properties': unit_props,
                'sigma_results': sigma_results
            }

            # Process final data if available
            if 'Lin Error' in excel_file.sheet_names:
                final_data = self._extract_system_b_data(excel_file, 'Lin Error')
                results['tracks'][track_id]['final_data'] = final_data

        return results

    def _find_system_a_sheets(self, sheet_names: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Find and organize System A sheets by track.

        Args:
            sheet_names: List of all sheet names in the Excel file

        Returns:
            Dictionary mapping track IDs to their sheets
        """
        tracks = {}

        # Look for TRK1 and TRK2 sheets
        for track_num in ['1', '2']:
            track_id = f'TRK{track_num}'
            untrimmed_pattern = f'SEC1 TRK{track_num} 0'
            trimmed_pattern = f'SEC1 TRK{track_num} TRM'

            sheets = {
                'untrimmed': None,
                'trimmed': None
            }

            for sheet in sheet_names:
                if untrimmed_pattern in sheet:
                    sheets['untrimmed'] = sheet
                elif trimmed_pattern in sheet:
                    sheets['trimmed'] = sheet

            if sheets['untrimmed']:  # Only add track if untrimmed data exists
                tracks[track_id] = sheets

        return tracks

    def _extract_system_a_data(
            self,
            excel_file: pd.ExcelFile,
            sheet_name: str,
            track_id: str
    ) -> DataExtraction:
        """
        Extract data from System A sheet.

        Args:
            excel_file: Pandas ExcelFile object
            sheet_name: Name of the sheet to extract from
            track_id: Track identifier (TRK1, TRK2)

        Returns:
            DataExtraction object containing the data
        """
        df = excel_file.parse(sheet_name, header=None)

        # Find data start row (skip headers)
        start_row = self._find_data_start_row(df, self.SYSTEM_A_COLUMNS['position'])

        # Extract columns
        data = DataExtraction(
            position=self._safe_extract_column(df, self.SYSTEM_A_COLUMNS['position'], start_row),
            error=self._safe_extract_column(df, self.SYSTEM_A_COLUMNS['error'], start_row),
            upper_limit=self._safe_extract_column(df, self.SYSTEM_A_COLUMNS['upper_limit'], start_row),
            lower_limit=self._safe_extract_column(df, self.SYSTEM_A_COLUMNS['lower_limit'], start_row),
            measured_volts=self._safe_extract_column(df, self.SYSTEM_A_COLUMNS['measured_volts'], start_row),
            theory_volts=self._safe_extract_column(df, self.SYSTEM_A_COLUMNS['theory_volts'], start_row),
            sheet_name=sheet_name,
            track_id=track_id
        )

        # Validate and clean data
        data = self._validate_and_clean_data(data)

        return data

    def _extract_system_b_data(self, excel_file: pd.ExcelFile, sheet_name: str) -> DataExtraction:
        """
        Extract data from System B sheet.

        Args:
            excel_file: Pandas ExcelFile object
            sheet_name: Name of the sheet to extract from

        Returns:
            DataExtraction object containing the data
        """
        df = excel_file.parse(sheet_name, header=None)

        # Find data start row
        start_row = self._find_data_start_row(df, self.SYSTEM_B_COLUMNS['position'])

        # Extract columns
        data = DataExtraction(
            position=self._safe_extract_column(df, self.SYSTEM_B_COLUMNS['position'], start_row),
            error=self._safe_extract_column(df, self.SYSTEM_B_COLUMNS['error'], start_row),
            upper_limit=self._safe_extract_column(df, self.SYSTEM_B_COLUMNS['upper_limit'], start_row),
            lower_limit=self._safe_extract_column(df, self.SYSTEM_B_COLUMNS['lower_limit'], start_row),
            sheet_name=sheet_name
        )

        # Validate and clean data
        data = self._validate_and_clean_data(data)

        return data

    def _extract_unit_properties_system_a(
            self,
            excel_file: pd.ExcelFile,
            sheet_name: str
    ) -> UnitProperties:
        """
        Extract unit properties from System A sheet.

        Args:
            excel_file: Pandas ExcelFile object
            sheet_name: Name of the sheet (typically untrimmed)

        Returns:
            UnitProperties object
        """
        df = excel_file.parse(sheet_name, header=None)

        props = UnitProperties()

        # Extract from specific cells
        try:
            # Unit length from B26
            props.unit_length = self._safe_extract_cell_value(df, 'B26')

            # Resistance from B10
            props.untrimmed_resistance = self._safe_extract_cell_value(df, 'B10')

        except Exception as e:
            self.logger.warning(f"Error extracting unit properties: {str(e)}")

        return props

    def _extract_unit_properties_system_b(
            self,
            excel_file: pd.ExcelFile,
            sheet_name: str
    ) -> UnitProperties:
        """
        Extract unit properties from System B sheet.

        Args:
            excel_file: Pandas ExcelFile object
            sheet_name: Name of the sheet

        Returns:
            UnitProperties object
        """
        df = excel_file.parse(sheet_name, header=None)

        props = UnitProperties()

        try:
            # Unit length from K1
            props.unit_length = self._safe_extract_cell_value(df, 'K1')

            # Resistance from R1
            props.untrimmed_resistance = self._safe_extract_cell_value(df, 'R1')

        except Exception as e:
            self.logger.warning(f"Error extracting unit properties: {str(e)}")

        return props

    def _calculate_sigma_gradient(
            self,
            position: np.ndarray,
            error: np.ndarray,
            unit_props: UnitProperties
    ) -> SigmaResults:
        """
        Calculate sigma gradient using MATLAB-compatible algorithm.

        This implements the exact calculations from the MATLAB code:
        1. Apply filter to error signal
        2. Calculate gradients with specified step size
        3. Calculate standard deviation of gradients
        4. Determine threshold and pass/fail status

        Args:
            position: Position array
            error: Error array
            unit_props: Unit properties for threshold calculation

        Returns:
            SigmaResults object containing all calculation results
        """
        # Apply filter to error signal (matching MATLAB's my_filtfiltfd2)
        filtered_error = self._apply_filter(error)

        # Calculate gradients with step size
        gradients = []
        gradient_positions = []

        for i in range(len(position) - self.GRADIENT_STEP):
            idx1 = i
            idx2 = i + self.GRADIENT_STEP

            dx = position[idx2] - position[idx1]
            dy = filtered_error[idx2] - filtered_error[idx1]

            if dx != 0:  # Avoid division by zero
                gradient = dy / dx
                gradients.append(gradient)
                gradient_positions.append(position[idx2])

        gradients = np.array(gradients)
        gradient_positions = np.array(gradient_positions)

        # Calculate sigma (standard deviation with ddof=1 for sample std)
        sigma_gradient = np.std(gradients, ddof=1) if len(gradients) > 0 else 0.0

        # Calculate threshold
        sigma_threshold = self._calculate_sigma_threshold(position, error, unit_props)

        # Determine pass/fail
        sigma_pass = sigma_gradient <= sigma_threshold

        return SigmaResults(
            sigma_gradient=sigma_gradient,
            sigma_threshold=sigma_threshold,
            sigma_pass=sigma_pass,
            gradients=gradients,
            filtered_error=filtered_error,
            gradient_positions=gradient_positions
        )

    def _apply_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply filter matching MATLAB's my_filtfiltfd2 implementation.

        This is a first-order digital filter applied forward and backward
        for zero phase distortion.

        Args:
            data: Input data array

        Returns:
            Filtered data array
        """
        alpha = self.FILTER_CUTOFF_FREQ / self.FILTER_SAMPLING_FREQ

        # Forward pass
        output = np.zeros_like(data, dtype=float)
        output[0] = data[0]

        for i in range(1, len(data)):
            output[i] = output[i - 1] + alpha * (data[i] - output[i - 1])

        # Backward pass
        output2 = np.zeros_like(data, dtype=float)
        output2[-1] = output[-1]

        for i in range(len(data) - 2, -1, -1):
            output2[i] = output2[i + 1] + alpha * (output[i] - output2[i + 1])

        return output2

    def _calculate_sigma_threshold(
            self,
            position: np.ndarray,
            error: np.ndarray,
            unit_props: UnitProperties
    ) -> float:
        """
        Calculate sigma threshold based on unit properties.

        Args:
            position: Position array
            error: Error array
            unit_props: Unit properties

        Returns:
            Calculated threshold value
        """
        # Calculate travel length if not provided
        if unit_props.travel_length is None:
            unit_props.travel_length = np.max(position) - np.min(position)

        # Calculate linearity spec if not provided
        if unit_props.linearity_spec is None:
            unit_props.linearity_spec = np.max(np.abs(error))

        # Use unit length if available, otherwise use travel length
        length = unit_props.unit_length if unit_props.unit_length else unit_props.travel_length

        # Default scaling factor (can be calibrated)
        scaling_factor = 24.0

        # Calculate threshold
        threshold = (unit_props.linearity_spec / length) * scaling_factor

        return threshold

    def _find_data_start_row(self, df: pd.DataFrame, position_col: int) -> int:
        """
        Find the row where actual data starts (skip headers).

        Args:
            df: DataFrame to search
            position_col: Column index for position data

        Returns:
            Row index where data starts
        """
        for i in range(min(20, len(df))):  # Check first 20 rows
            try:
                val = pd.to_numeric(df.iloc[i, position_col], errors='coerce')
                if pd.notna(val) and isinstance(val, (int, float)):
                    return i
            except:
                continue
        return 0

    def _safe_extract_column(
            self,
            df: pd.DataFrame,
            col_idx: int,
            start_row: int = 0
    ) -> np.ndarray:
        """
        Safely extract a column from DataFrame with error handling.

        Args:
            df: DataFrame to extract from
            col_idx: Column index
            start_row: Row to start extraction

        Returns:
            Numpy array of extracted values
        """
        try:
            if col_idx >= df.shape[1]:
                return np.array([])

            data = pd.to_numeric(df.iloc[start_row:, col_idx], errors='coerce')
            return data.dropna().values
        except Exception as e:
            self.logger.warning(f"Error extracting column {col_idx}: {str(e)}")
            return np.array([])

    def _safe_extract_cell_value(self, df: pd.DataFrame, cell_ref: str) -> Optional[float]:
        """
        Safely extract a value from a specific cell.

        Args:
            df: DataFrame to extract from
            cell_ref: Cell reference (e.g., 'B26')

        Returns:
            Extracted value or None
        """
        try:
            # Convert cell reference to row/col indices
            col_letter = ''.join(c for c in cell_ref if c.isalpha())
            row_num = int(''.join(c for c in cell_ref if c.isdigit()))

            # Convert column letter to index
            col_idx = 0
            for i, letter in enumerate(reversed(col_letter)):
                col_idx += (ord(letter.upper()) - ord('A') + 1) * (26 ** i)
            col_idx -= 1  # Zero-based index

            # Extract value
            value = df.iloc[row_num - 1, col_idx]
            return float(value) if pd.notna(value) else None

        except Exception as e:
            self.logger.warning(f"Error extracting cell {cell_ref}: {str(e)}")
            return None

    def _validate_and_clean_data(self, data: DataExtraction) -> DataExtraction:
        """
        Validate and clean extracted data.

        Args:
            data: DataExtraction object to validate

        Returns:
            Cleaned DataExtraction object
        """
        # Remove NaN values and ensure arrays have same length
        valid_indices = ~(np.isnan(data.position) | np.isnan(data.error))

        data.position = data.position[valid_indices]
        data.error = data.error[valid_indices]

        # Apply to optional arrays if they exist
        if data.upper_limit is not None and len(data.upper_limit) == len(valid_indices):
            data.upper_limit = data.upper_limit[valid_indices]

        if data.lower_limit is not None and len(data.lower_limit) == len(valid_indices):
            data.lower_limit = data.lower_limit[valid_indices]

        # Sort by position
        sort_indices = np.argsort(data.position)
        data.position = data.position[sort_indices]
        data.error = data.error[sort_indices]

        if data.upper_limit is not None:
            data.upper_limit = data.upper_limit[sort_indices]
        if data.lower_limit is not None:
            data.lower_limit = data.lower_limit[sort_indices]

        # Apply endpoint filtering (remove 7 points from each end)
        if len(data.position) > 14:
            data.position = data.position[7:-7]
            data.error = data.error[7:-7]
            if data.upper_limit is not None:
                data.upper_limit = data.upper_limit[7:-7]
            if data.lower_limit is not None:
                data.lower_limit = data.lower_limit[7:-7]

        return data

    def batch_process(self, folder_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process multiple files in a folder.

        Args:
            folder_path: Path to folder containing Excel files

        Returns:
            Dictionary containing results for all files
        """
        folder_path = Path(folder_path)

        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")

        results = {}
        excel_files = list(folder_path.glob('*.xls*'))

        self.logger.info(f"Found {len(excel_files)} Excel files to process")

        for file_path in excel_files:
            try:
                file_results = self.process_file(file_path)
                results[file_path.name] = file_results
            except Exception as e:
                self.logger.error(f"Failed to process {file_path.name}: {str(e)}")
                results[file_path.name] = {'error': str(e)}

        return results

    def export_results(self, results: Dict[str, Any], output_path: Union[str, Path]):
        """
        Export processing results to various formats.

        Args:
            results: Processing results dictionary
            output_path: Path for output file
        """
        output_path = Path(output_path)

        # Create summary DataFrame
        summary_data = []

        for filename, file_results in results.items():
            if 'error' in file_results:
                continue

            for track_id, track_data in file_results.get('tracks', {}).items():
                summary_data.append({
                    'filename': filename,
                    'system_type': file_results['file_info']['system_type'],
                    'track_id': track_id,
                    'sigma_gradient': track_data['sigma_results'].sigma_gradient,
                    'sigma_threshold': track_data['sigma_results'].sigma_threshold,
                    'sigma_pass': track_data['sigma_results'].sigma_pass,
                    'unit_length': track_data['unit_properties'].unit_length,
                    'travel_length': track_data['unit_properties'].travel_length,
                    'linearity_spec': track_data['unit_properties'].linearity_spec
                })

        # Save to Excel
        df = pd.DataFrame(summary_data)
        df.to_excel(output_path.with_suffix('.xlsx'), index=False)

        self.logger.info(f"Results exported to: {output_path.with_suffix('.xlsx')}")


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create processor instance
    processor = DataProcessor()

    # Example: Process a single file
    # result = processor.process_file("path/to/your/file.xlsx")
    # print(f"Sigma gradient: {result['tracks']['TRK1']['sigma_results'].sigma_gradient}")

    # Example: Process a folder
    # results = processor.batch_process("path/to/your/folder")
    # processor.export_results(results, "output/summary.xlsx")
    # In data_processor.py, add database integration:

    class LaserTrimDataProcessor:
        def __init__(self, config_file='config.json'):
            # Existing initialization...
            self.db_manager = DatabaseManager(self.config)

        def analyze_file(self, file_path):
            # Existing analysis...

            # Save to database
            if hasattr(self, 'db_manager'):
                run_id = getattr(self, 'current_run_id', None)
                if run_id:
                    self.db_manager.save_file_result(run_id, result)

            return result

        def process_folder(self, folder_path):
            # Create analysis run
            if hasattr(self, 'db_manager'):
                self.current_run_id = self.db_manager.create_analysis_run(
                    folder_path,
                    self.config.__dict__
                )

            # Existing processing...

            # Update run statistics
            if hasattr(self, 'db_manager') and hasattr(self, 'current_run_id'):
                self.db_manager.update_analysis_run(
                    self.current_run_id,
                    processed_files=len(results),
                    failed_files=len(errors),
                    total_files=total_files,
                    processing_time=time.time() - start_time
                )