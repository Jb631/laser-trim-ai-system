"""
Data Loader Module

Main orchestrator for loading and processing laser trim data files.
Coordinates system detection, data extraction, and sigma calculations.
"""

from core.config import Config
from ..constants import (
    SYSTEM_A, SYSTEM_B,
    SYSTEM_A_TRACKS
)
from .system_detector import SystemDetector
from .data_processor import DataExtractor
from .sigma_calculator import SigmaCalculator
from ..utils.excel_utils import ExcelReader
from .filter_utils import apply_matlab_filter


class DataLoader:
    """
    Main orchestrator for loading and processing laser trim data.

    This class coordinates:
    1. System detection (A or B)
    2. Sheet identification
    3. Data extraction
    4. Sigma calculations
    5. Unit property extraction
    """

    def __init__(self, config: Optional[Config] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the data loader.

        Args:
            config: Configuration object (uses default if None)
            logger: Logger instance (creates one if None)
        """
        self.config = config or Config()
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.system_detector = SystemDetector(logger=self.logger)
        self.excel_reader = ExcelReader(logger=self.logger)
        self.sigma_calculator = SigmaCalculator(
            filter_cutoff=self.config.filter_cutoff_frequency,
            filter_sampling_freq=self.config.filter_sampling_frequency,
            gradient_step=self.config.matlab_gradient_step,
            logger=self.logger
        )

    def load_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and process a single laser trim data file.

        Args:
            file_path: Path to the Excel file

        Returns:
            Dictionary containing all extracted and calculated data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.suffix.lower() in ['.xls', '.xlsx']:
            raise ValueError(f"Invalid file type: {file_path.suffix}. Expected .xls or .xlsx")

        self.logger.info(f"Loading file: {file_path.name}")

        try:
            # Step 1: Detect system type
            system_type = self.system_detector.detect_system(str(file_path))
            self.logger.info(f"Detected system type: {system_type}")

            # Step 2: Get sheet information
            sheets_info = self._identify_sheets(file_path, system_type)

            # Step 3: Initialize result structure
            result = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'system': system_type,
                'is_multi_track': system_type == SYSTEM_A and len(sheets_info.get('tracks', {})) > 1,
                'tracks': {},
                'metadata': self._extract_metadata(file_path)
            }

            # Step 4: Process based on system type
            if system_type == SYSTEM_A:
                result['tracks'] = self._process_system_a(file_path, sheets_info)
            else:  # SYSTEM_B
                result['tracks'] = self._process_system_b(file_path, sheets_info)

            # Step 5: Add summary statistics
            result['summary'] = self._calculate_summary(result['tracks'])

            self.logger.info(f"Successfully loaded {file_path.name}")
            return result

        except Exception as e:
            self.logger.error(f"Error loading {file_path.name}: {str(e)}")
            raise

    def _identify_sheets(self, file_path: Path, system_type: str) -> Dict[str, Any]:
        """
        Identify relevant sheets in the Excel file based on system type.

        Args:
            file_path: Path to Excel file
            system_type: Detected system type (A or B)

        Returns:
            Dictionary with sheet information
        """
        # Get all sheet names
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        if system_type == SYSTEM_A:
            return self._identify_system_a_sheets(sheet_names)
        else:
            return self._identify_system_b_sheets(sheet_names)

    def _identify_system_a_sheets(self, sheet_names: List[str]) -> Dict[str, Any]:
        """Identify System A sheets (multi-track support)."""
        sheets_info = {
            'tracks': {},
            'all_sheets': sheet_names
        }

        # Look for each track
        for track in SYSTEM_A_TRACKS:
            track_info = {
                'untrimmed': None,
                'trimmed': None,
                'all_trim_sheets': []
            }

            for sheet in sheet_names:
                sheet_upper = sheet.upper()

                # Check for untrimmed sheet (e.g., "SEC1 TRK1 0")
                if f"SEC1 {track} 0" in sheet_upper:
                    track_info['untrimmed'] = sheet

                # Check for trimmed sheets
                elif f"SEC1 {track}" in sheet_upper and ("TRM" in sheet_upper or "TRIM" in sheet_upper):
                    track_info['all_trim_sheets'].append(sheet)

            # Select final trimmed sheet (usually the last one)
            if track_info['all_trim_sheets']:
                track_info['trimmed'] = track_info['all_trim_sheets'][-1]

            # Only add track if we found data for it
            if track_info['untrimmed']:
                sheets_info['tracks'][track] = track_info

        return sheets_info

    def _identify_system_b_sheets(self, sheet_names: List[str]) -> Dict[str, Any]:
        """Identify System B sheets."""
        sheets_info = {
            'untrimmed': None,
            'final': None,
            'all_trim_sheets': [],
            'all_sheets': sheet_names
        }

        for sheet in sheet_names:
            if sheet == "test":
                sheets_info['untrimmed'] = sheet
            elif sheet == "Lin Error":
                sheets_info['final'] = sheet
            elif "Trim" in sheet:
                sheets_info['all_trim_sheets'].append(sheet)

        return sheets_info

    def _process_system_a(self, file_path: Path, sheets_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process System A file (potentially multi-track).

        Args:
            file_path: Path to Excel file
            sheets_info: Sheet identification info

        Returns:
            Dictionary with track data
        """
        tracks_data = {}
        data_extractor = DataExtractor(system_type=SYSTEM_A, logger=self.logger)

        for track_id, track_info in sheets_info['tracks'].items():
            self.logger.info(f"Processing track: {track_id}")

            try:
                # Extract untrimmed data
                untrimmed_data = data_extractor.extract_measurement_data(
                    str(file_path),
                    track_info['untrimmed']
                )

                # Extract trimmed data if available
                trimmed_data = None
                if track_info['trimmed']:
                    trimmed_data = data_extractor.extract_measurement_data(
                        str(file_path),
                        track_info['trimmed']
                    )

                # Extract unit properties
                unit_props = data_extractor.extract_unit_properties(
                    str(file_path),
                    track_info['untrimmed'],
                    track_info['trimmed']
                )

                # Calculate sigma gradient
                sigma_results = self._calculate_sigma(untrimmed_data)

                # Calculate linearity spec
                linearity_spec = self._calculate_linearity_spec(untrimmed_data)

                # Calculate sigma threshold
                sigma_threshold = self._calculate_sigma_threshold(
                    linearity_spec,
                    untrimmed_data['travel_length'],
                    unit_props.get('unit_length')
                )

                tracks_data[track_id] = {
                    'untrimmed_data': untrimmed_data,
                    'trimmed_data': trimmed_data,
                    'unit_properties': unit_props,
                    'sigma_gradient': sigma_results['sigma_gradient'],
                    'sigma_threshold': sigma_threshold,
                    'sigma_pass': sigma_results['sigma_gradient'] <= sigma_threshold,
                    'linearity_spec': linearity_spec,
                    'gradients': sigma_results['gradients'],
                    'gradient_positions': sigma_results['gradient_positions']
                }

            except Exception as e:
                self.logger.error(f"Error processing track {track_id}: {str(e)}")
                tracks_data[track_id] = {'error': str(e)}

        return tracks_data

    def _process_system_b(self, file_path: Path, sheets_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process System B file (single track).

        Args:
            file_path: Path to Excel file
            sheets_info: Sheet identification info

        Returns:
            Dictionary with track data
        """
        data_extractor = DataExtractor(system_type=SYSTEM_B, logger=self.logger)

        try:
            # For System B, we primarily use the final "Lin Error" sheet
            measurement_data = None
            if sheets_info['final']:
                measurement_data = data_extractor.extract_measurement_data(
                    str(file_path),
                    sheets_info['final']
                )
            elif sheets_info['untrimmed']:
                # Fallback to test sheet if Lin Error not found
                measurement_data = data_extractor.extract_measurement_data(
                    str(file_path),
                    sheets_info['untrimmed']
                )

            if not measurement_data:
                raise ValueError("No valid measurement data found")

            # Extract unit properties
            unit_props = data_extractor.extract_unit_properties(
                str(file_path),
                sheets_info['untrimmed'] or sheets_info['final'],
                sheets_info['final']
            )

            # Calculate sigma gradient
            sigma_results = self._calculate_sigma(measurement_data)

            # Calculate linearity spec
            linearity_spec = self._calculate_linearity_spec(measurement_data)

            # Calculate sigma threshold
            sigma_threshold = self._calculate_sigma_threshold(
                linearity_spec,
                measurement_data['travel_length'],
                unit_props.get('unit_length')
            )

            # System B has single track, use 'default' key
            return {
                'default': {
                    'measurement_data': measurement_data,
                    'unit_properties': unit_props,
                    'sigma_gradient': sigma_results['sigma_gradient'],
                    'sigma_threshold': sigma_threshold,
                    'sigma_pass': sigma_results['sigma_gradient'] <= sigma_threshold,
                    'linearity_spec': linearity_spec,
                    'gradients': sigma_results['gradients'],
                    'gradient_positions': sigma_results['gradient_positions']
                }
            }

        except Exception as e:
            self.logger.error(f"Error processing System B file: {str(e)}")
            return {'default': {'error': str(e)}}

    def _calculate_sigma(self, measurement_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate sigma gradient using the validated MATLAB algorithm.

        Args:
            measurement_data: Dictionary with position and error data

        Returns:
            Dictionary with sigma gradient and intermediate results
        """
        position = measurement_data['position']
        error = measurement_data['error']

        # Apply MATLAB filter
        filtered_error = apply_matlab_filter(
            error,
            cutoff_freq=self.config.filter_cutoff_frequency,
            sampling_freq=self.config.filter_sampling_frequency
        )

        # Calculate gradients
        gradients = []
        gradient_positions = []

        for i in range(len(position) - self.config.matlab_gradient_step):
            idx1 = i
            idx2 = i + self.config.matlab_gradient_step

            dx = position[idx2] - position[idx1]
            dy = filtered_error[idx2] - filtered_error[idx1]

            if dx != 0:  # Avoid division by zero
                gradient = dy / dx
                gradients.append(gradient)
                gradient_positions.append(position[idx2])

        # Calculate standard deviation (sigma)
        if gradients:
            sigma_gradient = np.std(gradients, ddof=1)  # Use sample standard deviation
        else:
            sigma_gradient = 0.0
            self.logger.warning("No gradients calculated - sigma set to 0")

        return {
            'sigma_gradient': sigma_gradient,
            'gradients': gradients,
            'gradient_positions': gradient_positions,
            'filtered_error': filtered_error
        }

    def _calculate_linearity_spec(self, measurement_data: Dict[str, Any]) -> float:
        """
        Calculate linearity specification from limit data.

        Args:
            measurement_data: Dictionary with measurement data

        Returns:
            Linearity specification value
        """
        upper_limit = measurement_data.get('upper_limit', [])
        lower_limit = measurement_data.get('lower_limit', [])

        # Filter out None values
        valid_upper = [val for val in upper_limit if val is not None]
        valid_lower = [val for val in lower_limit if val is not None]

        if valid_upper and valid_lower:
            # Calculate average limit width
            avg_upper = np.mean(valid_upper)
            avg_lower = np.mean(valid_lower)
            linearity_spec = (avg_upper - avg_lower) / 2
        else:
            # Fallback: use max absolute error
            error = measurement_data.get('error', [])
            linearity_spec = max(abs(e) for e in error) if error else 0.0
            self.logger.warning("Using max error as linearity spec (no valid limits)")

        return linearity_spec

    def _calculate_sigma_threshold(self, linearity_spec: float, travel_length: float,
                                   unit_length: Optional[float] = None) -> float:
        """
        Calculate sigma threshold based on specifications.

        Args:
            linearity_spec: Linearity specification
            travel_length: Total travel length
            unit_length: Unit length/angle (if available)

        Returns:
            Sigma threshold value
        """
        # Use unit length if available, otherwise use travel length
        length_to_use = unit_length if unit_length is not None else travel_length

        if length_to_use <= 0:
            self.logger.warning("Invalid length for threshold calculation, using default")
            return self.config.default_sigma_threshold

        # Formula: (linearity_spec / length) * scaling_factor
        threshold = (linearity_spec / length_to_use) * self.config.sigma_scaling_factor

        return threshold

    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract file metadata.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with metadata
        """
        import os
        from datetime import datetime

        stat = os.stat(file_path)

        # Extract model and serial from filename
        filename = file_path.stem
        parts = filename.split('_')

        model = parts[0] if parts else "Unknown"
        serial = parts[1] if len(parts) > 1 else "Unknown"

        return {
            'model': model,
            'serial': serial,
            'file_size': stat.st_size,
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat()
        }

    def _calculate_summary(self, tracks_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate summary statistics across all tracks.

        Args:
            tracks_data: Dictionary with all track data

        Returns:
            Summary statistics
        """
        summary = {
            'total_tracks': len(tracks_data),
            'passed_tracks': 0,
            'failed_tracks': 0,
            'avg_sigma_gradient': None,
            'max_sigma_gradient': None,
            'min_sigma_gradient': None
        }

        sigma_gradients = []

        for track_id, track_data in tracks_data.items():
            if 'error' in track_data:
                continue

            if 'sigma_gradient' in track_data:
                sigma_gradients.append(track_data['sigma_gradient'])

                if track_data.get('sigma_pass', False):
                    summary['passed_tracks'] += 1
                else:
                    summary['failed_tracks'] += 1

        if sigma_gradients:
            summary['avg_sigma_gradient'] = np.mean(sigma_gradients)
            summary['max_sigma_gradient'] = max(sigma_gradients)
            summary['min_sigma_gradient'] = min(sigma_gradients)

        return summary

    def load_batch(self, file_paths: List[Union[str, Path]],
                   progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Load and process multiple files.

        Args:
            file_paths: List of file paths to process
            progress_callback: Optional callback function(current, total, filename)

        Returns:
            List of results for each file
        """
        results = []
        total = len(file_paths)

        for i, file_path in enumerate(file_paths):
            try:
                if progress_callback:
                    progress_callback(i + 1, total, Path(file_path).name)

                result = self.load_file(file_path)
                results.append(result)

            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {str(e)}")
                results.append({
                    'file_name': Path(file_path).name,
                    'file_path': str(file_path),
                    'error': str(e)
                })

        return results