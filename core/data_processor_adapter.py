"""
Adapter to make DataProcessor compatible with GUI expectations
"""
from core.data_processor import DataProcessor, SystemType, UnitProperties, SigmaResults
from core.config import Config, ConfigManager
from pathlib import Path
from typing import Dict, Any, Optional


class LaserTrimDataProcessor(DataProcessor):
    """Adapter class for GUI compatibility"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else Config()

        # Apply configuration settings to data processor
        if hasattr(config, 'processing'):
            self.FILTER_SAMPLING_FREQ = config.processing.filter_sampling_freq
            self.FILTER_CUTOFF_FREQ = config.processing.filter_cutoff_freq
            self.GRADIENT_STEP = config.processing.gradient_step_size

    def analyze_file(self, file_path) -> Dict[str, Any]:
        """Adapter method for process_file with GUI expected format"""
        try:
            # Call parent process_file
            result = self.process_file(file_path)

            # Convert to GUI expected format
            gui_result = {
                'filename': Path(file_path).name,
                'file_info': result.get('file_info', {}),
                'tracks': result.get('tracks', {}),
                'overall_status': 'PASS',  # Default
                'analysis_results': {}
            }

            # Check if all tracks passed
            all_passed = True
            for track_id, track_data in result.get('tracks', {}).items():
                if 'sigma_results' in track_data:
                    if not track_data['sigma_results'].sigma_pass:
                        all_passed = False
                        break

            gui_result['overall_status'] = 'PASS' if all_passed else 'FAIL'

            # Add first track's results as analysis_results for backward compatibility
            first_track = list(result.get('tracks', {}).values())[0] if result.get('tracks') else {}
            if 'sigma_results' in first_track:
                gui_result['analysis_results'] = {
                    'sigma_gradient': first_track['sigma_results'].sigma_gradient,
                    'sigma_threshold': first_track['sigma_results'].sigma_threshold,
                    'sigma_pass': first_track['sigma_results'].sigma_pass
                }

            return gui_result

        except Exception as e:
            self.logger.error(f"Error analyzing file: {str(e)}")
            return {
                'filename': Path(file_path).name,
                'error': str(e),
                'overall_status': 'ERROR'
            }

    def process_folder(self, folder_path) -> Dict[str, Any]:
        """Process all files in a folder"""
        return self.batch_process(folder_path)