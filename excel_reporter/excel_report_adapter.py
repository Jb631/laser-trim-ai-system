"""
Adapter for Excel Report Generator compatibility
"""
from excel_reporter.excel_reporter import ExcelReporter
from typing import Dict, Any, List, Union
import logging


class ExcelReportGenerator(ExcelReporter):
    """Adapter class for GUI compatibility"""

    def __init__(self, config=None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def generate_report(self, results: Union[Dict, List], filename: str) -> str:
        """Generate report with GUI expected interface"""
        try:
            # Convert results to expected format
            report_data = {
                'file_results': [],
                'processing_summary': {
                    'total_files': 0,
                    'successful': 0,
                    'failed': 0
                }
            }

            # Handle both single result and list of results
            if isinstance(results, dict) and 'tracks' in results:
                results = [results]
            elif isinstance(results, dict) and not any(key in results for key in ['tracks', 'file_info', 'error']):
                # It's a dictionary of results
                results = list(results.values())
            elif not isinstance(results, list):
                results = [results]

            # Process each result
            for result in results:
                if isinstance(result, dict):
                    if 'error' in result and result['error']:
                        report_data['processing_summary']['failed'] += 1
                        continue

                    # Convert to report format
                    file_entry = self._convert_result_to_report_format(result)
                    if file_entry:
                        report_data['file_results'].append(file_entry)

                        # Update summary based on status
                        if file_entry.get('overall_status') == 'Pass':
                            report_data['processing_summary']['successful'] += 1
                        else:
                            report_data['processing_summary']['failed'] += 1

            report_data['processing_summary']['total_files'] = len(report_data['file_results'])

            # Call parent method
            return super().generate_report(report_data, filename, include_ai_insights=False)

        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise

    def _convert_result_to_report_format(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single result to report format"""
        file_entry = {
            'filename': result.get('filename', result.get('file_info', {}).get('filename', 'Unknown')),
            'model': 'Unknown',
            'serial': 'Unknown',
            'system': result.get('file_info', {}).get('system_type', 'Unknown'),
            'status': result.get('overall_status', 'Unknown'),
            'overall_status': 'Unknown'
        }

        # Extract model and serial from filename
        filename = file_entry['filename']
        if '_' in filename:
            parts = filename.split('_')
            file_entry['model'] = parts[0]
            if len(parts) > 1:
                file_entry['serial'] = parts[1].split('.')[0]

        # Add analysis results if available
        if 'analysis_results' in result:
            file_entry.update(result['analysis_results'])

        # Add track data if available
        if 'tracks' in result:
            # For multi-track files, we'll aggregate or use first track
            track_count = 0
            all_pass = True

            for track_id, track_data in result['tracks'].items():
                track_count += 1

                if 'sigma_results' in track_data:
                    # Use first track for main values
                    if track_count == 1:
                        file_entry['sigma_gradient'] = track_data['sigma_results'].sigma_gradient
                        file_entry['sigma_threshold'] = track_data['sigma_results'].sigma_threshold
                        file_entry['sigma_pass'] = track_data['sigma_results'].sigma_pass

                    # Check if any track failed
                    if not track_data['sigma_results'].sigma_pass:
                        all_pass = False

                # Add unit properties
                if 'unit_properties' in track_data and track_count == 1:
                    unit_props = track_data['unit_properties']
                    if hasattr(unit_props, 'unit_length'):
                        file_entry['unit_length'] = unit_props.unit_length
                    if hasattr(unit_props, 'linearity_spec'):
                        file_entry['linearity_spec'] = unit_props.linearity_spec

            file_entry['overall_status'] = 'Pass' if all_pass else 'Fail'
            file_entry['status'] = file_entry['overall_status']

            # If multi-track, store track details
            if track_count > 1:
                file_entry['tracks'] = {}
                for track_id, track_data in result['tracks'].items():
                    if 'sigma_results' in track_data:
                        file_entry['tracks'][track_id] = {
                            'sigma_gradient': track_data['sigma_results'].sigma_gradient,
                            'sigma_threshold': track_data['sigma_results'].sigma_threshold,
                            'sigma_pass': track_data['sigma_results'].sigma_pass,
                            'status': 'Pass' if track_data['sigma_results'].sigma_pass else 'Fail'
                        }

        return file_entry