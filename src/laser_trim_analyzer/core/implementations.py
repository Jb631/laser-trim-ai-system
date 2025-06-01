# laser_trim_analyzer/core/implementations.py
"""
Concrete implementations of the core interfaces.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import asyncio
from pathlib import Path
import logging
from datetime import datetime

from .interfaces import (
    TrimData, UnitProperties, SigmaMetrics, LinearityMetrics,
    TrimEffectiveness, SystemType, FileResult, TrackResult
)
from ..utils.filter_utils import apply_filter
from ..utils.excel_utils import extract_cell_value


class ExcelFileReader:
    """Implementation of FileReader for Excel files."""

    async def read_file(self, filepath: str) -> Dict[str, Any]:
        """Read Excel file and return sheet information."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._read_file_sync, filepath)

    def _read_file_sync(self, filepath: str) -> Dict[str, Any]:
        """Synchronous file reading."""
        try:
            excel_file = pd.ExcelFile(filepath)
            return {
                "filename": Path(filepath).name,
                "sheet_names": excel_file.sheet_names,
                "file_path": filepath
            }
        except Exception as e:
            raise IOError(f"Failed to read Excel file {filepath}: {e}")

    def get_sheet_names(self, filepath: str) -> List[str]:
        """Get list of sheet names in file."""
        try:
            excel_file = pd.ExcelFile(filepath)
            return excel_file.sheet_names
        except Exception as e:
            raise IOError(f"Failed to get sheet names from {filepath}: {e}")


class StandardDataExtractor:
    """Implementation of DataExtractor."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.system_mappings = self._get_system_mappings()

    async def extract_trim_data(
            self,
            filepath: str,
            sheet_name: str,
            system: SystemType
    ) -> TrimData:
        """Extract trim data from a sheet asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._extract_trim_data_sync, filepath, sheet_name, system
        )

    def _extract_trim_data_sync(
            self,
            filepath: str,
            sheet_name: str,
            system: SystemType
    ) -> TrimData:
        """Synchronous data extraction."""
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            columns = self.system_mappings[system.value]["columns"]

            # Find start row (skip headers)
            start_row = self._find_data_start_row(df, columns, system)

            # Extract data columns
            positions = pd.to_numeric(
                df.iloc[start_row:, columns["position"]],
                errors='coerce'
            ).tolist()

            errors = pd.to_numeric(
                df.iloc[start_row:, columns["error"]],
                errors='coerce'
            ).tolist()

            upper_limits = pd.to_numeric(
                df.iloc[start_row:, columns["upper_limit"]],
                errors='coerce'
            ).tolist()

            lower_limits = pd.to_numeric(
                df.iloc[start_row:, columns["lower_limit"]],
                errors='coerce'
            ).tolist()

            # Clean data - remove NaN rows
            cleaned_data = self._clean_trim_data(
                positions, errors, upper_limits, lower_limits
            )

            # Calculate full travel length before filtering
            if cleaned_data["positions"]:
                full_travel = max(cleaned_data["positions"]) - min(cleaned_data["positions"])
            else:
                full_travel = None

            # Apply end-point filtering if enough data
            if len(cleaned_data["positions"]) > 14:
                for key in cleaned_data:
                    cleaned_data[key] = cleaned_data[key][7:-7]

            return TrimData(
                positions=cleaned_data["positions"],
                errors=cleaned_data["errors"],
                upper_limits=cleaned_data["upper_limits"],
                lower_limits=cleaned_data["lower_limits"],
                sheet_name=sheet_name,
                full_travel_length=full_travel
            )

        except Exception as e:
            self.logger.error(f"Error extracting data from {sheet_name}: {e}")
            return TrimData([], [], [], [], sheet_name)

    async def extract_unit_properties(
            self,
            filepath: str,
            sheet_info: Dict[str, Any],
            system: SystemType
    ) -> UnitProperties:
        """Extract unit properties from sheets."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._extract_unit_properties_sync, filepath, sheet_info, system
        )

    def _extract_unit_properties_sync(
            self,
            filepath: str,
            sheet_info: Dict[str, Any],
            system: SystemType
    ) -> UnitProperties:
        """Synchronous property extraction."""
        mapping = self.system_mappings[system.value]
        cells = mapping["cells"]

        props = UnitProperties()

        # Extract from untrimmed sheet
        untrimmed_sheet = sheet_info.get("untrimmed")
        if untrimmed_sheet:
            # Unit length
            unit_length = extract_cell_value(
                filepath, untrimmed_sheet, cells["unit_length"], self.logger
            )
            if isinstance(unit_length, (int, float)) and unit_length > 0:
                props.unit_length = float(unit_length)

            # Untrimmed resistance
            if system == SystemType.SYSTEM_A:
                # Try B10 first for System A
                resistance = extract_cell_value(
                    filepath, untrimmed_sheet, "B10", self.logger
                )
            else:
                resistance = extract_cell_value(
                    filepath, untrimmed_sheet, cells["untrimmed_resistance"], self.logger
                )

            if isinstance(resistance, (int, float)) and resistance > 0:
                props.untrimmed_resistance = float(resistance)

        # Extract from trimmed sheet
        trimmed_sheet = sheet_info.get("final")
        if trimmed_sheet:
            if system == SystemType.SYSTEM_A:
                # Try B10 first for System A
                resistance = extract_cell_value(
                    filepath, trimmed_sheet, "B10", self.logger
                )
            else:
                resistance = extract_cell_value(
                    filepath, trimmed_sheet, cells["trimmed_resistance"], self.logger
                )

            if isinstance(resistance, (int, float)) and resistance > 0:
                props.trimmed_resistance = float(resistance)

        # Calculate resistance change
        if props.untrimmed_resistance and props.trimmed_resistance:
            props.resistance_change = props.trimmed_resistance - props.untrimmed_resistance
            props.resistance_change_percent = (
                    props.resistance_change / props.untrimmed_resistance * 100
            )

        return props

    def _find_data_start_row(
            self,
            df: pd.DataFrame,
            columns: Dict[str, int],
            system: SystemType
    ) -> int:
        """Find the first row with valid numeric data."""
        if system == SystemType.SYSTEM_A:
            # Look for first numeric value in position column
            for i in range(min(10, df.shape[0])):
                try:
                    val = df.iloc[i, columns["position"]]
                    if pd.notna(val) and isinstance(val, (int, float)):
                        return i
                except IndexError:
                    break
        return 0

    def _clean_trim_data(
            self,
            positions: List[float],
            errors: List[float],
            upper_limits: List[float],
            lower_limits: List[float]
    ) -> Dict[str, List[float]]:
        """Clean and filter trim data."""
        # Find valid indices
        valid_indices = []
        for i in range(len(positions)):
            if pd.notna(positions[i]) and pd.notna(errors[i]):
                valid_indices.append(i)

        # Create cleaned data
        cleaned = {
            "positions": [positions[i] for i in valid_indices],
            "errors": [errors[i] for i in valid_indices],
            "upper_limits": [
                upper_limits[i] if i < len(upper_limits) and pd.notna(upper_limits[i])
                else None for i in valid_indices
            ],
            "lower_limits": [
                lower_limits[i] if i < len(lower_limits) and pd.notna(lower_limits[i])
                else None for i in valid_indices
            ]
        }

        # Sort by position
        if cleaned["positions"]:
            sorted_indices = sorted(
                range(len(cleaned["positions"])),
                key=lambda i: cleaned["positions"][i]
            )
            for key in cleaned:
                cleaned[key] = [cleaned[key][i] for i in sorted_indices]

        return cleaned

    def _get_system_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Get system column and cell mappings."""
        return {
            "A": {
                "columns": {
                    "position": 7,  # Column H
                    "error": 6,  # Column G
                    "upper_limit": 8,  # Column I
                    "lower_limit": 9  # Column J
                },
                "cells": {
                    "unit_length": "B26",
                    "untrimmed_resistance": "B10",
                    "trimmed_resistance": "B10"
                }
            },
            "B": {
                "columns": {
                    "position": 8,  # Column I
                    "error": 3,  # Column D
                    "upper_limit": 5,  # Column F
                    "lower_limit": 6  # Column G
                },
                "cells": {
                    "unit_length": "K1",
                    "untrimmed_resistance": "R1",
                    "trimmed_resistance": "R1"
                }
            }
        }


class StandardMetricsCalculator:
    """Implementation of MetricsCalculator."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.default_scaling_factor = 24.0

    def calculate_sigma_metrics(
            self,
            data: TrimData,
            unit_properties: UnitProperties,
            model: str
    ) -> SigmaMetrics:
        """Calculate sigma gradient metrics."""
        # Apply filtering
        filtered_errors = apply_filter(data.errors)

        # Calculate gradients
        gradients = []
        step_size = 3  # MATLAB compatibility

        for i in range(len(data.positions) - step_size):
            dx = data.positions[i + step_size] - data.positions[i]
            dy = filtered_errors[i + step_size] - filtered_errors[i]
            if dx != 0:
                gradients.append(dy / dx)

        # Calculate sigma gradient
        sigma_gradient = np.std(gradients, ddof=1) if gradients else 0.0

        # Calculate threshold
        travel_length = data.full_travel_length or (
            max(data.positions) - min(data.positions) if data.positions else 0
        )

        # Determine linearity spec
        valid_upper = [u for u in data.upper_limits if u is not None]
        valid_lower = [l for l in data.lower_limits if l is not None]

        if valid_upper and valid_lower:
            linearity_spec = (np.mean(valid_upper) - np.mean(valid_lower)) / 2
        else:
            linearity_spec = max(abs(e) for e in data.errors) if data.errors else 0.01

        # Calculate threshold with model awareness
        sigma_threshold = self._calculate_model_aware_threshold(
            linearity_spec, travel_length, unit_properties.unit_length, model
        )

        # Special case for 8340-1
        if model.startswith("8340-1"):
            sigma_threshold = 0.4

        passed = sigma_gradient <= sigma_threshold
        margin = sigma_threshold - sigma_gradient

        return SigmaMetrics(
            gradient=sigma_gradient,
            threshold=sigma_threshold,
            passed=passed,
            margin=margin,
            scaling_factor=self.default_scaling_factor
        )

    def calculate_linearity_metrics(
            self,
            data: TrimData,
            spec: float
    ) -> LinearityMetrics:
        """Calculate linearity metrics with optimal offset."""
        # Calculate optimal offset
        optimal_offset = self._calculate_optimal_offset(
            data.errors, data.upper_limits, data.lower_limits
        )

        # Apply offset and check linearity
        shifted_errors = [e + optimal_offset for e in data.errors]

        fail_points = 0
        max_error = 0.0

        for i, error in enumerate(shifted_errors):
            if (data.upper_limits[i] is not None and
                    data.lower_limits[i] is not None):

                max_error = max(max_error, abs(error))

                if not (data.lower_limits[i] <= error <= data.upper_limits[i]):
                    fail_points += 1

        passed = fail_points == 0

        # Calculate deviation analysis
        positions = np.array(data.positions)
        errors = np.array(data.errors)

        slope, intercept = np.polyfit(positions, errors, 1)
        ideal_errors = slope * positions + intercept
        deviations = np.abs(errors - ideal_errors)

        max_dev_idx = np.argmax(deviations)

        return LinearityMetrics(
            spec=spec,
            optimal_offset=optimal_offset,
            final_error_raw=max(abs(e) for e in data.errors),
            final_error_shifted=max_error,
            passed=passed,
            fail_points=fail_points,
            max_deviation=deviations[max_dev_idx],
            max_deviation_position=positions[max_dev_idx]
        )

    def calculate_trim_effectiveness(
            self,
            untrimmed: TrimData,
            trimmed: TrimData
    ) -> TrimEffectiveness:
        """Calculate trim effectiveness metrics."""
        # RMS errors
        untrimmed_rms = np.sqrt(np.mean(np.square(untrimmed.errors)))
        trimmed_rms = np.sqrt(np.mean(np.square(trimmed.errors)))

        # Improvement percentage
        if untrimmed_rms > 0:
            improvement = ((untrimmed_rms - trimmed_rms) / untrimmed_rms) * 100
        else:
            improvement = 0.0

        # Max error reduction
        untrimmed_max = max(abs(e) for e in untrimmed.errors)
        trimmed_max = max(abs(e) for e in trimmed.errors)

        if untrimmed_max > 0:
            max_reduction = ((untrimmed_max - trimmed_max) / untrimmed_max) * 100
        else:
            max_reduction = 0.0

        return TrimEffectiveness(
            improvement_percent=improvement,
            untrimmed_rms_error=untrimmed_rms,
            trimmed_rms_error=trimmed_rms,
            max_error_reduction_percent=max_reduction
        )

    def calculate_failure_probability(
            self,
            sigma_metrics: SigmaMetrics,
            linearity_metrics: Optional[LinearityMetrics],
            unit_properties: UnitProperties
    ) -> float:
        """Calculate failure probability score."""
        # Normalized gradient (closer to 1 means closer to threshold)
        normalized_gradient = (
            sigma_metrics.gradient / sigma_metrics.threshold
            if sigma_metrics.threshold > 0 else 999
        )

        # Gradient margin (positive is good)
        gradient_margin = 1 - normalized_gradient

        # Weights for scoring
        weights = {
            "gradient_margin": 0.7,
            "linearity_spec": 0.3
        }

        # Calculate score (higher is better)
        linearity_factor = 0.02 / max(0.001, linearity_metrics.spec) if linearity_metrics else 1.0

        score = (
                weights["gradient_margin"] * gradient_margin +
                weights["linearity_spec"] * linearity_factor
        )

        # Convert to probability using sigmoid
        failure_probability = 1 / (1 + np.exp(2 * score))

        return failure_probability

    def _calculate_optimal_offset(
            self,
            errors: List[float],
            upper_limits: List[Optional[float]],
            lower_limits: List[Optional[float]]
    ) -> float:
        """Calculate optimal vertical offset for error data."""
        valid_errors = []
        midpoints = []

        for i in range(len(errors)):
            if upper_limits[i] is not None and lower_limits[i] is not None:
                valid_errors.append(errors[i])
                midpoints.append((upper_limits[i] + lower_limits[i]) / 2)

        if not valid_errors:
            return 0.0

        # Use median for robustness
        differences = np.array(valid_errors) - np.array(midpoints)
        return -np.median(differences)

    def _calculate_model_aware_threshold(
            self,
            linearity_spec: float,
            travel_length: float,
            unit_length: Optional[float],
            model: str
    ) -> float:
        """Calculate threshold with model-specific logic."""
        if model.startswith('8555'):
            # Empirical threshold for 8555
            base_threshold = 0.0025
            spec_factor = linearity_spec / 0.01 if linearity_spec > 0 else 1.0
            return base_threshold * spec_factor

        elif model.startswith('8340-1'):
            # Hard-coded for 8340-1
            return 0.4

        else:
            # Traditional calculation
            length_to_use = unit_length if unit_length else travel_length
            if length_to_use and length_to_use > 0:
                return (linearity_spec / length_to_use) * self.default_scaling_factor
            else:
                return self.default_scaling_factor * 0.02


class ExcelResultsFormatter:
    """Implementation of ResultsFormatter for Excel/HTML/Database outputs."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def format_for_excel(self, results: List[FileResult]) -> pd.DataFrame:
        """Format results for Excel export with flattened structure."""
        rows = []

        for file_result in results:
            # Base file information
            base_info = {
                'File': file_result.filename,
                'Model': file_result.model,
                'Serial': file_result.serial,
                'File Date': file_result.file_date.strftime('%Y-%m-%d'),
                'System': file_result.system.value,
                'Overall Status': file_result.overall_status.value
            }

            # Process each track
            for track_id, track in file_result.tracks.items():
                row = base_info.copy()
                row['Track ID'] = track_id
                row['Track Status'] = track.status.value

                # Sigma metrics
                row['Sigma Gradient'] = track.sigma_metrics.gradient
                row['Sigma Threshold'] = track.sigma_metrics.threshold
                row['Sigma Pass'] = track.sigma_metrics.passed

                # Unit properties
                row['Unit Length'] = track.unit_properties.unit_length
                row['Untrimmed Resistance'] = track.unit_properties.untrimmed_resistance
                row['Trimmed Resistance'] = track.unit_properties.trimmed_resistance
                row['Resistance Change'] = track.unit_properties.resistance_change
                row['Resistance Change (%)'] = track.unit_properties.resistance_change_percent

                # Linearity metrics
                if track.linearity_metrics:
                    row['Linearity Spec'] = track.linearity_metrics.spec
                    row['Optimal Offset'] = track.linearity_metrics.optimal_offset
                    row['Final Linearity Error (Raw)'] = track.linearity_metrics.final_error_raw
                    row['Final Linearity Error (Shifted)'] = track.linearity_metrics.final_error_shifted
                    row['Linearity Pass'] = track.linearity_metrics.passed
                    row['Linearity Fail Points'] = track.linearity_metrics.fail_points

                # Trim effectiveness
                if track.trim_effectiveness:
                    row['Trim Improvement (%)'] = track.trim_effectiveness.improvement_percent
                    row['Untrimmed RMS Error'] = track.trim_effectiveness.untrimmed_rms_error
                    row['Trimmed RMS Error'] = track.trim_effectiveness.trimmed_rms_error

                # Risk assessment
                row['Failure Probability'] = track.failure_probability
                row['Risk Category'] = track.risk_category.value

                # Additional fields
                row['Plot Path'] = track.plot_path
                row['Processing Time'] = file_result.processing_time

                rows.append(row)

        return pd.DataFrame(rows)

    def format_for_html(self, results: List[FileResult]) -> str:
        """Format results as HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Laser Trim Analysis Report</title>
            <meta charset="UTF-8">
            <style>{self._get_html_styles()}</style>
        </head>
        <body>
            <h1>🔬 Laser Trim Analysis Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            {self._generate_summary_section(results)}
            {self._generate_alerts_section(results)}
            {self._generate_detailed_results_section(results)}

            <footer>
                <p>Generated by Laser Trim Analyzer - Enhanced Edition</p>
            </footer>
        </body>
        </html>
        """
        return html

    def format_for_database(self, results: List[FileResult]) -> List[Dict[str, Any]]:
        """Format results for database storage."""
        db_records = []

        for file_result in results:
            # File-level record
            file_record = {
                'filename': file_result.filename,
                'file_path': file_result.filepath,
                'model': file_result.model,
                'serial': file_result.serial,
                'system': file_result.system.value,
                'has_multi_tracks': file_result.is_multi_track,
                'overall_status': file_result.overall_status.value,
                'processing_time': file_result.processing_time,
                'output_dir': file_result.output_directory,
                'timestamp': file_result.file_date,
                'tracks': {}
            }

            # Track-level records
            for track_id, track in file_result.tracks.items():
                track_record = {
                    'track_id': track_id,
                    'status': track.status.value,
                    'travel_length': track.untrimmed_data.full_travel_length,
                    'linearity_spec': track.sigma_metrics.threshold,
                    'sigma_gradient': track.sigma_metrics.gradient,
                    'sigma_threshold': track.sigma_metrics.threshold,
                    'sigma_pass': track.sigma_metrics.passed,
                    'unit_length': track.unit_properties.unit_length,
                    'untrimmed_resistance': track.unit_properties.untrimmed_resistance,
                    'trimmed_resistance': track.unit_properties.trimmed_resistance,
                    'resistance_change': track.unit_properties.resistance_change,
                    'resistance_change_percent': track.unit_properties.resistance_change_percent,
                    'failure_probability': track.failure_probability,
                    'risk_category': track.risk_category.value,
                    'plot_path': track.plot_path
                }

                # Add optional metrics
                if track.linearity_metrics:
                    track_record.update({
                        'optimal_offset': track.linearity_metrics.optimal_offset,
                        'final_linearity_error_raw': track.linearity_metrics.final_error_raw,
                        'final_linearity_error_shifted': track.linearity_metrics.final_error_shifted,
                        'linearity_pass': track.linearity_metrics.passed,
                        'linearity_fail_points': track.linearity_metrics.fail_points
                    })

                if track.trim_effectiveness:
                    track_record.update({
                        'trim_improvement_percent': track.trim_effectiveness.improvement_percent,
                        'untrimmed_rms_error': track.trim_effectiveness.untrimmed_rms_error,
                        'trimmed_rms_error': track.trim_effectiveness.trimmed_rms_error
                    })

                file_record['tracks'][track_id] = track_record

            db_records.append(file_record)

        return db_records

    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 20px; 
            line-height: 1.6; 
            color: #333; 
            background-color: #f9f9f9; 
        }
        h1, h2, h3 { color: #2c3e50; margin-top: 30px; }
        h1 { border-bottom: 3px solid #3498db; padding-bottom: 10px; }

        .summary-box { 
            border: 1px solid #ddd; 
            padding: 20px; 
            margin-bottom: 25px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        }

        .alert-box {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            color: #721c24; 
            border-radius: 8px; 
            padding: 20px; 
            margin-bottom: 25px; 
            border-left: 5px solid #dc3545;
        }

        .file-container { 
            background: white; 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            margin-bottom: 30px; 
            padding: 20px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }

        .status-pass { color: #27ae60; font-weight: bold; }
        .status-fail { color: #e74c3c; font-weight: bold; }
        .status-warning { color: #f39c12; font-weight: bold; }

        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin-top: 10px;
            background: white;
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: left; 
        }
        th { 
            background-color: #34495e; 
            color: white; 
        }
        tr:nth-child(even) { background-color: #f8f9fa; }
        """

    def _generate_summary_section(self, results: List[FileResult]) -> str:
        """Generate summary section for HTML report."""
        total_files = len(results)
        total_tracks = sum(len(r.tracks) for r in results)
        multi_track_files = sum(1 for r in results if r.is_multi_track)

        # Count status distribution
        pass_count = sum(1 for r in results for t in r.tracks.values()
                         if t.status.value == "Pass")
        fail_count = sum(1 for r in results for t in r.tracks.values()
                         if t.status.value == "Fail")

        # Risk distribution
        high_risk = sum(1 for r in results for t in r.tracks.values()
                        if t.risk_category.value == "High")

        return f"""
        <div class="summary-box">
            <h2>📊 Analysis Summary</h2>
            <p><strong>Total Files:</strong> {total_files}</p>
            <p><strong>Total Tracks:</strong> {total_tracks}</p>
            <p><strong>Multi-Track Files:</strong> {multi_track_files}</p>
            <p><strong>Pass/Fail:</strong> {pass_count} Pass / {fail_count} Fail</p>
            <p><strong>High Risk Units:</strong> {high_risk}</p>
        </div>
        """

    def _generate_alerts_section(self, results: List[FileResult]) -> str:
        """Generate alerts section for high-risk units."""
        alerts = []

        for file_result in results:
            for track_id, track in file_result.tracks.items():
                # Check for high risk
                if track.risk_category.value == "High":
                    alerts.append({
                        'type': 'high_risk',
                        'device': f"{file_result.filename} - {track_id}",
                        'message': f"High failure probability: {track.failure_probability:.2%}"
                    })

                # Check for 8340 models needing carbon screen check
                if (file_result.model.startswith("8340") and
                        not track.sigma_metrics.passed):
                    alerts.append({
                        'type': 'carbon_screen',
                        'device': f"{file_result.filename} - {track_id}",
                        'message': "Carbon screen check required"
                    })

        if not alerts:
            return ""

        html = '<div class="alert-box"><h2>⚠️ Maintenance Alerts</h2><ul>'
        for alert in alerts:
            html += f'<li><strong>{alert["device"]}:</strong> {alert["message"]}</li>'
        html += '</ul></div>'

        return html

    def _generate_detailed_results_section(self, results: List[FileResult]) -> str:
        """Generate detailed results section."""
        html = '<h2>📁 Detailed Results</h2>'

        for file_result in results:
            status_class = f"status-{file_result.overall_status.value.lower()}"

            html += f"""
            <div class="file-container">
                <h3>{file_result.filename}</h3>
                <p><strong>Model:</strong> {file_result.model} | 
                   <strong>Serial:</strong> {file_result.serial} | 
                   <strong>Status:</strong> <span class="{status_class}">
                   {file_result.overall_status.value}</span></p>

                <table>
                    <tr>
                        <th>Track</th>
                        <th>Status</th>
                        <th>Sigma Gradient</th>
                        <th>Sigma Pass</th>
                        <th>Linearity Pass</th>
                        <th>Risk Category</th>
                    </tr>
            """

            for track_id, track in file_result.tracks.items():
                track_status_class = f"status-{track.status.value.lower()}"
                linearity_pass = track.linearity_metrics.passed if track.linearity_metrics else "N/A"

                html += f"""
                    <tr>
                        <td>{track_id}</td>
                        <td class="{track_status_class}">{track.status.value}</td>
                        <td>{track.sigma_metrics.gradient:.6f}</td>
                        <td>{'✓' if track.sigma_metrics.passed else '✗'}</td>
                        <td>{'✓' if linearity_pass else '✗' if linearity_pass != "N/A" else "N/A"}</td>
                        <td>{track.risk_category.value}</td>
                    </tr>
                """

            html += '</table></div>'

        return html