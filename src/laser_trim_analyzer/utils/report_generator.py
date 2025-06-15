"""
Report Generator for Laser Trim Analyzer

Generates comprehensive HTML, PDF, and Excel reports from analysis results.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import pandas as pd

from laser_trim_analyzer.core.models import AnalysisResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive reports from analysis results."""

    def __init__(self):
        """Initialize report generator."""
        self.logger = logger

    def generate_html_report(
        self,
        results: List[AnalysisResult],
        output_path: Path,
        title: str = "Laser Trim Analysis Report"
    ) -> None:
        """
        Generate HTML report from analysis results.
        
        Args:
            results: List of analysis results
            output_path: Path to save HTML report
            title: Report title
        """
        try:
            # Calculate summary statistics
            total_files = len(results)
            passed_files = sum(1 for r in results if r.overall_status.value == "PASS")
            failed_files = sum(1 for r in results if r.overall_status.value == "FAIL")
            warning_files = sum(1 for r in results if r.overall_status.value == "WARNING")
            
            # Generate HTML content
            html_content = self._generate_html_content(
                results, title, total_files, passed_files, failed_files, warning_files
            )
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            raise

    def _generate_html_content(
        self,
        results: List[AnalysisResult],
        title: str,
        total_files: int,
        passed_files: int,
        failed_files: int,
        warning_files: int
    ) -> str:
        """Generate HTML content for the report."""
        
        # Calculate pass rate
        pass_rate = (passed_files / total_files * 100) if total_files > 0 else 0
        
        # Determine analysis mode information
        lm_mode_files = 0
        standard_mode_files = 0
        for result in results:
            # Check if LM compliance mode was used (this would be logged in processing)
            if hasattr(result, 'processing_metadata') and result.processing_metadata:
                if result.processing_metadata.get('lm_compliance_mode', False):
                    lm_mode_files += 1
                else:
                    standard_mode_files += 1
            else:
                # Default assumption if no metadata
                standard_mode_files += 1
        
        # Generate results table rows
        table_rows = []
        for result in results:
            primary_track = result.primary_track
            
            # Get validation info
            validation_status = getattr(result, 'overall_validation_status', 'Not Available')
            validation_grade = getattr(result, 'validation_grade', 'N/A')
            
            # Get analysis mode info
            analysis_mode = "Standard"
            if hasattr(result, 'processing_metadata') and result.processing_metadata:
                if result.processing_metadata.get('lm_compliance_mode', False):
                    analysis_mode = "LM Compliance"
            
            row = f"""
            <tr>
                <td>{result.metadata.filename}</td>
                <td>{result.metadata.model}</td>
                <td>{result.metadata.serial}</td>
                <td><span class="status-{result.overall_status.value.lower()}">{result.overall_status.value}</span></td>
                <td><span class="mode-{analysis_mode.lower().replace(' ', '-')}">{analysis_mode}</span></td>
                <td><span class="validation-{validation_status.value.lower() if hasattr(validation_status, 'value') else 'unknown'}">{validation_status.value if hasattr(validation_status, 'value') else validation_status}</span></td>
                <td>{validation_grade}</td>
                <td>{primary_track.sigma_analysis.sigma_gradient:.4f}</td>
                <td>{'✓' if primary_track.sigma_analysis.sigma_pass else '✗'}</td>
                <td>{'✓' if primary_track.linearity_analysis.linearity_pass else '✗'}</td>
                <td>{result.processing_time:.2f}s</td>
            </tr>
            """
            table_rows.append(row)
        
        # Generate analysis mode summary
        mode_summary = ""
        if lm_mode_files > 0 or standard_mode_files > 0:
            mode_summary = f"""
                <div class="summary-card mode-info">
                    <h3>Analysis Modes</h3>
                    <div class="mode-breakdown">
                        <div>LM Compliance: <strong>{lm_mode_files}</strong></div>
                        <div>Standard: <strong>{standard_mode_files}</strong></div>
                    </div>
                </div>
            """
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                .summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .summary-card {{
                    background: #ecf0f1;
                    padding: 20px;
                    border-radius: 6px;
                    text-align: center;
                    border-left: 4px solid #3498db;
                }}
                .summary-card h3 {{
                    margin: 0 0 10px 0;
                    color: #2c3e50;
                }}
                .summary-card .value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #3498db;
                }}
                .pass-rate {{
                    border-left-color: #27ae60;
                }}
                .pass-rate .value {{
                    color: #27ae60;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .status-pass {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .status-fail {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .status-warning {{
                    color: #f39c12;
                    font-weight: bold;
                }}
                .validation-validated {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .validation-warning {{
                    color: #f39c12;
                    font-weight: bold;
                }}
                .validation-failed {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .validation-not_validated {{
                    color: #95a5a6;
                }}
                .mode-lm-compliance {{
                    color: #e67e22;
                    font-weight: bold;
                }}
                .mode-standard {{
                    color: #3498db;
                    font-weight: bold;
                }}
                .mode-info {{
                    border-left-color: #9b59b6;
                }}
                .mode-info .value {{
                    color: #9b59b6;
                }}
                .mode-breakdown {{
                    display: flex;
                    justify-content: space-between;
                    font-size: 0.9em;
                }}
                .footer {{
                    margin-top: 30px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                
                <div class="summary">
                    <div class="summary-card">
                        <h3>Total Files</h3>
                        <div class="value">{total_files}</div>
                    </div>
                    <div class="summary-card pass-rate">
                        <h3>Pass Rate</h3>
                        <div class="value">{pass_rate:.1f}%</div>
                    </div>
                    <div class="summary-card">
                        <h3>Passed</h3>
                        <div class="value" style="color: #27ae60;">{passed_files}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Failed</h3>
                        <div class="value" style="color: #e74c3c;">{failed_files}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Warnings</h3>
                        <div class="value" style="color: #f39c12;">{warning_files}</div>
                    </div>
                    {mode_summary}
                </div>

                <table>
                    <thead>
                        <tr>
                            <th>Filename</th>
                            <th>Model</th>
                            <th>Serial</th>
                            <th>Status</th>
                            <th>Mode</th>
                            <th>Validation</th>
                            <th>Grade</th>
                            <th>Sigma Gradient</th>
                            <th>Sigma Pass</th>
                            <th>Linearity Pass</th>
                            <th>Processing Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(table_rows)}
                    </tbody>
                </table>

                <div class="footer">
                    <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Laser Trim Analyzer v2.0</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content

    def generate_summary_report(
        self,
        results: List[AnalysisResult],
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Generate summary statistics report.
        
        Args:
            results: List of analysis results
            output_path: Path to save JSON summary
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            # Calculate statistics
            total_files = len(results)
            if total_files == 0:
                return {}
            
            passed_files = sum(1 for r in results if r.overall_status.value == "PASS")
            failed_files = sum(1 for r in results if r.overall_status.value == "FAIL")
            warning_files = sum(1 for r in results if r.overall_status.value == "WARNING")
            
            # Validation statistics
            validated_files = sum(1 for r in results 
                                if hasattr(r, 'overall_validation_status') and 
                                r.overall_validation_status.value == "VALIDATED")
            
            # Processing time statistics
            processing_times = [r.processing_time for r in results]
            avg_processing_time = sum(processing_times) / len(processing_times)
            
            # Model statistics
            models = {}
            for result in results:
                model = result.metadata.model
                if model not in models:
                    models[model] = {'total': 0, 'passed': 0, 'failed': 0, 'warnings': 0}
                
                models[model]['total'] += 1
                if result.overall_status.value == "PASS":
                    models[model]['passed'] += 1
                elif result.overall_status.value == "FAIL":
                    models[model]['failed'] += 1
                else:
                    models[model]['warnings'] += 1
            
            summary = {
                'report_generated': datetime.now().isoformat(),
                'total_files': total_files,
                'passed_files': passed_files,
                'failed_files': failed_files,
                'warning_files': warning_files,
                'pass_rate': (passed_files / total_files * 100) if total_files > 0 else 0,
                'validated_files': validated_files,
                'validation_rate': (validated_files / total_files * 100) if total_files > 0 else 0,
                'average_processing_time': avg_processing_time,
                'model_statistics': models
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Summary report generated: {output_path}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
            raise
    
    def generate_comprehensive_excel_report(
        self,
        results: List[AnalysisResult],
        output_path: Path,
        include_raw_data: bool = False
    ) -> None:
        """
        Generate comprehensive Excel report with all analysis data including ML predictions.
        
        Args:
            results: List of analysis results
            output_path: Path to save Excel file
            include_raw_data: Whether to include raw position/error data
        """
        try:
            # Check if we have any results
            if not results:
                # Create an empty report with a message
                empty_df = pd.DataFrame([{"Message": "No analysis results to report"}])
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    empty_df.to_excel(writer, sheet_name='No Results', index=False)
                self.logger.warning("No results to export, created empty report")
                return
                
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 1. Summary Sheet - Always create this first to ensure at least one sheet exists
                summary_data = []
                
                try:
                    for result in results:
                        primary_track = result.primary_track
                        
                        # Basic info
                        row = {
                        'Filename': result.metadata.filename,
                        'Model': result.metadata.model,
                        'Serial': result.metadata.serial,
                        'System Type': result.metadata.system_type.value,
                        'Overall Status': result.overall_status.value,
                        'Processing Time (s)': result.processing_time,
                        'Analysis Date': result.metadata.test_date.isoformat() if result.metadata.test_date else None,
                    }
                    
                    # Validation info
                    if hasattr(result, 'overall_validation_status'):
                        row['Validation Status'] = getattr(result.overall_validation_status, 'value', str(result.overall_validation_status))
                    if hasattr(result, 'validation_grade'):
                        row['Validation Grade'] = result.validation_grade
                    
                    # Sigma analysis
                    row.update({
                        'Sigma Gradient': primary_track.sigma_analysis.sigma_gradient,
                        'Sigma Threshold': primary_track.sigma_analysis.sigma_threshold,
                        'Sigma Pass': primary_track.sigma_analysis.sigma_pass,
                    })
                    
                    # Linearity analysis
                    row.update({
                        'Linearity Spec': primary_track.linearity_analysis.linearity_spec,
                        'Linearity Pass': primary_track.linearity_analysis.linearity_pass,
                        'Linearity Fail Points': primary_track.linearity_analysis.linearity_fail_points,
                        'Optimal Offset': primary_track.linearity_analysis.optimal_offset,
                        'Final Linearity Error': primary_track.linearity_analysis.final_linearity_error_shifted,
                    })
                    
                    # ML Failure Prediction
                    if primary_track.failure_prediction:
                        row.update({
                            'Risk Category': primary_track.failure_prediction.risk_category.value,
                            'Failure Probability': primary_track.failure_prediction.failure_probability,
                            'Gradient Margin': primary_track.failure_prediction.gradient_margin,
                            'Failure Contributing Factors': ', '.join(primary_track.failure_prediction.contributing_factors),
                        })
                    else:
                        row.update({
                            'Risk Category': None,
                            'Failure Probability': None,
                            'Gradient Margin': None,
                            'Failure Contributing Factors': None,
                        })
                    
                    # Advanced analytics (if available)
                    if hasattr(primary_track, 'resistance_analysis') and primary_track.resistance_analysis:
                        row.update({
                            'Min Resistance': primary_track.resistance_analysis.min_resistance,
                            'Max Resistance': primary_track.resistance_analysis.max_resistance,
                            'Resistance Range': primary_track.resistance_analysis.resistance_range,
                            'Normalized Range': primary_track.resistance_analysis.normalized_range,
                        })
                    
                    if hasattr(primary_track, 'consistency_metrics') and primary_track.consistency_metrics:
                        row.update({
                            'Trim Effectiveness': primary_track.consistency_metrics.trim_effectiveness,
                            'Trim Effectiveness Grade': primary_track.consistency_metrics.trim_effectiveness_grade,
                            'Noise Level': primary_track.consistency_metrics.noise_level,
                            'Dynamic Range': primary_track.consistency_metrics.dynamic_range,
                        })
                    
                        summary_data.append(row)
                        
                except Exception as e:
                    self.logger.error(f"Error building summary data: {e}")
                    # summary_data may be partially populated or empty
                
                # Write summary sheet - always write this to ensure at least one sheet exists
                summary_df = pd.DataFrame(summary_data)
                if summary_df.empty:
                    # Add at least one row to avoid empty sheet
                    summary_df = pd.DataFrame([{"Message": "No data available"}])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # 2. Detailed Analysis Sheet (one row per track)
                try:
                    detailed_data = []
                    for result in results:
                        for track_id, track_data in result.tracks.items():
                            row = {
                            'Filename': result.metadata.filename,
                            'Model': result.metadata.model,
                            'Serial': result.metadata.serial,
                            'Track ID': track_id,
                            'Track Status': track_data.status.value,
                        }
                        
                        # All track-specific data
                        row.update({
                            'Sigma Gradient': track_data.sigma_analysis.sigma_gradient,
                            'Sigma Threshold': track_data.sigma_analysis.sigma_threshold,
                            'Sigma Pass': track_data.sigma_analysis.sigma_pass,
                            'Linearity Spec': track_data.linearity_analysis.linearity_spec,
                            'Linearity Pass': track_data.linearity_analysis.linearity_pass,
                            'Linearity Fail Points': track_data.linearity_analysis.linearity_fail_points,
                            'Data Points': len(track_data.position_data),
                            'Untrimmed Points': len(track_data.untrimmed_positions) if track_data.untrimmed_positions else 0,
                        })
                        
                        # ML predictions
                        if track_data.failure_prediction:
                            row.update({
                                'Risk Category': track_data.failure_prediction.risk_category.value,
                                'Failure Probability': track_data.failure_prediction.failure_probability,
                                'Gradient Margin': track_data.failure_prediction.gradient_margin,
                                'Contributing Factors': ', '.join(track_data.failure_prediction.contributing_factors),
                            })
                        
                            detailed_data.append(row)
                            
                except Exception as e:
                    self.logger.warning(f"Error building detailed data: {e}")
                    # Continue with partial data
                
                detailed_df = pd.DataFrame(detailed_data)
                if detailed_df.empty:
                    # Add at least one row to avoid empty sheet
                    detailed_df = pd.DataFrame([{"Message": "No detailed data available"}])
                detailed_df.to_excel(writer, sheet_name='Detailed Analysis', index=False)
                
                # 3. Model Performance Sheet
                model_stats = {}
                for result in results:
                    model = result.metadata.model
                    if model not in model_stats:
                        model_stats[model] = {
                            'Total Files': 0,
                            'Passed': 0,
                            'Failed': 0,
                            'Warnings': 0,
                            'Low Risk': 0,
                            'Medium Risk': 0,
                            'High Risk': 0,
                            'Avg Sigma Gradient': [],
                            'Avg Failure Probability': [],
                        }
                    
                    model_stats[model]['Total Files'] += 1
                    
                    if result.overall_status.value == "PASS":
                        model_stats[model]['Passed'] += 1
                    elif result.overall_status.value == "FAIL":
                        model_stats[model]['Failed'] += 1
                    else:
                        model_stats[model]['Warnings'] += 1
                    
                    primary_track = result.primary_track
                    model_stats[model]['Avg Sigma Gradient'].append(primary_track.sigma_analysis.sigma_gradient)
                    
                    if primary_track.failure_prediction:
                        risk = primary_track.failure_prediction.risk_category.value
                        model_stats[model][f'{risk} Risk'] += 1
                        model_stats[model]['Avg Failure Probability'].append(
                            primary_track.failure_prediction.failure_probability
                        )
                
                # Calculate averages
                model_performance = []
                for model, stats in model_stats.items():
                    row = {
                        'Model': model,
                        'Total Files': stats['Total Files'],
                        'Passed': stats['Passed'],
                        'Failed': stats['Failed'],
                        'Warnings': stats['Warnings'],
                        'Pass Rate (%)': (stats['Passed'] / stats['Total Files'] * 100) if stats['Total Files'] > 0 else 0,
                        'Low Risk': stats['Low Risk'],
                        'Medium Risk': stats['Medium Risk'],
                        'High Risk': stats['High Risk'],
                        'Avg Sigma Gradient': sum(stats['Avg Sigma Gradient']) / len(stats['Avg Sigma Gradient']) if stats['Avg Sigma Gradient'] else None,
                        'Avg Failure Probability': sum(stats['Avg Failure Probability']) / len(stats['Avg Failure Probability']) if stats['Avg Failure Probability'] else None,
                    }
                    model_performance.append(row)
                
                model_df = pd.DataFrame(model_performance)
                model_df.to_excel(writer, sheet_name='Model Performance', index=False)
                
                # 4. Raw Data Sheets (optional)
                if include_raw_data and len(results) <= 10:  # Limit to 10 files for raw data
                    for i, result in enumerate(results[:10]):
                        sheet_name = f'Raw_{i+1}_{result.metadata.model}_{result.metadata.serial}'[:31]
                        
                        raw_data = []
                        for track_id, track_data in result.tracks.items():
                            for j, (pos, err) in enumerate(zip(track_data.position_data, track_data.error_data)):
                                raw_data.append({
                                    'Track': track_id,
                                    'Index': j,
                                    'Position': pos,
                                    'Error': err,
                                    'Type': 'Trimmed'
                                })
                            
                            if track_data.untrimmed_positions:
                                for j, (pos, err) in enumerate(zip(track_data.untrimmed_positions, track_data.untrimmed_errors)):
                                    raw_data.append({
                                        'Track': track_id,
                                        'Index': j,
                                        'Position': pos,
                                        'Error': err,
                                        'Type': 'Untrimmed'
                                    })
                        
                        if raw_data:
                            raw_df = pd.DataFrame(raw_data)
                            raw_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Format the Excel file
                try:
                    self._format_excel_file(writer)
                except Exception as e:
                    self.logger.warning(f"Could not format Excel file: {e}")
                    # Continue - formatting is optional
                
            self.logger.info(f"Comprehensive Excel report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate Excel report: {e}")
            raise
    
    def _format_excel_file(self, writer):
        """Apply formatting to the Excel file."""
        try:
            # For openpyxl writer, apply basic column width adjustments only
            # This prevents Excel corruption issues
            if hasattr(writer, 'sheets'):
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    
                    # Auto-fit columns (approximate) - safe operation
                    for column_cells in worksheet.columns:
                        max_length = 0
                        column_letter = column_cells[0].column_letter if column_cells else 'A'
                        
                        for cell in column_cells:
                            try:
                                if cell.value and len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        
                        # Set column width with reasonable limits
                        adjusted_width = max(8, min(max_length + 2, 50))
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                    
        except Exception as e:
            # Don't fail the export due to formatting issues
            self.logger.warning(f"Could not apply Excel formatting: {e}") 