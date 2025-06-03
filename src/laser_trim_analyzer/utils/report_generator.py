"""
Report Generator for Laser Trim Analyzer

Generates comprehensive HTML, PDF, and Excel reports from analysis results.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

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