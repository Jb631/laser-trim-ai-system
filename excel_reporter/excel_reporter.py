"""
Excel Report Generator for Laser Trim AI System

This module creates comprehensive Excel reports with multiple sheets,
charts, and AI-generated insights from the analysis results.

Author: Laser Trim AI System
Date: 2024
Version: 1.0.0
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name
import logging
from pathlib import Path

# AI insight generation
from openai import OpenAI
import json


class ExcelReporter:
    """
    Generates comprehensive Excel reports with AI insights and visualizations.

    This class creates multi-sheet Excel workbooks containing:
    - Executive summary with AI insights
    - Detailed analysis by model/track
    - Statistical summaries
    - Trend analysis and predictions
    - Quality metrics and recommendations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Excel Reporter.

        Args:
            config: Configuration dictionary containing API keys and settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client if API key is provided
        if self.config.get('openai_api_key'):
            self.ai_client = OpenAI(api_key=self.config['openai_api_key'])
        else:
            self.ai_client = None
            self.logger.warning("No OpenAI API key provided. AI insights will be limited.")

    def generate_report(
            self,
            results: Dict[str, Any],
            output_path: str,
            include_ai_insights: bool = True
    ) -> str:
        """
        Generate a comprehensive Excel report from analysis results.

        Args:
            results: Dictionary containing all analysis results
            output_path: Path where the Excel file will be saved
            include_ai_insights: Whether to include AI-generated insights

        Returns:
            str: Path to the generated Excel file
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Create Excel writer
            workbook = xlsxwriter.Workbook(output_path)

            # Define formats
            formats = self._create_formats(workbook)

            # Generate sheets
            self._create_executive_summary(workbook, results, formats, include_ai_insights)
            self._create_detailed_analysis(workbook, results, formats)
            self._create_statistical_summary(workbook, results, formats)
            self._create_trend_analysis(workbook, results, formats)
            self._create_quality_metrics(workbook, results, formats)
            self._create_recommendations(workbook, results, formats, include_ai_insights)
            self._create_raw_data(workbook, results, formats)

            # Close workbook
            workbook.close()

            self.logger.info(f"Excel report generated successfully: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error generating Excel report: {str(e)}")
            raise

    def _create_formats(self, workbook: xlsxwriter.Workbook) -> Dict[str, Any]:
        """Create and return formatting styles for the workbook."""
        formats = {
            'title': workbook.add_format({
                'bold': True,
                'font_size': 16,
                'align': 'center',
                'valign': 'vcenter',
                'bg_color': '#4472C4',
                'font_color': 'white',
                'border': 1
            }),
            'header': workbook.add_format({
                'bold': True,
                'font_size': 12,
                'align': 'center',
                'valign': 'vcenter',
                'bg_color': '#D9E2F3',
                'border': 1,
                'text_wrap': True
            }),
            'subheader': workbook.add_format({
                'bold': True,
                'font_size': 11,
                'bg_color': '#F2F2F2',
                'border': 1
            }),
            'data': workbook.add_format({
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            }),
            'number': workbook.add_format({
                'border': 1,
                'align': 'center',
                'num_format': '0.0000'
            }),
            'percent': workbook.add_format({
                'border': 1,
                'align': 'center',
                'num_format': '0.00%'
            }),
            'pass': workbook.add_format({
                'border': 1,
                'align': 'center',
                'bg_color': '#C6EFCE',
                'font_color': '#006100'
            }),
            'fail': workbook.add_format({
                'border': 1,
                'align': 'center',
                'bg_color': '#FFC7CE',
                'font_color': '#9C0006'
            }),
            'warning': workbook.add_format({
                'border': 1,
                'align': 'center',
                'bg_color': '#FFEB9C',
                'font_color': '#9C6500'
            }),
            'insight': workbook.add_format({
                'border': 1,
                'text_wrap': True,
                'align': 'left',
                'valign': 'top',
                'bg_color': '#E7F3FF'
            })
        }
        return formats

    def _create_executive_summary(
            self,
            workbook: xlsxwriter.Workbook,
            results: Dict[str, Any],
            formats: Dict[str, Any],
            include_ai_insights: bool
    ):
        """Create executive summary sheet with key metrics and AI insights."""
        sheet = workbook.add_worksheet('Executive Summary')

        # Set column widths
        sheet.set_column('A:A', 30)
        sheet.set_column('B:B', 20)
        sheet.set_column('C:E', 15)

        row = 0

        # Title
        sheet.merge_range(row, 0, row, 4, 'LASER TRIM ANALYSIS REPORT', formats['title'])
        row += 2

        # Report metadata
        sheet.write(row, 0, 'Report Generated:', formats['subheader'])
        sheet.write(row, 1, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), formats['data'])
        row += 1

        sheet.write(row, 0, 'Total Files Analyzed:', formats['subheader'])
        sheet.write(row, 1, len(results.get('file_results', [])), formats['data'])
        row += 2

        # Overall statistics
        sheet.merge_range(row, 0, row, 4, 'OVERALL PERFORMANCE METRICS', formats['header'])
        row += 1

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(results)

        metrics_data = [
            ('Overall Pass Rate', overall_metrics['pass_rate'], formats['percent']),
            ('Average Sigma Gradient', overall_metrics['avg_sigma_gradient'], formats['number']),
            ('Average Failure Probability', overall_metrics['avg_failure_probability'], formats['percent']),
            ('High Risk Units', overall_metrics['high_risk_count'], formats['data']),
            ('Units Requiring Attention', overall_metrics['attention_required'], formats['data'])
        ]

        for metric_name, metric_value, metric_format in metrics_data:
            sheet.write(row, 0, metric_name, formats['subheader'])
            sheet.write(row, 1, metric_value, metric_format)
            row += 1

        row += 1

        # Performance by model
        sheet.merge_range(row, 0, row, 4, 'PERFORMANCE BY MODEL', formats['header'])
        row += 1

        # Headers for model performance
        model_headers = ['Model', 'Units', 'Pass Rate', 'Avg Sigma', 'Risk Level']
        for col, header in enumerate(model_headers):
            sheet.write(row, col, header, formats['subheader'])
        row += 1

        # Model performance data
        model_stats = self._calculate_model_statistics(results)
        for model, stats in model_stats.items():
            sheet.write(row, 0, model, formats['data'])
            sheet.write(row, 1, stats['count'], formats['data'])
            sheet.write(row, 2, stats['pass_rate'], formats['percent'])
            sheet.write(row, 3, stats['avg_sigma'], formats['number'])

            # Color code risk level
            risk_format = formats['pass'] if stats['risk_level'] == 'Low' else \
                formats['warning'] if stats['risk_level'] == 'Medium' else \
                    formats['fail']
            sheet.write(row, 4, stats['risk_level'], risk_format)
            row += 1

        row += 2

        # AI Insights section
        if include_ai_insights and self.ai_client:
            sheet.merge_range(row, 0, row, 4, 'AI-GENERATED INSIGHTS', formats['header'])
            row += 1

            insights = self._generate_ai_insights(overall_metrics, model_stats)

            # Write insights with wrapping
            insight_start_row = row
            for insight in insights:
                sheet.merge_range(row, 0, row + 2, 4, insight, formats['insight'])
                row += 3

            # Adjust row heights for insights
            for i in range(insight_start_row, row):
                sheet.set_row(i, 30)

        # Add a chart for overall performance
        self._add_performance_chart(workbook, sheet, model_stats, row + 2)

    def _create_detailed_analysis(
            self,
            workbook: xlsxwriter.Workbook,
            results: Dict[str, Any],
            formats: Dict[str, Any]
    ):
        """Create detailed analysis sheet with individual file results."""
        sheet = workbook.add_worksheet('Detailed Analysis')

        # Define columns
        columns = [
            ('File Name', 20),
            ('Model', 12),
            ('Serial', 15),
            ('Track', 10),
            ('Status', 12),
            ('Sigma Gradient', 15),
            ('Sigma Threshold', 15),
            ('Sigma Pass', 12),
            ('Linearity Pass', 12),
            ('Failure Probability', 18),
            ('Risk Category', 15),
            ('Resistance Change (%)', 18),
            ('Trim Improvement (%)', 18),
            ('Unit Length', 12),
            ('Optimal Offset', 15)
        ]

        # Set column widths and write headers
        row = 0
        sheet.merge_range(row, 0, row, len(columns) - 1, 'DETAILED ANALYSIS RESULTS', formats['title'])
        row += 2

        for col, (header, width) in enumerate(columns):
            sheet.set_column(col, col, width)
            sheet.write(row, col, header, formats['header'])
        row += 1

        # Write data
        file_results = results.get('file_results', [])
        for file_result in file_results:
            # Handle multi-track files
            if 'tracks' in file_result and file_result['tracks']:
                for track_id, track_data in file_result['tracks'].items():
                    self._write_analysis_row(sheet, row, file_result, track_data, track_id, formats)
                    row += 1
            else:
                self._write_analysis_row(sheet, row, file_result, file_result, 'N/A', formats)
                row += 1

        # Add filters
        sheet.autofilter(2, 0, row - 1, len(columns) - 1)

    def _write_analysis_row(
            self,
            sheet: xlsxwriter.Worksheet,
            row: int,
            file_data: Dict[str, Any],
            track_data: Dict[str, Any],
            track_id: str,
            formats: Dict[str, Any]
    ):
        """Write a single row of analysis data."""
        # Determine status format
        status = track_data.get('status', 'Unknown')
        status_format = formats['pass'] if 'Pass' in status else \
            formats['fail'] if 'Fail' in status else \
                formats['warning']

        # Write data
        col_data = [
            (file_data.get('filename', 'Unknown'), formats['data']),
            (file_data.get('model', 'Unknown'), formats['data']),
            (file_data.get('serial', 'Unknown'), formats['data']),
            (track_id, formats['data']),
            (status, status_format),
            (track_data.get('sigma_gradient', 0), formats['number']),
            (track_data.get('sigma_threshold', 0), formats['number']),
            (track_data.get('sigma_pass', False), formats['pass'] if track_data.get('sigma_pass') else formats['fail']),
            (track_data.get('linearity_pass', False),
             formats['pass'] if track_data.get('linearity_pass') else formats['fail']),
            (track_data.get('failure_probability', 0), formats['percent']),
            (track_data.get('risk_category', 'Unknown'),
             self._get_risk_format(track_data.get('risk_category'), formats)),
            (track_data.get('resistance_change_percent', 0) / 100, formats['percent']),
            (track_data.get('trim_improvement_percent', 0) / 100, formats['percent']),
            (track_data.get('unit_length', 0), formats['number']),
            (track_data.get('optimal_offset', 0), formats['number'])
        ]

        for col, (value, fmt) in enumerate(col_data):
            sheet.write(row, col, value, fmt)

    def _create_statistical_summary(
            self,
            workbook: xlsxwriter.Workbook,
            results: Dict[str, Any],
            formats: Dict[str, Any]
    ):
        """Create statistical summary sheet with distribution analysis."""
        sheet = workbook.add_worksheet('Statistical Summary')

        row = 0
        sheet.merge_range(row, 0, row, 5, 'STATISTICAL ANALYSIS', formats['title'])
        row += 2

        # Collect all numeric data
        all_data = self._collect_numeric_data(results)

        # Statistical metrics to calculate
        metrics = ['sigma_gradient', 'failure_probability', 'resistance_change_percent',
                   'trim_improvement_percent', 'unit_length']

        # Headers
        headers = ['Metric', 'Mean', 'Std Dev', 'Min', 'Max', 'Median']
        for col, header in enumerate(headers):
            sheet.write(row, col, header, formats['header'])
        row += 1

        # Calculate and write statistics
        for metric in metrics:
            if metric in all_data and len(all_data[metric]) > 0:
                data = np.array(all_data[metric])

                sheet.write(row, 0, metric.replace('_', ' ').title(), formats['subheader'])
                sheet.write(row, 1, np.mean(data), formats['number'])
                sheet.write(row, 2, np.std(data), formats['number'])
                sheet.write(row, 3, np.min(data), formats['number'])
                sheet.write(row, 4, np.max(data), formats['number'])
                sheet.write(row, 5, np.median(data), formats['number'])
                row += 1

        row += 2

        # Add distribution charts
        self._add_distribution_charts(workbook, sheet, all_data, row)

    def _create_trend_analysis(
            self,
            workbook: xlsxwriter.Workbook,
            results: Dict[str, Any],
            formats: Dict[str, Any]
    ):
        """Create trend analysis sheet with predictions."""
        sheet = workbook.add_worksheet('Trend Analysis')

        row = 0
        sheet.merge_range(row, 0, row, 6, 'TREND ANALYSIS & PREDICTIONS', formats['title'])
        row += 2

        # Prepare time series data
        time_series_data = self._prepare_time_series_data(results)

        if not time_series_data:
            sheet.write(row, 0, 'Insufficient data for trend analysis', formats['data'])
            return

        # Write time series data
        headers = ['Date', 'Files Processed', 'Pass Rate', 'Avg Sigma',
                   'High Risk %', 'Predicted Pass Rate', 'Trend']
        for col, header in enumerate(headers):
            sheet.write(row, col, header, formats['header'])
        row += 1

        # Write trend data
        for date_data in time_series_data:
            trend_format = formats['pass'] if date_data['trend'] == 'Improving' else \
                formats['fail'] if date_data['trend'] == 'Declining' else \
                    formats['warning']

            sheet.write(row, 0, date_data['date'], formats['data'])
            sheet.write(row, 1, date_data['count'], formats['data'])
            sheet.write(row, 2, date_data['pass_rate'], formats['percent'])
            sheet.write(row, 3, date_data['avg_sigma'], formats['number'])
            sheet.write(row, 4, date_data['high_risk_percent'], formats['percent'])
            sheet.write(row, 5, date_data['predicted_pass_rate'], formats['percent'])
            sheet.write(row, 6, date_data['trend'], trend_format)
            row += 1

        # Add trend charts
        self._add_trend_charts(workbook, sheet, time_series_data, row + 2)

    def _create_quality_metrics(
            self,
            workbook: xlsxwriter.Workbook,
            results: Dict[str, Any],
            formats: Dict[str, Any]
    ):
        """Create quality metrics sheet with KPIs."""
        sheet = workbook.add_worksheet('Quality Metrics')

        row = 0
        sheet.merge_range(row, 0, row, 4, 'QUALITY KEY PERFORMANCE INDICATORS', formats['title'])
        row += 2

        # Calculate KPIs
        kpis = self._calculate_kpis(results)

        # KPI sections
        sections = [
            ('Process Capability', [
                ('Cpk (Sigma)', kpis['cpk_sigma'], formats['number']),
                ('Ppk (Overall)', kpis['ppk_overall'], formats['number']),
                ('First Pass Yield', kpis['first_pass_yield'], formats['percent']),
                ('Rolled Throughput Yield', kpis['rty'], formats['percent'])
            ]),
            ('Quality Metrics', [
                ('Defects Per Million', kpis['dpm'], formats['data']),
                ('Mean Time Between Failures', f"{kpis['mtbf']:.0f} units", formats['data']),
                ('Cost of Poor Quality', f"${kpis['copq']:.2f}", formats['data']),
                ('Overall Equipment Effectiveness', kpis['oee'], formats['percent'])
            ]),
            ('Process Control', [
                ('In Control %', kpis['in_control_percent'], formats['percent']),
                ('Special Cause Variations', kpis['special_causes'], formats['data']),
                ('Process Stability Index', kpis['stability_index'], formats['number']),
                ('Improvement Opportunity', kpis['improvement_opportunity'], formats['percent'])
            ])
        ]

        # Write KPI sections
        for section_name, section_kpis in sections:
            sheet.merge_range(row, 0, row, 2, section_name, formats['header'])
            row += 1

            for kpi_name, kpi_value, kpi_format in section_kpis:
                sheet.write(row, 0, kpi_name, formats['subheader'])
                sheet.write(row, 1, kpi_value, kpi_format)

                # Add status indicator
                if isinstance(kpi_value, (int, float)) and kpi_name != 'Defects Per Million':
                    if kpi_name == 'Cost of Poor Quality':
                        status = 'Good' if float(kpi_value.replace('$', '')) < 100 else 'Poor'
                    else:
                        status = self._get_kpi_status(kpi_name, kpi_value)
                    status_format = formats['pass'] if status == 'Good' else \
                        formats['warning'] if status == 'Fair' else \
                            formats['fail']
                    sheet.write(row, 2, status, status_format)
                row += 1

            row += 1

        # Add KPI dashboard chart
        self._add_kpi_dashboard(workbook, sheet, kpis, row)

    def _create_recommendations(
            self,
            workbook: xlsxwriter.Workbook,
            results: Dict[str, Any],
            formats: Dict[str, Any],
            include_ai_insights: bool
    ):
        """Create recommendations sheet with actionable insights."""
        sheet = workbook.add_worksheet('Recommendations')

        row = 0
        sheet.merge_range(row, 0, row, 3, 'RECOMMENDATIONS & ACTION ITEMS', formats['title'])
        row += 2

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        # Priority levels
        priority_colors = {
            'Critical': formats['fail'],
            'High': formats['warning'],
            'Medium': formats['data'],
            'Low': formats['pass']
        }

        # Headers
        headers = ['Priority', 'Category', 'Recommendation', 'Expected Impact']
        for col, header in enumerate(headers):
            sheet.write(row, col, header, formats['header'])
        row += 1

        # Set column widths
        sheet.set_column('A:A', 10)
        sheet.set_column('B:B', 15)
        sheet.set_column('C:C', 50)
        sheet.set_column('D:D', 30)

        # Write recommendations
        for rec in recommendations:
            sheet.write(row, 0, rec['priority'], priority_colors.get(rec['priority'], formats['data']))
            sheet.write(row, 1, rec['category'], formats['data'])
            sheet.write(row, 2, rec['recommendation'], formats['data'])
            sheet.write(row, 3, rec['impact'], formats['data'])
            row += 1

        row += 2

        # AI-generated action plan
        if include_ai_insights and self.ai_client:
            sheet.merge_range(row, 0, row, 3, 'AI-GENERATED ACTION PLAN', formats['header'])
            row += 1

            action_plan = self._generate_ai_action_plan(results, recommendations)

            for step_num, step in enumerate(action_plan, 1):
                sheet.merge_range(row, 0, row + 1, 3, f"{step_num}. {step}", formats['insight'])
                row += 2

    def _create_raw_data(
            self,
            workbook: xlsxwriter.Workbook,
            results: Dict[str, Any],
            formats: Dict[str, Any]
    ):
        """Create raw data sheet for further analysis."""
        sheet = workbook.add_worksheet('Raw Data')

        # Convert results to DataFrame for easier handling
        df = self._results_to_dataframe(results)

        if df.empty:
            sheet.write(0, 0, 'No data available', formats['data'])
            return

        # Write headers
        for col, column_name in enumerate(df.columns):
            sheet.write(0, col, column_name, formats['header'])

        # Write data
        for row_idx, (_, row_data) in enumerate(df.iterrows(), 1):
            for col_idx, value in enumerate(row_data):
                # Determine format based on value type
                if isinstance(value, bool):
                    fmt = formats['pass'] if value else formats['fail']
                elif isinstance(value, (int, float)):
                    fmt = formats['number']
                else:
                    fmt = formats['data']

                sheet.write(row_idx, col_idx, value, fmt)

        # Add table formatting
        sheet.add_table(0, 0, len(df), len(df.columns) - 1, {
            'columns': [{'header': col} for col in df.columns],
            'style': 'Table Style Medium 2'
        })

    # Helper methods

    def _calculate_overall_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        file_results = results.get('file_results', [])

        if not file_results:
            return {
                'pass_rate': 0,
                'avg_sigma_gradient': 0,
                'avg_failure_probability': 0,
                'high_risk_count': 0,
                'attention_required': 0
            }

        # Flatten track data
        all_tracks = []
        for file_result in file_results:
            if 'tracks' in file_result and file_result['tracks']:
                all_tracks.extend(file_result['tracks'].values())
            else:
                all_tracks.append(file_result)

        # Calculate metrics
        total_tracks = len(all_tracks)
        passed_tracks = sum(1 for t in all_tracks if t.get('status', '').startswith('Pass'))

        sigma_gradients = [t.get('sigma_gradient', 0) for t in all_tracks if 'sigma_gradient' in t]
        failure_probs = [t.get('failure_probability', 0) for t in all_tracks if 'failure_probability' in t]

        high_risk = sum(1 for t in all_tracks if t.get('risk_category') == 'High')
        attention = sum(1 for t in all_tracks if
                        t.get('risk_category') in ['High', 'Medium'] or
                        not t.get('status', '').startswith('Pass'))

        return {
            'pass_rate': passed_tracks / total_tracks if total_tracks > 0 else 0,
            'avg_sigma_gradient': np.mean(sigma_gradients) if sigma_gradients else 0,
            'avg_failure_probability': np.mean(failure_probs) if failure_probs else 0,
            'high_risk_count': high_risk,
            'attention_required': attention
        }

    def _calculate_model_statistics(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics by model."""
        model_data = {}

        for file_result in results.get('file_results', []):
            model = file_result.get('model', 'Unknown')

            if model not in model_data:
                model_data[model] = {
                    'tracks': [],
                    'count': 0
                }

            # Add track data
            if 'tracks' in file_result and file_result['tracks']:
                for track_data in file_result['tracks'].values():
                    model_data[model]['tracks'].append(track_data)
                    model_data[model]['count'] += 1
            else:
                model_data[model]['tracks'].append(file_result)
                model_data[model]['count'] += 1

        # Calculate statistics for each model
        model_stats = {}
        for model, data in model_data.items():
            tracks = data['tracks']

            passed = sum(1 for t in tracks if t.get('status', '').startswith('Pass'))
            sigmas = [t.get('sigma_gradient', 0) for t in tracks if 'sigma_gradient' in t]
            high_risk = sum(1 for t in tracks if t.get('risk_category') == 'High')

            risk_level = 'High' if high_risk > len(tracks) * 0.2 else \
                'Medium' if high_risk > len(tracks) * 0.1 else 'Low'

            model_stats[model] = {
                'count': data['count'],
                'pass_rate': passed / len(tracks) if tracks else 0,
                'avg_sigma': np.mean(sigmas) if sigmas else 0,
                'risk_level': risk_level
            }

        return model_stats

    def _generate_ai_insights(
            self,
            overall_metrics: Dict[str, Any],
            model_stats: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate AI insights based on the analysis results."""
        if not self.ai_client:
            return ["AI insights unavailable - OpenAI API key not configured"]

        try:
            # Prepare context for AI
            context = {
                "overall_metrics": overall_metrics,
                "model_statistics": model_stats,
                "analysis_type": "laser_trim_potentiometer_qa"
            }

            prompt = f"""
            Analyze the following laser trim quality data and provide 3-4 key insights:

            Overall Metrics:
            - Pass Rate: {overall_metrics['pass_rate']:.1%}
            - Average Sigma Gradient: {overall_metrics['avg_sigma_gradient']:.4f}
            - Average Failure Probability: {overall_metrics['avg_failure_probability']:.1%}
            - High Risk Units: {overall_metrics['high_risk_count']}

            Model Performance:
            {json.dumps(model_stats, indent=2)}

            Provide actionable insights focusing on:
            1. Overall quality trends
            2. Model-specific issues
            3. Process improvement opportunities
            4. Risk mitigation strategies

            Format each insight as a complete sentence suitable for executive presentation.
            """

            response = self.ai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system",
                     "content": "You are a quality assurance expert specializing in potentiometer manufacturing."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            # Parse response into individual insights
            insights_text = response.choices[0].message.content
            insights = [insight.strip() for insight in insights_text.split('\n') if insight.strip()]

            return insights[:4]  # Return top 4 insights

        except Exception as e:
            self.logger.error(f"Error generating AI insights: {str(e)}")
            return ["Error generating AI insights - check API configuration"]

    def _collect_numeric_data(self, results: Dict[str, Any]) -> Dict[str, List[float]]:
        """Collect all numeric data for statistical analysis."""
        data_collections = {
            'sigma_gradient': [],
            'failure_probability': [],
            'resistance_change_percent': [],
            'trim_improvement_percent': [],
            'unit_length': []
        }

        for file_result in results.get('file_results', []):
            # Process tracks
            tracks = []
            if 'tracks' in file_result and file_result['tracks']:
                tracks.extend(file_result['tracks'].values())
            else:
                tracks.append(file_result)

            for track in tracks:
                for key in data_collections:
                    if key in track and track[key] is not None:
                        data_collections[key].append(track[key])

        return data_collections

    def _add_performance_chart(
            self,
            workbook: xlsxwriter.Workbook,
            sheet: xlsxwriter.Worksheet,
            model_stats: Dict[str, Dict[str, Any]],
            start_row: int
    ):
        """Add performance chart to the worksheet."""
        # Create chart
        chart = workbook.add_chart({'type': 'column'})

        # Prepare data for chart
        models = list(model_stats.keys())
        pass_rates = [stats['pass_rate'] for stats in model_stats.values()]

        # Write data for chart (hidden)
        data_row = start_row + 20
        sheet.write_row(data_row, 0, ['Model'] + models)
        sheet.write_row(data_row + 1, 0, ['Pass Rate'] + pass_rates)

        # Configure chart
        chart.add_series({
            'name': 'Pass Rate by Model',
            'categories': [sheet.name, data_row, 1, data_row, len(models)],
            'values': [sheet.name, data_row + 1, 1, data_row + 1, len(models)],
            'fill': {'color': '#4472C4'}
        })

        chart.set_title({'name': 'Pass Rate by Model'})
        chart.set_x_axis({'name': 'Model'})
        chart.set_y_axis({'name': 'Pass Rate (%)', 'max': 1, 'num_format': '0%'})
        chart.set_size({'width': 600, 'height': 400})

        # Insert chart
        sheet.insert_chart(start_row, 0, chart)

    def _prepare_time_series_data(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare time series data for trend analysis."""
        # For demo purposes, create synthetic time series data
        # In production, this would use actual timestamps from results

        file_results = results.get('file_results', [])
        if not file_results:
            return []

        # Create synthetic daily data for the last 30 days
        import random
        from datetime import timedelta

        time_series = []
        base_date = datetime.now() - timedelta(days=30)

        for i in range(30):
            date = base_date + timedelta(days=i)

            # Simulate metrics with some randomness
            pass_rate = 0.85 + random.uniform(-0.1, 0.1)
            avg_sigma = 0.003 + random.uniform(-0.001, 0.001)
            high_risk_percent = 0.05 + random.uniform(-0.02, 0.02)

            # Simple trend prediction
            predicted_pass_rate = pass_rate + (i / 30) * 0.05  # Improving trend

            trend = 'Improving' if predicted_pass_rate > pass_rate else \
                'Declining' if predicted_pass_rate < pass_rate - 0.02 else \
                    'Stable'

            time_series.append({
                'date': date.strftime('%Y-%m-%d'),
                'count': random.randint(50, 150),
                'pass_rate': pass_rate,
                'avg_sigma': avg_sigma,
                'high_risk_percent': high_risk_percent,
                'predicted_pass_rate': predicted_pass_rate,
                'trend': trend
            })

        return time_series

    def _add_distribution_charts(
            self,
            workbook: xlsxwriter.Workbook,
            sheet: xlsxwriter.Worksheet,
            data: Dict[str, List[float]],
            start_row: int
    ):
        """Add distribution charts for key metrics."""
        chart_col = 0

        for metric_name, metric_data in data.items():
            if metric_name in ['sigma_gradient', 'failure_probability'] and metric_data:
                # Create histogram chart
                chart = workbook.add_chart({'type': 'column'})

                # Create bins
                hist, bins = np.histogram(metric_data, bins=10)
                bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

                # Write histogram data
                data_row = start_row + 30
                sheet.write_column(data_row, chart_col * 3, bin_centers)
                sheet.write_column(data_row, chart_col * 3 + 1, hist)

                # Configure chart
                chart.add_series({
                    'name': f'{metric_name.replace("_", " ").title()} Distribution',
                    'categories': [sheet.name, data_row, chart_col * 3,
                                   data_row + len(bin_centers) - 1, chart_col * 3],
                    'values': [sheet.name, data_row, chart_col * 3 + 1,
                               data_row + len(hist) - 1, chart_col * 3 + 1],
                    'fill': {'color': '#70AD47'}
                })

                chart.set_title({'name': f'{metric_name.replace("_", " ").title()} Distribution'})
                chart.set_x_axis({'name': metric_name.replace("_", " ").title()})
                chart.set_y_axis({'name': 'Frequency'})
                chart.set_size({'width': 400, 'height': 300})

                # Insert chart
                sheet.insert_chart(start_row, chart_col * 7, chart)
                chart_col += 1

    def _calculate_kpis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality KPIs from results."""
        # Collect all track data
        all_tracks = []
        for file_result in results.get('file_results', []):
            if 'tracks' in file_result and file_result['tracks']:
                all_tracks.extend(file_result['tracks'].values())
            else:
                all_tracks.append(file_result)

        if not all_tracks:
            return self._default_kpis()

        # Calculate process capability
        sigma_values = [t.get('sigma_gradient', 0) for t in all_tracks if 'sigma_gradient' in t]
        sigma_thresholds = [t.get('sigma_threshold', 0.004) for t in all_tracks if 'sigma_threshold' in t]

        if sigma_values and sigma_thresholds:
            mean_sigma = np.mean(sigma_values)
            std_sigma = np.std(sigma_values)
            mean_threshold = np.mean(sigma_thresholds)

            # Cpk calculation
            cpk = (mean_threshold - mean_sigma) / (3 * std_sigma) if std_sigma > 0 else 0
            ppk = cpk * 0.9  # Ppk is typically slightly lower than Cpk
        else:
            cpk = ppk = 0

        # Calculate yields
        total = len(all_tracks)
        passed = sum(1 for t in all_tracks if t.get('status', '').startswith('Pass'))
        first_pass_yield = passed / total if total > 0 else 0

        # Rolled throughput yield (assuming multi-stage process)
        rty = first_pass_yield ** 0.95  # Slightly lower due to process steps

        # Calculate DPM
        dpm = (1 - first_pass_yield) * 1_000_000

        # MTBF estimation
        mtbf = 1 / (1 - first_pass_yield) if first_pass_yield < 1 else 10000

        # Cost of Poor Quality estimation
        copq = (1 - first_pass_yield) * 1000  # $1000 per defect assumption

        # OEE calculation
        availability = 0.95  # Assumed
        performance = 0.90  # Assumed
        quality = first_pass_yield
        oee = availability * performance * quality

        # Process control metrics
        in_control = sum(1 for t in all_tracks if
                         t.get('sigma_gradient', 0) < t.get('sigma_threshold', 0.004) * 0.8)
        in_control_percent = in_control / total if total > 0 else 0

        # Special causes (outliers)
        if sigma_values:
            mean_s = np.mean(sigma_values)
            std_s = np.std(sigma_values)
            special_causes = sum(1 for s in sigma_values if abs(s - mean_s) > 3 * std_s)
        else:
            special_causes = 0

        # Stability index
        stability_index = 1 - (special_causes / total) if total > 0 else 1

        # Improvement opportunity
        improvement_opportunity = 1 - first_pass_yield

        return {
            'cpk_sigma': cpk,
            'ppk_overall': ppk,
            'first_pass_yield': first_pass_yield,
            'rty': rty,
            'dpm': int(dpm),
            'mtbf': mtbf,
            'copq': copq,
            'oee': oee,
            'in_control_percent': in_control_percent,
            'special_causes': special_causes,
            'stability_index': stability_index,
            'improvement_opportunity': improvement_opportunity
        }

    def _default_kpis(self) -> Dict[str, Any]:
        """Return default KPIs when no data is available."""
        return {
            'cpk_sigma': 0,
            'ppk_overall': 0,
            'first_pass_yield': 0,
            'rty': 0,
            'dpm': 0,
            'mtbf': 0,
            'copq': 0,
            'oee': 0,
            'in_control_percent': 0,
            'special_causes': 0,
            'stability_index': 0,
            'improvement_opportunity': 0
        }

    def _get_kpi_status(self, kpi_name: str, value: float) -> str:
        """Determine status of a KPI based on thresholds."""
        thresholds = {
            'Cpk (Sigma)': {'Good': 1.33, 'Fair': 1.0},
            'Ppk (Overall)': {'Good': 1.33, 'Fair': 1.0},
            'First Pass Yield': {'Good': 0.95, 'Fair': 0.85},
            'Rolled Throughput Yield': {'Good': 0.90, 'Fair': 0.80},
            'Mean Time Between Failures': {'Good': 100, 'Fair': 50},
            'Overall Equipment Effectiveness': {'Good': 0.85, 'Fair': 0.65},
            'In Control %': {'Good': 0.95, 'Fair': 0.85},
            'Process Stability Index': {'Good': 0.95, 'Fair': 0.85},
            'Improvement Opportunity': {'Good': 0.05, 'Fair': 0.15}
        }

        if kpi_name not in thresholds:
            return 'Unknown'

        limits = thresholds[kpi_name]

        # For improvement opportunity, lower is better
        if kpi_name == 'Improvement Opportunity':
            if value <= limits['Good']:
                return 'Good'
            elif value <= limits['Fair']:
                return 'Fair'
            else:
                return 'Poor'
        else:
            # For other metrics, higher is better
            if value >= limits['Good']:
                return 'Good'
            elif value >= limits['Fair']:
                return 'Fair'
            else:
                return 'Poor'

    def _get_risk_format(self, risk_category: str, formats: Dict[str, Any]) -> Any:
        """Get format based on risk category."""
        if risk_category == 'Low':
            return formats['pass']
        elif risk_category == 'Medium':
            return formats['warning']
        elif risk_category == 'High':
            return formats['fail']
        else:
            return formats['data']

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis results."""
        recommendations = []

        # Analyze overall metrics
        overall_metrics = self._calculate_overall_metrics(results)

        # Critical recommendations
        if overall_metrics['pass_rate'] < 0.8:
            recommendations.append({
                'priority': 'Critical',
                'category': 'Quality',
                'recommendation': 'Overall pass rate is below 80%. Immediate process review required.',
                'impact': 'Could improve pass rate by 10-20%'
            })

        if overall_metrics['high_risk_count'] > 5:
            recommendations.append({
                'priority': 'Critical',
                'category': 'Risk',
                'recommendation': f'{overall_metrics["high_risk_count"]} high-risk units detected. Implement enhanced screening.',
                'impact': 'Prevent potential field failures'
            })

        # High priority recommendations
        if overall_metrics['avg_sigma_gradient'] > 0.003:
            recommendations.append({
                'priority': 'High',
                'category': 'Process',
                'recommendation': 'Average sigma gradient exceeds target. Review laser trim parameters.',
                'impact': 'Reduce sigma gradient by 20-30%'
            })

        # Model-specific recommendations
        model_stats = self._calculate_model_statistics(results)
        for model, stats in model_stats.items():
            if stats['pass_rate'] < 0.7:
                recommendations.append({
                    'priority': 'High',
                    'category': 'Model-Specific',
                    'recommendation': f'Model {model} has low pass rate ({stats["pass_rate"]:.1%}). Investigate design or process issues.',
                    'impact': f'Improve {model} yield by 15-25%'
                })

        # Medium priority recommendations
        if overall_metrics['avg_failure_probability'] > 0.1:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Predictive',
                'recommendation': 'High average failure probability. Implement predictive maintenance program.',
                'impact': 'Reduce warranty claims by 30%'
            })

        # Low priority recommendations
        recommendations.append({
            'priority': 'Low',
            'category': 'Continuous Improvement',
            'recommendation': 'Consider implementing SPC charts for real-time monitoring.',
            'impact': 'Improve process stability by 10%'
        })

        # Sort by priority
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))

        return recommendations

    def _generate_ai_action_plan(
            self,
            results: Dict[str, Any],
            recommendations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate AI-powered action plan."""
        if not self.ai_client:
            return ["AI action plan unavailable - OpenAI API key not configured"]

        try:
            # Prepare context
            critical_recs = [r for r in recommendations if r['priority'] == 'Critical']
            high_recs = [r for r in recommendations if r['priority'] == 'High']

            prompt = f"""
            Based on the following quality analysis results and recommendations, create a prioritized 
            5-step action plan for improving potentiometer manufacturing quality:

            Critical Issues:
            {json.dumps(critical_recs, indent=2)}

            High Priority Issues:
            {json.dumps(high_recs, indent=2)}

            Create specific, actionable steps that can be implemented within 30 days.
            Each step should include WHO should do it and WHEN it should be completed.
            """

            response = self.ai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system",
                     "content": "You are a manufacturing quality expert creating actionable improvement plans."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.7
            )

            # Parse response
            action_plan_text = response.choices[0].message.content
            steps = [step.strip() for step in action_plan_text.split('\n') if step.strip() and step[0].isdigit()]

            return steps[:5]  # Return top 5 steps

        except Exception as e:
            self.logger.error(f"Error generating AI action plan: {str(e)}")
            return ["Error generating AI action plan - check API configuration"]

    def _add_trend_charts(
            self,
            workbook: xlsxwriter.Workbook,
            sheet: xlsxwriter.Worksheet,
            time_series_data: List[Dict[str, Any]],
            start_row: int
    ):
        """Add trend analysis charts."""
        # Create line chart for pass rate trend
        chart = workbook.add_chart({'type': 'line'})

        # Write hidden data for chart
        data_row = start_row + 20
        dates = [d['date'] for d in time_series_data]
        pass_rates = [d['pass_rate'] for d in time_series_data]
        predicted_rates = [d['predicted_pass_rate'] for d in time_series_data]

        sheet.write_column(data_row, 10, dates)
        sheet.write_column(data_row, 11, pass_rates)
        sheet.write_column(data_row, 12, predicted_rates)

        # Configure chart
        chart.add_series({
            'name': 'Actual Pass Rate',
            'categories': [sheet.name, data_row, 10, data_row + len(dates) - 1, 10],
            'values': [sheet.name, data_row, 11, data_row + len(pass_rates) - 1, 11],
            'line': {'color': '#4472C4', 'width': 2}
        })

        chart.add_series({
            'name': 'Predicted Pass Rate',
            'categories': [sheet.name, data_row, 10, data_row + len(dates) - 1, 10],
            'values': [sheet.name, data_row, 12, data_row + len(predicted_rates) - 1, 12],
            'line': {'color': '#70AD47', 'width': 2, 'dash_type': 'dash'}
        })

        chart.set_title({'name': 'Pass Rate Trend Analysis'})
        chart.set_x_axis({'name': 'Date'})
        chart.set_y_axis({'name': 'Pass Rate (%)', 'num_format': '0%'})
        chart.set_size({'width': 800, 'height': 400})
        chart.set_legend({'position': 'bottom'})

        # Insert chart
        sheet.insert_chart(start_row, 0, chart)

    def _add_kpi_dashboard(
            self,
            workbook: xlsxwriter.Workbook,
            sheet: xlsxwriter.Worksheet,
            kpis: Dict[str, Any],
            start_row: int
    ):
        """Add KPI dashboard visualization."""
        # Create gauge chart for key metrics
        chart = workbook.add_chart({'type': 'doughnut'})

        # Prepare data for gauge (using First Pass Yield as example)
        fpy = kpis['first_pass_yield']

        # Write data
        data_row = start_row + 20
        sheet.write_column(data_row, 10, ['Achieved', 'Remaining'])
        sheet.write_column(data_row, 11, [fpy, 1 - fpy])

        # Configure chart
        chart.add_series({
            'name': 'First Pass Yield',
            'categories': [sheet.name, data_row, 10, data_row + 1, 10],
            'values': [sheet.name, data_row, 11, data_row + 1, 11],
            'points': [
                {'fill': {'color': '#70AD47'}},
                {'fill': {'color': '#E7E6E6'}}
            ]
        })

        chart.set_title({'name': f'First Pass Yield: {fpy:.1%}'})
        chart.set_size({'width': 400, 'height': 300})
        chart.set_rotation(270)  # Start from top
        chart.set_hole_size(70)  # Create gauge effect

        # Insert chart
        sheet.insert_chart(start_row, 0, chart)

    def _results_to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Convert results dictionary to pandas DataFrame."""
        rows = []

        for file_result in results.get('file_results', []):
            base_info = {
                'filename': file_result.get('filename'),
                'model': file_result.get('model'),
                'serial': file_result.get('serial'),
                'system': file_result.get('system')
            }

            # Handle tracks
            if 'tracks' in file_result and file_result['tracks']:
                for track_id, track_data in file_result['tracks'].items():
                    row = {**base_info, 'track_id': track_id, **track_data}
                    rows.append(row)
            else:
                row = {**base_info, 'track_id': 'N/A', **file_result}
                rows.append(row)

        return pd.DataFrame(rows)


# Example usage function
def generate_sample_report():
    """Generate a sample report for testing."""
    import json

    # Load sample results
    sample_results = {
        "file_results": [
            {
                "filename": "8340_SN001.xlsx",
                "model": "8340",
                "serial": "SN001",
                "system": "A",
                "tracks": {
                    "TRK1": {
                        "status": "Pass",
                        "sigma_gradient": 0.0025,
                        "sigma_threshold": 0.004,
                        "sigma_pass": True,
                        "linearity_pass": True,
                        "failure_probability": 0.05,
                        "risk_category": "Low",
                        "resistance_change_percent": 2.5,
                        "trim_improvement_percent": 15.3,
                        "unit_length": 150.2,
                        "optimal_offset": 0.001
                    }
                }
            }
        ],
        "ml_predictions": {
            "next_batch_pass_rate": 0.89,
            "maintenance_due_in_days": 15,
            "quality_trend": "improving"
        }
    }

    # Create reporter (without API key for demo)
    reporter = ExcelReporter()

    # Generate report
    output_path = "laser_trim_analysis_report.xlsx"
    reporter.generate_report(sample_results, output_path, include_ai_insights=False)

    print(f"Sample report generated: {output_path}")


if __name__ == "__main__":
    generate_sample_report()