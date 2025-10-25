"""
AI-Powered QA Analysis Module for Potentiometer Manufacturing

This module provides intelligent analysis capabilities including:
- Automatic insight generation
- Interactive QA assistant
- Professional PDF report generation
- Predictive maintenance alerts
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import warnings

warnings.filterwarnings('ignore')

# For the interactive assistant
import re
from collections import defaultdict


class AIQualityAnalyzer:
    """
    AI-powered quality analysis system for potentiometer manufacturing.
    Provides intelligent insights, predictions, and recommendations.
    """

    def __init__(self, db_path: str):
        """
        Initialize the AI analyzer with database connection.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self.insights = []
        self.recommendations = []
        self.anomalies = []

        # Best practice thresholds based on industry standards
        self.best_practices = {
            'sigma_gradient': {'excellent': 0.001, 'good': 0.005, 'acceptable': 0.01, 'poor': 0.02},
            'pass_rate': {'excellent': 0.98, 'good': 0.95, 'acceptable': 0.90, 'poor': 0.85},
            'resistance_change': {'excellent': 2.0, 'good': 5.0, 'acceptable': 10.0, 'poor': 15.0},
            'failure_probability': {'excellent': 0.05, 'good': 0.15, 'acceptable': 0.30, 'poor': 0.50}
        }

    def generate_insights(self, analysis_results: List[Dict[str, Any]],
                          historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate intelligent insights from analysis results.

        Args:
            analysis_results: Recent analysis results
            historical_data: Historical data for comparison

        Returns:
            Dictionary containing insights, patterns, and recommendations
        """
        self.insights = []
        self.recommendations = []
        self.anomalies = []

        # Convert results to DataFrame for easier analysis
        df = pd.DataFrame(analysis_results)

        # 1. Overall Performance Insights
        self._analyze_overall_performance(df)

        # 2. Model-Specific Insights
        self._analyze_model_performance(df)

        # 3. Trend Analysis (if historical data available)
        if historical_data is not None:
            self._analyze_trends(df, historical_data)

        # 4. Anomaly Detection
        self._detect_anomalies(df)

        # 5. Process Improvement Opportunities
        self._identify_improvements(df)

        # 6. Predict Future Issues
        predictions = self._predict_future_issues(df, historical_data)

        return {
            'insights': self.insights,
            'recommendations': self.recommendations,
            'anomalies': self.anomalies,
            'predictions': predictions,
            'summary': self._generate_executive_summary(df)
        }

    def _analyze_overall_performance(self, df: pd.DataFrame):
        """Analyze overall manufacturing performance."""
        if len(df) == 0:
            return

        # Calculate key metrics
        overall_pass_rate = df['Sigma Pass'].mean() if 'Sigma Pass' in df.columns else 0
        avg_sigma = df['Sigma Gradient'].mean() if 'Sigma Gradient' in df.columns else 0

        # Generate insights based on performance
        if overall_pass_rate >= self.best_practices['pass_rate']['excellent']:
            self.insights.append({
                'type': 'positive',
                'category': 'overall_performance',
                'message': f"Excellent overall performance with {overall_pass_rate * 100:.1f}% pass rate",
                'importance': 'high'
            })
        elif overall_pass_rate < self.best_practices['pass_rate']['acceptable']:
            self.insights.append({
                'type': 'warning',
                'category': 'overall_performance',
                'message': f"Pass rate of {overall_pass_rate * 100:.1f}% is below acceptable threshold",
                'importance': 'high'
            })
            self.recommendations.append(
                "Investigate root causes of low pass rate. Consider equipment calibration and material quality checks."
            )

        # Sigma gradient analysis
        if avg_sigma > self.best_practices['sigma_gradient']['acceptable']:
            self.insights.append({
                'type': 'warning',
                'category': 'sigma_gradient',
                'message': f"Average sigma gradient ({avg_sigma:.4f}) exceeds acceptable limits",
                'importance': 'high'
            })
            self.recommendations.append(
                "High sigma gradients indicate process variability. Check laser calibration and environmental conditions."
            )

    def _analyze_model_performance(self, df: pd.DataFrame):
        """Analyze performance by model."""
        if 'Model' not in df.columns:
            return

        model_performance = df.groupby('Model').agg({
            'Sigma Pass': 'mean',
            'Sigma Gradient': 'mean',
            'Failure Probability': 'mean'
        }).round(4)

        for model, metrics in model_performance.iterrows():
            # Check each model against best practices
            if metrics['Sigma Pass'] < self.best_practices['pass_rate']['good']:
                self.insights.append({
                    'type': 'warning',
                    'category': 'model_specific',
                    'message': f"Model {model} has below-average pass rate ({metrics['Sigma Pass'] * 100:.1f}%)",
                    'importance': 'medium',
                    'model': model
                })

            if metrics['Failure Probability'] > self.best_practices['failure_probability']['acceptable']:
                self.recommendations.append(
                    f"Model {model}: High failure risk detected. Review design specifications and tolerances."
                )

    def _detect_anomalies(self, df: pd.DataFrame):
        """Detect anomalous units using statistical methods."""
        numeric_columns = ['Sigma Gradient', 'Resistance Change (%)', 'Failure Probability']

        for col in numeric_columns:
            if col in df.columns:
                # Use IQR method for anomaly detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                anomalous_units = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

                if len(anomalous_units) > 0:
                    for _, unit in anomalous_units.iterrows():
                        self.anomalies.append({
                            'unit': f"{unit.get('Model', 'Unknown')}-{unit.get('Serial', 'Unknown')}",
                            'metric': col,
                            'value': unit[col],
                            'expected_range': f"{lower_bound:.4f} to {upper_bound:.4f}",
                            'severity': 'high' if abs(unit[col] - df[col].mean()) > 3 * df[col].std() else 'medium'
                        })

    def _analyze_trends(self, current_df: pd.DataFrame, historical_df: pd.DataFrame):
        """Analyze trends by comparing current to historical data."""
        # Convert timestamp to datetime if needed
        if 'timestamp' in historical_df.columns:
            historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])

            # Get data from last 30 days
            recent_historical = historical_df[
                historical_df['timestamp'] > datetime.now() - timedelta(days=30)
                ]

            if len(recent_historical) > 0:
                # Compare current to recent historical averages
                current_avg_sigma = current_df['Sigma Gradient'].mean()
                historical_avg_sigma = recent_historical['sigma_gradient'].mean()

                change_percent = ((current_avg_sigma - historical_avg_sigma) / historical_avg_sigma * 100)

                if abs(change_percent) > 10:
                    trend_type = 'increasing' if change_percent > 0 else 'improving'
                    self.insights.append({
                        'type': 'trend',
                        'category': 'sigma_gradient_trend',
                        'message': f"Sigma gradient {trend_type} by {abs(change_percent):.1f}% compared to 30-day average",
                        'importance': 'high'
                    })

                    if change_percent > 10:
                        self.recommendations.append(
                            "Sigma gradient trending upward. Schedule preventive maintenance for laser equipment."
                        )

    def _identify_improvements(self, df: pd.DataFrame):
        """Identify process improvement opportunities."""
        # Check for consistent issues across units
        if 'Risk Category' in df.columns:
            high_risk_pct = (df['Risk Category'] == 'High').mean()

            if high_risk_pct > 0.1:  # More than 10% high risk
                self.recommendations.append(
                    f"{high_risk_pct * 100:.1f}% of units are high risk. "
                    "Consider tightening incoming material specifications."
                )

        # Check resistance change patterns
        if 'Resistance Change (%)' in df.columns:
            avg_resistance_change = df['Resistance Change (%)'].abs().mean()

            if avg_resistance_change > self.best_practices['resistance_change']['good']:
                self.insights.append({
                    'type': 'improvement',
                    'category': 'resistance_control',
                    'message': f"Average resistance change ({avg_resistance_change:.1f}%) can be improved",
                    'importance': 'medium'
                })
                self.recommendations.append(
                    "Optimize trim parameters to minimize resistance change while maintaining linearity."
                )

    def _predict_future_issues(self, current_df: pd.DataFrame,
                               historical_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Predict potential future issues based on patterns."""
        predictions = {
            'maintenance_needed': [],
            'quality_risks': [],
            'trend_forecasts': []
        }

        if historical_df is not None and len(historical_df) > 100:
            # Simple trend analysis for maintenance prediction
            historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])

            # Group by week and calculate degradation
            weekly_sigma = historical_df.set_index('timestamp').resample('W')['sigma_gradient'].mean()

            if len(weekly_sigma) > 4:
                # Calculate trend using linear regression
                x = np.arange(len(weekly_sigma))
                y = weekly_sigma.values

                if not np.any(np.isnan(y)):
                    slope, intercept = np.polyfit(x, y, 1)

                    # Predict when sigma will exceed threshold
                    threshold = self.best_practices['sigma_gradient']['poor']
                    if slope > 0:  # Degrading
                        weeks_to_threshold = (threshold - y[-1]) / slope

                        if weeks_to_threshold < 4:
                            predictions['maintenance_needed'].append({
                                'equipment': 'Laser trimming system',
                                'urgency': 'high',
                                'predicted_date': datetime.now() + timedelta(weeks=weeks_to_threshold),
                                'reason': 'Sigma gradient trending toward unacceptable levels'
                            })

        # Predict quality risks based on current patterns
        if 'Model' in current_df.columns:
            model_risks = current_df.groupby('Model')['Failure Probability'].mean()

            for model, risk in model_risks.items():
                if risk > self.best_practices['failure_probability']['acceptable']:
                    predictions['quality_risks'].append({
                        'model': model,
                        'risk_level': 'high' if risk > 0.5 else 'medium',
                        'predicted_failures': int(risk * 100),
                        'recommendation': f"Review specifications for Model {model}"
                    })

        return predictions

    def _generate_executive_summary(self, df: pd.DataFrame) -> str:
        """Generate executive summary of findings."""
        summary_parts = []

        # Overall performance
        pass_rate = df['Sigma Pass'].mean() * 100 if 'Sigma Pass' in df.columns else 0
        summary_parts.append(f"Analyzed {len(df)} units with {pass_rate:.1f}% overall pass rate.")

        # Key findings
        if self.anomalies:
            summary_parts.append(f"Detected {len(self.anomalies)} anomalous units requiring attention.")

        # Top recommendation
        if self.recommendations:
            summary_parts.append(f"Primary recommendation: {self.recommendations[0]}")

        return " ".join(summary_parts)


class InteractiveQAAssistant:
    """
    Interactive assistant for answering questions about QA data.
    """

    def __init__(self, db_path: str):
        """Initialize the QA assistant."""
        self.db_path = db_path
        self.context = {}

        # Define question patterns and handlers
        self.patterns = {
            'pass_rate': r'(?:what is|show me|tell me about).*pass rate.*(?:for|of)?\s*(\w+)?',
            'sigma': r'(?:what is|show me|average).*sigma.*(?:for|of)?\s*(\w+)?',
            'compare': r'compare\s+(\w+)\s+(?:to|and|with)\s+(\w+)',
            'worst': r'(?:worst|poorest|lowest).*(?:performing|performance).*(?:model|unit)',
            'best': r'(?:best|highest|top).*(?:performing|performance).*(?:model|unit)',
            'explain': r'(?:explain|what is|define)\s+(.+)',
            'trend': r'(?:trend|trending|change).*(?:in|for)?\s*(.+)?',
            'risk': r'(?:risk|failure|problem).*(?:with|for)?\s*(\w+)?'
        }

        # Metric explanations for users
        self.explanations = {
            'sigma gradient': """
Sigma gradient measures the variability of error across the potentiometer's travel.
- Lower values (< 0.005) indicate consistent, stable performance
- Higher values (> 0.01) suggest irregular trimming or material issues
- It's calculated as the standard deviation of error gradients
Think of it as how "smooth" the potentiometer's response is.
            """,
            'linearity': """
Linearity measures how closely the potentiometer follows an ideal straight line.
- Perfect linearity means the output changes proportionally with position
- Deviations indicate the potentiometer doesn't track perfectly
- Measured as maximum deviation from the ideal response
Important for precision applications.
            """,
            'failure probability': """
Failure probability predicts the likelihood of early field failure.
- Combines multiple quality metrics using statistical models
- Values > 0.7 indicate high risk units
- Based on historical failure patterns
Helps prioritize units for additional testing or rework.
            """,
            'resistance change': """
Resistance change shows how much the trim process altered the total resistance.
- Typically should be < 10% for good process control
- Large changes may indicate over-trimming
- Affects the final tolerance of the potentiometer
Critical for meeting customer specifications.
            """
        }

    def answer_question(self, question: str, context_data: Optional[pd.DataFrame] = None) -> str:
        """
        Answer a question about the QA data.

        Args:
            question: Natural language question
            context_data: Optional DataFrame with current data context

        Returns:
            Natural language answer
        """
        question_lower = question.lower()

        # Load data if not provided
        if context_data is None:
            context_data = self._load_recent_data()

        # Try to match question patterns
        for pattern_name, pattern in self.patterns.items():
            match = re.search(pattern, question_lower)
            if match:
                return self._handle_pattern(pattern_name, match, context_data, question)

        # Default response
        return self._general_response(question, context_data)

    def _handle_pattern(self, pattern_name: str, match: re.Match,
                        data: pd.DataFrame, original_question: str) -> str:
        """Handle specific question patterns."""

        if pattern_name == 'pass_rate':
            model = match.group(1)
            return self._get_pass_rate_info(data, model)

        elif pattern_name == 'sigma':
            model = match.group(1)
            return self._get_sigma_info(data, model)

        elif pattern_name == 'compare':
            model1, model2 = match.group(1), match.group(2)
            return self._compare_models(data, model1, model2)

        elif pattern_name == 'worst':
            return self._get_worst_performing(data)

        elif pattern_name == 'best':
            return self._get_best_performing(data)

        elif pattern_name == 'explain':
            term = match.group(1).strip()
            return self._explain_term(term)

        elif pattern_name == 'trend':
            metric = match.group(1) if match.group(1) else 'sigma gradient'
            return self._analyze_trend(data, metric)

        elif pattern_name == 'risk':
            model = match.group(1)
            return self._get_risk_info(data, model)

        return "I couldn't understand that question. Try asking about pass rates, sigma gradients, or model comparisons."

    def _get_pass_rate_info(self, data: pd.DataFrame, model: Optional[str] = None) -> str:
        """Get pass rate information."""
        if model and 'Model' in data.columns:
            model_data = data[data['Model'].str.contains(model, case=False, na=False)]
            if len(model_data) > 0:
                pass_rate = model_data['Sigma Pass'].mean() * 100
                return f"Model {model} has a pass rate of {pass_rate:.1f}%. "                       f"This is based on {len(model_data)} units tested."
            else:
                return f"No data found for model {model}."
        else:
            overall_pass = data['Sigma Pass'].mean() * 100 if 'Sigma Pass' in data.columns else 0
            return f"The overall pass rate is {overall_pass:.1f}% across {len(data)} units tested."

    def _get_sigma_info(self, data: pd.DataFrame, model: Optional[str] = None) -> str:
        """Get sigma gradient information."""
        if 'Sigma Gradient' not in data.columns:
            return "Sigma gradient data is not available."

        if model and 'Model' in data.columns:
            model_data = data[data['Model'].str.contains(model, case=False, na=False)]
            if len(model_data) > 0:
                avg_sigma = model_data['Sigma Gradient'].mean()
                std_sigma = model_data['Sigma Gradient'].std()
                return f"Model {model} has an average sigma gradient of {avg_sigma:.4f} "                       f"(std dev: {std_sigma:.4f}). "                       f"{'This is excellent.' if avg_sigma < 0.005 else 'This could be improved.'}"
            else:
                return f"No sigma gradient data found for model {model}."
        else:
            avg_sigma = data['Sigma Gradient'].mean()
            return f"The average sigma gradient across all units is {avg_sigma:.4f}. "                   f"{'This indicates good process control.' if avg_sigma < 0.01 else 'Consider process optimization.'}"

    def _compare_models(self, data: pd.DataFrame, model1: str, model2: str) -> str:
        """Compare two models."""
        if 'Model' not in data.columns:
            return "Model comparison requires model data."

        m1_data = data[data['Model'].str.contains(model1, case=False, na=False)]
        m2_data = data[data['Model'].str.contains(model2, case=False, na=False)]

        if len(m1_data) == 0 or len(m2_data) == 0:
            return f"Insufficient data to compare {model1} and {model2}."

        comparison = []

        # Pass rates
        m1_pass = m1_data['Sigma Pass'].mean() * 100
        m2_pass = m2_data['Sigma Pass'].mean() * 100
        comparison.append(f"Pass rates: {model1} ({m1_pass:.1f}%) vs {model2} ({m2_pass:.1f}%)")

        # Sigma gradients
        m1_sigma = m1_data['Sigma Gradient'].mean()
        m2_sigma = m2_data['Sigma Gradient'].mean()
        comparison.append(f"Avg sigma: {model1} ({m1_sigma:.4f}) vs {model2} ({m2_sigma:.4f})")

        # Winner
        m1_score = m1_pass - (m1_sigma * 1000)  # Simple scoring
        m2_score = m2_pass - (m2_sigma * 1000)
        winner = model1 if m1_score > m2_score else model2

        comparison.append(f"Overall, {winner} shows better performance.")

        return " ".join(comparison)

    def _get_worst_performing(self, data: pd.DataFrame) -> str:
        """Identify worst performing units/models."""
        if 'Model' in data.columns and 'Sigma Pass' in data.columns:
            model_performance = data.groupby('Model')['Sigma Pass'].agg(['mean', 'count'])
            worst_model = model_performance['mean'].idxmin()
            pass_rate = model_performance.loc[worst_model, 'mean'] * 100
            count = model_performance.loc[worst_model, 'count']

            return f"The worst performing model is {worst_model} with only {pass_rate:.1f}% pass rate "                   f"across {count} units. Consider reviewing the specifications and process parameters for this model."

        return "Unable to determine worst performing model from available data."

    def _explain_term(self, term: str) -> str:
        """Explain a technical term."""
        term_lower = term.lower()

        for key, explanation in self.explanations.items():
            if key in term_lower or term_lower in key:
                return explanation.strip()

        # General explanations for common terms
        if 'pass' in term_lower or 'fail' in term_lower:
            return "Pass/fail status is determined by comparing the sigma gradient to a threshold. "                   "Units with sigma gradient below the threshold pass, indicating good quality."

        elif 'threshold' in term_lower:
            return "The threshold is the maximum acceptable value for sigma gradient. "                   "It's calculated based on the linearity specification and unit geometry."

        return f"I don't have a specific explanation for '{term}'. "               "Try asking about sigma gradient, linearity, failure probability, or resistance change."

    def _load_recent_data(self, days: int = 7) -> pd.DataFrame:
        """Load recent data from database."""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            
            # FIXED: Prevent SQL injection by using proper parameterization
            query = """
                SELECT 
                    model,
                    serial_number,
                    file_date,
                    sigma_gradient,
                    sigma_threshold,
                    sigma_pass,
                    linearity_error,
                    linearity_pass,
                    status
                FROM analysis_results 
                WHERE file_date >= date('now', '-' || ? || ' days')
                ORDER BY file_date DESC
            """
            
            # Use proper parameter binding - pass the number of days as an integer
            df = pd.read_sql_query(query, conn, params=[str(days)])
            conn.close()
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()


class ProfessionalReportGenerator:
    """
    Generate professional PDF reports with AI insights.
    """

    def __init__(self, output_dir: str):
        """Initialize report generator."""
        self.output_dir = output_dir
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2563eb'),
            spaceBefore=20,
            spaceAfter=10
        ))

        self.styles.add(ParagraphStyle(
            name='InsightText',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=10
        ))

    def generate_comprehensive_report(self,
                                      analysis_results: List[Dict[str, Any]],
                                      ai_insights: Dict[str, Any],
                                      company_name: str = "Potentiometer Manufacturing Co.") -> str:
        """
        Generate comprehensive PDF report with AI insights.

        Args:
            analysis_results: Analysis results data
            ai_insights: AI-generated insights
            company_name: Company name for report header

        Returns:
            Path to generated PDF report
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"QA_AI_Report_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)

        # Create document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        # Build content
        story = []

        # Title page
        story.extend(self._create_title_page(company_name))
        story.append(PageBreak())

        # Executive summary
        story.extend(self._create_executive_summary(ai_insights))
        story.append(PageBreak())

        # Detailed insights
        story.extend(self._create_insights_section(ai_insights))
        story.append(PageBreak())

        # Data analysis
        story.extend(self._create_data_analysis_section(analysis_results))
        story.append(PageBreak())

        # Recommendations
        story.extend(self._create_recommendations_section(ai_insights))
        story.append(PageBreak())

        # Predictive analysis
        if 'predictions' in ai_insights:
            story.extend(self._create_predictions_section(ai_insights['predictions']))

        # Build PDF
        doc.build(story)

        return filepath

    def _create_title_page(self, company_name: str) -> List:
        """Create title page content."""
        content = []

        # Company name
        content.append(Paragraph(company_name, self.styles['CustomTitle']))
        content.append(Spacer(1, 0.5 * inch))

        # Report title
        content.append(Paragraph(
            "Quality Analysis Report with AI Insights",
            self.styles['CustomTitle']
        ))
        content.append(Spacer(1, 0.3 * inch))

        # Date
        content.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%B %d, %Y')}",
            self.styles['Normal']
        ))

        content.append(Spacer(1, 2 * inch))

        # Report info
        info_data = [
            ['Report Type:', 'Comprehensive QA Analysis'],
            ['Analysis Method:', 'AI-Enhanced Statistical Analysis'],
            ['Confidence Level:', '95%'],
            ['Data Period:', 'Last 30 days']
        ]

        info_table = Table(info_data, colWidths=[2 * inch, 3 * inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))

        content.append(info_table)

        return content

    def _create_executive_summary(self, ai_insights: Dict[str, Any]) -> List:
        """Create executive summary section."""
        content = []

        content.append(Paragraph("Executive Summary", self.styles['SectionHeader']))

        # Summary text
        summary = ai_insights.get('summary', 'No summary available.')
        content.append(Paragraph(summary, self.styles['InsightText']))

        content.append(Spacer(1, 0.2 * inch))

        # Key metrics table
        if ai_insights.get('insights'):
            # Extract key numbers from insights
            positive_insights = [i for i in ai_insights['insights'] if i.get('type') == 'positive']
            warning_insights = [i for i in ai_insights['insights'] if i.get('type') == 'warning']

            metrics_data = [
                ['Metric', 'Status', 'Details'],
                ['Overall Health', 'Good' if len(positive_insights) > len(warning_insights) else 'Needs Attention',
                 f"{len(positive_insights)} positive, {len(warning_insights)} warnings"],
                ['Anomalies Detected', str(len(ai_insights.get('anomalies', []))),
                 'Requires investigation' if ai_insights.get('anomalies') else 'None'],
                ['Action Items', str(len(ai_insights.get('recommendations', []))),
                 'See recommendations section']
            ]

            metrics_table = Table(metrics_data, colWidths=[2 * inch, 1.5 * inch, 2.5 * inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            content.append(metrics_table)

        return content

    def _create_insights_section(self, ai_insights: Dict[str, Any]) -> List:
        """Create detailed insights section."""
        content = []

        content.append(Paragraph("Detailed AI Insights", self.styles['SectionHeader']))

        # Group insights by category
        insights_by_category = defaultdict(list)
        for insight in ai_insights.get('insights', []):
            insights_by_category[insight.get('category', 'general')].append(insight)

        for category, insights in insights_by_category.items():
            # Category header
            category_name = category.replace('_', ' ').title()
            content.append(Paragraph(f"{category_name}:", self.styles['Heading3']))

            # Insights list
            for insight in insights:
                icon = "✓" if insight['type'] == 'positive' else "⚠"
                text = f"{icon} {insight['message']}"

                if insight['type'] == 'positive':
                    style = 'InsightText'
                else:
                    style = 'InsightText'  # Could create warning style

                content.append(Paragraph(text, self.styles[style]))

            content.append(Spacer(1, 0.1 * inch))

        # Anomalies section
        if ai_insights.get('anomalies'):
            content.append(Paragraph("Anomalies Detected", self.styles['Heading3']))

            anomaly_data = [['Unit', 'Metric', 'Value', 'Expected Range', 'Severity']]
            for anomaly in ai_insights['anomalies'][:10]:  # Limit to 10
                anomaly_data.append([
                    anomaly['unit'],
                    anomaly['metric'],
                    f"{anomaly['value']:.4f}",
                    anomaly['expected_range'],
                    anomaly['severity'].upper()
                ])

            anomaly_table = Table(anomaly_data)
            anomaly_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            content.append(anomaly_table)

        return content

    def _create_data_analysis_section(self, analysis_results: List[Dict[str, Any]]) -> List:
        """Create data analysis section with charts."""
        content = []

        content.append(Paragraph("Statistical Analysis", self.styles['SectionHeader']))

        # Convert to DataFrame for analysis
        df = pd.DataFrame(analysis_results)

        if not df.empty:
            # Create and save charts
            chart_paths = self._create_analysis_charts(df)

            # Add charts to report
            for chart_path in chart_paths:
                if os.path.exists(chart_path):
                    img = Image(chart_path, width=5 * inch, height=3 * inch)
                    content.append(img)
                    content.append(Spacer(1, 0.2 * inch))

        return content

    def _create_analysis_charts(self, df: pd.DataFrame) -> List[str]:
        """Create analysis charts and save as images."""
        chart_paths = []

        # Chart 1: Pass rate by model
        if 'Model' in df.columns and 'Sigma Pass' in df.columns:
            plt.figure(figsize=(8, 5))
            model_pass = df.groupby('Model')['Sigma Pass'].mean() * 100
            model_pass.plot(kind='bar', color='skyblue', edgecolor='black')
            plt.title('Pass Rate by Model')
            plt.xlabel('Model')
            plt.ylabel('Pass Rate (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()

            chart1_path = os.path.join(self.output_dir, 'chart_pass_rate.png')
            plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
            plt.close()
            chart_paths.append(chart1_path)

        # Chart 2: Sigma gradient distribution
        if 'Sigma Gradient' in df.columns:
            plt.figure(figsize=(8, 5))
            plt.hist(df['Sigma Gradient'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
            plt.axvline(df['Sigma Gradient'].mean(), color='red', linestyle='--', label='Mean')
            plt.title('Sigma Gradient Distribution')
            plt.xlabel('Sigma Gradient')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()

            chart2_path = os.path.join(self.output_dir, 'chart_sigma_dist.png')
            plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
            plt.close()
            chart_paths.append(chart2_path)

        return chart_paths

    def _create_recommendations_section(self, ai_insights: Dict[str, Any]) -> List:
        """Create recommendations section."""
        content = []

        content.append(Paragraph("Recommendations", self.styles['SectionHeader']))

        recommendations = ai_insights.get('recommendations', [])

        if recommendations:
            # Priority levels
            high_priority = []
            medium_priority = []
            low_priority = []

            # Categorize recommendations (simple heuristic)
            for rec in recommendations:
                if any(word in rec.lower() for word in ['immediately', 'urgent', 'critical']):
                    high_priority.append(rec)
                elif any(word in rec.lower() for word in ['consider', 'review', 'check']):
                    medium_priority.append(rec)
                else:
                    low_priority.append(rec)

            # High priority
            if high_priority:
                content.append(Paragraph("High Priority Actions:", self.styles['Heading3']))
                for i, rec in enumerate(high_priority, 1):
                    content.append(Paragraph(f"{i}. {rec}", self.styles['InsightText']))
                content.append(Spacer(1, 0.1 * inch))

            # Medium priority
            if medium_priority:
                content.append(Paragraph("Medium Priority Actions:", self.styles['Heading3']))
                for i, rec in enumerate(medium_priority, 1):
                    content.append(Paragraph(f"{i}. {rec}", self.styles['InsightText']))
                content.append(Spacer(1, 0.1 * inch))

            # Low priority
            if low_priority:
                content.append(Paragraph("Additional Suggestions:", self.styles['Heading3']))
                for i, rec in enumerate(low_priority, 1):
                    content.append(Paragraph(f"{i}. {rec}", self.styles['InsightText']))
        else:
            content.append(Paragraph(
                "No specific recommendations at this time. Continue monitoring current processes.",
                self.styles['InsightText']
            ))

        return content

    def _create_predictions_section(self, predictions: Dict[str, Any]) -> List:
        """Create predictive analysis section."""
        content = []

        content.append(Paragraph("Predictive Analysis", self.styles['SectionHeader']))

        # Maintenance predictions
        if predictions.get('maintenance_needed'):
            content.append(Paragraph("Predicted Maintenance Requirements:", self.styles['Heading3']))

            maint_data = [['Equipment', 'Urgency', 'Predicted Date', 'Reason']]
            for pred in predictions['maintenance_needed']:
                maint_data.append([
                    pred['equipment'],
                    pred['urgency'].upper(),
                    pred['predicted_date'].strftime('%Y-%m-%d'),
                    pred['reason']
                ])

            maint_table = Table(maint_data, colWidths=[2 * inch, 1 * inch, 1.5 * inch, 2.5 * inch])
            maint_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.orange),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            content.append(maint_table)
            content.append(Spacer(1, 0.2 * inch))

        # Quality risk predictions
        if predictions.get('quality_risks'):
            content.append(Paragraph("Quality Risk Predictions:", self.styles['Heading3']))

            risk_data = [['Model', 'Risk Level', 'Predicted Failures (%)', 'Action']]
            for risk in predictions['quality_risks']:
                risk_data.append([
                    risk['model'],
                    risk['risk_level'].upper(),
                    f"{risk['predicted_failures']}%",
                    risk['recommendation']
                ])

            risk_table = Table(risk_data, colWidths=[1.5 * inch, 1 * inch, 1.5 * inch, 3 * inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            content.append(risk_table)

        return content


class PredictiveMaintenanceSystem:
    """
    Predictive maintenance system for equipment monitoring.
    """

    def __init__(self, db_path: str):
        """Initialize predictive maintenance system."""
        self.db_path = db_path
        self.equipment_profiles = {
            'laser_system': {
                'degradation_rate': 0.0001,  # Per day
                'threshold': 0.02,
                'maintenance_interval': 90  # Days
            },
            'alignment_system': {
                'degradation_rate': 0.00005,
                'threshold': 0.015,
                'maintenance_interval': 180
            }
        }

    def predict_equipment_maintenance(self, historical_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Predict when equipment will need maintenance.

        Args:
            historical_data: Historical performance data

        Returns:
            List of maintenance predictions
        """
        predictions = []

        if historical_data.empty:
            return predictions

        # Analyze trends for each equipment type
        for equipment, profile in self.equipment_profiles.items():
            prediction = self._analyze_equipment_trend(
                historical_data,
                equipment,
                profile
            )
            if prediction:
                predictions.append(prediction)

        return predictions

    def _analyze_equipment_trend(self, data: pd.DataFrame,
                                 equipment: str, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze trend for specific equipment."""
        # Simple linear regression on sigma gradient over time
        if 'timestamp' not in data.columns or 'sigma_gradient' not in data.columns:
            return None

        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp')

        # Calculate days from first measurement
        data['days'] = (data['timestamp'] - data['timestamp'].min()).dt.days

        # Fit linear model
        if len(data) < 10:
            return None

        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            data['days'],
            data['sigma_gradient']
        )

        # Predict when threshold will be reached
        current_value = data['sigma_gradient'].iloc[-1]

        if slope > 0:  # Degrading
            days_to_threshold = (profile['threshold'] - current_value) / slope

            if days_to_threshold < profile['maintenance_interval']:
                return {
                    'equipment': equipment.replace('_', ' ').title(),
                    'current_value': current_value,
                    'threshold': profile['threshold'],
                    'days_to_threshold': int(days_to_threshold),
                    'predicted_date': datetime.now() + timedelta(days=days_to_threshold),
                    'confidence': abs(r_value),
                    'urgency': 'high' if days_to_threshold < 30 else 'medium',
                    'recommendation': self._get_maintenance_recommendation(equipment, days_to_threshold)
                }

        return None

    def _get_maintenance_recommendation(self, equipment: str, days_to_threshold: int) -> str:
        """Get specific maintenance recommendation."""
        if equipment == 'laser_system':
            if days_to_threshold < 14:
                return "Schedule immediate laser calibration and alignment check."
            elif days_to_threshold < 30:
                return "Plan laser maintenance within next 2 weeks. Check power output and beam quality."
            else:
                return "Monitor laser performance weekly. Schedule maintenance within 30 days."

        elif equipment == 'alignment_system':
            if days_to_threshold < 30:
                return "Check mechanical alignment and adjust if necessary."
            else:
                return "Schedule routine alignment verification."

        return "Schedule preventive maintenance."

    def detect_anomalous_patterns(self, recent_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect anomalous patterns that might indicate equipment issues.

        Args:
            recent_data: Recent measurement data

        Returns:
            List of detected anomalies
        """
        anomalies = []

        if 'sigma_gradient' not in recent_data.columns:
            return anomalies

        # Check for sudden changes
        if len(recent_data) > 10:
            recent_data = recent_data.sort_values('timestamp')

            # Calculate rolling statistics
            window = min(5, len(recent_data) // 2)
            rolling_mean = recent_data['sigma_gradient'].rolling(window).mean()
            rolling_std = recent_data['sigma_gradient'].rolling(window).std()

            # Detect points outside 3 standard deviations
            for i in range(window, len(recent_data)):
                value = recent_data['sigma_gradient'].iloc[i]
                mean = rolling_mean.iloc[i]
                std = rolling_std.iloc[i]

                if std > 0 and abs(value - mean) > 3 * std:
                    anomalies.append({
                        'timestamp': recent_data['timestamp'].iloc[i],
                        'type': 'sudden_change',
                        'value': value,
                        'expected_range': f"{mean - 2 * std:.4f} to {mean + 2 * std:.4f}",
                        'severity': 'high',
                        'possible_cause': 'Equipment malfunction or material change',
                        'action': 'Investigate immediately and check equipment calibration'
                    })

        # Check for increasing variance (indicates instability)
        if len(recent_data) > 20:
            first_half_std = recent_data['sigma_gradient'].iloc[:len(recent_data) // 2].std()
            second_half_std = recent_data['sigma_gradient'].iloc[len(recent_data) // 2:].std()

            if second_half_std > first_half_std * 1.5:
                anomalies.append({
                    'timestamp': datetime.now(),
                    'type': 'increasing_variance',
                    'value': second_half_std / first_half_std,
                    'severity': 'medium',
                    'possible_cause': 'Equipment wearing out or environmental changes',
                    'action': 'Schedule equipment inspection'
                })

        return anomalies


# Example usage and integration
def main():
    """Example of how to use the AI-powered QA system."""
    import os

    # Setup paths
    db_path = os.path.join(os.path.expanduser("~"), "LaserTrimResults", "analysis_history.db")
    output_dir = os.path.join(os.path.expanduser("~"), "LaserTrimResults", "AI_Reports")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize AI components
    ai_analyzer = AIQualityAnalyzer(db_path)
    qa_assistant = InteractiveQAAssistant(db_path)
    report_generator = ProfessionalReportGenerator(output_dir)
    maintenance_system = PredictiveMaintenanceSystem(db_path)

    # Load recent analysis results (example)
    # In real use, this would come from your analysis system
    analysis_results = [
        {
            'File': '8340_A12345.xlsx',
            'Model': '8340',
            'Serial': 'A12345',
            'Sigma Pass': True,
            'Sigma Gradient': 0.0045,
            'Failure Probability': 0.15,
            'Risk Category': 'Low',
            'Resistance Change (%)': 3.5
        },
        # ... more results
    ]

    # Generate AI insights
    print("Generating AI insights...")
    insights = ai_analyzer.generate_insights(analysis_results)

    # Print insights
    print("\nAI INSIGHTS:")
    print("=" * 50)
    for insight in insights['insights']:
        print(f"- {insight['message']} ({insight['type']})")

    print("\nRECOMMENDATIONS:")
    for rec in insights['recommendations']:
        print(f"- {rec}")

    # Interactive Q&A example
    print("\nINTERACTIVE Q&A:")
    print("=" * 50)

    questions = [
        "What is the pass rate for model 8340?",
        "Explain sigma gradient",
        "Which model is performing worst?",
        "Compare 8340 and 8555"
    ]

    for question in questions:
        print(f"\nQ: {question}")
        answer = qa_assistant.answer_question(question, pd.DataFrame(analysis_results))
        print(f"A: {answer}")

    # Generate professional report
    print("\nGenerating professional PDF report...")
    report_path = report_generator.generate_comprehensive_report(
        analysis_results,
        insights,
        "Acme Potentiometer Manufacturing"
    )
    print(f"Report saved to: {report_path}")

    # Predictive maintenance example
    print("\nPREDICTIVE MAINTENANCE:")
    print("=" * 50)

    # Load historical data for predictions
    import sqlite3
    conn = sqlite3.connect(db_path)
    historical_data = pd.read_sql_query(
        "SELECT * FROM track_results WHERE timestamp >= datetime('now', '-90 days')",
        conn
    )
    conn.close()

    if not historical_data.empty:
        maintenance_predictions = maintenance_system.predict_equipment_maintenance(historical_data)

        for pred in maintenance_predictions:
            print(f"\n{pred['equipment']}:")
            print(f"  - Predicted maintenance: {pred['predicted_date'].strftime('%Y-%m-%d')}")
            print(f"  - Days until threshold: {pred['days_to_threshold']}")
            print(f"  - Recommendation: {pred['recommendation']}")

    print("\nAI analysis complete!")


if __name__ == "__main__":
    main()