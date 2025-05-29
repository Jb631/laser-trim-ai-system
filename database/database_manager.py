"""
Database Manager for Laser Trim AI System

This module handles all database operations including storing analysis results,
querying historical data, and managing the database connection.

Author: Laser Trim AI System
Date: 2024
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import pandas as pd
import numpy as np

from core.config import Config


class DatabaseManager:
    """Manages all database operations for the Laser Trim AI System."""

    def __init__(self, config: Config):
        """
        Initialize database manager.

        Args:
            config: System configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(config.OUTPUT_DIR) / "laser_trim_ai.db"
        self._init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    input_folder TEXT,
                    total_files INTEGER,
                    processed_files INTEGER,
                    failed_files INTEGER,
                    processing_time REAL,
                    configuration TEXT,
                    notes TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    filename TEXT,
                    model TEXT,
                    serial TEXT,
                    system TEXT,
                    sigma_gradient REAL,
                    sigma_threshold REAL,
                    sigma_pass BOOLEAN,
                    linearity_pass BOOLEAN,
                    overall_status TEXT,
                    failure_probability REAL,
                    risk_category TEXT,
                    ml_prediction TEXT,
                    ml_confidence REAL,
                    raw_data TEXT,
                    FOREIGN KEY (run_id) REFERENCES analysis_runs(id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model TEXT,
                    total_units INTEGER,
                    pass_rate REAL,
                    avg_sigma_gradient REAL,
                    std_sigma_gradient REAL,
                    avg_failure_probability REAL,
                    trend_direction TEXT,
                    anomaly_count INTEGER
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    file_id INTEGER,
                    anomaly_type TEXT,
                    severity TEXT,
                    description TEXT,
                    parameters TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (file_id) REFERENCES file_results(id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    file_id INTEGER,
                    model_name TEXT,
                    prediction TEXT,
                    confidence REAL,
                    actual_result TEXT,
                    correct BOOLEAN,
                    features_used TEXT,
                    FOREIGN KEY (file_id) REFERENCES file_results(id)
                )
            ''')

            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model ON file_results(model)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON file_results(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON file_results(overall_status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk ON file_results(risk_category)')

            self.logger.info(f"Database initialized at {self.db_path}")

    def create_analysis_run(self, input_folder: str, configuration: Dict) -> int:
        """
        Create a new analysis run entry.

        Args:
            input_folder: Path to input data folder
            configuration: Configuration used for this run

        Returns:
            Run ID for the new analysis run
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analysis_runs (input_folder, configuration, total_files)
                VALUES (?, ?, 0)
            ''', (input_folder, json.dumps(configuration)))
            return cursor.lastrowid

    def update_analysis_run(self, run_id: int, **kwargs):
        """Update analysis run with results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Build update query dynamically
            updates = []
            values = []
            for key, value in kwargs.items():
                updates.append(f"{key} = ?")
                values.append(value)

            values.append(run_id)
            query = f"UPDATE analysis_runs SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, values)

    def save_file_result(self, run_id: int, result: Dict[str, Any]) -> int:
        """
        Save individual file analysis result.

        Args:
            run_id: ID of the analysis run
            result: Analysis result dictionary

        Returns:
            ID of the saved file result
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Extract ML predictions if available
            ml_data = result.get('ml_analysis', {})

            cursor.execute('''
                INSERT INTO file_results (
                    run_id, filename, model, serial, system,
                    sigma_gradient, sigma_threshold, sigma_pass,
                    linearity_pass, overall_status, failure_probability,
                    risk_category, ml_prediction, ml_confidence, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                result.get('filename', ''),
                result.get('model', ''),
                result.get('serial', ''),
                result.get('system', ''),
                result.get('sigma_gradient', 0.0),
                result.get('sigma_threshold', 0.0),
                result.get('sigma_pass', False),
                result.get('linearity_pass', False),
                result.get('overall_status', 'Unknown'),
                result.get('failure_probability', 0.0),
                result.get('risk_category', 'Unknown'),
                ml_data.get('prediction', ''),
                ml_data.get('confidence', 0.0),
                json.dumps(result)
            ))

            file_id = cursor.lastrowid

            # Save ML prediction details if available
            if ml_data:
                self._save_ml_prediction(file_id, ml_data, result.get('overall_status', ''))

            # Check for anomalies
            anomalies = self._detect_anomalies(result)
            for anomaly in anomalies:
                self._save_anomaly(file_id, anomaly)

            return file_id

    def _save_ml_prediction(self, file_id: int, ml_data: Dict, actual_result: str):
        """Save ML prediction details."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            prediction = ml_data.get('prediction', '')
            correct = prediction.lower() == actual_result.lower()

            cursor.execute('''
                INSERT INTO ml_predictions (
                    file_id, model_name, prediction, confidence,
                    actual_result, correct, features_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_id,
                ml_data.get('model_name', 'default'),
                prediction,
                ml_data.get('confidence', 0.0),
                actual_result,
                correct,
                json.dumps(ml_data.get('features', []))
            ))

    def _detect_anomalies(self, result: Dict) -> List[Dict]:
        """Detect anomalies in analysis results."""
        anomalies = []

        # Check for unusually high sigma gradient
        sigma_gradient = result.get('sigma_gradient', 0)
        if sigma_gradient > self.config.ANOMALY_THRESHOLDS['high_sigma']:
            anomalies.append({
                'type': 'high_sigma_gradient',
                'severity': 'high',
                'description': f'Sigma gradient ({sigma_gradient:.4f}) exceeds threshold',
                'parameters': {'sigma_gradient': sigma_gradient}
            })

        # Check for failure probability anomaly
        failure_prob = result.get('failure_probability', 0)
        if failure_prob > self.config.ANOMALY_THRESHOLDS['high_failure_prob']:
            anomalies.append({
                'type': 'high_failure_probability',
                'severity': 'medium',
                'description': f'High failure probability detected ({failure_prob:.2%})',
                'parameters': {'failure_probability': failure_prob}
            })

        # Check for resistance anomalies
        resistance_change = abs(result.get('resistance_change_percent', 0))
        if resistance_change > self.config.ANOMALY_THRESHOLDS['resistance_change']:
            anomalies.append({
                'type': 'resistance_anomaly',
                'severity': 'medium',
                'description': f'Large resistance change detected ({resistance_change:.1f}%)',
                'parameters': {'resistance_change': resistance_change}
            })

        return anomalies

    def _save_anomaly(self, file_id: int, anomaly: Dict):
        """Save detected anomaly."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO anomalies (
                    file_id, anomaly_type, severity, description, parameters
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                file_id,
                anomaly['type'],
                anomaly['severity'],
                anomaly['description'],
                json.dumps(anomaly['parameters'])
            ))

    def get_historical_data(self,
                            model: Optional[str] = None,
                            days_back: int = 30,
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve historical analysis data.

        Args:
            model: Filter by model number (optional)
            days_back: Number of days to look back
            limit: Maximum number of records to return

        Returns:
            DataFrame with historical data
        """
        query = '''
            SELECT * FROM file_results
            WHERE timestamp > datetime('now', ?)
        '''
        params = [f'-{days_back} days']

        if model:
            query += ' AND model = ?'
            params.append(model)

        query += ' ORDER BY timestamp DESC'

        if limit:
            query += f' LIMIT {limit}'

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])

        # Parse JSON fields
        if not df.empty:
            df['raw_data'] = df['raw_data'].apply(json.loads)

        return df

    def get_model_performance_history(self, model: str, days_back: int = 90) -> pd.DataFrame:
        """Get performance history for a specific model."""
        query = '''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_units,
                AVG(CASE WHEN sigma_pass THEN 1 ELSE 0 END) as pass_rate,
                AVG(sigma_gradient) as avg_sigma,
                AVG(failure_probability) as avg_failure_prob,
                SUM(CASE WHEN risk_category = 'High' THEN 1 ELSE 0 END) as high_risk_count
            FROM file_results
            WHERE model = ? AND timestamp > datetime('now', ?)
            GROUP BY DATE(timestamp)
            ORDER BY date
        '''

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn,
                                   params=(model, f'-{days_back} days'),
                                   parse_dates=['date'])
        return df

    def update_model_performance(self):
        """Update model performance statistics."""
        query = '''
            SELECT 
                model,
                COUNT(*) as total_units,
                AVG(CASE WHEN sigma_pass AND linearity_pass THEN 1 ELSE 0 END) as pass_rate,
                AVG(sigma_gradient) as avg_sigma,
                STDEV(sigma_gradient) as std_sigma,
                AVG(failure_probability) as avg_failure_prob,
                SUM(CASE WHEN risk_category = 'High' THEN 1 ELSE 0 END) as anomaly_count
            FROM file_results
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY model
        '''

        with self.get_connection() as conn:
            cursor = conn.cursor()
            results = cursor.execute(query).fetchall()

            for row in results:
                # Calculate trend
                trend = self._calculate_trend(row['model'])

                cursor.execute('''
                    INSERT INTO model_performance (
                        model, total_units, pass_rate, avg_sigma_gradient,
                        std_sigma_gradient, avg_failure_probability,
                        trend_direction, anomaly_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['model'], row['total_units'], row['pass_rate'],
                    row['avg_sigma'], row['std_sigma'], row['avg_failure_prob'],
                    trend, row['anomaly_count']
                ))

    def _calculate_trend(self, model: str) -> str:
        """Calculate performance trend for a model."""
        # Get recent vs older performance
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Recent week
            recent = cursor.execute('''
                SELECT AVG(CASE WHEN sigma_pass THEN 1 ELSE 0 END) as pass_rate
                FROM file_results
                WHERE model = ? AND timestamp > datetime('now', '-7 days')
            ''', (model,)).fetchone()

            # Previous week
            older = cursor.execute('''
                SELECT AVG(CASE WHEN sigma_pass THEN 1 ELSE 0 END) as pass_rate
                FROM file_results
                WHERE model = ? 
                AND timestamp > datetime('now', '-14 days')
                AND timestamp <= datetime('now', '-7 days')
            ''', (model,)).fetchone()

            if recent and older and recent['pass_rate'] and older['pass_rate']:
                diff = recent['pass_rate'] - older['pass_rate']
                if diff > 0.05:
                    return 'improving'
                elif diff < -0.05:
                    return 'declining'

        return 'stable'

    def get_anomaly_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get summary of recent anomalies."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Count by type
            type_counts = cursor.execute('''
                SELECT anomaly_type, COUNT(*) as count
                FROM anomalies
                WHERE timestamp > datetime('now', ?)
                GROUP BY anomaly_type
            ''', (f'-{days_back} days',)).fetchall()

            # Count by severity
            severity_counts = cursor.execute('''
                SELECT severity, COUNT(*) as count
                FROM anomalies
                WHERE timestamp > datetime('now', ?)
                GROUP BY severity
            ''', (f'-{days_back} days',)).fetchall()

            # Unresolved anomalies
            unresolved = cursor.execute('''
                SELECT COUNT(*) as count
                FROM anomalies
                WHERE resolved = 0 AND timestamp > datetime('now', ?)
            ''', (f'-{days_back} days',)).fetchone()

        return {
            'by_type': {row['anomaly_type']: row['count'] for row in type_counts},
            'by_severity': {row['severity']: row['count'] for row in severity_counts},
            'unresolved_count': unresolved['count'] if unresolved else 0
        }

    def get_ml_model_accuracy(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get ML model accuracy statistics."""
        query = '''
            SELECT 
                model_name,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct_predictions,
                AVG(confidence) as avg_confidence
            FROM ml_predictions
            WHERE timestamp > datetime('now', '-30 days')
        '''

        if model_name:
            query += ' AND model_name = ?'
            params = (model_name,)
        else:
            params = ()

        query += ' GROUP BY model_name'

        with self.get_connection() as conn:
            cursor = conn.cursor()
            results = cursor.execute(query, params).fetchall()

        accuracy_stats = {}
        for row in results:
            accuracy = row['correct_predictions'] / row['total_predictions'] if row['total_predictions'] > 0 else 0
            accuracy_stats[row['model_name']] = {
                'accuracy': accuracy,
                'total_predictions': row['total_predictions'],
                'avg_confidence': row['avg_confidence']
            }

        return accuracy_stats

    def export_to_excel(self, output_path: Path, days_back: int = 30):
        """Export historical data to Excel file."""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # File results
            df_results = self.get_historical_data(days_back=days_back)
            df_results.to_excel(writer, sheet_name='Analysis Results', index=False)

            # Model performance
            with self.get_connection() as conn:
                df_perf = pd.read_sql_query(
                    'SELECT * FROM model_performance ORDER BY timestamp DESC LIMIT 100',
                    conn, parse_dates=['timestamp']
                )
            df_perf.to_excel(writer, sheet_name='Model Performance', index=False)

            # Anomalies
            with self.get_connection() as conn:
                df_anomalies = pd.read_sql_query(
                    '''SELECT a.*, f.filename, f.model 
                       FROM anomalies a 
                       JOIN file_results f ON a.file_id = f.id
                       WHERE a.timestamp > datetime('now', ?)
                       ORDER BY a.timestamp DESC''',
                    conn, params=(f'-{days_back} days',), parse_dates=['timestamp']
                )
            df_anomalies.to_excel(writer, sheet_name='Anomalies', index=False)

        self.logger.info(f"Exported database to {output_path}")

    def cleanup_old_data(self, days_to_keep: int = 365):
        """Remove old data to prevent database bloat."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Delete old file results and related data
            cursor.execute('''
                DELETE FROM file_results
                WHERE timestamp < datetime('now', ?)
            ''', (f'-{days_to_keep} days',))

            deleted = cursor.rowcount
            self.logger.info(f"Cleaned up {deleted} old records")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Table row counts
            tables = ['analysis_runs', 'file_results', 'model_performance',
                      'anomalies', 'ml_predictions']

            for table in tables:
                count = cursor.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                stats[f'{table}_count'] = count

            # Database file size
            stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)

            # Date range
            dates = cursor.execute('''
                SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest
                FROM file_results
            ''').fetchone()

            if dates['oldest']:
                stats['oldest_record'] = dates['oldest']
                stats['newest_record'] = dates['newest']

        return stats