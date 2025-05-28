"""
Database Tests for Laser Trim AI System

Comprehensive test suite for database functionality including
storage, retrieval, analysis, and reporting features.

Author: Laser Trim AI System
Date: 2024
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd

from database_manager import DatabaseManager
from historical_analyzer import HistoricalAnalyzer
from trend_reporter import TrendReporter
from data_migrator import DataMigrator
from ..config import Config


class TestDatabaseManager(unittest.TestCase):
    """Test database manager functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.OUTPUT_DIR = self.test_dir
        self.db = DatabaseManager(self.config)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def test_database_creation(self):
        """Test database is created correctly."""
        db_path = Path(self.test_dir) / "laser_trim_ai.db"
        self.assertTrue(db_path.exists())

    def test_create_analysis_run(self):
        """Test creating analysis run."""
        run_id = self.db.create_analysis_run(
            input_folder="/test/input",
            configuration={'test': True}
        )
        self.assertIsInstance(run_id, int)
        self.assertGreater(run_id, 0)

    def test_save_file_result(self):
        """Test saving file analysis result."""
        # Create run
        run_id = self.db.create_analysis_run("/test", {})

        # Create test result
        result = {
            'filename': 'test_file.xlsx',
            'model': 'TEST123',
            'serial': 'S12345',
            'system': 'A',
            'sigma_gradient': 0.0234,
            'sigma_threshold': 0.0300,
            'sigma_pass': True,
            'linearity_pass': True,
            'overall_status': 'Pass',
            'failure_probability': 0.05,
            'risk_category': 'Low',
            'ml_analysis': {
                'prediction': 'Pass',
                'confidence': 0.95,
                'model_name': 'test_model'
            }
        }

        # Save result
        file_id = self.db.save_file_result(run_id, result)
        self.assertIsInstance(file_id, int)

        # Verify saved data
        df = self.db.get_historical_data(limit=1)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['model'], 'TEST123')

    def test_anomaly_detection(self):
        """Test anomaly detection and storage."""
        run_id = self.db.create_analysis_run("/test", {})

        # Create result with anomaly
        result = {
            'filename': 'anomaly_test.xlsx',
            'model': 'TEST123',
            'sigma_gradient': 0.1,  # Very high
            'sigma_threshold': 0.03,
            'sigma_pass': False,
            'overall_status': 'Fail',
            'failure_probability': 0.9,
            'risk_category': 'High',
            'resistance_change_percent': 25  # High change
        }

        file_id = self.db.save_file_result(run_id, result)

        # Check anomalies were detected
        anomaly_summary = self.db.get_anomaly_summary(days_back=1)
        self.assertGreater(sum(anomaly_summary['by_type'].values()), 0)

    def test_model_performance_update(self):
        """Test model performance statistics update."""
        # Add some test data
        run_id = self.db.create_analysis_run("/test", {})

        for i in range(10):
            result = {
                'filename': f'test_{i}.xlsx',
                'model': 'MODEL_A' if i < 5 else 'MODEL_B',
                'sigma_gradient': 0.02 + i * 0.001,
                'sigma_threshold': 0.03,
                'sigma_pass': i < 8,
                'linearity_pass': i < 7,
                'failure_probability': i * 0.1,
                'risk_category': 'Low' if i < 5 else 'High'
            }
            self.db.save_file_result(run_id, result)

        # Update performance
        self.db.update_model_performance()

        # Check performance data exists
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            count = cursor.execute('SELECT COUNT(*) FROM model_performance').fetchone()[0]
            self.assertGreater(count, 0)

    def test_ml_accuracy_tracking(self):
        """Test ML model accuracy tracking."""
        run_id = self.db.create_analysis_run("/test", {})

        # Add results with ML predictions
        for i in range(20):
            result = {
                'filename': f'ml_test_{i}.xlsx',
                'model': 'TEST',
                'sigma_gradient': 0.02,
                'overall_status': 'Pass' if i % 2 == 0 else 'Fail',
                'ml_analysis': {
                    'prediction': 'Pass' if i % 3 == 0 else 'Fail',
                    'confidence': 0.8 + (i % 10) * 0.02,
                    'model_name': 'rf_classifier'
                }
            }
            self.db.save_file_result(run_id, result)

        # Get accuracy stats
        accuracy = self.db.get_ml_model_accuracy()
        self.assertIn('rf_classifier', accuracy)
        self.assertGreater(accuracy['rf_classifier']['total_predictions'], 0)

    def test_export_to_excel(self):
        """Test exporting data to Excel."""
        # Add test data
        run_id = self.db.create_analysis_run("/test", {})

        for i in range(5):
            result = {
                'filename': f'export_test_{i}.xlsx',
                'model': f'MODEL_{i}',
                'sigma_gradient': 0.02 + i * 0.001,
                'sigma_pass': True
            }
            self.db.save_file_result(run_id, result)

        # Export
        export_path = Path(self.test_dir) / 'export_test.xlsx'
        self.db.export_to_excel(export_path, days_back=1)

        self.assertTrue(export_path.exists())

        # Verify content
        df = pd.read_excel(export_path, sheet_name='Analysis Results')
        self.assertEqual(len(df), 5)

    def test_cleanup_old_data(self):
        """Test cleaning up old data."""
        run_id = self.db.create_analysis_run("/test", {})

        # Add old data (manually set timestamp)
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            old_date = (datetime.now() - timedelta(days=400)).isoformat()

            cursor.execute('''
                INSERT INTO file_results (run_id, filename, timestamp)
                VALUES (?, ?, ?)
            ''', (run_id, 'old_file.xlsx', old_date))

        # Add recent data
        self.db.save_file_result(run_id, {'filename': 'new_file.xlsx'})

        # Clean up
        self.db.cleanup_old_data(days_to_keep=365)

        # Check old data is gone
        df = self.db.get_historical_data(days_back=500)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['filename'], 'new_file.xlsx')


class TestHistoricalAnalyzer(unittest.TestCase):
    """Test historical analysis functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.OUTPUT_DIR = self.test_dir
        self.db = DatabaseManager(self.config)
        self.analyzer = HistoricalAnalyzer(self.db, self.config)

        # Add test data
        self._add_test_data()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def _add_test_data(self):
        """Add test data for analysis."""
        run_id = self.db.create_analysis_run("/test", {})

        # Add data with trends
        for day in range(30):
            date = datetime.now() - timedelta(days=29 - day)

            # Simulate improving trend
            base_pass_rate = 0.7 + day * 0.01

            for i in range(10):
                result = {
                    'filename': f'test_{day}_{i}.xlsx',
                    'model': 'MODEL_A' if i < 5 else 'MODEL_B',
                    'sigma_gradient': 0.025 - day * 0.0002,
                    'sigma_threshold': 0.03,
                    'sigma_pass': i / 10 < base_pass_rate,
                    'linearity_pass': i / 10 < base_pass_rate - 0.05,
                    'failure_probability': max(0, 0.3 - day * 0.01),
                    'risk_category': 'High' if i > 7 else 'Low'
                }

                # Manually set timestamp
                file_id = self.db.save_file_result(run_id, result)
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'UPDATE file_results SET timestamp = ? WHERE id = ?',
                        (date.isoformat(), file_id)
                    )

    def test_analyze_model_trends(self):
        """Test model trend analysis."""
        analysis = self.analyzer.analyze_model_trends('MODEL_A', days_back=30)

        self.assertIn('model', analysis)
        self.assertEqual(analysis['model'], 'MODEL_A')
        self.assertIn('pass_rate_trend', analysis)
        self.assertIn('sigma_analysis', analysis)

        # Should detect improving trend
        if 'pass_rate_trend' in analysis and 'direction' in analysis['pass_rate_trend']:
            self.assertEqual(analysis['pass_rate_trend']['direction'], 'improving')

    def test_compare_models(self):
        """Test model comparison."""
        comparison = self.analyzer.compare_models(['MODEL_A', 'MODEL_B'], days_back=30)

        self.assertEqual(len(comparison), 2)
        self.assertIn('model', comparison.columns)
        self.assertIn('pass_rate', comparison.columns)

    def test_detect_anomaly_clusters(self):
        """Test anomaly cluster detection."""
        # Add some anomalies
        run_id = self.db.create_analysis_run("/test", {})

        for i in range(10):
            result = {
                'filename': f'anomaly_{i}.xlsx',
                'model': 'ANOMALY_MODEL',
                'sigma_gradient': 0.08,  # High
                'failure_probability': 0.8,
                'risk_category': 'High'
            }
            self.db.save_file_result(run_id, result)

        clusters = self.analyzer.detect_anomaly_clusters(days_back=1)

        self.assertIn('clusters', clusters)
        self.assertIn('total_anomalies', clusters)
        self.assertGreater(clusters['total_anomalies'], 0)

    def test_improvement_recommendations(self):
        """Test improvement recommendation generation."""
        recommendations = self.analyzer.generate_improvement_recommendations('MODEL_A')

        self.assertIsInstance(recommendations, list)
        if recommendations:
            self.assertIn('recommendation', recommendations[0])
            self.assertIn('priority', recommendations[0])

    def test_cost_impact_calculation(self):
        """Test cost impact analysis."""
        cost_analysis = self.analyzer.calculate_cost_impact('MODEL_A', days_back=30)

        self.assertIn('costs', cost_analysis)
        self.assertIn('total', cost_analysis['costs'])
        self.assertIn('potential_savings', cost_analysis)


class TestTrendReporter(unittest.TestCase):
    """Test trend reporting functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.OUTPUT_DIR = self.test_dir
        self.db = DatabaseManager(self.config)
        self.analyzer = HistoricalAnalyzer(self.db, self.config)
        self.reporter = TrendReporter(self.db, self.analyzer, self.config)

        # Add test data
        self._add_test_data()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def _add_test_data(self):
        """Add test data for reporting."""
        run_id = self.db.create_analysis_run("/test", {})

        for i in range(20):
            result = {
                'filename': f'report_test_{i}.xlsx',
                'model': f'MODEL_{i % 3}',
                'sigma_gradient': 0.02 + (i % 5) * 0.002,
                'sigma_pass': i % 4 != 0,
                'linearity_pass': i % 5 != 0,
                'risk_category': ['Low', 'Medium', 'High'][i % 3]
            }
            self.db.save_file_result(run_id, result)

    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        report_path = self.reporter.generate_comprehensive_report(
            Path(self.test_dir),
            days_back=30
        )

        self.assertTrue(report_path.exists())
        self.assertTrue(report_path.suffix == '.xlsx')

        # Check visualizations were created
        viz_dir = list(Path(self.test_dir).glob('trend_visualizations_*'))
        self.assertEqual(len(viz_dir), 1)

    def test_generate_quick_report(self):
        """Test quick report generation."""
        report_path = Path(self.test_dir) / 'quick_report.txt'

        summary = self.reporter.generate_quick_report('MODEL_0', report_path)

        self.assertTrue(report_path.exists())
        self.assertIn('model', summary)
        self.assertEqual(summary['model'], 'MODEL_0')


class TestDataMigrator(unittest.TestCase):
    """Test data migration functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.OUTPUT_DIR = self.test_dir
        self.db = DatabaseManager(self.config)
        self.migrator = DataMigrator(self.db, self.config)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def test_import_from_excel(self):
        """Test importing from Excel file."""
        # Create test Excel file
        test_data = pd.DataFrame({
            'File': ['test1.xlsx', 'test2.xlsx'],
            'Model': ['MODEL_A', 'MODEL_B'],
            'Serial': ['S001', 'S002'],
            'Sigma Gradient': [0.02, 0.03],
            'Sigma Pass': [True, False],
            'Overall Status': ['Pass', 'Fail']
        })

        excel_path = Path(self.test_dir) / 'test_import.xlsx'
        test_data.to_excel(excel_path, index=False)

        # Import
        result = self.migrator.import_from_excel(excel_path)

        self.assertTrue(result['success'])
        self.assertEqual(result['imported'], 2)
        self.assertEqual(result['failed'], 0)

        # Verify data in database
        df = self.db.get_historical_data()
        self.assertEqual(len(df), 2)

    def test_import_from_json(self):
        """Test importing from JSON file."""
        # Create test JSON file
        test_data = [
            {
                'filename': 'json_test1.xlsx',
                'model': 'JSON_MODEL',
                'sigma_gradient': 0.025,
                'sigma_pass': True
            },
            {
                'filename': 'json_test2.xlsx',
                'model': 'JSON_MODEL',
                'sigma_gradient': 0.035,
                'sigma_pass': False
            }
        ]

        json_path = Path(self.test_dir) / 'test_import.json'
        with open(json_path, 'w') as f:
            json.dump(test_data, f)

        # Import
        result = self.migrator.import_from_json(json_path)

        self.assertTrue(result['success'])
        self.assertEqual(result['imported'], 2)

    def test_validate_import(self):
        """Test import validation."""
        # Valid data
        valid_data = [
            {
                'filename': 'test.xlsx',
                'model': 'MODEL',
                'sigma_gradient': 0.02
            }
        ]

        errors = self.migrator.validate_import(valid_data)
        self.assertEqual(len(errors), 0)

        # Invalid data
        invalid_data = [
            {
                'filename': 'test.xlsx',
                # Missing model
                'sigma_gradient': 'not_a_number'
            }
        ]

        errors = self.migrator.validate_import(invalid_data)
        self.assertGreater(len(errors), 0)

    def test_export_for_backup(self):
        """Test backup export."""
        # Add some data
        run_id = self.db.create_analysis_run("/test", {})
        for i in range(5):
            self.db.save_file_result(run_id, {
                'filename': f'backup_test_{i}.xlsx',
                'model': 'BACKUP_MODEL',
                'sigma_gradient': 0.02
            })

        # Export backup
        backup_path = self.migrator.export_for_backup(Path(self.test_dir))

        self.assertTrue(backup_path.exists())

        # Verify backup content
        df = pd.read_excel(backup_path, sheet_name='Analysis Results')
        self.assertEqual(len(df), 5)


if __name__ == '__main__':
    unittest.main()