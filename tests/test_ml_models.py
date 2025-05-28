"""
Unit Tests for Machine Learning Models
=====================================

Comprehensive tests for the ML models including:
- Threshold optimization
- Failure prediction
- Drift detection
- Model persistence
- Feature importance

Author: QA Specialist
Date: 2024
"""

import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# Import modules to test
from config import Config
from ml_models import LaserTrimMLModels, create_ml_models, train_all_models


class TestMLModels(unittest.TestCase):
    """Test suite for ML models."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Create test config
        self.config = Config()
        self.config.output_dir = self.test_dir

        # Create ML models instance
        self.ml_models = LaserTrimMLModels(self.config)

        # Create synthetic test data
        self.test_data = self._create_synthetic_data(1000)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def _create_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """Create synthetic historical data for testing."""
        np.random.seed(42)

        # Generate features
        data = {
            'sigma_gradient': np.random.normal(0.5, 0.1, n_samples),
            'linearity_spec': np.random.normal(0.02, 0.005, n_samples),
            'travel_length': np.random.normal(150, 20, n_samples),
            'unit_length': np.random.normal(140, 15, n_samples),
            'resistance_change': np.random.normal(5, 2, n_samples),
            'resistance_change_percent': np.random.normal(2, 0.5, n_samples),
            'sigma_threshold': np.random.normal(0.6, 0.1, n_samples),
            'timestamp': [datetime.now() - timedelta(days=x) for x in range(n_samples)],
            'model': np.random.choice(['8340', '8555', '6845'], n_samples),
            'passed': np.random.choice([True, False], n_samples, p=[0.85, 0.15])
        }

        # Add error data
        data['error_data'] = [
            np.random.normal(0, 0.01, 100).tolist() for _ in range(n_samples)
        ]

        # Add zone analysis
        data['zone_analysis'] = [
            {'worst_zone': np.random.randint(1, 6), 'variance': np.random.random()}
            for _ in range(n_samples)
        ]

        return pd.DataFrame(data)

    def test_initialization(self):
        """Test ML models initialization."""
        self.assertIsNotNone(self.ml_models)
        self.assertEqual(len(self.ml_models.models), 4)
        self.assertEqual(len(self.ml_models.scalers), 3)
        self.assertTrue(self.ml_models.models_dir.exists())

    def test_feature_preparation(self):
        """Test feature engineering."""
        features = self.ml_models.prepare_features(self.test_data)

        # Check basic features
        self.assertIn('sigma_gradient', features.columns)
        self.assertIn('linearity_spec', features.columns)
        self.assertIn('travel_length', features.columns)

        # Check calculated features
        self.assertIn('sigma_to_spec_ratio', features.columns)
        self.assertIn('length_ratio', features.columns)

        # Check error statistics
        self.assertIn('error_mean', features.columns)
        self.assertIn('error_std', features.columns)
        self.assertIn('error_max', features.columns)

        # Check time features
        self.assertIn('hour_of_day', features.columns)
        self.assertIn('day_of_week', features.columns)
        self.assertIn('is_weekend', features.columns)

        # Check model encoding
        self.assertTrue(any('model_' in col for col in features.columns))

        # Check no missing values
        self.assertFalse(features.isnull().any().any())

    def test_threshold_optimizer_training(self):
        """Test threshold optimizer training."""
        results = self.ml_models.train_threshold_optimizer(self.test_data)

        self.assertIsNotNone(self.ml_models.models['threshold_optimizer'])
        self.assertIn('mae', results)
        self.assertIn('r2_score', results)
        self.assertIn('feature_importance', results)
        self.assertIn('best_params', results)

        # Check performance metrics
        self.assertLess(results['mae'], 0.2)  # MAE should be reasonable
        self.assertGreater(results['r2_score'], 0.5)  # R² should be decent

        # Check feature importance
        self.assertGreater(len(results['feature_importance']), 0)

        # Test retraining protection
        results2 = self.ml_models.train_threshold_optimizer(self.test_data)
        self.assertNotIn('error', results2)

    def test_failure_predictor_training(self):
        """Test failure predictor training."""
        results = self.ml_models.train_failure_predictor(self.test_data)

        if 'error' not in results:
            self.assertIsNotNone(self.ml_models.models['failure_predictor'])
            self.assertIn('accuracy', results)
            self.assertIn('precision', results)
            self.assertIn('recall', results)
            self.assertIn('confusion_matrix', results)
            self.assertIn('feature_importance', results)

            # Check if target accuracy is reported
            self.assertIn('target_accuracy_met', results)

            # Check selected features
            self.assertIn('selected_features', results)
            self.assertGreater(len(results['selected_features']), 0)

    def test_drift_detector_training(self):
        """Test drift detector training."""
        results = self.ml_models.train_drift_detector(self.test_data)

        self.assertIsNotNone(self.ml_models.models['drift_detector'])
        self.assertIn('n_anomalies', results)
        self.assertIn('anomaly_rate', results)
        self.assertIn('feature_drift_scores', results)

        # Check anomaly rate is reasonable
        self.assertLess(results['anomaly_rate'], 0.2)
        self.assertGreater(results['anomaly_rate'], 0.0)

    def test_threshold_prediction(self):
        """Test threshold prediction."""
        # Train model first
        self.ml_models.train_threshold_optimizer(self.test_data)

        # Test prediction
        test_features = self.test_data.iloc[0].to_dict()
        prediction = self.ml_models.predict_optimal_threshold(test_features)

        self.assertIn('optimal_threshold', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('feature_contributions', prediction)

        # Check prediction is reasonable
        self.assertGreater(prediction['optimal_threshold'], 0)
        self.assertLess(prediction['optimal_threshold'], 2.0)
        self.assertGreater(prediction['confidence'], 0.5)

    def test_failure_prediction(self):
        """Test failure prediction."""
        # Train model first
        train_results = self.ml_models.train_failure_predictor(self.test_data)

        if 'error' not in train_results:
            # Test prediction
            test_features = self.test_data.iloc[0].to_dict()
            prediction = self.ml_models.predict_failure_probability(test_features)

            self.assertIn('failure_probability', prediction)
            self.assertIn('failure_prediction', prediction)
            self.assertIn('risk_level', prediction)
            self.assertIn('risk_factors', prediction)

            # Check probability is valid
            self.assertGreaterEqual(prediction['failure_probability'], 0)
            self.assertLessEqual(prediction['failure_probability'], 1)

            # Check risk level
            self.assertIn(prediction['risk_level'],
                          ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])

    def test_drift_detection(self):
        """Test drift detection."""
        # Train model first
        self.ml_models.train_drift_detector(self.test_data)

        # Test normal case
        normal_features = self.test_data.iloc[0].to_dict()
        detection = self.ml_models.detect_manufacturing_drift(normal_features)

        self.assertIn('is_drift', detection)
        self.assertIn('anomaly_score', detection)
        self.assertIn('severity', detection)
        self.assertIn('drift_indicators', detection)
        self.assertIn('recommendation', detection)

        # Test anomaly case (extreme values)
        anomaly_features = normal_features.copy()
        anomaly_features['sigma_gradient'] = 2.0  # Very high
        anomaly_features['resistance_change'] = 50  # Very high

        anomaly_detection = self.ml_models.detect_manufacturing_drift(anomaly_features)

        # Should detect as anomaly
        self.assertTrue(anomaly_detection['anomaly_score'] < detection['anomaly_score'])

    def test_model_persistence(self):
        """Test model saving and loading."""
        # Train models
        self.ml_models.train_threshold_optimizer(self.test_data)
        self.ml_models.train_failure_predictor(self.test_data)
        self.ml_models.train_drift_detector(self.test_data)

        # Save models
        version = self.ml_models.save_models('test_version')
        self.assertIsNotNone(version)

        # Check files exist
        save_dir = Path(version)
        self.assertTrue(save_dir.exists())
        self.assertTrue((save_dir / 'threshold_optimizer.joblib').exists())
        self.assertTrue((save_dir / 'scalers.joblib').exists())
        self.assertTrue((save_dir / 'feature_importance.json').exists())
        self.assertTrue((save_dir / 'metadata.json').exists())

        # Create new instance and load
        new_ml_models = LaserTrimMLModels(self.config)
        success = new_ml_models.load_models('test_version')

        self.assertTrue(success)
        self.assertIsNotNone(new_ml_models.models['threshold_optimizer'])
        self.assertIsNotNone(new_ml_models.models['drift_detector'])

        # Test loaded model works
        test_features = self.test_data.iloc[0].to_dict()
        prediction = new_ml_models.predict_optimal_threshold(test_features)
        self.assertIn('optimal_threshold', prediction)

    def test_feature_importance_report(self):
        """Test feature importance reporting."""
        # Train models
        self.ml_models.train_threshold_optimizer(self.test_data)
        self.ml_models.train_failure_predictor(self.test_data)

        # Get report
        report = self.ml_models.get_feature_importance_report()

        self.assertIn('threshold_optimizer', report)
        self.assertIn('failure_predictor', report)
        self.assertIn('summary', report)

        # Check threshold optimizer report
        if report['threshold_optimizer']:
            self.assertIn('top_features', report['threshold_optimizer'])
            self.assertGreater(len(report['threshold_optimizer']['top_features']), 0)

        # Check summary
        self.assertIn('most_important_overall', report['summary'])
        self.assertIn('feature_categories', report['summary'])

        # Check categories
        categories = report['summary']['feature_categories']
        self.assertIn('measurements', categories)
        self.assertIn('statistics', categories)
        self.assertIn('ratios', categories)

    def test_train_all_models(self):
        """Test training all models at once."""
        results = train_all_models(self.ml_models, self.test_data)

        self.assertIn('threshold_optimizer', results)
        self.assertIn('failure_predictor', results)
        self.assertIn('drift_detector', results)
        self.assertIn('saved_version', results)

        # Check all models are trained
        self.assertIsNotNone(self.ml_models.models['threshold_optimizer'])
        self.assertIsNotNone(self.ml_models.models['drift_detector'])

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty data
        empty_data = pd.DataFrame()
        results = self.ml_models.train_threshold_optimizer(empty_data)
        self.assertIn('error', results)

        # Test with insufficient data
        small_data = self.test_data.head(5)
        results = self.ml_models.train_threshold_optimizer(small_data)
        self.assertIn('error', results)

        # Test prediction without training
        new_models = LaserTrimMLModels(self.config)
        prediction = new_models.predict_optimal_threshold({'sigma_gradient': 0.5})
        self.assertIn('error', prediction)

        # Test with missing features
        incomplete_data = self.test_data.drop(columns=['sigma_gradient'])
        features = self.ml_models.prepare_features(incomplete_data)
        self.assertEqual(features['sigma_gradient'].iloc[0], 0)  # Should fill with 0

    def test_real_world_scenario(self):
        """Test realistic usage scenario."""
        # Simulate historical data with trends
        n_days = 90
        historical_data = []

        for day in range(n_days):
            # Simulate daily production with drift
            drift_factor = 1 + (day / n_days) * 0.1  # 10% drift over time

            for _ in range(10):  # 10 units per day
                unit_data = {
                    'sigma_gradient': np.random.normal(0.5 * drift_factor, 0.05),
                    'linearity_spec': np.random.normal(0.02, 0.002),
                    'travel_length': np.random.normal(150, 10),
                    'unit_length': np.random.normal(140, 10),
                    'resistance_change': np.random.normal(5 * drift_factor, 1),
                    'resistance_change_percent': np.random.normal(2 * drift_factor, 0.3),
                    'sigma_threshold': 0.6,
                    'timestamp': datetime.now() - timedelta(days=n_days - day),
                    'model': np.random.choice(['8340', '8555']),
                    'error_data': np.random.normal(0, 0.01, 100).tolist()
                }
                unit_data['passed'] = unit_data['sigma_gradient'] < unit_data['sigma_threshold']
                historical_data.append(unit_data)

        df = pd.DataFrame(historical_data)

        # Train models
        results = train_all_models(self.ml_models, df)

        # Should detect drift in recent production
        recent_unit = df.iloc[-1].to_dict()
        drift_detection = self.ml_models.detect_manufacturing_drift(recent_unit)

        # Recent units should show some drift
        self.assertIsNotNone(drift_detection['anomaly_score'])

        # Generate feature importance report
        importance_report = self.ml_models.get_feature_importance_report()
        self.assertGreater(len(importance_report['summary']['most_important_overall']), 0)


class TestMLIntegration(unittest.TestCase):
    """Test ML integration with data processor."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.output_dir = self.test_dir

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_factory_function(self):
        """Test ML models factory function."""
        ml_models = create_ml_models(self.config)
        self.assertIsInstance(ml_models, LaserTrimMLModels)
        self.assertEqual(ml_models.config, self.config)

    def test_model_versioning(self):
        """Test model versioning system."""
        ml_models = create_ml_models(self.config)

        # Create synthetic data
        data = pd.DataFrame({
            'sigma_gradient': np.random.normal(0.5, 0.1, 100),
            'linearity_spec': np.random.normal(0.02, 0.005, 100),
            'travel_length': np.random.normal(150, 20, 100),
            'sigma_threshold': np.random.normal(0.6, 0.1, 100),
            'passed': np.random.choice([True, False], 100)
        })

        # Train and save version 1
        ml_models.train_threshold_optimizer(data)
        version1 = ml_models.save_models('v1.0')

        # Modify data and train version 2
        data['sigma_gradient'] *= 1.1
        ml_models.train_threshold_optimizer(data, force_retrain=True)
        version2 = ml_models.save_models('v2.0')

        # Both versions should exist
        self.assertTrue(Path(version1).exists())
        self.assertTrue(Path(version2).exists())

        # Load version 1 and check it's different
        ml_models_v1 = create_ml_models(self.config)
        ml_models_v1.load_models('v1.0')

        # Predictions should be different
        test_features = {'sigma_gradient': 0.5, 'linearity_spec': 0.02}
        pred_v2 = ml_models.predict_optimal_threshold(test_features)
        pred_v1 = ml_models_v1.predict_optimal_threshold(test_features)

        # Should have different thresholds (due to different training data)
        self.assertNotEqual(
            pred_v1.get('optimal_threshold', 0),
            pred_v2.get('optimal_threshold', 0)
        )


if __name__ == '__main__':
    unittest.main()