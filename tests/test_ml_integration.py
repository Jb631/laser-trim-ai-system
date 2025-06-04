"""
Tests for ML integration in the laser trim analyzer.

Tests:
- ML model loading and initialization
- Real-time predictions during processing
- Threshold optimization
- Failure prediction
- Model training and updating
"""

import pytest
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import joblib

from laser_trim_analyzer.core.config import Config
from laser_trim_analyzer.core.models import RiskCategory
from laser_trim_analyzer.ml.engine import (
    MLEngine, ModelConfig, FeatureEngineering,
    ModelVersionControl
)
from laser_trim_analyzer.ml.models import (
    ThresholdOptimizer, FailurePredictor, DriftDetector,
    ModelFactory
)
from laser_trim_analyzer.ml.predictors import MLPredictor
from laser_trim_analyzer.database.manager import DatabaseManager


class TestMLModels:
    """Test individual ML model implementations."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        n_samples = 1000

        # Generate features
        data = pd.DataFrame({
            'sigma_gradient': np.random.normal(0.02, 0.01, n_samples),
            'sigma_threshold': np.random.normal(0.04, 0.005, n_samples),
            'unit_length': np.random.normal(300, 20, n_samples),
            'travel_length': np.random.normal(290, 15, n_samples),
            'linearity_spec': np.random.normal(0.05, 0.01, n_samples),
            'resistance_change_percent': np.random.normal(2, 1, n_samples),
            'trim_improvement_percent': np.random.normal(40, 10, n_samples),
            'final_linearity_error_shifted': np.random.normal(0.02, 0.005, n_samples),
            'failure_probability': np.random.random(n_samples),
            'range_utilization_percent': np.random.normal(80, 10, n_samples)
        })

        # Create target for threshold optimization (optimal thresholds)
        y_threshold = data['sigma_gradient'] * 1.5 + np.random.normal(0, 0.002, n_samples)

        # Create target for failure prediction (binary)
        y_failure = (data['failure_probability'] > 0.7).astype(int)

        return data, y_threshold, y_failure

    def test_threshold_optimizer(self, sample_training_data):
        """Test threshold optimizer model."""
        X, y_threshold, _ = sample_training_data

        # Create model
        model = ModelFactory.create_threshold_optimizer()

        # Train model
        metrics = model.train(X, y_threshold)

        # Verify training succeeded
        assert model.is_trained
        assert metrics['r2_score'] > 0.5  # Should explain some variance
        assert metrics['rmse'] < 0.01  # Reasonable error

        # Test prediction
        test_features = X.iloc[0:1]
        prediction = model.predict(test_features)

        assert len(prediction) == 1
        assert 0 < prediction[0] < 0.1  # Reasonable threshold range

        # Test recommendation with confidence
        recommendation = model.recommend_threshold(
            X.iloc[0].to_dict(),
            confidence_level=0.95
        )

        assert 'recommended_threshold' in recommendation
        assert 'confidence_interval' in recommendation
        assert recommendation['confidence_interval'][0] < recommendation['recommended_threshold']
        assert recommendation['confidence_interval'][1] > recommendation['recommended_threshold']

    def test_failure_predictor(self, sample_training_data):
        """Test failure predictor model."""
        X, _, y_failure = sample_training_data

        # Create model
        model = ModelFactory.create_failure_predictor()

        # Train model
        metrics = model.train(X, y_failure)

        # Verify training succeeded
        assert model.is_trained
        assert metrics['accuracy'] > 0.7
        assert metrics['f1_score'] > 0.5

        # Test prediction
        test_features = X.iloc[0:10]
        predictions = model.predict(test_features)
        probabilities = model.predict_proba(test_features)

        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= p <= 1 for p in probabilities)

        # Test risk assessment
        risk_assessment = model.get_risk_assessment(test_features)

        assert len(risk_assessment) == 10
        assert 'failure_probability' in risk_assessment.columns
        assert 'risk_category' in risk_assessment.columns

    def test_drift_detector(self, sample_training_data):
        """Test drift detector model."""
        X, _, _ = sample_training_data

        # Use first 80% as baseline
        baseline_data = X.iloc[:800]

        # Create model
        model = ModelFactory.create_drift_detector()

        # Train on baseline
        metrics = model.train(baseline_data)

        assert model.is_trained
        assert 'anomaly_score_mean' in metrics
        assert 'detected_anomalies' in metrics

        # Test on new data with drift
        drift_data = X.iloc[800:].copy()
        # Introduce drift
        drift_data['sigma_gradient'] += 0.01

        # Detect drift
        drift_indicators = model.predict(drift_data)

        assert len(drift_indicators) == len(drift_data)
        assert any(drift_indicators)  # Should detect some drift

        # Analyze drift patterns
        drift_analysis = model.analyze_drift(drift_data)

        assert 'overall_drift_rate' in drift_analysis
        assert 'drift_trend' in drift_analysis
        assert 'feature_drift' in drift_analysis

        # Get drift report
        report = model.get_drift_report(drift_data)

        assert 'summary' in report
        assert 'recommendations' in report
        assert report['summary']['drift_detected']  # Should detect drift


class TestMLEngine:
    """Test ML engine and infrastructure."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def ml_engine(self, temp_dir):
        """Create ML engine instance."""
        data_path = temp_dir / "data"
        models_path = temp_dir / "models"

        engine = MLEngine(str(data_path), str(models_path))
        return engine

    @pytest.fixture
    def training_data(self):
        """Create comprehensive training data."""
        n_samples = 500

        df = pd.DataFrame({
            'model': np.random.choice(['8340', '8555', '6845'], n_samples),
            'serial': [f'S{i:05d}' for i in range(n_samples)],
            'sigma_gradient': np.random.normal(0.02, 0.01, n_samples),
            'sigma_threshold': np.random.normal(0.04, 0.005, n_samples),
            'sigma_pass': np.random.random(n_samples) > 0.2,
            'linearity_pass': np.random.random(n_samples) > 0.15,
            'unit_length': np.random.normal(300, 20, n_samples),
            'resistance_change_percent': np.random.normal(2, 1, n_samples),
            'failure_probability': np.random.random(n_samples),
            'risk_category': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'timestamp': pd.date_range(end=datetime.now(), periods=n_samples, freq='H')
        })

        # Add computed target
        df['optimal_threshold'] = df['sigma_gradient'] * 1.8 + np.random.normal(0, 0.002, n_samples)

        return df

    def test_model_registration_and_training(self, ml_engine, training_data):
        """Test model registration and training."""
        # Register threshold optimizer
        config = ModelConfig({
            'model_type': 'threshold_optimizer',
            'version': '1.0.0',
            'features': ['sigma_gradient', 'unit_length', 'resistance_change_percent'],
            'target': 'optimal_threshold',
            'hyperparameters': {'n_estimators': 50, 'max_depth': 5}
        })

        ml_engine.register_model('threshold_opt_v1', ThresholdOptimizer, config)

        # Train model
        model = ml_engine.train_model(
            'threshold_opt_v1',
            ThresholdOptimizer,
            training_data,
            save=True
        )

        assert model.is_trained
        assert 'threshold_opt_v1' in ml_engine.models

        # Verify model was saved
        versions = ml_engine.version_control.list_versions('threshold_opt_v1')
        assert len(versions) > 0

    def test_feature_engineering(self, ml_engine, training_data):
        """Test feature engineering pipeline."""
        # Apply feature engineering
        engineered_data = ml_engine.feature_engineering.create_features(training_data)

        # Check new features were created
        assert len(engineered_data.columns) > len(training_data.columns)

        # Check specific engineered features
        assert 'sigma_ratio' in engineered_data.columns
        assert 'sigma_margin' in engineered_data.columns
        assert 'hour' in engineered_data.columns  # Time feature

        # Verify feature statistics
        assert len(ml_engine.feature_engineering.feature_stats) > 0

    def test_model_versioning(self, ml_engine, training_data):
        """Test model version control."""
        # Train multiple versions
        config = ModelConfig({
            'model_type': 'failure_predictor',
            'version': '1.0.0',
            'features': ['sigma_gradient', 'linearity_pass', 'failure_probability'],
            'target': 'risk_category'
        })

        ml_engine.register_model('failure_model', FailurePredictor, config)

        # Train v1
        model_v1 = ml_engine.train_model('failure_model', FailurePredictor, training_data)

        # Modify config and train v2
        config.hyperparameters = {'n_estimators': 200}
        config.version = '1.1.0'
        ml_engine.model_configs['failure_model'] = config

        model_v2 = ml_engine.train_model('failure_model', FailurePredictor, training_data)

        # Check versions
        versions = ml_engine.version_control.list_versions('failure_model')
        assert len(versions) >= 2

        # Load specific version
        loaded_model = ml_engine.version_control.load_model(
            FailurePredictor,
            'failure_model',
            '1.0.0'
        )
        assert loaded_model.is_trained

    def test_automated_retraining(self, ml_engine, training_data):
        """Test automated retraining pipeline."""
        # Register model with retraining criteria
        config = ModelConfig({
            'model_type': 'threshold_optimizer',
            'version': '1.0.0',
            'features': ['sigma_gradient', 'unit_length'],
            'target': 'optimal_threshold',
            'retraining_criteria': {
                'min_samples': 100,
                'max_days_since_training': 1,  # Force retraining
                'performance_threshold': 0.99  # High threshold to force retraining
            }
        })

        ml_engine.register_model('auto_retrain_model', ThresholdOptimizer, config)

        # Initial training
        ml_engine.train_model('auto_retrain_model', ThresholdOptimizer, training_data[:200])

        # Mock data source
        def data_source():
            return training_data

        # Run automated retraining
        results = ml_engine.automated_retraining_pipeline(data_source)

        assert 'auto_retrain_model' in results
        # Should retrain due to criteria
        assert results['auto_retrain_model']['status'] == 'retrained'

    def test_model_performance_tracking(self, ml_engine, training_data):
        """Test performance tracking over time."""
        config = ModelConfig({
            'model_type': 'threshold_optimizer',
            'version': '1.0.0',
            'features': ['sigma_gradient'],
            'target': 'optimal_threshold'
        })

        ml_engine.register_model('tracked_model', ThresholdOptimizer, config)

        # Train and evaluate multiple times
        for i in range(3):
            subset = training_data[i * 100:(i + 1) * 100]
            model = ml_engine.train_model('tracked_model', ThresholdOptimizer, subset, save=False)
            ml_engine.evaluate_model('tracked_model', subset)

        # Get performance history
        history = ml_engine.get_model_performance_history('tracked_model')

        assert len(history) >= 3
        assert 'timestamp' in history.columns
        assert 'type' in history.columns

        # Generate report
        report = ml_engine.generate_model_report('tracked_model')

        assert 'performance_metrics' in report
        assert 'performance_trend' in report


class TestMLPredictor:
    """Test real-time ML predictor."""

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory."""
        temp_dir = Path(tempfile.mkdtemp()) / "ml_models"
        temp_dir.mkdir(parents=True)
        yield temp_dir
        shutil.rmtree(temp_dir.parent)

    @pytest.fixture
    def mock_models(self, temp_model_dir):
        """Create mock pre-trained models."""
        # Create simple mock models
        from sklearn.ensemble import RandomForestClassifier, IsolationForest
        from sklearn.preprocessing import StandardScaler

        # Failure predictor
        failure_model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_mock = np.random.rand(100, 10)
        y_mock = np.random.randint(0, 2, 100)
        failure_model.fit(X_mock, y_mock)

        scaler = StandardScaler()
        scaler.fit(X_mock)

        model_data = {
            'model': failure_model,
            'scaler': scaler,
            'feature_stats': {},
            'version': '1.0',
            'trained_date': datetime.now().isoformat()
        }

        with open(temp_model_dir / 'failure_predictor.pkl', 'wb') as f:
            joblib.dump(model_data, f)

        # Anomaly detector
        anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        anomaly_model.fit(X_mock)

        anomaly_data = {
            'model': anomaly_model,
            'scaler': scaler,
            'feature_stats': {},
            'version': '1.0',
            'trained_date': datetime.now().isoformat()
        }

        with open(temp_model_dir / 'anomaly_detector.pkl', 'wb') as f:
            joblib.dump(anomaly_data, f)

        # Feature statistics
        feature_stats = {
            'Sigma Gradient': {'mean': 0.02, 'std': 0.01},
            'Unit Length': {'mean': 300, 'std': 20}
        }

        import json
        with open(temp_model_dir / 'feature_stats.json', 'w') as f:
            json.dump(feature_stats, f)

    @pytest.fixture
    def predictor(self, temp_model_dir, mock_models):
        """Create ML predictor instance."""
        predictor = MLPredictor(
            model_dir=str(temp_model_dir),
            cache_size=100
        )
        return predictor

    def test_real_time_prediction(self, predictor):
        """Test real-time prediction during analysis."""
        # Sample analysis data
        analysis_data = {
            'File': 'test_8340_A12345.xlsx',
            'Model': '8340',
            'Serial': 'A12345',
            'Sigma Gradient': 0.025,
            'Sigma Threshold': 0.04,
            'Unit Length': 300,
            'Travel Length': 295,
            'Linearity Spec': 0.05,
            'Resistance Change (%)': 2.5,
            'Trim Improvement (%)': 45,
            'Final Linearity Error (Shifted)': 0.02,
            'Failure Probability': 0.3,
            'Range Utilization (%)': 85
        }

        # Get predictions
        predictions = predictor.predict_real_time(analysis_data)

        # Verify predictions structure
        assert 'timestamp' in predictions
        assert 'predictions' in predictions
        assert 'recommendations' in predictions
        assert 'warnings' in predictions

        # Check specific predictions
        assert 'failure_probability' in predictions['predictions']
        assert 'is_anomaly' in predictions['predictions']
        assert isinstance(predictions['predictions']['failure_probability'], float)

    def test_prediction_caching(self, predictor):
        """Test that predictions are cached."""
        analysis_data = {
            'File': 'test.xlsx',
            'Model': '8340',
            'Sigma Gradient': 0.025,
            'Sigma Threshold': 0.04,
            'Unit Length': 300
        }

        # First prediction
        pred1 = predictor.predict_real_time(analysis_data)
        assert not pred1.get('cached', False)

        # Second prediction (should be cached)
        pred2 = predictor.predict_real_time(analysis_data)
        # Cache is internal, but timing should be faster
        assert pred2['prediction_time_ms'] < pred1['prediction_time_ms']

    def test_anomaly_detection(self, predictor):
        """Test anomaly detection in predictions."""
        # Create anomalous data
        anomalous_data = {
            'File': 'anomaly.xlsx',
            'Model': '8340',
            'Sigma Gradient': 0.1,  # Very high
            'Sigma Threshold': 0.04,
            'Unit Length': 600,  # Unusual
            'Failure Probability': 0.9  # High risk
        }

        predictions = predictor.predict_real_time(anomalous_data)

        # Should detect anomaly
        assert predictions['predictions'].get('is_anomaly', False)

        # Should have warnings
        assert len(predictions['warnings']) > 0
        assert any('anomaly' in w['type'].lower() for w in predictions['warnings'])

    def test_threshold_suggestions(self, predictor):
        """Test threshold optimization suggestions."""
        # Add mock threshold data
        predictor.thresholds = {
            '8340': {
                'recommended': 0.035,
                'confidence': 0.9,
                'sample_count': 1000
            }
        }

        analysis_data = {
            'File': 'test.xlsx',
            'Model': '8340',
            'Sigma Gradient': 0.03,
            'Sigma Threshold': 0.05,  # Higher than recommended
            'Unit Length': 300
        }

        predictions = predictor.predict_real_time(analysis_data)

        # Should suggest threshold adjustment
        assert 'suggested_threshold' in predictions['predictions']
        assert predictions['predictions']['suggested_threshold'] < 0.05

        # Should have recommendation
        assert any('threshold' in r.lower() for r in predictions['recommendations'])

    def test_performance_statistics(self, predictor):
        """Test performance tracking."""
        # Make several predictions
        for i in range(5):
            data = {
                'File': f'test_{i}.xlsx',
                'Model': '8340',
                'Sigma Gradient': 0.02 + i * 0.001,
                'Sigma Threshold': 0.04,
                'Unit Length': 300
            }
            predictor.predict_real_time(data)

        # Get performance stats
        stats = predictor.get_performance_stats()

        assert stats['average_prediction_time_ms'] > 0
        assert stats['cache_hit_rate'] > 0  # Some should be cached
        assert 'models_loaded' in stats
        assert len(stats['models_loaded']) > 0


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.fixture
    def full_system(self, temp_dir):
        """Create full integrated system."""
        # Configuration
        config = Config(
            database=Config.DatabaseConfig(
                enabled=True,
                path=temp_dir / "test.db"
            ),
            ml=Config.MLConfig(
                enabled=True,
                model_path=temp_dir / "models"
            )
        )

        # Database
        db_manager = DatabaseManager(str(config.database.path))
        db_manager.init_db()

        # ML Engine
        ml_engine = MLEngine(
            str(temp_dir / "data"),
            str(config.ml.model_path)
        )

        # ML Predictor
        ml_predictor = None  # Would be initialized with trained models

        # Processor
        from laser_trim_analyzer.core.processor import LaserTrimProcessor
        processor = LaserTrimProcessor(
            config=config,
            db_manager=db_manager,
            ml_predictor=ml_predictor
        )

        return {
            'config': config,
            'processor': processor,
            'db_manager': db_manager,
            'ml_engine': ml_engine
        }

    def test_end_to_end_workflow(self, full_system, temp_dir):
        """Test complete workflow from file to ML predictions."""
        # Create test file
        file_path = temp_dir / "test_8340_A12345.xlsx"
        self._create_test_file(file_path)

        # Process file
        processor = full_system['processor']
        result = asyncio.run(processor.process_file(file_path, temp_dir))

        # Verify processing succeeded
        assert result is not None
        assert result.overall_status in [AnalysisStatus.PASS, AnalysisStatus.FAIL, AnalysisStatus.WARNING]

        # Verify database storage
        assert result.db_id is not None

        # Query from database
        db_manager = full_system['db_manager']
        historical = db_manager.get_historical_data(model='8340', days_back=1)
        assert len(historical) > 0

        # Train ML models with historical data
        if len(historical) > 10:  # Need enough data
            ml_engine = full_system['ml_engine']

            # Convert to training data
            training_data = self._convert_to_training_data(historical)

            # Register and train model
            config = ModelConfig(
                model_type='threshold_optimizer',
                features=['sigma_gradient', 'unit_length'],
                target='sigma_threshold'
            )
            ml_engine.register_model('test_model', ThresholdOptimizer, config)

            model = ml_engine.train_model('test_model', ThresholdOptimizer, training_data)
            assert model.is_trained

    def _create_test_file(self, file_path):
        """Helper to create test Excel file."""
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Simple test data
            df = pd.DataFrame({
                'G': np.random.normal(0, 0.01, 100),  # Error
                'H': np.linspace(0, 100, 100),  # Position
                'I': [0.05] * 100,  # Upper limit
                'J': [-0.05] * 100,  # Lower limit
            })
            df.to_excel(writer, sheet_name='SEC1 TRK1 0', index=False)

    def _convert_to_training_data(self, db_results):
        """Convert database results to training DataFrame."""
        data = []
        for result in db_results:
            for track in result.tracks:
                data.append({
                    'model': result.model,
                    'sigma_gradient': track.sigma_gradient,
                    'sigma_threshold': track.sigma_threshold,
                    'unit_length': track.unit_length or 300,
                    'sigma_pass': track.sigma_pass
                })
        return pd.DataFrame(data)