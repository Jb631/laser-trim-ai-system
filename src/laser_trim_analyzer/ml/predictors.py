"""
ML Predictor Integration for Laser Trim Analyzer v2.

This module provides a clean interface between the ML components and the main processor,
handling model loading, caching, and real-time predictions.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass

from laser_trim_analyzer.core.config import Config
from laser_trim_analyzer.core.models import (
    AnalysisResult, TrackData, RiskCategory, AnalysisStatus
)
from laser_trim_analyzer.ml.engine import MLEngine, ModelConfig
from laser_trim_analyzer.ml.models import ModelFactory
from laser_trim_analyzer.database.models import MLPrediction as DBMLPrediction

# Import the existing ML predictor implementation
from laser_trim_analyzer.ml.ml_predictor_class import MLPredictor as MLPredictorImpl


@dataclass
class PredictionResult:
    """Structured prediction result."""
    failure_probability: float
    risk_category: str
    is_anomaly: bool
    anomaly_score: float
    suggested_threshold: Optional[float]
    confidence_score: float
    warnings: List[Dict[str, Any]]
    recommendations: List[str]
    feature_importance: Optional[Dict[str, float]] = None


class MLPredictor:
    """
    ML Predictor for real-time analysis integration.

    This class wraps the existing ML predictor implementation and provides
    a clean interface for the processor while managing models through MLEngine.
    """

    def __init__(
            self,
            config: Config,
            ml_engine: Optional[MLEngine] = None,
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ML Predictor.

        Args:
            config: Application configuration
            ml_engine: ML Engine instance for model management
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Initialize ML Engine if not provided
        if ml_engine is None:
            ml_engine = MLEngine(
                data_path=str(config.data_directory),
                models_path=str(config.ml.model_path),
                logger=self.logger
            )
        self.ml_engine = ml_engine

        # Initialize the implementation predictor
        self._impl_predictor = None
        self._models_loaded = False

        # Performance tracking
        self.prediction_count = 0
        self.total_prediction_time = 0.0

        # Feature names mapping
        self.feature_mapping = {
            'sigma_gradient': 'Sigma Gradient',
            'sigma_threshold': 'Sigma Threshold',
            'unit_length': 'Unit Length',
            'travel_length': 'Travel Length',
            'linearity_spec': 'Linearity Spec',
            'resistance_change_percent': 'Resistance Change (%)',
            'trim_improvement_percent': 'Trim Improvement (%)',
            'final_linearity_error_shifted': 'Final Linearity Error (Shifted)',
            'failure_probability': 'Failure Probability',
            'range_utilization_percent': 'Range Utilization (%)'
        }

    def initialize(self) -> bool:
        """
        Initialize ML models and predictor.

        Returns:
            True if initialization successful
        """
        try:
            # Create model directory if it doesn't exist
            model_dir = self.config.ml.model_path
            model_dir.mkdir(parents=True, exist_ok=True)

            # Initialize implementation predictor
            self._impl_predictor = MLPredictorImpl(
                model_dir=str(model_dir),
                db_manager=None,  # We'll use our own DB manager
                cache_size=1000,
                update_interval_hours=24,
                logger=self.logger
            )

            # Load or register models with MLEngine
            self._register_models()

            # Check if models need training
            if self._check_models_need_training():
                self.logger.info("ML models need training. Will use default predictions.")
                # Don't fail initialization, just mark models as not loaded
                self._models_loaded = False
            else:
                self._models_loaded = True

            self.logger.info("ML Predictor initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize ML Predictor: {e}")
            return False

    def _register_models(self):
        """Register ML models with the engine."""
        from laser_trim_analyzer.ml.models import ThresholdOptimizer, FailurePredictor, DriftDetector
        
        # Create default model configs
        failure_config = ModelConfig({
            'model_type': 'failure_predictor',
            'version': '1.0',
            'features': ['sigma_gradient', 'sigma_threshold', 'linearity_pass',
                      'resistance_change_percent', 'trim_improvement_percent',
                      'failure_probability', 'worst_zone'],
            'hyperparameters': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5
            }
        })
        
        threshold_config = ModelConfig({
            'model_type': 'threshold_optimizer',
            'version': '1.0',
            'features': ['sigma_gradient', 'linearity_spec', 'travel_length',
                      'unit_length', 'resistance_change_percent'],
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        })
        
        drift_config = ModelConfig({
            'model_type': 'drift_detector',
            'version': '1.0',
            'features': ['sigma_gradient', 'linearity_spec', 'resistance_change_percent',
                      'travel_length', 'unit_length'],
            'hyperparameters': {
                'n_estimators': 100,
                'contamination': 0.05,
                'max_samples': 'auto'
            }
        })

        # Register failure predictor
        self.ml_engine.register_model(
            'failure_predictor',
            FailurePredictor,
            failure_config
        )

        # Register threshold optimizer
        self.ml_engine.register_model(
            'threshold_optimizer',
            ThresholdOptimizer,
            threshold_config
        )

        # Register drift detector
        self.ml_engine.register_model(
            'drift_detector',
            DriftDetector,
            drift_config
        )

    def _check_models_need_training(self) -> bool:
        """Check if models need initial training."""
        model_files = ['failure_predictor.pkl', 'anomaly_detector.pkl', 'threshold_optimizer.pkl']
        model_dir = self.config.ml.model_path

        for model_file in model_files:
            if not (model_dir / model_file).exists():
                return True
        return False

    async def predict(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """
        Make predictions for an analysis result.

        Args:
            analysis_result: Complete analysis result

        Returns:
            Dictionary with predictions for all tracks
        """
        if not self._models_loaded:
            # Return empty predictions instead of error
            return {}

        predictions = {}

        # Process each track
        for track_id, track_data in analysis_result.tracks.items():
            try:
                track_predictions = await self._predict_for_track(
                    track_data,
                    analysis_result.metadata.model,
                    analysis_result.metadata.serial,
                    track_id
                )
                predictions[track_id] = track_predictions

            except Exception as e:
                self.logger.error(f"Prediction failed for track {track_id}: {e}")
                predictions[track_id] = self._get_default_predictions()

        # Add file-level predictions
        predictions['overall'] = self._aggregate_predictions(predictions)

        return predictions

    async def _predict_for_track(
            self,
            track_data: TrackData,
            model: str,
            serial: str,
            track_id: str
    ) -> PredictionResult:
        """Make predictions for a single track."""
        import time
        start_time = time.time()

        # Prepare data for predictor
        analysis_data = self._prepare_track_data(track_data, model, serial, track_id)

        # Get predictions from implementation
        ml_predictions = self._impl_predictor.predict_real_time(analysis_data)

        # Parse predictions into structured result
        pred_data = ml_predictions.get('predictions', {})

        result = PredictionResult(
            failure_probability=pred_data.get('failure_probability', 0.0),
            risk_category=self._determine_risk_category(
                pred_data.get('failure_probability', 0.0)
            ),
            is_anomaly=pred_data.get('is_anomaly', False),
            anomaly_score=pred_data.get('anomaly_score', 0.0),
            suggested_threshold=pred_data.get('suggested_threshold'),
            confidence_score=pred_data.get('quality_score', 0.0) / 100.0,
            warnings=ml_predictions.get('warnings', []),
            recommendations=ml_predictions.get('recommendations', [])
        )

        # Track performance
        self.prediction_count += 1
        self.total_prediction_time += (time.time() - start_time)

        return result

    def _prepare_track_data(
            self,
            track_data: TrackData,
            model: str,
            serial: str,
            track_id: str
    ) -> Dict[str, Any]:
        """Prepare track data for ML predictor."""
        data = {
            'File': f"{model}_{serial}.xlsx",
            'Model': model,
            'Serial': serial,
            'Track': track_id,
            'Sigma Gradient': track_data.sigma_analysis.sigma_gradient,
            'Sigma Threshold': track_data.sigma_analysis.sigma_threshold,
            'Sigma Pass': track_data.sigma_analysis.sigma_pass,
            'Unit Length': track_data.unit_properties.unit_length,
            'Travel Length': track_data.travel_length,
            'Linearity Spec': track_data.linearity_analysis.linearity_spec,
            'Linearity Pass': track_data.linearity_analysis.linearity_pass,
            'Resistance Change (%)': track_data.resistance_analysis.resistance_change_percent,
            'Risk Category': track_data.failure_prediction.risk_category.value if track_data.failure_prediction else 'Unknown'
        }

        # Add optional fields
        if track_data.trim_effectiveness:
            data['Trim Improvement (%)'] = track_data.trim_effectiveness.improvement_percent

        if track_data.linearity_analysis:
            data['Final Linearity Error (Shifted)'] = track_data.linearity_analysis.final_linearity_error_shifted

        if track_data.failure_prediction:
            data['Failure Probability'] = track_data.failure_prediction.failure_probability

        if track_data.dynamic_range:
            data['Range Utilization (%)'] = track_data.dynamic_range.range_utilization_percent

        return data

    def _determine_risk_category(self, failure_probability: float) -> str:
        """Determine risk category from failure probability."""
        if failure_probability > self.config.analysis.high_risk_threshold:
            return RiskCategory.HIGH.value
        elif failure_probability > self.config.analysis.low_risk_threshold:
            return RiskCategory.MEDIUM.value
        else:
            return RiskCategory.LOW.value

    def _aggregate_predictions(
            self,
            track_predictions: Dict[str, PredictionResult]
    ) -> Dict[str, Any]:
        """Aggregate track predictions for overall assessment."""
        if not track_predictions:
            return {}

        # Filter out 'overall' key if it exists
        track_preds = {k: v for k, v in track_predictions.items() if k != 'overall'}

        if not track_preds:
            return {}

        # Calculate overall metrics
        failure_probs = [p.failure_probability for p in track_preds.values()]
        anomaly_flags = [p.is_anomaly for p in track_preds.values()]

        # Collect all warnings and recommendations
        all_warnings = []
        all_recommendations = []

        for pred in track_preds.values():
            all_warnings.extend(pred.warnings)
            all_recommendations.extend(pred.recommendations)

        # Remove duplicates
        unique_recommendations = list(set(all_recommendations))

        return {
            'max_failure_probability': max(failure_probs),
            'avg_failure_probability': np.mean(failure_probs),
            'any_anomaly': any(anomaly_flags),
            'anomaly_count': sum(anomaly_flags),
            'overall_risk': self._determine_risk_category(max(failure_probs)),
            'total_warnings': len(all_warnings),
            'unique_recommendations': len(unique_recommendations),
            'top_recommendations': unique_recommendations[:3]  # Top 3 recommendations
        }

    def _get_default_predictions(self) -> PredictionResult:
        """Get default predictions when ML fails."""
        return PredictionResult(
            failure_probability=0.0,
            risk_category=RiskCategory.UNKNOWN.value,
            is_anomaly=False,
            anomaly_score=0.0,
            suggested_threshold=None,
            confidence_score=0.0,
            warnings=[],
            recommendations=[]
        )

    def check_for_drift(self, recent_results: List[AnalysisResult]) -> Dict[str, Any]:
        """
        Check for manufacturing drift in recent results.

        Args:
            recent_results: List of recent analysis results

        Returns:
            Drift analysis results
        """
        if not self._models_loaded or not recent_results:
            return {}

        try:
            # Convert results to DataFrame
            data_for_drift = []
            for result in recent_results:
                for track_id, track in result.tracks.items():
                    data_for_drift.append({
                        'timestamp': result.metadata.file_date,
                        'model': result.metadata.model,
                        'sigma_gradient': track.sigma_analysis.sigma_gradient,
                        'linearity_spec': track.linearity_analysis.linearity_spec,
                        'resistance_change_percent': track.resistance_analysis.resistance_change_percent or 0,
                        'travel_length': track.travel_length,
                        'unit_length': track.unit_properties.unit_length or 0
                    })

            df = pd.DataFrame(data_for_drift)

            # Use MLEngine's drift detector if available
            if 'drift_detector' in self.ml_engine.models:
                drift_detector = self.ml_engine.models['drift_detector']
                drift_analysis = drift_detector.analyze_drift(df)
                return drift_analysis

            return {'drift_detected': False, 'message': 'Drift detector not available'}

        except Exception as e:
            self.logger.error(f"Drift detection failed: {e}")
            return {'error': str(e)}

    def suggest_threshold_adjustments(
            self,
            model: str,
            recent_results: List[AnalysisResult]
    ) -> Optional[Dict[str, float]]:
        """
        Suggest threshold adjustments for a model.

        Args:
            model: Model number
            recent_results: Recent analysis results for the model

        Returns:
            Suggested thresholds or None
        """
        if not self._models_loaded:
            return None

        try:
            # Get recommended thresholds from predictor
            if hasattr(self._impl_predictor, 'thresholds'):
                model_thresholds = self._impl_predictor.thresholds.get(model)
                if model_thresholds:
                    return {
                        'recommended': model_thresholds.get('recommended'),
                        'confidence': model_thresholds.get('confidence', 0.8),
                        'based_on_samples': model_thresholds.get('sample_count', 0)
                    }

            return None

        except Exception as e:
            self.logger.error(f"Threshold suggestion failed: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get ML predictor performance statistics."""
        stats = {
            'predictions_made': self.prediction_count,
            'average_prediction_time': (
                    self.total_prediction_time / max(1, self.prediction_count)
            ),
            'models_loaded': self._models_loaded
        }

        # Add implementation stats if available
        if self._impl_predictor:
            impl_stats = self._impl_predictor.get_performance_stats()
            stats.update({
                'cache_hit_rate': impl_stats.get('cache_hit_rate', 0),
                'anomaly_detection_rate': impl_stats.get('anomaly_detection_rate', 0),
                'model_versions': impl_stats.get('model_versions', {})
            })

        return stats

    def update_models(self) -> bool:
        """
        Check for and load updated models.

        Returns:
            True if models were updated
        """
        if self._impl_predictor:
            self._impl_predictor.check_for_model_updates()
            return True
        return False

    def format_ml_predictions_for_db(
            self,
            predictions: Dict[str, Any],
            analysis_id: int
    ) -> List[DBMLPrediction]:
        """
        Format ML predictions for database storage.

        Args:
            predictions: ML predictions dictionary
            analysis_id: Database ID of the analysis

        Returns:
            List of database prediction objects
        """
        db_predictions = []

        for track_id, pred in predictions.items():
            if track_id == 'overall':
                continue

            if isinstance(pred, PredictionResult):
                db_pred = DBMLPrediction(
                    analysis_id=analysis_id,
                    prediction_type='comprehensive',
                    model_version=self._impl_predictor.model_versions.get(
                        'failure_predictor', '1.0'
                    ) if self._impl_predictor else '1.0',
                    predicted_failure_probability=pred.failure_probability,
                    predicted_risk_category=pred.risk_category,
                    confidence_score=pred.confidence_score,
                    recommendations=pred.recommendations,
                    drift_detected=pred.is_anomaly,
                    drift_percentage=abs(pred.anomaly_score) * 100 if pred.is_anomaly else None
                )

                # Add threshold recommendation if available
                if pred.suggested_threshold:
                    db_pred.recommended_threshold = pred.suggested_threshold

                db_predictions.append(db_pred)

        return db_predictions