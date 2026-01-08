"""
ML Manager for Laser Trim Analyzer v3.

Orchestrates all per-model ML components:
- ModelPredictor: Failure probability prediction
- ModelThresholdOptimizer: Optimal sigma threshold
- ModelDriftDetector: Quality drift detection
- ModelProfiler: Statistical profiling

Handles training, persistence, and application to database.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from laser_trim_analyzer.ml.predictor import (
    ModelPredictor, PredictorTrainingResult, extract_features, FEATURE_COLUMNS
)
from laser_trim_analyzer.ml.threshold_optimizer import (
    ModelThresholdOptimizer, ThresholdResult
)
from laser_trim_analyzer.ml.drift_detector import (
    ModelDriftDetector, DriftResult, DriftDirection
)
from laser_trim_analyzer.ml.profiler import (
    ModelProfiler, ModelProfile, calculate_cross_model_metrics
)
from laser_trim_analyzer.database.models import StatusType

logger = logging.getLogger(__name__)


@dataclass
class ModelTrainingResult:
    """Result of training all ML components for a single model."""
    model_name: str
    success: bool

    # Component results
    predictor_trained: bool = False
    predictor_accuracy: float = 0.0
    threshold_calculated: bool = False
    threshold_value: Optional[float] = None
    threshold_confidence: float = 0.0
    drift_baseline_set: bool = False
    profile_built: bool = False

    # Sample counts
    n_samples: int = 0
    n_pass: int = 0
    n_fail: int = 0

    error: Optional[str] = None


@dataclass
class TrainingProgress:
    """Progress update during training."""
    current_model: str
    models_complete: int
    models_total: int
    phase: str  # 'gathering', 'training', 'profiling'
    message: str


@dataclass
class ApplyProgress:
    """Progress update during database application."""
    records_complete: int
    records_total: int
    models_updated: int
    message: str


class MLManager:
    """
    Manages all per-model ML components.

    Responsibilities:
    - Train ML components from database data
    - Apply learned thresholds and predictions to database
    - Provide threshold/prediction lookups during analysis
    - Persist and load trained state

    Usage:
    1. After processing files: call train_all_models()
    2. To apply ML to existing data: call apply_to_database()
    3. During analysis: call get_threshold() or get_failure_probability()
    """

    # Minimum samples for different components
    MIN_PREDICTOR_SAMPLES = 50
    MIN_THRESHOLD_SAMPLES = 20
    MIN_DRIFT_SAMPLES = 30
    MIN_PROFILE_SAMPLES = 10

    def __init__(self, db_manager: Any, ml_storage_path: Optional[Path] = None):
        """
        Initialize ML Manager.

        Args:
            db_manager: DatabaseManager instance for data access
            ml_storage_path: Path for storing ML model files (pickles)
        """
        self.db = db_manager
        self.storage_path = ml_storage_path or Path("data/ml_models")

        # Per-model components (lazy loaded)
        self.predictors: Dict[str, ModelPredictor] = {}
        self.threshold_optimizers: Dict[str, ModelThresholdOptimizer] = {}
        self.drift_detectors: Dict[str, ModelDriftDetector] = {}
        self.profilers: Dict[str, ModelProfiler] = {}

        # State
        self.is_loaded: bool = False
        self.last_training_date: Optional[datetime] = None
        self.trained_models: List[str] = []
        self.models_needing_data: Dict[str, int] = {}  # model -> samples needed

    def get_predictor(self, model_name: str) -> ModelPredictor:
        """Get or create predictor for a model."""
        if model_name not in self.predictors:
            self.predictors[model_name] = ModelPredictor(model_name)
        return self.predictors[model_name]

    def get_threshold_optimizer(self, model_name: str) -> ModelThresholdOptimizer:
        """Get or create threshold optimizer for a model."""
        if model_name not in self.threshold_optimizers:
            self.threshold_optimizers[model_name] = ModelThresholdOptimizer(model_name)
        return self.threshold_optimizers[model_name]

    def get_drift_detector(self, model_name: str) -> ModelDriftDetector:
        """Get or create drift detector for a model."""
        if model_name not in self.drift_detectors:
            self.drift_detectors[model_name] = ModelDriftDetector(model_name)
        return self.drift_detectors[model_name]

    def get_profiler(self, model_name: str) -> ModelProfiler:
        """Get or create profiler for a model."""
        if model_name not in self.profilers:
            self.profilers[model_name] = ModelProfiler(model_name)
        return self.profilers[model_name]

    def get_threshold(self, model_name: str) -> Optional[float]:
        """
        Get learned threshold for a model.

        Args:
            model_name: Product model number

        Returns:
            Learned threshold, or None to use formula fallback
        """
        if model_name in self.threshold_optimizers:
            optimizer = self.threshold_optimizers[model_name]
            if optimizer.is_calculated:
                return optimizer.threshold
        return None

    def get_failure_probability(
        self,
        model_name: str,
        features: Dict[str, float]
    ) -> Optional[float]:
        """
        Get failure probability prediction.

        Args:
            model_name: Product model number
            features: Feature dict from extract_features()

        Returns:
            Failure probability (0-1), or None if not trained
        """
        if model_name in self.predictors:
            predictor = self.predictors[model_name]
            if predictor.is_trained:
                return predictor.predict_failure_probability(features)
        return None

    def train_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> ModelTrainingResult:
        """
        Train all ML components for a specific model.

        Args:
            model_name: Product model number
            progress_callback: Optional callback for progress updates

        Returns:
            ModelTrainingResult with training status
        """
        result = ModelTrainingResult(model_name=model_name, success=False)

        try:
            if progress_callback:
                progress_callback(f"Gathering data for {model_name}...")

            # Get training data from database
            data = self._get_training_data(model_name)

            if data is None or len(data) < self.MIN_PROFILE_SAMPLES:
                result.error = f"Insufficient data: {len(data) if data is not None else 0} samples"
                self.models_needing_data[model_name] = self.MIN_THRESHOLD_SAMPLES - (len(data) if data is not None else 0)
                return result

            result.n_samples = len(data)
            result.n_pass = int(data['passed'].sum())
            result.n_fail = result.n_samples - result.n_pass

            # 1. Build profile (always, needs least data)
            if progress_callback:
                progress_callback(f"Building profile for {model_name}...")

            profiler = self.get_profiler(model_name)
            profiler.build_profile(data)
            result.profile_built = profiler.is_profiled

            # 2. Calculate threshold (needs 20+ samples)
            if len(data) >= self.MIN_THRESHOLD_SAMPLES:
                if progress_callback:
                    progress_callback(f"Calculating threshold for {model_name}...")

                optimizer = self.get_threshold_optimizer(model_name)
                threshold_result = optimizer.calculate_threshold(
                    sigma_values=data['sigma_gradient'],
                    passed=data['passed'],
                    fail_points=data.get('linearity_fail_points'),
                    linearity_spec=data['linearity_spec'].iloc[0] if 'linearity_spec' in data.columns else None
                )
                result.threshold_calculated = optimizer.is_calculated
                result.threshold_value = threshold_result.threshold
                result.threshold_confidence = threshold_result.confidence

            # 3. Set drift baseline using 70/30 split (needs 30+ samples)
            if len(data) >= self.MIN_DRIFT_SAMPLES:
                if progress_callback:
                    progress_callback(f"Setting drift baseline for {model_name}...")

                detector = self.get_drift_detector(model_name)

                # Sort by file_date for temporal split
                data_with_sigma = data[data['sigma_gradient'].notna()].copy()
                if 'file_date' in data_with_sigma.columns:
                    data_with_sigma = data_with_sigma.sort_values('file_date')

                # Use oldest 70% for baseline, newest 30% for detection
                baseline_cutoff_idx = int(len(data_with_sigma) * 0.7)
                baseline_data = data_with_sigma.iloc[:baseline_cutoff_idx]
                detection_data = data_with_sigma.iloc[baseline_cutoff_idx:]

                if len(baseline_data) >= self.MIN_DRIFT_SAMPLES:
                    # Determine cutoff date (date of last baseline sample)
                    cutoff_date = None
                    if 'file_date' in baseline_data.columns and len(baseline_data) > 0:
                        last_baseline_date = baseline_data['file_date'].iloc[-1]
                        if pd.notna(last_baseline_date):
                            cutoff_date = last_baseline_date

                    sigma_values = baseline_data['sigma_gradient'].values
                    result.drift_baseline_set = detector.set_baseline(sigma_values, cutoff_date)

                    # Reset detector state before running on detection period
                    if result.drift_baseline_set:
                        detector.reset()

                        # Run drift detection on the newest 30%
                        for sigma in detection_data['sigma_gradient'].values:
                            detector.detect(sigma)

                        logger.info(
                            f"DriftDetector[{model_name}] using 70/30 split - "
                            f"Baseline: {len(baseline_data)} samples, "
                            f"Detection: {len(detection_data)} samples, "
                            f"Cutoff: {cutoff_date}"
                        )

            # 4. Train predictor (needs 50+ samples)
            if len(data) >= self.MIN_PREDICTOR_SAMPLES:
                if progress_callback:
                    progress_callback(f"Training predictor for {model_name}...")

                predictor = self.get_predictor(model_name)

                # Prepare features
                features = self._extract_features_from_data(data)
                labels = ~data['passed']  # 1 = failed
                severity = data.get('linearity_fail_points')

                training_result = predictor.train(features, labels, severity)
                result.predictor_trained = training_result.success
                if training_result.metrics:
                    result.predictor_accuracy = training_result.metrics.accuracy

            result.success = True

            # Track trained model
            if model_name not in self.trained_models:
                self.trained_models.append(model_name)

            threshold_str = f"{result.threshold_value:.6f}" if result.threshold_value else "N/A"
            logger.info(
                f"MLManager trained {model_name} - "
                f"Threshold: {threshold_str}, "
                f"Predictor: {'Yes' if result.predictor_trained else 'No'}, "
                f"Drift: {'Yes' if result.drift_baseline_set else 'No'}"
            )

            return result

        except Exception as e:
            logger.exception(f"Error training model {model_name}: {e}")
            result.error = str(e)
            return result

    def train_all_models(
        self,
        min_samples: int = 20,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None
    ) -> Dict[str, ModelTrainingResult]:
        """
        Train ML for all models with sufficient data.

        Args:
            min_samples: Minimum samples required for training
            progress_callback: Optional callback for progress updates

        Returns:
            Dict of model_name -> ModelTrainingResult
        """
        results = {}

        # Get list of models from database
        models = self._get_model_list()

        if not models:
            logger.warning("No models found in database")
            return results

        logger.info(f"MLManager training {len(models)} models...")

        for i, model_name in enumerate(models):
            if progress_callback:
                progress_callback(TrainingProgress(
                    current_model=model_name,
                    models_complete=i,
                    models_total=len(models),
                    phase='training',
                    message=f"Training {model_name} ({i+1}/{len(models)})"
                ))

            # Simple callback for single model
            def model_callback(msg: str):
                if progress_callback:
                    progress_callback(TrainingProgress(
                        current_model=model_name,
                        models_complete=i,
                        models_total=len(models),
                        phase='training',
                        message=msg
                    ))

            results[model_name] = self.train_model(model_name, model_callback)

        # Calculate cross-model metrics
        if progress_callback:
            progress_callback(TrainingProgress(
                current_model='',
                models_complete=len(models),
                models_total=len(models),
                phase='profiling',
                message='Calculating cross-model metrics...'
            ))

        cross_metrics = calculate_cross_model_metrics(self.profilers)
        for model_name, (difficulty, quality) in cross_metrics.items():
            if model_name in self.profilers:
                self.profilers[model_name].set_comparative_metrics(difficulty, quality)

        self.last_training_date = datetime.now()

        # Save state to database immediately after training
        # This ensures drift detector state is persisted before user views Trends page
        self._save_state_to_db()

        # Summary
        trained_count = sum(1 for r in results.values() if r.success)
        logger.info(
            f"MLManager training complete - "
            f"{trained_count}/{len(models)} models trained"
        )

        return results

    def apply_to_database(
        self,
        progress_callback: Optional[Callable[[ApplyProgress], None]] = None,
        run_drift_detection: bool = True
    ) -> Dict[str, Any]:
        """
        Apply learned ML to all database records.

        Updates:
        - sigma_threshold and sigma_pass using learned thresholds
        - Track and analysis status based on new sigma_pass values
        - Runs drift detection on detection-period samples only

        Optimized for performance:
        - Uses bulk SQL UPDATE for threshold/sigma_pass (fast)
        - Only loads tracks for status recalculation and drift detection
        - Commits in batches per model

        Args:
            progress_callback: Optional callback for progress updates
            run_drift_detection: Whether to run drift detection (default True)

        Returns:
            Dict with counts and drift alerts:
            {'updated': N, 'skipped': M, 'errors': E, 'drift_alerts': [...]}
        """
        counts = {'updated': 0, 'skipped': 0, 'errors': 0, 'drift_alerts': []}

        try:
            from sqlalchemy import update, case, and_
            from laser_trim_analyzer.database.models import (
                TrackResult, AnalysisResult, QAAlert, AlertType
            )

            models_updated = set()

            with self.db.session() as session:
                # Count total trained models for progress
                total_models = len(self.trained_models)

                if progress_callback:
                    progress_callback(ApplyProgress(
                        records_complete=0,
                        records_total=total_models,
                        models_updated=0,
                        message=f"Applying thresholds to {total_models} models..."
                    ))

                for model_idx, model_name in enumerate(self.trained_models):
                    optimizer = self.threshold_optimizers.get(model_name)
                    if not optimizer or not optimizer.is_calculated:
                        continue

                    new_threshold = optimizer.threshold
                    detector = self.drift_detectors.get(model_name)

                    # OPTIMIZATION 1: Bulk update sigma_threshold and sigma_pass using SQL
                    # Use subquery instead of loading IDs into Python (much faster for large datasets)
                    import time
                    model_start_time = time.time()
                    track_count = 0
                    analysis_updates = 0

                    try:
                        from sqlalchemy import select

                        # Use subquery instead of loading all IDs into Python
                        # SQLite can optimize this much better than a massive IN list
                        analysis_subquery = (
                            select(AnalysisResult.id)
                            .where(AnalysisResult.model == model_name)
                        )

                        # Check if there are any analyses for this model
                        analysis_count = session.query(AnalysisResult.id).filter(
                            AnalysisResult.model == model_name
                        ).count()

                        if analysis_count == 0:
                            continue

                        # Bulk update sigma_threshold for all tracks of this model
                        result1 = session.execute(
                            update(TrackResult)
                            .where(TrackResult.analysis_id.in_(analysis_subquery))
                            .values(sigma_threshold=new_threshold)
                        )
                        track_count = result1.rowcount  # Get count from UPDATE result

                        # Bulk update sigma_pass based on threshold comparison
                        # sigma_pass = True if sigma_gradient <= threshold, else False
                        session.execute(
                            update(TrackResult)
                            .where(
                                and_(
                                    TrackResult.analysis_id.in_(analysis_subquery),
                                    TrackResult.sigma_gradient.isnot(None)
                                )
                            )
                            .values(
                                sigma_pass=case(
                                    (TrackResult.sigma_gradient <= new_threshold, True),
                                    else_=False
                                )
                            )
                        )

                        # OPTIMIZATION 2: Bulk update track status based on sigma_pass and linearity_pass
                        # Status = PASS if both pass, FAIL if both fail, WARNING otherwise
                        # NOTE: Use .name (not .value) for SQLite - SQLAlchemy stores enum NAME not value
                        session.execute(
                            update(TrackResult)
                            .where(TrackResult.analysis_id.in_(analysis_subquery))
                            .values(
                                status=case(
                                    # Both pass -> PASS
                                    (and_(
                                        TrackResult.sigma_pass == True,
                                        TrackResult.linearity_pass == True
                                    ), StatusType.PASS.name),
                                    # Both fail -> FAIL
                                    (and_(
                                        TrackResult.sigma_pass == False,
                                        TrackResult.linearity_pass == False
                                    ), StatusType.FAIL.name),
                                    # Mixed -> WARNING
                                    else_=StatusType.WARNING.name
                                )
                            )
                        )

                        # Track count already captured from first UPDATE rowcount
                        counts['updated'] += track_count
                        models_updated.add(model_name)

                    except Exception as e:
                        logger.warning(f"Error in bulk update for {model_name}: {e}")
                        counts['errors'] += 1

                    # OPTIMIZATION 3: Update analysis overall_status in bulk using subqueries
                    # This is faster than iterating over each analysis
                    try:
                        from sqlalchemy import func, exists
                        from sqlalchemy.orm import aliased

                        # Update analyses that have any ERROR tracks -> ERROR
                        # NOTE: Use .name (not .value) - SQLAlchemy stores enum NAME
                        result = session.execute(
                            update(AnalysisResult)
                            .where(AnalysisResult.model == model_name)
                            .where(
                                exists(
                                    select(TrackResult.id)
                                    .where(TrackResult.analysis_id == AnalysisResult.id)
                                    .where(TrackResult.status == StatusType.ERROR.name)
                                )
                            )
                            .values(overall_status=StatusType.ERROR.name)
                        )
                        analysis_updates += result.rowcount

                        # Update analyses that have FAIL but no ERROR -> FAIL
                        result = session.execute(
                            update(AnalysisResult)
                            .where(AnalysisResult.model == model_name)
                            .where(~exists(
                                select(TrackResult.id)
                                .where(TrackResult.analysis_id == AnalysisResult.id)
                                .where(TrackResult.status == StatusType.ERROR.name)
                            ))
                            .where(exists(
                                select(TrackResult.id)
                                .where(TrackResult.analysis_id == AnalysisResult.id)
                                .where(TrackResult.status == StatusType.FAIL.name)
                            ))
                            .values(overall_status=StatusType.FAIL.name)
                        )
                        analysis_updates += result.rowcount

                        # Update analyses that have WARNING but no ERROR/FAIL -> WARNING
                        result = session.execute(
                            update(AnalysisResult)
                            .where(AnalysisResult.model == model_name)
                            .where(~exists(
                                select(TrackResult.id)
                                .where(TrackResult.analysis_id == AnalysisResult.id)
                                .where(TrackResult.status == StatusType.ERROR.name)
                            ))
                            .where(~exists(
                                select(TrackResult.id)
                                .where(TrackResult.analysis_id == AnalysisResult.id)
                                .where(TrackResult.status == StatusType.FAIL.name)
                            ))
                            .where(exists(
                                select(TrackResult.id)
                                .where(TrackResult.analysis_id == AnalysisResult.id)
                                .where(TrackResult.status == StatusType.WARNING.name)
                            ))
                            .values(overall_status=StatusType.WARNING.name)
                        )
                        analysis_updates += result.rowcount

                        # Update remaining analyses (all tracks PASS) -> PASS
                        result = session.execute(
                            update(AnalysisResult)
                            .where(AnalysisResult.model == model_name)
                            .where(~exists(
                                select(TrackResult.id)
                                .where(TrackResult.analysis_id == AnalysisResult.id)
                                .where(TrackResult.status.in_([
                                    StatusType.ERROR.name, StatusType.FAIL.name, StatusType.WARNING.name
                                ]))
                            ))
                            .values(overall_status=StatusType.PASS.name)
                        )
                        analysis_updates += result.rowcount

                    except Exception as e:
                        logger.warning(f"Error updating analysis status for {model_name}: {e}")

                    # OPTIMIZATION 4: Drift detection - only load detection-period tracks
                    if run_drift_detection and detector and detector.has_baseline:
                        try:
                            # Only get tracks AFTER baseline cutoff date
                            cutoff_date = detector.baseline_cutoff_date

                            if cutoff_date:
                                detection_tracks = (
                                    session.query(TrackResult.sigma_gradient, AnalysisResult.id)
                                    .join(AnalysisResult)
                                    .filter(AnalysisResult.model == model_name)
                                    .filter(AnalysisResult.file_date > cutoff_date)
                                    .filter(TrackResult.sigma_gradient.isnot(None))
                                    .order_by(AnalysisResult.file_date)
                                    .all()
                                )

                                # Reset detector before running on detection period
                                detector.reset()

                                # Run drift detection only on detection-period samples
                                for sigma, analysis_id in detection_tracks:
                                    drift_result = detector.detect(sigma)

                                    # Only create alert on first detection per model
                                    if drift_result.is_drifting and drift_result.message:
                                        # Check if we already logged this drift
                                        if model_name not in [a['model'] for a in counts['drift_alerts']]:
                                            counts['drift_alerts'].append({
                                                'model': model_name,
                                                'direction': drift_result.direction.value,
                                                'severity': drift_result.severity,
                                            })

                        except Exception as e:
                            logger.warning(f"Error in drift detection for {model_name}: {e}")

                    # Commit after each model to avoid holding locks too long
                    session.commit()

                    # Log completion with timing and counts for verification
                    model_elapsed = time.time() - model_start_time
                    logger.info(
                        f"Apply complete for {model_name}: "
                        f"{track_count} tracks, {analysis_updates} analyses updated "
                        f"in {model_elapsed:.1f}s"
                    )

                    if progress_callback:
                        progress_callback(ApplyProgress(
                            records_complete=model_idx + 1,
                            records_total=total_models,
                            models_updated=len(models_updated),
                            message=f"Completed {model_name}: {track_count} tracks, {analysis_updates} analyses ({model_elapsed:.1f}s)"
                        ))

                # Save updated drift state
                self._save_state_to_db()

        except Exception as e:
            logger.exception(f"Error applying ML to database: {e}")
            counts['errors'] += 1

        logger.info(
            f"MLManager apply complete - "
            f"Updated: {counts['updated']}, Skipped: {counts['skipped']}, "
            f"Errors: {counts['errors']}, Drift alerts: {len(counts['drift_alerts'])}"
        )

        return counts

    def get_drift_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current drift status for all models.

        Returns:
            Dict of model_name -> drift info
        """
        status = {}

        for model_name, detector in self.drift_detectors.items():
            if detector.has_baseline:
                lower, center, upper = detector.get_control_limits()

                # Compute direction fresh based on current EWMA vs baseline mean
                # This ensures direction reflects CURRENT state, not historical drift
                if detector.is_drifting and detector.ewma_value is not None and detector.baseline_mean is not None:
                    if detector.ewma_value > detector.baseline_mean:
                        direction = 'up'
                    elif detector.ewma_value < detector.baseline_mean:
                        direction = 'down'
                    else:
                        direction = None
                else:
                    direction = None

                status[model_name] = {
                    'has_baseline': True,
                    'is_drifting': detector.is_drifting,
                    'direction': direction,
                    'baseline_mean': detector.baseline_mean,
                    'baseline_std': detector.baseline_std,
                    'control_limits': {
                        'lower': lower,
                        'center': center,
                        'upper': upper,
                    },
                    'samples_since_baseline': detector.samples_since_baseline,
                    'drift_start_date': detector.drift_start_date.isoformat() if detector.drift_start_date else None,
                }
            else:
                status[model_name] = {'has_baseline': False}

        return status

    def _get_model_list(self) -> List[str]:
        """Get list of unique models from database."""
        try:
            from laser_trim_analyzer.database.models import AnalysisResult

            with self.db.session() as session:
                models = (
                    session.query(AnalysisResult.model)
                    .distinct()
                    .all()
                )
                return [m[0] for m in models if m[0]]
        except Exception as e:
            logger.error(f"Error getting model list: {e}")
            return []

    def _get_training_data(self, model_name: str) -> Optional[pd.DataFrame]:
        """
        Get training data for a specific model from database.

        Combines:
        - Trim file results (TrackResult)
        - Final Test results (FinalTestTrack) when linked

        Returns DataFrame with columns:
        - sigma_gradient, linearity_error, linearity_fail_points
        - linearity_pass, sigma_pass (outcomes)
        - linearity_spec, file_date
        - passed (True if linearity passed)
        - source ('trim' or 'final_test')
        """
        try:
            from laser_trim_analyzer.database.models import (
                AnalysisResult, TrackResult, FinalTestResult, FinalTestTrack
            )

            records = []

            with self.db.session() as session:
                # Get trim data
                trim_results = (
                    session.query(TrackResult, AnalysisResult.file_date)
                    .join(AnalysisResult)
                    .filter(AnalysisResult.model == model_name)
                    .all()
                )

                for track, file_date in trim_results:
                    records.append({
                        'sigma_gradient': track.sigma_gradient,
                        'linearity_error': track.final_linearity_error_shifted or 0,
                        'linearity_fail_points': track.linearity_fail_points or 0,
                        'optimal_offset': track.optimal_offset or 0,
                        'linearity_spec': track.linearity_spec or 0.01,
                        'linearity_pass': track.linearity_pass if track.linearity_pass is not None else True,
                        'sigma_pass': track.sigma_pass,
                        'passed': track.linearity_pass if track.linearity_pass is not None else True,
                        'file_date': file_date,
                        'source': 'trim',
                    })

                # Get Final Test data (higher priority when linked)
                final_results = (
                    session.query(FinalTestTrack, FinalTestResult.file_date)
                    .join(FinalTestResult)
                    .filter(FinalTestResult.model == model_name)
                    .all()
                )

                for track, file_date in final_results:
                    # Final Test data overwrites trim outcome for linked records
                    records.append({
                        'sigma_gradient': None,  # Final test doesn't have sigma
                        'linearity_error': track.linearity_error or 0,
                        'linearity_fail_points': track.linearity_fail_points or 0,
                        'optimal_offset': 0,
                        'linearity_spec': track.linearity_spec or 0.01,
                        'linearity_pass': track.linearity_pass if track.linearity_pass is not None else True,
                        'sigma_pass': None,
                        'passed': track.linearity_pass if track.linearity_pass is not None else True,
                        'file_date': file_date,
                        'source': 'final_test',
                    })

            if not records:
                return None

            df = pd.DataFrame(records)

            # Filter to records with sigma_gradient for predictor training
            df_with_sigma = df[df['sigma_gradient'].notna()].copy()

            return df_with_sigma if len(df_with_sigma) > 0 else df

        except Exception as e:
            logger.error(f"Error getting training data for {model_name}: {e}")
            return None

    def _extract_features_from_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract feature columns from training data."""
        features = pd.DataFrame()

        features['sigma_gradient'] = data.get('sigma_gradient', 0)
        features['linearity_error'] = data.get('linearity_error', 0).abs()
        features['fail_points'] = data.get('linearity_fail_points', 0)
        features['optimal_offset'] = data.get('optimal_offset', 0).abs()
        features['linearity_spec'] = data.get('linearity_spec', 0.01)

        # Derived features
        spec = features['linearity_spec'].replace(0, 0.01)
        features['sigma_to_spec'] = features['sigma_gradient'] / spec
        features['error_to_spec'] = features['linearity_error'] / spec

        return features.fillna(0)

    def save_all(self) -> bool:
        """
        Save all trained ML state to disk.

        Saves:
        - Predictor models (pickle files)
        - State to database (via model_ml_state table)

        Returns:
            True if successful
        """
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            predictors_path = self.storage_path / "predictors"
            predictors_path.mkdir(exist_ok=True)

            # Save predictors
            for model_name, predictor in self.predictors.items():
                if predictor.is_trained:
                    path = predictors_path / f"{model_name}.pkl"
                    predictor.save(path)

            # Save state to database
            self._save_state_to_db()

            logger.info(f"MLManager saved {len(self.predictors)} predictors")
            return True

        except Exception as e:
            logger.error(f"Error saving ML state: {e}")
            return False

    def load_all(self) -> bool:
        """
        Load all trained ML state from disk and database.

        Returns:
            True if successful
        """
        try:
            # Load state from database
            self._load_state_from_db()

            # Load predictor models
            predictors_path = self.storage_path / "predictors"
            if predictors_path.exists():
                for pkl_file in predictors_path.glob("*.pkl"):
                    model_name = pkl_file.stem
                    predictor = self.get_predictor(model_name)
                    predictor.load(pkl_file)

            self.is_loaded = True
            logger.info(f"MLManager loaded {len(self.predictors)} predictors")
            return True

        except Exception as e:
            logger.error(f"Error loading ML state: {e}")
            return False

    def _save_state_to_db(self) -> None:
        """Save ML state to database model_ml_state table."""
        try:
            from laser_trim_analyzer.database.models import ModelMLState

            with self.db.session() as session:
                for model_name in self.trained_models:
                    # Get or create state record
                    state = session.query(ModelMLState).filter(
                        ModelMLState.model == model_name
                    ).first()

                    if not state:
                        state = ModelMLState(model=model_name)
                        session.add(state)

                    # Update from threshold optimizer
                    optimizer = self.threshold_optimizers.get(model_name)
                    if optimizer and optimizer.is_calculated:
                        state.is_trained = True
                        state.sigma_threshold = optimizer.threshold
                        state.threshold_confidence = optimizer.confidence
                        state.threshold_method = optimizer.method
                        state.n_pass = optimizer.n_pass
                        state.n_fail = optimizer.n_fail
                        state.pass_sigma_mean = optimizer.pass_sigma_mean
                        state.pass_sigma_std = optimizer.pass_sigma_std
                        state.pass_sigma_max = optimizer.pass_sigma_max
                        state.fail_sigma_min = optimizer.fail_sigma_min
                        state.fail_sigma_mean = optimizer.fail_sigma_mean
                        state.avg_fail_severity = optimizer.avg_fail_severity
                        state.training_samples = optimizer.n_samples
                        state.training_date = optimizer.calculated_date

                    # Update from predictor
                    predictor = self.predictors.get(model_name)
                    if predictor and predictor.is_trained:
                        state.predictor_trained = True
                        if predictor.metrics:
                            state.predictor_accuracy = predictor.metrics.accuracy
                            state.predictor_precision = predictor.metrics.precision
                            state.predictor_recall = predictor.metrics.recall
                            state.predictor_f1 = predictor.metrics.f1
                            state.predictor_auc = predictor.metrics.auc_roc
                        state.feature_importance = predictor.feature_importance

                    # Update from profiler
                    profiler = self.profilers.get(model_name)
                    if profiler and profiler.profile:
                        p = profiler.profile
                        if p.sigma:
                            state.sigma_mean = p.sigma.mean
                            state.sigma_std = p.sigma.std
                            state.sigma_p5 = p.sigma.p5
                            state.sigma_p50 = p.sigma.p50
                            state.sigma_p95 = p.sigma.p95
                        if p.linearity_error:
                            state.error_mean = p.linearity_error.mean
                            state.error_std = p.linearity_error.std
                        state.pass_rate = p.pass_rate
                        state.fail_rate = p.fail_rate
                        state.linearity_pass_rate = p.linearity_pass_rate
                        state.avg_fail_points = p.avg_fail_points
                        state.track_correlation = p.track_correlation
                        state.spec_margin_percent = p.spec_margin_percent
                        state.difficulty_score = p.difficulty_score
                        state.quality_percentile = p.quality_percentile
                        state.linearity_spec = p.linearity_spec

                    # Update from drift detector
                    detector = self.drift_detectors.get(model_name)
                    if detector and detector.has_baseline:
                        state.drift_has_baseline = True
                        state.drift_baseline_mean = detector.baseline_mean
                        state.drift_baseline_std = detector.baseline_std
                        state.drift_baseline_p5 = detector.baseline_p5
                        state.drift_baseline_p50 = detector.baseline_p50
                        state.drift_baseline_p95 = detector.baseline_p95
                        state.drift_baseline_samples = detector.baseline_samples
                        state.drift_baseline_cutoff_date = detector.baseline_cutoff_date
                        state.cusum_pos = detector.cusum_pos
                        state.cusum_neg = detector.cusum_neg
                        state.peak_cusum = detector._peak_cusum
                        state.ewma_value = detector.ewma_value
                        state.is_drifting = detector.is_drifting
                        state.drift_direction = detector.drift_direction.value if detector.drift_direction else None
                        state.drift_start_date = detector.drift_start_date
                        state.samples_since_baseline = detector.samples_since_baseline

                session.commit()
                logger.info(f"Saved ML state for {len(self.trained_models)} models to database")

        except Exception as e:
            logger.error(f"Error saving ML state to database: {e}")

    def _load_state_from_db(self) -> None:
        """Load ML state from database model_ml_state table."""
        try:
            from laser_trim_analyzer.database.models import ModelMLState

            with self.db.session() as session:
                states = session.query(ModelMLState).filter(
                    ModelMLState.is_trained == True
                ).all()

                for state in states:
                    model_name = state.model

                    # Load threshold optimizer state
                    if state.sigma_threshold is not None:
                        optimizer = self.get_threshold_optimizer(model_name)
                        optimizer.threshold = state.sigma_threshold
                        optimizer.confidence = state.threshold_confidence
                        optimizer.method = state.threshold_method
                        optimizer.n_samples = state.training_samples or 0
                        optimizer.n_pass = state.n_pass or 0
                        optimizer.n_fail = state.n_fail or 0
                        optimizer.pass_sigma_mean = state.pass_sigma_mean or 0
                        optimizer.pass_sigma_std = state.pass_sigma_std or 0
                        optimizer.pass_sigma_max = state.pass_sigma_max or 0
                        optimizer.fail_sigma_min = state.fail_sigma_min or 0
                        optimizer.fail_sigma_mean = state.fail_sigma_mean or 0
                        optimizer.avg_fail_severity = state.avg_fail_severity or 0
                        optimizer.is_calculated = True
                        optimizer.calculated_date = state.training_date

                    # Load drift detector state
                    if state.drift_has_baseline:
                        detector = self.get_drift_detector(model_name)
                        detector.has_baseline = True
                        detector.baseline_mean = state.drift_baseline_mean
                        detector.baseline_std = state.drift_baseline_std
                        detector.baseline_p5 = state.drift_baseline_p5
                        detector.baseline_p50 = state.drift_baseline_p50
                        detector.baseline_p95 = state.drift_baseline_p95
                        detector.baseline_samples = state.drift_baseline_samples or 0
                        detector.baseline_cutoff_date = state.drift_baseline_cutoff_date
                        detector.cusum_pos = state.cusum_pos or 0
                        detector.cusum_neg = state.cusum_neg or 0
                        detector._peak_cusum = state.peak_cusum or 0
                        detector.ewma_value = state.ewma_value
                        detector.is_drifting = state.is_drifting or False
                        if state.drift_direction:
                            detector.drift_direction = DriftDirection(state.drift_direction)
                        detector.drift_start_date = state.drift_start_date
                        detector.samples_since_baseline = state.samples_since_baseline or 0

                    # Load profiler state
                    if state.sigma_mean is not None or state.pass_rate is not None:
                        from laser_trim_analyzer.ml.profiler import (
                            ModelProfiler, ModelProfile, ProfileStatistics
                        )
                        profiler = self.get_profiler(model_name)
                        profile = ModelProfile(model_name=model_name)

                        # Restore sigma statistics
                        if state.sigma_mean is not None:
                            profile.sigma = ProfileStatistics(
                                mean=state.sigma_mean or 0,
                                std=state.sigma_std or 0,
                                p5=state.sigma_p5 or 0,
                                p50=state.sigma_p50 or 0,
                                p95=state.sigma_p95 or 0,
                            )

                        # Restore linearity error statistics
                        if state.error_mean is not None:
                            profile.linearity_error = ProfileStatistics(
                                mean=state.error_mean or 0,
                                std=state.error_std or 0,
                            )

                        # Restore quality metrics
                        profile.pass_rate = state.pass_rate or 0
                        profile.fail_rate = state.fail_rate or 0
                        profile.linearity_pass_rate = state.linearity_pass_rate or 0
                        profile.avg_fail_points = state.avg_fail_points or 0
                        profile.track_correlation = state.track_correlation or 0
                        profile.spec_margin_percent = state.spec_margin_percent or 0
                        profile.difficulty_score = state.difficulty_score or 0.5
                        profile.quality_percentile = state.quality_percentile or 0.5
                        profile.linearity_spec = state.linearity_spec
                        profile.sample_count = state.training_samples or 0
                        profile.profiled_date = state.training_date

                        profiler.profile = profile
                        profiler.is_profiled = True

                    # Track as trained
                    if model_name not in self.trained_models:
                        self.trained_models.append(model_name)

                logger.info(f"Loaded ML state for {len(states)} models from database")

        except Exception as e:
            logger.error(f"Error loading ML state from database: {e}")

    def get_training_status(self) -> Dict[str, Any]:
        """Get summary of training status for all models."""
        trained = []
        needs_data = []
        not_trained = []

        for model_name in self._get_model_list():
            if model_name in self.trained_models:
                optimizer = self.threshold_optimizers.get(model_name)
                predictor = self.predictors.get(model_name)
                trained.append({
                    'model': model_name,
                    'threshold': optimizer.threshold if optimizer else None,
                    'confidence': optimizer.confidence if optimizer else None,
                    'predictor_accuracy': predictor.metrics.accuracy if predictor and predictor.metrics else None,
                    'samples': optimizer.n_samples if optimizer else 0,
                })
            elif model_name in self.models_needing_data:
                needs_data.append({
                    'model': model_name,
                    'samples_needed': self.models_needing_data[model_name],
                })
            else:
                not_trained.append(model_name)

        return {
            'trained_count': len(trained),
            'needs_data_count': len(needs_data),
            'not_trained_count': len(not_trained),
            'last_training_date': self.last_training_date,
            'trained': trained,
            'needs_data': needs_data,
            'not_trained': not_trained,
        }

    def get_model_insights(self, model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive insights for a model.

        Returns combined info from predictor, threshold, drift, and profiler.
        """
        insights = {
            'model_name': model_name,
            'has_data': False,
        }

        # Threshold info
        optimizer = self.threshold_optimizers.get(model_name)
        if optimizer and optimizer.is_calculated:
            insights['threshold'] = optimizer.get_statistics()
            insights['has_data'] = True

        # Predictor info
        predictor = self.predictors.get(model_name)
        if predictor and predictor.is_trained:
            insights['predictor'] = predictor.get_state_dict()
            insights['feature_importance'] = predictor.get_feature_importance()

        # Drift info
        detector = self.drift_detectors.get(model_name)
        if detector and detector.has_baseline:
            insights['drift'] = detector.get_statistics()
            lower, center, upper = detector.get_control_limits()
            insights['control_limits'] = {
                'lower': lower,
                'center': center,
                'upper': upper,
            }

        # Profile info
        profiler = self.profilers.get(model_name)
        if profiler and profiler.is_profiled:
            insights['profile'] = profiler.get_profile_dict()
            insights['insights'] = [
                {'category': i.category, 'severity': i.severity, 'message': i.message}
                for i in profiler.get_insights()
            ]

        return insights
