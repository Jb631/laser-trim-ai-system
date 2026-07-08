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
import threading
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
        # Protected by _components_lock for thread-safe access — training
        # runs on a background thread while the GUI thread reads predictions.
        self._components_lock = threading.RLock()
        self.predictors: Dict[str, ModelPredictor] = {}
        self.threshold_optimizers: Dict[str, ModelThresholdOptimizer] = {}
        self.drift_detectors: Dict[str, ModelDriftDetector] = {}
        self.composite_models: Dict[str, "CompositeRiskModel"] = {}
        self.profilers: Dict[str, ModelProfiler] = {}

        # State
        self.is_loaded: bool = False
        self.last_training_date: Optional[datetime] = None
        self.trained_models: List[str] = []
        self.models_needing_data: Dict[str, int] = {}  # model -> samples needed

    def get_predictor(self, model_name: str) -> ModelPredictor:
        """Get or create predictor for a model (thread-safe)."""
        with self._components_lock:
            if model_name not in self.predictors:
                self.predictors[model_name] = ModelPredictor(model_name)
            return self.predictors[model_name]

    def get_threshold_optimizer(self, model_name: str) -> ModelThresholdOptimizer:
        """Get or create threshold optimizer for a model (thread-safe)."""
        with self._components_lock:
            if model_name not in self.threshold_optimizers:
                self.threshold_optimizers[model_name] = ModelThresholdOptimizer(model_name)
            return self.threshold_optimizers[model_name]

    def get_drift_detector(self, model_name: str) -> ModelDriftDetector:
        """Get or create drift detector for a model (thread-safe)."""
        with self._components_lock:
            if model_name not in self.drift_detectors:
                self.drift_detectors[model_name] = ModelDriftDetector(model_name)
            return self.drift_detectors[model_name]

    def get_profiler(self, model_name: str) -> ModelProfiler:
        """Get or create profiler for a model (thread-safe)."""
        with self._components_lock:
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
        # Snapshot the optimizer reference under the lock so a concurrent
        # train_model() insert can't reshape the dict between the membership
        # check and the dereference.
        with self._components_lock:
            optimizer = self.threshold_optimizers.get(model_name)
        if optimizer is not None and optimizer.is_calculated:
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
        with self._components_lock:
            predictor = self.predictors.get(model_name)
        if predictor is not None and predictor.is_trained:
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
            # Calibrate the per-model sigma threshold on the UNTRIMMED (raw-element)
            # sigma vs the post-trim trim outcome -- "how noisy can a raw element be
            # and still trim to pass linearity?". Post-trim sigma was the wrong,
            # weakly-linked input (smoothness != deviation magnitude).
            thr_data = (data[data['untrimmed_sigma_gradient'].notna()]
                        if 'untrimmed_sigma_gradient' in data.columns else data.iloc[0:0])
            if len(thr_data) >= self.MIN_THRESHOLD_SAMPLES:
                if progress_callback:
                    progress_callback(f"Calculating threshold for {model_name}...")

                optimizer = self.get_threshold_optimizer(model_name)
                threshold_result = optimizer.calculate_threshold(
                    sigma_values=thr_data['untrimmed_sigma_gradient'],
                    passed=thr_data['passed'],
                    fail_points=thr_data.get('linearity_fail_points'),
                    linearity_spec=thr_data['linearity_spec'].iloc[0] if 'linearity_spec' in thr_data.columns else None
                )
                result.threshold_calculated = optimizer.is_calculated
                result.threshold_value = threshold_result.threshold
                result.threshold_confidence = threshold_result.confidence

            # 3. Set drift baseline using 70/30 split (needs 30+ samples)
            if len(data) >= self.MIN_DRIFT_SAMPLES:
                if progress_callback:
                    progress_callback(f"Setting drift baseline for {model_name}...")

                detector = self.get_drift_detector(model_name)

                # D-SIGMA: monitor drift on the UNTRIMMED-sweep sigma (the upstream
                # element-production signal), NOT post-trim sigma -- the latter is
                # corrected by trimming and only LAGS the process, so it's the wrong
                # signal for "is the process drifting?". Post-trim sigma remains the
                # product-quality gate (the threshold optimizer above).
                drift_col = 'untrimmed_sigma_gradient'
                if drift_col in data.columns:
                    data_with_sigma = data[data[drift_col].notna()].copy()
                else:
                    data_with_sigma = data.iloc[0:0].copy()
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

                    sigma_values = baseline_data[drift_col].values
                    result.drift_baseline_set = detector.set_baseline(sigma_values, cutoff_date)

                    # Reset detector state before running on detection period
                    if result.drift_baseline_set:
                        detector.reset()

                        # Run drift detection on the newest 30%
                        for sigma in detection_data[drift_col].values:
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
                # Group by serial so repeated trims of one physical unit can't
                # straddle the train/test split (optimistic-bias leakage).
                groups = data['serial'] if 'serial' in data.columns else None

                training_result = predictor.train(features, labels, severity, groups=groups)
                result.predictor_trained = training_result.success
                if training_result.metrics:
                    result.predictor_accuracy = training_result.metrics.accuracy

            # Composite trim-risk model (2026-06-01 plan). Reuses the same
            # per-model training frame; grouped-CV + deploy-gate inside.
            try:
                self._train_composite_risk(model_name, data)
            except Exception as e:
                logger.warning("Composite risk training failed for %s: %s", model_name, e)

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

            # Simple callback for single model — bind loop variables by value
            def model_callback(msg: str, _name=model_name, _i=i):
                if progress_callback:
                    progress_callback(TrainingProgress(
                        current_model=_name,
                        models_complete=_i,
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
            from sqlalchemy import update, case, and_, func
            from laser_trim_analyzer.database.models import (
                TrackResult, AnalysisResult, QAAlert, AlertType
            )

            models_updated = set()

            # Drop the explicit outer _write_lock — db.session() already takes
            # it (RLock-reentrant) on enter, and holding it for the entire
            # multi-model loop blocked every other DB consumer (dashboard
            # refresh, incremental check, ML staleness lookup) for the full
            # duration of a multi-second apply.
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
                        analysis_subquery = None  # Defined here so prediction block can use it

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

                            # Bulk update sigma_pass: the threshold is calibrated on
                            # UNTRIMMED sigma, so gate on untrimmed (raw-element) sigma,
                            # falling back to post-trim only when the sweep is absent.
                            session.execute(
                                update(TrackResult)
                                .where(
                                    and_(
                                        TrackResult.analysis_id.in_(analysis_subquery),
                                        func.coalesce(
                                            TrackResult.untrimmed_sigma_gradient,
                                            TrackResult.sigma_gradient,
                                        ).isnot(None),
                                    )
                                )
                                .values(
                                    sigma_pass=case(
                                        (func.coalesce(
                                            TrackResult.untrimmed_sigma_gradient,
                                            TrackResult.sigma_gradient,
                                        ) <= new_threshold, True),
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
                            session.rollback()
                            counts['errors'] += 1
                            continue  # Skip remaining steps for this model

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
                            session.rollback()
                            continue  # Skip remaining steps for this model

                        # OPTIMIZATION 3.5: Update failure_probability using ML predictor
                        # Uses batch prediction (single DataFrame + predict_proba call)
                        # instead of per-track loop. ~50-100x faster for large models.
                        predictor = self.predictors.get(model_name)
                        if predictor and predictor.is_trained and analysis_subquery is not None:
                            try:
                                # Load tracks with features needed for prediction
                                tracks = (
                                    session.query(
                                        TrackResult.id,
                                        TrackResult.sigma_gradient,
                                        TrackResult.final_linearity_error_shifted,
                                        TrackResult.linearity_fail_points,
                                        TrackResult.optimal_offset,
                                        TrackResult.linearity_spec,
                                    )
                                    .filter(TrackResult.analysis_id.in_(analysis_subquery))
                                    .filter(TrackResult.sigma_gradient.isnot(None))
                                    .all()
                                )

                                if tracks:
                                    # Build feature dicts for all tracks
                                    track_ids = []
                                    features_list = []
                                    for track in tracks:
                                        sigma = track.sigma_gradient or 0.0
                                        lin_error = abs(track.final_linearity_error_shifted or 0.0)
                                        lin_spec = track.linearity_spec or 0.01
                                        track_ids.append(track.id)
                                        features_list.append({
                                            'sigma_gradient': sigma,
                                            'linearity_error': lin_error,
                                            'fail_points': track.linearity_fail_points or 0,
                                            'optimal_offset': track.optimal_offset or 0.0,
                                            'linearity_spec': lin_spec,
                                            'sigma_to_spec': sigma / lin_spec if lin_spec > 0 else 0.0,
                                            'error_to_spec': lin_error / lin_spec if lin_spec > 0 else 0.0,
                                        })

                                    # Single batch prediction — one DataFrame, one
                                    # scaler.transform, one predict_proba call
                                    probabilities = predictor.predict_batch(features_list)

                                    update_mappings = [
                                        {'id': tid, 'failure_probability': prob}
                                        for tid, prob in zip(track_ids, probabilities)
                                        if prob is not None
                                    ]

                                    if update_mappings:
                                        session.bulk_update_mappings(TrackResult, update_mappings)

                                    logger.info(f"Updated {len(update_mappings)} failure predictions for {model_name}")

                            except Exception as e:
                                logger.warning(f"Error updating failure predictions for {model_name}: {e}")

                        # OPTIMIZATION 4: Drift detection - only load detection-period tracks
                        if run_drift_detection and detector and detector.has_baseline:
                            try:
                                # Only get tracks AFTER baseline cutoff date
                                cutoff_date = detector.baseline_cutoff_date

                                if cutoff_date:
                                    # D-SIGMA: detect on the same UNTRIMMED signal the
                                    # baseline was set on (post-trim sigma is the wrong,
                                    # lagging signal for process drift).
                                    detection_tracks = (
                                        session.query(TrackResult.untrimmed_sigma_gradient, AnalysisResult.id)
                                        .join(AnalysisResult)
                                        .filter(AnalysisResult.model == model_name)
                                        .filter(AnalysisResult.file_date > cutoff_date)
                                        .filter(TrackResult.untrimmed_sigma_gradient.isnot(None))
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

                                                # Persist drift alert to DB
                                                try:
                                                    # Map float severity (0-1) to string category
                                                    if drift_result.severity >= 0.75:
                                                        severity_str = "Critical"
                                                    elif drift_result.severity >= 0.5:
                                                        severity_str = "High"
                                                    elif drift_result.severity >= 0.25:
                                                        severity_str = "Medium"
                                                    else:
                                                        severity_str = "Low"

                                                    alert = QAAlert(
                                                        analysis_id=analysis_id,
                                                        alert_type=AlertType.DRIFT_DETECTED,
                                                        severity=severity_str,
                                                        message=drift_result.message,
                                                        metric_name="sigma_gradient",
                                                        metric_value=drift_result.cusum_value,
                                                    )
                                                    session.add(alert)
                                                except Exception as e:
                                                    logger.warning(f"Failed to persist drift alert: {e}")

                            except Exception as e:
                                logger.warning(f"Error in drift detection for {model_name}: {e}")

                        # Commit after each model to avoid holding locks too long
                        session.commit()

                        # Score composite_trim_risk_score for deployed models.
                        # Done AFTER the commit (a separate raw-sqlite3 connection)
                        # so the bulk UPDATE doesn't contend with the session's
                        # uncommitted writes -- that contention raises
                        # "database is locked" and would silently skip scoring.
                        try:
                            self._score_composite_for_model(
                                str(self.db.database_path), model_name
                            )
                        except Exception as e:
                            logger.warning("Composite scoring failed for %s: %s", model_name, e)

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

                    # Save updated drift state using the existing session to
                    # avoid opening a nested session on the StaticPool connection
                    self._save_state_to_db(existing_session=session)
                    session.commit()

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
                # Get trim data. Exclude UNTRIMMED tracks — test-sweep-only
                # files have no sigma/linearity to train on, and the .notna()
                # filter below has a "return everything" fallback that would
                # otherwise leak them into training when nothing else exists.
                trim_results = (
                    session.query(TrackResult, AnalysisResult.file_date, AnalysisResult.serial)
                    .join(AnalysisResult)
                    .filter(AnalysisResult.model == model_name)
                    .filter(TrackResult.status != StatusType.UNTRIMMED.name)
                    .all()
                )

                for track, file_date, serial in trim_results:
                    records.append({
                        'sigma_gradient': track.sigma_gradient,
                        # Upstream process signal used for DRIFT (D-SIGMA): the
                        # untrimmed-sweep sigma, not the trim-corrected one.
                        'untrimmed_sigma_gradient': track.untrimmed_sigma_gradient,
                        'linearity_error': track.final_linearity_error_shifted or 0,
                        'linearity_fail_points': track.linearity_fail_points or 0,
                        'optimal_offset': track.optimal_offset or 0,
                        'linearity_spec': track.linearity_spec or 0.01,
                        'linearity_pass': track.linearity_pass if track.linearity_pass is not None else True,
                        'sigma_pass': track.sigma_pass,
                        'passed': track.linearity_pass if track.linearity_pass is not None else True,
                        'file_date': file_date,
                        # serial groups repeated trims of one physical unit so they
                        # don't leak across the train/test split (a unit can be
                        # re-trimmed many times -- valid, not a duplicate).
                        'serial': serial,
                        'source': 'trim',
                        # Composite trim-risk features (task 6): worst raw-element
                        # deviation, resistance shift mode, and trim-headroom mode.
                        'untrimmed_error_max': track.untrimmed_error_max,
                        'resistance_change_percent': track.resistance_change_percent,
                        'trim_pass_count': track.trim_pass_count,
                    })

                # Get Final Test data (higher priority when linked)
                final_results = (
                    session.query(FinalTestTrack, FinalTestResult.file_date, FinalTestResult.serial)
                    .join(FinalTestResult)
                    .filter(FinalTestResult.model == model_name)
                    .all()
                )

                for track, file_date, serial in final_results:
                    # Final Test data overwrites trim outcome for linked records
                    records.append({
                        'sigma_gradient': None,  # Final test doesn't have sigma
                        'untrimmed_sigma_gradient': None,  # nor an untrimmed sweep
                        'linearity_error': track.linearity_error or 0,
                        'linearity_fail_points': track.linearity_fail_points or 0,
                        'optimal_offset': 0,
                        'linearity_spec': track.linearity_spec or 0.01,
                        'linearity_pass': track.linearity_pass if track.linearity_pass is not None else True,
                        'sigma_pass': None,
                        'passed': track.linearity_pass if track.linearity_pass is not None else True,
                        'file_date': file_date,
                        'serial': serial,
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

    def _save_state_to_db(self, existing_session=None) -> None:
        """Save ML state to database model_ml_state table.

        Args:
            existing_session: If provided, use this session instead of opening
                a new one.  The caller is responsible for committing.  When
                *not* provided the method opens (and commits) its own session.
        """
        try:
            from laser_trim_analyzer.database.models import ModelMLState

            def _do_save(session):
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
                        state.consecutive_recovered = detector._consecutive_recovered

            if existing_session is not None:
                # Use caller's session -- caller is responsible for commit
                _do_save(existing_session)
            else:
                # Open our own session and commit
                with self.db.session() as session:
                    _do_save(session)
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
                        detector._consecutive_recovered = getattr(state, 'consecutive_recovered', 0) or 0

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

                    # Restore predictor trained status from DB
                    if state.predictor_trained:
                        if model_name in self.predictors:
                            self.predictors[model_name].is_trained = True

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

    def _train_composite_risk(self, model_name: str, data) -> "CompositeRiskModel":
        """Train and persist the per-model composite trim-risk model."""
        from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
        crm = CompositeRiskModel(model_name)
        res = crm.train(data)
        logger.info(
            "Composite risk [%s]: cv_auc=%.3f best_single=%.3f conf=%.3f deployed=%s (%s)",
            model_name, res.cv_auc, res.best_single_auc, res.confidence,
            res.deployed, res.reason,
        )
        out_dir = self.storage_path / "composite_risk"
        out_dir.mkdir(parents=True, exist_ok=True)
        crm.save(out_dir / f"{model_name}.pkl")
        if not hasattr(self, "composite_models"):
            self.composite_models = {}
        self.composite_models[model_name] = crm
        return crm

    def _score_composite_for_model(self, db_path: str, model_name: str) -> int:
        """Write composite_trim_risk_score for all of a model's tracks.
        Only runs for deployed models; returns rows scored."""
        import sqlite3
        from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel, FEATURES
        crm = getattr(self, "composite_models", {}).get(model_name)
        if crm is None:
            p = self.storage_path / "composite_risk" / f"{model_name}.pkl"
            if not p.exists():
                return 0
            crm = CompositeRiskModel.load(p)
        if not (crm.is_trained and crm.result and crm.result.deployed):
            return 0
        cols = ", ".join(FEATURES)
        con = sqlite3.connect(db_path); cur = con.cursor()
        rows = cur.execute(
            f"SELECT t.id, {cols} FROM track_results t "
            f"JOIN analysis_results ar ON t.analysis_id = ar.id WHERE ar.model = ?",
            (model_name,)).fetchall()
        scored = 0
        for r in rows:
            tid = r[0]
            feat = {f: r[1 + i] for i, f in enumerate(FEATURES)}
            s = crm.predict_proba(feat)
            if s == s:  # not NaN
                cur.execute("UPDATE track_results SET composite_trim_risk_score=? WHERE id=?",
                            (s, tid)); scored += 1
        con.commit(); con.close()
        return scored

    def get_adjustment_recommendations(self, model: str) -> List[Dict[str, Any]]:
        """
        Generate adjustment recommendations based on model history and specs.

        Returns list of {recommendation, priority, rationale} dicts.
        """
        recommendations = []

        try:
            # Check drift
            detector = self.drift_detectors.get(model)
            if detector and detector.has_baseline and detector.is_drifting:
                direction = detector.drift_direction.value if detector.drift_direction else "unknown"
                recommendations.append({
                    "recommendation": "Investigate process drift",
                    "priority": "High",
                    "rationale": (
                        f"Drift detection triggered for {model} (direction: {direction}). "
                        f"Review recent process changes, material lots, or equipment."
                    ),
                })

            # Check pass rate from profile
            profiler = self.profilers.get(model)
            if profiler and profiler.is_profiled:
                pass_rate = profiler.profile.pass_rate if profiler.profile else None
                if pass_rate is not None and pass_rate < 0.6:
                    recommendations.append({
                        "recommendation": "Prioritize for root cause analysis",
                        "priority": "High",
                        "rationale": (
                            f"Pass rate of {pass_rate*100:.0f}% is below 60%. "
                            f"This model is consuming significant rework resources."
                        ),
                    })

        except Exception as e:
            logger.debug(f"Could not generate recommendations for {model}: {e}")

        return recommendations


def get_drifting_models(db, sensitivity_preset: str = "standard"):
    """Return sorted list of currently-flagged models.

    Reads model_metric_state directly (no historical re-scan).  Each
    model's overall_tier is computed by hydrating its 8 MetricDetector
    rows and running the worst-of aggregation.  Sorted by (tier desc,
    magnitude desc).

    Returns empty list when nothing is above Stable.

    ``sensitivity_preset`` is currently unused -- the active thresholds
    are whatever was written to model_metric_state at last training
    time.  Callers wanting a "what-if at a different preset" view
    should use ``preview_alert_count(db, preset)`` instead.  The
    parameter is preserved on the signature for API symmetry with the
    other Spec 2 entry points; future versions may honor it by
    recomputing thresholds in-flight.
    """
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, ModelAlertSummary,
    )

    summaries: list[ModelAlertSummary] = []
    with db.session() as s:
        # Get unique models that have at least one row
        models = [
            r[0] for r in s.query(ModelMetricState.model).distinct().all()
        ]

    for model in models:
        status = get_model_drift_status(db, model)
        if status.overall_tier > DriftTier.STABLE:
            summaries.append(ModelAlertSummary(
                model=model,
                tier=status.overall_tier,
                alert_type=status.worst_alert_type,
                worst_metric=status.worst_metric or "",
                magnitude=status.per_metric[status.worst_metric].magnitude
                          if status.worst_metric else 0.0,
            ))

    summaries.sort(key=lambda r: (int(r.tier), r.magnitude), reverse=True)
    return summaries


# Triage gate: minimum |recent shift from baseline| (σ) for a flagged model to be
# *hidden*. 0.0 = hide nothing (the cautious default). The V6 drift signal is weak
# and we don't yet know what a "real" shift looks like, so we surface every flagged
# model and let the honest shift number + sort do the prioritizing rather than
# silently dropping anything. Raise this once the real-vs-noise boundary is known.
TRIAGE_MIN_SIGMA_SHIFT: float = 0.0


def get_triage_alerts(db, min_sigma_shift: float = TRIAGE_MIN_SIGMA_SHIFT):
    """Triage feed: flagged models enriched with the honest recent sigma-shift, sorted
    by how far they have ACTUALLY moved (|shift| desc) rather than by the abstract
    CUSUM/EWMA magnitude.

    Why this exists: the detector's `magnitude` is "distance past the control limit",
    which can read in the hundreds for a sub-1σ shift (the CUSUM stays elevated after a
    transient that has since recovered). Sorting/labelling by the real shift surfaces
    the genuine drifts and makes the 0.01σ "still flagged but recovered" cases obvious
    instead of letting them dominate. `min_sigma_shift` can hide confirmed-tiny shifts,
    but defaults to 0.0 (hide nothing) — see TRIAGE_MIN_SIGMA_SHIFT.
    """
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.export.evidence import compute_recent_means

    summaries = get_drifting_models(db)
    for a in summaries:
        if not a.worst_metric:
            continue
        with db.session() as s:
            row = (s.query(ModelMetricState.baseline_mean, ModelMetricState.baseline_std)
                   .filter(ModelMetricState.model == a.model,
                           ModelMetricState.metric == a.worst_metric).first())
        recent = compute_recent_means(db, a.model).get(a.worst_metric)
        if row and recent is not None and row[1]:
            a.sigma_shift = (recent - row[0]) / row[1]
    return _order_triage_alerts(summaries, min_sigma_shift)


def active_model_set(db, recent_days: int = 90, mps_models=None) -> set:
    """The set of models considered ACTIVE (current production).

    A model is active if it has data within `recent_days` of the dataset's most recent
    date, OR it's user-pinned on the MPS list. Anchored to the dataset's latest date
    (not wall-clock now) so a loaded historical batch still resolves sensibly — the same
    reasoning as the 'recent' window elsewhere. Triage uses this to focus on models the
    operator is actually running; legacy models (e.g. a unit last processed in 2016)
    shouldn't dominate 'what to look at today'.
    """
    from datetime import timedelta
    from sqlalchemy import func
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, SmoothnessResult as DBSR)
    active = set(mps_models or [])
    with db.session() as s:
        latest = s.query(func.max(DBAR.file_date)).scalar()
        if latest is None:
            return active
        cutoff = latest - timedelta(days=recent_days)
        for (m,) in (s.query(DBAR.model)
                     .filter(DBAR.file_date >= cutoff, DBAR.model.isnot(None)).distinct()):
            active.add(m)
        for (m,) in (s.query(DBSR.model)
                     .filter(DBSR.file_date >= cutoff, DBSR.model.isnot(None)).distinct()):
            active.add(m)
    return active


def _order_triage_alerts(summaries, min_sigma_shift: float = TRIAGE_MIN_SIGMA_SHIFT):
    """Gate + order enriched alert summaries. Pure (no DB) so it's directly testable.

    TIER FIRST, then |actual shift| within the tier. Worst-severity-on-top is a
    deliberate, protected V6 decision ("nervous about losing the tiers"); a raw shift
    must NOT reorder across tiers, or a Warning could jump above a Drift and slow-drift
    (CUSUM) models — half the mission — would sink below noisier step-changes. Within a
    tier, sort by the honest baseline shift (replacing `magnitude` as the tiebreaker) so
    the biggest real movers lead and recovered/sub-noise alerts fall to the bottom of
    their tier. `min_sigma_shift` can hide confirmed-tiny shifts (default 0.0 = hide
    nothing; selectivity is owned by the per-model preset thresholds, not this gate).
    """
    kept = [a for a in summaries
            if a.sigma_shift is None or abs(a.sigma_shift) >= min_sigma_shift]
    kept.sort(key=lambda r: (int(r.tier),
                             abs(r.sigma_shift) if r.sigma_shift is not None else -1.0),
              reverse=True)
    return kept


def get_model_drift_status(db, model: str):
    """Return full per-metric breakdown for one model.

    Hydrates MetricDetector instances from model_metric_state rows and
    asks the container for its current status.
    """
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MetricDetector, MultiMetricDriftDetector,
    )

    metrics = {}
    # Materialize row data inside the session to avoid DetachedInstanceError
    # when expire_on_commit=True causes attributes to be reloaded post-close.
    rows_by_metric: dict = {}
    with db.session() as s:
        rows = s.query(ModelMetricState).filter(
            ModelMetricState.model == model,
        ).all()
        for r in rows:
            rows_by_metric[r.metric] = {
                "baseline_mean": r.baseline_mean,
                "baseline_std": r.baseline_std,
                "baseline_count": r.baseline_count,
                "is_trained": r.is_trained,
                "h_warning": r.h_warning,
                "h_drift": r.h_drift,
                "h_oc": r.h_oc,
                "L_warning": r.L_warning,
                "L_drift": r.L_drift,
                "L_oc": r.L_oc,
                "z_warning": r.z_warning,
                "z_drift": r.z_drift,
                "z_oc": r.z_oc,
                "cusum_pos": r.cusum_pos,
                "cusum_neg": r.cusum_neg,
                "ewma_state": r.ewma_state,
                "recent_window": r.recent_window,
                "last_updated": r.last_updated,
            }

    from collections import deque
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        COMPOSITE_METRIC, STEP_CHANGE_WINDOW,
    )

    for metric_name in WATCHED_METRICS:
        row = rows_by_metric.get(metric_name)
        if row is None:
            # No DB row -> create a placeholder untrained detector
            metrics[metric_name] = MetricDetector(
                metric=metric_name,
                baseline_mean=0.0,
                baseline_std=0.0,
                baseline_count=0,
                is_trained=False,
            )
        else:
            det = MetricDetector(
                metric=metric_name,
                baseline_mean=row["baseline_mean"] or 0.0,
                baseline_std=row["baseline_std"] or 0.0,
                baseline_count=row["baseline_count"],
                is_trained=row["is_trained"],
                h_per_tier={
                    "WARNING": row["h_warning"] or 0.0,
                    "DRIFT": row["h_drift"] or 0.0,
                    "OUT_OF_CONTROL": row["h_oc"] or 0.0,
                },
                L_per_tier={
                    "WARNING": row["L_warning"] or 0.0,
                    "DRIFT": row["L_drift"] or 0.0,
                    "OUT_OF_CONTROL": row["L_oc"] or 0.0,
                },
                z_per_tier={
                    "WARNING": row["z_warning"] or 0.0,
                    "DRIFT": row["z_drift"] or 0.0,
                    "OUT_OF_CONTROL": row["z_oc"] or 0.0,
                },
                cusum_pos=row["cusum_pos"] or 0.0,
                cusum_neg=row["cusum_neg"] or 0.0,
                ewma_state=row["ewma_state"],
                # Persisted step-change window: without it, hydrated detectors
                # woke with an empty window — STEP_CHANGE was unreachable at
                # read time and the σ-shift tiebreaker was always 0.
                recent_window=deque(
                    [float(v) for v in (row["recent_window"] or [])],
                    maxlen=STEP_CHANGE_WINDOW),
            )
            metrics[metric_name] = det

    # Composite staleness: if the composite's sample marker lags the rest of
    # this model's metrics by more than 30 days, scores have stopped being
    # produced (e.g. un-deployed after a retrain). It must then stop demoting
    # its input family, or the whole trim-effort family goes silent.
    comp_row = rows_by_metric.get(COMPOSITE_METRIC)
    if comp_row is not None and comp_row["is_trained"]:
        sibling_dates = [
            r["last_updated"] for m, r in rows_by_metric.items()
            if m != COMPOSITE_METRIC and r["is_trained"] and r["last_updated"] is not None
        ]
        comp_date = comp_row["last_updated"]
        if sibling_dates and (
            comp_date is None
            or (max(sibling_dates) - comp_date).days > 30
        ):
            metrics[COMPOSITE_METRIC].represents_family = False

    container = MultiMetricDriftDetector(model=model, metrics=metrics)
    return container.get_status()


def preview_alert_count(db, sensitivity_preset: str) -> dict:
    """Count models that would flag at each tier under the candidate preset.

    Cheap -- doesn't re-scan history.  Recomputes per-tier thresholds for
    each existing row, then evaluates against cached runtime state.
    """
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, target_fp_for_tier, WATCHED_METRICS,
    )
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        MetricDetector, MultiMetricDriftDetector, compute_thresholds,
    )

    counts = {"warning": 0, "drift": 0, "out_of_control": 0}

    with db.session() as s:
        models = [
            r[0] for r in s.query(ModelMetricState.model).distinct().all()
        ]

    for model in models:
        # Build detectors with candidate-preset thresholds in place of cached ones
        metrics = {}
        # Materialize row data inside the session to avoid DetachedInstanceError
        row_data: list[dict] = []
        with db.session() as s:
            rows = s.query(ModelMetricState).filter(
                ModelMetricState.model == model
            ).all()
            for r in rows:
                row_data.append({
                    "metric": r.metric,
                    "is_trained": r.is_trained,
                    "baseline_mean": r.baseline_mean,
                    "baseline_std": r.baseline_std,
                    "baseline_count": r.baseline_count,
                    "cusum_pos": r.cusum_pos,
                    "cusum_neg": r.cusum_neg,
                    "ewma_state": r.ewma_state,
                })

        for row in row_data:
            if not row["is_trained"] or row["baseline_std"] is None:
                continue
            # Shared helper applies the Bonferroni correction — preview must
            # compute thresholds EXACTLY like training or the counts lie
            # (pre-2026-07-06 this path skipped the /n_metrics correction).
            from laser_trim_analyzer.ml.drift_training import corrected_tier_thresholds
            thresholds = corrected_tier_thresholds(sensitivity_preset, row["baseline_std"])
            h_per_tier = {t.name: thresholds[t][0] for t in thresholds}
            L_per_tier = {t.name: thresholds[t][1] for t in thresholds}
            z_per_tier = {t.name: thresholds[t][2] for t in thresholds}

            metrics[row["metric"]] = MetricDetector(
                metric=row["metric"],
                baseline_mean=row["baseline_mean"],
                baseline_std=row["baseline_std"],
                baseline_count=row["baseline_count"],
                is_trained=True,
                h_per_tier=h_per_tier,
                L_per_tier=L_per_tier,
                z_per_tier=z_per_tier,
                cusum_pos=row["cusum_pos"] or 0.0,
                cusum_neg=row["cusum_neg"] or 0.0,
                ewma_state=row["ewma_state"],
            )

        container = MultiMetricDriftDetector(model=model, metrics=metrics)
        status = container.get_status()
        if status.overall_tier == DriftTier.WARNING:
            counts["warning"] += 1
        elif status.overall_tier == DriftTier.DRIFT:
            counts["drift"] += 1
        elif status.overall_tier == DriftTier.OUT_OF_CONTROL:
            counts["out_of_control"] += 1

    return counts


def list_known_models(db):
    """One ModelSummary per distinct model across analysis_results + smoothness_results.

    Cost: ONE inventory session (GROUP BY model, MAX(file_date)) + ONE
    get_drifting_models call for tiers.  Independent of model count (fixes the
    per-model-session N+1).  Non-flagged models default to DriftTier.STABLE.
    """
    from sqlalchemy import func
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, SmoothnessResult as DBSR,
    )
    from laser_trim_analyzer.ml.drift_types import DriftTier, ModelSummary

    last_seen = {}
    with db.session() as s:
        for model, last in (s.query(DBAR.model, func.max(DBAR.file_date))
                            .group_by(DBAR.model).all()):
            if model:
                last_seen[model] = last
        for model, last in (s.query(DBSR.model, func.max(DBSR.file_date))
                            .group_by(DBSR.model).all()):
            if not model:
                continue
            if model not in last_seen:
                last_seen[model] = last
            elif last is not None and (last_seen[model] is None or last > last_seen[model]):
                last_seen[model] = last

    flagged = {a.model: a.tier for a in get_drifting_models(db)}
    return sorted(
        (ModelSummary(model=m, tier=flagged.get(m, DriftTier.STABLE),
                      last_processed=last_seen.get(m)) for m in last_seen),
        key=lambda x: x.model,
    )


def apply_sensitivity_preset(db, preset: str) -> int:
    """Recompute per-tier (h, L, z) in place from each trained row's cached baseline_std
    using `preset`'s target FP rates. Preserves baseline_* and runtime cusum/ewma state
    (no history re-scan). Returns rows updated. Lets Settings 'Save preset' change what
    Triage flags without a full retrain (get_drifting_models ignores its preset arg)."""
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.drift_types import DriftTier
    from laser_trim_analyzer.ml.drift_training import corrected_tier_thresholds

    updated = 0
    with db.session() as s:
        rows = s.query(ModelMetricState).filter(
            ModelMetricState.is_trained == True,                # noqa: E712
            ModelMetricState.baseline_std.isnot(None)).all()
        for row in rows:
            # Shared helper = same Bonferroni-corrected math as training, so a
            # saved preset produces the thresholds a retrain would (pre-2026-07-06
            # this path skipped the correction: ~9x looser in FP target).
            thresholds = corrected_tier_thresholds(preset, row.baseline_std)
            for tier, (hc, lc, zc) in (
                (DriftTier.WARNING, ("h_warning", "L_warning", "z_warning")),
                (DriftTier.DRIFT, ("h_drift", "L_drift", "z_drift")),
                (DriftTier.OUT_OF_CONTROL, ("h_oc", "L_oc", "z_oc")),
            ):
                h, L, z = thresholds[tier]
                setattr(row, hc, h)
                setattr(row, lc, L)
                setattr(row, zc, z)
            # NOTE: deliberately NOT touching row.last_updated — it is the
            # advance_drift_state sample marker, not a wall-clock audit field.
            # Overwriting it with now() (the old behavior) told the legacy
            # date-fallback that everything before this moment was consumed.
            updated += 1
        s.commit()
    return updated
