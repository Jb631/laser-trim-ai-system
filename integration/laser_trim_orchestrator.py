"""
Laser Trim AI System - Master Orchestrator
==========================================

This module provides seamless integration of all components:
- Data Processing
- Machine Learning Analysis
- Database Storage
- Report Generation
- GUI Interface

Author: QA Team
Date: 2024
Version: 2.0.0
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
import traceback

# Component imports
from config import Config, ConfigManager
from data_processor import DataProcessor, SystemType
from ml_models import LaserTrimMLModels, create_ml_models
from excel_reporter import ExcelReporter
from database import DatabaseManager, HistoricalAnalyzer, TrendReporter, DataMigrator


class ProcessingStatus(Enum):
    """Status of file processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ML_ANALYSIS = "ml_analysis"
    DB_SAVING = "db_saving"
    REPORT_GEN = "report_generation"


@dataclass
class ProcessingResult:
    """Container for processing results."""
    file_path: str
    status: ProcessingStatus
    data_result: Optional[Dict[str, Any]] = None
    ml_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    warnings: List[str] = None


class LaserTrimOrchestrator:
    """
    Master orchestrator for the Laser Trim AI System.

    Coordinates all components and provides a unified interface
    for processing, analysis, and excel_reporting.
    """

    def __init__(self, config_path: Optional[str] = None,
                 enable_parallel: bool = True,
                 enable_ml: bool = True,
                 enable_db: bool = True):
        """
        Initialize the orchestrator with all components.

        Args:
            config_path: Path to configuration file
            enable_parallel: Enable parallel processing
            enable_ml: Enable ML analysis
            enable_db: Enable database storage
        """
        self.logger = self._setup_logger()
        self.logger.info("Initializing Laser Trim AI System Orchestrator...")

        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config

        # Processing settings
        self.enable_parallel = enable_parallel
        self.enable_ml = enable_ml
        self.enable_db = enable_db

        # Initialize components
        self._initialize_components()

        # Processing state
        self.current_run_id = None
        self.processing_queue = []
        self.results_cache = {}

        # Performance metrics
        self.metrics = {
            'files_processed': 0,
            'files_failed': 0,
            'total_processing_time': 0,
            'ml_predictions_made': 0,
            'reports_generated': 0
        }

        self.logger.info("Orchestrator initialization complete")

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('LaserTrimOrchestrator')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # File handler
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / f'orchestrator_{datetime.now().strftime("%Y%m%d")}.log'
            )
            file_handler.setLevel(logging.DEBUG)

            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger

    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Data processor
            self.data_processor = DataProcessor(logger=self.logger)
            self.logger.info("✓ Data processor initialized")

            # ML models
            if self.enable_ml:
                self.ml_models = create_ml_models(self.config)
                self._load_ml_models()
                self.logger.info("✓ ML models initialized")
            else:
                self.ml_models = None
                self.logger.info("⚠ ML models disabled")

            # Database
            if self.enable_db:
                self.db_manager = DatabaseManager(self.config)
                self.historical_analyzer = HistoricalAnalyzer(self.db_manager, self.config)
                self.trend_reporter = TrendReporter(
                    self.db_manager, self.historical_analyzer, self.config
                )
                self.data_migrator = DataMigrator(self.db_manager, self.config)
                self.logger.info("✓ Database components initialized")
            else:
                self.db_manager = None
                self.logger.info("⚠ Database disabled")

            # Report generator
            report_config = {}
            if hasattr(self.config, 'openai_api_key'):
                report_config['openai_api_key'] = self.config.openai_api_key

            self.report_generator = ExcelReporter(report_config)
            self.logger.info("✓ Report generator initialized")

            # Thread pool for parallel processing
            if self.enable_parallel:
                self.executor = ThreadPoolExecutor(max_workers=4)
                self.logger.info("✓ Parallel processing enabled (4 workers)")
            else:
                self.executor = None
                self.logger.info("⚠ Parallel processing disabled")

        except Exception as e:
            self.logger.error(f"Component initialization failed: {str(e)}")
            raise

    def _load_ml_models(self):
        """Load pre-trained ML models."""
        try:
            # Try to load latest models
            models_dir = Path(self.config.output.output_dir) / 'ml_models'
            if models_dir.exists():
                latest_version = self._find_latest_model_version(models_dir)
                if latest_version:
                    self.ml_models.load_models(latest_version)
                    self.logger.info(f"✓ Loaded ML models version: {latest_version}")
                else:
                    self.logger.warning("No pre-trained ML models found")
            else:
                self.logger.warning("ML models directory not found")
        except Exception as e:
            self.logger.warning(f"Could not load ML models: {str(e)}")

    def _find_latest_model_version(self, models_dir: Path) -> Optional[str]:
        """Find the latest model version."""
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('models_v')]
        if model_dirs:
            # Sort by modification time
            latest = max(model_dirs, key=lambda d: d.stat().st_mtime)
            return latest.name.replace('models_', '')
        return None

    # ==================== Main Processing Methods ====================

    async def process_folder_async(self, folder_path: Union[str, Path],
                                   output_dir: Optional[Union[str, Path]] = None,
                                   generate_report: bool = True,
                                   report_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Asynchronously process all files in a folder.

        Args:
            folder_path: Path to folder containing Excel files
            output_dir: Output directory for results
            generate_report: Whether to generate Excel report
            report_name: Custom report name

        Returns:
            Dictionary with processing results
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")

        # Set output directory
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(self.config.output.output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all Excel files
        excel_files = list(folder_path.glob('*.xls*'))
        self.logger.info(f"Found {len(excel_files)} Excel files to process")

        # Create database run
        if self.enable_db and self.db_manager:
            self.current_run_id = self.db_manager.create_analysis_run(
                str(folder_path),
                asdict(self.config)
            )

        # Process files
        start_time = datetime.now()
        results = await self._process_files_parallel(excel_files) if self.enable_parallel else \
            self._process_files_sequential(excel_files)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Update database run
        if self.enable_db and self.db_manager and self.current_run_id:
            successful = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
            failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)

            self.db_manager.update_analysis_run(
                self.current_run_id,
                processed_files=successful,
                failed_files=failed,
                total_files=len(excel_files),
                processing_time=processing_time
            )

        # Generate report
        report_path = None
        if generate_report:
            report_path = self._generate_comprehensive_report(
                results, output_dir, report_name
            )

        # Prepare summary
        summary = self._create_processing_summary(results, processing_time, report_path)

        # Save results to file
        self._save_results_to_file(summary, output_dir)

        # Update metrics
        self._update_metrics(results)

        return summary

    def process_folder(self, folder_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for folder processing."""
        return asyncio.run(self.process_folder_async(folder_path, **kwargs))

    async def _process_files_parallel(self, files: List[Path]) -> List[ProcessingResult]:
        """Process files in parallel."""
        self.logger.info(f"Processing {len(files)} files in parallel...")

        # Create tasks
        tasks = []
        for file_path in files:
            task = asyncio.create_task(self._process_single_file_async(file_path))
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    file_path=str(files[i]),
                    status=ProcessingStatus.FAILED,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    def _process_files_sequential(self, files: List[Path]) -> List[ProcessingResult]:
        """Process files sequentially."""
        self.logger.info(f"Processing {len(files)} files sequentially...")

        results = []
        for i, file_path in enumerate(files):
            self.logger.info(f"Processing file {i + 1}/{len(files)}: {file_path.name}")
            result = self._process_single_file(file_path)
            results.append(result)

        return results

    async def _process_single_file_async(self, file_path: Path) -> ProcessingResult:
        """Asynchronously process a single file."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._process_single_file, file_path)

    def _process_single_file(self, file_path: Path) -> ProcessingResult:
        """
        Process a single file through the complete pipeline.

        Args:
            file_path: Path to Excel file

        Returns:
            ProcessingResult with all analysis data
        """
        result = ProcessingResult(
            file_path=str(file_path),
            status=ProcessingStatus.PROCESSING
        )

        start_time = datetime.now()

        try:
            # Step 1: Data processing
            self.logger.debug(f"Processing data from {file_path.name}")
            data_result = self.data_processor.process_file(file_path)
            result.data_result = data_result

            # Step 2: ML analysis
            if self.enable_ml and self.ml_models:
                result.status = ProcessingStatus.ML_ANALYSIS
                ml_result = self._perform_ml_analysis(data_result)
                result.ml_result = ml_result

            # Step 3: Database storage
            if self.enable_db and self.db_manager and self.current_run_id:
                result.status = ProcessingStatus.DB_SAVING
                self._save_to_database(data_result, result.ml_result)

            # Calculate processing time
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.status = ProcessingStatus.COMPLETED

            self.logger.info(f"✓ Completed {file_path.name} in {result.processing_time:.2f}s")

        except Exception as e:
            result.status = ProcessingStatus.FAILED
            result.error = str(e)
            result.processing_time = (datetime.now() - start_time).total_seconds()

            self.logger.error(f"✗ Failed {file_path.name}: {str(e)}")
            self.logger.debug(traceback.format_exc())

        return result

    def _perform_ml_analysis(self, data_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ML analysis on processed data."""
        ml_results = {
            'predictions': {},
            'insights': [],
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Analyze each track
            for track_id, track_data in data_result.get('tracks', {}).items():
                if 'sigma_results' not in track_data:
                    continue

                # Prepare features
                features = {
                    'sigma_gradient': track_data['sigma_results'].sigma_gradient,
                    'sigma_threshold': track_data['sigma_results'].sigma_threshold,
                    'linearity_spec': track_data['unit_properties'].linearity_spec or 0,
                    'travel_length': track_data['unit_properties'].travel_length or 0,
                    'unit_length': track_data['unit_properties'].unit_length or 0,
                    'model': data_result['file_info']['filename'].split('_')[0],
                    'timestamp': datetime.now()
                }

                # Get predictions
                track_predictions = {}

                # Optimal threshold
                threshold_pred = self.ml_models.predict_optimal_threshold(features)
                track_predictions['optimal_threshold'] = threshold_pred

                # Failure probability
                failure_pred = self.ml_models.predict_failure_probability(features)
                track_predictions['failure_risk'] = failure_pred

                # Drift detection
                drift_pred = self.ml_models.detect_manufacturing_drift(features)
                track_predictions['drift_detection'] = drift_pred

                ml_results['predictions'][track_id] = track_predictions

                # Generate insights
                if failure_pred['risk_level'] in ['HIGH', 'CRITICAL']:
                    ml_results['insights'].append(
                        f"Track {track_id}: High failure risk detected ({failure_pred['failure_probability']:.1%})"
                    )

                if drift_pred['is_drift']:
                    ml_results['insights'].append(
                        f"Track {track_id}: Manufacturing drift detected - {drift_pred['recommendation']}"
                    )

        except Exception as e:
            self.logger.error(f"ML analysis error: {str(e)}")
            ml_results['error'] = str(e)

        return ml_results

    def _save_to_database(self, data_result: Dict[str, Any], ml_result: Optional[Dict[str, Any]]):
        """Save results to database."""
        try:
            # Prepare combined result
            db_record = {
                'filename': data_result['file_info']['filename'],
                'model': data_result['file_info']['filename'].split('_')[0],
                'serial': data_result['file_info']['filename'].split('_')[1] if '_' in data_result['file_info'][
                    'filename'] else 'Unknown',
                'system': data_result['file_info']['system_type'],
                'timestamp': datetime.now()
            }

            # Add data from first track (or aggregate)
            first_track = list(data_result['tracks'].values())[0] if data_result['tracks'] else {}
            if first_track and 'sigma_results' in first_track:
                db_record.update({
                    'sigma_gradient': first_track['sigma_results'].sigma_gradient,
                    'sigma_threshold': first_track['sigma_results'].sigma_threshold,
                    'sigma_pass': first_track['sigma_results'].sigma_pass,
                    'linearity_pass': True,  # You may want to calculate this
                    'overall_status': 'Pass' if first_track['sigma_results'].sigma_pass else 'Fail'
                })

            # Add ML predictions
            if ml_result and 'predictions' in ml_result:
                first_pred = list(ml_result['predictions'].values())[0] if ml_result['predictions'] else {}
                if first_pred:
                    db_record.update({
                        'failure_probability': first_pred.get('failure_risk', {}).get('failure_probability', 0),
                        'risk_category': first_pred.get('failure_risk', {}).get('risk_level', 'Unknown'),
                        'ml_analysis': {
                            'prediction': first_pred.get('failure_risk', {}).get('failure_prediction', False),
                            'confidence': first_pred.get('failure_risk', {}).get('confidence', 0),
                            'model_name': 'ensemble_v2'
                        }
                    })

            # Save to database
            self.db_manager.save_file_result(self.current_run_id, db_record)

        except Exception as e:
            self.logger.error(f"Database save error: {str(e)}")

    def _generate_comprehensive_report(self, results: List[ProcessingResult],
                                       output_dir: Path,
                                       report_name: Optional[str] = None) -> Path:
        """Generate comprehensive Excel report."""
        self.logger.info("Generating comprehensive report...")

        # Prepare report data
        report_data = {
            'file_results': [],
            'ml_predictions': {},
            'processing_summary': {
                'total_files': len(results),
                'successful': sum(1 for r in results if r.status == ProcessingStatus.COMPLETED),
                'failed': sum(1 for r in results if r.status == ProcessingStatus.FAILED)
            }
        }

        # Convert results to report format
        for result in results:
            if result.status == ProcessingStatus.COMPLETED and result.data_result:
                file_entry = self._convert_to_report_format(result)
                if file_entry:
                    report_data['file_results'].append(file_entry)

        # Add ML insights
        if self.enable_ml:
            report_data['ml_predictions'] = self._aggregate_ml_insights(results)

        # Generate report
        if not report_name:
            report_name = f"laser_trim_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        report_path = output_dir / report_name

        try:
            self.report_generator.generate_report(
                report_data,
                str(report_path),
                include_ai_insights=bool(getattr(self.config, 'openai_api_key', None))
            )

            self.logger.info(f"✓ Report generated: {report_path}")
            self.metrics['reports_generated'] += 1

            return report_path

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return None

    def _convert_to_report_format(self, result: ProcessingResult) -> Optional[Dict[str, Any]]:
        """Convert processing result to report format."""
        if not result.data_result:
            return None

        data = result.data_result
        report_entry = {
            'filename': Path(result.file_path).name,
            'model': Path(result.file_path).stem.split('_')[0],
            'serial': Path(result.file_path).stem.split('_')[1] if '_' in Path(result.file_path).stem else 'Unknown',
            'system': data['file_info']['system_type']
        }

        # Handle multi-track files
        if len(data.get('tracks', {})) > 1:
            report_entry['tracks'] = {}
            for track_id, track_data in data['tracks'].items():
                track_entry = self._extract_track_data(track_data)
                if result.ml_result and 'predictions' in result.ml_result:
                    track_entry.update(self._extract_ml_data(
                        result.ml_result['predictions'].get(track_id, {})
                    ))
                report_entry['tracks'][track_id] = track_entry
        else:
            # Single track or combined data
            first_track = list(data['tracks'].values())[0] if data['tracks'] else {}
            report_entry.update(self._extract_track_data(first_track))

            if result.ml_result and 'predictions' in result.ml_result:
                first_pred = list(result.ml_result['predictions'].values())[0] if result.ml_result[
                    'predictions'] else {}
                report_entry.update(self._extract_ml_data(first_pred))

        return report_entry

    def _extract_track_data(self, track_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant data from track results."""
        extracted = {
            'status': 'Unknown',
            'sigma_gradient': 0,
            'sigma_threshold': 0,
            'sigma_pass': False,
            'linearity_pass': True
        }

        if 'sigma_results' in track_data:
            sigma = track_data['sigma_results']
            extracted.update({
                'sigma_gradient': sigma.sigma_gradient,
                'sigma_threshold': sigma.sigma_threshold,
                'sigma_pass': sigma.sigma_pass,
                'status': 'Pass' if sigma.sigma_pass else 'Fail'
            })

        if 'unit_properties' in track_data:
            props = track_data['unit_properties']
            extracted.update({
                'unit_length': props.unit_length or 0,
                'travel_length': props.travel_length or 0,
                'linearity_spec': props.linearity_spec or 0
            })

        return extracted

    def _extract_ml_data(self, ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ML prediction data."""
        extracted = {}

        if 'failure_risk' in ml_predictions:
            risk = ml_predictions['failure_risk']
            extracted.update({
                'failure_probability': risk.get('failure_probability', 0),
                'risk_category': risk.get('risk_level', 'Unknown')
            })

        if 'optimal_threshold' in ml_predictions:
            threshold = ml_predictions['optimal_threshold']
            extracted['optimal_threshold'] = threshold.get('optimal_threshold', 0)

        return extracted

    def _aggregate_ml_insights(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Aggregate ML insights across all results."""
        insights = {
            'overall_risk_assessment': 'Low',
            'drift_detected': False,
            'high_risk_units': 0,
            'recommendations': []
        }

        high_risk_count = 0
        drift_count = 0

        for result in results:
            if result.ml_result and 'predictions' in result.ml_result:
                for track_pred in result.ml_result['predictions'].values():
                    # Check failure risk
                    if 'failure_risk' in track_pred:
                        risk_level = track_pred['failure_risk'].get('risk_level', '')
                        if risk_level in ['HIGH', 'CRITICAL']:
                            high_risk_count += 1

                    # Check drift
                    if 'drift_detection' in track_pred:
                        if track_pred['drift_detection'].get('is_drift', False):
                            drift_count += 1

        insights['high_risk_units'] = high_risk_count
        insights['drift_detected'] = drift_count > 0

        # Overall risk assessment
        total_units = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
        if total_units > 0:
            risk_ratio = high_risk_count / total_units
            if risk_ratio > 0.2:
                insights['overall_risk_assessment'] = 'High'
            elif risk_ratio > 0.1:
                insights['overall_risk_assessment'] = 'Medium'

        # Generate recommendations
        if high_risk_count > 0:
            insights['recommendations'].append(
                f"Immediate inspection recommended for {high_risk_count} high-risk units"
            )

        if drift_count > 0:
            insights['recommendations'].append(
                "Manufacturing drift detected - review process parameters"
            )

        return insights

    def _create_processing_summary(self, results: List[ProcessingResult],
                                   processing_time: float,
                                   report_path: Optional[Path]) -> Dict[str, Any]:
        """Create summary of processing results."""
        successful = [r for r in results if r.status == ProcessingStatus.COMPLETED]
        failed = [r for r in results if r.status == ProcessingStatus.FAILED]

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'processing_time': processing_time,
            'average_time_per_file': processing_time / len(results) if results else 0,
            'report_path': str(report_path) if report_path else None,
            'metrics': dict(self.metrics),
            'results': []
        }

        # Add individual file results
        for result in results:
            file_summary = {
                'filename': Path(result.file_path).name,
                'status': result.status.value,
                'processing_time': result.processing_time
            }

            if result.error:
                file_summary['error'] = result.error

            if result.data_result and 'tracks' in result.data_result:
                # Add sigma results
                tracks_summary = {}
                for track_id, track_data in result.data_result['tracks'].items():
                    if 'sigma_results' in track_data:
                        tracks_summary[track_id] = {
                            'sigma_gradient': track_data['sigma_results'].sigma_gradient,
                            'sigma_pass': track_data['sigma_results'].sigma_pass
                        }
                file_summary['tracks'] = tracks_summary

            summary['results'].append(file_summary)

        return summary

    def _save_results_to_file(self, summary: Dict[str, Any], output_dir: Path):
        """Save processing summary to JSON file."""
        summary_path = output_dir / 'processing_summary.json'

        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            self.logger.info(f"✓ Summary saved to: {summary_path}")

        except Exception as e:
            self.logger.error(f"Failed to save summary: {str(e)}")

    def _update_metrics(self, results: List[ProcessingResult]):
        """Update performance metrics."""
        for result in results:
            if result.status == ProcessingStatus.COMPLETED:
                self.metrics['files_processed'] += 1
                if result.ml_result:
                    self.metrics['ml_predictions_made'] += 1
            elif result.status == ProcessingStatus.FAILED:
                self.metrics['files_failed'] += 1

            self.metrics['total_processing_time'] += result.processing_time

    # ==================== Utility Methods ====================

    def train_ml_models(self, historical_data_path: Optional[str] = None,
                        days_back: int = 90) -> Dict[str, Any]:
        """
        Train or retrain ML models.

        Args:
            historical_data_path: Path to historical data file
            days_back: Days of historical data to use from database

        Returns:
            Training results
        """
        if not self.ml_models:
            return {'error': 'ML models not enabled'}

        self.logger.info("Training ML models...")

        # Load historical data
        if historical_data_path and Path(historical_data_path).exists():
            # Load from file
            historical_data = pd.read_csv(historical_data_path)
            self.logger.info(f"Loaded {len(historical_data)} records from file")
        elif self.enable_db and self.db_manager:
            # Load from database
            historical_data = self.db_manager.get_historical_data(days_back=days_back)
            self.logger.info(f"Loaded {len(historical_data)} records from database")
        else:
            return {'error': 'No historical data source available'}

        if len(historical_data) < 100:
            return {'error': f'Insufficient data for training ({len(historical_data)} records)'}

        # Train models
        from ml_models import train_all_models
        results = train_all_models(self.ml_models, historical_data)

        self.logger.info(f"✓ ML models trained and saved to: {results.get('saved_version')}")

        return results

    def generate_trend_report(self, output_dir: Union[str, Path],
                              days_back: int = 30) -> Optional[Path]:
        """Generate trend analysis report."""
        if not self.enable_db or not self.trend_reporter:
            self.logger.error("Database not enabled - cannot generate trend report")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            report_path = self.trend_reporter.generate_comprehensive_report(
                output_dir, days_back
            )
            self.logger.info(f"✓ Trend report generated: {report_path}")
            return report_path

        except Exception as e:
            self.logger.error(f"Trend report generation failed: {str(e)}")
            return None

    def import_legacy_data(self, legacy_path: Union[str, Path]) -> Dict[str, Any]:
        """Import legacy data into the system."""
        if not self.enable_db or not self.data_migrator:
            return {'error': 'Database not enabled'}

        legacy_path = Path(legacy_path)

        if legacy_path.is_file():
            # Import single file
            if legacy_path.suffix == '.xlsx':
                return self.data_migrator.import_from_excel(legacy_path)
            elif legacy_path.suffix == '.json':
                return self.data_migrator.import_from_json(legacy_path)
            else:
                return {'error': f'Unsupported file type: {legacy_path.suffix}'}
        elif legacy_path.is_dir():
            # Import directory
            return self.data_migrator.import_legacy_data(legacy_path)
        else:
            return {'error': f'Path not found: {legacy_path}'}

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        status = {
            'components': {
                'data_processor': 'active',
                'ml_models': 'active' if self.enable_ml else 'disabled',
                'database': 'active' if self.enable_db else 'disabled',
                'report_generator': 'active',
                'parallel_processing': 'active' if self.enable_parallel else 'disabled'
            },
            'metrics': dict(self.metrics),
            'configuration': {
                'config_path': str(self.config_manager.config_path) if self.config_manager.config_path else 'default',
                'output_dir': str(self.config.output.output_dir)
            }
        }

        # Add ML model info
        if self.ml_models:
            status['ml_models'] = {
                'threshold_optimizer': self.ml_models.models['threshold_optimizer'] is not None,
                'failure_predictor': self.ml_models.models['failure_predictor'] is not None,
                'drift_detector': self.ml_models.models['drift_detector'] is not None
            }

        # Add database info
        if self.db_manager:
            db_stats = self.db_manager.get_database_stats()
            status['database'] = db_stats

        return status

    def cleanup(self):
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)

        self.logger.info("Orchestrator cleanup complete")


# ==================== Integration Functions ====================

def create_orchestrator(config_path: Optional[str] = None,
                        enable_all: bool = True) -> LaserTrimOrchestrator:
    """
    Create and configure the orchestrator.

    Args:
        config_path: Path to configuration file
        enable_all: Enable all features (ML, DB, parallel)

    Returns:
        Configured orchestrator instance
    """
    return LaserTrimOrchestrator(
        config_path=config_path,
        enable_parallel=enable_all,
        enable_ml=enable_all,
        enable_db=enable_all
    )


def process_with_full_pipeline(input_folder: str,
                               output_folder: Optional[str] = None,
                               config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Process folder with full pipeline - convenience function.

    Args:
        input_folder: Folder containing Excel files
        output_folder: Output folder for results
        config_path: Configuration file path

    Returns:
        Processing summary
    """
    orchestrator = create_orchestrator(config_path)

    try:
        return orchestrator.process_folder(
            input_folder,
            output_dir=output_folder,
            generate_report=True
        )
    finally:
        orchestrator.cleanup()


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example: Process a folder with full pipeline
    results = process_with_full_pipeline(
        input_folder="data/laser_trim_files",
        output_folder="output/analysis_results"
    )

    print(f"\nProcessing Complete!")
    print(f"Files processed: {results['successful']}/{results['total_files']}")
    print(f"Time taken: {results['processing_time']:.2f} seconds")
    if results['report_path']:
        print(f"Report saved to: {results['report_path']}")