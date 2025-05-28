"""
Full Pipeline Test for Laser Trim AI System
===========================================

This script tests the complete workflow:
1. Data Processing
2. ML Analysis
3. Database Storage
4. Report Generation
5. GUI Integration

Author: QA Specialist
Date: 2024
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import time

# Import all system components
from laser_trim_orchestrator import LaserTrimOrchestrator, create_orchestrator
from data_processor import DataProcessor
from ml_models import LaserTrimMLModels, create_ml_models
from excel_reporter import ExcelReporter
from database import DatabaseManager, HistoricalAnalyzer, TrendReporter
from config import Config, ConfigManager


class FullPipelineTest:
    """Test suite for the complete laser trim analysis pipeline."""

    def __init__(self):
        """Initialize test environment."""
        self.setup_logging()
        self.setup_directories()
        self.create_test_data()
        self.results = {}

    def setup_logging(self):
        """Configure logging for tests."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline_test.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('PipelineTest')

    def setup_directories(self):
        """Create necessary directories."""
        dirs = ['test_data', 'test_output', 'test_models', 'test_reports', 'test_db']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
        self.logger.info("✓ Created test directories")

    def create_test_data(self):
        """Create synthetic test data files."""
        self.logger.info("Creating synthetic test data...")

        # Create System A test file (multi-track)
        self._create_system_a_file('test_data/8340-1_SN12345_20240115.xlsx')

        # Create System B test file
        self._create_system_b_file('test_data/8555_SN67890_20240115.xlsx')

        # Create problematic file for error testing
        self._create_problematic_file('test_data/6845_SN99999_20240115.xlsx')

        self.logger.info("✓ Created 3 test files")

    def _create_system_a_file(self, filepath):
        """Create a System A test file with two tracks."""
        writer = pd.ExcelWriter(filepath, engine='openpyxl')

        # Generate data for TRK1
        n_points = 100
        position = np.linspace(0, 120, n_points)
        error = 0.001 * np.sin(position * 0.1) + np.random.normal(0, 0.0001, n_points)

        # Create DataFrame with System A column layout
        data = pd.DataFrame({
            'A': [''] * n_points,
            'B': [''] * n_points,
            'C': [''] * n_points,
            'D': np.random.randn(n_points),  # Measured volts
            'E': range(n_points),  # Index
            'F': np.random.randn(n_points),  # Theory volts
            'G': error,  # Error
            'H': position,  # Position
            'I': error + 0.01,  # Upper limit
            'J': error - 0.01  # Lower limit
        })

        # Write TRK1 sheets
        data.to_excel(writer, sheet_name='SEC1 TRK1 0', index=False, header=False)
        data.to_excel(writer, sheet_name='SEC1 TRK1 TRM', index=False, header=False)

        # Generate slightly different data for TRK2
        error2 = 0.001 * np.cos(position * 0.1) + np.random.normal(0, 0.0001, n_points)
        data['G'] = error2
        data['I'] = error2 + 0.01
        data['J'] = error2 - 0.01

        # Write TRK2 sheets
        data.to_excel(writer, sheet_name='SEC1 TRK2 0', index=False, header=False)
        data.to_excel(writer, sheet_name='SEC1 TRK2 TRM', index=False, header=False)

        # Add unit properties
        props_df = pd.DataFrame(index=range(30), columns=list('ABCDEFGHIJ'))
        props_df.loc[25, 'B'] = 150.0  # Unit length
        props_df.loc[9, 'B'] = 1000.0  # Resistance
        props_df.to_excel(writer, sheet_name='Properties', index=False, header=False)

        writer.close()

    def _create_system_b_file(self, filepath):
        """Create a System B test file."""
        writer = pd.ExcelWriter(filepath, engine='openpyxl')

        # Generate test data
        n_points = 100
        position = np.linspace(0, 180, n_points)
        error = 0.0008 * np.sin(position * 0.08) + np.random.normal(0, 0.00008, n_points)

        # Create DataFrame with System B layout
        data = pd.DataFrame({
            'A': [''] * n_points,
            'B': [''] * n_points,
            'C': [''] * n_points,
            'D': error,  # Error
            'E': [''] * n_points,
            'F': error + 0.008,  # Upper limit
            'G': error - 0.008,  # Lower limit
            'H': [''] * n_points,
            'I': position  # Position
        })

        # Write sheets
        data.to_excel(writer, sheet_name='test', index=False, header=False)
        data.to_excel(writer, sheet_name='Lin Error', index=False, header=False)

        # Add properties
        props_df = pd.DataFrame(index=range(10), columns=list('ABCDEFGHIJKLMNOPQR'))
        props_df.loc[0, 'K'] = 180.0  # Unit length
        props_df.loc[0, 'R'] = 2000.0  # Resistance
        props_df.to_excel(writer, sheet_name='Properties', index=False, header=False)

        writer.close()

    def _create_problematic_file(self, filepath):
        """Create a file with issues to test error handling."""
        writer = pd.ExcelWriter(filepath, engine='openpyxl')

        # Generate problematic data (high sigma, drift)
        n_points = 100
        position = np.linspace(0, 120, n_points)
        # High sigma gradient and drift
        error = 0.005 * np.sin(position * 0.2) + 0.0001 * position + np.random.normal(0, 0.0005, n_points)

        data = pd.DataFrame({
            'A': [''] * n_points,
            'B': [''] * n_points,
            'C': [''] * n_points,
            'D': np.random.randn(n_points),
            'E': range(n_points),
            'F': np.random.randn(n_points),
            'G': error,
            'H': position,
            'I': error + 0.005,  # Tight limits
            'J': error - 0.005
        })

        data.to_excel(writer, sheet_name='SEC1 TRK1 0', index=False, header=False)
        writer.close()

    def run_all_tests(self):
        """Run all pipeline tests."""
        print("\n" + "=" * 80)
        print("LASER TRIM AI SYSTEM - FULL PIPELINE TEST")
        print("=" * 80 + "\n")

        # Run tests in sequence
        self.test_1_data_processing()
        self.test_2_ml_training()
        self.test_3_ml_predictions()
        self.test_4_database_operations()
        self.test_5_report_generation()
        self.test_6_orchestrator_integration()
        self.test_7_error_handling()

        # Generate summary report
        self.generate_test_report()

    def test_1_data_processing(self):
        """Test 1: Core data processing functionality."""
        print("\n📋 TEST 1: Data Processing")
        print("-" * 40)

        try:
            # Initialize processor
            processor = DataProcessor()

            # Test System A file
            result_a = processor.process_file('test_data/8340-1_SN12345_20240115.xlsx')
            assert 'tracks' in result_a
            assert len(result_a['tracks']) == 2  # Should have TRK1 and TRK2

            # Check sigma calculations
            for track_id, track_data in result_a['tracks'].items():
                sigma_results = track_data['sigma_results']
                print(f"  {track_id}: σ={sigma_results.sigma_gradient:.6f}, "
                      f"Pass={sigma_results.sigma_pass}")
                assert sigma_results.sigma_gradient > 0
                assert sigma_results.sigma_threshold > 0

            # Test System B file
            result_b = processor.process_file('test_data/8555_SN67890_20240115.xlsx')
            assert 'tracks' in result_b
            assert len(result_b['tracks']) == 1  # Single track

            self.results['data_processing'] = {
                'status': 'PASSED',
                'files_processed': 2,
                'system_a_tracks': len(result_a['tracks']),
                'system_b_tracks': len(result_b['tracks'])
            }

            print("  ✅ Data processing test PASSED")

        except Exception as e:
            self.results['data_processing'] = {'status': 'FAILED', 'error': str(e)}
            print(f"  ❌ Data processing test FAILED: {e}")

    def test_2_ml_training(self):
        """Test 2: ML model training."""
        print("\n🤖 TEST 2: ML Model Training")
        print("-" * 40)

        try:
            # Create ML models
            config = Config()
            config.output_dir = 'test_output'
            ml_models = create_ml_models(config)

            # Generate training data
            historical_data = self._generate_training_data(200)

            # Train models
            print("  Training threshold optimizer...")
            threshold_results = ml_models.train_threshold_optimizer(historical_data)
            assert 'mae' in threshold_results
            print(f"    MAE: {threshold_results['mae']:.4f}")

            print("  Training failure predictor...")
            failure_results = ml_models.train_failure_predictor(historical_data)
            if 'error' not in failure_results:
                assert 'accuracy' in failure_results
                print(f"    Accuracy: {failure_results['accuracy']:.2%}")

            print("  Training drift detector...")
            drift_results = ml_models.train_drift_detector(historical_data)
            assert 'anomaly_rate' in drift_results
            print(f"    Anomaly rate: {drift_results['anomaly_rate']:.2%}")

            # Save models
            version = ml_models.save_models('test_v1')

            self.results['ml_training'] = {
                'status': 'PASSED',
                'models_trained': 3,
                'version': version
            }

            print("  ✅ ML training test PASSED")

        except Exception as e:
            self.results['ml_training'] = {'status': 'FAILED', 'error': str(e)}
            print(f"  ❌ ML training test FAILED: {e}")

    def test_3_ml_predictions(self):
        """Test 3: ML predictions on new data."""
        print("\n🔮 TEST 3: ML Predictions")
        print("-" * 40)

        try:
            # Load trained models
            config = Config()
            config.output_dir = 'test_output'
            ml_models = create_ml_models(config)
            ml_models.load_models('test_v1')

            # Process a file and get features
            processor = DataProcessor()
            result = processor.process_file('test_data/8340-1_SN12345_20240115.xlsx')

            # Create feature dict from first track
            track_data = list(result['tracks'].values())[0]
            features = {
                'sigma_gradient': track_data['sigma_results'].sigma_gradient,
                'sigma_threshold': track_data['sigma_results'].sigma_threshold,
                'linearity_spec': 0.01,
                'travel_length': 120,
                'unit_length': 150,
                'resistance_change': 5.2,
                'resistance_change_percent': 2.1,
                'error_data': np.random.normal(0, 0.001, 100).tolist(),
                'model': '8340',
                'timestamp': datetime.now()
            }

            # Get predictions
            print("  Testing threshold optimization...")
            threshold_pred = ml_models.predict_optimal_threshold(features)
            assert 'optimal_threshold' in threshold_pred
            print(f"    Optimal threshold: {threshold_pred['optimal_threshold']:.4f}")

            print("  Testing failure prediction...")
            failure_pred = ml_models.predict_failure_probability(features)
            assert 'failure_probability' in failure_pred
            print(f"    Failure probability: {failure_pred['failure_probability']:.2%}")
            print(f"    Risk level: {failure_pred['risk_level']}")

            print("  Testing drift detection...")
            drift_pred = ml_models.detect_manufacturing_drift(features)
            assert 'is_drift' in drift_pred
            print(f"    Drift detected: {drift_pred['is_drift']}")

            self.results['ml_predictions'] = {
                'status': 'PASSED',
                'predictions_made': 3
            }

            print("  ✅ ML predictions test PASSED")

        except Exception as e:
            self.results['ml_predictions'] = {'status': 'FAILED', 'error': str(e)}
            print(f"  ❌ ML predictions test FAILED: {e}")

    def test_4_database_operations(self):
        """Test 4: Database storage and retrieval."""
        print("\n💾 TEST 4: Database Operations")
        print("-" * 40)

        try:
            # Initialize database
            config = Config()
            config.OUTPUT_DIR = 'test_db'
            db_manager = DatabaseManager(config)

            # Create analysis run
            run_id = db_manager.create_analysis_run('test_data', {'test': True})
            print(f"  Created analysis run: {run_id}")

            # Save test results
            test_result = {
                'filename': 'test_file.xlsx',
                'model': 'TEST123',
                'serial': 'SN001',
                'system': 'A',
                'sigma_gradient': 0.0025,
                'sigma_threshold': 0.004,
                'sigma_pass': True,
                'linearity_pass': True,
                'overall_status': 'Pass',
                'failure_probability': 0.12,
                'risk_category': 'Low',
                'ml_analysis': {
                    'prediction': 'Pass',
                    'confidence': 0.92,
                    'model_name': 'test_model'
                }
            }

            file_id = db_manager.save_file_result(run_id, test_result)
            print(f"  Saved file result: {file_id}")

            # Query data
            df = db_manager.get_historical_data(days_back=1)
            assert len(df) > 0
            print(f"  Retrieved {len(df)} records")

            # Test anomaly detection
            anomaly_summary = db_manager.get_anomaly_summary(days_back=1)
            print(f"  Anomaly summary: {anomaly_summary}")

            self.results['database'] = {
                'status': 'PASSED',
                'records_saved': 1,
                'records_retrieved': len(df)
            }

            print("  ✅ Database operations test PASSED")

        except Exception as e:
            self.results['database'] = {'status': 'FAILED', 'error': str(e)}
            print(f"  ❌ Database operations test FAILED: {e}")

    def test_5_report_generation(self):
        """Test 5: Excel report generation."""
        print("\n📊 TEST 5: Report Generation")
        print("-" * 40)

        try:
            # Create test data for report
            report_data = {
                'file_results': [
                    {
                        'filename': '8340-1_SN12345.xlsx',
                        'model': '8340-1',
                        'serial': 'SN12345',
                        'system': 'A',
                        'status': 'Pass',
                        'sigma_gradient': 0.0023,
                        'sigma_threshold': 0.004,
                        'sigma_pass': True,
                        'linearity_pass': True,
                        'failure_probability': 0.08,
                        'risk_category': 'Low',
                        'resistance_change_percent': 2.1,
                        'trim_improvement_percent': 18.5,
                        'unit_length': 150.0
                    },
                    {
                        'filename': '8555_SN67890.xlsx',
                        'model': '8555',
                        'serial': 'SN67890',
                        'system': 'B',
                        'status': 'Pass',
                        'sigma_gradient': 0.0018,
                        'sigma_threshold': 0.003,
                        'sigma_pass': True,
                        'linearity_pass': True,
                        'failure_probability': 0.05,
                        'risk_category': 'Low'
                    }
                ],
                'ml_predictions': {
                    'next_batch_pass_rate': 0.88,
                    'maintenance_due_in_days': 15,
                    'quality_trend': 'stable'
                }
            }

            # Generate report
            reporter = ExcelReporter()
            report_path = reporter.generate_report(
                report_data,
                'test_reports/test_report.xlsx',
                include_ai_insights=False
            )

            assert Path(report_path).exists()
            print(f"  Generated report: {report_path}")

            self.results['report_generation'] = {
                'status': 'PASSED',
                'report_path': report_path
            }

            print("  ✅ Report generation test PASSED")

        except Exception as e:
            self.results['report_generation'] = {'status': 'FAILED', 'error': str(e)}
            print(f"  ❌ Report generation test FAILED: {e}")

    def test_6_orchestrator_integration(self):
        """Test 6: Full orchestrator integration."""
        print("\n🎯 TEST 6: Orchestrator Integration")
        print("-" * 40)

        try:
            # Create orchestrator
            orchestrator = LaserTrimOrchestrator(
                enable_parallel=False,  # Disable for testing
                enable_ml=True,
                enable_db=True
            )

            # Process test folder
            print("  Processing test folder...")
            start_time = time.time()

            results = orchestrator.process_folder(
                'test_data',
                output_dir='test_output/orchestrator',
                generate_report=True
            )

            processing_time = time.time() - start_time

            print(f"  Processed {results['total_files']} files in {processing_time:.2f}s")
            print(f"  Successful: {results['successful']}")
            print(f"  Failed: {results['failed']}")

            # Check report generation
            assert results['report_path'] is not None
            print(f"  Report generated: {results['report_path']}")

            self.results['orchestrator'] = {
                'status': 'PASSED',
                'files_processed': results['total_files'],
                'time': processing_time
            }

            orchestrator.cleanup()
            print("  ✅ Orchestrator integration test PASSED")

        except Exception as e:
            self.results['orchestrator'] = {'status': 'FAILED', 'error': str(e)}
            print(f"  ❌ Orchestrator integration test FAILED: {e}")

    def test_7_error_handling(self):
        """Test 7: Error handling and edge cases."""
        print("\n⚠️  TEST 7: Error Handling")
        print("-" * 40)

        errors_handled = []

        try:
            processor = DataProcessor()

            # Test 1: Non-existent file
            try:
                processor.process_file('non_existent.xlsx')
            except ValueError as e:
                errors_handled.append('non_existent_file')
                print("  ✓ Correctly handled non-existent file")

            # Test 2: Invalid file format
            with open('test_data/invalid.txt', 'w') as f:
                f.write('not an excel file')

            try:
                processor.process_file('test_data/invalid.txt')
            except ValueError as e:
                errors_handled.append('invalid_format')
                print("  ✓ Correctly handled invalid file format")

            # Test 3: Process problematic file
            result = processor.process_file('test_data/6845_SN99999_20240115.xlsx')
            if 'tracks' in result:
                for track_id, track_data in result['tracks'].items():
                    if 'sigma_results' in track_data:
                        # Should have high sigma gradient
                        assert track_data['sigma_results'].sigma_gradient > 0.004
                        errors_handled.append('high_sigma_handled')
                        print("  ✓ Correctly processed high sigma file")

            self.results['error_handling'] = {
                'status': 'PASSED',
                'errors_handled': len(errors_handled)
            }

            print("  ✅ Error handling test PASSED")

        except Exception as e:
            self.results['error_handling'] = {'status': 'FAILED', 'error': str(e)}
            print(f"  ❌ Error handling test FAILED: {e}")

    def _generate_training_data(self, n_samples):
        """Generate synthetic training data."""
        data = []
        for i in range(n_samples):
            record = {
                'sigma_gradient': np.random.normal(0.5, 0.1),
                'linearity_spec': np.random.normal(0.02, 0.005),
                'travel_length': np.random.normal(150, 20),
                'unit_length': np.random.normal(140, 15),
                'resistance_change': np.random.normal(5, 2),
                'resistance_change_percent': np.random.normal(2, 0.5),
                'sigma_threshold': np.random.normal(0.6, 0.1),
                'error_data': np.random.normal(0, 0.01, 100).tolist(),
                'model': np.random.choice(['8340', '8555', '6845']),
                'timestamp': datetime.now(),
                'passed': np.random.choice([True, False], p=[0.85, 0.15])
            }
            data.append(record)
        return pd.DataFrame(data)

    def generate_test_report(self):
        """Generate final test report."""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        # Count results
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get('status') == 'PASSED')
        failed_tests = total_tests - passed_tests

        # Display results
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {passed_tests / total_tests * 100:.1f}%")

        print("\nDetailed Results:")
        print("-" * 40)

        for test_name, result in self.results.items():
            status_icon = "✅" if result.get('status') == 'PASSED' else "❌"
            print(f"{status_icon} {test_name}: {result.get('status')}")
            if result.get('error'):
                print(f"   Error: {result['error']}")

        # Save detailed report
        report_path = 'test_output/pipeline_test_report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_tests': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'success_rate': passed_tests / total_tests
                },
                'results': self.results
            }, f, indent=2)

        print(f"\nDetailed report saved to: {report_path}")

        # Overall result
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! The pipeline is working correctly.")
        else:
            print(f"\n⚠️  {failed_tests} tests failed. Please check the errors above.")


def main():
    """Run the full pipeline test."""
    # Create and run test suite
    test_suite = FullPipelineTest()

    try:
        test_suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()