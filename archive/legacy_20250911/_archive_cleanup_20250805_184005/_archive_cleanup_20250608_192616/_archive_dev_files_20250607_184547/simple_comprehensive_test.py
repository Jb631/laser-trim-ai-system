#!/usr/bin/env python3
"""
Simplified Comprehensive Test Script for Laser Trim Analyzer v2

This script performs extensive testing without external dependencies like psutil.
"""

import asyncio
import logging
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from laser_trim_analyzer.core.config import Config
from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.utils.logging_utils import setup_logging
from laser_trim_analyzer.core.models import AnalysisStatus, SystemType


class SimpleTestSuite:
    """Simplified test suite for comprehensive testing."""
    
    def __init__(self):
        self.results = {
            'single_file_tests': [],
            'batch_tests': [],
            'edge_case_tests': [],
            'database_tests': [],
            'performance_metrics': {},
            'summary': {}
        }
        self.test_dir = Path(__file__).parent / 'test_files' / 'System A test files'
        self.output_dir = Path(__file__).parent / 'test_outputs'
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        setup_logging(log_file=self.output_dir / 'test_log.log')
        
        # Initialize components
        self.config = Config()
        self.db_manager = None
        self.processor = None
        
    async def setup(self):
        """Initialize test environment."""
        self.logger.info("Setting up test environment...")
        
        # Initialize database
        try:
            self.db_manager = DatabaseManager(self.config)
            await self.db_manager.initialize()
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            self.db_manager = None
        
        # Initialize processor without ML for simplicity
        self.processor = LaserTrimProcessor(
            self.config, 
            self.db_manager,
            ml_predictor=None  # Skip ML for this test
        )
        
    async def run_all_tests(self):
        """Run all test categories."""
        start_time = time.time()
        
        await self.setup()
        
        # 1. Test single file processing
        await self.test_single_file_processing()
        
        # 2. Test batch processing
        await self.test_batch_processing()
        
        # 3. Test edge cases
        await self.test_edge_cases()
        
        # 4. Test database operations
        await self.test_database_operations()
        
        # 5. Generate summary report
        self.generate_summary_report(time.time() - start_time)
        
        # Cleanup
        await self.cleanup()
        
    async def test_single_file_processing(self):
        """Test single file processing with various product numbers."""
        self.logger.info("\n=== Testing Single File Processing ===")
        
        # Select diverse test files
        test_files = [
            # Different product numbers from various years and systems
            ("2475-10_19_TEST DATA_11-16-2023_6-10 PM.xls", "2475 series - Recent"),
            ("5409A_5_TEST DATA_8-22-2017_9-02 AM.xls", "5409 series - 2017"),
            ("6126_74_TEST DATA_2-24-2025_12-18 PM.xls", "6126 series - 2025"),
            ("7063-A_17_TEST DATA_2-19-2021_9-15 AM.xls", "7063 series - Multi-variant"),
            ("8275A_10_TEST DATA_5-21-2025_4-48 PM.xls", "8275 series - Latest"),
            ("8755-1_31_TEST DATA_6-13-2023_9-41 AM.xls", "8755 series - Complex"),
            # Edge cases
            ("8232-1 BLUE_39_TEST DATA_8-12-2019_4-32 PM.xls", "File with spaces"),
            ("8394-6_101 REDUNDANT_TEST DATA_3-25-2022_9-32 AM.xls", "Special naming"),
            ("7953B_20_TEST DATA_12-22-2015_7-30 AM.xls", "Old file - 2015"),
            ("8736_Shop18_initial lin.xls", "Non-standard filename"),
        ]
        
        for filename, description in test_files:
            file_path = self.test_dir / filename
            if not file_path.exists():
                self.logger.warning(f"Test file not found: {filename}")
                self.results['single_file_tests'].append({
                    'file': filename,
                    'description': description,
                    'status': 'skipped',
                    'reason': 'File not found'
                })
                continue
                
            self.logger.info(f"Testing {description}: {filename}")
            
            try:
                start_time = time.time()
                
                # Process file
                result = await self.processor.process_file(
                    file_path,
                    output_dir=self.output_dir / 'single_files',
                    progress_callback=lambda msg, prog: self.logger.debug(f"{msg}: {prog:.1%}")
                )
                
                processing_time = time.time() - start_time
                
                # Collect results
                test_result = {
                    'file': filename,
                    'description': description,
                    'status': 'success',
                    'processing_time': processing_time,
                    'overall_status': result.overall_status.value,
                    'validation_status': result.overall_validation_status.value,
                    'system_type': result.metadata.system_type.value,
                    'model': result.metadata.model,
                    'tracks': len(result.tracks),
                    'warnings': len(result.validation_warnings),
                    'errors': len(result.processing_errors)
                }
                
                # Extract key metrics from primary track
                if result.tracks:
                    primary_track = next(iter(result.tracks.values()))
                    test_result.update({
                        'sigma_gradient': round(primary_track.sigma_analysis.sigma_gradient, 4),
                        'sigma_pass': primary_track.sigma_analysis.sigma_pass,
                        'linearity_error': round(primary_track.linearity_analysis.final_linearity_error_shifted, 4),
                        'linearity_pass': primary_track.linearity_analysis.linearity_pass,
                        'resistance_change': round(primary_track.resistance_analysis.resistance_change_percent or 0, 2),
                        'validation_grade': primary_track.sigma_analysis.validation_result.validation_grade if hasattr(primary_track.sigma_analysis, 'validation_result') and primary_track.sigma_analysis.validation_result else 'N/A'
                    })
                
                self.results['single_file_tests'].append(test_result)
                self.logger.info(f"✓ Processed successfully in {processing_time:.2f}s - Status: {result.overall_status.value}")
                
            except Exception as e:
                self.logger.error(f"✗ Failed to process {filename}: {e}")
                self.results['single_file_tests'].append({
                    'file': filename,
                    'description': description,
                    'status': 'failed',
                    'error': str(e),
                    'error_type': type(e).__name__
                })
    
    async def test_batch_processing(self):
        """Test batch processing with different batch sizes."""
        self.logger.info("\n=== Testing Batch Processing ===")
        
        batch_tests = [
            ("Small batch (5 files)", 5),
            ("Medium batch (15 files)", 15),
            ("Large batch (30 files)", 30),
        ]
        
        # Get all available test files
        all_files = list(self.test_dir.glob("*.xls"))[:50]  # Limit to 50 files
        
        for test_name, batch_size in batch_tests:
            if len(all_files) < batch_size:
                self.logger.warning(f"Not enough files for {test_name}")
                continue
                
            self.logger.info(f"Testing {test_name}")
            
            # Select diverse files for batch
            batch_files = all_files[:batch_size]
            
            try:
                start_time = time.time()
                
                # Process batch
                results = await self.processor.process_batch(
                    batch_files,
                    output_dir=self.output_dir / 'batch_processing',
                    progress_callback=lambda msg, prog: self.logger.debug(f"{msg}: {prog:.1%}"),
                    max_workers=4  # Limit workers for testing
                )
                
                processing_time = time.time() - start_time
                
                # Analyze results
                successful = len(results)
                failed = batch_size - successful
                pass_count = sum(1 for r in results.values() if r.overall_status == AnalysisStatus.PASS)
                warning_count = sum(1 for r in results.values() if r.overall_status == AnalysisStatus.WARNING)
                fail_count = sum(1 for r in results.values() if r.overall_status == AnalysisStatus.FAIL)
                
                # Calculate average metrics
                avg_sigma = 0
                avg_linearity = 0
                count = 0
                for result in results.values():
                    if result.tracks:
                        track = next(iter(result.tracks.values()))
                        avg_sigma += track.sigma_analysis.sigma_gradient
                        avg_linearity += track.linearity_analysis.final_linearity_error_shifted
                        count += 1
                
                if count > 0:
                    avg_sigma /= count
                    avg_linearity /= count
                
                batch_result = {
                    'test_name': test_name,
                    'batch_size': batch_size,
                    'status': 'success',
                    'processing_time': processing_time,
                    'files_per_second': successful / processing_time if processing_time > 0 else 0,
                    'successful': successful,
                    'failed': failed,
                    'pass_count': pass_count,
                    'warning_count': warning_count,
                    'fail_count': fail_count,
                    'avg_sigma_gradient': round(avg_sigma, 4),
                    'avg_linearity_error': round(avg_linearity, 4)
                }
                
                self.results['batch_tests'].append(batch_result)
                self.logger.info(f"✓ Batch processed in {processing_time:.2f}s - {successful}/{batch_size} successful ({batch_result['files_per_second']:.2f} files/sec)")
                
            except Exception as e:
                self.logger.error(f"✗ Batch processing failed: {e}")
                self.results['batch_tests'].append({
                    'test_name': test_name,
                    'batch_size': batch_size,
                    'status': 'failed',
                    'error': str(e),
                    'error_type': type(e).__name__
                })
    
    async def test_edge_cases(self):
        """Test edge cases and error handling."""
        self.logger.info("\n=== Testing Edge Cases ===")
        
        # Test 1: Non-existent file
        self.logger.info("Testing: Non-existent file")
        try:
            await self.processor.process_file(Path("non_existent_file.xls"))
            self.results['edge_case_tests'].append({
                'name': 'Non-existent file',
                'status': 'failed',
                'error': 'Should have raised FileNotFoundError'
            })
        except FileNotFoundError:
            self.results['edge_case_tests'].append({
                'name': 'Non-existent file',
                'status': 'success',
                'result': 'FileNotFoundError raised as expected'
            })
            self.logger.info("✓ FileNotFoundError raised as expected")
        except Exception as e:
            self.results['edge_case_tests'].append({
                'name': 'Non-existent file',
                'status': 'failed',
                'error': f"Wrong error type: {type(e).__name__}"
            })
            self.logger.error(f"✗ Wrong error type: {type(e).__name__}")
        
        # Test 2: Empty batch
        self.logger.info("Testing: Empty batch")
        try:
            result = await self.processor.process_batch([])
            if result == {}:
                self.results['edge_case_tests'].append({
                    'name': 'Empty batch',
                    'status': 'success',
                    'result': 'Empty dict returned as expected'
                })
                self.logger.info("✓ Empty batch handled correctly")
            else:
                self.results['edge_case_tests'].append({
                    'name': 'Empty batch',
                    'status': 'failed',
                    'error': f"Unexpected result: {result}"
                })
        except Exception as e:
            self.results['edge_case_tests'].append({
                'name': 'Empty batch',
                'status': 'failed',
                'error': str(e)
            })
            self.logger.error(f"✗ Empty batch error: {e}")
        
        # Test 3: Invalid file extension
        self.logger.info("Testing: Invalid file extension")
        try:
            # Create a temporary text file
            temp_file = self.output_dir / "test.txt"
            temp_file.write_text("Not an Excel file")
            
            await self.processor.process_file(temp_file)
            self.results['edge_case_tests'].append({
                'name': 'Invalid file extension',
                'status': 'failed',
                'error': 'Should have raised ValidationError'
            })
            
            # Cleanup
            temp_file.unlink()
            
        except Exception as e:
            self.results['edge_case_tests'].append({
                'name': 'Invalid file extension',
                'status': 'success',
                'result': f'{type(e).__name__} raised as expected'
            })
            self.logger.info(f"✓ {type(e).__name__} raised for invalid extension")
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()
        
        # Test 4: Duplicate files in batch
        self.logger.info("Testing: Duplicate files in batch")
        try:
            test_file = next(self.test_dir.glob("*.xls"))
            duplicate_batch = [test_file, test_file, test_file]  # Same file 3 times
            
            results = await self.processor.process_batch(duplicate_batch)
            
            self.results['edge_case_tests'].append({
                'name': 'Duplicate files in batch',
                'status': 'success',
                'result': f'Processed {len(results)} unique files from {len(duplicate_batch)} inputs'
            })
            self.logger.info(f"✓ Duplicate handling: {len(results)} results from {len(duplicate_batch)} inputs")
            
        except Exception as e:
            self.results['edge_case_tests'].append({
                'name': 'Duplicate files in batch',
                'status': 'failed',
                'error': str(e)
            })
            self.logger.error(f"✗ Duplicate handling failed: {e}")
    
    async def test_database_operations(self):
        """Test database functionality."""
        self.logger.info("\n=== Testing Database Operations ===")
        
        if not self.db_manager:
            self.logger.warning("Database manager not available, skipping tests")
            self.results['database_tests'].append({
                'test': 'Database availability',
                'status': 'skipped',
                'reason': 'Database manager not initialized'
            })
            return
        
        # Test 1: Check database tables exist
        try:
            # This will create tables if they don't exist
            await self.db_manager.initialize()
            self.results['database_tests'].append({
                'test': 'Database initialization',
                'status': 'success'
            })
            self.logger.info("✓ Database initialized successfully")
        except Exception as e:
            self.results['database_tests'].append({
                'test': 'Database initialization',
                'status': 'failed',
                'error': str(e)
            })
            self.logger.error(f"✗ Database initialization failed: {e}")
            return
        
        # Test 2: Save and retrieve analysis
        try:
            # Process a small file first
            test_files = list(self.test_dir.glob("*.xls"))[:1]
            if test_files:
                result = await self.processor.process_file(test_files[0])
                
                # Save to database
                saved_id = await self.db_manager.save_analysis(result)
                
                self.results['database_tests'].append({
                    'test': 'Save analysis',
                    'status': 'success',
                    'saved_id': saved_id
                })
                self.logger.info(f"✓ Saved analysis with ID: {saved_id}")
                
                # Retrieve analysis
                retrieved = await self.db_manager.get_analysis(saved_id)
                
                if retrieved:
                    self.results['database_tests'].append({
                        'test': 'Retrieve analysis',
                        'status': 'success',
                        'model': retrieved.metadata.model
                    })
                    self.logger.info("✓ Retrieved analysis successfully")
                else:
                    self.results['database_tests'].append({
                        'test': 'Retrieve analysis',
                        'status': 'failed',
                        'error': 'No data retrieved'
                    })
            
        except Exception as e:
            self.logger.error(f"✗ Database save/retrieve failed: {e}")
            self.results['database_tests'].append({
                'test': 'Save/Retrieve analysis',
                'status': 'failed',
                'error': str(e)
            })
        
        # Test 3: Query recent analyses
        try:
            recent = await self.db_manager.get_recent_analyses(limit=5)
            self.results['database_tests'].append({
                'test': 'Query recent analyses',
                'status': 'success',
                'count': len(recent)
            })
            self.logger.info(f"✓ Retrieved {len(recent)} recent analyses")
        except Exception as e:
            self.results['database_tests'].append({
                'test': 'Query recent analyses',
                'status': 'failed',
                'error': str(e)
            })
            self.logger.error(f"✗ Query recent analyses failed: {e}")
    
    def generate_summary_report(self, total_time: float):
        """Generate comprehensive test summary report."""
        self.logger.info("\n=== Generating Test Summary ===")
        
        # Calculate summary statistics
        total_tests = 0
        successful_tests = 0
        failed_tests = 0
        
        # Count results from each category
        for category in ['single_file_tests', 'batch_tests', 'edge_case_tests', 'database_tests']:
            tests = self.results.get(category, [])
            category_success = sum(1 for t in tests if t.get('status') == 'success')
            category_failed = sum(1 for t in tests if t.get('status') == 'failed')
            category_skipped = sum(1 for t in tests if t.get('status') == 'skipped')
            
            total_tests += len(tests)
            successful_tests += category_success
            failed_tests += category_failed
        
        # Create summary
        self.results['summary'] = {
            'total_time': round(total_time, 2),
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': round((successful_tests / total_tests * 100) if total_tests > 0 else 0, 1)
        }
        
        # Save results to JSON
        report_path = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*70)
        print("COMPREHENSIVE TEST SUMMARY - LASER TRIM ANALYZER V2")
        print("="*70)
        print(f"Total Test Time: {total_time:.2f} seconds")
        print(f"Total Tests Run: {total_tests}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Failed Tests: {failed_tests}")
        print(f"Success Rate: {self.results['summary']['success_rate']:.1f}%")
        print("\n--- Single File Processing ---")
        single_tests = self.results.get('single_file_tests', [])
        if single_tests:
            print(f"Files Tested: {len(single_tests)}")
            print(f"Success: {sum(1 for t in single_tests if t['status'] == 'success')}")
            print(f"Failed: {sum(1 for t in single_tests if t['status'] == 'failed')}")
            print(f"Average Processing Time: {sum(t.get('processing_time', 0) for t in single_tests if t.get('status') == 'success') / max(1, sum(1 for t in single_tests if t.get('status') == 'success')):.2f}s")
            
            # Show some example results
            print("\nSample Results:")
            for test in single_tests[:3]:
                if test['status'] == 'success':
                    print(f"  - {test['description']}: {test['overall_status']} (σ={test.get('sigma_gradient', 'N/A')}, Lin={test.get('linearity_error', 'N/A')})")
        
        print("\n--- Batch Processing ---")
        batch_tests = self.results.get('batch_tests', [])
        if batch_tests:
            for test in batch_tests:
                if test['status'] == 'success':
                    print(f"  - {test['test_name']}: {test['successful']}/{test['batch_size']} files @ {test['files_per_second']:.2f} files/sec")
                else:
                    print(f"  - {test['test_name']}: FAILED - {test.get('error', 'Unknown error')}")
        
        print("\n--- Edge Cases ---")
        edge_tests = self.results.get('edge_case_tests', [])
        for test in edge_tests:
            print(f"  - {test['name']}: {test['status'].upper()}")
        
        print("\n--- Database Operations ---")
        db_tests = self.results.get('database_tests', [])
        for test in db_tests:
            print(f"  - {test['test']}: {test['status'].upper()}")
        
        print(f"\nDetailed report saved to: {report_path}")
        print("="*70)
        
        # Print any notable errors
        all_errors = []
        for category in ['single_file_tests', 'batch_tests', 'edge_case_tests', 'database_tests']:
            for test in self.results.get(category, []):
                if test.get('status') == 'failed' and 'error' in test:
                    all_errors.append(f"{category}: {test.get('file', test.get('name', 'Unknown'))} - {test['error']}")
        
        if all_errors:
            print("\n--- Notable Errors ---")
            for error in all_errors[:5]:  # Show first 5 errors
                print(f"  • {error}")
            if len(all_errors) > 5:
                print(f"  ... and {len(all_errors) - 5} more errors")
    
    async def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up test resources...")
        
        if self.db_manager:
            await self.db_manager.close()


async def main():
    """Main test execution function."""
    print("Starting Comprehensive Test Suite for Laser Trim Analyzer v2")
    print("This will test various Excel files and app functionality")
    print("=" * 70)
    
    test_suite = SimpleTestSuite()
    
    try:
        await test_suite.run_all_tests()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)