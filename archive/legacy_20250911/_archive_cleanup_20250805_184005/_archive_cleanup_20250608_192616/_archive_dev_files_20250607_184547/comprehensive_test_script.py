#!/usr/bin/env python3
"""
Comprehensive Test Script for Laser Trim Analyzer v2

This script performs extensive testing of the laser trim analyzer with various
Excel files to ensure full functionality, including:
- Single file processing
- Batch processing
- Different product numbers
- Edge cases and error handling
- Database operations
- GUI functionality
- Performance metrics
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
import psutil
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from laser_trim_analyzer.core.config import Config
from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.ml.ml_manager import MLManager
from laser_trim_analyzer.utils.logging_utils import setup_logging
from laser_trim_analyzer.core.models import AnalysisStatus, SystemType
from laser_trim_analyzer.gui.modern_gui import LaserTrimAnalyzerGUI
import customtkinter as ctk


class ComprehensiveTestSuite:
    """Test suite for comprehensive testing of the laser trim analyzer."""
    
    def __init__(self):
        self.results = {
            'single_file_tests': [],
            'batch_tests': [],
            'edge_case_tests': [],
            'database_tests': [],
            'gui_tests': [],
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
        self.ml_manager = None
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
        
        # Initialize ML manager
        try:
            self.ml_manager = MLManager(self.config, self.db_manager)
            self.ml_manager.initialize()
            self.logger.info("ML Manager initialized successfully")
        except Exception as e:
            self.logger.error(f"ML Manager initialization failed: {e}")
            self.ml_manager = None
        
        # Initialize processor
        ml_predictor = self.ml_manager.predictor if self.ml_manager else None
        self.processor = LaserTrimProcessor(
            self.config, 
            self.db_manager,
            ml_predictor
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
        
        # 5. Test GUI functionality (limited without actual display)
        await self.test_gui_functionality()
        
        # 6. Generate summary report
        self.generate_summary_report(time.time() - start_time)
        
        # Cleanup
        await self.cleanup()
        
    async def test_single_file_processing(self):
        """Test single file processing with various product numbers."""
        self.logger.info("\n=== Testing Single File Processing ===")
        
        # Select diverse test files
        test_files = [
            # Different product numbers
            ("2475-10_19_TEST DATA_11-16-2023_6-10 PM.xls", "2475 series"),
            ("5409A_5_TEST DATA_8-22-2017_9-02 AM.xls", "5409 series"),
            ("6126_74_TEST DATA_2-24-2025_12-18 PM.xls", "6126 series"),
            ("7063-A_17_TEST DATA_2-19-2021_9-15 AM.xls", "7063 series"),
            ("8275A_10_TEST DATA_5-21-2025_4-48 PM.xls", "8275 series"),
            ("8755-1_31_TEST DATA_6-13-2023_9-41 AM.xls", "8755 series"),
        ]
        
        for filename, description in test_files:
            file_path = self.test_dir / filename
            if not file_path.exists():
                self.logger.warning(f"Test file not found: {filename}")
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
                    'tracks': len(result.tracks),
                    'warnings': len(result.validation_warnings),
                    'errors': len(result.processing_errors)
                }
                
                # Extract key metrics
                if result.tracks:
                    primary_track = next(iter(result.tracks.values()))
                    test_result.update({
                        'sigma_gradient': primary_track.sigma_analysis.sigma_gradient,
                        'sigma_pass': primary_track.sigma_analysis.sigma_pass,
                        'linearity_error': primary_track.linearity_analysis.final_linearity_error_shifted,
                        'linearity_pass': primary_track.linearity_analysis.linearity_pass,
                        'resistance_change': primary_track.resistance_analysis.resistance_change_percent
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
                    'traceback': traceback.format_exc()
                })
    
    async def test_batch_processing(self):
        """Test batch processing with different batch sizes."""
        self.logger.info("\n=== Testing Batch Processing ===")
        
        batch_tests = [
            ("Small batch (10 files)", 10),
            ("Medium batch (25 files)", 25),
            ("Large batch (50 files)", 50),
        ]
        
        # Get all available test files
        all_files = list(self.test_dir.glob("*.xls"))[:100]  # Limit to 100 files
        
        for test_name, batch_size in batch_tests:
            if len(all_files) < batch_size:
                self.logger.warning(f"Not enough files for {test_name}")
                continue
                
            self.logger.info(f"Testing {test_name}")
            
            batch_files = all_files[:batch_size]
            
            try:
                start_time = time.time()
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Process batch
                results = await self.processor.process_batch(
                    batch_files,
                    output_dir=self.output_dir / 'batch_processing',
                    progress_callback=lambda msg, prog: self.logger.debug(f"{msg}: {prog:.1%}")
                )
                
                processing_time = time.time() - start_time
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = memory_after - memory_before
                
                # Analyze results
                successful = len(results)
                failed = batch_size - successful
                pass_count = sum(1 for r in results.values() if r.overall_status == AnalysisStatus.PASS)
                warning_count = sum(1 for r in results.values() if r.overall_status == AnalysisStatus.WARNING)
                fail_count = sum(1 for r in results.values() if r.overall_status == AnalysisStatus.FAIL)
                
                batch_result = {
                    'test_name': test_name,
                    'batch_size': batch_size,
                    'status': 'success',
                    'processing_time': processing_time,
                    'files_per_second': successful / processing_time if processing_time > 0 else 0,
                    'memory_used_mb': memory_used,
                    'successful': successful,
                    'failed': failed,
                    'pass_count': pass_count,
                    'warning_count': warning_count,
                    'fail_count': fail_count
                }
                
                self.results['batch_tests'].append(batch_result)
                self.logger.info(f"✓ Batch processed in {processing_time:.2f}s - {successful}/{batch_size} successful")
                
            except Exception as e:
                self.logger.error(f"✗ Batch processing failed: {e}")
                self.results['batch_tests'].append({
                    'test_name': test_name,
                    'batch_size': batch_size,
                    'status': 'failed',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
    
    async def test_edge_cases(self):
        """Test edge cases and error handling."""
        self.logger.info("\n=== Testing Edge Cases ===")
        
        edge_case_tests = [
            # Test non-existent file
            {
                'name': 'Non-existent file',
                'test': lambda: self.processor.process_file(Path("non_existent.xls")),
                'expected_error': 'FileNotFoundError'
            },
            # Test invalid file path
            {
                'name': 'Invalid file path',
                'test': lambda: self.processor.process_file(Path("/invalid/../../../path.xls")),
                'expected_error': 'ValidationError'
            },
            # Test empty batch
            {
                'name': 'Empty batch',
                'test': lambda: self.processor.process_batch([]),
                'expected_result': {}
            },
            # Test file with spaces in name
            {
                'name': 'File with spaces',
                'file': "8232-1 BLUE_39_TEST DATA_8-12-2019_4-32 PM.xls",
                'test': None  # Will be set dynamically
            },
            # Test file with special characters
            {
                'name': 'File with special characters',
                'file': "8394-6_101 REDUNDANT_TEST DATA_3-25-2022_9-32 AM.xls",
                'test': None
            }
        ]
        
        for test_case in edge_case_tests:
            self.logger.info(f"Testing: {test_case['name']}")
            
            try:
                # Set up test function for file-based tests
                if 'file' in test_case and test_case['file']:
                    file_path = self.test_dir / test_case['file']
                    if file_path.exists():
                        test_case['test'] = lambda fp=file_path: self.processor.process_file(fp)
                    else:
                        self.logger.warning(f"Edge case file not found: {test_case['file']}")
                        continue
                
                if test_case['test']:
                    result = await test_case['test']()
                    
                    if 'expected_error' in test_case:
                        # Should have raised an error
                        self.results['edge_case_tests'].append({
                            'name': test_case['name'],
                            'status': 'failed',
                            'error': 'Expected error not raised'
                        })
                        self.logger.error(f"✗ Expected error '{test_case['expected_error']}' not raised")
                    else:
                        # Success case
                        self.results['edge_case_tests'].append({
                            'name': test_case['name'],
                            'status': 'success',
                            'result': str(type(result))
                        })
                        self.logger.info(f"✓ Edge case handled successfully")
                        
            except Exception as e:
                error_type = type(e).__name__
                
                if 'expected_error' in test_case and error_type == test_case['expected_error']:
                    self.results['edge_case_tests'].append({
                        'name': test_case['name'],
                        'status': 'success',
                        'error_caught': error_type
                    })
                    self.logger.info(f"✓ Expected error caught: {error_type}")
                else:
                    self.results['edge_case_tests'].append({
                        'name': test_case['name'],
                        'status': 'failed',
                        'error': str(e),
                        'error_type': error_type
                    })
                    self.logger.error(f"✗ Unexpected error: {e}")
    
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
        
        # Test 1: Save analysis result
        try:
            # Process a file first
            test_file = next(self.test_dir.glob("*.xls"))
            result = await self.processor.process_file(test_file)
            
            # Save to database
            saved_id = await self.db_manager.save_analysis(result)
            
            self.results['database_tests'].append({
                'test': 'Save analysis',
                'status': 'success',
                'saved_id': saved_id
            })
            self.logger.info(f"✓ Saved analysis with ID: {saved_id}")
            
            # Test 2: Retrieve analysis
            retrieved = await self.db_manager.get_analysis(saved_id)
            
            if retrieved:
                self.results['database_tests'].append({
                    'test': 'Retrieve analysis',
                    'status': 'success',
                    'retrieved_model': retrieved.metadata.model
                })
                self.logger.info("✓ Retrieved analysis successfully")
            else:
                raise Exception("Failed to retrieve saved analysis")
                
            # Test 3: Query analyses
            recent_analyses = await self.db_manager.get_recent_analyses(limit=10)
            
            self.results['database_tests'].append({
                'test': 'Query recent analyses',
                'status': 'success',
                'count': len(recent_analyses)
            })
            self.logger.info(f"✓ Retrieved {len(recent_analyses)} recent analyses")
            
            # Test 4: Get statistics
            stats = await self.db_manager.get_statistics()
            
            self.results['database_tests'].append({
                'test': 'Get statistics',
                'status': 'success',
                'total_analyses': stats.get('total_analyses', 0)
            })
            self.logger.info(f"✓ Database statistics: {stats.get('total_analyses', 0)} total analyses")
            
        except Exception as e:
            self.logger.error(f"✗ Database test failed: {e}")
            self.results['database_tests'].append({
                'test': 'Database operations',
                'status': 'failed',
                'error': str(e)
            })
    
    async def test_gui_functionality(self):
        """Test GUI functionality (limited without display)."""
        self.logger.info("\n=== Testing GUI Functionality ===")
        
        try:
            # Test GUI initialization
            root = ctk.CTk()
            root.withdraw()  # Hide window for testing
            
            gui = LaserTrimAnalyzerGUI(root)
            
            self.results['gui_tests'].append({
                'test': 'GUI initialization',
                'status': 'success'
            })
            self.logger.info("✓ GUI initialized successfully")
            
            # Test page navigation
            pages = ['single_file', 'batch_processing', 'ml_tools', 'ai_insights', 'settings']
            for page in pages:
                try:
                    gui.show_page(page)
                    self.results['gui_tests'].append({
                        'test': f'Navigate to {page}',
                        'status': 'success'
                    })
                    self.logger.info(f"✓ Navigation to {page} successful")
                except Exception as e:
                    self.results['gui_tests'].append({
                        'test': f'Navigate to {page}',
                        'status': 'failed',
                        'error': str(e)
                    })
                    self.logger.error(f"✗ Navigation to {page} failed: {e}")
            
            # Cleanup
            root.destroy()
            
        except Exception as e:
            self.logger.error(f"✗ GUI test failed: {e}")
            self.results['gui_tests'].append({
                'test': 'GUI functionality',
                'status': 'failed',
                'error': str(e)
            })
    
    def generate_summary_report(self, total_time: float):
        """Generate comprehensive test summary report."""
        self.logger.info("\n=== Generating Test Summary ===")
        
        # Calculate summary statistics
        total_tests = 0
        successful_tests = 0
        failed_tests = 0
        
        # Single file tests
        single_file_success = sum(1 for t in self.results['single_file_tests'] if t['status'] == 'success')
        single_file_failed = len(self.results['single_file_tests']) - single_file_success
        total_tests += len(self.results['single_file_tests'])
        successful_tests += single_file_success
        failed_tests += single_file_failed
        
        # Batch tests
        batch_success = sum(1 for t in self.results['batch_tests'] if t['status'] == 'success')
        batch_failed = len(self.results['batch_tests']) - batch_success
        total_tests += len(self.results['batch_tests'])
        successful_tests += batch_success
        failed_tests += batch_failed
        
        # Edge case tests
        edge_success = sum(1 for t in self.results['edge_case_tests'] if t['status'] == 'success')
        edge_failed = len(self.results['edge_case_tests']) - edge_success
        total_tests += len(self.results['edge_case_tests'])
        successful_tests += edge_success
        failed_tests += edge_failed
        
        # Database tests
        db_success = sum(1 for t in self.results['database_tests'] if t['status'] == 'success')
        db_failed = sum(1 for t in self.results['database_tests'] if t['status'] == 'failed')
        total_tests += db_success + db_failed
        successful_tests += db_success
        failed_tests += db_failed
        
        # GUI tests
        gui_success = sum(1 for t in self.results['gui_tests'] if t['status'] == 'success')
        gui_failed = sum(1 for t in self.results['gui_tests'] if t['status'] == 'failed')
        total_tests += gui_success + gui_failed
        successful_tests += gui_success
        failed_tests += gui_failed
        
        # Performance metrics
        if self.results['single_file_tests']:
            avg_single_time = sum(t['processing_time'] for t in self.results['single_file_tests'] 
                                if 'processing_time' in t) / len(self.results['single_file_tests'])
        else:
            avg_single_time = 0
        
        # Create summary
        self.results['summary'] = {
            'total_time': total_time,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
            'single_file_tests': {
                'total': len(self.results['single_file_tests']),
                'success': single_file_success,
                'failed': single_file_failed,
                'avg_processing_time': avg_single_time
            },
            'batch_tests': {
                'total': len(self.results['batch_tests']),
                'success': batch_success,
                'failed': batch_failed
            },
            'edge_case_tests': {
                'total': len(self.results['edge_case_tests']),
                'success': edge_success,
                'failed': edge_failed
            },
            'database_tests': {
                'total': db_success + db_failed,
                'success': db_success,
                'failed': db_failed
            },
            'gui_tests': {
                'total': gui_success + gui_failed,
                'success': gui_success,
                'failed': gui_failed
            }
        }
        
        # Save results to JSON
        report_path = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST SUMMARY")
        print("="*60)
        print(f"Total Test Time: {total_time:.2f} seconds")
        print(f"Total Tests Run: {total_tests}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Failed Tests: {failed_tests}")
        print(f"Success Rate: {self.results['summary']['success_rate']:.1f}%")
        print("\nDetailed Results:")
        print(f"- Single File Tests: {single_file_success}/{len(self.results['single_file_tests'])}")
        print(f"- Batch Tests: {batch_success}/{len(self.results['batch_tests'])}")
        print(f"- Edge Case Tests: {edge_success}/{len(self.results['edge_case_tests'])}")
        print(f"- Database Tests: {db_success}/{db_success + db_failed}")
        print(f"- GUI Tests: {gui_success}/{gui_success + gui_failed}")
        print(f"\nDetailed report saved to: {report_path}")
        print("="*60)
        
        # Create visual summary if matplotlib available
        try:
            self.create_visual_summary()
        except Exception as e:
            self.logger.warning(f"Could not create visual summary: {e}")
    
    def create_visual_summary(self):
        """Create visual summary charts."""
        import matplotlib.pyplot as plt
        
        # Test category results
        categories = ['Single File', 'Batch', 'Edge Cases', 'Database', 'GUI']
        success_counts = [
            self.results['summary']['single_file_tests']['success'],
            self.results['summary']['batch_tests']['success'],
            self.results['summary']['edge_case_tests']['success'],
            self.results['summary']['database_tests']['success'],
            self.results['summary']['gui_tests']['success']
        ]
        failed_counts = [
            self.results['summary']['single_file_tests']['failed'],
            self.results['summary']['batch_tests']['failed'],
            self.results['summary']['edge_case_tests']['failed'],
            self.results['summary']['database_tests']['failed'],
            self.results['summary']['gui_tests']['failed']
        ]
        
        # Create bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Stacked bar chart
        x = range(len(categories))
        width = 0.6
        ax1.bar(x, success_counts, width, label='Success', color='green', alpha=0.8)
        ax1.bar(x, failed_counts, width, bottom=success_counts, label='Failed', color='red', alpha=0.8)
        ax1.set_ylabel('Number of Tests')
        ax1.set_title('Test Results by Category')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Success rate pie chart
        success = self.results['summary']['successful_tests']
        failed = self.results['summary']['failed_tests']
        ax2.pie([success, failed], labels=['Success', 'Failed'], 
                colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
        ax2.set_title('Overall Success Rate')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_summary_visual.png', dpi=150)
        plt.close()
        
        # Processing time analysis for single files
        if self.results['single_file_tests']:
            times = [t['processing_time'] for t in self.results['single_file_tests'] 
                    if 'processing_time' in t]
            if times:
                plt.figure(figsize=(10, 6))
                plt.hist(times, bins=20, alpha=0.7, color='blue', edgecolor='black')
                plt.xlabel('Processing Time (seconds)')
                plt.ylabel('Number of Files')
                plt.title('Single File Processing Time Distribution')
                plt.axvline(sum(times)/len(times), color='red', linestyle='dashed', 
                           linewidth=2, label=f'Mean: {sum(times)/len(times):.2f}s')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(self.output_dir / 'processing_time_distribution.png', dpi=150)
                plt.close()
    
    async def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up test resources...")
        
        if self.db_manager:
            await self.db_manager.close()
        
        # Force garbage collection
        import gc
        gc.collect()


async def main():
    """Main test execution function."""
    print("Starting Comprehensive Test Suite for Laser Trim Analyzer v2")
    print("=" * 60)
    
    test_suite = ComprehensiveTestSuite()
    
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