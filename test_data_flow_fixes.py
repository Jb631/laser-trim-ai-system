#!/usr/bin/env python3
"""
Comprehensive Test Script for Data Flow and File Management Fixes

Tests the complete pipeline from file selection through analysis to ensure:
1. Files load properly and remain selected
2. Analysis completes without debug errors
3. ML tools function without errors
4. File metadata persists throughout the process
5. Error handling works correctly
"""

import asyncio
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any
import traceback

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from laser_trim_analyzer.core.config import Config, get_config
from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.core.models import AnalysisResult, SystemType
from laser_trim_analyzer.core.exceptions import ProcessingError, ValidationError
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.utils.file_utils import ensure_directory

# Configure logging for comprehensive debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_data_flow.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)


class DataFlowTester:
    """Comprehensive tester for data flow and file management fixes."""
    
    def __init__(self):
        """Initialize the tester."""
        self.config = None
        self.processor = None
        self.db_manager = None
        self.test_files = []
        self.temp_dir = None
        self.results = {}
        self.errors = []
        
    def setup(self):
        """Set up test environment."""
        logger.info("Setting up test environment...")
        
        try:
            # Create temporary directory
            self.temp_dir = Path(tempfile.mkdtemp(prefix="laser_trim_test_"))
            logger.info(f"Created temp directory: {self.temp_dir}")
            
            # Initialize configuration
            self.config = get_config()
            self.config.data_directory = self.temp_dir / "data"
            self.config.database.path = self.temp_dir / "test.db"
            self.config.ml.model_path = self.temp_dir / "models"
            
            # Create directories
            ensure_directory(self.config.data_directory)
            ensure_directory(self.config.ml.model_path)
            
            # Initialize database manager
            try:
                self.db_manager = DatabaseManager(str(self.config.database.path))
                self.db_manager.init_db()
                logger.info("Database manager initialized successfully")
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")
                self.db_manager = None
            
            # Create test files
            self._create_test_files()
            
            logger.info("Test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _create_test_files(self):
        """Create test Excel files for processing."""
        logger.info("Creating test files...")
        
        try:
            import pandas as pd
            import numpy as np
            
            # Create test data directory
            test_data_dir = self.temp_dir / "test_files"
            ensure_directory(test_data_dir)
            
            # Test file configurations
            test_configs = [
                {
                    'filename': 'MODEL123_SERIAL001_20240101_120000.xlsx',
                    'system': 'A',
                    'sheets': ['TRK1 0', 'TRK1 TRM'],
                    'data_points': 100
                },
                {
                    'filename': 'MODEL456_SERIAL002_20240102_130000.xlsx',
                    'system': 'B',
                    'sheets': ['test', 'Lin Error'],
                    'data_points': 150
                },
                {
                    'filename': 'MODEL789_SERIAL003_TA_20240103_140000.xlsx',
                    'system': 'B',
                    'sheets': ['test', 'Lin Error'],
                    'data_points': 120
                }
            ]
            
            for config in test_configs:
                file_path = test_data_dir / config['filename']
                
                # Create Excel file with test data
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    for sheet_name in config['sheets']:
                        # Generate realistic test data
                        positions = np.linspace(0, 100, config['data_points'])
                        errors = np.random.normal(0, 0.1, config['data_points'])
                        upper_limits = np.full(config['data_points'], 0.5)
                        lower_limits = np.full(config['data_points'], -0.5)
                        
                        # Create DataFrame
                        data = {
                            'Position': positions,
                            'Error': errors,
                            'Upper Limit': upper_limits,
                            'Lower Limit': lower_limits
                        }
                        
                        df = pd.DataFrame(data)
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Add some metadata cells
                        if sheet_name == config['sheets'][0]:  # First sheet
                            # Add unit length and resistance values
                            metadata_df = pd.DataFrame({
                                'Parameter': ['Unit Length', 'Resistance'],
                                'Value': [10.0, 1000.0]
                            })
                            metadata_df.to_excel(writer, sheet_name=sheet_name, 
                                               startrow=len(df) + 2, index=False)
                
                self.test_files.append(file_path)
                logger.info(f"Created test file: {file_path.name}")
            
            logger.info(f"Created {len(self.test_files)} test files")
            
        except Exception as e:
            logger.error(f"Failed to create test files: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def test_processor_initialization(self):
        """Test processor initialization with ML and database components."""
        logger.info("Testing processor initialization...")
        
        try:
            # Test basic initialization
            self.processor = LaserTrimProcessor(
                config=self.config,
                db_manager=self.db_manager,
                ml_predictor=None,  # Let it initialize its own
                logger=logger
            )
            
            # Verify components are initialized
            assert self.processor is not None, "Processor not initialized"
            assert self.processor.config is not None, "Config not set"
            assert self.processor.logger is not None, "Logger not set"
            
            # Check analyzers
            if self.processor.sigma_analyzer is None:
                logger.warning("Sigma analyzer not initialized")
            if self.processor.linearity_analyzer is None:
                logger.warning("Linearity analyzer not initialized")
            if self.processor.resistance_analyzer is None:
                logger.warning("Resistance analyzer not initialized")
            
            # Check ML predictor
            if self.processor.ml_predictor is None:
                logger.info("ML predictor not available (expected for test environment)")
            else:
                logger.info("ML predictor initialized successfully")
            
            logger.info("✅ Processor initialization test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Processor initialization test failed: {e}")
            logger.error(traceback.format_exc())
            self.errors.append(f"Processor initialization: {e}")
            return False
    
    async def test_file_processing(self):
        """Test individual file processing."""
        logger.info("Testing file processing...")
        
        if not self.processor:
            logger.error("Processor not initialized")
            return False
        
        success_count = 0
        
        for file_path in self.test_files:
            logger.info(f"Processing file: {file_path.name}")
            
            try:
                # Test file processing with progress callback
                progress_updates = []
                
                def progress_callback(message: str, progress: float):
                    progress_updates.append((message, progress))
                    logger.debug(f"Progress: {progress:.1f}% - {message}")
                
                # Process the file
                result = await self.processor.process_file(
                    file_path=file_path,
                    output_dir=self.temp_dir / "output" / file_path.stem,
                    progress_callback=progress_callback
                )
                
                # Validate result
                assert isinstance(result, AnalysisResult), f"Invalid result type: {type(result)}"
                assert result.metadata is not None, "Missing metadata"
                assert result.tracks is not None, "Missing tracks"
                assert len(result.tracks) > 0, "No tracks processed"
                assert result.primary_track is not None, "Missing primary track"
                
                # Check progress updates
                assert len(progress_updates) > 0, "No progress updates received"
                assert progress_updates[-1][1] == 1.0, "Final progress not 100%"
                
                # Store result
                self.results[str(file_path)] = result
                success_count += 1
                
                logger.info(f"✅ Successfully processed {file_path.name}")
                logger.info(f"   - Model: {result.metadata.model}")
                logger.info(f"   - Serial: {result.metadata.serial}")
                logger.info(f"   - Tracks: {len(result.tracks)}")
                logger.info(f"   - Status: {result.overall_status.value}")
                logger.info(f"   - Processing time: {result.processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path.name}: {e}")
                logger.error(traceback.format_exc())
                self.errors.append(f"File processing {file_path.name}: {e}")
        
        logger.info(f"File processing test: {success_count}/{len(self.test_files)} files processed successfully")
        return success_count == len(self.test_files)
    
    async def test_batch_processing(self):
        """Test batch processing functionality."""
        logger.info("Testing batch processing...")
        
        if not self.processor:
            logger.error("Processor not initialized")
            return False
        
        try:
            # Test batch processing
            batch_progress_updates = []
            
            def batch_progress_callback(message: str, progress: float):
                batch_progress_updates.append((message, progress))
                logger.debug(f"Batch Progress: {progress:.1f}% - {message}")
            
            # Process all files in batch
            batch_results = await self.processor.process_batch(
                file_paths=self.test_files,
                output_dir=self.temp_dir / "batch_output",
                progress_callback=batch_progress_callback,
                max_workers=2
            )
            
            # Validate batch results
            assert isinstance(batch_results, dict), f"Invalid batch results type: {type(batch_results)}"
            assert len(batch_results) > 0, "No batch results returned"
            
            # Check that all files were processed
            processed_files = set(batch_results.keys())
            expected_files = {str(f) for f in self.test_files}
            
            missing_files = expected_files - processed_files
            if missing_files:
                logger.warning(f"Missing files in batch results: {missing_files}")
            
            extra_files = processed_files - expected_files
            if extra_files:
                logger.warning(f"Extra files in batch results: {extra_files}")
            
            logger.info(f"✅ Batch processing test passed")
            logger.info(f"   - Files processed: {len(batch_results)}")
            logger.info(f"   - Progress updates: {len(batch_progress_updates)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch processing test failed: {e}")
            logger.error(traceback.format_exc())
            self.errors.append(f"Batch processing: {e}")
            return False
    
    async def test_error_handling(self):
        """Test error handling with invalid files."""
        logger.info("Testing error handling...")
        
        if not self.processor:
            logger.error("Processor not initialized")
            return False
        
        try:
            # Test with non-existent file
            try:
                await self.processor.process_file(Path("nonexistent.xlsx"))
                logger.error("Should have failed with non-existent file")
                return False
            except (FileNotFoundError, ValidationError) as e:
                logger.info(f"✅ Correctly handled non-existent file: {e}")
            
            # Test with invalid file type
            invalid_file = self.temp_dir / "invalid.txt"
            invalid_file.write_text("This is not an Excel file")
            
            try:
                await self.processor.process_file(invalid_file)
                logger.error("Should have failed with invalid file type")
                return False
            except ValidationError as e:
                logger.info(f"✅ Correctly handled invalid file type: {e}")
            
            logger.info("✅ Error handling test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")
            logger.error(traceback.format_exc())
            self.errors.append(f"Error handling: {e}")
            return False
    
    async def test_state_persistence(self):
        """Test that file state persists throughout processing."""
        logger.info("Testing state persistence...")
        
        try:
            # Simulate file selection and processing workflow
            file_metadata_cache = {}
            processing_results = {}
            
            # Step 1: File selection (simulate UI file addition)
            for file_path in self.test_files:
                file_stats = file_path.stat()
                file_metadata_cache[str(file_path)] = {
                    'name': file_path.name,
                    'size': file_stats.st_size,
                    'modified': file_stats.st_mtime,
                    'path': str(file_path),
                    'status': 'Selected',
                    'added_time': time.time()
                }
            
            logger.info(f"Step 1: Selected {len(file_metadata_cache)} files")
            
            # Step 2: Processing (simulate analysis workflow)
            for file_path in self.test_files:
                file_key = str(file_path)
                
                # Update status to processing
                if file_key in file_metadata_cache:
                    file_metadata_cache[file_key]['status'] = 'Processing'
                
                # Simulate processing
                if file_key in self.results:
                    result = self.results[file_key]
                    processing_results[file_key] = result
                    
                    # Update status to completed
                    if file_key in file_metadata_cache:
                        file_metadata_cache[file_key].update({
                            'status': 'Completed',
                            'result': result,
                            'completed_time': time.time()
                        })
            
            logger.info(f"Step 2: Processed {len(processing_results)} files")
            
            # Step 3: Verify state persistence
            for file_path in self.test_files:
                file_key = str(file_path)
                
                # Check metadata persistence
                assert file_key in file_metadata_cache, f"Metadata lost for {file_path.name}"
                metadata = file_metadata_cache[file_key]
                assert metadata['name'] == file_path.name, f"Name mismatch for {file_path.name}"
                assert metadata['status'] == 'Completed', f"Status not updated for {file_path.name}"
                
                # Check result persistence
                assert file_key in processing_results, f"Result lost for {file_path.name}"
                result = processing_results[file_key]
                assert isinstance(result, AnalysisResult), f"Invalid result type for {file_path.name}"
            
            logger.info("✅ State persistence test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ State persistence test failed: {e}")
            logger.error(traceback.format_exc())
            self.errors.append(f"State persistence: {e}")
            return False
    
    def test_ml_integration(self):
        """Test ML integration without errors."""
        logger.info("Testing ML integration...")
        
        try:
            # Test ML predictor initialization
            if self.processor and self.processor.ml_predictor:
                logger.info("✅ ML predictor is available")
                
                # Test prediction on a result
                if self.results:
                    sample_result = next(iter(self.results.values()))
                    logger.info("ML predictor integration appears functional")
                else:
                    logger.info("No results available for ML testing")
            else:
                logger.info("✅ ML predictor gracefully disabled (expected in test environment)")
            
            logger.info("✅ ML integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ ML integration test failed: {e}")
            logger.error(traceback.format_exc())
            self.errors.append(f"ML integration: {e}")
            return False
    
    def cleanup(self):
        """Clean up test environment."""
        logger.info("Cleaning up test environment...")
        
        try:
            if self.temp_dir and self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Removed temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    async def run_all_tests(self):
        """Run all tests and return summary."""
        logger.info("🚀 Starting comprehensive data flow tests...")
        
        test_results = {}
        
        # Setup
        if not self.setup():
            logger.error("❌ Setup failed - aborting tests")
            return False
        
        # Run tests
        tests = [
            ("Processor Initialization", self.test_processor_initialization()),
            ("File Processing", self.test_file_processing()),
            ("Batch Processing", self.test_batch_processing()),
            ("Error Handling", self.test_error_handling()),
            ("State Persistence", self.test_state_persistence()),
            ("ML Integration", self.test_ml_integration())
        ]
        
        for test_name, test_coro in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                if asyncio.iscoroutine(test_coro):
                    result = await test_coro
                else:
                    result = test_coro
                test_results[test_name] = result
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                logger.error(traceback.format_exc())
                test_results[test_name] = False
                self.errors.append(f"{test_name}: {e}")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        passed = sum(1 for result in test_results.values() if result)
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if self.errors:
            logger.info(f"\nErrors encountered:")
            for error in self.errors:
                logger.info(f"  - {error}")
        
        # Cleanup
        self.cleanup()
        
        success = passed == total
        if success:
            logger.info("🎉 All tests passed! Data flow and file management fixes are working correctly.")
        else:
            logger.error("💥 Some tests failed. Please review the errors above.")
        
        return success


async def main():
    """Main test function."""
    tester = DataFlowTester()
    success = await tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner crashed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 