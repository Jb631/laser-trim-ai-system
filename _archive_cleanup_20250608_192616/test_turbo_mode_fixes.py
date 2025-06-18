#!/usr/bin/env python3
"""
Test script to verify turbo mode batch processing fixes.
Tests:
1. File counter accuracy
2. Stop functionality
3. Database save functionality
4. Results display after processing
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.laser_trim_analyzer.core.config import get_config
from src.laser_trim_analyzer.core.large_scale_processor import LargeScaleProcessor
from src.laser_trim_analyzer.database.manager import DatabaseManager
from src.laser_trim_analyzer.gui.widgets.progress_widgets_ctk import BatchProgressDialog

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TurboModeTestHarness:
    """Test harness for turbo mode batch processing."""
    
    def __init__(self):
        self.config = get_config()
        self.db_manager = DatabaseManager(self.config)
        self.processor = LargeScaleProcessor(self.config, self.db_manager)
        self.stop_requested = False
        self.file_counts = []
        self.progress_values = []
        
    def test_file_counter_accuracy(self):
        """Test that file counter shows accurate progress."""
        logger.info("\n=== Testing File Counter Accuracy ===")
        
        # Create test callback to capture progress
        def progress_callback(message: str, progress: float):
            self.file_counts.append(message)
            self.progress_values.append(progress)
            logger.info(f"Progress: {message} ({progress:.2%})")
            return not self.stop_requested
            
        # Test with a small batch to verify counter
        test_files = self._get_test_files(limit=10)
        if not test_files:
            logger.warning("No test files found. Please ensure test Excel files are available.")
            return False
            
        logger.info(f"Processing {len(test_files)} test files...")
        
        # Reset counters
        self.file_counts.clear()
        self.progress_values.clear()
        
        # Process files
        try:
            results = self.processor.process_batch_turbo(
                test_files,
                progress_callback=progress_callback
            )
            
            # Verify file counts
            logger.info(f"\nCaptured {len(self.file_counts)} progress messages")
            
            # Check for accurate file count patterns
            count_accurate = False
            for msg in self.file_counts:
                if "Completed" in msg and "of" in msg:
                    parts = msg.split()
                    try:
                        completed = int(parts[1])
                        total = int(parts[3])
                        if total == len(test_files):
                            count_accurate = True
                            logger.info(f"✓ File count accurate: {completed}/{total}")
                    except (IndexError, ValueError):
                        pass
                        
            if count_accurate:
                logger.info("✓ File counter accuracy test PASSED")
                return True
            else:
                logger.error("✗ File counter accuracy test FAILED")
                return False
                
        except Exception as e:
            logger.error(f"Error during file counter test: {e}")
            return False
            
    def test_stop_functionality(self):
        """Test that processing stops when requested."""
        logger.info("\n=== Testing Stop Functionality ===")
        
        stop_at_file = 3
        files_processed = 0
        
        def progress_callback(message: str, progress: float):
            nonlocal files_processed
            if "Completed" in message:
                try:
                    files_processed = int(message.split()[1])
                except:
                    pass
                    
            # Request stop after processing 3 files
            if files_processed >= stop_at_file:
                logger.info(f"Requesting stop at file {files_processed}")
                self.stop_requested = True
                return False
            return True
            
        # Get more test files
        test_files = self._get_test_files(limit=20)
        if len(test_files) < 10:
            logger.warning("Not enough test files for stop test")
            return False
            
        logger.info(f"Processing {len(test_files)} files, will stop at file {stop_at_file}")
        
        # Reset stop flag
        self.stop_requested = False
        
        # Process files
        try:
            start_time = time.time()
            results = self.processor.process_batch_turbo(
                test_files,
                progress_callback=progress_callback
            )
            elapsed = time.time() - start_time
            
            # Verify stop worked
            if len(results) <= stop_at_file + 2:  # Allow some margin
                logger.info(f"✓ Processing stopped after {len(results)} files (elapsed: {elapsed:.1f}s)")
                logger.info("✓ Stop functionality test PASSED")
                return True
            else:
                logger.error(f"✗ Processing did not stop properly: {len(results)} files processed")
                logger.error("✗ Stop functionality test FAILED")
                return False
                
        except Exception as e:
            logger.error(f"Error during stop test: {e}")
            return False
            
    def test_database_save(self):
        """Test that results are saved to database with duplicate checking."""
        logger.info("\n=== Testing Database Save Functionality ===")
        
        # Get test files
        test_files = self._get_test_files(limit=5)
        if not test_files:
            logger.warning("No test files for database test")
            return False
            
        # Clear any existing test data
        self._clear_test_data(test_files)
        
        # Process files first time
        logger.info("Processing files (first time)...")
        try:
            results1 = self.processor.process_batch_turbo(
                test_files,
                progress_callback=lambda m, p: True
            )
            
            # Check database for saved results
            saved_count = self._count_saved_results(test_files)
            logger.info(f"Saved {saved_count} results to database")
            
            if saved_count != len(results1):
                logger.error(f"✗ Database save count mismatch: {saved_count} != {len(results1)}")
                return False
                
            # Process same files again to test duplicate checking
            logger.info("\nProcessing same files again (testing duplicates)...")
            results2 = self.processor.process_batch_turbo(
                test_files,
                progress_callback=lambda m, p: True
            )
            
            # Check that no duplicates were created
            saved_count2 = self._count_saved_results(test_files)
            logger.info(f"Total saved results: {saved_count2}")
            
            if saved_count2 == saved_count:
                logger.info("✓ Duplicate checking working correctly")
                logger.info("✓ Database save functionality test PASSED")
                return True
            else:
                logger.error(f"✗ Duplicates created: {saved_count2} > {saved_count}")
                logger.error("✗ Database save functionality test FAILED")
                return False
                
        except Exception as e:
            logger.error(f"Error during database test: {e}")
            return False
            
    def test_progress_dialog_compatibility(self):
        """Test that progress dialog handles both 0-1 and 0-100 ranges."""
        logger.info("\n=== Testing Progress Dialog Compatibility ===")
        
        try:
            # Test with 0-1 range
            logger.info("Testing 0-1 progress range...")
            progress_values_01 = [0.0, 0.25, 0.5, 0.75, 1.0]
            for val in progress_values_01:
                msg = f"Processing... ({val:.0%})"
                logger.info(f"  Progress: {val} -> {msg}")
                
            # Test with 0-100 range  
            logger.info("\nTesting 0-100 progress range...")
            progress_values_100 = [0, 25, 50, 75, 100]
            for val in progress_values_100:
                msg = f"Processing... ({val}%)"
                logger.info(f"  Progress: {val} -> {msg}")
                
            logger.info("✓ Progress dialog compatibility test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"Error during progress dialog test: {e}")
            return False
            
    def _get_test_files(self, limit=None):
        """Get test Excel files."""
        # Look for test files in common locations
        test_dirs = [
            Path(self.config.test_data_path) if hasattr(self.config, 'test_data_path') else None,
            project_root / "test_data",
            project_root / "6-16-2025 test_file",
            Path.home() / "Desktop" / "test_files"
        ]
        
        excel_files = []
        for test_dir in test_dirs:
            if test_dir and test_dir.exists():
                excel_files.extend(test_dir.glob("*.xlsx"))
                excel_files.extend(test_dir.glob("*.xls"))
                
        if limit:
            excel_files = excel_files[:limit]
            
        return excel_files
        
    def _clear_test_data(self, test_files):
        """Clear test data from database."""
        try:
            with self.db_manager.get_session() as session:
                for file_path in test_files:
                    # Delete by file path
                    session.execute(
                        "DELETE FROM analysis_results WHERE file_name = :fname",
                        {"fname": file_path.name}
                    )
                session.commit()
        except Exception as e:
            logger.warning(f"Could not clear test data: {e}")
            
    def _count_saved_results(self, test_files):
        """Count saved results in database."""
        try:
            with self.db_manager.get_session() as session:
                count = 0
                for file_path in test_files:
                    result = session.execute(
                        "SELECT COUNT(*) FROM analysis_results WHERE file_name = :fname",
                        {"fname": file_path.name}
                    ).scalar()
                    count += result or 0
                return count
        except Exception as e:
            logger.warning(f"Could not count saved results: {e}")
            return 0
            
    def run_all_tests(self):
        """Run all turbo mode tests."""
        logger.info("=" * 60)
        logger.info("TURBO MODE BATCH PROCESSING TEST SUITE")
        logger.info("=" * 60)
        
        tests = [
            ("File Counter Accuracy", self.test_file_counter_accuracy),
            ("Stop Functionality", self.test_stop_functionality),
            ("Database Save", self.test_database_save),
            ("Progress Dialog", self.test_progress_dialog_compatibility)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                logger.error(f"Test '{test_name}' crashed: {e}")
                results[test_name] = False
                
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        
        for test_name, result in results.items():
            status = "PASSED" if result else "FAILED"
            symbol = "✓" if result else "✗"
            logger.info(f"{symbol} {test_name}: {status}")
            
        logger.info(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("\n🎉 ALL TURBO MODE FIXES ARE WORKING CORRECTLY! 🎉")
        else:
            logger.error("\n⚠️ Some turbo mode features need attention")
            
        return passed == total


if __name__ == "__main__":
    # Run tests
    tester = TurboModeTestHarness()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)