#!/usr/bin/env python3
"""
Test Excel file processing with the Laser Trim Analyzer

This script tests the actual processing of Excel files to verify calculations.
"""

import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('excel_processing_test.log')
    ]
)

logger = logging.getLogger(__name__)

def test_single_file_processing():
    """Test processing of a single Excel file."""
    logger.info("Starting Excel file processing test...")
    
    try:
        # Import required modules
        from laser_trim_analyzer.core.processor import LaserTrimProcessor
        from laser_trim_analyzer.core.config import get_config
        
        # Get configuration
        config = get_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize processor
        processor = LaserTrimProcessor(config)
        logger.info("Processor initialized successfully")
        
        # Find test files
        test_files_dir = Path(__file__).parent.parent / "test_files" / "System A test files"
        if not test_files_dir.exists():
            logger.error(f"Test files directory not found: {test_files_dir}")
            return False
            
        # Get first few Excel files
        excel_files = list(test_files_dir.glob("*.xls"))[:3]  # Test first 3 files
        
        if not excel_files:
            logger.error("No Excel files found in test directory")
            return False
            
        logger.info(f"Found {len(excel_files)} test files")
        
        # Process each file
        results = []
        for file_path in excel_files:
            logger.info(f"\nProcessing: {file_path.name}")
            
            try:
                # Process the file (handle both sync and async)
                import asyncio
                import inspect
                
                if inspect.iscoroutinefunction(processor.process_file):
                    # It's async, run it with asyncio
                    result = asyncio.run(processor.process_file(file_path))
                else:
                    # It's sync
                    result = processor.process_file(file_path)
                
                # Log summary
                logger.info(f"  Model: {result.metadata.model}")
                logger.info(f"  Serial: {result.metadata.serial}")
                logger.info(f"  System Type: {result.metadata.system}")
                logger.info(f"  Processing Time: {result.processing_time:.2f}s")
                logger.info(f"  Overall Status: {result.overall_status}")
                logger.info(f"  Tracks Analyzed: {len(result.tracks)}")
                
                # Log track details
                for track_id, track in result.tracks.items():
                    logger.info(f"\n  Track {track_id}:")
                    logger.info(f"    Status: {track.overall_status}")
                    
                    if track.sigma_analysis:
                        logger.info(f"    Sigma Analysis:")
                        logger.info(f"      Gradient: {track.sigma_analysis.sigma_gradient:.6f}")
                        logger.info(f"      Threshold: {track.sigma_analysis.sigma_threshold:.6f}")
                        logger.info(f"      Pass: {track.sigma_analysis.sigma_pass}")
                        
                    if track.linearity_analysis:
                        logger.info(f"    Linearity Analysis:")
                        logger.info(f"      Spec: {track.linearity_analysis.linearity_spec:.4f}")
                        logger.info(f"      Pass: {track.linearity_analysis.linearity_pass}")
                        
                    if track.resistance_analysis:
                        logger.info(f"    Resistance Analysis:")
                        if hasattr(track.resistance_analysis, 'resistance_before'):
                            logger.info(f"      Before: {track.resistance_analysis.resistance_before:.2f}")
                            logger.info(f"      After: {track.resistance_analysis.resistance_after:.2f}")
                            logger.info(f"      Change: {track.resistance_analysis.resistance_change_percent:.2f}%")
                
                # Store result
                results.append({
                    'file': file_path.name,
                    'model': result.metadata.model,
                    'serial': result.metadata.serial,
                    'status': str(result.overall_status),
                    'tracks': len(result.tracks),
                    'processing_time': result.processing_time
                })
                
            except Exception as e:
                logger.error(f"  Failed to process {file_path.name}: {e}", exc_info=True)
                results.append({
                    'file': file_path.name,
                    'error': str(e)
                })
        
        # Save results summary
        summary_file = Path('test_results_summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'files_tested': len(excel_files),
                'results': results
            }, f, indent=2)
        
        logger.info(f"\nTest completed. Summary saved to {summary_file}")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

def test_batch_processing():
    """Test batch processing functionality."""
    logger.info("\n\nStarting batch processing test...")
    
    try:
        from laser_trim_analyzer.core.processor import LaserTrimProcessor
        from laser_trim_analyzer.core.config import get_config
        import asyncio
        
        # Get configuration
        config = get_config()
        processor = LaserTrimProcessor(config)
        
        # Find test files
        test_files_dir = Path(__file__).parent.parent / "test_files" / "System A test files"
        excel_files = list(test_files_dir.glob("*.xls"))[:5]  # Test first 5 files
        
        logger.info(f"Testing batch processing with {len(excel_files)} files")
        
        # Simple synchronous batch processing
        batch_results = []
        for i, file_path in enumerate(excel_files):
            logger.info(f"Processing file {i+1}/{len(excel_files)}: {file_path.name}")
            try:
                import inspect
                if inspect.iscoroutinefunction(processor.process_file):
                    # It's async, run it with asyncio
                    result = asyncio.run(processor.process_file(file_path))
                else:
                    # It's sync
                    result = processor.process_file(file_path)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
        
        # Summary
        logger.info(f"\nBatch processing completed:")
        logger.info(f"  Total files: {len(excel_files)}")
        logger.info(f"  Successful: {len(batch_results)}")
        logger.info(f"  Failed: {len(excel_files) - len(batch_results)}")
        
        # Calculate statistics
        total_tracks = sum(len(r.tracks) for r in batch_results)
        passed_tracks = sum(
            sum(1 for t in r.tracks.values() if str(t.overall_status) == 'Pass')
            for r in batch_results
        )
        
        logger.info(f"  Total tracks: {total_tracks}")
        logger.info(f"  Passed tracks: {passed_tracks}")
        logger.info(f"  Pass rate: {passed_tracks/total_tracks*100:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Batch test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Test single file processing
    single_success = test_single_file_processing()
    
    # Test batch processing
    batch_success = test_batch_processing()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Single file test: {'PASSED' if single_success else 'FAILED'}")
    logger.info(f"Batch processing test: {'PASSED' if batch_success else 'FAILED'}")
    
    sys.exit(0 if (single_success and batch_success) else 1)