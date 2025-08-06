#!/usr/bin/env python3
"""
Test a single Excel file processing
"""

import sys
import logging
from pathlib import Path
import asyncio
import inspect

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_single_file():
    """Test processing of a single Excel file."""
    try:
        from laser_trim_analyzer.core.processor import LaserTrimProcessor
        from laser_trim_analyzer.core.config import get_config
        
        # Get configuration
        config = get_config()
        processor = LaserTrimProcessor(config)
        
        # Get a test file
        test_file = Path(__file__).parent.parent / "test_files" / "System A test files" / "2475-10_19_TEST DATA_11-16-2023_6-10 PM.xls"
        
        if not test_file.exists():
            logger.error(f"Test file not found: {test_file}")
            return False
            
        logger.info(f"Processing: {test_file.name}")
        
        # Process the file
        if inspect.iscoroutinefunction(processor.process_file):
            result = asyncio.run(processor.process_file(test_file))
        else:
            result = processor.process_file(test_file)
            
        # Display results
        logger.info("\n" + "="*60)
        logger.info("FILE METADATA")
        logger.info("="*60)
        logger.info(f"Model: {result.metadata.model}")
        logger.info(f"Serial: {result.metadata.serial}")
        logger.info(f"System: {result.metadata.system}")
        logger.info(f"File Date: {result.metadata.file_date}")
        logger.info(f"Processing Time: {result.processing_time:.2f}s")
        logger.info(f"Overall Status: {result.overall_status}")
        logger.info(f"Validation Status: {result.overall_validation_status}")
        
        logger.info("\n" + "="*60)
        logger.info("TRACK ANALYSIS")
        logger.info("="*60)
        
        for track_id, track in result.tracks.items():
            logger.info(f"\nTrack {track_id}:")
            logger.info(f"  Overall Status: {track.status}")
            
            # Sigma Analysis
            if track.sigma_analysis:
                sigma = track.sigma_analysis
                logger.info(f"  Sigma Analysis:")
                logger.info(f"    Gradient: {sigma.sigma_gradient:.6f}")
                logger.info(f"    Threshold: {sigma.sigma_threshold:.6f}")
                logger.info(f"    Pass: {sigma.sigma_pass}")
                if hasattr(sigma, 'improvement_percent'):
                    logger.info(f"    Improvement: {sigma.improvement_percent:.2f}%")
                logger.info(f"    Compliance: {sigma.industry_compliance}")
                if hasattr(sigma, 'validation_result') and sigma.validation_result:
                    logger.info(f"    Validation Grade: {sigma.validation_result.validation_grade}")
            
            # Linearity Analysis
            if track.linearity_analysis:
                lin = track.linearity_analysis
                logger.info(f"  Linearity Analysis:")
                logger.info(f"    Spec: {lin.linearity_spec:.4f}")
                logger.info(f"    Pass: {lin.linearity_pass}")
                if hasattr(lin, 'linearity_error'):
                    logger.info(f"    Error: {lin.linearity_error:.4f}")
                logger.info(f"    Grade: {lin.industry_grade}")
                if hasattr(lin, 'validation_result') and lin.validation_result:
                    logger.info(f"    Validation Grade: {lin.validation_result.validation_grade}")
            
            # Resistance Analysis
            if track.resistance_analysis:
                res = track.resistance_analysis
                logger.info(f"  Resistance Analysis:")
                if hasattr(res, 'resistance_before'):
                    logger.info(f"    Before: {res.resistance_before:.2f}")
                    logger.info(f"    After: {res.resistance_after:.2f}")
                    logger.info(f"    Change: {res.resistance_change_percent:.2f}%")
                logger.info(f"    Stability: {res.resistance_stability_grade}")
            
            # Risk Assessment
            if hasattr(track, 'risk_category'):
                logger.info(f"  Risk Category: {track.risk_category}")
        
        # Validation Summary
        if result.validation_summary:
            logger.info("\n" + "="*60)
            logger.info("VALIDATION SUMMARY")
            logger.info("="*60)
            for key, value in result.validation_summary.items():
                logger.info(f"{key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_single_file()
    sys.exit(0 if success else 1)