#!/usr/bin/env python3
"""
Test validation grades to ensure they're calculated properly
"""

import sys
import logging
from pathlib import Path
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_validation_grades():
    """Test validation grade calculation."""
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
        result = await processor.process_file(test_file)
            
        # Display validation results
        logger.info("\n" + "="*60)
        logger.info("VALIDATION GRADE ANALYSIS")
        logger.info("="*60)
        logger.info(f"Overall Validation Grade: {result.validation_grade}")
        logger.info(f"Overall Validation Status: {result.overall_validation_status}")
        
        logger.info("\n" + "="*60)
        logger.info("INDIVIDUAL ANALYSIS GRADES")
        logger.info("="*60)
        
        for track_id, track in result.tracks.items():
            logger.info(f"\nTrack {track_id}:")
            
            # Sigma Analysis
            if track.sigma_analysis and track.sigma_analysis.validation_result:
                vr = track.sigma_analysis.validation_result
                logger.info(f"  Sigma Analysis:")
                logger.info(f"    Grade: {vr.validation_grade}")
                logger.info(f"    Deviation: {vr.deviation_percent:.2f}%")
                logger.info(f"    Is Valid: {vr.is_valid}")
                logger.info(f"    Tolerance Used: {vr.tolerance_used}%")
                logger.info(f"    Expected: {vr.expected_value:.6f}")
                logger.info(f"    Actual: {vr.actual_value:.6f}")
            
            # Linearity Analysis
            if track.linearity_analysis and track.linearity_analysis.validation_result:
                vr = track.linearity_analysis.validation_result
                logger.info(f"  Linearity Analysis:")
                logger.info(f"    Grade: {vr.validation_grade}")
                logger.info(f"    Deviation: {vr.deviation_percent:.2f}%")
                logger.info(f"    Is Valid: {vr.is_valid}")
                logger.info(f"    Tolerance Used: {vr.tolerance_used}%")
                logger.info(f"    Expected: {vr.expected_value:.4f}")
                logger.info(f"    Actual: {vr.actual_value:.4f}")
            
            # Resistance Analysis
            if track.resistance_analysis and track.resistance_analysis.validation_result:
                vr = track.resistance_analysis.validation_result
                logger.info(f"  Resistance Analysis:")
                logger.info(f"    Grade: {vr.validation_grade}")
                logger.info(f"    Deviation: {vr.deviation_percent:.2f}%")
                logger.info(f"    Is Valid: {vr.is_valid}")
                logger.info(f"    Tolerance Used: {vr.tolerance_used}%")
                
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_validation_grades())
    sys.exit(0 if success else 1)