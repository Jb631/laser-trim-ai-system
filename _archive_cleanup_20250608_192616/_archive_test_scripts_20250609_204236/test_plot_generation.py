#!/usr/bin/env python3
"""Test script to check plot generation."""

import asyncio
import logging
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, 'src')

from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.utils.file_utils import ensure_directory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_plot_generation():
    """Test plot generation with a sample file."""
    
    # Get configuration
    config = get_config()
    logger.info(f"Configuration loaded")
    logger.info(f"generate_plots: {config.processing.generate_plots}")
    logger.info(f"data_directory: {config.data_directory}")
    logger.info(f"Resolved data_directory: {config.data_directory.expanduser().resolve()}")
    
    # Create processor
    processor = LaserTrimProcessor(config)
    
    # Find a test file
    test_files_dir = Path("test_files/System A test files")
    test_files = list(test_files_dir.glob("*.xls"))
    
    if not test_files:
        logger.error("No test files found!")
        return
    
    test_file = test_files[0]
    logger.info(f"Using test file: {test_file}")
    
    # Create output directory
    base_dir = config.data_directory.expanduser().resolve()
    output_dir = base_dir / "test_plot_generation" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_directory(output_dir)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output directory exists: {output_dir.exists()}")
    
    # Process file
    try:
        result = await processor.process_file(
            file_path=test_file,
            output_dir=output_dir,
            progress_callback=lambda msg, prog: logger.info(f"Progress: {msg} ({prog:.1%})")
        )
        
        logger.info(f"Processing completed successfully")
        logger.info(f"Result: {result.metadata.model} - {result.metadata.serial}")
        
        # Check for plots
        plot_files = list(output_dir.glob("*.png"))
        logger.info(f"Plot files generated: {len(plot_files)}")
        for plot_file in plot_files:
            logger.info(f"  - {plot_file.name} ({plot_file.stat().st_size} bytes)")
        
        # Check track data for plot paths
        for track_id, track_data in result.tracks.items():
            logger.info(f"Track {track_id} plot_path: {getattr(track_data, 'plot_path', 'Not set')}")
            
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_plot_generation())