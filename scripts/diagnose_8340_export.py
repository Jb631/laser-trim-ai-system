#!/usr/bin/env python3
"""
Diagnostic script to investigate batch export issues with model 8340
showing mostly zeros and empty failure analysis sheet.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.utils.enhanced_excel_export import EnhancedExcelExporter

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_model_8340():
    """Diagnose issues with model 8340 data in exports."""
    try:
        # Initialize components
        config = get_config()
        db_manager = DatabaseManager(config)
        
        logger.info("=== DIAGNOSING MODEL 8340 EXPORT ISSUES ===")
        
        # 1. Check database for model 8340 data
        logger.info("\n1. Checking database for model 8340 data...")
        with db_manager.get_session() as session:
            from laser_trim_analyzer.database.models import AnalysisResult as DBAnalysisResult, TrackResult as DBTrackResult
            
            # Get all 8340 records
            results_8340 = session.query(DBAnalysisResult).filter(
                DBAnalysisResult.model.like('%8340%')
            ).all()
            
            logger.info(f"Found {len(results_8340)} records for model 8340")
            
            if results_8340:
                # Sample the first few records
                for i, result in enumerate(results_8340[:3]):
                    logger.info(f"\nRecord {i+1}:")
                    logger.info(f"  Model: {result.model}")
                    logger.info(f"  Serial: {result.serial}")
                    logger.info(f"  Filename: {result.filename}")
                    logger.info(f"  Tracks: {len(result.tracks)}")
                    
                    # Check track data
                    for j, track in enumerate(result.tracks[:2]):
                        logger.info(f"  Track {j+1} ({track.track_id}):")
                        logger.info(f"    Sigma Gradient: {track.sigma_gradient}")
                        logger.info(f"    Linearity Pass: {track.linearity_pass}")
                        logger.info(f"    Resistance Change: {track.resistance_change_percent}")
                        logger.info(f"    Final Linearity Error: {track.final_linearity_error_shifted}")
                        logger.info(f"    Failure Probability: {track.failure_probability}")
                        logger.info(f"    Range Utilization: {track.range_utilization_percent}")
        
        # 2. Test the export functionality directly
        logger.info("\n2. Testing export functionality...")
        
        # Get some test data
        test_results = db_manager.get_historical_data(
            model="8340",
            limit=10,
            include_tracks=True
        )
        
        if test_results:
            logger.info(f"Retrieved {len(test_results)} test results for export")
            
            # Create a test export
            exporter = EnhancedExcelExporter()
            
            # Convert to the format expected by exporter
            export_data = {}
            for result in test_results:
                export_data[result.filename] = result
            
            # Create test export file
            test_export_path = Path("test_8340_export.xlsx")
            exporter.export_batch_results(
                results=export_data,
                output_path=test_export_path,
                include_raw_data=True,
                progress_callback=lambda msg, pct: logger.info(f"Export progress: {msg} ({pct*100:.1f}%)")
            )
            
            logger.info(f"\nTest export created: {test_export_path}")
            
            # Read back the export and check for issues
            logger.info("\n3. Analyzing exported data...")
            
            # Read summary sheet
            df_summary = pd.read_excel(test_export_path, sheet_name='Summary')
            logger.info(f"\nSummary sheet shape: {df_summary.shape}")
            logger.info(f"Columns: {list(df_summary.columns)}")
            
            # Check for 8340 data
            df_8340 = df_summary[df_summary['Model'].str.contains('8340', na=False)]
            logger.info(f"\nFound {len(df_8340)} rows for model 8340 in summary")
            
            if not df_8340.empty:
                # Check for zero values
                numeric_cols = df_8340.select_dtypes(include=[np.number]).columns
                zero_counts = {}
                for col in numeric_cols:
                    zero_count = (df_8340[col] == 0).sum()
                    zero_counts[col] = zero_count
                    if zero_count > len(df_8340) * 0.5:  # More than 50% zeros
                        logger.warning(f"  Column '{col}' has {zero_count}/{len(df_8340)} zeros!")
                
                # Sample some data
                logger.info("\nSample of 8340 data:")
                important_cols = ['Model', 'Serial', 'Sigma Gradient', 'Linearity Pass', 
                                'Resistance Change %', 'Risk Category']
                available_cols = [col for col in important_cols if col in df_8340.columns]
                logger.info(df_8340[available_cols].head().to_string())
            
            # Check failure analysis sheet
            try:
                df_failure = pd.read_excel(test_export_path, sheet_name='Failure Analysis')
                logger.info(f"\nFailure Analysis sheet shape: {df_failure.shape}")
                if df_failure.empty:
                    logger.warning("Failure Analysis sheet is EMPTY!")
                    
                    # Check if there should be failures
                    high_risk_count = 0
                    fail_count = 0
                    for result in test_results:
                        for track in result.tracks:
                            if track.risk_category and track.risk_category.value == 'HIGH':
                                high_risk_count += 1
                            if track.status and track.status.value == 'FAIL':
                                fail_count += 1
                    
                    logger.info(f"Expected failures: {fail_count} failed tracks, {high_risk_count} high risk tracks")
                    
            except Exception as e:
                logger.error(f"Could not read Failure Analysis sheet: {e}")
            
            # Clean up test file
            if test_export_path.exists():
                test_export_path.unlink()
                logger.info("\nTest export file deleted")
                
        else:
            logger.warning("No test results found for model 8340")
            
        # 3. Check the export utility itself
        logger.info("\n4. Checking export utility code...")
        export_file = Path(__file__).parent.parent / "src" / "laser_trim_analyzer" / "utils" / "enhanced_excel_export.py"
        if export_file.exists():
            with open(export_file, 'r') as f:
                content = f.read()
                
            # Look for potential issues
            if "sigma_gradient" in content:
                logger.info("✓ Export utility references sigma_gradient")
            else:
                logger.warning("✗ Export utility may not properly handle sigma_gradient")
                
            if "failure_probability" in content or "risk_category" in content:
                logger.info("✓ Export utility handles failure analysis data")
            else:
                logger.warning("✗ Export utility may not properly handle failure analysis")
                
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}", exc_info=True)

if __name__ == "__main__":
    diagnose_model_8340()