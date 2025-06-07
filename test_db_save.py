#!/usr/bin/env python3
"""
Test script to debug database save issues.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.core.models import (
    AnalysisResult, FileMetadata, TrackData, SigmaAnalysis,
    AnalysisStatus, SystemType
)
from laser_trim_analyzer.core.config import get_config

def create_test_analysis():
    """Create a minimal test analysis result."""
    metadata = FileMetadata(
        filename="test_file.xls",
        file_path=Path("/tmp/test_file.xls"),
        file_date=datetime.now(),
        model="TEST123",
        serial="001",
        system=SystemType.SYSTEM_A,
        has_multi_tracks=False
    )
    
    # Create minimal track data
    track_data = TrackData(
        track_id="Track1",
        travel_length=10.0,
        status=AnalysisStatus.PASS,
        sigma_analysis=SigmaAnalysis(
            sigma_gradient=0.5,
            sigma_threshold=1.0,
            sigma_pass=True
        )
    )
    
    # Create analysis result
    analysis = AnalysisResult(
        metadata=metadata,
        tracks={"Track1": track_data},
        overall_status=AnalysisStatus.PASS,
        processing_time=1.0,
        timestamp=datetime.now()
    )
    
    return analysis

def test_database_save():
    """Test database saving functionality."""
    print("=" * 60)
    print("Testing Database Save Functionality")
    print("=" * 60)
    
    try:
        # Get config
        config = get_config()
        print(f"✓ Configuration loaded")
        
        # Initialize database manager
        print("\nInitializing database manager...")
        db_manager = DatabaseManager(config)
        print(f"✓ Database manager initialized")
        
        # Create test analysis
        print("\nCreating test analysis...")
        analysis = create_test_analysis()
        print(f"✓ Test analysis created: {analysis.metadata.model}-{analysis.metadata.serial}")
        
        # Try to save using debug method
        print("\nAttempting to save analysis...")
        try:
            if hasattr(db_manager, 'save_analysis_result'):
                print("Using save_analysis_result (debug method)...")
                analysis_id = db_manager.save_analysis_result(analysis)
            else:
                print("Using save_analysis (normal method)...")
                analysis_id = db_manager.save_analysis(analysis)
            
            print(f"✓ Analysis saved successfully with ID: {analysis_id}")
            
            # Validate the save
            print(f"\nValidating saved analysis...")
            is_valid = db_manager.validate_saved_analysis(analysis_id)
            if is_valid:
                print(f"✓ Analysis validated successfully")
            else:
                print(f"✗ Analysis validation failed")
                
        except Exception as e:
            print(f"✗ Save failed: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            
        # Test batch save
        print("\n" + "=" * 60)
        print("Testing Batch Save...")
        print("=" * 60)
        
        # Create multiple analyses
        analyses = []
        for i in range(3):
            metadata = FileMetadata(
                filename=f"batch_test_{i}.xls",
                file_path=Path(f"/tmp/batch_test_{i}.xls"),
                file_date=datetime.now(),
                model="BATCH123",
                serial=f"00{i+1}",
                system=SystemType.SYSTEM_A,
                has_multi_tracks=False
            )
            
            track_data = TrackData(
                track_id="Track1",
                travel_length=10.0,
                status=AnalysisStatus.PASS,
                sigma_analysis=SigmaAnalysis(
                    sigma_gradient=0.5,
                    sigma_threshold=1.0,
                    sigma_pass=True
                )
            )
            
            analysis = AnalysisResult(
                metadata=metadata,
                tracks={"Track1": track_data},
                overall_status=AnalysisStatus.PASS,
                processing_time=1.0,
                timestamp=datetime.now()
            )
            analyses.append(analysis)
        
        print(f"Created {len(analyses)} test analyses for batch save")
        
        try:
            print("Attempting batch save...")
            saved_ids = db_manager.save_analysis_batch(analyses)
            successful_saves = len([id for id in saved_ids if id is not None])
            print(f"✓ Batch save completed: {successful_saves}/{len(analyses)} successful")
            
            if successful_saves < len(analyses):
                print(f"✗ Some saves failed!")
                for i, id in enumerate(saved_ids):
                    if id is None:
                        print(f"  - Analysis {i} failed to save")
                        
        except Exception as e:
            print(f"✗ Batch save failed: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"\n✗ Test failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_database_save()