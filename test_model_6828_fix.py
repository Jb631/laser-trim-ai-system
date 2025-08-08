#!/usr/bin/env python3
"""
Test Script for Model 6828 Data Retrieval Fix

This script verifies that the Model Summary page fix resolves the 
"No valid sigma gradient values found!" error and blank Excel export issue.

Run this script from the project root directory:
    python test_model_6828_fix.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_model_6828_data_retrieval():
    """Test the complete Model 6828 data retrieval fix."""
    
    print("=" * 60)
    print("MODEL 6828 DATA RETRIEVAL FIX VALIDATION")
    print("=" * 60)
    
    try:
        # Import required modules
        from laser_trim_analyzer.database.manager import DatabaseManager
        from laser_trim_analyzer.core.config import get_config
        from sqlalchemy.orm import sessionmaker
        from laser_trim_analyzer.database.models import AnalysisResult, TrackResult
        import pandas as pd
        
        # Get config and database connection
        config = get_config()
        db_path = f"sqlite:///{config.database.path.absolute()}"
        print(f"✓ Using database: {db_path}")
        
        db_manager = DatabaseManager(db_path)
        Session = sessionmaker(bind=db_manager.engine)
        session = Session()
        
        # Test 1: Check if model 6828 data exists
        print("\n1. CHECKING MODEL 6828 DATABASE RECORDS...")
        analyses = session.query(AnalysisResult).filter(AnalysisResult.model == '6828').all()
        print(f"   Found {len(analyses)} analyses for model 6828")
        
        if not analyses:
            print("   ❌ No model 6828 data found in database!")
            print("   This suggests a database connection issue or missing data.")
            return False
            
        # Test 2: Examine sample data structure
        print("\n2. EXAMINING DATA STRUCTURE...")
        sample_analysis = analyses[0]
        tracks = session.query(TrackResult).filter(TrackResult.analysis_id == sample_analysis.id).all()
        print(f"   Sample analysis ID {sample_analysis.id}: {len(tracks)} tracks")
        
        if tracks:
            sample_track = tracks[0]
            print(f"   Sample track fields:")
            print(f"     sigma_gradient: {sample_track.sigma_gradient}")
            print(f"     sigma_threshold: {sample_track.sigma_threshold}")
            print(f"     linearity_spec: {sample_track.linearity_spec}")
            print(f"     resistance_change_percent: {sample_track.resistance_change_percent}")
            print(f"     failure_probability: {sample_track.failure_probability}")
            print(f"     unit_length: {sample_track.unit_length}")
        
        # Test 3: Simulate the FIXED Model Summary data extraction
        print("\n3. TESTING FIXED DATA EXTRACTION LOGIC...")
        
        # This mimics the exact logic from the fixed ModelSummaryPage._load_model_data()
        data_rows = []
        total_sigma_values = 0
        
        for analysis in analyses[:5]:  # Test first 5 for speed
            tracks = session.query(TrackResult).filter(TrackResult.analysis_id == analysis.id).all()
            
            for track in tracks:
                # Use the FIXED logic - access fields directly from track record
                row = {
                    'analysis_id': analysis.id,
                    'filename': analysis.filename,
                    'model': analysis.model,
                    'serial': analysis.serial,
                    'track_id': track.track_id,
                    
                    # Core fields that were broken before (100% NULL in Excel)
                    'sigma_gradient': track.sigma_gradient,
                    'sigma_threshold': track.sigma_threshold,
                    'sigma_pass': track.sigma_pass,
                    'linearity_spec': track.linearity_spec,
                    'linearity_pass': track.linearity_pass,
                    'resistance_change_percent': track.resistance_change_percent,
                    'failure_probability': track.failure_probability,
                    'unit_length': track.unit_length,
                    'untrimmed_resistance': track.untrimmed_resistance,
                    'trimmed_resistance': track.trimmed_resistance,
                    
                    # Advanced fields added for completeness
                    'travel_length': track.travel_length,
                    'gradient_margin': track.gradient_margin,
                    'trim_improvement_percent': track.trim_improvement_percent,
                    'range_utilization_percent': track.range_utilization_percent,
                }
                
                data_rows.append(row)
                
                # Count valid sigma values
                if track.sigma_gradient is not None:
                    total_sigma_values += 1
        
        print(f"   Extracted {len(data_rows)} track records")
        print(f"   Found {total_sigma_values} non-null sigma gradient values")
        
        # Test 4: Validate the fix resolves the original issue
        print("\n4. VALIDATING FIX RESULTS...")
        
        if total_sigma_values == 0:
            print("   ❌ ISSUE PERSISTS: Still no valid sigma gradient values found!")
            print("   This indicates either:")
            print("     - Data was never saved to database properly")
            print("     - Database schema mismatch")
            print("     - Different database being used than expected")
            return False
        else:
            print(f"   ✅ SUCCESS: Found {total_sigma_values} valid sigma gradient values!")
            
        # Create DataFrame like Model Summary page does
        model_data = pd.DataFrame(data_rows)
        sigma_values = model_data['sigma_gradient'].dropna()
        
        if len(sigma_values) > 0:
            print(f"   ✅ Sigma statistics: Min={sigma_values.min():.6f}, Max={sigma_values.max():.6f}")
            print(f"   ✅ Mean={sigma_values.mean():.6f}, Std={sigma_values.std():.6f}")
        
        # Test 5: Check for previously problematic fields
        print("\n5. CHECKING PREVIOUSLY BLANK FIELDS...")
        
        problematic_fields = [
            'sigma_gradient', 'sigma_threshold', 'linearity_spec', 
            'resistance_change_percent', 'failure_probability', 
            'unit_length', 'untrimmed_resistance', 'trimmed_resistance'
        ]
        
        all_fields_fixed = True
        for field in problematic_fields:
            non_null_count = model_data[field].notna().sum()
            total_count = len(model_data)
            percent_populated = (non_null_count / total_count * 100) if total_count > 0 else 0
            
            print(f"   {field}: {non_null_count}/{total_count} populated ({percent_populated:.1f}%)")
            
            if non_null_count == 0:
                all_fields_fixed = False
                print(f"     ❌ {field} still 100% NULL!")
        
        if all_fields_fixed and total_sigma_values > 0:
            print("\n🎉 OVERALL RESULT: FIX SUCCESSFUL!")
            print("   - 'No valid sigma gradient values found!' error should be resolved")
            print("   - Excel exports should now show actual data instead of blank fields")
            print("   - Model Summary page should display charts and metrics")
            return True
        else:
            print("\n❌ OVERALL RESULT: ISSUES REMAIN")
            print("   Some fields are still returning NULL values")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'session' in locals():
            session.close()


def test_gui_model_summary():
    """Test the actual Model Summary page functionality."""
    
    print("\n" + "=" * 60)
    print("GUI MODEL SUMMARY PAGE TEST")
    print("=" * 60)
    
    try:
        # Import the actual Model Summary page class
        from laser_trim_analyzer.gui.pages.model_summary_page import ModelSummaryPage
        from laser_trim_analyzer.database.manager import DatabaseManager
        from laser_trim_analyzer.core.config import get_config
        
        # Mock parent and main_window for testing
        class MockMainWindow:
            def __init__(self):
                config = get_config()
                db_path = f"sqlite:///{config.database.path.absolute()}"
                self.db_manager = DatabaseManager(db_path)
        
        class MockParent:
            pass
        
        print("✓ Creating Model Summary page instance...")
        main_window = MockMainWindow()
        parent = MockParent()
        
        # Create Model Summary page (this should work with the fix)
        model_summary_page = ModelSummaryPage(parent, main_window)
        
        print("✓ Loading model 6828 data...")
        # This calls the fixed _load_model_data method
        model_summary_page._load_model_data('6828')
        
        if model_summary_page.model_data is not None and len(model_summary_page.model_data) > 0:
            print(f"✅ SUCCESS: Loaded {len(model_summary_page.model_data)} records for model 6828")
            
            # Check sigma values specifically
            sigma_values = model_summary_page.model_data['sigma_gradient'].dropna()
            print(f"✅ Found {len(sigma_values)} valid sigma gradient values")
            
            if len(sigma_values) > 0:
                print(f"✅ Sigma range: {sigma_values.min():.6f} to {sigma_values.max():.6f}")
                print("🎉 Model Summary page fix is WORKING!")
                return True
            else:
                print("❌ Model Summary page still shows no sigma gradient values")
                return False
        else:
            print("❌ Model Summary page failed to load any data")
            return False
            
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing Model 6828 Data Retrieval Fix...")
    print("This validates the fix for 'No valid sigma gradient values found!' error")
    print()
    
    # Test 1: Direct database validation
    db_test_passed = test_model_6828_data_retrieval()
    
    # Test 2: GUI integration test
    gui_test_passed = test_gui_model_summary()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Database Test: {'✅ PASS' if db_test_passed else '❌ FAIL'}")
    print(f"GUI Test: {'✅ PASS' if gui_test_passed else '❌ FAIL'}")
    
    if db_test_passed and gui_test_passed:
        print("\n🎉 ALL TESTS PASSED - Model 6828 fix is working correctly!")
        print("\nNext steps:")
        print("1. Open the GUI and navigate to Model Summary > Model 6828")
        print("2. Verify you see charts and data instead of error messages")
        print("3. Export to Excel and confirm numeric fields have actual values")
    else:
        print("\n❌ SOME TESTS FAILED - Additional investigation needed")
        print("\nTroubleshooting suggestions:")
        print("1. Check database path in config matches your actual data location")
        print("2. Verify model 6828 data exists in the database")
        print("3. Check for any database connection or permission issues")