#!/usr/bin/env python3
"""
Investigate why model 8340 batch processing data isn't being saved to database.
This script checks various aspects of the batch processing save mechanism.
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.laser_trim_analyzer.core.config import get_config
from src.laser_trim_analyzer.database.manager import DatabaseManager
from src.laser_trim_analyzer.database.models import (
    AnalysisResult, TrackResult, MLPrediction, QAAlert
)
from sqlalchemy import func, or_
import json

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_database_for_8340(session):
    """Check database for any references to model 8340."""
    logger.info("=" * 80)
    logger.info("CHECKING DATABASE FOR MODEL 8340 REFERENCES")
    logger.info("=" * 80)
    
    # Check AnalysisResult table
    logger.info("\n1. Checking AnalysisResult table...")
    
    # Try different query patterns
    patterns = ['8340', '%8340%', '%8340', '8340%']
    total_found = 0
    
    for pattern in patterns:
        count = session.query(func.count(AnalysisResult.id)).filter(
            AnalysisResult.model.like(pattern)
        ).scalar()
        logger.info(f"   Pattern '{pattern}': {count} records")
        if count > 0:
            # Get a sample
            sample = session.query(AnalysisResult).filter(
                AnalysisResult.model.like(pattern)
            ).first()
            logger.info(f"   Sample: ID={sample.id}, Model='{sample.model}', Serial='{sample.serial}'")
        total_found += count
    
    # Also check for exact match
    exact_count = session.query(func.count(AnalysisResult.id)).filter(
        AnalysisResult.model == '8340'
    ).scalar()
    logger.info(f"   Exact match '8340': {exact_count} records")
    
    # Check case variations
    case_variations = ['8340', 'P8340', 'p8340', 'M8340', 'm8340']
    for variant in case_variations:
        count = session.query(func.count(AnalysisResult.id)).filter(
            AnalysisResult.model == variant
        ).scalar()
        if count > 0:
            logger.info(f"   Found variant '{variant}': {count} records")
    
    logger.info(f"\n   Total 8340-related records: {total_found}")
    
    # Check recent entries
    logger.info("\n2. Checking recent entries (last 100)...")
    recent = session.query(AnalysisResult).order_by(
        AnalysisResult.timestamp.desc()
    ).limit(100).all()
    
    models_found = {}
    for r in recent:
        model = r.model or 'NONE'
        models_found[model] = models_found.get(model, 0) + 1
    
    logger.info("   Recent models processed:")
    for model, count in sorted(models_found.items()):
        logger.info(f"   - {model}: {count} entries")
    
    # Check for any model containing '340'
    logger.info("\n3. Checking for any model containing '340'...")
    any_340 = session.query(AnalysisResult).filter(
        AnalysisResult.model.contains('340')
    ).limit(10).all()
    
    if any_340:
        logger.info(f"   Found {len(any_340)} models containing '340':")
        for r in any_340:
            logger.info(f"   - Model: '{r.model}', Serial: '{r.serial}', Date: {r.timestamp}")
    else:
        logger.info("   No models containing '340' found")


def check_validation_rules():
    """Check if there are validation rules that might reject model 8340."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECKING VALIDATION RULES")
    logger.info("=" * 80)
    
    # Check security validator
    try:
        from src.laser_trim_analyzer.utils.validators import SecurityValidator
        validator = SecurityValidator()
        
        # Test model 8340
        test_data = {
            'model': '8340',
            'serial': 'TEST123',
            'operator': 'TestOp'
        }
        
        logger.info("\n1. Testing SecurityValidator with model '8340'...")
        try:
            is_valid = validator.validate_metadata(test_data)
            logger.info(f"   Validation result: {is_valid}")
        except Exception as e:
            logger.error(f"   Validation failed: {str(e)}")
            
        # Test variations
        variations = ['P8340', 'p8340', 'M8340', 'm8340', '8340-01']
        for variant in variations:
            test_data['model'] = variant
            try:
                is_valid = validator.validate_metadata(test_data)
                logger.info(f"   Model '{variant}': {'VALID' if is_valid else 'INVALID'}")
            except Exception as e:
                logger.info(f"   Model '{variant}': REJECTED - {str(e)}")
                
    except ImportError:
        logger.warning("   Could not import SecurityValidator")
    
    # Check for duplicate detection
    logger.info("\n2. Checking duplicate detection logic...")
    try:
        from src.laser_trim_analyzer.database.manager import DatabaseManager
        # This would need actual implementation check
        logger.info("   Would need to check DatabaseManager._is_duplicate_result method")
    except:
        pass


def simulate_batch_save():
    """Simulate what happens during batch save for model 8340."""
    logger.info("\n" + "=" * 80)
    logger.info("SIMULATING BATCH SAVE PROCESS")
    logger.info("=" * 80)
    
    config = get_config()
    db_manager = DatabaseManager(config)
    
    # Create a test result for model 8340
    test_result = {
        'file_path': '/test/path/8340_test.xlsx',
        'metadata': {
            'model': '8340',
            'serial': 'TEST123',
            'operator': 'TestOp',
            'test_date': datetime.now().isoformat(),
            'work_order': 'WO123',
            'revision': 'A'
        },
        'num_tracks': 4,
        'all_tracks_data': {
            'Track1': {'status': 'Pass', 'resistance': [100.5, 101.2, 99.8]},
            'Track2': {'status': 'Pass', 'resistance': [200.1, 199.5, 201.0]},
            'Track3': {'status': 'Pass', 'resistance': [150.2, 149.8, 150.5]},
            'Track4': {'status': 'Pass', 'resistance': [250.0, 249.5, 250.8]}
        },
        'resistance_data': {
            'Track1': [100.5, 101.2, 99.8],
            'Track2': [200.1, 199.5, 201.0],
            'Track3': [150.2, 149.8, 150.5],
            'Track4': [250.0, 249.5, 250.8]
        },
        'statistics': {
            'Track1': {'mean': 100.5, 'std': 0.7, 'cv': 0.7},
            'Track2': {'mean': 200.2, 'std': 0.75, 'cv': 0.37},
            'Track3': {'mean': 150.17, 'std': 0.35, 'cv': 0.23},
            'Track4': {'mean': 250.1, 'std': 0.65, 'cv': 0.26}
        },
        'validation_results': {
            'overall_status': 'Pass',
            'failure_reasons': [],
            'track_statuses': {
                'Track1': 'Pass',
                'Track2': 'Pass', 
                'Track3': 'Pass',
                'Track4': 'Pass'
            }
        },
        'ml_predictions': {
            'yield_forecast': 0.95,
            'confidence': 0.88
        }
    }
    
    logger.info("\n1. Attempting to save test result for model 8340...")
    
    try:
        # Get the save method
        if hasattr(db_manager, 'save_analysis_result'):
            logger.info("   Using save_analysis_result method...")
            saved_id = db_manager.save_analysis_result(test_result)
            if saved_id:
                logger.info(f"   SUCCESS: Saved with ID {saved_id}")
                
                # Verify it was saved
                with db_manager.get_session() as session:
                    verify = session.query(AnalysisResult).filter_by(id=saved_id).first()
                    if verify:
                        logger.info(f"   Verified: Model='{verify.model}', Serial='{verify.serial}'")
                    else:
                        logger.error("   ERROR: Could not verify saved record")
            else:
                logger.error("   FAILED: save_analysis_result returned None")
        else:
            logger.error("   ERROR: save_analysis_result method not found")
            
    except Exception as e:
        logger.error(f"   ERROR during save: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if hasattr(db_manager, 'close'):
            db_manager.close()


def check_batch_processing_logs():
    """Check if there are any batch processing logs that show errors."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECKING FOR BATCH PROCESSING LOGS")
    logger.info("=" * 80)
    
    # Check common log locations
    log_dirs = [
        Path("logs"),
        Path.home() / ".laser_trim_analyzer" / "logs",
        Path(os.environ.get('LOCALAPPDATA', '')) / "LaserTrimAnalyzer" / "logs",
        Path(".")  # Current directory
    ]
    
    for log_dir in log_dirs:
        if log_dir.exists():
            logger.info(f"\nChecking {log_dir}...")
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                for log_file in sorted(log_files)[-5:]:  # Last 5 files
                    logger.info(f"   Found: {log_file.name} ({log_file.stat().st_size} bytes)")
                    
                    # Check for 8340 references
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if '8340' in content:
                                logger.info(f"      *** Contains '8340' references!")
                                # Find relevant lines
                                for line in content.splitlines():
                                    if '8340' in line and any(word in line.lower() for word in ['error', 'fail', 'skip', 'duplicate', 'invalid']):
                                        logger.info(f"      > {line.strip()}")
                    except:
                        pass


def main():
    """Main investigation function."""
    logger.info("Starting investigation of model 8340 batch processing issue...")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Initialize database
    config = get_config()
    db_manager = DatabaseManager(config)
    
    try:
        with db_manager.get_session() as session:
            # Check database
            check_database_for_8340(session)
        
        # Check validation rules
        check_validation_rules()
        
        # Check logs
        check_batch_processing_logs()
        
        # Simulate save
        simulate_batch_save()
        
        logger.info("\n" + "=" * 80)
        logger.info("INVESTIGATION SUMMARY")
        logger.info("=" * 80)
        logger.info("""
Possible reasons why model 8340 isn't in the database:

1. Save to Database checkbox was not checked during batch processing
2. Validation failure (model format, security rules)
3. Duplicate detection (file already processed)
4. Database save error (check logs)
5. Model name mismatch (stored differently than '8340')

Recommendations:
1. Ensure "Save to Database" is checked before batch processing
2. Check application logs for errors during processing
3. Verify file metadata contains proper model number
4. Try processing a single 8340 file with verbose logging
        """)
        
    except Exception as e:
        logger.error(f"Investigation error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(db_manager, 'close'):
            db_manager.close()


if __name__ == "__main__":
    main()