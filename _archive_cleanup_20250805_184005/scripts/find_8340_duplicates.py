#!/usr/bin/env python3
"""
Find model 8340 records that are being detected as duplicates.
This script checks multiple database locations and investigates the duplicate detection logic.
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
from src.laser_trim_analyzer.database.models import AnalysisResult
from sqlalchemy import func, or_, and_, text
import json

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_all_database_locations():
    """Check all possible database locations for 8340 records."""
    logger.info("=" * 80)
    logger.info("CHECKING ALL DATABASE LOCATIONS")
    logger.info("=" * 80)
    
    # Possible database locations
    db_locations = [
        # Production paths
        Path("D:/LaserTrimData/production.db"),
        Path("C:/Users/Jayma/AppData/Local/LaserTrimAnalyzer/database/laser_trim_local.db"),
        
        # Development paths
        Path(os.environ.get('LOCALAPPDATA', '')) / "LaserTrimAnalyzer" / "dev" / "laser_trim_dev.db",
        Path(os.environ.get('USERPROFILE', '')) / "Documents" / "LaserTrimAnalyzer" / "dev" / "data" / "laser_trim_dev.db",
        
        # Legacy paths
        Path("laser_trim_analyzer.db"),
        Path("data/laser_trim_analyzer.db"),
    ]
    
    for db_path in db_locations:
        if db_path.exists():
            logger.info(f"\nFound database at: {db_path}")
            logger.info(f"Size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            # Try to connect and check for 8340 records
            try:
                from sqlalchemy import create_engine
                from sqlalchemy.orm import sessionmaker
                
                engine = create_engine(f"sqlite:///{db_path}")
                Session = sessionmaker(bind=engine)
                session = Session()
                
                # Check if analysis_results table exists
                result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results'"))
                if result.fetchone():
                    # Count 8340 records
                    count = session.execute(text("SELECT COUNT(*) FROM analysis_results WHERE model LIKE '%8340%'")).scalar()
                    logger.info(f"   8340 records found: {count}")
                    
                    if count > 0:
                        # Get some samples
                        samples = session.execute(text("""
                            SELECT id, model, serial, timestamp, filename 
                            FROM analysis_results 
                            WHERE model LIKE '%8340%' 
                            LIMIT 5
                        """)).fetchall()
                        
                        logger.info("   Sample records:")
                        for sample in samples:
                            logger.info(f"      ID: {sample[0]}, Model: {sample[1]}, Serial: {sample[2]}, Date: {sample[3]}, File: {sample[4]}")
                else:
                    logger.info("   No analysis_results table found")
                
                session.close()
                engine.dispose()
                
            except Exception as e:
                logger.error(f"   Error checking database: {str(e)}")
        else:
            logger.debug(f"Database not found at: {db_path}")


def check_duplicate_detection_logic():
    """Examine the duplicate detection logic in DatabaseManager."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMINING DUPLICATE DETECTION LOGIC")
    logger.info("=" * 80)
    
    # Get the duplicate detection method
    config = get_config()
    db_manager = DatabaseManager(config)
    
    # Test with a sample 8340 result
    test_metadata = {
        'model': '8340-1-82',
        'serial': 'TEST',
        'test_date': '2025-04-17 05:59:00'
    }
    
    logger.info("\nTesting duplicate detection for model '8340-1-82'...")
    
    try:
        with db_manager.get_session() as session:
            # Check using the actual duplicate detection query
            existing = session.query(AnalysisResult).filter(
                and_(
                    AnalysisResult.model == test_metadata['model'],
                    AnalysisResult.serial == test_metadata['serial']
                )
            ).first()
            
            if existing:
                logger.info(f"   Found existing record: ID={existing.id}, Date={existing.timestamp}")
            else:
                logger.info("   No existing record found")
                
            # Also check with just model
            model_only = session.query(AnalysisResult).filter(
                AnalysisResult.model.like('%8340%')
            ).count()
            logger.info(f"   Total records with '8340' in model: {model_only}")
            
    except Exception as e:
        logger.error(f"   Error in duplicate detection: {str(e)}")


def test_batch_export_query():
    """Test the exact query used in batch export to understand why it returns zeros."""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING BATCH EXPORT QUERY")
    logger.info("=" * 80)
    
    config = get_config()
    db_manager = DatabaseManager(config)
    
    try:
        with db_manager.get_session() as session:
            # Test different model formats
            model_formats = ['8340', '8340-1-82', '8340-1-83', '8340%']
            
            for model_format in model_formats:
                logger.info(f"\nTesting model format: '{model_format}'")
                
                # Exact match
                exact = session.query(AnalysisResult).filter(
                    AnalysisResult.model == model_format
                ).count()
                logger.info(f"   Exact match: {exact} records")
                
                # LIKE match
                like = session.query(AnalysisResult).filter(
                    AnalysisResult.model.like(f'%{model_format}%')
                ).count()
                logger.info(f"   LIKE match: {like} records")
                
                # Case-insensitive
                ilike = session.query(func.count(AnalysisResult.id)).filter(
                    func.lower(AnalysisResult.model).like(f'%{model_format.lower()}%')
                ).scalar()
                logger.info(f"   Case-insensitive: {ilike} records")
                
    except Exception as e:
        logger.error(f"Export query test error: {str(e)}")


def check_current_environment():
    """Check which database environment is currently active."""
    logger.info("\n" + "=" * 80)
    logger.info("CURRENT ENVIRONMENT CHECK")
    logger.info("=" * 80)
    
    # Check environment variable
    env = os.environ.get('LTA_ENV', 'production')
    logger.info(f"LTA_ENV: {env}")
    
    # Get config and check database path
    config = get_config()
    logger.info(f"Database path from config: {config.database.path}")
    
    # Check if database file exists
    db_path = Path(config.database.path)
    if db_path.exists():
        logger.info(f"Database file exists: YES ({db_path.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        logger.info(f"Database file exists: NO")


def main():
    """Main investigation function."""
    logger.info("Starting investigation of model 8340 duplicate detection issue...")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Check environment
    check_current_environment()
    
    # Check all database locations
    check_all_database_locations()
    
    # Check duplicate detection
    check_duplicate_detection_logic()
    
    # Test export queries
    test_batch_export_query()
    
    logger.info("\n" + "=" * 80)
    logger.info("INVESTIGATION SUMMARY")
    logger.info("=" * 80)
    logger.info("""
Based on the logs, model 8340 records ARE in the database but being skipped as duplicates.
The issue appears to be:

1. Records exist from previous batch processing (April 2025)
2. New batch processing detects these as duplicates and skips them
3. The export might be querying a different database or using different criteria

Recommendations:
1. Check if you're running in the correct environment (production vs development)
2. Verify the database path matches where the original records were saved
3. Consider if you want to update existing records instead of skipping duplicates
4. The export might need to handle the full model format (e.g., '8340-1-82' vs just '8340')
    """)


if __name__ == "__main__":
    main()