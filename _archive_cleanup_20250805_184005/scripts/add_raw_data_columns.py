#!/usr/bin/env python3
"""
Database migration script to add position_data and error_data columns to track_results table.

This script safely adds the new columns to existing databases without losing any data.
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text, inspect
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_database_url():
    """Get the database URL from environment or default location."""
    # Check environment variable first
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        return db_url
    
    # Default to user's home directory
    db_path = Path.home() / '.laser_trim_analyzer' / 'laser_trim.db'
    return f"sqlite:///{db_path}"

def add_raw_data_columns():
    """Add position_data and error_data columns if they don't exist."""
    try:
        # Create engine
        engine = create_engine(get_database_url())
        
        # Check if table exists
        inspector = inspect(engine)
        if 'track_results' not in inspector.get_table_names():
            logger.info("Table 'track_results' doesn't exist yet, migration not needed")
            return
        
        # Check if columns already exist
        existing_columns = [col['name'] for col in inspector.get_columns('track_results')]
        
        columns_to_add = []
        if 'position_data' not in existing_columns:
            columns_to_add.append('position_data')
        if 'error_data' not in existing_columns:
            columns_to_add.append('error_data')
        
        if not columns_to_add:
            logger.info("Columns already exist, no migration needed")
            return
        
        # Add missing columns
        with engine.begin() as conn:
            for column in columns_to_add:
                logger.info(f"Adding column: {column}")
                # SQLite doesn't have a JSON type, so we use TEXT
                try:
                    conn.execute(text(f"ALTER TABLE track_results ADD COLUMN {column} TEXT"))
                    logger.info(f"Successfully added column: {column}")
                except Exception as e:
                    logger.warning(f"Could not add column {column}: {e}")
        
        logger.info(f"Migration completed for columns: {', '.join(columns_to_add)}")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting database migration...")
    add_raw_data_columns()
    logger.info("Migration completed successfully!")