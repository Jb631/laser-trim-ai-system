#!/usr/bin/env python3
"""
Fix All Database Enum Values

This script updates all enum values in the database to their proper case.
It handles risk categories, status types, and any other enum fields.

Usage:
    python scripts/fix_database_enums.py [--dry-run]
"""

import os
import sys
import argparse
from pathlib import Path
import sqlite3
import logging

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from laser_trim_analyzer.core.config import get_config
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you've run: pip install -e .")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_database_enums(dry_run=False):
    """Fix all enum values in the database."""
    config = get_config()
    
    print("=" * 80)
    print("Database Enum Value Fixer")
    print("=" * 80)
    
    if dry_run:
        print("DRY RUN MODE - No changes will be made")
    
    # Get database path
    db_path = os.path.expandvars(str(config.database.path))
    
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return
    
    print(f"Database: {db_path}")
    print()
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Define all enum fixes
    enum_fixes = [
        # Risk Category fixes in track_results
        {
            'table': 'track_results',
            'column': 'risk_category',
            'fixes': [
                ("'high'", "'HIGH'"),
                ("'medium'", "'MEDIUM'"),
                ("'low'", "'LOW'"),
                ("'unknown'", "'UNKNOWN'"),
                ("'High'", "'HIGH'"),
                ("'Medium'", "'MEDIUM'"),
                ("'Low'", "'LOW'"),
                ("'Unknown'", "'UNKNOWN'"),
            ]
        },
        # Risk Category fixes in prediction_risk
        {
            'table': 'prediction_risk',
            'column': 'predicted_risk_category',
            'fixes': [
                ("'high'", "'HIGH'"),
                ("'medium'", "'MEDIUM'"),
                ("'low'", "'LOW'"),
                ("'unknown'", "'UNKNOWN'"),
                ("'High'", "'HIGH'"),
                ("'Medium'", "'MEDIUM'"),
                ("'Low'", "'LOW'"),
                ("'Unknown'", "'UNKNOWN'"),
            ]
        },
        # Status Type fixes
        {
            'table': 'track_results',
            'column': 'status',
            'fixes': [
                ("'pass'", "'PASS'"),
                ("'fail'", "'FAIL'"),
                ("'warning'", "'WARNING'"),
                ("'error'", "'ERROR'"),
                ("'Pass'", "'PASS'"),
                ("'Fail'", "'FAIL'"),
                ("'Warning'", "'WARNING'"),
                ("'Error'", "'ERROR'"),
            ]
        },
        {
            'table': 'analysis_results',
            'column': 'overall_status',
            'fixes': [
                ("'pass'", "'PASS'"),
                ("'fail'", "'FAIL'"),
                ("'warning'", "'WARNING'"),
                ("'error'", "'ERROR'"),
                ("'Pass'", "'PASS'"),
                ("'Fail'", "'FAIL'"),
                ("'Warning'", "'WARNING'"),
                ("'Error'", "'ERROR'"),
                ("'pending'", "'PROCESSING_FAILED'"),
                ("'Pending'", "'PROCESSING_FAILED'"),
                ("''", "'ERROR'"),  # Empty strings to ERROR
            ]
        },
        # System Type fixes
        {
            'table': 'analysis_results',
            'column': 'system_type',
            'fixes': [
                ("'a'", "'A'"),
                ("'b'", "'B'"),
                ("'unknown'", "'UNKNOWN'"),
                ("'Unknown'", "'UNKNOWN'"),
            ]
        },
        # Alert Type fixes
        {
            'table': 'qa_alerts',
            'column': 'alert_type',
            'fixes': [
                ("'carbon_screen'", "'CARBON_SCREEN'"),
                ("'high_risk'", "'HIGH_RISK'"),
                ("'drift_detected'", "'DRIFT_DETECTED'"),
                ("'threshold_exceeded'", "'THRESHOLD_EXCEEDED'"),
                ("'maintenance_required'", "'MAINTENANCE_REQUIRED'"),
                ("'sigma_fail'", "'SIGMA_FAIL'"),
                ("'process_error'", "'PROCESS_ERROR'"),
            ]
        }
    ]
    
    total_fixes = 0
    
    for enum_fix in enum_fixes:
        table = enum_fix['table']
        column = enum_fix['column']
        
        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
        if not cursor.fetchone():
            print(f"Table '{table}' does not exist, skipping...")
            continue
        
        print(f"\nFixing {column} in {table}:")
        
        for old_value, new_value in enum_fix['fixes']:
            # Count affected rows
            count_query = f"SELECT COUNT(*) FROM {table} WHERE {column} = {old_value}"
            cursor.execute(count_query)
            count = cursor.fetchone()[0]
            
            if count > 0:
                print(f"  Found {count} rows with {column} = {old_value}")
                
                if not dry_run:
                    # Update the values
                    update_query = f"UPDATE {table} SET {column} = {new_value} WHERE {column} = {old_value}"
                    cursor.execute(update_query)
                    print(f"  ✓ Updated to {new_value}")
                else:
                    print(f"  Would update to {new_value}")
                
                total_fixes += count
    
    # Also check for NULL values that should be set to defaults
    null_fixes = [
        ('track_results', 'risk_category', "'UNKNOWN'"),
        ('track_results', 'status', "'ERROR'"),
        ('analysis_results', 'overall_status', "'ERROR'"),
        ('analysis_results', 'system_type', "'UNKNOWN'"),
    ]
    
    print("\nChecking for NULL values:")
    for table, column, default_value in null_fixes:
        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
        if not cursor.fetchone():
            continue
            
        count_query = f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL"
        cursor.execute(count_query)
        count = cursor.fetchone()[0]
        
        if count > 0:
            print(f"  Found {count} NULL values in {table}.{column}")
            
            if not dry_run:
                update_query = f"UPDATE {table} SET {column} = {default_value} WHERE {column} IS NULL"
                cursor.execute(update_query)
                print(f"  ✓ Set to {default_value}")
            else:
                print(f"  Would set to {default_value}")
            
            total_fixes += count
    
    if not dry_run:
        conn.commit()
        print(f"\n✓ Database updated successfully! Fixed {total_fixes} values.")
    else:
        print(f"\nDry run complete. Would fix {total_fixes} values.")
    
    conn.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Fix database enum values')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be fixed without making changes')
    args = parser.parse_args()
    
    fix_database_enums(dry_run=args.dry_run)


if __name__ == "__main__":
    main()