#!/usr/bin/env python3
"""Test script to verify home page refresh after analysis."""

import time
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.core.config import get_config

def test_timestamp_consistency():
    """Test that timestamps are handled consistently."""
    
    print("Testing timestamp consistency...")
    
    # Get database manager
    config = get_config()
    db_manager = DatabaseManager(config.database)
    
    # Check current UTC time
    utc_now = datetime.now(timezone.utc)
    local_now = datetime.now()
    
    print(f"Current UTC time: {utc_now}")
    print(f"Current local time: {local_now}")
    print(f"Timezone offset: {local_now.astimezone().utcoffset()}")
    
    # Query for today's data using UTC
    today_start_utc = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"\nQuerying for data from UTC: {today_start_utc}")
    
    results = db_manager.get_historical_data(
        start_date=today_start_utc,
        include_tracks=False
    )
    
    print(f"Found {len(results)} results for today (UTC)")
    
    if results:
        # Show first few results with timestamps
        for i, result in enumerate(results[:5]):
            print(f"\nResult {i+1}:")
            print(f"  Model: {result.model}")
            print(f"  Serial: {result.serial}")
            print(f"  Timestamp (UTC): {result.timestamp}")
            print(f"  Timestamp (local): {result.timestamp.replace(tzinfo=timezone.utc).astimezone()}")
            print(f"  Status: {result.overall_status}")
    
    # Also check with days_back parameter
    print("\n\nChecking with days_back=1...")
    results_days_back = db_manager.get_historical_data(
        days_back=1,
        include_tracks=False
    )
    print(f"Found {len(results_days_back)} results from last 24 hours")
    
    # Close database connection
    db_manager.close()
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_timestamp_consistency()