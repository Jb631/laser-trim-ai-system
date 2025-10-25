#!/usr/bin/env python3
"""
Check ML training data in the database to verify sample counts.

This script examines the database to show exactly how many training
samples are available for ML model training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from src.laser_trim_analyzer.database.manager import DatabaseManager
from src.laser_trim_analyzer.core.config import get_config
from src.laser_trim_analyzer.database.models import AnalysisRecord, Track

def check_training_data():
    """Check and display training data statistics."""
    config = get_config()
    db_manager = DatabaseManager(config)
    
    print("ML Training Data Analysis")
    print("=" * 60)
    
    try:
        # Get all historical records
        all_records = db_manager.get_historical_data(days=0)  # 0 = all data
        print(f"\nTotal analysis records in database: {len(all_records)}")
        
        # Count records by time period
        now = datetime.now()
        periods = {
            "Last 7 days": 7,
            "Last 30 days": 30,
            "Last 90 days": 90,
            "Last 180 days": 180,
            "Last 365 days": 365
        }
        
        print("\nRecords by time period:")
        for period_name, days in periods.items():
            cutoff = now - timedelta(days=days)
            recent_records = [r for r in all_records if r.file_date and r.file_date >= cutoff]
            print(f"  {period_name}: {len(recent_records)} records")
        
        # Analyze tracks per record
        total_tracks = 0
        records_with_tracks = 0
        records_without_tracks = 0
        track_counts = {}
        model_counts = {}
        
        print("\nDetailed analysis:")
        for record in all_records:
            if hasattr(record, 'tracks') and record.tracks:
                num_tracks = len(record.tracks)
                total_tracks += num_tracks
                records_with_tracks += 1
                
                # Track distribution
                track_counts[num_tracks] = track_counts.get(num_tracks, 0) + 1
                
                # Model distribution
                model = record.model or 'Unknown'
                if model not in model_counts:
                    model_counts[model] = {'records': 0, 'tracks': 0}
                model_counts[model]['records'] += 1
                model_counts[model]['tracks'] += num_tracks
            else:
                records_without_tracks += 1
        
        print(f"\nTrack statistics:")
        print(f"  Records with tracks: {records_with_tracks}")
        print(f"  Records without tracks: {records_without_tracks}")
        print(f"  Total tracks (training samples): {total_tracks}")
        
        if records_with_tracks > 0:
            avg_tracks = total_tracks / records_with_tracks
            print(f"  Average tracks per record: {avg_tracks:.2f}")
        
        print(f"\nTrack count distribution:")
        for track_count in sorted(track_counts.keys()):
            print(f"  {track_count} tracks: {track_counts[track_count]} records")
        
        print(f"\nModel distribution:")
        for model in sorted(model_counts.keys()):
            stats = model_counts[model]
            print(f"  {model}: {stats['records']} records, {stats['tracks']} tracks")
        
        # Check if 794 samples is plausible
        print(f"\n794 samples analysis:")
        if total_tracks > 0:
            print(f"  Actual total tracks: {total_tracks}")
            if total_tracks == 794:
                print("  ✓ The 794 samples count is EXACT!")
            elif abs(total_tracks - 794) < 50:
                print(f"  ~ The 794 samples count is CLOSE (difference: {abs(total_tracks - 794)})")
            else:
                print(f"  ✗ The 794 samples count differs significantly (difference: {abs(total_tracks - 794)})")
        
        # Sample some actual data
        print(f"\nSample of first 5 records with tracks:")
        sample_count = 0
        for record in all_records[:20]:  # Check first 20 records
            if hasattr(record, 'tracks') and record.tracks and sample_count < 5:
                print(f"\n  Record ID: {record.id}")
                print(f"    Date: {record.file_date}")
                print(f"    Model: {record.model}")
                print(f"    Serial: {record.serial}")
                print(f"    Tracks: {len(record.tracks)}")
                for track in record.tracks[:2]:  # Show first 2 tracks
                    print(f"      - Track {track.track_id}: sigma={track.sigma_gradient:.6f}")
                sample_count += 1
        
    except Exception as e:
        print(f"\nError analyzing training data: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db_manager.close()

if __name__ == "__main__":
    check_training_data()