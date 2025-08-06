#!/usr/bin/env python3
"""Test script to verify the tracks.values() fix."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import DBAnalysisResult

def test_tracks_access():
    """Test accessing tracks from database results."""
    # Initialize database
    db = DatabaseManager(":memory:")
    
    # Get some results (may be empty)
    with db.get_session() as session:
        results = session.query(DBAnalysisResult).limit(5).all()
        
        print(f"Found {len(results)} results in database")
        
        for result in results:
            print(f"\nResult: {result.model} - {result.serial}")
            print(f"  tracks type: {type(result.tracks)}")
            print(f"  tracks length: {len(result.tracks)}")
            
            # Test the fix - this should work with lists
            if isinstance(result.tracks, dict):
                print("  ERROR: tracks is a dict (unexpected for DB results)")
                tracks_iter = result.tracks.values()
            else:
                print("  OK: tracks is a list (expected for DB results)")
                tracks_iter = result.tracks
                
            # Iterate through tracks
            for i, track in enumerate(tracks_iter):
                print(f"  Track {i}: {track.track_id} - Status: {track.status}")

if __name__ == "__main__":
    test_tracks_access()