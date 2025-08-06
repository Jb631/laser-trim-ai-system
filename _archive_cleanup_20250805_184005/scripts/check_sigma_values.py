#!/usr/bin/env python3
"""
Check sigma gradient values in the database and identify any issues.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from laser_trim_analyzer.database.models import Base, TrackResult, AnalysisResult
from laser_trim_analyzer.core.config import load_config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_sigma_values(env='development'):
    """Check sigma gradient values in the database."""
    
    # Load configuration
    config = load_config(env)
    db_path = config.database.path
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found at: {db_path}")
        return
    
    logger.info(f"Checking database at: {db_path}")
    
    # Create engine and session
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get total track count
        total_tracks = session.query(func.count(TrackResult.id)).scalar()
        logger.info(f"Total tracks in database: {total_tracks}")
        
        # Check for NULL sigma gradients
        null_sigma = session.query(func.count(TrackResult.id)).filter(
            TrackResult.sigma_gradient.is_(None)
        ).scalar()
        logger.info(f"Tracks with NULL sigma gradient: {null_sigma}")
        
        # Check for zero sigma gradients
        zero_sigma = session.query(func.count(TrackResult.id)).filter(
            TrackResult.sigma_gradient == 0
        ).scalar()
        logger.info(f"Tracks with zero sigma gradient: {zero_sigma}")
        
        # Check for very small sigma gradients (< 0.0001)
        small_sigma = session.query(func.count(TrackResult.id)).filter(
            TrackResult.sigma_gradient < 0.0001,
            TrackResult.sigma_gradient > 0
        ).scalar()
        logger.info(f"Tracks with very small sigma gradient (0 < σ < 0.0001): {small_sigma}")
        
        # Get statistics for non-zero values
        non_zero_tracks = session.query(TrackResult).filter(
            TrackResult.sigma_gradient > 0
        ).all()
        
        if non_zero_tracks:
            sigma_values = [track.sigma_gradient for track in non_zero_tracks]
            import numpy as np
            
            logger.info(f"\nStatistics for non-zero sigma gradients ({len(sigma_values)} tracks):")
            logger.info(f"  Min: {np.min(sigma_values):.6f}")
            logger.info(f"  Max: {np.max(sigma_values):.6f}")
            logger.info(f"  Mean: {np.mean(sigma_values):.6f}")
            logger.info(f"  Std: {np.std(sigma_values):.6f}")
            logger.info(f"  Median: {np.median(sigma_values):.6f}")
            
            # Show distribution
            logger.info("\nDistribution of sigma values:")
            ranges = [(0, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, float('inf'))]
            for low, high in ranges:
                count = len([s for s in sigma_values if low <= s < high])
                pct = (count / len(sigma_values)) * 100
                logger.info(f"  [{low:.3f}, {high:.3f}): {count} ({pct:.1f}%)")
        
        # Check specific models
        logger.info("\nChecking by model:")
        models = session.query(AnalysisResult.model).distinct().all()
        
        for (model,) in models:
            if model:
                tracks = session.query(TrackResult).join(AnalysisResult).filter(
                    AnalysisResult.model == model
                ).all()
                
                if tracks:
                    sigma_values = [t.sigma_gradient for t in tracks if t.sigma_gradient is not None]
                    zero_count = len([t for t in tracks if t.sigma_gradient == 0])
                    
                    if sigma_values:
                        logger.info(f"\n  Model {model}:")
                        logger.info(f"    Total tracks: {len(tracks)}")
                        logger.info(f"    Zero values: {zero_count} ({zero_count/len(tracks)*100:.1f}%)")
                        logger.info(f"    Non-zero mean: {np.mean([s for s in sigma_values if s > 0]):.6f}")
                        
                        # Show some sample files with zero values
                        if zero_count > 0:
                            zero_tracks = session.query(TrackResult, AnalysisResult).join(
                                AnalysisResult
                            ).filter(
                                AnalysisResult.model == model,
                                TrackResult.sigma_gradient == 0
                            ).limit(5).all()
                            
                            logger.info(f"    Sample files with zero sigma:")
                            for track, analysis in zero_tracks:
                                logger.info(f"      - {analysis.filename} (Track: {track.track_id})")
        
    finally:
        session.close()
        engine.dispose()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check sigma gradient values in database')
    parser.add_argument('--env', default='development', help='Environment to use')
    args = parser.parse_args()
    
    check_sigma_values(args.env)