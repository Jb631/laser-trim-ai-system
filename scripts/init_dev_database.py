#!/usr/bin/env python3
"""
Initialize Development Database

This script sets up a fresh development database for the Laser Trim Analyzer.
It creates the necessary directories, initializes the database schema, and
optionally seeds test data.

Usage:
    python scripts/init_dev_database.py [--seed-data] [--clean]
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set development environment
os.environ['LTA_ENV'] = 'development'

from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import Base


def clean_development_environment():
    """Clean the development environment by removing existing data."""
    config = get_config()
    
    print("Cleaning Development Environment")
    print("=" * 80)
    
    # Get paths from config
    db_path_str = os.path.expandvars(str(config.database.path))
    db_path = Path(db_path_str)
    
    data_dir_str = os.path.expandvars(str(config.data_directory))
    data_dir = Path(data_dir_str)
    
    models_dir_str = os.path.expandvars(str(config.ml.model_path))
    models_dir = Path(models_dir_str)
    
    log_dir_str = os.path.expandvars(str(config.log_directory))
    log_dir = Path(log_dir_str)
    
    # List of paths to clean
    paths_to_clean = [
        (db_path, "Database"),
        (data_dir, "Data directory"),
        (models_dir, "ML models directory"),
        (log_dir, "Logs directory")
    ]
    
    for path, description in paths_to_clean:
        if path.exists():
            if path.is_file():
                print(f"  Removing {description}: {path}")
                path.unlink()
            elif path.is_dir():
                print(f"  Removing {description}: {path}")
                shutil.rmtree(path)
        else:
            print(f"  {description} does not exist: {path}")
    
    print("\n✓ Development environment cleaned")


def init_development_database(seed_data=False, clean=False):
    """Initialize the development database."""
    
    if clean:
        clean_development_environment()
        print()
    
    print("Initializing Development Database")
    print("=" * 80)
    
    # Get configuration
    config = get_config()
    
    print(f"Environment: {os.environ.get('LTA_ENV', 'production')}")
    print(f"Configuration loaded from: {config.app_name}")
    print(f"Debug mode: {config.debug}")
    
    # Display paths - config should already have expanded paths
    db_path_str = str(config.database.path)
    data_dir_str = str(config.data_directory)
    models_dir_str = str(config.ml.model_path)
    log_dir_str = str(config.log_directory)
    
    print(f"\nPaths:")
    print(f"  Database: {db_path_str}")
    print(f"  Data: {data_dir_str}")
    print(f"  Models: {models_dir_str}")
    print(f"  Logs: {log_dir_str}")
    
    # Create directories
    print("\n1. Creating directories...")
    for path_str, desc in [
        (db_path_str, "Database"),
        (data_dir_str, "Data"),
        (models_dir_str, "ML Models"),
        (log_dir_str, "Logs")
    ]:
        path = Path(path_str)
        if path.suffix:  # It's a file, create parent dir
            path = path.parent
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ {desc} directory: {path}")
        except Exception as e:
            print(f"   ✗ Failed to create {desc} directory: {e}")
            return False
    
    # Initialize database
    print("\n2. Initializing database...")
    try:
        db_manager = DatabaseManager(config, echo=config.database.echo)
        
        # Drop existing tables if cleaning
        if clean:
            print("   Dropping existing tables...")
            db_manager.init_db(drop_existing=True)
        else:
            # Just ensure tables exist
            db_manager.init_db(drop_existing=False)
        
        print("   ✓ Database initialized successfully")
        
        # Check database status
        with db_manager.get_session() as session:
            from laser_trim_analyzer.database.models import AnalysisResult
            count = session.query(AnalysisResult).count()
            print(f"   ✓ Database contains {count} analysis records")
        
    except Exception as e:
        print(f"   ✗ Failed to initialize database: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Seed test data if requested
    if seed_data:
        print("\n3. Seeding test data...")
        try:
            seed_test_data(db_manager)
            print("   ✓ Test data seeded successfully")
        except Exception as e:
            print(f"   ✗ Failed to seed test data: {e}")
            import traceback
            traceback.print_exc()
    
    # Create marker files
    print("\n4. Creating marker files...")
    
    # Create a README in the data directory
    readme_path = Path(data_dir_str) / "README.md"
    readme_content = f"""# Development Data Directory

This directory contains development data for the Laser Trim Analyzer.

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Environment: Development

## Structure

- `/data` - Analysis data files
- `/exports` - Excel exports
- `/plots` - Generated plots
- `/temp` - Temporary files

## Notes

This is a development environment. Data here should not be used for production.
"""
    
    try:
        readme_path.write_text(readme_content)
        print(f"   ✓ Created README at: {readme_path}")
    except Exception as e:
        print(f"   ✗ Failed to create README: {e}")
    
    print("\n✅ Development environment initialized successfully!")
    print("\nTo use this environment, set the environment variable:")
    print("  Windows: set LTA_ENV=development")
    print("  Linux/Mac: export LTA_ENV=development")
    print("\nThen run the application:")
    print("  python src/__main__.py")
    
    return True


def seed_test_data(db_manager):
    """Seed the database with test data."""
    from laser_trim_analyzer.core.models import (
        AnalysisResult, TrackData, AnalysisStatus, 
        SystemType, RiskCategory
    )
    import numpy as np
    
    # Create some test analysis results
    test_results = []
    
    models = ["8340", "8555", "8575"]
    
    for i in range(5):
        for model in models:
            # Create mock track data
            tracks = []
            for track_num in range(1, 6):
                track = TrackData(
                    track_number=track_num,
                    pre_trim_mean=100.0 + np.random.normal(0, 5),
                    pre_trim_std=2.0 + np.random.normal(0, 0.5),
                    post_trim_mean=150.0 + np.random.normal(0, 3),
                    post_trim_std=1.0 + np.random.normal(0, 0.2),
                    drift_percentage=np.random.uniform(-0.5, 0.5),
                    units_tested=1000,
                    units_passed=int(950 + np.random.randint(-50, 50)),
                    yield_percentage=95.0 + np.random.uniform(-5, 5),
                    cpk=1.5 + np.random.uniform(-0.3, 0.5),
                    risk_category=np.random.choice(list(RiskCategory))
                )
                tracks.append(track)
            
            result = AnalysisResult(
                filename=f"test_{model}_batch_{i+1}.xlsx",
                timestamp=datetime.now(),
                model=model,
                serial=f"TEST{i+1:04d}",
                date_of_trim=datetime.now(),
                tracks=tracks,
                num_zones=5,
                zone_results={},
                filter_parameters={
                    "sampling_frequency": 100,
                    "cutoff_frequency": 40
                },
                sigma_thresholds={"scaling_factor": 24.0},
                overall_yield=95.0 + np.random.uniform(-5, 5),
                risk_assessment={
                    "overall_risk": np.random.choice(["low", "medium", "high"]),
                    "risk_factors": []
                },
                warnings=[],
                position_data=None,
                error_data=None,
                system_type=SystemType.FIVE_TRACK,
                status=AnalysisStatus.COMPLETED
            )
            
            test_results.append(result)
    
    # Save to database
    for result in test_results:
        db_manager.save_analysis(result)
    
    print(f"   Added {len(test_results)} test analysis records")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize development database for Laser Trim Analyzer"
    )
    parser.add_argument(
        "--seed-data",
        action="store_true",
        help="Seed the database with test data"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing development environment before initializing"
    )
    
    args = parser.parse_args()
    
    success = init_development_database(
        seed_data=args.seed_data,
        clean=args.clean
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())