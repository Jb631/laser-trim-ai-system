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
        SystemType, RiskCategory, FileMetadata, UnitProperties,
        SigmaAnalysis, LinearityAnalysis, ResistanceAnalysis,
        FailurePrediction, TrimEffectiveness, ValidationStatus
    )
    import numpy as np
    from pathlib import Path
    
    # Create some test analysis results
    test_results = []
    
    models = ["8340", "8555", "8575"]
    
    for i in range(5):
        for model in models:
            # Create file metadata - use temp directory
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "laser_trim_test_data"
            temp_dir.mkdir(exist_ok=True)
            test_file = temp_dir / f"test_{model}_batch_{i+1}.xlsx"
            # Create empty file
            test_file.touch()
            
            metadata = FileMetadata(
                filename=f"test_{model}_batch_{i+1}.xlsx",
                file_path=test_file,
                file_date=datetime.now(),
                model=model,
                serial=f"TEST{i+1:04d}",
                system=SystemType.SYSTEM_A if i % 2 == 0 else SystemType.SYSTEM_B,
                has_multi_tracks=True if model == "8575" else False
            )
            
            # Create mock track data
            tracks = {}
            num_tracks = 5 if model == "8575" else 1
            
            for track_num in range(1, num_tracks + 1):
                track_id = f"TRK{track_num}" if num_tracks > 1 else "default"
                
                # Generate some test data
                travel_length = 100.0 + np.random.uniform(-10, 10)
                num_points = 1000
                positions = np.linspace(0, travel_length, num_points).tolist()
                errors = (np.random.normal(0, 0.5, num_points) + np.sin(np.linspace(0, 2*np.pi, num_points)) * 0.3).tolist()
                
                # Create unit properties
                unit_props = UnitProperties(
                    unit_length=travel_length,
                    untrimmed_resistance=10000.0 + np.random.normal(0, 100),
                    trimmed_resistance=10500.0 + np.random.normal(0, 100)
                )
                
                # Create sigma analysis
                sigma = SigmaAnalysis(
                    sigma_gradient=0.8 + np.random.uniform(-0.2, 0.2),
                    sigma_threshold=1.0,
                    sigma_pass=np.random.choice([True, False], p=[0.9, 0.1]),
                    gradient_margin=0.2 + np.random.uniform(-0.1, 0.1),
                    scaling_factor=24.0,
                    validation_status=ValidationStatus.VALIDATED
                )
                
                # Create linearity analysis
                linearity = LinearityAnalysis(
                    linearity_spec=0.5,
                    optimal_offset=0.1 + np.random.uniform(-0.05, 0.05),
                    final_linearity_error_raw=0.3 + np.random.uniform(-0.1, 0.1),
                    final_linearity_error_shifted=0.2 + np.random.uniform(-0.1, 0.1),
                    linearity_pass=np.random.choice([True, False], p=[0.85, 0.15]),
                    linearity_fail_points=np.random.randint(0, 5),
                    validation_status=ValidationStatus.VALIDATED
                )
                
                # Create resistance analysis
                resistance = ResistanceAnalysis(
                    untrimmed_resistance=unit_props.untrimmed_resistance,
                    trimmed_resistance=unit_props.trimmed_resistance,
                    resistance_change=unit_props.resistance_change,
                    resistance_change_percent=unit_props.resistance_change_percent,
                    validation_status=ValidationStatus.VALIDATED
                )
                
                # Create failure prediction
                import random
                risk_cat = random.choice([RiskCategory.HIGH, RiskCategory.MEDIUM, RiskCategory.LOW])
                failure_pred = FailurePrediction(
                    failure_probability=np.random.uniform(0.01, 0.2),
                    risk_category=risk_cat,
                    gradient_margin=sigma.gradient_margin,
                    contributing_factors={"sigma": 0.4, "linearity": 0.3, "resistance": 0.3}
                )
                
                # Create trim effectiveness
                trim_effect = TrimEffectiveness(
                    improvement_percent=60.0 + np.random.uniform(-20, 20),
                    untrimmed_rms_error=0.8 + np.random.uniform(-0.2, 0.2),
                    trimmed_rms_error=0.3 + np.random.uniform(-0.1, 0.1),
                    max_error_reduction_percent=70.0 + np.random.uniform(-10, 10),
                    validation_status=ValidationStatus.VALIDATED
                )
                
                # Determine status
                if sigma.sigma_pass and linearity.linearity_pass:
                    status = AnalysisStatus.PASS
                    status_reason = ""
                elif not sigma.sigma_pass:
                    status = AnalysisStatus.FAIL
                    status_reason = "Sigma gradient exceeds threshold"
                else:
                    status = AnalysisStatus.WARNING
                    status_reason = "Linearity test failed"
                
                track = TrackData(
                    track_id=track_id,
                    status=status,
                    status_reason=status_reason,
                    travel_length=travel_length,
                    position_data=positions,
                    error_data=errors,
                    upper_limits=[1.0] * num_points,
                    lower_limits=[-1.0] * num_points,
                    unit_properties=unit_props,
                    sigma_analysis=sigma,
                    linearity_analysis=linearity,
                    resistance_analysis=resistance,
                    failure_prediction=failure_pred,
                    trim_effectiveness=trim_effect,
                    overall_validation_status=ValidationStatus.VALIDATED
                )
                
                tracks[track_id] = track
            
            # Determine overall status
            all_pass = all(track.status == AnalysisStatus.PASS for track in tracks.values())
            any_fail = any(track.status == AnalysisStatus.FAIL for track in tracks.values())
            
            if all_pass:
                overall_status = AnalysisStatus.PASS
            elif any_fail:
                overall_status = AnalysisStatus.FAIL
            else:
                overall_status = AnalysisStatus.WARNING
            
            result = AnalysisResult(
                metadata=metadata,
                overall_status=overall_status,
                processing_time=np.random.uniform(0.5, 2.0),
                tracks=tracks,
                overall_validation_status=ValidationStatus.VALIDATED
            )
            
            test_results.append(result)
    
    # Save to database
    saved_count = 0
    for result in test_results:
        try:
            db_manager.save_analysis(result)
            saved_count += 1
        except Exception as e:
            print(f"   ✗ Failed to save test record: {e}")
    
    print(f"   ✓ Added {saved_count} test analysis records")


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