"""
Laser Trim AI System - Integration Examples
==========================================

This script demonstrates how to use the integrated system
for various workflows and use cases.

Author: QA Team
Date: 2024
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Import the orchestrator
from laser_trim_orchestrator import (
    LaserTrimOrchestrator,
    create_orchestrator,
    process_with_full_pipeline
)


def example_1_basic_processing():
    """Example 1: Basic folder processing with all features."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Folder Processing")
    print("=" * 60)

    # Process folder with full pipeline
    results = process_with_full_pipeline(
        input_folder="data/laser_trim_files",
        output_folder="output/basic_analysis"
    )

    # Display results
    print(f"\n✓ Processing Complete!")
    print(f"  - Total files: {results['total_files']}")
    print(f"  - Successful: {results['successful']}")
    print(f"  - Failed: {results['failed']}")
    print(f"  - Processing time: {results['processing_time']:.2f}s")
    print(f"  - Report: {results['report_path']}")

    # Show individual file results
    print("\nFile Results:")
    for file_result in results['results'][:5]:  # Show first 5
        print(f"  - {file_result['filename']}: {file_result['status']}")
        if 'tracks' in file_result:
            for track_id, track_data in file_result['tracks'].items():
                print(f"    • {track_id}: σ={track_data['sigma_gradient']:.4f}, "
                      f"Pass={track_data['sigma_pass']}")


def example_2_custom_configuration():
    """Example 2: Processing with custom configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 60)

    # Create orchestrator with custom settings
    orchestrator = LaserTrimOrchestrator(
        config_path="config/custom_config.json",
        enable_parallel=True,  # Enable parallel processing
        enable_ml=True,  # Enable ML analysis
        enable_db=False  # Disable database (for speed)
    )

    # Process folder
    results = orchestrator.process_folder(
        "data/test_files",
        output_dir="output/custom_analysis",
        generate_report=True,
        report_name="custom_report.xlsx"
    )

    print(f"\n✓ Custom processing complete!")
    print(f"  - Files processed: {results['successful']}")
    print(f"  - Average time per file: {results['average_time_per_file']:.2f}s")

    # Clean up
    orchestrator.cleanup()


def example_3_ml_training():
    """Example 3: Train ML models with historical data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: ML Model Training")
    print("=" * 60)

    orchestrator = create_orchestrator()

    # Option 1: Train from database (last 90 days)
    if orchestrator.enable_db:
        print("\nTraining from database...")
        results = orchestrator.train_ml_models(days_back=90)

        if 'error' not in results:
            print(f"✓ Models trained successfully!")
            print(f"  - Threshold optimizer MAE: {results['threshold_optimizer']['mae']:.4f}")
            print(f"  - Failure predictor accuracy: {results['failure_predictor']['accuracy']:.2%}")
            print(f"  - Models saved to: {results['saved_version']}")

    # Option 2: Train from CSV file
    else:
        print("\nTraining from CSV file...")
        results = orchestrator.train_ml_models(
            historical_data_path="data/historical_data.csv"
        )

    orchestrator.cleanup()


def example_4_real_time_monitoring():
    """Example 4: Real-time monitoring workflow."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Real-Time Monitoring")
    print("=" * 60)

    orchestrator = create_orchestrator()

    # Monitor a directory for new files
    watch_dir = Path("data/incoming")
    processed_dir = Path("data/processed")

    print(f"\nMonitoring {watch_dir} for new files...")
    print("Press Ctrl+C to stop\n")

    try:
        import time
        while True:
            # Check for new Excel files
            new_files = list(watch_dir.glob("*.xls*"))

            if new_files:
                print(f"\nFound {len(new_files)} new files!")

                for file_path in new_files:
                    # Process single file
                    result = orchestrator._process_single_file(file_path)

                    # Display result
                    status = "✓" if result.status.value == "completed" else "✗"
                    print(f"{status} {file_path.name}: {result.status.value}")

                    if result.ml_result and 'predictions' in result.ml_result:
                        # Check for high risk
                        for track_id, pred in result.ml_result['predictions'].items():
                            risk = pred.get('failure_risk', {})
                            if risk.get('risk_level') in ['HIGH', 'CRITICAL']:
                                print(f"  ⚠️  ALERT: {track_id} - High failure risk!")

                    # Move processed file
                    processed_dir.mkdir(exist_ok=True)
                    file_path.rename(processed_dir / file_path.name)

            time.sleep(5)  # Check every 5 seconds

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

    orchestrator.cleanup()


def example_5_batch_analysis_with_trends():
    """Example 5: Batch analysis with trend reporting."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Batch Analysis with Trends")
    print("=" * 60)

    orchestrator = create_orchestrator()

    # Process batch
    print("\nProcessing batch...")
    results = orchestrator.process_folder(
        "data/production_batch_20240115",
        output_dir="output/batch_analysis"
    )

    # Generate trend report
    if orchestrator.enable_db:
        print("\nGenerating trend report...")
        trend_report = orchestrator.generate_trend_report(
            output_dir="output/batch_analysis",
            days_back=30
        )

        if trend_report:
            print(f"✓ Trend report generated: {trend_report}")

    # Get system metrics
    status = orchestrator.get_system_status()

    print("\n📊 System Metrics:")
    print(f"  - Files processed: {status['metrics']['files_processed']}")
    print(f"  - ML predictions: {status['metrics']['ml_predictions_made']}")
    print(f"  - Reports generated: {status['metrics']['reports_generated']}")
    print(f"  - Total processing time: {status['metrics']['total_processing_time']:.1f}s")

    orchestrator.cleanup()


def example_6_legacy_data_import():
    """Example 6: Import legacy data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Legacy Data Import")
    print("=" * 60)

    orchestrator = create_orchestrator()

    if not orchestrator.enable_db:
        print("Database not enabled - skipping example")
        return

    # Import from Excel
    print("\nImporting from legacy Excel file...")
    result = orchestrator.import_legacy_data("data/legacy/historical_results.xlsx")

    if result.get('success'):
        print(f"✓ Imported {result['imported']} records")
        print(f"  - Failed: {result['failed']}")
        print(f"  - Run ID: {result['run_id']}")

    # Import from directory
    print("\nImporting from legacy directory...")
    result = orchestrator.import_legacy_data("data/legacy/2023_results")

    if 'total_imported' in result:
        print(f"✓ Imported {result['total_imported']} total records")
        print(f"  - Excel files: {result['excel_files']}")
        print(f"  - JSON files: {result['json_files']}")
        print(f"  - CSV files: {result['csv_files']}")

    orchestrator.cleanup()


def example_7_production_workflow():
    """Example 7: Complete production workflow."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Production Workflow")
    print("=" * 60)

    # This example shows a complete production workflow

    # Step 1: Initialize system
    print("\n1. Initializing system...")
    orchestrator = create_orchestrator(config_path="config/production_config.json")

    # Step 2: Check system status
    print("\n2. Checking system status...")
    status = orchestrator.get_system_status()

    all_ready = all(
        status['components'][comp] == 'active'
        for comp in ['data_processor', 'ml_models', 'database', 'report_generator']
    )

    if not all_ready:
        print("⚠️  Some components are not active!")
        for comp, state in status['components'].items():
            print(f"  - {comp}: {state}")
    else:
        print("✓ All systems ready!")

    # Step 3: Process today's production
    print("\n3. Processing today's production files...")
    today = datetime.now().strftime('%Y%m%d')

    results = orchestrator.process_folder(
        f"data/production/{today}",
        output_dir=f"output/production/{today}",
        generate_report=True
    )

    # Step 4: Check for issues
    print("\n4. Checking for quality issues...")

    high_risk_count = 0
    drift_detected = False

    for file_result in results['results']:
        if file_result['status'] == 'completed':
            # Check each file's ML results
            # (In real implementation, this would be in the results)
            pass

    # Step 5: Generate daily summary
    print("\n5. Generating daily summary...")

    summary = {
        'date': today,
        'total_units': results['total_files'],
        'pass_rate': results['successful'] / results['total_files'] if results['total_files'] > 0 else 0,
        'high_risk_units': high_risk_count,
        'drift_detected': drift_detected,
        'report_location': results['report_path']
    }

    # Save summary
    summary_path = Path(f"output/production/{today}/daily_summary.json")
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Production workflow complete!")
    print(f"  Daily summary: {summary_path}")

    # Step 6: Optional - Retrain models if needed
    if results['successful'] > 100:  # If we have enough new data
        print("\n6. Retraining ML models with new data...")
        train_results = orchestrator.train_ml_models(days_back=7)

        if 'error' not in train_results:
            print("✓ Models retrained with latest data")

    orchestrator.cleanup()


def main():
    """Run examples based on user selection."""
    print("\n" + "=" * 80)
    print("LASER TRIM AI SYSTEM - INTEGRATION EXAMPLES")
    print("=" * 80)

    examples = {
        '1': ('Basic Processing', example_1_basic_processing),
        '2': ('Custom Configuration', example_2_custom_configuration),
        '3': ('ML Training', example_3_ml_training),
        '4': ('Real-Time Monitoring', example_4_real_time_monitoring),
        '5': ('Batch Analysis with Trends', example_5_batch_analysis_with_trends),
        '6': ('Legacy Data Import', example_6_legacy_data_import),
        '7': ('Production Workflow', example_7_production_workflow)
    }

    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    print("\n  0. Run all examples (except monitoring)")
    print("  q. Quit")

    choice = input("\nSelect example (0-7 or q): ").strip().lower()

    if choice == 'q':
        print("Exiting...")
        return
    elif choice == '0':
        # Run all except real-time monitoring
        for key, (name, func) in examples.items():
            if key != '4':  # Skip monitoring
                try:
                    func()
                except Exception as e:
                    print(f"\n❌ Error in {name}: {str(e)}")
    elif choice in examples:
        try:
            examples[choice][1]()
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    # Ensure required directories exist
    for dir_name in ['data', 'output', 'logs', 'config']:
        Path(dir_name).mkdir(exist_ok=True)

    main()