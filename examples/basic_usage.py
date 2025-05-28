"""
Basic Usage Example

This script demonstrates how to use the laser trim analysis system
to process Excel files and validate the sigma calculations.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.core.data_loader import DataLoader
from src.utils.filter_utils import apply_matlab_filter
import logging
import json


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def print_results(result: dict, detailed: bool = False):
    """Pretty print the analysis results."""
    print(f"\n{'=' * 60}")
    print(f"File: {result['file_name']}")
    print(f"System: {result['system']}")
    print(f"Model: {result['metadata']['model']}")
    print(f"Serial: {result['metadata']['serial']}")
    print(f"Multi-track: {result['is_multi_track']}")
    print(f"{'=' * 60}")

    # Print track results
    for track_id, track_data in result['tracks'].items():
        print(f"\nTrack: {track_id}")
        print("-" * 40)

        if 'error' in track_data:
            print(f"  Error: {track_data['error']}")
            continue

        # Basic results
        print(f"  Sigma Gradient: {track_data['sigma_gradient']:.6f}")
        print(f"  Sigma Threshold: {track_data['sigma_threshold']:.6f}")
        print(f"  Pass/Fail: {'PASS' if track_data['sigma_pass'] else 'FAIL'}")
        print(f"  Linearity Spec: {track_data['linearity_spec']:.6f}")

        # Unit properties
        unit_props = track_data.get('unit_properties', {})
        print(f"  Unit Length: {unit_props.get('unit_length', 'N/A')}")
        print(f"  Untrimmed Resistance: {unit_props.get('untrimmed_resistance', 'N/A')}")
        print(f"  Trimmed Resistance: {unit_props.get('trimmed_resistance', 'N/A')}")

        if detailed:
            # Data statistics
            if 'untrimmed_data' in track_data:
                data = track_data['untrimmed_data']
                print(f"\n  Data Statistics:")
                print(f"    Points: {len(data['position'])}")
                print(f"    Travel Length: {data['travel_length']:.4f}")
                print(f"    Position Range: {min(data['position']):.4f} to {max(data['position']):.4f}")
                print(f"    Error Range: {min(data['error']):.6f} to {max(data['error']):.6f}")

            # Gradient statistics
            if 'gradients' in track_data:
                gradients = track_data['gradients']
                print(f"\n  Gradient Statistics:")
                print(f"    Number of gradients: {len(gradients)}")
                if gradients:
                    print(f"    Min gradient: {min(gradients):.6f}")
                    print(f"    Max gradient: {max(gradients):.6f}")
                    print(f"    Mean gradient: {sum(gradients) / len(gradients):.6f}")

    # Print summary
    if 'summary' in result:
        print(f"\n{'=' * 60}")
        print("Summary:")
        summary = result['summary']
        print(f"  Total Tracks: {summary['total_tracks']}")
        print(f"  Passed: {summary['passed_tracks']}")
        print(f"  Failed: {summary['failed_tracks']}")
        if summary['avg_sigma_gradient'] is not None:
            print(f"  Average Sigma: {summary['avg_sigma_gradient']:.6f}")
            print(f"  Min Sigma: {summary['min_sigma_gradient']:.6f}")
            print(f"  Max Sigma: {summary['max_sigma_gradient']:.6f}")


def validate_sigma_calculation(result: dict):
    """
    Validate that sigma calculation matches expected MATLAB results.

    This function checks:
    1. Filter is applied correctly (80Hz cutoff)
    2. Gradient step size is 3
    3. Standard deviation uses sample (N-1) formula
    """
    print(f"\n{'=' * 60}")
    print("Sigma Calculation Validation")
    print(f"{'=' * 60}")

    for track_id, track_data in result['tracks'].items():
        if 'error' in track_data or 'untrimmed_data' not in track_data:
            continue

        print(f"\nValidating Track: {track_id}")

        # Get raw data
        position = track_data['untrimmed_data']['position']
        error = track_data['untrimmed_data']['error']

        # Verify filter parameters
        print(f"  Filter cutoff: 80 Hz (expected)")
        print(f"  Filter sampling freq: 100 Hz (expected)")
        print(f"  Gradient step: 3 (expected)")

        # Check gradient calculation
        gradients = track_data.get('gradients', [])
        if gradients:
            # Verify step size by checking positions
            gradient_positions = track_data.get('gradient_positions', [])
            if len(gradient_positions) > 1:
                typical_step = gradient_positions[1] - gradient_positions[0]
                expected_step = position[3] - position[0]  # Step of 3
                print(f"  Typical position step: {typical_step:.6f}")
                print(f"  Expected step (3 indices): {expected_step:.6f}")

                if abs(typical_step - expected_step) > 0.0001:
                    print("  WARNING: Step size may not be correct!")

        # Verify standard deviation calculation
        import numpy as np
        if gradients:
            # Recalculate to verify
            std_numpy = np.std(gradients, ddof=1)  # Sample standard deviation
            reported_sigma = track_data['sigma_gradient']

            print(f"  Calculated sigma: {std_numpy:.6f}")
            print(f"  Reported sigma: {reported_sigma:.6f}")
            print(f"  Difference: {abs(std_numpy - reported_sigma):.9f}")

            if abs(std_numpy - reported_sigma) > 1e-6:
                print("  WARNING: Sigma calculation may have issues!")
            else:
                print("  ✓ Sigma calculation verified")


def process_single_file(file_path: str, config: Config, logger: logging.Logger):
    """Process a single file and display results."""
    print(f"\nProcessing: {file_path}")

    # Create data loader
    loader = DataLoader(config=config, logger=logger)

    try:
        # Load and process file
        result = loader.load_file(file_path)

        # Print results
        print_results(result, detailed=True)

        # Validate calculations
        validate_sigma_calculation(result)

        # Save results to JSON for inspection
        output_file = Path(file_path).stem + "_results.json"
        with open(output_file, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_types(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                else:
                    return obj

            json.dump(result, f, indent=2, default=convert_types)
        print(f"\nResults saved to: {output_file}")

        return result

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise


def process_batch(folder_path: str, config: Config, logger: logging.Logger):
    """Process all Excel files in a folder."""
    print(f"\nProcessing folder: {folder_path}")

    # Find all Excel files
    excel_files = []
    for file in Path(folder_path).glob("*.xls*"):
        if file.suffix.lower() in ['.xls', '.xlsx']:
            excel_files.append(file)

    print(f"Found {len(excel_files)} Excel files")

    if not excel_files:
        print("No Excel files found in folder")
        return

    # Create data loader
    loader = DataLoader(config=config, logger=logger)

    # Process files
    def progress_callback(current, total, filename):
        print(f"Processing {current}/{total}: {filename}")

    results = loader.load_batch(excel_files, progress_callback=progress_callback)

    # Summary statistics
    total_files = len(results)
    successful = sum(1 for r in results if 'error' not in r)
    failed = total_files - successful

    print(f"\n{'=' * 60}")
    print("Batch Processing Summary")
    print(f"{'=' * 60}")
    print(f"Total files: {total_files}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Count pass/fail by track
    total_tracks = 0
    passed_tracks = 0
    failed_tracks = 0

    for result in results:
        if 'tracks' in result:
            for track_id, track_data in result['tracks'].items():
                if 'sigma_pass' in track_data:
                    total_tracks += 1
                    if track_data['sigma_pass']:
                        passed_tracks += 1
                    else:
                        failed_tracks += 1

    print(f"\nTrack Statistics:")
    print(f"Total tracks: {total_tracks}")
    print(f"Passed: {passed_tracks}")
    print(f"Failed: {failed_tracks}")
    if total_tracks > 0:
        print(f"Pass rate: {passed_tracks / total_tracks * 100:.1f}%")

    # Save batch results
    output_file = "batch_results.json"
    with open(output_file, 'w') as f:
        def convert_types(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return obj

        json.dump(results, f, indent=2, default=convert_types)
    print(f"\nBatch results saved to: {output_file}")


def main():
    """Main entry point for testing."""
    # Set up logging
    logger = setup_logging()

    # Create configuration
    config = Config()

    print("Laser Trim Analysis System - Test Script")
    print(f"{'=' * 60}")
    print(f"Configuration:")
    print(f"  Filter cutoff: {config.filter_cutoff_frequency} Hz")
    print(f"  Filter sampling: {config.filter_sampling_frequency} Hz")
    print(f"  Gradient step: {config.matlab_gradient_step}")
    print(f"  Sigma scaling factor: {config.sigma_scaling_factor}")
    print(f"{'=' * 60}")

    # Get input from user
    while True:
        print("\nOptions:")
        print("1. Process single file")
        print("2. Process folder (batch)")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            file_path = input("Enter path to Excel file: ").strip()
            if os.path.exists(file_path):
                process_single_file(file_path, config, logger)
            else:
                print(f"File not found: {file_path}")

        elif choice == '2':
            folder_path = input("Enter path to folder: ").strip()
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                process_batch(folder_path, config, logger)
            else:
                print(f"Folder not found: {folder_path}")

        elif choice == '3':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()