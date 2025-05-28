"""
Example Usage Script for AI-Powered Laser Trim Analysis System

This script demonstrates how to use the core data processing engine
for analyzing laser trim data files.

Author: QA Team
Date: 2024
Version: 1.0.0
"""

import os
import sys
from pathlib import Path
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from data_processor import DataProcessor, SystemType
from config import ConfigManager, create_configured_processor


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('laser_trim_analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def process_single_file_example(file_path: str):
    """
    Example: Process a single laser trim data file.

    Args:
        file_path: Path to the Excel file
    """
    logger = setup_logging()
    logger.info("=== Single File Processing Example ===")

    # Create processor
    processor = DataProcessor()

    try:
        # Process the file
        results = processor.process_file(file_path)

        # Display results
        print(f"\nFile: {results['file_info']['filename']}")
        print(f"System Type: {results['file_info']['system_type']}")
        print(f"Number of Tracks: {len(results['tracks'])}")

        # Display results for each track
        for track_id, track_data in results['tracks'].items():
            print(f"\n--- Track: {track_id} ---")

            # Sigma results
            sigma_results = track_data['sigma_results']
            print(f"Sigma Gradient: {sigma_results.sigma_gradient:.6f}")
            print(f"Sigma Threshold: {sigma_results.sigma_threshold:.6f}")
            print(f"Sigma Pass: {'PASS' if sigma_results.sigma_pass else 'FAIL'}")

            # Unit properties
            unit_props = track_data['unit_properties']
            print(f"Unit Length: {unit_props.unit_length}")
            print(f"Travel Length: {unit_props.travel_length}")
            print(f"Linearity Spec: {unit_props.linearity_spec}")

            # Data statistics
            untrimmed_data = track_data['untrimmed_data']
            print(f"Data Points: {len(untrimmed_data.position)}")
            print(f"Position Range: {untrimmed_data.position.min():.3f} to {untrimmed_data.position.max():.3f}")
            print(f"Error Range: {untrimmed_data.error.min():.6f} to {untrimmed_data.error.max():.6f}")

        return results

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise


def batch_processing_example(folder_path: str):
    """
    Example: Process all Excel files in a folder.

    Args:
        folder_path: Path to folder containing Excel files
    """
    logger = setup_logging()
    logger.info("=== Batch Processing Example ===")

    # Create processor
    processor = DataProcessor()

    # Process all files
    results = processor.batch_process(folder_path)

    # Summary statistics
    total_files = len(results)
    successful = sum(1 for r in results.values() if 'error' not in r)
    failed = total_files - successful

    print(f"\nBatch Processing Summary:")
    print(f"Total Files: {total_files}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Collect all sigma results
    sigma_data = []
    for filename, file_results in results.items():
        if 'error' not in file_results:
            for track_id, track_data in file_results.get('tracks', {}).items():
                sigma_data.append({
                    'filename': filename,
                    'track_id': track_id,
                    'sigma_gradient': track_data['sigma_results'].sigma_gradient,
                    'sigma_threshold': track_data['sigma_results'].sigma_threshold,
                    'sigma_pass': track_data['sigma_results'].sigma_pass
                })

    # Create summary DataFrame
    df_summary = pd.DataFrame(sigma_data)

    # Export results
    output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    processor.export_results(results, output_file)
    print(f"\nResults exported to: {output_file}")

    # Display statistics
    if not df_summary.empty:
        print(f"\nSigma Analysis Summary:")
        print(f"Total Tracks Analyzed: {len(df_summary)}")
        print(f"Pass Rate: {df_summary['sigma_pass'].mean() * 100:.1f}%")
        print(f"Average Sigma Gradient: {df_summary['sigma_gradient'].mean():.6f}")
        print(
            f"Sigma Gradient Range: {df_summary['sigma_gradient'].min():.6f} to {df_summary['sigma_gradient'].max():.6f}")

    return results, df_summary


def configuration_example():
    """Example: Using configuration management."""
    logger = setup_logging()
    logger.info("=== Configuration Example ===")

    # Create configuration manager
    config_mgr = ConfigManager()

    # Create default configuration
    config_path = "config/analysis_config.json"
    config_mgr.create_default_config_file(config_path)
    print(f"Created default configuration: {config_path}")

    # Modify configuration for specific requirements
    config_mgr.update_setting('processing', 'default_scaling_factor', 25.0)
    config_mgr.update_setting('processing', 'min_data_points', 30)
    config_mgr.update_setting('output', 'decimal_places', 4)

    # Save modified configuration
    config_mgr.save_config()

    # Create processor with configuration
    processor = create_configured_processor(config_path)
    print("\nProcessor created with custom configuration")

    return processor


def visualization_example(results: dict):
    """
    Example: Visualize analysis results.

    Args:
        results: Processing results from a single file
    """
    logger = setup_logging()
    logger.info("=== Visualization Example ===")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Laser Trim Analysis - {results['file_info']['filename']}", fontsize=16)

    for idx, (track_id, track_data) in enumerate(results['tracks'].items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col] if len(results['tracks']) > 1 else axes.flatten()[0]

        # Extract data
        untrimmed_data = track_data['untrimmed_data']
        sigma_results = track_data['sigma_results']

        # Plot 1: Error vs Position
        ax.plot(untrimmed_data.position, untrimmed_data.error, 'b-', label='Raw Error', alpha=0.7)
        ax.plot(untrimmed_data.position, sigma_results.filtered_error, 'r-', label='Filtered Error', linewidth=2)

        # Add tolerance limits if available
        if untrimmed_data.upper_limit is not None:
            ax.plot(untrimmed_data.position, untrimmed_data.upper_limit, 'g--', label='Upper Limit', alpha=0.5)
            ax.plot(untrimmed_data.position, untrimmed_data.lower_limit, 'g--', label='Lower Limit', alpha=0.5)

        ax.set_xlabel('Position')
        ax.set_ylabel('Error')
        ax.set_title(
            f'Track {track_id} - Sigma: {sigma_results.sigma_gradient:.6f} ({"PASS" if sigma_results.sigma_pass else "FAIL"})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots if only one track
    if len(results['tracks']) == 1:
        for ax in axes.flatten()[1:]:
            ax.set_visible(False)

    plt.tight_layout()

    # Save plot
    plot_file = f"analysis_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")

    return fig


def advanced_analysis_example(results: dict):
    """
    Example: Advanced analysis and reporting.

    Args:
        results: Processing results from single or batch processing
    """
    logger = setup_logging()
    logger.info("=== Advanced Analysis Example ===")

    # Create detailed report
    report_lines = []
    report_lines.append("LASER TRIM ANALYSIS REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Analyze each file
    for filename, file_results in results.items():
        if 'error' in file_results:
            report_lines.append(f"\nFile: {filename} - ERROR: {file_results['error']}")
            continue

        report_lines.append(f"\nFile: {filename}")
        report_lines.append(f"System Type: {file_results['file_info']['system_type']}")

        # Analyze each track
        for track_id, track_data in file_results['tracks'].items():
            report_lines.append(f"\n  Track: {track_id}")

            # Sigma analysis
            sigma_results = track_data['sigma_results']
            report_lines.append(f"    Sigma Gradient: {sigma_results.sigma_gradient:.6f}")
            report_lines.append(f"    Sigma Threshold: {sigma_results.sigma_threshold:.6f}")
            report_lines.append(f"    Status: {'PASS' if sigma_results.sigma_pass else 'FAIL'}")

            # Calculate margin
            margin = (
                                 sigma_results.sigma_threshold - sigma_results.sigma_gradient) / sigma_results.sigma_threshold * 100
            report_lines.append(f"    Margin: {margin:.1f}%")

            # Gradient statistics
            if len(sigma_results.gradients) > 0:
                report_lines.append(f"    Gradient Stats:")
                report_lines.append(f"      Mean: {sigma_results.gradients.mean():.6f}")
                report_lines.append(f"      Std: {sigma_results.gradients.std():.6f}")
                report_lines.append(f"      Min: {sigma_results.gradients.min():.6f}")
                report_lines.append(f"      Max: {sigma_results.gradients.max():.6f}")

    # Save report
    report_file = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nDetailed report saved to: {report_file}")

    # Print summary to console
    print("\n" + "\n".join(report_lines[:20]) + "\n...")  # First 20 lines


def main():
    """Main function demonstrating various usage examples."""
    print("AI-Powered Laser Trim Analysis System - Examples")
    print("=" * 50)

    # Example 1: Configuration
    print("\n1. Setting up configuration...")
    processor = configuration_example()

    # Example 2: Single file processing
    # Replace with your actual file path
    test_file = "data/sample_laser_trim_data.xlsx"
    if os.path.exists(test_file):
        print("\n2. Processing single file...")
        results = process_single_file_example(test_file)

        # Example 3: Visualization
        print("\n3. Creating visualizations...")
        visualization_example(results)
    else:
        print(f"\n2. Skipping single file example - file not found: {test_file}")

    # Example 4: Batch processing
    # Replace with your actual folder path
    test_folder = "data/laser_trim_files"
    if os.path.exists(test_folder):
        print("\n4. Batch processing folder...")
        batch_results, summary_df = batch_processing_example(test_folder)

        # Example 5: Advanced analysis
        print("\n5. Generating advanced analysis report...")
        advanced_analysis_example(batch_results)
    else:
        print(f"\n4. Skipping batch processing - folder not found: {test_folder}")

    print("\n" + "=" * 50)
    print("Examples completed. Check generated files for results.")


if __name__ == "__main__":
    main()