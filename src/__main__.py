"""
Laser Trim AI System - Main Entry Point

This is the main entry point for the laser trim analysis system.
It can be run as a module: python -m src
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional, List
import json

from .config import Config
from .core.data_loader import DataLoader
from .core.sigma_calculator import SigmaCalculator


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger('laser_trim_ai')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='Laser Trim AI System - Analyze potentiometer trim data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python -m src analyze path/to/file.xlsx

  # Process all files in a folder
  python -m src analyze path/to/folder --batch

  # Process with custom output
  python -m src analyze input.xlsx --output results.json

  # Validate sigma calculation
  python -m src validate file.xlsx

  # Show configuration
  python -m src config
        """
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze laser trim data')
    analyze_parser.add_argument('input', help='Input file or folder path')
    analyze_parser.add_argument('--batch', action='store_true',
                                help='Process all Excel files in folder')
    analyze_parser.add_argument('--output', '-o', help='Output file path')
    analyze_parser.add_argument('--format', choices=['json', 'csv', 'excel'],
                                default='json', help='Output format')
    analyze_parser.add_argument('--detailed', '-d', action='store_true',
                                help='Include detailed data in output')

    # Validate command
    validate_parser = subparsers.add_parser('validate',
                                            help='Validate sigma calculations')
    validate_parser.add_argument('input', help='Input file path')
    validate_parser.add_argument('--reference', help='Reference sigma value')

    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'),
                               help='Set configuration value')
    config_parser.add_argument('--reset', action='store_true',
                               help='Reset to default configuration')

    # Global options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log to file')
    parser.add_argument('--config-file', help='Custom configuration file')

    return parser


def analyze_command(args, config: Config, logger: logging.Logger):
    """Handle analyze command."""
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1

    # Create data loader
    loader = DataLoader(config=config, logger=logger)

    # Process based on mode
    if args.batch or input_path.is_dir():
        # Batch processing
        if input_path.is_file():
            logger.error("--batch specified but input is a file, not a directory")
            return 1

        # Find all Excel files
        excel_files = list(input_path.glob("**/*.xls*"))
        excel_files = [f for f in excel_files if f.suffix.lower() in ['.xls', '.xlsx']]

        if not excel_files:
            logger.warning(f"No Excel files found in {input_path}")
            return 0

        logger.info(f"Found {len(excel_files)} Excel files to process")

        # Process files
        def progress_callback(current, total, filename):
            logger.info(f"Processing [{current}/{total}] {filename}")

        results = loader.load_batch(excel_files, progress_callback=progress_callback)

    else:
        # Single file processing
        if not input_path.is_file():
            logger.error("Input is not a file")
            return 1

        logger.info(f"Processing single file: {input_path}")
        result = loader.load_file(input_path)
        results = [result]

    # Prepare output
    if args.detailed:
        output_data = results
    else:
        # Summary only
        output_data = []
        for result in results:
            summary = {
                'file_name': result['file_name'],
                'system': result['system'],
                'model': result['metadata']['model'],
                'serial': result['metadata']['serial'],
                'tracks': {}
            }

            for track_id, track_data in result.get('tracks', {}).items():
                if 'error' not in track_data:
                    summary['tracks'][track_id] = {
                        'sigma_gradient': track_data.get('sigma_gradient'),
                        'sigma_threshold': track_data.get('sigma_threshold'),
                        'sigma_pass': track_data.get('sigma_pass'),
                        'unit_length': track_data.get('unit_properties', {}).get('unit_length')
                    }

            output_data.append(summary)

    # Save output
    output_path = args.output
    if not output_path:
        if args.batch:
            output_path = "batch_results.json"
        else:
            output_path = input_path.stem + "_results.json"

    save_results(output_data, output_path, args.format, logger)

    # Print summary
    print_analysis_summary(results, logger)

    return 0


def validate_command(args, config: Config, logger: logging.Logger):
    """Handle validate command."""
    input_path = Path(args.input)

    if not input_path.exists() or not input_path.is_file():
        logger.error(f"Input file does not exist: {input_path}")
        return 1

    logger.info(f"Validating sigma calculation for: {input_path}")

    # Create data loader
    loader = DataLoader(config=config, logger=logger)

    # Load file
    result = loader.load_file(input_path)

    # Validate each track
    print("\n" + "=" * 60)
    print("SIGMA CALCULATION VALIDATION")
    print("=" * 60)

    all_valid = True

    for track_id, track_data in result.get('tracks', {}).items():
        if 'error' in track_data:
            continue

        print(f"\nTrack: {track_id}")
        print("-" * 40)

        # Check configuration
        print(f"Configuration:")
        print(f"  Filter cutoff: {config.filter_cutoff_frequency} Hz (expected: 80)")
        print(f"  Sampling freq: {config.filter_sampling_frequency} Hz (expected: 100)")
        print(f"  Gradient step: {config.matlab_gradient_step} (expected: 3)")

        # Validate sigma
        sigma = track_data.get('sigma_gradient')
        if sigma is not None:
            print(f"\nCalculated sigma: {sigma:.6f}")

            if args.reference:
                try:
                    ref_sigma = float(args.reference)
                    diff = abs(sigma - ref_sigma)
                    print(f"Reference sigma: {ref_sigma:.6f}")
                    print(f"Difference: {diff:.6f} ({diff / ref_sigma * 100:.2f}%)")

                    if diff > 0.0001:
                        print("WARNING: Significant difference from reference!")
                        all_valid = False
                    else:
                        print("✓ Sigma matches reference")
                except ValueError:
                    logger.error(f"Invalid reference value: {args.reference}")

        # Additional validation checks
        if 'gradients' in track_data:
            gradients = track_data['gradients']
            print(f"\nGradient statistics:")
            print(f"  Count: {len(gradients)}")
            if gradients:
                import numpy as np
                print(f"  Mean: {np.mean(gradients):.6f}")
                print(f"  Std (N-1): {np.std(gradients, ddof=1):.6f}")
                print(f"  Min: {min(gradients):.6f}")
                print(f"  Max: {max(gradients):.6f}")

    print("\n" + "=" * 60)
    if all_valid:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
    print("=" * 60)

    return 0 if all_valid else 1


def config_command(args, config: Config, logger: logging.Logger):
    """Handle config command."""
    if args.reset:
        # Reset to defaults
        config = Config()
        logger.info("Configuration reset to defaults")

    if args.set:
        # Set configuration value
        key, value = args.set
        if hasattr(config, key):
            # Try to convert value to appropriate type
            current_value = getattr(config, key)
            try:
                if isinstance(current_value, bool):
                    setattr(config, key, value.lower() in ['true', '1', 'yes'])
                elif isinstance(current_value, int):
                    setattr(config, key, int(value))
                elif isinstance(current_value, float):
                    setattr(config, key, float(value))
                else:
                    setattr(config, key, value)
                logger.info(f"Set {key} = {value}")
            except ValueError:
                logger.error(f"Invalid value for {key}: {value}")
                return 1
        else:
            logger.error(f"Unknown configuration key: {key}")
            return 1

    # Display configuration
    print("\n" + "=" * 60)
    print("CURRENT CONFIGURATION")
    print("=" * 60)

    for key, value in config.__dict__.items():
        if not key.startswith('_'):
            print(f"{key}: {value}")

    print("=" * 60)

    return 0


def save_results(data: List[dict], output_path: str, format: str, logger: logging.Logger):
    """Save results in specified format."""
    output_path = Path(output_path)

    try:
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        elif format == 'csv':
            # Flatten data for CSV
            import pandas as pd
            flattened = []

            for result in data:
                base_info = {
                    'file_name': result.get('file_name'),
                    'system': result.get('system'),
                    'model': result.get('model', {}).get('model') if isinstance(result.get('model'),
                                                                                dict) else result.get('model'),
                    'serial': result.get('serial', {}).get('serial') if isinstance(result.get('serial'),
                                                                                   dict) else result.get('serial')
                }

                for track_id, track_data in result.get('tracks', {}).items():
                    if isinstance(track_data, dict) and 'error' not in track_data:
                        row = base_info.copy()
                        row['track_id'] = track_id
                        row['sigma_gradient'] = track_data.get('sigma_gradient')
                        row['sigma_threshold'] = track_data.get('sigma_threshold')
                        row['sigma_pass'] = track_data.get('sigma_pass')
                        flattened.append(row)

            df = pd.DataFrame(flattened)
            df.to_csv(output_path, index=False)

        elif format == 'excel':
            # Create Excel file with multiple sheets
            import pandas as pd

            with pd.ExcelWriter(output_path) as writer:
                # Summary sheet
                summary_data = []
                for result in data:
                    for track_id, track_data in result.get('tracks', {}).items():
                        if isinstance(track_data, dict) and 'error' not in track_data:
                            summary_data.append({
                                'File': result.get('file_name'),
                                'System': result.get('system'),
                                'Model': result.get('model', {}).get('model') if isinstance(result.get('model'),
                                                                                            dict) else result.get(
                                    'model'),
                                'Serial': result.get('serial', {}).get('serial') if isinstance(result.get('serial'),
                                                                                               dict) else result.get(
                                    'serial'),
                                'Track': track_id,
                                'Sigma Gradient': track_data.get('sigma_gradient'),
                                'Sigma Threshold': track_data.get('sigma_threshold'),
                                'Pass/Fail': 'PASS' if track_data.get('sigma_pass') else 'FAIL'
                            })

                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)

                # Raw data sheet (if detailed)
                if any('untrimmed_data' in t for r in data for t in r.get('tracks', {}).values() if
                       isinstance(t, dict)):
                    df_raw = pd.DataFrame(data)
                    df_raw.to_excel(writer, sheet_name='Raw Data', index=False)

        logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


def print_analysis_summary(results: List[dict], logger: logging.Logger):
    """Print analysis summary to console."""
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    total_files = len(results)
    total_tracks = 0
    passed_tracks = 0
    failed_tracks = 0

    for result in results:
        for track_id, track_data in result.get('tracks', {}).items():
            if isinstance(track_data, dict) and 'sigma_pass' in track_data:
                total_tracks += 1
                if track_data['sigma_pass']:
                    passed_tracks += 1
                else:
                    failed_tracks += 1

    print(f"Total files processed: {total_files}")
    print(f"Total tracks analyzed: {total_tracks}")
    print(f"Passed: {passed_tracks}")
    print(f"Failed: {failed_tracks}")

    if total_tracks > 0:
        print(f"Pass rate: {passed_tracks / total_tracks * 100:.1f}%")

    print("=" * 60)


def main():
    """Main entry point for the laser trim AI system."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_level, args.log_file)

    # Load configuration
    if args.config_file:
        # Load custom config (not implemented in this version)
        logger.info(f"Loading config from: {args.config_file}")
        config = Config()  # For now, just use default
    else:
        config = Config()

    # Handle commands
    if not args.command:
        parser.print_help()
        return 0

    try:
        if args.command == 'analyze':
            return analyze_command(args, config, logger)

        elif args.command == 'validate':
            return validate_command(args, config, logger)

        elif args.command == 'config':
            return config_command(args, config, logger)

        else:
            logger.error(f"Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug("Stack trace:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())