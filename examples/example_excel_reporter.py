"""
Excel Reporter Example Usage

This script demonstrates how to use the Excel Report Generator
with the Laser Trim AI System.

Author: Laser Trim AI System
Date: 2024
Version: 1.0.0
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from excel_reporter import ExcelReporter
from excel_config import ExcelReportConfig, get_production_config
from data_processor import LaserTrimDataProcessor
from ml_models import LaserTrimMLModels
from config import Config


def process_and_report(input_folder: str, output_folder: str, config_path: str = None):
    """
    Complete workflow: process data, run ML analysis, and generate Excel report.

    Args:
        input_folder: Folder containing laser trim data files
        output_folder: Folder for output files
        config_path: Optional path to configuration file
    """
    print(f"\n{'=' * 60}")
    print("LASER TRIM AI SYSTEM - Excel Report Generation")
    print(f"{'=' * 60}\n")

    # Load configuration
    config = Config()
    if config_path and os.path.exists(config_path):
        config = Config(config_path)
        print(f"✅ Loaded configuration from: {config_path}")
    else:
        print("📋 Using default configuration")

    # Initialize components
    print("\n🔧 Initializing components...")
    processor = LaserTrimDataProcessor(config)
    ml_models = LaserTrimMLModels(config)

    # Create Excel report configuration
    excel_config = get_production_config()
    if config.openai_api_key:
        excel_config.set('api.openai_api_key', config.openai_api_key)

    reporter = ExcelReporter(excel_config.to_dict())
    print("✅ Components initialized")

    # Process data files
    print(f"\n📂 Processing files in: {input_folder}")
    file_results = []

    for filename in os.listdir(input_folder):
        if filename.endswith(('.xlsx', '.xls')):
            print(f"\n  Processing: {filename}")
            filepath = os.path.join(input_folder, filename)

            try:
                # Process file
                result = processor.process_file(filepath)

                if result['success']:
                    # Run ML predictions
                    ml_result = ml_models.predict_quality(result)

                    # Combine results
                    combined_result = {
                        'filename': filename,
                        'model': result.get('model', 'Unknown'),
                        'serial': result.get('serial', 'Unknown'),
                        'system': result.get('system', 'Unknown'),
                        'status': ml_result['predicted_status'],
                        'sigma_gradient': result.get('sigma_gradient', 0),
                        'sigma_threshold': result.get('sigma_threshold', 0.004),
                        'sigma_pass': result.get('sigma_pass', False),
                        'linearity_pass': result.get('linearity_pass', False),
                        'failure_probability': ml_result['failure_probability'],
                        'risk_category': ml_result['risk_category'],
                        'confidence': ml_result['confidence'],
                        'resistance_change_percent': result.get('resistance_change_percent', 0),
                        'trim_improvement_percent': result.get('trim_improvement_percent', 0),
                        'unit_length': result.get('unit_length', 0),
                        'optimal_offset': result.get('optimal_offset', 0)
                    }

                    # Handle multi-track files
                    if 'tracks' in result:
                        combined_result['tracks'] = {}
                        for track_id, track_data in result['tracks'].items():
                            track_ml = ml_models.predict_quality(track_data)
                            combined_result['tracks'][track_id] = {
                                **track_data,
                                'predicted_status': track_ml['predicted_status'],
                                'failure_probability': track_ml['failure_probability'],
                                'risk_category': track_ml['risk_category']
                            }

                    file_results.append(combined_result)
                    print(f"    ✅ Processed successfully")
                    print(f"    📊 Status: {combined_result['status']}")
                    print(f"    🎯 Risk: {combined_result['risk_category']}")
                else:
                    print(f"    ❌ Processing failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"    ❌ Error: {str(e)}")

    # Prepare results for report
    print(f"\n📊 Processed {len(file_results)} files successfully")

    # Get ML predictions for batch
    if file_results:
        print("\n🤖 Running batch ML analysis...")
        batch_predictions = ml_models.analyze_batch(file_results)

        report_data = {
            'file_results': file_results,
            'ml_predictions': batch_predictions,
            'processing_timestamp': datetime.now().isoformat(),
            'config_version': config.version
        }
    else:
        print("\n⚠️  No files were successfully processed")
        report_data = {
            'file_results': [],
            'ml_predictions': {},
            'processing_timestamp': datetime.now().isoformat()
        }

    # Generate Excel report
    print("\n📝 Generating Excel report...")
    report_filename = f"laser_trim_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    report_path = os.path.join(output_folder, report_filename)

    try:
        generated_path = reporter.generate_report(
            report_data,
            report_path,
            include_ai_insights=bool(config.openai_api_key)
        )

        print(f"\n✅ Excel report generated successfully!")
        print(f"📄 Report saved to: {generated_path}")

        # Print summary
        print_report_summary(report_data)

        return generated_path

    except Exception as e:
        print(f"\n❌ Error generating report: {str(e)}")
        return None


def generate_demo_report():
    """Generate a demonstration report with sample data."""
    print("\n🎯 Generating demonstration Excel report...")

    # Create sample data
    demo_data = {
        "file_results": [
            {
                "filename": "8340-1_SN12345_20240115.xlsx",
                "model": "8340-1",
                "serial": "SN12345",
                "system": "A",
                "tracks": {
                    "TRK1": {
                        "status": "Pass",
                        "sigma_gradient": 0.0023,
                        "sigma_threshold": 0.004,
                        "sigma_pass": True,
                        "linearity_pass": True,
                        "failure_probability": 0.05,
                        "risk_category": "Low",
                        "resistance_change_percent": 2.3,
                        "trim_improvement_percent": 18.5,
                        "unit_length": 150.0,
                        "optimal_offset": 0.0008
                    },
                    "TRK2": {
                        "status": "Warning",
                        "sigma_gradient": 0.0038,
                        "sigma_threshold": 0.004,
                        "sigma_pass": True,
                        "linearity_pass": False,
                        "failure_probability": 0.45,
                        "risk_category": "Medium",
                        "resistance_change_percent": 3.8,
                        "trim_improvement_percent": 12.2,
                        "unit_length": 149.5,
                        "optimal_offset": 0.0012
                    }
                }
            },
            {
                "filename": "8555_SN98765_20240115.xlsx",
                "model": "8555",
                "serial": "SN98765",
                "system": "B",
                "status": "Pass",
                "sigma_gradient": 0.0015,
                "sigma_threshold": 0.003,
                "sigma_pass": True,
                "linearity_pass": True,
                "failure_probability": 0.02,
                "risk_category": "Low",
                "resistance_change_percent": 1.5,
                "trim_improvement_percent": 25.3,
                "unit_length": 180.0,
                "optimal_offset": 0.0003
            },
            {
                "filename": "6845_SN54321_20240115.xlsx",
                "model": "6845",
                "serial": "SN54321",
                "system": "A",
                "status": "Fail",
                "sigma_gradient": 0.0052,
                "sigma_threshold": 0.0045,
                "sigma_pass": False,
                "linearity_pass": True,
                "failure_probability": 0.85,
                "risk_category": "High",
                "resistance_change_percent": 6.2,
                "trim_improvement_percent": 5.8,
                "unit_length": 120.0,
                "optimal_offset": 0.0018
            }
        ],
        "ml_predictions": {
            "next_batch_pass_rate": 0.82,
            "maintenance_due_in_days": 8,
            "quality_trend": "improving",
            "recommended_actions": [
                "Adjust laser power for model 6845",
                "Schedule calibration for System A",
                "Review trim parameters for high resistance change units"
            ]
        }
    }

    # Create reporter with demo configuration
    demo_config = ExcelReportConfig({
        "report": {
            "include_ai_insights": False,  # No API key for demo
            "include_charts": True,
            "include_raw_data": True
        }
    })

    reporter = ExcelReporter(demo_config.to_dict())

    # Generate report
    output_path = "demo_laser_trim_report.xlsx"
    generated_path = reporter.generate_report(
        demo_data,
        output_path,
        include_ai_insights=False
    )

    print(f"\n✅ Demo report generated: {generated_path}")
    print("\n📊 Demo Report Contents:")
    print("  - Executive Summary with overall metrics")
    print("  - Detailed Analysis of all units")
    print("  - Statistical Summary with distributions")
    print("  - Trend Analysis and predictions")
    print("  - Quality KPIs and metrics")
    print("  - Recommendations for improvement")
    print("  - Raw data export")

    return generated_path


def print_report_summary(report_data: dict):
    """Print a summary of the report data."""
    file_results = report_data.get('file_results', [])

    if not file_results:
        return

    # Calculate summary statistics
    total_units = 0
    passed_units = 0
    high_risk = 0

    for result in file_results:
        if 'tracks' in result:
            for track in result['tracks'].values():
                total_units += 1
                if track.get('status', '').startswith('Pass'):
                    passed_units += 1
                if track.get('risk_category') == 'High':
                    high_risk += 1
        else:
            total_units += 1
            if result.get('status', '').startswith('Pass'):
                passed_units += 1
            if result.get('risk_category') == 'High':
                high_risk += 1

    pass_rate = (passed_units / total_units * 100) if total_units > 0 else 0

    print(f"\n📊 REPORT SUMMARY")
    print(f"{'=' * 40}")
    print(f"Total Units Analyzed: {total_units}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print(f"High Risk Units: {high_risk}")

    # ML predictions summary
    ml_predictions = report_data.get('ml_predictions', {})
    if ml_predictions:
        print(f"\n🤖 ML PREDICTIONS")
        print(f"{'=' * 40}")
        print(f"Next Batch Pass Rate: {ml_predictions.get('next_batch_pass_rate', 0) * 100:.1f}%")
        print(f"Maintenance Due: {ml_predictions.get('maintenance_due_in_days', 'N/A')} days")
        print(f"Quality Trend: {ml_predictions.get('quality_trend', 'Unknown')}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Excel reports for laser trim analysis"
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Generate a demonstration report with sample data'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input folder containing laser trim data files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./reports',
        help='Output folder for generated reports (default: ./reports)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    if args.demo:
        # Generate demo report
        generate_demo_report()
    elif args.input:
        # Process actual data
        if not os.path.exists(args.input):
            print(f"❌ Input folder not found: {args.input}")
            return

        process_and_report(args.input, args.output, args.config)
    else:
        # Show help if no action specified
        parser.print_help()
        print("\n💡 Examples:")
        print("  Generate demo report:")
        print("    python example_excel_reporter.py --demo")
        print("\n  Process actual data:")
        print("    python example_excel_reporter.py --input ./data --output ./reports")


if __name__ == "__main__":
    main()