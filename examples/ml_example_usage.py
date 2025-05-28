"""
ML Models Usage Examples
========================

This script demonstrates how to use the machine learning models
for laser trim analysis in real-world scenarios.

Author: QA Specialist
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import your modules
from config import Config
from data_processor import LaserTrimDataProcessor
from ml_models import LaserTrimMLModels, create_ml_models, train_all_models


def example_4_feature_analysis():
    """Example 4: Feature importance and model insights."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Feature Importance Analysis")
    print("=" * 60)

    # Setup
    config = Config()
    ml_models = create_ml_models(config)

    # Train models if needed
    historical_data = create_synthetic_historical_data(500)
    train_all_models(ml_models, historical_data)

    # Get feature importance report
    print("\nGenerating feature importance report...")
    report = ml_models.get_feature_importance_report()

    # Display threshold optimizer features
    print("\nThreshold Optimizer - Top Features:")
    for feature, importance in report['threshold_optimizer']['top_features'][:5]:
        print(f"  {feature:30s} {importance:.4f}")

    # Display failure predictor features
    print("\nFailure Predictor - Top Features:")
    if 'top_features' in report['failure_predictor']:
        for feature, importance in report['failure_predictor']['top_features'][:5]:
            print(f"  {feature:30s} {importance:.4f}")

    # Display overall most important features
    print("\nMost Important Features Overall:")
    for feature, importance in report['summary']['most_important_overall'][:5]:
        print(f"  {feature:30s} {importance:.4f}")

    # Feature categories
    print("\nFeature Categories:")
    categories = report['summary']['feature_categories']
    for category, features in categories.items():
        if features:
            print(f"  {category}: {len(features)} features")
            print(f"    Examples: {', '.join(features[:3])}")

    # Create feature importance visualization
    create_feature_importance_plot(report)


def example_5_continuous_monitoring():
    """Example 5: Continuous monitoring with alerts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Continuous Production Monitoring")
    print("=" * 60)

    # Setup
    config = Config()
    ml_models = create_ml_models(config)
    ml_models.load_models('latest')

    # Monitoring parameters
    alert_threshold = 0.7  # 70% failure probability triggers alert
    drift_window = 20  # Check drift every 20 units

    print(f"\nMonitoring production with:")
    print(f"  - Alert threshold: {alert_threshold:.0%}")
    print(f"  - Drift check window: {drift_window} units")
    print("\nPress Ctrl+C to stop monitoring\n")

    # Monitoring loop
    unit_buffer = []
    alert_count = 0

    try:
        for i in range(100):  # Simulate 100 units
            # Simulate new unit data
            unit_data = {
                'unit_id': f'PROD_{datetime.now().strftime("%Y%m%d")}_{i:04d}',
                'sigma_gradient': np.random.normal(0.5 + i * 0.001, 0.03),
                'linearity_spec': np.random.normal(0.02, 0.002),
                'travel_length': np.random.normal(150, 8),
                'unit_length': np.random.normal(140, 7),
                'resistance_change': np.random.normal(5, 1),
                'resistance_change_percent': np.random.normal(2, 0.3),
                'error_data': np.random.normal(0, 0.01, 100).tolist(),
                'model': np.random.choice(['8340', '8555']),
                'timestamp': datetime.now()
            }

            # Real-time analysis
            failure_pred = ml_models.predict_failure_probability(unit_data)
            optimal_threshold = ml_models.predict_optimal_threshold(unit_data)

            # Check for alerts
            if failure_pred['failure_probability'] > alert_threshold:
                alert_count += 1
                print(f"⚠️  ALERT: Unit {unit_data['unit_id']} - "
                      f"High failure risk ({failure_pred['failure_probability']:.1%})")
                print(f"   Risk Level: {failure_pred['risk_level']}")
                print(f"   Recommended Action: Immediate inspection")

            # Add to buffer for drift detection
            unit_buffer.append(unit_data)

            # Check for drift every N units
            if len(unit_buffer) >= drift_window:
                print(f"\n🔍 Drift check after {len(unit_buffer)} units...")

                # Check last unit for drift
                drift_result = ml_models.detect_manufacturing_drift(unit_buffer[-1])

                if drift_result['is_drift']:
                    print(f"   ⚠️  DRIFT DETECTED!")
                    print(f"   Severity: {drift_result['severity']}")
                    print(f"   {drift_result['recommendation']}")
                else:
                    print(f"   ✓ No drift detected")

                # Clear buffer
                unit_buffer = []

            # Simulate production delay
            import time
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")

    print(f"\nMonitoring Summary:")
    print(f"  Units processed: {i + 1}")
    print(f"  Alerts triggered: {alert_count}")
    print(f"  Alert rate: {alert_count / (i + 1):.1%}")


# Helper Functions

def create_synthetic_historical_data(n_samples: int) -> pd.DataFrame:
    """Create synthetic historical data for examples."""
    np.random.seed(42)

    data = []
    for i in range(n_samples):
        # Simulate different models with different characteristics
        model = np.random.choice(['8340', '8555', '6845'], p=[0.5, 0.3, 0.2])

        # Model-specific parameters
        if model == '8340':
            base_sigma = 0.5
            base_resistance = 5.0
        elif model == '8555':
            base_sigma = 0.45
            base_resistance = 4.5
        else:
            base_sigma = 0.55
            base_resistance = 5.5

        # Add time-based drift
        days_ago = n_samples - i
        drift_factor = 1 + (i / n_samples) * 0.05  # 5% drift over time

        unit = {
            'sigma_gradient': np.random.normal(base_sigma * drift_factor, 0.05),
            'linearity_spec': np.random.normal(0.02, 0.003),
            'travel_length': np.random.normal(150, 10),
            'unit_length': np.random.normal(140, 8),
            'resistance_change': np.random.normal(base_resistance * drift_factor, 0.5),
            'resistance_change_percent': np.random.normal(2 * drift_factor, 0.3),
            'sigma_threshold': 0.6,
            'error_data': np.random.normal(0, 0.01, 100).tolist(),
            'model': model,
            'timestamp': datetime.now() - timedelta(days=days_ago),
            'zone_analysis': {
                'worst_zone': np.random.randint(1, 6),
                'variance': np.random.random()
            }
        }

        # Determine pass/fail
        unit['passed'] = unit['sigma_gradient'] < unit['sigma_threshold']

        data.append(unit)

    return pd.DataFrame(data)


def create_batch_visualization(batch_df: pd.DataFrame):
    """Create visualization for batch analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Batch Production Analysis', fontsize=16)

    # 1. Sigma gradient trend
    ax1 = axes[0, 0]
    ax1.plot(batch_df.index, batch_df['sigma_gradient'], 'b-', alpha=0.6)
    ax1.axhline(y=0.6, color='r', linestyle='--', label='Threshold')
    ax1.set_xlabel('Unit Number')
    ax1.set_ylabel('Sigma Gradient')
    ax1.set_title('Sigma Gradient Trend')
    ax1.legend()

    # 2. Failure probability distribution
    ax2 = axes[0, 1]
    ax2.hist(batch_df['failure_probability'], bins=20, alpha=0.7, color='orange')
    ax2.axvline(x=0.5, color='r', linestyle='--', label='High Risk Threshold')
    ax2.set_xlabel('Failure Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Failure Risk Distribution')
    ax2.legend()

    # 3. Drift detection over time
    ax3 = axes[1, 0]
    drift_units = batch_df[batch_df['drift_detected']]
    ax3.scatter(batch_df.index, batch_df['sigma_gradient'],
                c=batch_df['drift_detected'], cmap='RdYlGn_r', alpha=0.6)
    ax3.set_xlabel('Unit Number')
    ax3.set_ylabel('Sigma Gradient')
    ax3.set_title('Drift Detection (Red = Drift)')

    # 4. Risk heatmap
    ax4 = axes[1, 1]
    risk_matrix = batch_df[['sigma_gradient', 'resistance_change', 'failure_probability']].corr()
    sns.heatmap(risk_matrix, annot=True, cmap='coolwarm', center=0, ax=ax4)
    ax4.set_title('Risk Factor Correlations')

    plt.tight_layout()
    plt.savefig('batch_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'batch_analysis.png'")


def create_feature_importance_plot(report: dict):
    """Create feature importance visualization."""
    # Extract top features
    features = []
    importances = []

    for feature, importance in report['summary']['most_important_overall'][:10]:
        features.append(feature)
        importances.append(importance)

    # Create horizontal bar plot
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(features))

    plt.barh(y_pos, importances, alpha=0.8)
    plt.yticks(y_pos, features)
    plt.xlabel('Importance Score')
    plt.title('Top 10 Most Important Features for ML Models')
    plt.tight_layout()

    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nFeature importance plot saved as 'feature_importance.png'")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("LASER TRIM ML MODELS - USAGE EXAMPLES")
    print("=" * 80)

    # Run examples
    print("\nWhich example would you like to run?")
    print("1. Basic model training")
    print("2. Real-time production analysis")
    print("3. Batch analysis with trends")
    print("4. Feature importance analysis")
    print("5. Continuous monitoring")
    print("6. Run all examples")

    choice = input("\nEnter choice (1-6): ").strip()

    if choice == '1':
        example_1_basic_training()
    elif choice == '2':
        example_2_real_time_analysis()
    elif choice == '3':
        example_3_batch_analysis()
    elif choice == '4':
        example_4_feature_analysis()
    elif choice == '5':
        example_5_continuous_monitoring()
    elif choice == '6':
        # Run all except continuous monitoring
        ml_models = example_1_basic_training()
        example_2_real_time_analysis()
        example_3_batch_analysis()
        example_4_feature_analysis()
        print("\n✓ All examples completed!")
        print("\nNote: Run example 5 separately for continuous monitoring.")
    else:
        print("Invalid choice. Please run again and select 1-6.")

    print("\n" + "=" * 80)
    print("Examples completed. Check generated files:")
    print("- historical_data.csv (synthetic data)")
    print("- ml_models/ (saved models)")
    print("- batch_analysis.png (batch visualization)")
    print("- feature_importance.png (feature analysis)")
    print("=" * 80)


if __name__ == "__main__":
    main()
1
_basic_training():
"""Example 1: Basic model training with historical data."""
print("\n" + "=" * 60)
print("EXAMPLE 1: Basic Model Training")
print("=" * 60)

# Setup
config = Config()
ml_models = create_ml_models(config)

# Load historical data (in practice, this would come from your database)
print("\nLoading historical data...")
historical_data = pd.read_csv('historical_data.csv')  # Your data file

# For demo, create synthetic data if file doesn't exist
if not Path('historical_data.csv').exists():
    print("Creating synthetic historical data for demo...")
    historical_data = create_synthetic_historical_data(1000)
    historical_data.to_csv('historical_data.csv', index=False)

# Train all models
print("\nTraining ML models...")
results = train_all_models(ml_models, historical_data)

# Display results
print("\nTraining Results:")
print(f"Threshold Optimizer - MAE: {results['threshold_optimizer'].get('mae', 'N/A'):.4f}")
print(f"Failure Predictor - Accuracy: {results['failure_predictor'].get('accuracy', 'N/A'):.2%}")
print(f"Drift Detector - Anomalies Found: {results['drift_detector'].get('n_anomalies', 'N/A')}")
print(f"Models saved to: {results['saved_version']}")

return ml_models


def example_2_real_time_analysis():
    """Example 2: Real-time analysis of production unit."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Real-Time Production Analysis")
    print("=" * 60)

    # Setup
    config = Config()
    processor = LaserTrimDataProcessor(config)
    ml_models = create_ml_models(config)

    # Load pre-trained models
    print("\nLoading pre-trained models...")
    ml_models.load_models('latest')  # Load latest version

    # Process new unit data
    print("\nProcessing new production unit...")
    unit_file = 'unit_8340_A12345.xlsx'  # Your data file

    # For demo, create synthetic unit data
    unit_data = {
        'sigma_gradient': 0.52,
        'linearity_spec': 0.018,
        'travel_length': 152.3,
        'unit_length': 141.7,
        'resistance_change': 4.8,
        'resistance_change_percent': 1.9,
        'error_data': np.random.normal(0, 0.008, 100).tolist(),
        'model': '8340',
        'timestamp': datetime.now()
    }

    # Get ML predictions
    print("\n1. Threshold Optimization:")
    threshold_pred = ml_models.predict_optimal_threshold(unit_data)
    print(f"   Current Threshold: 0.60")
    print(f"   Optimal Threshold: {threshold_pred['optimal_threshold']:.3f}")
    print(f"   Confidence: {threshold_pred['confidence']:.1%}")
    print("   Top Contributing Features:")
    for feat, contrib in list(threshold_pred['feature_contributions'].items())[:3]:
        print(f"   - {feat}: {contrib['importance']:.3f}")

    print("\n2. Failure Risk Assessment:")
    failure_pred = ml_models.predict_failure_probability(unit_data)
    print(f"   Failure Probability: {failure_pred['failure_probability']:.1%}")
    print(f"   Risk Level: {failure_pred['risk_level']}")
    print("   Risk Factors:")
    for factor in failure_pred['risk_factors'][:3]:
        print(f"   - {factor['feature']}: {factor['value']:.3f} "
              f"(importance: {factor['importance']:.3f})")

    print("\n3. Manufacturing Drift Check:")
    drift_check = ml_models.detect_manufacturing_drift(unit_data)
    print(f"   Drift Detected: {'Yes' if drift_check['is_drift'] else 'No'}")
    print(f"   Anomaly Score: {drift_check['anomaly_score']:.3f}")
    print(f"   Severity: {drift_check['severity']}")
    print(f"   Recommendation: {drift_check['recommendation']}")


def example_3_batch_analysis():
    """Example 3: Batch analysis with trend detection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch Analysis with Trends")
    print("=" * 60)

    # Setup
    config = Config()
    ml_models = create_ml_models(config)

    # Load models
    ml_models.load_models('latest')

    # Simulate batch of units from today's production
    print("\nAnalyzing today's production batch...")
    batch_data = []

    for i in range(50):
        unit = {
            'unit_id': f'UNIT_{i:04d}',
            'sigma_gradient': np.random.normal(0.48 + i * 0.002, 0.02),  # Gradual drift
            'linearity_spec': np.random.normal(0.02, 0.001),
            'travel_length': np.random.normal(150, 5),
            'unit_length': np.random.normal(140, 5),
            'resistance_change': np.random.normal(5 + i * 0.02, 0.5),
            'resistance_change_percent': np.random.normal(2, 0.2),
            'error_data': np.random.normal(0, 0.01, 100).tolist(),
            'model': '8340',
            'timestamp': datetime.now()
        }

        # Get predictions
        failure_prob = ml_models.predict_failure_probability(unit)['failure_probability']
        drift_detected = ml_models.detect_manufacturing_drift(unit)['is_drift']

        unit['failure_probability'] = failure_prob
        unit['drift_detected'] = drift_detected
        batch_data.append(unit)

    # Convert to DataFrame for analysis
    batch_df = pd.DataFrame(batch_data)

    # Analysis summary
    print(f"\nBatch Summary:")
    print(f"Total Units: {len(batch_df)}")
    print(f"Average Failure Risk: {batch_df['failure_probability'].mean():.1%}")
    print(f"High Risk Units (>50%): {(batch_df['failure_probability'] > 0.5).sum()}")
    print(f"Drift Detected: {batch_df['drift_detected'].sum()} units")

    # Trend analysis
    print(f"\nTrend Analysis:")
    print(
        f"Sigma Gradient Trend: {'Increasing' if batch_df['sigma_gradient'].corr(batch_df.index) > 0.5 else 'Stable'}")
    print(
        f"Resistance Change Trend: {'Increasing' if batch_df['resistance_change'].corr(batch_df.index) > 0.5 else 'Stable'}")

    # Create visualization
    create_batch_visualization(batch_df)


def example_