"""
ML Model Trainer for Laser Trim Analyzer

This script trains the ML models used by MLPredictor for real-time predictions.
Run this periodically to update models with new data.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sqlite3
from typing import Dict, Any, Tuple


class MLModelTrainer:
    """Train ML models for laser trim analysis predictions."""

    def __init__(self, db_path: str, output_dir: str):
        """
        Initialize the model trainer.

        Args:
            db_path: Path to the SQLite database
            output_dir: Directory to save trained models
        """
        self.db_path = db_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Feature names for all models
        self.feature_names = [
            'sigma_gradient', 'sigma_threshold', 'unit_length',
            'travel_length', 'linearity_spec', 'resistance_change_percent',
            'trim_improvement_percent', 'final_linearity_error_shifted',
            'failure_probability', 'range_utilization_percent'
        ]

    def load_training_data(self, days_back: int = 365) -> pd.DataFrame:
        """Load historical data for training."""
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT 
            t.*,
            a.model,
            a.serial
        FROM track_results t
        JOIN analysis_results a ON t.analysis_id = a.id
        WHERE a.timestamp >= datetime('now', ?)
        """

        df = pd.read_sql_query(query, conn, params=[f'-{days_back} days'])
        conn.close()

        print(f"Loaded {len(df)} records for training")
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """Prepare features and calculate statistics."""
        # Select feature columns
        feature_cols = []
        for name in self.feature_names:
            if name in df.columns:
                feature_cols.append(name)

        X = df[feature_cols].copy()

        # Calculate feature statistics for handling missing values
        feature_stats = {}
        for col in X.columns:
            feature_stats[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'min': X[col].min(),
                'max': X[col].max(),
                'median': X[col].median()
            }

        # Fill missing values with mean
        X = X.fillna(X.mean())

        return X, feature_stats

    def train_failure_predictor(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train failure prediction model."""
        print("\n=== Training Failure Predictor ===")

        # Prepare features
        X, feature_stats = self.prepare_features(df)

        # Create target: high risk units are likely to fail
        # You can adjust this logic based on your actual failure data
        y = ((df['failure_probability'] > 0.5) |
             (df['sigma_pass'] == 0) |
             (df['risk_category'] == 'High')).astype(int)

        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]

        print(f"Training samples: {len(X)}")
        print(f"Failure rate: {y.mean():.2%}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 5 Important Features:")
        print(feature_importance.head())

        # Save model
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_stats': feature_stats,
            'feature_importance': feature_importance.to_dict('records'),
            'version': '1.0',
            'trained_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_accuracy': (y_pred == y_test).mean()
        }

        output_path = os.path.join(self.output_dir, 'failure_predictor.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\nFailure predictor saved to: {output_path}")
        return model_data

    def train_anomaly_detector(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train anomaly detection model."""
        print("\n=== Training Anomaly Detector ===")

        # Prepare features
        X, feature_stats = self.prepare_features(df)

        # For anomaly detection, we train on "normal" units only
        normal_mask = (
                (df['sigma_pass'] == 1) &
                (df['risk_category'] != 'High') &
                (df['failure_probability'] < 0.3)
        )
        X_normal = X[normal_mask]

        print(f"Training on {len(X_normal)} normal samples")

        # Scale features
        scaler = StandardScaler()
        X_normal_scaled = scaler.fit_transform(X_normal)

        # Train Isolation Forest
        model = IsolationForest(
            contamination=0.05,  # Expect 5% anomalies
            random_state=42,
            n_estimators=100
        )
        model.fit(X_normal_scaled)

        # Test on all data
        X_all_scaled = scaler.transform(X)
        anomaly_predictions = model.predict(X_all_scaled)
        anomaly_scores = model.score_samples(X_all_scaled)

        # Analyze results
        anomaly_rate = (anomaly_predictions == -1).mean()
        print(f"\nAnomaly rate on all data: {anomaly_rate:.2%}")

        # Check which units are detected as anomalies
        df_with_anomalies = df.copy()
        df_with_anomalies['is_anomaly'] = (anomaly_predictions == -1)
        df_with_anomalies['anomaly_score'] = anomaly_scores

        print("\nAnomaly detection by risk category:")
        print(df_with_anomalies.groupby('risk_category')['is_anomaly'].mean())

        # Save model
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_stats': feature_stats,
            'version': '1.0',
            'trained_date': datetime.now().isoformat(),
            'training_samples': len(X_normal),
            'expected_anomaly_rate': 0.05,
            'actual_anomaly_rate': anomaly_rate
        }

        output_path = os.path.join(self.output_dir, 'anomaly_detector.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\nAnomaly detector saved to: {output_path}")
        return model_data

    def calculate_optimal_thresholds(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate optimal thresholds for each model."""
        print("\n=== Calculating Optimal Thresholds ===")

        thresholds = {}

        # Group by model
        for model in df['model'].unique():
            model_data = df[df['model'] == model]

            if len(model_data) < 10:
                continue

            # Calculate statistics
            sigma_values = model_data['sigma_gradient'].dropna()

            # Method 1: Statistical approach (mean + 3*std)
            mean_sigma = sigma_values.mean()
            std_sigma = sigma_values.std()
            statistical_threshold = mean_sigma + 3 * std_sigma

            # Method 2: Percentile approach (95th percentile)
            percentile_threshold = sigma_values.quantile(0.95)

            # Method 3: Based on actual pass/fail data
            passing_units = model_data[model_data['sigma_pass'] == 1]['sigma_gradient']
            if len(passing_units) > 0:
                data_driven_threshold = passing_units.max() * 1.1  # 10% margin
            else:
                data_driven_threshold = statistical_threshold

            # Use the most conservative (highest) threshold
            recommended_threshold = max(statistical_threshold, percentile_threshold, data_driven_threshold)

            # Calculate confidence based on sample size
            confidence = min(0.95, len(model_data) / 100)

            thresholds[model] = {
                'recommended': float(recommended_threshold),
                'statistical': float(statistical_threshold),
                'percentile_95': float(percentile_threshold),
                'data_driven': float(data_driven_threshold),
                'confidence': float(confidence),
                'sample_count': len(model_data),
                'current_pass_rate': float(model_data['sigma_pass'].mean())
            }

            print(f"\nModel {model}:")
            print(f"  Samples: {len(model_data)}")
            print(f"  Recommended threshold: {recommended_threshold:.6f}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Current pass rate: {model_data['sigma_pass'].mean():.2%}")

        # Save thresholds
        output_path = os.path.join(self.output_dir, 'recommended_thresholds.json')
        with open(output_path, 'w') as f:
            json.dump(thresholds, f, indent=2)

        print(f"\nThresholds saved to: {output_path}")
        return thresholds

    def save_feature_statistics(self, feature_stats: Dict[str, Dict[str, float]]) -> None:
        """Save feature statistics for handling missing values."""
        # Convert numpy types to Python types for JSON serialization
        stats_serializable = {}
        for feature, stats in feature_stats.items():
            stats_serializable[feature] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in stats.items()
            }

        output_path = os.path.join(self.output_dir, 'feature_stats.json')
        with open(output_path, 'w') as f:
            json.dump(stats_serializable, f, indent=2)

        print(f"\nFeature statistics saved to: {output_path}")

    def train_all_models(self, days_back: int = 365) -> None:
        """Train all models and save them."""
        print("=" * 60)
        print("ML MODEL TRAINING FOR LASER TRIM ANALYZER")
        print(f"Training with last {days_back} days of data")
        print("=" * 60)

        # Load data
        df = self.load_training_data(days_back)

        if len(df) < 100:
            print("\nWARNING: Not enough data for training (need at least 100 samples)")
            return

        # Train models
        failure_model = self.train_failure_predictor(df)
        anomaly_model = self.train_anomaly_detector(df)

        # Calculate thresholds
        thresholds = self.calculate_optimal_thresholds(df)

        # Save feature statistics
        _, feature_stats = self.prepare_features(df)
        self.save_feature_statistics(feature_stats)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print(f"Models saved to: {self.output_dir}")
        print("=" * 60)

    def evaluate_model_performance(self, days_to_test: int = 30) -> Dict[str, Any]:
        """Evaluate model performance on recent data."""
        print("\n=== Evaluating Model Performance ===")

        # Load recent data for testing
        df_test = self.load_training_data(days_to_test)

        if len(df_test) < 10:
            print("Not enough recent data for evaluation")
            return {}

        # Load models
        results = {}

        # Test failure predictor
        try:
            with open(os.path.join(self.output_dir, 'failure_predictor.pkl'), 'rb') as f:
                failure_data = pickle.load(f)

            X_test, _ = self.prepare_features(df_test)
            X_test = X_test.fillna(X_test.mean())

            # Create target
            y_test = ((df_test['failure_probability'] > 0.5) |
                      (df_test['sigma_pass'] == 0) |
                      (df_test['risk_category'] == 'High')).astype(int)

            # Predict
            X_test_scaled = failure_data['scaler'].transform(X_test)
            y_pred = failure_data['model'].predict(X_test_scaled)

            accuracy = (y_pred == y_test).mean()
            results['failure_predictor_accuracy'] = accuracy
            print(f"Failure predictor accuracy on recent data: {accuracy:.2%}")

        except Exception as e:
            print(f"Error evaluating failure predictor: {e}")

        return results


def main():
    """Example usage of the model trainer."""
    import argparse

    parser = argparse.ArgumentParser(description='Train ML models for Laser Trim Analyzer')
    parser.add_argument('--db-path', type=str, required=True,
                        help='Path to the SQLite database')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save trained models')
    parser.add_argument('--days-back', type=int, default=365,
                        help='Number of days of historical data to use')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate existing models on recent data')

    args = parser.parse_args()

    # Create trainer
    trainer = MLModelTrainer(args.db_path, args.output_dir)

    if args.evaluate:
        # Evaluate existing models
        trainer.evaluate_model_performance()
    else:
        # Train new models
        trainer.train_all_models(args.days_back)


if __name__ == "__main__":
    # For testing without command line args
    db_path = os.path.join(os.path.expanduser("~"), "LaserTrimResults", "analysis_history.db")
    output_dir = os.path.join(os.path.expanduser("~"), "LaserTrimResults", "ml_models")

    if os.path.exists(db_path):
        trainer = MLModelTrainer(db_path, output_dir)
        trainer.train_all_models(days_back=365)
    else:
        print(f"Database not found at: {db_path}")
        print("Please run the laser trim analyzer first to create the database.")