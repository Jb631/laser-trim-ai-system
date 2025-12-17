"""
Model Info Analyzers for ML Tools Page

These methods analyze model information from the ML manager
without requiring direct access to model objects.
"""

from typing import Dict, Any, List
import numpy as np
from datetime import datetime


def analyze_model_info(model_name: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze individual model from its info dictionary."""
    analysis = {
        'name': model_name,
        'type': model_name,
        'is_trained': info.get('trained', False),
        'training_samples': 0,  # Will be populated when trained
        'last_trained': info.get('last_training'),
        'performance_metrics': info.get('performance', {}),
        'feature_importance': {},
        'prediction_stats': {},
        'version': '1.0.0',
        'status': info.get('status', 'Unknown'),
        'description': info.get('description', ''),
        'feature_count': info.get('feature_count', 0),
        'features': info.get('features', [])
    }
    
    # Use actual feature importance from model if available
    if analysis['is_trained']:
        # Extract feature importance from model info
        feature_importance = info.get('feature_importance', {})
        if feature_importance:
            analysis['feature_importance'] = feature_importance
        else:
            # Leave empty if no real data available
            analysis['feature_importance'] = {}
        
        # Extract prediction stats from model info
        prediction_stats = info.get('prediction_stats', {})
        if prediction_stats:
            analysis['prediction_stats'] = prediction_stats
        else:
            # Use defaults based on model type
            analysis['prediction_stats'] = {
                'model_category': 'classification' if 'failure_predictor' in model_name else 'regression',
                'complexity_score': info.get('complexity_score', 0.0),
                'efficiency_score': info.get('efficiency_score', 0.0)
            }
    
    return analysis


def compare_model_performance_info(models_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compare performance across models using info dictionaries."""
    comparison = {
        'best_model': None,
        'worst_model': None,
        'average_accuracy': 0,
        'accuracy_spread': 0,
        'models_by_performance': [],
        'overall_ranking': [],  # Add this for the chart
        'summary': {}
    }
    
    # Collect performance metrics
    model_scores = {}
    all_model_data = {}
    
    for model_name, info in models_info.items():
        is_trained = info.get('trained', False)
        performance = info.get('performance', {})
        
        # Get accuracy/score
        accuracy = performance.get('accuracy', performance.get('r2_score', 0)) if is_trained else 0
        
        # Calculate composite score (accuracy * efficiency factor)
        efficiency = 0.85 if is_trained else 0.5  # Trained models get higher efficiency
        composite = accuracy * efficiency
        
        model_data = {
            'is_trained': is_trained,
            'accuracy': accuracy,
            'efficiency': efficiency,
            'composite': composite,
            'performance_metrics': performance
        }
        
        all_model_data[model_name] = model_data
        
        if is_trained:
            model_scores[model_name] = accuracy
    
    # Create overall ranking (all models, trained and untrained)
    overall_ranking = sorted(
        all_model_data.items(),
        key=lambda x: x[1]['composite'],
        reverse=True
    )
    comparison['overall_ranking'] = overall_ranking
    
    if model_scores:
        # Sort by score for trained models only
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['models_by_performance'] = [
            {'name': name, 'score': score} for name, score in sorted_models
        ]
        
        comparison['best_model'] = sorted_models[0][0] if sorted_models else None
        comparison['worst_model'] = sorted_models[-1][0] if sorted_models else None
        
        scores = list(model_scores.values())
        comparison['average_accuracy'] = np.mean(scores)
        comparison['accuracy_spread'] = np.max(scores) - np.min(scores) if len(scores) > 1 else 0
    
    # Summary
    comparison['summary'] = {
        'total_models': len(models_info),
        'trained_models': len(model_scores),
        'untrained_models': len(models_info) - len(model_scores),
        'best_model': comparison['best_model'],
        'average_accuracy': comparison['average_accuracy']
    }
    
    return comparison


def compare_feature_importance_info(models_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compare feature importance across models."""
    comparison = {
        'common_features': [],
        'feature_rankings': {},
        'consensus_features': []
    }
    
    # Since we're using info dicts, we'll use the features list
    all_features = set()
    for info in models_info.values():
        features = info.get('features', [])
        all_features.update(features)
    
    comparison['common_features'] = list(all_features)
    
    # Extract actual feature importance from models if available
    if all_features:
        # Aggregate feature importance scores from all models
        feature_scores = {}
        model_count = {}
        
        for model_name, info in models_info.items():
            if info.get('trained', False) and 'feature_importance' in info:
                for feature, importance in info['feature_importance'].items():
                    if feature not in feature_scores:
                        feature_scores[feature] = 0.0
                        model_count[feature] = 0
                    feature_scores[feature] += importance
                    model_count[feature] += 1
        
        # Average the scores
        for feature in feature_scores:
            if model_count[feature] > 0:
                feature_scores[feature] /= model_count[feature]
        
        if feature_scores:
            comparison['feature_rankings'] = dict(
                sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            )
            
            # Top 50% are consensus features
            if feature_scores.values():
                threshold = np.median(list(feature_scores.values()))
                comparison['consensus_features'] = [
                    f for f, score in feature_scores.items() if score >= threshold
                ]
        else:
            # No real feature importance data available
            comparison['feature_rankings'] = {}
            comparison['consensus_features'] = []
    
    return comparison


def analyze_resource_usage_info(models_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze resource usage across models."""
    usage = {
        'training_times': {},
        'memory_usage': {},
        'prediction_speeds': {},
        'efficiency_ranking': []
    }
    
    # Extract actual resource usage data from models
    for model_name, info in models_info.items():
        if info.get('trained', False):
            # Get actual resource metrics from model info
            performance = info.get('performance', {})
            
            # Extract training time
            training_time = performance.get('training_time')
            if training_time is None:
                # Try to extract from other fields
                training_time = info.get('training_duration', 0.0)
            if training_time:
                usage['training_times'][model_name] = training_time
                
            # Extract memory usage
            memory_usage = performance.get('memory_usage_mb')
            if memory_usage is None:
                memory_usage = info.get('memory_footprint', 0.0)
            if memory_usage:
                usage['memory_usage'][model_name] = memory_usage
                
            # Extract prediction speed
            prediction_speed = performance.get('avg_prediction_time_ms')
            if prediction_speed is None:
                prediction_speed = info.get('inference_time', 0.0)
            if prediction_speed:
                usage['prediction_speeds'][model_name] = prediction_speed
    
    # Calculate efficiency ranking
    efficiency_scores = {}
    for model_name in usage['training_times']:
        # Simple efficiency metric
        train_time = usage['training_times'][model_name]
        memory = usage['memory_usage'][model_name]
        speed = usage['prediction_speeds'][model_name]
        
        # Lower is better for all metrics
        efficiency = 1.0 / (train_time * 0.3 + memory * 0.002 + speed * 0.5)
        efficiency_scores[model_name] = efficiency
    
    usage['efficiency_ranking'] = sorted(
        efficiency_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return usage


def analyze_prediction_quality_info(models_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze prediction quality across models."""
    quality = {
        'accuracy_distribution': {},
        'confidence_analysis': {},
        'error_patterns': {},
        'reliability_scores': {}
    }
    
    accuracies = []
    
    for model_name, info in models_info.items():
        if info.get('trained', False):
            performance = info.get('performance', {})
            accuracy = performance.get('accuracy', performance.get('r2_score', 0))
            accuracies.append(accuracy)
            
            # Extract actual reliability metrics from model
            reliability = info.get('reliability_metrics', {})
            quality['reliability_scores'][model_name] = {
                'accuracy': accuracy,
                'consistency': reliability.get('consistency', accuracy * 0.95),  # Default to 95% of accuracy
                'robustness': reliability.get('robustness', accuracy * 0.90)  # Default to 90% of accuracy
            }
    
    if accuracies:
        quality['accuracy_distribution'] = {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies),
            'range': np.max(accuracies) - np.min(accuracies)
        }
    
    return quality