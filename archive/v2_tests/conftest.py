"""
Pytest configuration and shared fixtures for Laser Trim Analyzer tests.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


@pytest.fixture
def sample_data():
    """Generate sample sigma gradient data for testing."""
    np.random.seed(42)  # Reproducible results
    return np.random.normal(loc=0.5, scale=0.1, size=100)


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'sigma_gradient': np.random.normal(loc=0.5, scale=0.1, size=100),
        'linearity_error': np.random.normal(loc=0.02, scale=0.005, size=100),
        'model_number': ['8340'] * 100,
        'trim_date': pd.date_range('2024-01-01', periods=100, freq='D')
    })


@pytest.fixture
def edge_case_data():
    """Generate edge case data for boundary testing."""
    return {
        'empty': np.array([]),
        'single': np.array([0.5]),
        'two_points': np.array([0.3, 0.7]),
        'all_same': np.array([0.5] * 10),
        'with_nan': np.array([0.5, np.nan, 0.6, 0.4]),
        'with_inf': np.array([0.5, np.inf, 0.6, 0.4]),
        'with_neg_inf': np.array([0.5, -np.inf, 0.6, 0.4]),
        'all_nan': np.array([np.nan] * 10),
        'zeros': np.array([0.0] * 10),
        'negative': np.array([-0.5, -0.3, -0.1]),
    }


@pytest.fixture
def spec_limits():
    """Standard specification limits for laser trim analysis."""
    return {
        'LSL': 0.3,  # Lower Spec Limit
        'USL': 0.7,  # Upper Spec Limit
        'target': 0.5,  # Target value
    }
