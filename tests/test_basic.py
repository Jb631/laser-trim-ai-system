"""Basic tests for project structure."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that main modules can be imported."""
    try:
        import src
        import src.core
        import src.data
        import src.ml_models
        import src.reporting
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")

def test_project_structure():
    """Test that project structure is correct."""
    required_dirs = [
        "src/core",
        "src/ml_models",
        "src/data",
        "src/reporting",
        "src/gui",
        "tests",
        "docs",
        "config"
    ]

    for directory in required_dirs:
        assert Path(directory).exists(), f"Missing directory: {directory}"

def test_configuration():
    """Test that configuration file exists."""
    config_file = Path("config/default_config.yaml")
    assert config_file.exists(), "Missing default configuration file"
