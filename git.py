#!/usr/bin/env python3
"""
GitHub Repository Setup Script for Laser Trim AI System
Run this to create the initial project structure
"""

import os
import sys
import subprocess
from pathlib import Path


def create_directory_structure():
    """Create the project directory structure."""
    print("🚀 Setting up Laser Trim AI System repository...")

    # Define directory structure
    directories = [
        "src/core",
        "src/ml_models",
        "src/data",
        "src/reporting",
        "src/gui",
        "tests",
        "docs",
        "examples/sample_data",
        "scripts",
        "config"
    ]

    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")

    # Create __init__.py files
    init_dirs = [
        "src",
        "src/core",
        "src/ml_models",
        "src/data",
        "src/reporting",
        "src/gui",
        "tests"
    ]

    for directory in init_dirs:
        init_file = Path(directory) / "__init__.py"
        init_file.touch()
        print(f"✓ Created {init_file}")


def create_readme():
    """Create README.md file."""
    readme_content = """# Laser Trim AI System

AI-powered quality analysis system for potentiometer laser trim testing.

## Features
- Automated sigma gradient calculation with validated algorithms
- Machine learning for threshold optimization  
- Predictive failure analysis
- Automated Excel report generation
- Real-time quality monitoring

## Current Status
- [ ] Core data pipeline
- [ ] Sigma calculations
- [ ] ML threshold optimization
- [ ] Failure prediction
- [ ] Excel reporting
- [ ] GUI application

## Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/laser-trim-ai-system.git
cd laser-trim-ai-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\\Scripts\\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start
```python
from src.core import LaserTrimAnalyzer

analyzer = LaserTrimAnalyzer()
results = analyzer.process_folder("path/to/data")
analyzer.generate_report(results, "output.xlsx")
```

## Development
See docs/development.md for development guidelines.
"""

    with open("README.md", "w") as f:
        f.write(readme_content)
    print("✓ Created README.md")


def create_requirements():
    """Create requirements.txt file."""
    requirements_content = """# Core
pandas>=1.3.0
numpy>=1.20.0
openpyxl>=3.0.0
xlsxwriter>=3.0.0

# Machine Learning
scikit-learn>=1.0.0
joblib>=1.0.0
xgboost>=1.5.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.6.0

# GUI (optional - uncomment one)
# PyQt6>=6.2.0
# PySide6>=6.2.0
# customtkinter>=5.0.0

# Database
sqlalchemy>=1.4.0

# Testing
pytest>=6.0.0
pytest-cov>=3.0.0

# Development
black>=22.0.0
flake8>=4.0.0
mypy>=0.910
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    print("✓ Created requirements.txt")


def create_gitignore():
    """Create .gitignore file."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.project
.pydevproject

# Data files
*.xlsx
*.xls
*.csv
!examples/sample_data/*

# Model files
*.pkl
*.h5
*.joblib
models/*.pkl
models/*.h5

# Output directories
reports/
output/
logs/
results/

# Logs
*.log
logs/

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Config files with secrets
config/production.yaml
config/local.yaml
config/secrets.yaml

# Test data
tests/test_data/
tests/fixtures/large_data/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Documentation builds
docs/_build/
docs/_static/
docs/_templates/
"""

    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("✓ Created .gitignore")


def create_setup_py():
    """Create setup.py file."""
    setup_content = '''from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="laser-trim-ai-system",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@company.com",
    description="AI-powered quality analysis for laser trim testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/laser-trim-ai-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Quality Control",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "openpyxl>=3.0.0",
        "scikit-learn>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "laser-trim-ai=src.cli:main",
        ],
    },
)
'''

    with open("setup.py", "w") as f:
        f.write(setup_content)
    print("✓ Created setup.py")


def create_sample_config():
    """Create sample configuration file."""
    config_content = """# Default configuration for Laser Trim AI System

system:
  name: "Laser Trim AI System"
  version: "0.1.0"

data:
  supported_formats: [".xlsx", ".xls", ".csv"]
  batch_size: 100

analysis:
  sigma_threshold_default: 0.5
  sigma_scaling_factor: 24.0
  filter_sampling_frequency: 100
  filter_cutoff_frequency: 80

ml_models:
  failure_prediction:
    enabled: true
    retrain_interval_days: 30
    min_training_samples: 100

  threshold_optimization:
    enabled: true
    safety_factor: 1.1

reporting:
  excel:
    include_charts: true
    include_statistics: true

  formats: ["xlsx", "html", "pdf"]

paths:
  data: "data/"
  models: "models/"
  reports: "reports/"
  logs: "logs/"
"""

    config_path = Path("config/default_config.yaml")
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        f.write(config_content)
    print("✓ Created config/default_config.yaml")


def create_sample_test():
    """Create a sample test file."""
    test_content = '''"""Basic tests for project structure."""

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
'''

    with open("tests/test_basic.py", "w") as f:
        f.write(test_content)
    print("✓ Created tests/test_basic.py")


def initialize_git():
    """Initialize git repository."""
    try:
        # Check if git is available
        subprocess.run(["git", "--version"], check=True, capture_output=True)

        # Initialize repository
        subprocess.run(["git", "init"], check=True)
        print("✓ Initialized git repository")

        # Add all files
        subprocess.run(["git", "add", "."], check=True)
        print("✓ Added files to git")

        # Create initial commit
        subprocess.run(["git", "commit", "-m", "Initial commit: Project structure"], check=True)
        print("✓ Created initial commit")

        return True
    except subprocess.CalledProcessError:
        print("⚠️  Git not found or error occurred. Please install git and run:")
        print("   git init")
        print("   git add .")
        print('   git commit -m "Initial commit: Project structure"')
        return False
    except FileNotFoundError:
        print("⚠️  Git not found. Please install git from https://git-scm.com/")
        return False


def main():
    """Main function to set up the project."""
    print("=" * 60)
    print("Laser Trim AI System - Project Setup")
    print("=" * 60)

    # Check if we're in an empty directory or should create new one
    current_files = os.listdir(".")
    if current_files and not all(f.startswith('.') for f in current_files):
        response = input("Current directory is not empty. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return

    # Create structure
    create_directory_structure()
    create_readme()
    create_requirements()
    create_gitignore()
    create_setup_py()
    create_sample_config()
    create_sample_test()

    print("\n" + "=" * 60)
    print("✅ Project structure created successfully!")
    print("=" * 60)

    # Initialize git
    git_initialized = initialize_git()

    # Next steps
    print("\n📋 Next Steps:")
    print("1. Create a new repository on GitHub:")
    print("   • Go to https://github.com/new")
    print("   • Name: laser-trim-ai-system")
    print("   • Set as Private")
    print("   • DON'T initialize with README (we already have one)")

    if git_initialized:
        print("\n2. Connect to GitHub:")
        print("   git remote add origin https://github.com/YOUR_USERNAME/laser-trim-ai-system.git")
        print("   git branch -M main")
        print("   git push -u origin main")
    else:
        print("\n2. Initialize git and connect to GitHub:")
        print("   git init")
        print("   git add .")
        print('   git commit -m "Initial commit"')
        print("   git remote add origin https://github.com/YOUR_USERNAME/laser-trim-ai-system.git")
        print("   git branch -M main")
        print("   git push -u origin main")

    print("\n3. Share the repository URL with me for our first session!")
    print("\n4. Optional: Create and activate virtual environment:")
    print("   python -m venv venv")
    print("   # Windows: venv\\Scripts\\activate")
    print("   # Linux/Mac: source venv/bin/activate")
    print("   pip install -r requirements.txt")


if __name__ == "__main__":
    main()