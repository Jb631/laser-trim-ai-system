# Installation Guide for Laser Trim Analyzer v2

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for development installation)
- Windows, macOS, or Linux operating system

## Quick Installation

### Basic Installation

```bash
# Clone the repository (if installing from source)
git clone https://github.com/yourcompany/laser-trim-analyzer.git
cd laser-trim-analyzer

# Install the package
pip install -e .
```

### Full Installation (with all optional features)

```bash
# Install with all optional dependencies
pip install -e ".[all]"
```

## Detailed Installation Options

### 1. Basic Installation (Core Features Only)

```bash
pip install -e .
```

This installs:
- Core analysis functionality
- Basic GUI
- Database support
- Standard ML features

### 2. Development Installation

```bash
pip install -e ".[dev]"
```

Includes:
- Testing frameworks (pytest, coverage)
- Code formatters (black, ruff)
- Type checkers (mypy)
- Jupyter notebook support

### 3. Advanced ML Installation

```bash
pip install -e ".[ml-advanced]"
```

Includes:
- TensorFlow
- PyTorch
- Transformers
- Hyperparameter optimization (Optuna)

### 4. Performance Tools Installation

```bash
pip install -e ".[performance]"
```

Includes:
- Line profiler
- Py-spy profiler
- Scalene profiler

### 5. Complete Installation

```bash
pip install -e ".[all]"
```

Installs everything including dev tools, advanced ML, and performance tools.

## Verifying Installation

### 1. Check CLI Installation

```bash
# Check if the CLI is installed
lta --version

# Or use the full command
laser-trim-analyzer --version
```

### 2. Check GUI Installation

```bash
# Launch the GUI
laser-trim-analyzer-gui
```

### 3. Run Tests

```bash
# Run basic tests
pytest tests/

# Run with coverage
pytest --cov=laser_trim_analyzer tests/
```

### 4. Check Imports

```python
# Test in Python
python -c "import laser_trim_analyzer; print(laser_trim_analyzer.__version__)"
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd laser-trim-analyzer
   
   # Reinstall in editable mode
   pip install -e . --force-reinstall
   ```

2. **Missing Dependencies**
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   ```

3. **GUI Not Launching**
   - Ensure tkinter is installed:
     - Windows: Usually included with Python
     - macOS: `brew install python-tk`
     - Linux: `sudo apt-get install python3-tk`

4. **Database Connection Issues**
   - SQLite should work out of the box
   - For PostgreSQL/MySQL, install appropriate drivers:
     ```bash
     pip install psycopg2-binary  # PostgreSQL
     pip install pymysql          # MySQL
     ```

5. **ML Features Not Working**
   ```bash
   # Install ML dependencies
   pip install -e ".[ml-advanced]"
   ```

### Platform-Specific Notes

#### Windows
- Some packages may require Visual C++ Build Tools
- Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

#### macOS
- May need to install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

#### Linux
- May need to install system packages:
  ```bash
  # Ubuntu/Debian
  sudo apt-get update
  sudo apt-get install python3-dev python3-tk
  
  # Fedora/RHEL
  sudo dnf install python3-devel python3-tkinter
  ```

## Virtual Environment (Recommended)

### Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install package
pip install -e .

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create conda environment
conda create -n laser-trim-analyzer python=3.10
conda activate laser-trim-analyzer

# Install package
pip install -e .

# Deactivate when done
conda deactivate
```

## Post-Installation Setup

### 1. Initialize Database

```bash
lta db init
```

### 2. Configure Settings

```bash
# Copy example config
cp config/default.yaml config/local.yaml

# Edit settings
lta config edit
```

### 3. Set Up Cache

```bash
# Initialize cache with performance preset
lta cache setup --preset performance
```

### 4. Verify ML Models

```bash
# Check ML status
lta ml status
```

## Updating

### Update from Git

```bash
git pull origin main
pip install -e . --upgrade
```

### Update Dependencies

```bash
pip install -r requirements.txt --upgrade
```

## Uninstallation

```bash
# Uninstall package
pip uninstall laser-trim-analyzer

# Remove configuration files (optional)
rm -rf ~/.laser_trim_analyzer
```

## Getting Help

- Check the documentation: `docs/`
- Run `lta --help` for CLI help
- Submit issues: https://github.com/yourcompany/laser-trim-analyzer/issues