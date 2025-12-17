# Laser Trim Analyzer v3

A streamlined, production-ready quality analysis platform for potentiometer laser trim data.

## Features

- **Multi-track Analysis**: Full support for System A and System B files
- **Sigma Gradient Analysis**: Pass/fail determination with ML-optimized thresholds
- **Linearity Analysis**: Error vs position with spec limit validation
- **Database Integration**: SQLite database for historical tracking
- **Trend Analysis**: Model-level trends and alerts
- **Excel Export**: Clean, professional Excel reports
- **Modern GUI**: Dark-themed, intuitive interface

## Quick Start

### Running from Source

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run the application
python src/__main__.py
```

### Deployed Executable

1. Run `deploy.bat` to build the executable
2. Find the package in `dist/LaserTrimAnalyzer-v3.x.x/`
3. Copy to your work computer
4. Run `LaserTrimAnalyzer.exe`

## Project Structure

```
laser-trim-analyzer/
├── src/
│   └── laser_trim_analyzer/    # Main application code
│       ├── core/               # Parser, processor, models
│       ├── database/           # SQLite database manager
│       ├── gui/                # CustomTkinter GUI
│       │   ├── pages/          # Dashboard, Process, Analyze, Trends, Settings
│       │   └── widgets/        # Chart widget
│       ├── ml/                 # Threshold optimizer, drift detector
│       └── export/             # Excel export
├── test_files/                 # Sample data files for testing
├── archive/                    # Archived V2 code and docs
├── deploy.bat                  # Build deployment package
└── pyproject.toml              # Project configuration
```

## Configuration

V3 uses a self-contained configuration system:
- Database: `./data/analysis.db` (relative to app)
- User settings: `~/.laser_trim_analyzer/config.yaml`

No external config files required - everything is self-contained for easy deployment.

## Development

### Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Building Deployment Package

```bash
deploy.bat
```

This creates a versioned folder (e.g., `LaserTrimAnalyzer-v3.0.0-20251216/`) ready for distribution.

## Version History

- **v3.0.0** (2025-12-16): Complete redesign with simplified architecture
  - Reduced from 110 files to ~30 files
  - Self-contained configuration
  - Improved ML integration
  - Excel-only export
  - Dark theme GUI

## License

MIT License - see [LICENSE](LICENSE) file.
