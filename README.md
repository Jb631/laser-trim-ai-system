# README.md
# Laser Trim Analyzer v2

A modern, AI-enhanced quality analysis platform for potentiometer laser trim data.

## Features

- **Multi-track Analysis**: Full support for System A (TRK1/TRK2) and System B files
- **Advanced Analytics**: Sigma gradient, linearity optimization, failure prediction
- **Machine Learning**: Automated threshold optimization and predictive maintenance
- **Database Integration**: Historical tracking and trend analysis
- **Modern GUI**: Clean, intuitive interface with real-time insights
- **AI Integration**: Optional cloud AI for anomaly detection and recommendations

## Installation

### Requirements
- Python 3.10 or higher
- Windows 10/11, macOS, or Linux

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourcompany/laser-trim-analyzer-v2.git
cd laser-trim-analyzer-v2
```

2. Create a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install the package:
```bash
pip install -e .
```

## Quick Start

### GUI Application
```bash
laser-trim-analyzer-gui
```

### Command Line
```bash
lta analyze /path/to/data --output /path/to/results
```

### Python API
```python
from laser_trim_analyzer import Config, AnalysisResult
from laser_trim_analyzer.analysis import SigmaAnalyzer

# Load configuration
config = Config()

# Analyze a file
analyzer = SigmaAnalyzer(config)
result = analyzer.analyze_file("path/to/file.xlsx")

# Access results
print(f"Sigma gradient: {result.primary_track.sigma_analysis.sigma_gradient}")
print(f"Pass/Fail: {result.overall_status}")
```

## Configuration

Configuration is managed through YAML files in the `config/` directory:

- `default.yaml`: Default settings
- `production.yaml`: Production environment settings

You can also use environment variables with the `LTA_` prefix:

```bash
export LTA_DEBUG=true
export LTA_DATABASE__PATH=/custom/path/to/db
```

## Project Structure

```
src/laser_trim_analyzer/
├── core/           # Core models and configuration
├── analysis/       # Analysis engines
├── database/       # Database management
├── ml/            # Machine learning components
├── gui/           # Graphical interface
├── api/           # API client for AI services
└── utils/         # Utility functions
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/

# Lint
ruff src/

# Type check
mypy src/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Documentation

Full documentation is available at [docs/index.md](docs/index.md).

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please contact the QA team or create an issue in the repository.