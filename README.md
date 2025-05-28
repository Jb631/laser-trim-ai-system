# AI-Powered Laser Trim Analysis System

## Overview

This is a modern, AI-powered quality assurance system for analyzing laser trim data from potentiometer manufacturing. The system provides accurate sigma gradient calculations that match existing MATLAB implementations while offering a modern Python-based architecture ready for AI/ML integration.

## Features

### Core Data Processing Engine ✅
- **Excel File Support**: Handles both .xls and .xlsx formats
- **Automatic System Detection**: Identifies System A/B configurations automatically
- **Multi-Track Support**: Processes TRK1/TRK2 data independently
- **MATLAB-Compatible Calculations**: Exact sigma gradient calculations matching legacy systems
- **Comprehensive Validation**: Built-in data validation and error handling

### Key Capabilities
- **Sigma Gradient Analysis**: Measures error variability with proven algorithms
- **Threshold Calculation**: Automatic pass/fail determination
- **Model-Specific Calibration**: Handles special cases like 8340-1 models
- **Batch Processing**: Process entire folders of files efficiently
- **Configurable Parameters**: JSON-based configuration system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jb631/laser-trim-ai-system.git
cd laser-trim-ai-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Single File Processing
```python
from data_processor import DataProcessor

# Create processor
processor = DataProcessor()

# Process a file
results = processor.process_file("path/to/your/file.xlsx")

# Display results
for track_id, track_data in results['tracks'].items():
    sigma = track_data['sigma_results']
    print(f"Track {track_id}: Sigma = {sigma.sigma_gradient:.6f} ({'PASS' if sigma.sigma_pass else 'FAIL'})")
```

### Batch Processing
```python
# Process entire folder
results = processor.batch_process("path/to/folder")

# Export summary
processor.export_results(results, "summary.xlsx")
```

### Using Configuration
```python
from config import create_configured_processor

# Create processor with custom configuration
processor = create_configured_processor("config/custom_config.json")
```

## Project Structure

```
laser-trim-ai-system/
├── core/
│   ├── data_processor.py      # Core data processing engine
│   ├── config.py              # Configuration management
│   └── test_data_processor.py # Unit tests
├── examples/
│   └── example_usage.py       # Usage examples
├── config/
│   └── default_config.json    # Default configuration
├── data/                      # Sample data (not in repo)
├── docs/                      # Documentation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Configuration

The system uses JSON configuration files for easy customization:

```json
{
    "processing": {
        "filter_sampling_freq": 100,
        "filter_cutoff_freq": 80,
        "gradient_step_size": 3,
        "default_scaling_factor": 24.0
    },
    "system_a": {
        "columns": {
            "error": 6,
            "position": 7,
            "upper_limit": 8,
            "lower_limit": 9
        }
    }
}
```

## Data Format

### System A Files
- Multiple tracks (TRK1, TRK2)
- Sheet naming: "SEC1 TRK1 0" (untrimmed), "SEC1 TRK1 TRM" (trimmed)
- Unit properties in cells B26 (length) and B10 (resistance)

### System B Files
- Single track configuration
- Sheets: "test" (untrimmed), "Lin Error" (final)
- Unit properties in cells K1 (length) and R1 (resistance)

## Validation

The sigma gradient calculation implements the exact MATLAB algorithm:

1. **Filtering**: First-order digital filter (forward-backward)
2. **Gradient Calculation**: Fixed step size (default: 3 points)
3. **Standard Deviation**: Sample standard deviation (ddof=1)
4. **Threshold**: (Linearity Spec / Unit Length) × Scaling Factor

## Testing

Run the test suite to verify functionality:

```bash
python -m unittest test_data_processor.py
```

## Roadmap

- [x] Core Data Processing Engine
- [ ] Machine Learning Models
- [ ] Excel Report Generator
- [ ] GUI Application
- [ ] Database Integration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is proprietary software for internal QA use.

## Support

For questions or issues, please contact the QA Team.

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Core Engine Complete ✅