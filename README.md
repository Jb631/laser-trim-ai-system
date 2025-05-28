# Laser Trim AI System

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
venv\Scripts\activate
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
