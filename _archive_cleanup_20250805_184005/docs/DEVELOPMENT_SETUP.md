# Development Setup Guide

This guide will help you set up a development environment for the Laser Trim Analyzer.

## Overview

The Laser Trim Analyzer supports multiple environments:
- **Production**: For deployed application use
- **Development**: For local development and testing
- **Deployment**: For packaged application distribution

## Prerequisites

- Python 3.8 or higher
- Git (for version control)
- Windows OS (application is Windows-specific due to tkinterdnd2)

## Initial Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd laser_trim_analyzer_v2
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -e .
```

This installs the application in editable mode with all dependencies.

### 4. Initialize Development Environment

```bash
python scripts/init_dev_database.py --clean --seed-data
```

This will:
- Create development directories
- Initialize a fresh development database
- Optionally seed test data for development
- Create necessary configuration files

## Running the Application

### Development Mode

Use the provided batch file:
```bash
run_dev.bat
```

Or manually:
```bash
set LTA_ENV=development
python src/__main__.py
```

### Production Mode

```bash
python src/__main__.py
```

## Configuration

### Environment Variables

- `LTA_ENV`: Controls which configuration file to use
  - `development`: Uses `config/development.yaml`
  - `production`: Uses `config/production.yaml` (default)

### Configuration Files

1. **config/development.yaml**
   - Local database in `%LOCALAPPDATA%`
   - Debug mode enabled
   - Lower thresholds for testing
   - Local API endpoints

2. **config/production.yaml**
   - Production database path
   - Performance optimizations
   - Production API endpoints
   - Higher validation thresholds

3. **config/deployment.yaml**
   - Supports single-user and multi-user modes
   - Configurable paths for deployment

## Directory Structure

### Development Environment
```
%LOCALAPPDATA%/LaserTrimAnalyzer/dev/
├── laser_trim_dev.db      # Development database
├── models/                 # ML models
└── logs/                   # Application logs

%USERPROFILE%/Documents/LaserTrimAnalyzer/dev/
├── data/                   # Analysis data
├── exports/                # Excel exports
├── plots/                  # Generated plots
└── temp/                   # Temporary files
```

### Production Environment
```
D:/LaserTrimData/
├── production.db           # Production database
├── models/                 # ML models
├── Production/             # Production data
└── Logs/                   # Production logs
```

## Testing

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test
```bash
pytest tests/test_core_functionality.py
```

### Run with Coverage
```bash
pytest --cov=laser_trim_analyzer tests/
```

## Common Development Tasks

### Reset Development Database
```bash
python scripts/init_dev_database.py --clean
```

### Seed Test Data
```bash
python scripts/init_dev_database.py --seed-data
```

### Switch Between Environments
```bash
# Development
set LTA_ENV=development

# Production
set LTA_ENV=production
```

### Check Database Contents
```python
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.core.config import get_config

config = get_config()
db = DatabaseManager(config)

with db.get_session() as session:
    count = session.query(AnalysisResult).count()
    print(f"Database contains {count} records")
```

## Troubleshooting

### Database Issues

1. **Permission Errors**
   - Ensure you have write permissions to the directories
   - Try running as administrator if needed

2. **Database Locked**
   - Close any other instances of the application
   - Check for zombie processes

3. **Wrong Database**
   - Check `LTA_ENV` environment variable
   - Verify configuration file is being loaded

### Import Errors

1. **Module Not Found**
   - Ensure you installed with `pip install -e .`
   - Check virtual environment is activated

2. **tkinterdnd2 Issues**
   - This is Windows-specific
   - May need Visual C++ redistributables

## Best Practices

1. **Always use development mode for testing**
   - Prevents corruption of production data
   - Easier to reset and test

2. **Keep configurations separate**
   - Don't commit production credentials
   - Use environment-specific settings

3. **Test database operations**
   - Use transactions for testing
   - Roll back test data after tests

4. **Document configuration changes**
   - Update this guide when adding new settings
   - Keep CHANGELOG.md updated

## Additional Resources

- [CLAUDE.md](../CLAUDE.md) - Project guidelines and rules
- [CHANGELOG.md](../CHANGELOG.md) - Recent changes and fixes
- [README.md](../README.md) - General project information