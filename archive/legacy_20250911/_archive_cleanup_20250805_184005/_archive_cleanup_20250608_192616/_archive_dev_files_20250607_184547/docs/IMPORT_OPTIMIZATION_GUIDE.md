# Import Optimization Guide

This guide provides best practices and tools for optimizing imports in the Laser Trim Analyzer codebase.

## Table of Contents

1. [Overview](#overview)
2. [Import Analysis Tools](#import-analysis-tools)
3. [Lazy Loading System](#lazy-loading-system)
4. [Best Practices](#best-practices)
5. [Performance Impact](#performance-impact)
6. [Code Structure Guidelines](#code-structure-guidelines)
7. [Automated Cleanup](#automated-cleanup)

## Overview

Efficient import management is crucial for:
- Reducing application startup time
- Minimizing memory footprint
- Avoiding circular dependencies
- Improving code maintainability

## Import Analysis Tools

### Using the Import Analyzer

```python
from laser_trim_analyzer.utils.import_optimizer import ImportAnalyzer

# Analyze a single file
analyzer = ImportAnalyzer()
analysis = analyzer.analyze_file(Path("src/module.py"))

# Analyze entire project
results = analyzer.analyze_project()

# Check for issues
for file_path, analysis in results.items():
    if analysis.unused_imports:
        print(f"{file_path}: {len(analysis.unused_imports)} unused imports")
    if analysis.circular_imports:
        print(f"{file_path}: Circular imports detected!")
```

### Command Line Usage

```bash
# Analyze current directory
python -m laser_trim_analyzer.utils.optimize_imports_cli

# Analyze specific file
python -m laser_trim_analyzer.utils.optimize_imports_cli src/module.py --verbose

# Show performance metrics
python -m laser_trim_analyzer.utils.optimize_imports_cli --show-performance

# Optimize imports (dry run)
python -m laser_trim_analyzer.utils.optimize_imports_cli --optimize

# Apply optimizations
python -m laser_trim_analyzer.utils.optimize_imports_cli --optimize --apply
```

## Lazy Loading System

### Basic Usage

```python
from laser_trim_analyzer.utils.lazy_imports import lazy_imports

# Register a module for lazy loading
lazy_imports.register('heavy_module', 'hm')

# Use the module (loaded on first access)
data = lazy_imports.hm.process_data()
```

### Using Decorators

```python
from laser_trim_analyzer.utils.lazy_imports import lazy_import

@lazy_import('pandas', 'pd')
def process_dataframe():
    # pandas is imported only when this function is called
    return pd.DataFrame({'a': [1, 2, 3]})
```

### Temporary Imports

```python
from laser_trim_analyzer.utils.lazy_imports import temporary_import

# Import is unloaded after use
with temporary_import('scipy.stats', 'stats') as stats:
    result = stats.norm.pdf(0)
```

### Heavy Dependencies

The following modules are pre-registered for lazy loading:
- `numpy` (alias: `np`)
- `pandas` (alias: `pd`)
- `matplotlib.pyplot` (alias: `plt`)
- `scipy`
- `tensorflow` (alias: `tf`)
- `torch`
- `sklearn`
- `cv2`
- `PIL.Image` (alias: `Image`)
- `plotly`
- `seaborn` (alias: `sns`)

## Best Practices

### 1. Import Organization

Follow PEP 8 conventions:

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from laser_trim_analyzer.core import models
from laser_trim_analyzer.utils import validators
```

### 2. Avoid Circular Imports

**Bad:**
```python
# module_a.py
from module_b import function_b

def function_a():
    return function_b()

# module_b.py
from module_a import function_a  # Circular!

def function_b():
    return function_a()
```

**Good:**
```python
# module_a.py
def function_a():
    from module_b import function_b  # Import inside function
    return function_b()

# module_b.py
def function_b():
    from module_a import function_a
    return function_a()
```

### 3. Use Type Checking Imports

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # These imports are only for type hints
    import pandas as pd
    from laser_trim_analyzer.core.models import AnalysisResult

def process_data(df: 'pd.DataFrame') -> 'AnalysisResult':
    # Actual import when needed
    import pandas as pd
    from laser_trim_analyzer.core.models import AnalysisResult
    ...
```

### 4. Lazy Load Heavy Dependencies

```python
# Bad: Import at module level
import matplotlib.pyplot as plt

def plot_data(data):
    plt.plot(data)

# Good: Import when needed
def plot_data(data):
    import matplotlib.pyplot as plt
    plt.plot(data)
```

## Performance Impact

### Measuring Import Time

```python
from laser_trim_analyzer.utils.import_optimizer import ImportAnalyzer

analyzer = ImportAnalyzer()
analysis = analyzer.analyze_file(Path("module.py"))

# Check import times
for module, time in analysis.import_time.items():
    if time > 0.1:  # Slow imports (>100ms)
        print(f"{module}: {time:.3f}s")
```

### Common Performance Issues

1. **Heavy modules at startup**
   - Solution: Use lazy loading
   
2. **Importing entire packages**
   - Bad: `import scipy`
   - Good: `from scipy import stats`
   
3. **Unnecessary imports**
   - Use the analyzer to find and remove unused imports

## Code Structure Guidelines

### 1. Module Organization

- Keep modules focused and cohesive
- Avoid modules with >20 imports (sign of doing too much)
- Group related functionality

### 2. Dependency Hierarchy

```
core/
  ├── interfaces.py    # No external dependencies
  ├── models.py        # Minimal dependencies
  └── implementations.py # Can depend on models/interfaces

analysis/
  ├── base.py          # Abstract base classes
  └── specific.py      # Can depend on base

utils/
  └── helpers.py       # Should not depend on core business logic
```

### 3. Import Patterns by Layer

**Core Layer:**
- Minimal external dependencies
- No GUI dependencies
- Prefer standard library

**Analysis Layer:**
- Can import numpy, pandas for computation
- Lazy load visualization libraries

**GUI Layer:**
- Import UI frameworks
- Lazy load heavy processing libraries

## Automated Cleanup

### Remove Unused Imports

```bash
# Dry run
python -m laser_trim_analyzer.utils.optimize_imports_cli --optimize

# Apply changes
python -m laser_trim_analyzer.utils.optimize_imports_cli --optimize --apply
```

### Sort Imports

```python
from laser_trim_analyzer.utils.import_optimizer import ImportCleaner

ImportCleaner.sort_imports(Path("module.py"))
```

### Continuous Integration

Add to your CI pipeline:

```yaml
- name: Check imports
  run: |
    python -m laser_trim_analyzer.utils.optimize_imports_cli --analyze-only --json > import_report.json
    # Fail if unused imports found
    python -c "import json; data=json.load(open('import_report.json')); exit(1 if any(a['unused_imports'] for a in data.values()) else 0)"
```

## Example Optimization Session

```bash
# 1. Analyze current state
$ python -m laser_trim_analyzer.utils.optimize_imports_cli --verbose --show-performance

# 2. Identify heavy imports
$ python -m laser_trim_analyzer.utils.optimize_imports_cli --show-heavy

# 3. Check for circular dependencies
$ python -m laser_trim_analyzer.utils.optimize_imports_cli --show-circular

# 4. Optimize (dry run first)
$ python -m laser_trim_analyzer.utils.optimize_imports_cli --optimize

# 5. Apply optimizations
$ python -m laser_trim_analyzer.utils.optimize_imports_cli --optimize --apply

# 6. Verify improvements
$ python -m laser_trim_analyzer.utils.optimize_imports_cli --show-performance
```

## Monitoring Import Performance

Use the built-in monitoring:

```python
from laser_trim_analyzer.utils.lazy_imports import import_monitor

# After running your application
stats = import_monitor.get_stats()
print(f"Total import time: {stats['total_time']:.2f}s")
print(f"Slowest imports: {stats['slowest_imports'][:5]}")
```

## Troubleshooting

### Common Issues

1. **ImportError after optimization**
   - Check if removed import was used indirectly
   - Use `--dry-run` first

2. **Circular import after refactoring**
   - Use the analyzer to detect cycles
   - Move imports inside functions if needed

3. **Performance regression**
   - Profile import times before/after
   - Consider preloading critical modules

### Debug Mode

```python
import logging
logging.getLogger('laser_trim_analyzer.utils.import_optimizer').setLevel(logging.DEBUG)
logging.getLogger('laser_trim_analyzer.utils.lazy_imports').setLevel(logging.DEBUG)
```

This will show detailed information about import analysis and lazy loading.