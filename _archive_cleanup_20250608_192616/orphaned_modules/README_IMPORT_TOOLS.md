# Import Optimization Tools

This directory contains tools for analyzing and optimizing Python imports throughout the laser trim analyzer codebase.

## Quick Start

### Analyze imports in your code:
```bash
# Analyze current directory
python -m laser_trim_analyzer.utils.optimize_imports_cli

# Analyze specific file
python -m laser_trim_analyzer.utils.optimize_imports_cli src/module.py

# Show detailed analysis
python -m laser_trim_analyzer.utils.optimize_imports_cli --verbose --show-performance
```

### Optimize imports:
```bash
# Dry run (see what would change)
python -m laser_trim_analyzer.utils.optimize_imports_cli --optimize

# Apply optimizations
python -m laser_trim_analyzer.utils.optimize_imports_cli --optimize --apply
```

## Components

### 1. Import Optimizer (`import_optimizer.py`)
- **ImportAnalyzer**: Analyzes Python files for import issues
- **ImportOptimizer**: Applies optimizations based on analysis
- **ImportCleaner**: Utilities for sorting and cleaning imports

### 2. Lazy Imports (`lazy_imports.py`)
- **LazyModule**: Deferred module loading
- **LazyImporter**: Global lazy import manager
- **Decorators**: `@lazy_import` for function-level lazy loading
- **Context managers**: `temporary_import` for scoped imports

### 3. CLI Tool (`optimize_imports_cli.py`)
Command-line interface for running import analysis and optimization.

## Common Use Cases

### Finding Unused Imports
```python
from laser_trim_analyzer.utils.import_optimizer import ImportAnalyzer

analyzer = ImportAnalyzer()
analysis = analyzer.analyze_file(Path("module.py"))

for unused in analysis.unused_imports:
    print(f"Line {unused.line}: {unused.module}")
```

### Implementing Lazy Loading
```python
from laser_trim_analyzer.utils.lazy_imports import lazy_imports, lazy_import

# Method 1: Global lazy imports
np = lazy_imports.np  # Not loaded until first use
array = np.array([1, 2, 3])  # Loaded here

# Method 2: Decorator
@lazy_import('pandas', 'pd')
def process_data():
    return pd.DataFrame({'a': [1, 2, 3]})

# Method 3: Property pattern
class Analyzer:
    @property
    def numpy(self):
        return lazy_imports.np
```

### Detecting Circular Imports
```bash
python -m laser_trim_analyzer.utils.optimize_imports_cli --show-circular
```

## Performance Benefits

1. **Faster Startup**: Heavy modules loaded only when needed
2. **Lower Memory**: Unused modules never loaded
3. **Better Testing**: Easier to mock dependencies
4. **Cleaner Code**: Unused imports removed

## Integration Examples

See `lazy_loading_example.py` for patterns to integrate into existing code.

## Best Practices

1. **Always lazy-load**:
   - numpy, pandas, matplotlib
   - scipy, sklearn, tensorflow
   - Any module taking >50ms to import

2. **Import at function level** for rarely-used features:
   ```python
   def advanced_analysis(data):
       import scipy.stats  # Only imported when function called
       return scipy.stats.describe(data)
   ```

3. **Use TYPE_CHECKING** for type hints:
   ```python
   from typing import TYPE_CHECKING
   
   if TYPE_CHECKING:
       import pandas as pd
   
   def process(df: 'pd.DataFrame'):
       import pandas as pd  # Actual import
       ...
   ```

## Monitoring

Track import performance:
```python
from laser_trim_analyzer.utils.lazy_imports import import_monitor

# After running application
stats = import_monitor.get_stats()
print(f"Total import time: {stats['total_time']:.2f}s")
print(f"Slowest imports: {stats['slowest_imports']}")
```