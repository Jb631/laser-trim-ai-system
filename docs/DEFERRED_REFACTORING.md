# Deferred Refactoring Issues

**Created:** 2026-01-07
**Source:** Code Review completed 2026-01-07

These 7 low-severity issues were identified during code review but deferred as they require structural refactoring better suited for a dedicated sprint.

---

## Summary

| ID | Issue | File(s) | Effort |
|----|-------|---------|--------|
| L-010 | Inconsistent logging levels | Multiple | Medium |
| L-011 | Missing return type hints | Multiple | Medium |
| L-012 | Long methods | trends.py | Medium |
| L-013 | Duplicated chart code | compare.py | High |
| L-014 | Duplicated chart code | export.py | High |
| L-015 | FIFO cache inefficiency | hashing.py | Low |
| L-017 | Missing __init__ type hints | GUI pages | Low |
| L-018 | Missing dataclass docs | ML modules | Low |

---

## Issue Details

### L-010: Inconsistent Logging Levels

**Files:** Multiple across codebase
**Category:** Best Practices

**Problem:**
Mix of `logger.info` and `logger.debug` for similar operations across modules. No clear standard for when to use each level.

**Solution:**
Audit all logging calls and standardize:
- `DEBUG`: Flow tracing, variable values, internal state
- `INFO`: Significant events (file processed, model trained, export complete)
- `WARNING`: Recoverable issues (fallback used, retry attempted)
- `ERROR`: Failures that affect functionality

**Files to audit:**
- `src/laser_trim_analyzer/core/*.py`
- `src/laser_trim_analyzer/ml/*.py`
- `src/laser_trim_analyzer/gui/pages/*.py`

---

### L-011: Missing Return Type Hints

**Files:** Multiple across codebase
**Category:** Best Practices

**Problem:**
Some methods lack return type hints, especially `-> None` for void methods.

**Solution:**
Add return type hints to all public methods:
```python
# Before
def _create_ui(self):
    ...

# After
def _create_ui(self) -> None:
    ...
```

**Priority files:**
- GUI page classes (`__init__`, `on_show`, `on_hide`, etc.)
- Widget classes
- Core processor methods

---

### L-012: Long Methods in trends.py

**File:** `src/laser_trim_analyzer/gui/pages/trends.py`
**Category:** Best Practices

**Problem:**
Several methods exceed 50 lines, reducing readability and making testing difficult.

**Solution:**
Break into smaller helper methods with clear responsibilities:

1. `_build_summary_section()` - Extract from `_create_ui()`
2. `_build_charts_section()` - Extract from `_create_ui()`
3. `_calculate_trend_metrics()` - Extract calculation logic
4. `_format_trend_display()` - Extract formatting logic

---

### L-013 & L-014: Duplicated Chart Code

**Files:**
- `src/laser_trim_analyzer/gui/pages/compare.py`
- `src/laser_trim_analyzer/gui/pages/export.py`
**Category:** Code Duplication

**Problem:**
Similar chart plotting patterns exist in:
- `compare.py`: `_plot_comparison()` and `_plot_comparison_export()`
- `export.py`: Multiple export methods with similar chart setup

**Solution:**
Create a shared `ChartExporter` utility class:

```python
# src/laser_trim_analyzer/gui/utils/chart_exporter.py

class ChartExporter:
    """Shared chart export utilities."""

    COLORS = {
        'final_test': '#27ae60',
        'trim': '#3498db',
        'spec_limit': '#e74c3c',
        'pass': '#27ae60',
        'fail': '#e74c3c',
    }

    @staticmethod
    def setup_comparison_chart(ax, data: Dict, mode: str = 'display') -> None:
        """
        Set up a comparison chart.

        Args:
            ax: Matplotlib axes
            data: Chart data dict
            mode: 'display' for dark theme, 'export' for light theme
        """
        ...

    @staticmethod
    def add_spec_limits(ax, upper: List, lower: List, positions: List) -> None:
        """Add spec limit shading to chart."""
        ...

    @staticmethod
    def export_to_file(fig, path: Path, dpi: int = 300) -> None:
        """Export figure to file with proper cleanup."""
        ...
```

**Refactoring steps:**
1. Create `chart_exporter.py` with shared methods
2. Update `compare.py` to use shared utilities
3. Update `export.py` to use shared utilities
4. Remove duplicated code

---

### L-015: FIFO Cache Inefficiency

**File:** `src/laser_trim_analyzer/utils/hashing.py`
**Line:** 61-64
**Category:** Optimization

**Problem:**
Removing items from dict by iterating over keys is O(n):
```python
keys_to_remove = list(_hash_cache.keys())[:_cache_max_size // 2]
for key in keys_to_remove:
    del _hash_cache[key]
```

**Solution:**
Use `functools.lru_cache` for automatic LRU eviction:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def _calculate_hash(file_path: str, mtime: float) -> str:
    """Calculate hash for a file (cached by path+mtime)."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def calculate_file_hash(file_path: Union[str, Path], use_cache: bool = True) -> str:
    """Calculate SHA256 hash of a file."""
    path = Path(file_path)
    path_str = str(path.resolve())
    mtime = path.stat().st_mtime

    if use_cache:
        return _calculate_hash(path_str, mtime)
    else:
        # Bypass cache
        return _calculate_hash.__wrapped__(path_str, mtime)

def clear_hash_cache() -> None:
    """Clear the hash cache."""
    _calculate_hash.cache_clear()
```

**Note:** This changes the API slightly - `get_cache_size()` would need to use `_calculate_hash.cache_info().currsize`.

---

### L-017: Missing __init__ Type Hints in GUI Pages

**Files:** All GUI page classes
**Category:** Best Practices

**Problem:**
`parent` and `app` parameters lack type hints in `__init__` methods.

**Solution:**
```python
# Before
def __init__(self, parent, app):
    super().__init__(parent)
    ...

# After
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from laser_trim_analyzer.gui.app import LaserTrimApp

def __init__(self, parent: ctk.CTkFrame, app: 'LaserTrimApp') -> None:
    super().__init__(parent)
    ...
```

**Files to update:**
- `src/laser_trim_analyzer/gui/pages/dashboard.py`
- `src/laser_trim_analyzer/gui/pages/process.py`
- `src/laser_trim_analyzer/gui/pages/analyze.py`
- `src/laser_trim_analyzer/gui/pages/compare.py`
- `src/laser_trim_analyzer/gui/pages/trends.py`
- `src/laser_trim_analyzer/gui/pages/settings.py`
- `src/laser_trim_analyzer/gui/pages/export.py`

---

### L-018: Missing Dataclass Documentation

**Files:** ML module dataclasses
**Category:** Best Practices

**Problem:**
Some dataclass fields lack documentation explaining their purpose.

**Solution:**
Add field descriptions using docstrings:

```python
@dataclass
class ThresholdResult:
    """
    Result of threshold calculation.

    Attributes:
        threshold: Calculated sigma threshold value
        confidence: Confidence level (0-1) based on sample size and separation
        method: Calculation method used ('separation', 'percentile', 'weighted', 'fallback')
        pass_sigma_mean: Mean sigma of passing samples
        pass_sigma_std: Standard deviation of passing samples
        ...
    """
    threshold: float
    confidence: float
    method: str
    ...
```

**Dataclasses to document:**
- `ml/predictor.py`: `PredictionResult`, `ModelPredictorState`
- `ml/threshold_optimizer.py`: `ThresholdResult`, `ThresholdOptimizerState`
- `ml/drift_detector.py`: `DriftResult`, `DriftDetectorState`
- `ml/profiler.py`: `ProfileStatistics`, `ModelProfile`, `ModelInsight`

---

## Priority Order

For a future refactoring sprint, recommended order:

1. **L-013 + L-014** (High impact) - Chart code deduplication reduces maintenance burden
2. **L-012** (Medium impact) - Breaking up trends.py improves testability
3. **L-010** (Medium impact) - Logging standardization helps debugging
4. **L-011 + L-017** (Low impact) - Type hints improve IDE support
5. **L-015** (Low impact) - Cache optimization is minor perf gain
6. **L-018** (Low impact) - Documentation can be done incrementally

---

## Code Review Completion Summary

**Original Review:** 2026-01-07
**Total Issues Found:** 34

| Severity | Fixed | Won't Fix | Deferred | Total |
|----------|-------|-----------|----------|-------|
| High | 4 | 0 | 0 | 4 |
| Medium | 11 | 1 | 0 | 12 |
| Low | 9 | 2 | 7 | 18 |
| **Total** | **24** | **3** | **7** | **34** |

**Completion Rate:** 79.4% (27/34 resolved)
