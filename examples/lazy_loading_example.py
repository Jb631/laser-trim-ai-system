"""Example of implementing lazy loading in the laser trim analyzer codebase."""

from typing import Optional, TYPE_CHECKING
from .lazy_imports import lazy_imports, lazy_import, LazyModule

# Type checking imports
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure


class OptimizedAnalyzer:
    """Example analyzer using lazy loading for heavy dependencies."""
    
    def __init__(self):
        self._np: Optional[LazyModule] = None
        self._pd: Optional[LazyModule] = None
        self._plt: Optional[LazyModule] = None
        
    @property
    def np(self) -> 'np':
        """Lazy-loaded numpy."""
        if self._np is None:
            self._np = lazy_imports.np
        return self._np
        
    @property
    def pd(self) -> 'pd':
        """Lazy-loaded pandas."""
        if self._pd is None:
            self._pd = lazy_imports.pd
        return self._pd
        
    @property
    def plt(self) -> 'plt':
        """Lazy-loaded matplotlib."""
        if self._plt is None:
            self._plt = lazy_imports.plt
        return self._plt
        
    def process_data(self, data: list) -> 'np.ndarray':
        """Process data using numpy (loaded on demand)."""
        return self.np.array(data)
        
    def create_dataframe(self, data: dict) -> 'pd.DataFrame':
        """Create dataframe (pandas loaded on demand)."""
        return self.pd.DataFrame(data)
        
    @lazy_import('scipy.stats', 'stats')
    def calculate_statistics(self, data: list) -> dict:
        """Calculate statistics (scipy loaded on demand)."""
        array = self.np.array(data)
        return {
            'mean': float(self.np.mean(array)),
            'std': float(self.np.std(array)),
            'skew': float(stats.skew(array)),
            'kurtosis': float(stats.kurtosis(array))
        }
        
    def create_plot(self, x: list, y: list) -> 'Figure':
        """Create plot (matplotlib loaded on demand)."""
        fig, ax = self.plt.subplots()
        ax.plot(x, y)
        return fig


# Pattern for module-level lazy imports
def get_heavy_analyzer():
    """Factory function that imports heavy analyzer only when needed."""
    from laser_trim_analyzer.analysis.heavy_analyzer import HeavyAnalyzer
    return HeavyAnalyzer()


# Pattern for conditional imports based on features
class FeatureBasedAnalyzer:
    """Analyzer that loads modules based on enabled features."""
    
    def __init__(self, enable_ml: bool = False, enable_plotting: bool = False):
        self.enable_ml = enable_ml
        self.enable_plotting = enable_plotting
        self._ml_engine = None
        self._plot_engine = None
        
    @property
    def ml_engine(self):
        """Lazy-loaded ML engine."""
        if self.enable_ml and self._ml_engine is None:
            from laser_trim_analyzer.ml.engine import MLEngine
            self._ml_engine = MLEngine()
        return self._ml_engine
        
    @property
    def plot_engine(self):
        """Lazy-loaded plotting engine."""
        if self.enable_plotting and self._plot_engine is None:
            from laser_trim_analyzer.utils.plotting_utils import PlottingEngine
            self._plot_engine = PlottingEngine()
        return self._plot_engine
        
    def analyze(self, data):
        """Analyze data using available engines."""
        results = {'basic_analysis': self._basic_analysis(data)}
        
        if self.enable_ml and self.ml_engine:
            results['ml_predictions'] = self.ml_engine.predict(data)
            
        if self.enable_plotting and self.plot_engine:
            results['plots'] = self.plot_engine.create_plots(data)
            
        return results
        
    def _basic_analysis(self, data):
        """Basic analysis without heavy dependencies."""
        return {
            'count': len(data),
            'first': data[0] if data else None,
            'last': data[-1] if data else None
        }


# Pattern for GUI components with lazy loading
class LazyGUIComponent:
    """GUI component that loads UI libraries on demand."""
    
    def __init__(self):
        self._tk = None
        self._ctk = None
        
    def create_window(self, use_custom_tkinter: bool = True):
        """Create window with lazy-loaded GUI library."""
        if use_custom_tkinter:
            if self._ctk is None:
                import customtkinter as ctk
                self._ctk = ctk
            return self._ctk.CTk()
        else:
            if self._tk is None:
                import tkinter as tk
                self._tk = tk
            return self._tk.Tk()


# Example of converting existing code to use lazy loading
class OriginalProcessor:
    """Original implementation with eager imports."""
    import numpy as np
    import pandas as pd
    from scipy import signal
    
    def process(self, data):
        return self.np.array(data)


class OptimizedProcessor:
    """Optimized implementation with lazy imports."""
    
    @lazy_import('numpy', 'np')
    @lazy_import('pandas', 'pd') 
    @lazy_import('scipy.signal', 'signal')
    def process(self, data):
        return np.array(data)


# Best practices example
class BestPracticesAnalyzer:
    """Demonstrates best practices for lazy loading."""
    
    def __init__(self):
        # Don't load anything in __init__
        self._cache = {}
        
    def light_operation(self, data: list) -> float:
        """Light operation using only standard library."""
        return sum(data) / len(data) if data else 0.0
        
    @lazy_import('numpy', 'np')
    def heavy_operation(self, data: list) -> 'np.ndarray':
        """Heavy operation that needs numpy."""
        array = np.array(data)
        return np.fft.fft(array)
        
    def conditional_operation(self, data: list, use_advanced: bool = False):
        """Operation that conditionally uses heavy libraries."""
        if use_advanced:
            # Import only when needed
            import scipy.stats as stats
            return stats.describe(data)
        else:
            # Use standard library for basic stats
            return {
                'mean': sum(data) / len(data),
                'min': min(data),
                'max': max(data)
            }
            
    def cached_heavy_operation(self, key: str):
        """Cached operation to avoid repeated imports."""
        if key in self._cache:
            return self._cache[key]
            
        # Import and compute only once
        import tensorflow as tf
        model = tf.keras.models.Sequential()
        # ... model setup ...
        
        self._cache[key] = model
        return model