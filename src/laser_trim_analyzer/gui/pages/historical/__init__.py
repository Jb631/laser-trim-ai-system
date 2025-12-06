"""
Historical page module - split from historical_page.py for maintainability.

This module provides the HistoricalPage with its component mixins:
- HistoricalPage: Main historical data analysis page
- AnalyticsMixin: Trend analysis, correlation, prediction, anomaly detection
- SPCMixin: Control charts, capability, Pareto, drift detection

Usage:
    from laser_trim_analyzer.gui.pages.historical import HistoricalPage

    # Or import from original location (backward compatible):
    from laser_trim_analyzer.gui.pages.historical_page import HistoricalPage
"""

from laser_trim_analyzer.gui.pages.historical.analytics_mixin import AnalyticsMixin
from laser_trim_analyzer.gui.pages.historical.spc_mixin import SPCMixin

__all__ = [
    'AnalyticsMixin',
    'SPCMixin',
]
