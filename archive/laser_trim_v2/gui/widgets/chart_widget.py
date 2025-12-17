"""
ChartWidget for QA Dashboard

A versatile matplotlib wrapper widget with zoom, pan, and export functionality.
Supports multiple chart types for QA data visualization.

REFACTORED: This module now re-exports ChartWidget from the modular charts/ package.
The implementation has been split into smaller, maintainable modules:
- charts/base.py: Core infrastructure (ChartWidgetBase)
- charts/basic_charts.py: Standard charts (BasicChartMixin)
- charts/quality_charts.py: Quality dashboards (QualityChartMixin)
- charts/analytics_charts.py: Advanced analytics (AnalyticsChartMixin)

Backward compatibility is preserved - all existing imports will work:
    from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
"""

# Re-export ChartWidget and components from the modular package
from laser_trim_analyzer.gui.widgets.charts import (
    ChartWidget,
    ChartWidgetBase,
    BasicChartMixin,
    QualityChartMixin,
    AnalyticsChartMixin,
)

# For backward compatibility with any code that references these directly
__all__ = [
    'ChartWidget',
    'ChartWidgetBase',
    'BasicChartMixin',
    'QualityChartMixin',
    'AnalyticsChartMixin',
]
