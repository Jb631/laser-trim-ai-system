"""
Chart widgets module - split from chart_widget.py for maintainability.

This module provides ChartWidget and its component mixins:
- ChartWidget: Main chart widget class (backward compatible)
- BasicChartMixin: Line, bar, scatter, histogram, box, pie charts
- QualityChartMixin: Quality dashboards, gauges, metrics cards
- AnalyticsChartMixin: Control charts, capability, pattern analysis

Usage:
    from laser_trim_analyzer.gui.widgets.charts import ChartWidget

    # Or import from original location (backward compatible):
    from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
"""

from laser_trim_analyzer.gui.widgets.charts.base import ChartWidgetBase
from laser_trim_analyzer.gui.widgets.charts.basic_charts import BasicChartMixin
from laser_trim_analyzer.gui.widgets.charts.quality_charts import QualityChartMixin
from laser_trim_analyzer.gui.widgets.charts.analytics_charts import AnalyticsChartMixin


class ChartWidget(AnalyticsChartMixin, QualityChartMixin, BasicChartMixin, ChartWidgetBase):
    """
    Enhanced matplotlib chart widget for QA data visualization.

    Combines all chart functionality through mixins:
    - ChartWidgetBase: Core infrastructure, setup, theme management
    - BasicChartMixin: Standard chart types (line, bar, scatter, etc.)
    - QualityChartMixin: Quality dashboards, gauges, KPI cards
    - AnalyticsChartMixin: Advanced analytics (control charts, patterns)

    Features:
    - Multiple chart types (line, bar, scatter, histogram, heatmap)
    - Interactive zoom and pan
    - Export to various formats
    - Customizable styling
    - Real-time updates
    - CustomTkinter integration
    """
    pass  # All functionality comes from mixins


__all__ = [
    'ChartWidget',
    'ChartWidgetBase',
    'BasicChartMixin',
    'QualityChartMixin',
    'AnalyticsChartMixin',
]
