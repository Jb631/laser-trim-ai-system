"""
GUI Widgets for v3.

Widgets:
- chart: Single chart widget for all visualizations (~400 lines)
- table: Data tables (TODO)
- cards: Metric cards and alerts (TODO)

Simplified from v2's 3000+ line chart system.
"""

from laser_trim_v3.gui.widgets.chart import ChartWidget, ChartStyle, COLORS

__all__ = [
    "ChartWidget",
    "ChartStyle",
    "COLORS",
]
