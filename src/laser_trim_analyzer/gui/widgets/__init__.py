"""
GUI Widgets for v3.

Widgets:
- chart: Single chart widget for all visualizations (~400 lines)
- table: Data tables (TODO)
- cards: Metric cards and alerts (TODO)

Simplified from v2's 3000+ line chart system.

NOTE: ChartWidget uses lazy imports to defer matplotlib loading.
Import directly from chart module when needed:
    from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle
"""

# Lazy imports - don't import ChartWidget at module level to avoid loading matplotlib
# Users should import directly from the chart module when needed

__all__ = [
    "ChartWidget",
    "ChartStyle",
    "COLORS",
]


def __getattr__(name):
    """Lazy import for ChartWidget and related classes."""
    if name in ("ChartWidget", "ChartStyle", "COLORS"):
        from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle, COLORS
        globals()["ChartWidget"] = ChartWidget
        globals()["ChartStyle"] = ChartStyle
        globals()["COLORS"] = COLORS
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
