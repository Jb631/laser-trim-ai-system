"""
GUI Pages for v3.

Pages:
- dashboard: At-a-glance health overview
- process: Import and process files
- analyze: View and compare results
- trends: Historical analysis and ML insights
- export: Export charts from database
- settings: Configuration
"""

from laser_trim_analyzer.gui.pages.dashboard import DashboardPage
from laser_trim_analyzer.gui.pages.process import ProcessPage
from laser_trim_analyzer.gui.pages.analyze import AnalyzePage
from laser_trim_analyzer.gui.pages.trends import TrendsPage
from laser_trim_analyzer.gui.pages.export import ExportPage
from laser_trim_analyzer.gui.pages.settings import SettingsPage

__all__ = [
    "DashboardPage",
    "ProcessPage",
    "AnalyzePage",
    "TrendsPage",
    "ExportPage",
    "SettingsPage",
]
