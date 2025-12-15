"""
GUI Pages for v3.

Pages:
- dashboard: At-a-glance health overview
- process: Import and process files
- analyze: View and compare results
- trends: Historical analysis and ML insights
- settings: Configuration
"""

from laser_trim_v3.gui.pages.dashboard import DashboardPage
from laser_trim_v3.gui.pages.process import ProcessPage
from laser_trim_v3.gui.pages.analyze import AnalyzePage
from laser_trim_v3.gui.pages.trends import TrendsPage
from laser_trim_v3.gui.pages.settings import SettingsPage

__all__ = [
    "DashboardPage",
    "ProcessPage",
    "AnalyzePage",
    "TrendsPage",
    "SettingsPage",
]
