"""
GUI Pages for v4.

Pages:
- dashboard: At-a-glance health overview
- quality_health: Operational quality status — better/worse/action per model
- process: Import and process files
- analyze: View and compare results
- trends: Historical analysis and ML insights
- export: Export charts from database
- settings: Configuration
"""

from laser_trim_analyzer.gui.pages.dashboard import DashboardPage
from laser_trim_analyzer.gui.pages.quality_health import QualityHealthPage
from laser_trim_analyzer.gui.pages.process import ProcessPage
from laser_trim_analyzer.gui.pages.analyze import AnalyzePage
from laser_trim_analyzer.gui.pages.trends import TrendsPage
from laser_trim_analyzer.gui.pages.export import ExportPage
from laser_trim_analyzer.gui.pages.settings import SettingsPage

__all__ = [
    "DashboardPage",
    "QualityHealthPage",
    "ProcessPage",
    "AnalyzePage",
    "TrendsPage",
    "ExportPage",
    "SettingsPage",
]
