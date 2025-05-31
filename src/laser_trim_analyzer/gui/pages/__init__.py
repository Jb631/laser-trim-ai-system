"""
GUI Pages for Laser Trim Analyzer

This package contains all the page implementations for the application.
Each page represents a different section of functionality.
"""

from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.pages.home_page import HomePage
from laser_trim_analyzer.gui.pages.analysis_page import AnalysisPage
from laser_trim_analyzer.gui.pages.historical_page import HistoricalPage
from laser_trim_analyzer.gui.pages.ml_tools_page import MLToolsPage
from laser_trim_analyzer.gui.pages.ai_insights_page import AIInsightsPage
from laser_trim_analyzer.gui.pages.settings_page import SettingsPage

__all__ = [
    'BasePage',
    'HomePage',
    'AnalysisPage',
    'HistoricalPage',
    'MLToolsPage',
    'AIInsightsPage',
    'SettingsPage',
]