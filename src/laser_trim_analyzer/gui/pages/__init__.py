"""
GUI Pages for Laser Trim Analyzer

This package contains all the page implementations for the application.
Each page represents a different section of functionality.
"""

# Import base page with CustomTkinter support
from laser_trim_analyzer.gui.pages.base_page_ctk import BasePage

# Import pages with error handling
try:
    from laser_trim_analyzer.gui.pages.home_page import HomePage
except ImportError as e:
    print(f"Warning: Could not import HomePage: {e}")
    HomePage = None



try:
    from laser_trim_analyzer.gui.pages.historical_page import HistoricalPage
except ImportError as e:
    print(f"Warning: Could not import HistoricalPage: {e}")
    HistoricalPage = None

try:
    from laser_trim_analyzer.gui.pages.model_summary_page import ModelSummaryPage
except ImportError as e:
    print(f"Warning: Could not import ModelSummaryPage: {e}")
    ModelSummaryPage = None

try:
    from laser_trim_analyzer.gui.pages.ml_tools_page import MLToolsPage
except ImportError as e:
    print(f"Warning: Could not import MLToolsPage: {e}")
    MLToolsPage = None

try:
    from laser_trim_analyzer.gui.pages.ai_insights_page import AIInsightsPage
except ImportError as e:
    print(f"Warning: Could not import AIInsightsPage: {e}")
    AIInsightsPage = None

try:
    from laser_trim_analyzer.gui.pages.settings_page import SettingsPage
except ImportError as e:
    print(f"Warning: Could not import SettingsPage: {e}")
    SettingsPage = None

try:
    from laser_trim_analyzer.gui.pages.single_file_page import SingleFilePage
except ImportError as e:
    print(f"Warning: Could not import SingleFilePage: {e}")
    SingleFilePage = None

try:
    from laser_trim_analyzer.gui.pages.batch_processing_page import BatchProcessingPage
except ImportError as e:
    print(f"Warning: Could not import BatchProcessingPage: {e}")
    BatchProcessingPage = None

try:
    from laser_trim_analyzer.gui.pages.multi_track_page import MultiTrackPage
except ImportError as e:
    print(f"Warning: Could not import MultiTrackPage: {e}")
    MultiTrackPage = None

__all__ = [
    'BasePage',
    'HomePage',
    'HistoricalPage',
    'ModelSummaryPage',
    'MLToolsPage',
    'AIInsightsPage',
    'SettingsPage',
    'SingleFilePage',
    'BatchProcessingPage',
    'MultiTrackPage',
    # 'MLModelInfoAnalyzers',
]