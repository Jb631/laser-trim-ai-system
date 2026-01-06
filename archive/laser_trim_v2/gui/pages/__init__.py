"""
GUI Pages for Laser Trim Analyzer

This package contains all the page implementations for the application.
Each page represents a different section of functionality.
"""

import logging

# Set up logger for import issues
logger = logging.getLogger(__name__)

# Import base page with CustomTkinter support
from laser_trim_analyzer.gui.pages.base_page_ctk import BasePage

# Import pages with error handling
try:
    from laser_trim_analyzer.gui.pages.home_page import HomePage
except ImportError as e:
    logger.error(f"Could not import HomePage: {e}", exc_info=True)
    HomePage = None



try:
    from laser_trim_analyzer.gui.pages.historical_page import HistoricalPage
except ImportError as e:
    logger.error(f"Could not import HistoricalPage: {e}", exc_info=True)
    HistoricalPage = None

try:
    from laser_trim_analyzer.gui.pages.model_summary_page import ModelSummaryPage
except ImportError as e:
    logger.error(f"Could not import ModelSummaryPage: {e}", exc_info=True)
    ModelSummaryPage = None

try:
    from laser_trim_analyzer.gui.pages.ml_tools_page import MLToolsPage
except ImportError as e:
    logger.error(f"Could not import MLToolsPage: {e}", exc_info=True)
    MLToolsPage = None

try:
    from laser_trim_analyzer.gui.pages.ai_insights_page import AIInsightsPage
except ImportError as e:
    logger.error(f"Could not import AIInsightsPage: {e}", exc_info=True)
    AIInsightsPage = None

try:
    from laser_trim_analyzer.gui.pages.settings_page import SettingsPage
except ImportError as e:
    logger.error(f"Could not import SettingsPage: {e}", exc_info=True)
    SettingsPage = None

try:
    from laser_trim_analyzer.gui.pages.single_file_page import SingleFilePage
except ImportError as e:
    logger.error(f"Could not import SingleFilePage: {e}", exc_info=True)
    SingleFilePage = None

try:
    from laser_trim_analyzer.gui.pages.batch_processing_page import BatchProcessingPage
except ImportError as e:
    logger.error(f"Could not import BatchProcessingPage: {e}", exc_info=True)
    BatchProcessingPage = None

try:
    from laser_trim_analyzer.gui.pages.multi_track_page import MultiTrackPage
except ImportError as e:
    logger.error(f"Could not import MultiTrackPage: {e}", exc_info=True)
    MultiTrackPage = None

try:
    from laser_trim_analyzer.gui.pages.final_test_comparison_page import FinalTestComparisonPage
except ImportError as e:
    logger.error(f"Could not import FinalTestComparisonPage: {e}", exc_info=True)
    FinalTestComparisonPage = None

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
    'FinalTestComparisonPage',
    # 'MLModelInfoAnalyzers',
]