"""GUI widgets for Laser Trim Analyzer."""

from laser_trim_analyzer.gui.widgets.stat_card import StatCard
from laser_trim_analyzer.gui.widgets.file_drop_zone import FileDropZone
from laser_trim_analyzer.gui.widgets.status_bar import StatusBar
from laser_trim_analyzer.gui.widgets.metric_card_ctk import MetricCard
from laser_trim_analyzer.gui.widgets.file_analysis_widget import FileAnalysisWidget
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
from laser_trim_analyzer.gui.widgets.simple_chart import SimpleChartWidget
from laser_trim_analyzer.gui.widgets.chart_placeholder import (
    ChartPlaceholder,
    create_chart_placeholder,
    create_loading_placeholder,
    create_error_placeholder,
    create_action_placeholder
)
from laser_trim_analyzer.gui.widgets.alert_banner import AlertBanner, AlertStack
from laser_trim_analyzer.gui.widgets.analysis_display import AnalysisDisplayWidget
from laser_trim_analyzer.gui.widgets.plot_viewer import PlotViewerWidget
from laser_trim_analyzer.gui.widgets.animated_widgets import (
    AnimatedProgressBar,
    FadeInFrame,
    SlideInFrame,
    AnimatedButton,
    LoadingSpinner,
    AnimatedNotification,
    AccessibilityHelper
)
from laser_trim_analyzer.gui.widgets.batch_results_widget_ctk import BatchResultsWidget
from laser_trim_analyzer.gui.widgets.track_viewer import IndividualTrackViewer
from laser_trim_analyzer.gui.widgets.progress_widgets_ctk import (
    ProgressDialog,
    BatchProgressDialog,
    SimpleProgressBar,
    ProgressIndicator
)

import customtkinter as ctk


def add_mousewheel_support(widget, canvas=None):
    """
    Add mouse wheel scrolling support to a widget.
    
    Args:
        widget: The widget to add scrolling to (supports Canvas, Treeview, Listbox)
        canvas: Optional canvas widget if the widget is inside a canvas (for scrollable frames)
        
    Note:
        CTkComboBox is not supported because it uses tkinter.Menu for dropdowns,
        which run in their own event loop and cannot be customized with mousewheel events.
    """
    # CTkComboBox is not supported - skip it
    if isinstance(widget, ctk.CTkComboBox):
        return  # Cannot add mousewheel support to Menu-based dropdowns
        
    else:
        # Original implementation for scrollable widgets
        def _on_mousewheel(event):
            # Determine which widget to scroll
            scroll_widget = canvas if canvas else widget
            
            # Different widgets have different scroll methods
            if hasattr(scroll_widget, 'yview_scroll'):
                scroll_widget.yview_scroll(int(-1*(event.delta/120)), "units")
            elif hasattr(scroll_widget, 'yview'):
                # Fallback for other scrollable widgets
                try:
                    scroll_widget.yview('scroll', int(-1*(event.delta/120)), 'units')
                except:
                    pass
        
        # Bind mouse wheel directly to the widget only
        widget.bind("<MouseWheel>", _on_mousewheel)


__all__ = [
    "StatCard",
    "FileDropZone",
    "StatusBar",
    "MetricCard",
    "FileAnalysisWidget",
    "ChartWidget",
    "SimpleChartWidget",
    "ChartPlaceholder",
    "create_chart_placeholder",
    "create_loading_placeholder",
    "create_error_placeholder",
    "create_action_placeholder",
    "AlertBanner",
    "AlertStack",
    "AnalysisDisplayWidget",
    "PlotViewerWidget",
    "AnimatedProgressBar",
    "FadeInFrame",
    "SlideInFrame",
    "AnimatedButton",
    "LoadingSpinner",
    "AnimatedNotification",
    "AccessibilityHelper",
    "BatchResultsWidget",
    "IndividualTrackViewer",
    "ProgressDialog",
    "BatchProgressDialog",
    "SimpleProgressBar",
    "ProgressIndicator",
    "add_mousewheel_support"
]