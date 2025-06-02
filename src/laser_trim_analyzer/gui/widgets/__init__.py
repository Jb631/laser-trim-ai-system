"""GUI widgets for Laser Trim Analyzer."""

from laser_trim_analyzer.gui.widgets.stat_card import StatCard
from laser_trim_analyzer.gui.widgets.file_drop_zone import FileDropZone
from laser_trim_analyzer.gui.widgets.progress_widget import ProgressWidget
from laser_trim_analyzer.gui.widgets.status_bar import StatusBar
from laser_trim_analyzer.gui.widgets.metric_card import MetricCard
from laser_trim_analyzer.gui.widgets.file_analysis_widget import FileAnalysisWidget
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
from laser_trim_analyzer.gui.widgets.alert_banner import AlertBanner, AlertStack


def add_mousewheel_support(widget, canvas=None):
    """
    Add mouse wheel scrolling support to a widget.
    
    Args:
        widget: The widget to add scrolling to (usually a Canvas, Treeview, or Listbox)
        canvas: Optional canvas widget if the widget is inside a canvas (for scrollable frames)
    """
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
    
    # Bind mouse wheel to the widget
    widget.bind("<MouseWheel>", _on_mousewheel)
    
    # Also bind when mouse enters/leaves the widget area
    def _bind_to_mousewheel(event):
        widget.bind_all("<MouseWheel>", _on_mousewheel)
    
    def _unbind_from_mousewheel(event):
        widget.unbind_all("<MouseWheel>")
    
    widget.bind('<Enter>', _bind_to_mousewheel)
    widget.bind('<Leave>', _unbind_from_mousewheel)


__all__ = [
    "StatCard",
    "FileDropZone",
    "ProgressWidget",
    "StatusBar",
    "MetricCard",
    "FileAnalysisWidget",
    "ChartWidget",
    "AlertBanner",
    "AlertStack",
    "add_mousewheel_support"
]