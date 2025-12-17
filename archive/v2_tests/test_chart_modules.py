"""
Tests for chart widget modules (Phase 4 file splitting).

These tests verify:
1. Chart module imports work correctly
2. Chart mixin classes have expected methods
3. ChartWidget combines all mixins properly
4. Module exports are correct

Note: These tests avoid instantiating GUI components to prevent GUI windows opening.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestChartModuleImports:
    """Test that chart module imports work correctly."""

    def test_import_chart_widget(self):
        """Test ChartWidget can be imported from charts package."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidget
        assert ChartWidget is not None

    def test_import_chart_widget_base(self):
        """Test ChartWidgetBase can be imported."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidgetBase
        assert ChartWidgetBase is not None

    def test_import_basic_chart_mixin(self):
        """Test BasicChartMixin can be imported."""
        from laser_trim_analyzer.gui.widgets.charts import BasicChartMixin
        assert BasicChartMixin is not None

    def test_import_quality_chart_mixin(self):
        """Test QualityChartMixin can be imported."""
        from laser_trim_analyzer.gui.widgets.charts import QualityChartMixin
        assert QualityChartMixin is not None

    def test_import_analytics_chart_mixin(self):
        """Test AnalyticsChartMixin can be imported."""
        from laser_trim_analyzer.gui.widgets.charts import AnalyticsChartMixin
        assert AnalyticsChartMixin is not None

    def test_backward_compatible_import(self):
        """Test ChartWidget can be imported from original location."""
        from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
        assert ChartWidget is not None


class TestChartWidgetInheritance:
    """Test ChartWidget class inheritance structure."""

    def test_chart_widget_inherits_base(self):
        """Test ChartWidget inherits from ChartWidgetBase."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidget, ChartWidgetBase
        assert issubclass(ChartWidget, ChartWidgetBase)

    def test_chart_widget_inherits_basic_mixin(self):
        """Test ChartWidget inherits from BasicChartMixin."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidget, BasicChartMixin
        assert issubclass(ChartWidget, BasicChartMixin)

    def test_chart_widget_inherits_quality_mixin(self):
        """Test ChartWidget inherits from QualityChartMixin."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidget, QualityChartMixin
        assert issubclass(ChartWidget, QualityChartMixin)

    def test_chart_widget_inherits_analytics_mixin(self):
        """Test ChartWidget inherits from AnalyticsChartMixin."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidget, AnalyticsChartMixin
        assert issubclass(ChartWidget, AnalyticsChartMixin)


class TestChartWidgetBaseMethods:
    """Test ChartWidgetBase has expected methods."""

    def test_has_setup_ui_method(self):
        """Test ChartWidgetBase has _setup_ui method."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidgetBase
        assert hasattr(ChartWidgetBase, '_setup_ui')

    def test_has_apply_theme_to_axes(self):
        """Test ChartWidgetBase has _apply_theme_to_axes method."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidgetBase
        assert hasattr(ChartWidgetBase, '_apply_theme_to_axes')

    def test_has_show_placeholder(self):
        """Test ChartWidgetBase has show_placeholder method."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidgetBase
        assert hasattr(ChartWidgetBase, 'show_placeholder')

    def test_has_show_loading(self):
        """Test ChartWidgetBase has show_loading method."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidgetBase
        assert hasattr(ChartWidgetBase, 'show_loading')

    def test_has_show_error(self):
        """Test ChartWidgetBase has show_error method."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidgetBase
        assert hasattr(ChartWidgetBase, 'show_error')

    def test_has_refresh_theme(self):
        """Test ChartWidgetBase has refresh_theme method."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidgetBase
        assert hasattr(ChartWidgetBase, 'refresh_theme')

    def test_has_clear_chart(self):
        """Test ChartWidgetBase has clear_chart method."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidgetBase
        assert hasattr(ChartWidgetBase, 'clear_chart')

    def test_has_export_chart(self):
        """Test ChartWidgetBase has _export_chart method."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidgetBase
        assert hasattr(ChartWidgetBase, '_export_chart')

    def test_has_update_chart(self):
        """Test ChartWidgetBase has update_chart method."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidgetBase
        assert hasattr(ChartWidgetBase, 'update_chart')

    def test_has_add_reference_lines(self):
        """Test ChartWidgetBase has add_reference_lines method."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidgetBase
        assert hasattr(ChartWidgetBase, 'add_reference_lines')

    def test_has_add_chart_annotation(self):
        """Test ChartWidgetBase has add_chart_annotation method."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidgetBase
        assert hasattr(ChartWidgetBase, 'add_chart_annotation')


class TestBasicChartMixinMethods:
    """Test BasicChartMixin has expected chart methods."""

    def test_has_plot_line(self):
        """Test BasicChartMixin has plot_line method."""
        from laser_trim_analyzer.gui.widgets.charts import BasicChartMixin
        assert hasattr(BasicChartMixin, 'plot_line')

    def test_has_plot_bar(self):
        """Test BasicChartMixin has plot_bar method."""
        from laser_trim_analyzer.gui.widgets.charts import BasicChartMixin
        assert hasattr(BasicChartMixin, 'plot_bar')

    def test_has_plot_scatter(self):
        """Test BasicChartMixin has plot_scatter method."""
        from laser_trim_analyzer.gui.widgets.charts import BasicChartMixin
        assert hasattr(BasicChartMixin, 'plot_scatter')

    def test_has_plot_histogram(self):
        """Test BasicChartMixin has plot_histogram method."""
        from laser_trim_analyzer.gui.widgets.charts import BasicChartMixin
        assert hasattr(BasicChartMixin, 'plot_histogram')

    def test_has_plot_box(self):
        """Test BasicChartMixin has plot_box method."""
        from laser_trim_analyzer.gui.widgets.charts import BasicChartMixin
        assert hasattr(BasicChartMixin, 'plot_box')

    def test_has_plot_pie(self):
        """Test BasicChartMixin has plot_pie method."""
        from laser_trim_analyzer.gui.widgets.charts import BasicChartMixin
        assert hasattr(BasicChartMixin, 'plot_pie')

    def test_has_update_chart_data(self):
        """Test BasicChartMixin has update_chart_data method."""
        from laser_trim_analyzer.gui.widgets.charts import BasicChartMixin
        assert hasattr(BasicChartMixin, 'update_chart_data')


class TestQualityChartMixinMethods:
    """Test QualityChartMixin has expected quality chart methods."""

    def test_has_plot_quality_dashboard(self):
        """Test QualityChartMixin has plot_quality_dashboard method."""
        from laser_trim_analyzer.gui.widgets.charts import QualityChartMixin
        assert hasattr(QualityChartMixin, 'plot_quality_dashboard')

    def test_has_plot_gauge(self):
        """Test QualityChartMixin has plot_gauge method."""
        from laser_trim_analyzer.gui.widgets.charts import QualityChartMixin
        assert hasattr(QualityChartMixin, 'plot_gauge')

    def test_has_plot_quality_dashboard_cards(self):
        """Test QualityChartMixin has plot_quality_dashboard_cards method."""
        from laser_trim_analyzer.gui.widgets.charts import QualityChartMixin
        assert hasattr(QualityChartMixin, 'plot_quality_dashboard_cards')


class TestAnalyticsChartMixinMethods:
    """Test AnalyticsChartMixin has expected analytics methods."""

    def test_has_plot_enhanced_control_chart(self):
        """Test AnalyticsChartMixin has plot_enhanced_control_chart method."""
        from laser_trim_analyzer.gui.widgets.charts import AnalyticsChartMixin
        assert hasattr(AnalyticsChartMixin, 'plot_enhanced_control_chart')

    def test_has_plot_process_capability_histogram(self):
        """Test AnalyticsChartMixin has plot_process_capability_histogram method."""
        from laser_trim_analyzer.gui.widgets.charts import AnalyticsChartMixin
        assert hasattr(AnalyticsChartMixin, 'plot_process_capability_histogram')

    def test_has_plot_early_warning_system(self):
        """Test AnalyticsChartMixin has plot_early_warning_system method."""
        from laser_trim_analyzer.gui.widgets.charts import AnalyticsChartMixin
        assert hasattr(AnalyticsChartMixin, 'plot_early_warning_system')

    def test_has_plot_failure_pattern_analysis(self):
        """Test AnalyticsChartMixin has plot_failure_pattern_analysis method."""
        from laser_trim_analyzer.gui.widgets.charts import AnalyticsChartMixin
        assert hasattr(AnalyticsChartMixin, 'plot_failure_pattern_analysis')

    def test_has_plot_performance_scorecard(self):
        """Test AnalyticsChartMixin has plot_performance_scorecard method."""
        from laser_trim_analyzer.gui.widgets.charts import AnalyticsChartMixin
        assert hasattr(AnalyticsChartMixin, 'plot_performance_scorecard')


class TestChartWidgetCombinedMethods:
    """Test ChartWidget has all methods from all mixins."""

    def test_has_all_basic_methods(self):
        """Test ChartWidget has all BasicChartMixin methods."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidget
        basic_methods = ['plot_line', 'plot_bar', 'plot_scatter', 'plot_histogram',
                        'plot_box', 'plot_pie', 'update_chart_data']
        for method in basic_methods:
            assert hasattr(ChartWidget, method), f"Missing method: {method}"

    def test_has_all_quality_methods(self):
        """Test ChartWidget has all QualityChartMixin methods."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidget
        quality_methods = ['plot_quality_dashboard', 'plot_gauge', 'plot_quality_dashboard_cards']
        for method in quality_methods:
            assert hasattr(ChartWidget, method), f"Missing method: {method}"

    def test_has_all_analytics_methods(self):
        """Test ChartWidget has all AnalyticsChartMixin methods."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidget
        analytics_methods = ['plot_enhanced_control_chart', 'plot_process_capability_histogram',
                            'plot_early_warning_system', 'plot_failure_pattern_analysis',
                            'plot_performance_scorecard']
        for method in analytics_methods:
            assert hasattr(ChartWidget, method), f"Missing method: {method}"

    def test_has_all_base_methods(self):
        """Test ChartWidget has all ChartWidgetBase methods."""
        from laser_trim_analyzer.gui.widgets.charts import ChartWidget
        base_methods = ['show_placeholder', 'show_loading', 'show_error',
                       'clear_chart', 'refresh_theme', 'update_chart']
        for method in base_methods:
            assert hasattr(ChartWidget, method), f"Missing method: {method}"


class TestChartModuleExports:
    """Test chart module __all__ exports."""

    def test_all_exports_defined(self):
        """Test __all__ is defined in charts module."""
        from laser_trim_analyzer.gui.widgets import charts
        assert hasattr(charts, '__all__')

    def test_chart_widget_in_exports(self):
        """Test ChartWidget is in __all__."""
        from laser_trim_analyzer.gui.widgets import charts
        assert 'ChartWidget' in charts.__all__

    def test_chart_widget_base_in_exports(self):
        """Test ChartWidgetBase is in __all__."""
        from laser_trim_analyzer.gui.widgets import charts
        assert 'ChartWidgetBase' in charts.__all__

    def test_basic_chart_mixin_in_exports(self):
        """Test BasicChartMixin is in __all__."""
        from laser_trim_analyzer.gui.widgets import charts
        assert 'BasicChartMixin' in charts.__all__

    def test_quality_chart_mixin_in_exports(self):
        """Test QualityChartMixin is in __all__."""
        from laser_trim_analyzer.gui.widgets import charts
        assert 'QualityChartMixin' in charts.__all__

    def test_analytics_chart_mixin_in_exports(self):
        """Test AnalyticsChartMixin is in __all__."""
        from laser_trim_analyzer.gui.widgets import charts
        assert 'AnalyticsChartMixin' in charts.__all__


class TestQAColorsAttribute:
    """Test ChartWidgetBase has QA colors for manufacturing visualization."""

    def test_has_qa_colors_attribute(self):
        """Test ChartWidgetBase has qa_colors definition."""
        from laser_trim_analyzer.gui.widgets.charts.base import ChartWidgetBase
        # Check the class definition includes qa_colors setup
        import inspect
        source = inspect.getsource(ChartWidgetBase.__init__)
        assert 'qa_colors' in source

    def test_qa_colors_include_pass_fail(self):
        """Test qa_colors include pass/fail status colors."""
        from laser_trim_analyzer.gui.widgets.charts.base import ChartWidgetBase
        import inspect
        source = inspect.getsource(ChartWidgetBase.__init__)
        assert "'pass'" in source
        assert "'fail'" in source

    def test_qa_colors_include_control_limits(self):
        """Test qa_colors include control chart colors."""
        from laser_trim_analyzer.gui.widgets.charts.base import ChartWidgetBase
        import inspect
        source = inspect.getsource(ChartWidgetBase.__init__)
        assert "'control_center'" in source or "'control_limits'" in source
