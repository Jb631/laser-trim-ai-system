"""
Home Page for Laser Trim Analyzer

Displays dashboard with key metrics and recent activity.
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import threading

from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.widgets.stat_card import StatCard
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget


class HomePage(BasePage):
    """
    Home page showing dashboard and recent activity.

    Features:
    - Key metrics dashboard
    - Recent activity list
    - Quick action buttons
    - Real-time stats from database
    """

    def __init__(self, parent: ttk.Frame, main_window: Any):
        """Initialize home page."""
        # Initialize stat cards dict before parent init
        self.stat_cards = {}
        self.activity_tree = None
        self.trend_chart = None

        super().__init__(parent, main_window)

        # Start background refresh
        self._start_auto_refresh()

    def _create_page(self):
        """Create home page content."""
        # Main container with padding
        container = ttk.Frame(self, style='TFrame')
        container.pack(fill='both', expand=True, padx=20, pady=20)

        # Title
        title_label = ttk.Label(
            container,
            text="Quality Analysis Dashboard",
            style='Title.TLabel'
        )
        title_label.pack(anchor='w', pady=(0, 10))

        # Subtitle with current date
        subtitle_label = ttk.Label(
            container,
            text=f"Welcome back! Today is {datetime.now().strftime('%B %d, %Y')}",
            font=('Segoe UI', 11),
            foreground=self.colors['text_secondary']
        )
        subtitle_label.pack(anchor='w', pady=(0, 20))

        # Create main content area with two columns
        content_frame = ttk.Frame(container)
        content_frame.pack(fill='both', expand=True)

        # Left column - Stats and charts
        left_column = ttk.Frame(content_frame)
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Right column - Recent activity
        right_column = ttk.Frame(content_frame)
        right_column.pack(side='right', fill='both', expand=True, padx=(10, 0))

        # Create sections
        self._create_stats_section(left_column)
        self._create_trend_section(left_column)
        self._create_quick_actions_section(left_column)
        self._create_activity_section(right_column)

    def _create_stats_section(self, parent):
        """Create statistics cards section."""
        # Stats frame
        stats_frame = ttk.LabelFrame(
            parent,
            text="Today's Performance",
            padding=15
        )
        stats_frame.pack(fill='x', pady=(0, 20))

        # Create 2x2 grid of stat cards
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill='x')

        # Configure grid
        for i in range(2):
            stats_grid.columnconfigure(i, weight=1, minsize=200)

        # Create stat cards
        self.stat_cards['units_tested'] = StatCard(
            stats_grid,
            title="Units Tested",
            value=0,
            unit="",
            color_scheme="default"
        )
        self.stat_cards['units_tested'].grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.stat_cards['pass_rate'] = StatCard(
            stats_grid,
            title="Pass Rate",
            value=0.0,
            unit="%",
            color_scheme="success"
        )
        self.stat_cards['pass_rate'].grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.stat_cards['avg_sigma'] = StatCard(
            stats_grid,
            title="Avg Sigma Gradient",
            value=0.0,
            unit="",
            color_scheme="warning"
        )
        self.stat_cards['avg_sigma'].grid(row=1, column=0, padx=5, pady=5, sticky='ew')

        self.stat_cards['high_risk'] = StatCard(
            stats_grid,
            title="High Risk Units",
            value=0,
            unit="",
            color_scheme="danger"
        )
        self.stat_cards['high_risk'].grid(row=1, column=1, padx=5, pady=5, sticky='ew')

    def _create_trend_section(self, parent):
        """Create trend chart section."""
        # Trend frame
        trend_frame = ttk.LabelFrame(
            parent,
            text="7-Day Pass Rate Trend",
            padding=15
        )
        trend_frame.pack(fill='both', expand=True, pady=(0, 20))

        # Create chart widget
        self.trend_chart = ChartWidget(
            trend_frame,
            chart_type='line',
            title="",
            figsize=(6, 3)
        )
        self.trend_chart.pack(fill='both', expand=True)

    def _create_quick_actions_section(self, parent):
        """Create quick actions section."""
        # Actions frame
        actions_frame = ttk.LabelFrame(
            parent,
            text="Quick Actions",
            padding=15
        )
        actions_frame.pack(fill='x')

        # Button container
        btn_container = ttk.Frame(actions_frame)
        btn_container.pack(fill='x')

        # Quick action buttons
        ttk.Button(
            btn_container,
            text="📁 New Analysis",
            style='Primary.TButton',
            command=self._quick_new_analysis
        ).pack(side='left', padx=(0, 10))

        ttk.Button(
            btn_container,
            text="📊 View Reports",
            command=self._view_reports
        ).pack(side='left', padx=(0, 10))

        ttk.Button(
            btn_container,
            text="🔄 Refresh Data",
            command=self.refresh
        ).pack(side='left')

    def _create_activity_section(self, parent):
        """Create recent activity section."""
        # Activity frame
        activity_frame = ttk.LabelFrame(
            parent,
            text="Recent Activity",
            padding=15
        )
        activity_frame.pack(fill='both', expand=True)

        # Create treeview for activity
        columns = ('Time', 'Action', 'Details', 'Status')
        self.activity_tree = ttk.Treeview(
            activity_frame,
            columns=columns,
            show='tree headings',
            height=15
        )

        # Configure columns
        self.activity_tree.column('#0', width=0, stretch=False)
        self.activity_tree.column('Time', width=100)
        self.activity_tree.column('Action', width=150)
        self.activity_tree.column('Details', width=200)
        self.activity_tree.column('Status', width=80)

        # Set headings
        for col in columns:
            self.activity_tree.heading(col, text=col)

        # Create scrollbar
        scrollbar = ttk.Scrollbar(
            activity_frame,
            orient='vertical',
            command=self.activity_tree.yview
        )
        self.activity_tree.configure(yscrollcommand=scrollbar.set)

        # Pack
        self.activity_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Add tags for status colors
        self.activity_tree.tag_configure('pass', foreground=self.colors['success'])
        self.activity_tree.tag_configure('fail', foreground=self.colors['danger'])
        self.activity_tree.tag_configure('warning', foreground=self.colors['warning'])

    def refresh(self):
        """Refresh dashboard data from database."""
        if not self.db_manager:
            return

        # Run database queries in background thread
        threading.Thread(target=self._refresh_data_background, daemon=True).start()

    def _refresh_data_background(self):
        """Background thread to refresh data."""
        try:
            # Get today's stats
            today_stats = self._get_today_stats()

            # Get 7-day trend
            trend_data = self._get_trend_data()

            # Get recent activity
            recent_activity = self._get_recent_activity()

            # Update UI in main thread
            self.after(0, self._update_ui, today_stats, trend_data, recent_activity)

        except Exception as e:
            self.logger.error(f"Error refreshing dashboard: {e}")

    def _get_today_stats(self) -> Dict[str, Any]:
        """Get today's statistics from database."""
        try:
            # Query today's results
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            # Get today's analyses
            results = self.db_manager.get_historical_data(
                start_date=today_start,
                include_tracks=True
            )

            # Calculate stats
            total_units = len(results)
            passed_units = sum(1 for r in results if r.overall_status.value == 'Pass')
            pass_rate = (passed_units / total_units * 100) if total_units > 0 else 0

            # Get track-level stats
            all_tracks = []
            for result in results:
                all_tracks.extend(result.tracks)

            # Calculate average sigma
            sigma_values = [t.sigma_gradient for t in all_tracks if t.sigma_gradient is not None]
            avg_sigma = sum(sigma_values) / len(sigma_values) if sigma_values else 0

            # Count high risk
            high_risk = sum(1 for t in all_tracks if t.risk_category and t.risk_category.value == 'High')

            return {
                'units_tested': total_units,
                'pass_rate': pass_rate,
                'avg_sigma': avg_sigma,
                'high_risk': high_risk
            }

        except Exception as e:
            self.logger.error(f"Error getting today's stats: {e}")
            return {
                'units_tested': 0,
                'pass_rate': 0,
                'avg_sigma': 0,
                'high_risk': 0
            }

    def _get_trend_data(self) -> List[Dict[str, Any]]:
        """Get 7-day trend data."""
        try:
            trend_data = []

            for i in range(7):
                date = datetime.now() - timedelta(days=i)
                date_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
                date_end = date_start + timedelta(days=1)

                # Get data for this day
                results = self.db_manager.get_historical_data(
                    start_date=date_start,
                    end_date=date_end
                )

                # Calculate pass rate
                total = len(results)
                passed = sum(1 for r in results if r.overall_status.value == 'Pass')
                pass_rate = (passed / total * 100) if total > 0 else None

                if pass_rate is not None:
                    trend_data.append({
                        'date': date.strftime('%m/%d'),
                        'pass_rate': pass_rate
                    })

            # Reverse to show oldest to newest
            trend_data.reverse()

            return trend_data

        except Exception as e:
            self.logger.error(f"Error getting trend data: {e}")
            return []

    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent activity from database."""
        try:
            # Get recent analyses
            results = self.db_manager.get_historical_data(
                days_back=1,
                limit=20,
                include_tracks=False
            )

            activities = []
            for result in results:
                # Determine status tag
                status = result.overall_status.value
                if status == 'Pass':
                    tag = 'pass'
                elif status == 'Fail':
                    tag = 'fail'
                else:
                    tag = 'warning'

                activities.append({
                    'time': result.timestamp.strftime('%H:%M:%S'),
                    'action': 'Analysis Complete',
                    'details': f"{result.model} - {result.serial}",
                    'status': status,
                    'tag': tag
                })

            return activities

        except Exception as e:
            self.logger.error(f"Error getting recent activity: {e}")
            return []

    def _update_ui(self, stats: Dict[str, Any], trend_data: List[Dict[str, Any]],
                   activities: List[Dict[str, Any]]):
        """Update UI with refreshed data."""
        # Update stat cards
        self.stat_cards['units_tested'].update_value(stats['units_tested'])
        self.stat_cards['pass_rate'].update_value(stats['pass_rate'])
        self.stat_cards['avg_sigma'].update_value(stats['avg_sigma'])
        self.stat_cards['high_risk'].update_value(stats['high_risk'])

        # Update trend chart
        if trend_data:
            dates = [d['date'] for d in trend_data]
            values = [d['pass_rate'] for d in trend_data]

            self.trend_chart.clear_chart()
            self.trend_chart.plot_line(
                dates, values,
                label="Pass Rate",
                color='primary',
                marker='o',
                xlabel="Date",
                ylabel="Pass Rate (%)"
            )
            self.trend_chart.add_threshold_lines({'Target': 95})

        # Update activity list
        # Clear existing items
        for item in self.activity_tree.get_children():
            self.activity_tree.delete(item)

        # Add new items
        for activity in activities:
            self.activity_tree.insert(
                '',
                'end',
                values=(
                    activity['time'],
                    activity['action'],
                    activity['details'],
                    activity['status']
                ),
                tags=(activity['tag'],)
            )

    def _start_auto_refresh(self):
        """Start automatic refresh timer."""
        # Refresh every 30 seconds when visible
        if self.is_visible:
            self.refresh()
        self.after(30000, self._start_auto_refresh)  # 30 seconds

    def _quick_new_analysis(self):
        """Quick action to start new analysis."""
        self.main_window._show_page('analysis')
        # Trigger file browser
        if hasattr(self.main_window.pages['analysis'], 'browse_files'):
            self.main_window.pages['analysis'].browse_files()

    def _view_reports(self):
        """Quick action to view reports."""
        self.main_window._show_page('historical')

    def _show_high_risk_details(self):
        """Show details of high risk units."""
        # Could open a dialog or navigate to filtered view
        self.main_window._show_page('historical')
        # TODO: Add filtering for high risk units

    def on_show(self):
        """Called when page is shown."""
        # Refresh data when page is shown
        self.refresh()