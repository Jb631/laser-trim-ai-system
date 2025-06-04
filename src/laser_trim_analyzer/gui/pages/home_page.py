"""
Home Page for Laser Trim Analyzer

Displays dashboard with key metrics and recent activity.
"""

import customtkinter as ctk
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import threading
import time

from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.widgets.metric_card import MetricCard
from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget


class HomePage(BasePage):
    """
    Home page showing dashboard and recent activity.

    Features:
    - Key metrics dashboard
    - Recent activity list
    - Quick action buttons
    - Real-time stats from database
    - Responsive design
    """

    def __init__(self, parent, main_window: Any):
        """Initialize home page."""
        # Initialize stat cards dict before parent init
        self.stat_cards = {}
        self.activity_list = None
        self.trend_chart = None

        super().__init__(parent, main_window)

        # Start background refresh
        self._start_auto_refresh()

    def _create_page(self):
        """Create home page content with responsive design (matching batch processing theme)."""
        # Main scrollable container (matching batch processing theme)
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Create sections in order (matching batch processing pattern)
        self._create_header()
        self._create_stats_section()
        self._create_activity_section()
        self._create_quick_actions_section()

    def _create_header(self):
        """Create header section (matching batch processing theme)."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Quality Analysis Dashboard",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=15)

        # Subtitle with current date
        self.subtitle_label = ctk.CTkLabel(
            self.header_frame,
            text=f"Welcome back! Today is {datetime.now().strftime('%B %d, %Y')}",
            font=ctk.CTkFont(size=12)
        )
        self.subtitle_label.pack(pady=(0, 15))

    def _create_stats_section(self):
        """Create statistics cards section (matching batch processing theme)."""
        self.stats_frame = ctk.CTkFrame(self.main_container)
        self.stats_frame.pack(fill='x', pady=(0, 20))

        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="Today's Performance:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.stats_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Metrics container
        self.stats_metrics_frame = ctk.CTkFrame(self.stats_frame)
        self.stats_metrics_frame.pack(fill='x', padx=15, pady=(0, 15))

        # Create metric cards (matching batch processing layout)
        self.stat_cards['units_tested'] = MetricCard(
            self.stats_metrics_frame,
            title="Units Tested",
            value="0",
            color_scheme="neutral"
        )
        self.stat_cards['units_tested'].pack(side='left', fill='x', expand=True, padx=(10, 5), pady=10)

        self.stat_cards['pass_rate'] = MetricCard(
            self.stats_metrics_frame,
            title="Pass Rate",
            value="0%",
            color_scheme="success"
        )
        self.stat_cards['pass_rate'].pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        self.stat_cards['validation_rate'] = MetricCard(
            self.stats_metrics_frame,
            title="Validation Success",
            value="0%",
            color_scheme="info"
        )
        self.stat_cards['validation_rate'].pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        self.stat_cards['avg_sigma'] = MetricCard(
            self.stats_metrics_frame,
            title="Avg Sigma Gradient",
            value="0.000",
            color_scheme="warning"
        )
        self.stat_cards['avg_sigma'].pack(side='left', fill='x', expand=True, padx=(5, 10), pady=10)

    def _create_activity_section(self):
        """Create recent activity section (matching batch processing theme)."""
        self.activity_frame = ctk.CTkFrame(self.main_container)
        self.activity_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.activity_label = ctk.CTkLabel(
            self.activity_frame,
            text="Recent Activity:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.activity_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Activity list
        self.activity_list = ctk.CTkTextbox(
            self.activity_frame,
            height=200,
            state="disabled"
        )
        self.activity_list.pack(fill='both', expand=True, padx=15, pady=(0, 15))

    def _create_quick_actions_section(self):
        """Create quick actions section (matching batch processing theme)."""
        self.actions_frame = ctk.CTkFrame(self.main_container)
        self.actions_frame.pack(fill='x', pady=(0, 20))

        self.actions_label = ctk.CTkLabel(
            self.actions_frame,
            text="Quick Actions:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.actions_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Actions container
        self.actions_container = ctk.CTkFrame(self.actions_frame)
        self.actions_container.pack(fill='x', padx=15, pady=(0, 15))

        self.new_analysis_button = ctk.CTkButton(
            self.actions_container,
            text="New Analysis",
            command=self._quick_new_analysis,
            width=120,
            height=40
        )
        self.new_analysis_button.pack(side='left', padx=(10, 10), pady=10)

        self.view_reports_button = ctk.CTkButton(
            self.actions_container,
            text="View Reports",
            command=self._view_reports,
            width=120,
            height=40
        )
        self.view_reports_button.pack(side='left', padx=(0, 10), pady=10)

        self.high_risk_button = ctk.CTkButton(
            self.actions_container,
            text="High Risk Units",
            command=self._show_high_risk_details,
            width=120,
            height=40,
            fg_color="orange",
            hover_color="darkorange"
        )
        self.high_risk_button.pack(side='left', padx=(0, 10), pady=10)

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
        """Get today's statistics."""
        if not self.main_window.db_manager:
            return {
                "units_tested": 0,
                "pass_rate": 0.0,
                "validation_rate": 0.0,
                "avg_sigma": 0.0,
                "avg_industry_grade": "--",
                "high_risk_count": 0
            }

        try:
            # Get today's data
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            results = self.main_window.db_manager.get_historical_data(
                start_date=today_start,
                include_tracks=True
            )

            if not results:
                return {
                    "units_tested": 0,
                    "pass_rate": 0.0,
                    "validation_rate": 0.0,
                    "avg_sigma": 0.0,
                    "avg_industry_grade": "--",
                    "high_risk_count": 0
                }

            # Calculate basic metrics
            total_units = len(results)
            passed_units = sum(1 for r in results if r.overall_status.value == "Pass")
            pass_rate = (passed_units / total_units * 100) if total_units > 0 else 0

            # Calculate validation metrics
            validated_units = sum(1 for r in results if hasattr(r, 'overall_validation_status') and 
                                r.overall_validation_status and r.overall_validation_status.value == "Validated")
            validation_rate = (validated_units / total_units * 100) if total_units > 0 else 0

            # Calculate sigma gradient average
            sigma_values = []
            industry_grades = []
            high_risk_count = 0

            for result in results:
                if result.tracks:
                    primary_track = result.tracks[0]
                    if hasattr(primary_track, 'sigma_gradient') and primary_track.sigma_gradient:
                        sigma_values.append(primary_track.sigma_gradient)
                    
                    # Count high risk units
                    if (hasattr(primary_track, 'risk_category') and 
                        primary_track.risk_category and 
                        primary_track.risk_category.value == "High"):
                        high_risk_count += 1
                    
                    # Collect industry grades for averaging
                    if hasattr(result, 'validation_grade') and result.validation_grade:
                        grade = result.validation_grade
                        if grade not in ["Not Validated", "Incomplete"]:
                            industry_grades.append(grade)

            avg_sigma = sum(sigma_values) / len(sigma_values) if sigma_values else 0.0

            # Calculate average industry grade
            if industry_grades:
                grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0, "F": 0}
                avg_grade_value = sum(grade_values.get(g, 0) for g in industry_grades) / len(industry_grades)
                
                if avg_grade_value >= 3.5:
                    avg_industry_grade = "A"
                elif avg_grade_value >= 2.5:
                    avg_industry_grade = "B"
                elif avg_grade_value >= 1.5:
                    avg_industry_grade = "C"
                elif avg_grade_value >= 0.5:
                    avg_industry_grade = "D"
                else:
                    avg_industry_grade = "F"
            else:
                avg_industry_grade = "--"

            return {
                "units_tested": total_units,
                "pass_rate": pass_rate,
                "validation_rate": validation_rate,
                "avg_sigma": avg_sigma,
                "avg_industry_grade": avg_industry_grade,
                "high_risk_count": high_risk_count
            }

        except Exception as e:
            self.logger.error(f"Error getting today's stats: {e}")
            return {
                "units_tested": 0,
                "pass_rate": 0.0,
                "validation_rate": 0.0,
                "avg_sigma": 0.0,
                "avg_industry_grade": "--",
                "high_risk_count": 0
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
        # Check if we have any data at all
        has_data = stats.get('units_tested', 0) > 0 or len(activities) > 0
        
        # Update stat cards - only update the ones that exist
        try:
            if 'units_tested' in self.stat_cards:
                self.stat_cards['units_tested'].update_value(str(stats.get('units_tested', 0)))
            if 'pass_rate' in self.stat_cards:
                self.stat_cards['pass_rate'].update_value(f"{stats.get('pass_rate', 0):.1f}%")
            if 'validation_rate' in self.stat_cards:
                self.stat_cards['validation_rate'].update_value(f"{stats.get('validation_rate', 0):.1f}%")
            if 'avg_sigma' in self.stat_cards:
                self.stat_cards['avg_sigma'].update_value(f"{stats.get('avg_sigma', 0):.4f}")
        except Exception as e:
            self.logger.warning(f"Error updating stat cards: {e}")

        # Update activity list using the text widget
        try:
            if hasattr(self, 'activity_list') and self.activity_list:
                self.activity_list.configure(state="normal")
                self.activity_list.delete('1.0', 'end')
                
                if activities:
                    for activity in activities[-10:]:  # Show last 10 activities
                        timestamp = activity.get('timestamp', 'Unknown time')
                        action = activity.get('action', 'Unknown action')
                        details = activity.get('details', '')
                        
                        activity_text = f"[{timestamp}] {action}"
                        if details:
                            activity_text += f": {details}"
                        activity_text += "\n"
                        
                        self.activity_list.insert('end', activity_text)
                else:
                    if not has_data:
                        # Show empty state guidance
                        empty_state_message = (
                            "No analysis data found.\n\n"
                            "To get started:\n"
                            "• Go to the Analysis tab\n"
                            "• Load Excel files for processing\n"
                            "• Build up a history of analysis results\n"
                            "• Return here to see your dashboard\n\n"
                            "Once you have data, this area will show:\n"
                            "• Recent analysis completions\n"
                            "• Processing status updates\n"
                            "• Quality alerts and notifications"
                        )
                    else:
                        empty_state_message = "No recent activity today"
                    
                    self.activity_list.insert('1.0', empty_state_message)
                
                self.activity_list.configure(state="disabled")
                self.activity_list.see('end')
        except Exception as e:
            self.logger.warning(f"Error updating activity list: {e}")

        self.logger.debug("Home page UI updated successfully")

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

    def _update_home_ui(self):
        """Update the home page UI with current data."""
        try:
            # Throttle updates to prevent excessive database queries
            current_time = time.time()
            if hasattr(self, '_last_update_time'):
                time_since_last_update = current_time - self._last_update_time
                if time_since_last_update < 5.0:  # Minimum 5 seconds between updates
                    return
            
            self._last_update_time = current_time
            
            # Update recent files
            self._update_recent_files()
            
            # Update statistics with timeout
            try:
                self._update_statistics_with_timeout()
            except Exception as e:
                self.logger.warning(f"Statistics update failed: {e}")
                
            # Update productivity metrics
            try:
                self._update_productivity_metrics()
            except Exception as e:
                self.logger.warning(f"Productivity metrics update failed: {e}")
                
            self.logger.debug("Home page UI updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating home UI: {e}")
            # Don't let UI update errors crash the application
    
    def _update_statistics_with_timeout(self):
        """Update statistics with timeout to prevent hanging."""
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def fetch_stats():
            try:
                if self.db_manager:
                    stats = self.db_manager.get_statistics()
                    result_queue.put(('success', stats))
                else:
                    result_queue.put(('error', 'No database connection'))
            except Exception as e:
                result_queue.put(('error', str(e)))
        
        # Run database query in background thread with timeout
        thread = threading.Thread(target=fetch_stats)
        thread.daemon = True
        thread.start()
        thread.join(timeout=2.0)  # 2 second timeout
        
        try:
            result_type, data = result_queue.get_nowait()
            if result_type == 'success':
                self._display_statistics(data)
            else:
                self.logger.warning(f"Statistics fetch failed: {data}")
        except queue.Empty:
            self.logger.warning("Statistics fetch timed out")
            # Use cached data if available
            if hasattr(self, '_cached_stats'):
                self._display_statistics(self._cached_stats)