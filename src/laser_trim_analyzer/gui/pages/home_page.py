"""
Home Page for Laser Trim Analyzer

Displays dashboard with key metrics and recent activity.
"""

import customtkinter as ctk
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import threading
import time

from laser_trim_analyzer.gui.pages.base_page_ctk import BasePage
from laser_trim_analyzer.gui.widgets.metric_card_ctk import MetricCard
from laser_trim_analyzer.core.models import AnalysisStatus


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
        self.is_visible = False  # Track visibility state
        self._last_refresh_time = None  # Track last refresh time
        self._pending_refresh = False  # Track if refresh is needed
        
        # Performance cache to reduce database queries
        self._cache_timeout = 15  # Cache data for 15 seconds
        self._cache = {
            'today_stats': None,
            'trend_data': None,
            'recent_activity': None,
            'last_update': 0
        }

        super().__init__(parent, main_window)

        # Subscribe to analysis complete events during initialization
        # This ensures we catch events even when page is not visible
        if hasattr(self.main_window, 'subscribe_to_event'):
            self._analysis_complete_handler = self._on_analysis_complete
            self.main_window.subscribe_to_event('analysis_complete', self._analysis_complete_handler)
            self.logger.info("Subscribed to analysis_complete event during initialization")

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

        # Metrics container with transparent background
        self.stats_metrics_frame = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        self.stats_metrics_frame.pack(fill='x', padx=15, pady=(0, 15))

        # Create metric cards (matching batch processing layout)
        self.stat_cards['units_tested'] = MetricCard(
            self.stats_metrics_frame,
            title="Units Tested",
            value="0",
            color_scheme="neutral",
            show_sparkline=False  # Disable sparkline to avoid white box
        )
        self.stat_cards['units_tested'].pack(side='left', fill='x', expand=True, padx=(10, 5), pady=10)

        self.stat_cards['pass_rate'] = MetricCard(
            self.stats_metrics_frame,
            title="Pass Rate",
            value="0%",
            color_scheme="success",
            show_sparkline=False  # Disable sparkline to avoid white box
        )
        self.stat_cards['pass_rate'].pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        self.stat_cards['validation_rate'] = MetricCard(
            self.stats_metrics_frame,
            title="Validation Success",
            value="0%",
            color_scheme="info",
            show_sparkline=False  # Disable sparkline to avoid white box
        )
        self.stat_cards['validation_rate'].pack(side='left', fill='x', expand=True, padx=(5, 5), pady=10)

        self.stat_cards['avg_sigma'] = MetricCard(
            self.stats_metrics_frame,
            title="Avg Sigma Gradient",
            value="0.000",
            color_scheme="warning",
            show_sparkline=False  # Disable sparkline to avoid white box
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

        # Actions container with transparent background
        self.actions_container = ctk.CTkFrame(self.actions_frame, fg_color="transparent")
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
        self.logger.info("Home page refresh called")
        
        # Track last refresh time
        self._last_refresh_time = time.time()
        
        if not hasattr(self.main_window, 'db_manager') or not self.main_window.db_manager:
            self.logger.warning("No database manager available for refresh")
            # Update UI with empty data
            self._update_ui({}, [], [])
            return

        # Run database queries in background thread
        self.logger.info("Starting background refresh thread")
        threading.Thread(target=self._refresh_data_background, daemon=True).start()

    def _refresh_data_background(self):
        """Background thread to refresh data with caching."""
        try:
            current_time = time.time()
            
            # Check if cache is still valid (but always refresh if pending)
            cache_valid = (current_time - self._cache['last_update']) < self._cache_timeout
            if cache_valid and not self._pending_refresh:
                # Use cached data
                self.logger.info("Using cached data for performance")
                try:
                    if self.winfo_exists():
                        self.after(0, self._update_ui, 
                                 self._cache['today_stats'] or {},
                                 self._cache['trend_data'] or [],
                                 self._cache['recent_activity'] or [])
                except Exception:
                    pass  # Widget was destroyed
                return
            elif self._pending_refresh:
                self.logger.info("Cache bypass: pending refresh requested")
                self._pending_refresh = False
            
            # Cache expired, fetch fresh data
            self.logger.info("Cache expired, fetching fresh data")
            
            # Get today's stats
            today_stats = self._get_today_stats()

            # Get 7-day trend
            trend_data = self._get_trend_data()

            # Get recent activity
            recent_activity = self._get_recent_activity()
            
            # Update cache
            self._cache.update({
                'today_stats': today_stats,
                'trend_data': trend_data,
                'recent_activity': recent_activity,
                'last_update': current_time
            })

            # Update UI in main thread safely
            try:
                if self.winfo_exists():
                    self.after(0, self._update_ui, today_stats, trend_data, recent_activity)
            except Exception:
                pass  # Widget was destroyed

        except Exception as e:
            self.logger.error(f"Error refreshing dashboard: {e}")

    def _get_today_stats(self) -> Dict[str, Any]:
        """Get today's statistics."""
        if not hasattr(self.main_window, 'db_manager') or not self.main_window.db_manager:
            return {
                "units_tested": 0,
                "pass_rate": 0.0,
                "validation_rate": 0.0,
                "avg_sigma": 0.0,
                "avg_industry_grade": "--",
                "high_risk_count": 0
            }

        try:
            # Get today's data - use local time for "today" but convert to UTC for database query
            # This ensures "Today's Performance" shows data from today in the user's timezone
            local_today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Calculate UTC offset
            from datetime import timezone
            local_now = datetime.now()
            utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
            utc_offset_hours = round((local_now - utc_now).total_seconds() / 3600)
            
            # Convert local midnight to UTC for database query
            utc_today_start = local_today_start - timedelta(hours=utc_offset_hours)
            
            self.logger.info(f"Local today starts at: {local_today_start}")
            self.logger.info(f"UTC offset: {utc_offset_hours} hours")
            self.logger.info(f"Querying for today's data starting from UTC: {utc_today_start}")
            
            results = self.main_window.db_manager.get_historical_data(
                start_date=utc_today_start,
                include_tracks=True
            )
            self.logger.info(f"Found {len(results)} results for today")

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
            passed_units = 0
            for r in results:
                try:
                    # Handle both enum and string status values
                    if hasattr(r.overall_status, 'value'):
                        status = r.overall_status.value
                    else:
                        status = str(r.overall_status)
                    if status == "Pass":
                        passed_units += 1
                except Exception:
                    continue
            pass_rate = (passed_units / total_units * 100) if total_units > 0 else 0

            # Calculate validation metrics
            # For now, consider all passing units as validated since validation status isn't stored in DB
            validated_units = passed_units
            validation_rate = pass_rate

            # Calculate sigma gradient average
            sigma_values = []
            industry_grades = []
            high_risk_count = 0

            for result in results:
                try:
                    if result.tracks and len(result.tracks) > 0:
                        # Get primary track - tracks is a list from DB, not a dict
                        primary_track = result.tracks[0]
                        
                        if primary_track:
                            # Get sigma gradient directly from track
                            if hasattr(primary_track, 'sigma_gradient') and primary_track.sigma_gradient is not None:
                                sigma_values.append(primary_track.sigma_gradient)
                        
                            # Count high risk units
                            if (hasattr(primary_track, 'risk_category') and 
                                primary_track.risk_category and 
                                hasattr(primary_track.risk_category, 'value') and
                                primary_track.risk_category.value == "High"):
                                high_risk_count += 1
                except (IndexError, AttributeError) as e:
                    self.logger.warning(f"Error accessing track data: {e}")
                    continue

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
        """Get 7-day trend data with single optimized query."""
        try:
            # Use days_back parameter instead of start_date to avoid datetime format issues
            # This is more reliable with SQLite
            all_results = self.main_window.db_manager.get_historical_data(
                days_back=7,
                include_tracks=False  # Don't need track data for trend
            )
            
            # Group by date
            daily_data = {}
            for result in all_results:
                date_key = result.timestamp.date()
                if date_key not in daily_data:
                    daily_data[date_key] = {'total': 0, 'passed': 0}
                
                daily_data[date_key]['total'] += 1
                
                # Handle both enum and string status values
                try:
                    if hasattr(result.overall_status, 'value'):
                        status = result.overall_status.value
                    else:
                        status = str(result.overall_status)
                    if status == 'Pass':
                        daily_data[date_key]['passed'] += 1
                except Exception:
                    continue
            
            # Build trend data for last 7 days
            trend_data = []
            for i in range(6, -1, -1):  # 6 days ago to today
                date = datetime.now() - timedelta(days=i)
                date_key = date.date()
                
                if date_key in daily_data:
                    data = daily_data[date_key]
                    pass_rate = (data['passed'] / data['total'] * 100) if data['total'] > 0 else 0
                    trend_data.append({
                        'date': date.strftime('%m/%d'),
                        'pass_rate': pass_rate
                    })
                else:
                    # No data for this day
                    trend_data.append({
                        'date': date.strftime('%m/%d'),
                        'pass_rate': 0
                    })

            return trend_data

        except Exception as e:
            self.logger.error(f"Error getting trend data: {e}")
            return []

    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent activity from database."""
        try:
            # Log database status
            self.logger.info(f"Getting recent activity - DB manager available: {self.main_window.db_manager is not None}")
            
            if not self.main_window.db_manager:
                self.logger.warning("No database manager available")
                return []
            
            # Get recent analyses from last 24 hours primarily, then expand if needed
            now = datetime.now()
            activities = []
            
            # First try to get today's activities - use UTC naive datetime
            today_start_utc = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            results = self.main_window.db_manager.get_historical_data(
                start_date=today_start_utc,
                limit=50,
                include_tracks=False
            )
            
            # If no activities today, get from last 7 days
            if not results:
                self.logger.info("No activities today, checking last 7 days")
                results = self.main_window.db_manager.get_historical_data(
                    days_back=7,
                    limit=20,
                    include_tracks=False
                )
            
            self.logger.info(f"Found {len(results)} recent activities from database")

            for result in results:
                # Determine status tag - handle both enum and string values
                try:
                    if hasattr(result.overall_status, 'value'):
                        status = result.overall_status.value
                    else:
                        status = str(result.overall_status)
                except Exception:
                    status = 'Unknown'
                
                if status == 'Pass':
                    tag = 'pass'
                elif status == 'Fail':
                    tag = 'fail'
                else:
                    tag = 'warning'

                # Get model and serial directly from DB result
                model = result.model if hasattr(result, 'model') else 'Unknown'
                serial = result.serial if hasattr(result, 'serial') else 'Unknown'
                
                # Format timestamp with date info for clarity - convert from UTC to local
                if hasattr(result, 'timestamp') and result.timestamp:
                    # Assume database timestamp is in UTC (naive), convert to local time for display
                    from datetime import timezone
                    import pytz
                    
                    # Make the naive UTC timestamp aware
                    timestamp_utc = pytz.UTC.localize(result.timestamp)
                    # Convert to local timezone
                    timestamp_local = timestamp_utc.astimezone()
                    
                    # If today, show just time
                    if timestamp_local.date() == now.date():
                        time_str = timestamp_local.strftime('%H:%M:%S')
                    # If yesterday
                    elif timestamp_local.date() == (now - timedelta(days=1)).date():
                        time_str = f"Yesterday {timestamp_local.strftime('%H:%M')}"
                    # Otherwise show date
                    else:
                        time_str = timestamp_local.strftime('%m/%d %H:%M')
                else:
                    time_str = 'Unknown'
                
                # Determine action based on context or always use 'Analysis Complete' for now
                # In future, could track different types of activities
                action = 'Analysis Complete'
                
                activities.append({
                    'time': time_str,
                    'action': action,
                    'details': f"{model} - {serial}",
                    'status': status,
                    'tag': tag,
                    'timestamp': result.timestamp if hasattr(result, 'timestamp') else None
                })
            
            # Sort by timestamp (newest first)
            activities.sort(key=lambda x: x.get('timestamp') or datetime.min, reverse=True)
            
            return activities

        except Exception as e:
            self.logger.error(f"Error getting recent activity: {e}")
            return []

    def _update_ui(self, stats: Dict[str, Any], trend_data: List[Dict[str, Any]],
                   activities: List[Dict[str, Any]]):
        """Update UI with refreshed data."""
        self.logger.info(f"_update_ui called with {len(activities)} activities")
        
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
                    # Show most recent 10 activities (already sorted newest first)
                    for activity in activities[:10]:
                        timestamp = activity.get('time', 'Unknown time')
                        action = activity.get('action', 'Unknown action')
                        details = activity.get('details', '')
                        status = activity.get('status', 'Unknown')
                        
                        activity_text = f"[{timestamp}] {action}"
                        if details:
                            activity_text += f": {details}"
                        if status and status != 'Unknown':
                            activity_text += f" ({status})"
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
        # Refresh every 30 seconds when visible for more responsive updates
        if hasattr(self, 'is_visible') and self.is_visible:
            self.logger.debug("Auto-refresh triggered")
            self.refresh()
        self.after(30000, self._start_auto_refresh)  # 30 seconds

    def _quick_new_analysis(self):
        """Quick action to start new analysis."""
        try:
            if hasattr(self.main_window, '_show_page'):
                self.main_window._show_page('single_file')
                # Trigger file browser
                if 'single_file' in self.main_window.pages and hasattr(self.main_window.pages['single_file'], '_browse_file'):
                    # Call the browse method after a short delay to ensure page is fully loaded
                    self.after(100, self.main_window.pages['single_file']._browse_file)
        except Exception as e:
            self.logger.error(f"Error navigating to single file page: {e}")

    def _view_reports(self):
        """Quick action to view reports."""
        try:
            if hasattr(self.main_window, '_show_page'):
                self.main_window._show_page('historical')
        except Exception as e:
            self.logger.error(f"Error navigating to historical page: {e}")

    def _show_high_risk_details(self):
        """Show details of high risk units."""
        try:
            # Navigate to historical page with high risk filter applied
            if hasattr(self.main_window, '_show_page'):
                self.main_window._show_page('historical', risk_category='High')
        except Exception as e:
            self.logger.error(f"Error navigating to historical page with filter: {e}")

    def on_show(self):
        """Called when page is shown."""
        self.logger.info("Home page on_show called")
        # Mark page as visible
        self.is_visible = True
        
        # Check if we need to refresh due to missed updates
        current_time = time.time()
        if self._pending_refresh or (self._last_refresh_time and current_time - self._last_refresh_time > 5):
            self.logger.info("Refreshing home page data on show (pending updates or stale data)")
            self._pending_refresh = False
            self.refresh()
        else:
            # Always refresh when shown to ensure latest data
            self.refresh()
    
    def on_hide(self):
        """Called when page is hidden."""
        # Mark page as not visible
        self.is_visible = False
        # Note: We keep the event subscription active to catch events while hidden
    
    def _on_analysis_complete(self, data):
        """Handle analysis complete event."""
        try:
            self.logger.info(f"Received analysis complete event: {data.get('type', 'unknown')}")
            
            # Clear cache to force fresh data on next refresh
            self._cache['last_update'] = 0
            self.logger.info("Cache cleared to force fresh data after analysis")
            
            # If page is visible, update immediately
            if self.is_visible:
                # Force immediate UI update with temporary message
                self._show_immediate_feedback(data)
                
                # Schedule database refresh with forced cache refresh
                self.after(500, self._force_refresh)  # 0.5 second delay to ensure DB is updated
            else:
                # Page is not visible, mark that we need to refresh when shown
                self.logger.info("Page not visible - marking refresh as pending")
                self._pending_refresh = True
        except Exception as e:
            self.logger.error(f"Error handling analysis complete event: {e}")
    
    def _force_refresh(self):
        """Force a complete refresh, ignoring cache."""
        self._cache['last_update'] = 0  # Clear cache
        self.refresh()
    
    def _show_immediate_feedback(self, data):
        """Show immediate feedback in the UI while waiting for database refresh."""
        # Update stat cards if we have results data
        if data.get('type') == 'batch_complete' and data.get('results'):
            try:
                results = data.get('results', {})
                # Calculate immediate stats from results
                total_count = len(results)
                passed_count = sum(1 for r in results.values() 
                                 if r and hasattr(r, 'overall_status') and 
                                 r.overall_status == AnalysisStatus.PASS)
                pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0
                
                # Update stat cards immediately
                if 'units_tested' in self.stat_cards:
                    # Just show the batch count for now, full refresh will update totals
                    self.stat_cards['units_tested'].update_value(str(total_count))
                if 'pass_rate' in self.stat_cards and total_count > 0:
                    # Update pass rate
                    self.stat_cards['pass_rate'].update_value(f"{pass_rate:.1f}%")
            except Exception as e:
                self.logger.debug(f"Could not update stats immediately: {e}")
        elif data.get('result'):
            # Handle single file result
            try:
                result = data.get('result')
                if result and hasattr(result, 'overall_status'):
                    # Show immediate feedback for single file
                    self.logger.info(f"Single file analysis complete: {getattr(result, 'model', 'Unknown')}")
                    # Don't update stat cards here - let the full refresh handle it
            except Exception as e:
                self.logger.debug(f"Could not process single file result: {e}")
        
        # Extract better data from the analysis complete event
        try:
            if 'model' not in data and hasattr(data.get('result', None), 'model'):
                data['model'] = data['result'].model
            if 'serial' not in data and hasattr(data.get('result', None), 'serial'):
                data['serial'] = data['result'].serial
            if 'status' not in data and hasattr(data.get('result', None), 'overall_status'):
                status = data['result'].overall_status
                data['status'] = status.value if hasattr(status, 'value') else str(status)
        except Exception as e:
            self.logger.debug(f"Could not extract additional data: {e}")
        
        # Show immediate feedback in activity list
        if hasattr(self, 'activity_list') and self.activity_list:
            try:
                self.activity_list.configure(state="normal")
                
                # Add temporary message based on event type
                now = datetime.now().strftime('%H:%M:%S')
                
                if data.get('type') == 'batch_complete':
                    # Handle batch processing completion
                    successful_count = data.get('successful_count', 0)
                    total_count = data.get('total_count', 0)
                    temp_message = f"[{now}] Batch processing complete: {successful_count}/{total_count} files processed\n"
                else:
                    # Handle single file analysis
                    model = data.get('model', 'Unknown')
                    serial = data.get('serial', 'Unknown')
                    status = data.get('status', 'Unknown')
                    temp_message = f"[{now}] Analysis complete: {model} - {serial} ({status})\n"
                
                current_content = self.activity_list.get('1.0', 'end').strip()
                
                # Add new message to top
                self.activity_list.delete('1.0', 'end')
                self.activity_list.insert('1.0', temp_message)
                
                # Keep only the most recent messages
                if current_content and current_content != "No recent activity today":
                    lines = current_content.split('\n')
                    # Keep up to 9 previous entries (10 total with new one)
                    for line in lines[:9]:
                        if line.strip():
                            self.activity_list.insert('end', line + '\n')
                
                self.activity_list.configure(state="disabled")
                self.activity_list.see('1.0')  # Scroll to top
            except Exception as e:
                self.logger.error(f"Error showing temporary activity: {e}")
    
    def _update_statistics_with_timeout(self):
        """Update statistics with timeout to prevent hanging."""
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def fetch_stats():
            try:
                if hasattr(self.main_window, 'db_manager') and self.main_window.db_manager:
                    stats = self.main_window.db_manager.get_statistics()
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
    
    def cleanup(self):
        """Clean up resources when page is destroyed."""
        # Unsubscribe from events
        if hasattr(self.main_window, 'unsubscribe_from_event') and hasattr(self, '_analysis_complete_handler'):
            self.main_window.unsubscribe_from_event('analysis_complete', self._analysis_complete_handler)
            self.logger.info("Unsubscribed from analysis_complete event during cleanup")
        
        # Stop any background threads if needed
        super().cleanup() if hasattr(super(), 'cleanup') else None