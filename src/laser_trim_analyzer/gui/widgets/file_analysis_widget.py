"""
FileAnalysisWidget for QA Dashboard

A comprehensive widget for displaying file analysis status with expandable track details.
Perfect for showing real-time analysis progress and results.
"""

import customtkinter as ctk
from tkinter import ttk, font
from typing import Dict, List, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TrackDetailWidget(ctk.CTkFrame):
    """Widget for displaying individual track analysis details."""

    def __init__(self, parent, track_id: str, track_data: dict, **kwargs):
        super().__init__(parent, **kwargs)

        self.track_id = track_id
        self.track_data = track_data

        # Colors
        self.colors = {
            'pass': '#27ae60',
            'fail': '#e74c3c',
            'warning': '#f39c12',
            'bg_light': '#f8f9fa',
            'border': '#dee2e6'
        }

        self._setup_ui()

    def _setup_ui(self):
        """Set up track detail UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        
        # Track header
        header_frame = ctk.CTkFrame(self)
        header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        header_frame.grid_columnconfigure(0, weight=1)

        track_label = ctk.CTkLabel(header_frame, text=f"Track {self.track_id}",
                                   font=ctk.CTkFont(size=12, weight="bold"))
        track_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        # Status badge
        status = self.track_data.get('status', 'Unknown')
        status_color = self._get_status_color(status)
        status_label = ctk.CTkLabel(header_frame, text=status.upper(),
                                    font=ctk.CTkFont(size=10, weight="bold"),
                                    text_color=status_color)
        status_label.grid(row=0, column=1, sticky="e", padx=10, pady=5)

        # Metrics grid
        metrics_frame = ctk.CTkFrame(self)
        metrics_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        metrics_frame.grid_columnconfigure([0, 1], weight=1)

        # Key metrics
        metrics = [
            ('Sigma Gradient', self.track_data.get('sigma_gradient', 'N/A'),
             self.track_data.get('sigma_pass', False)),
            ('Linearity', self.track_data.get('linearity_error', 'N/A'),
             self.track_data.get('linearity_pass', False)),
            ('Risk', self.track_data.get('risk_category', 'N/A'), None),
            ('Status', status, None)
        ]

        for i, (label, value, pass_status) in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            metric_frame = ctk.CTkFrame(metrics_frame)
            metric_frame.grid(row=row, column=col, sticky="ew", padx=5, pady=2)
            metric_frame.grid_columnconfigure(1, weight=1)

            label_widget = ctk.CTkLabel(metric_frame, text=f"{label}:",
                                        font=ctk.CTkFont(size=10))
            label_widget.grid(row=0, column=0, sticky="w", padx=5, pady=2)

            # Format value
            if isinstance(value, float):
                value_text = f"{value:.4f}"
            elif isinstance(value, bool):
                value_text = "PASS" if value else "FAIL"
            else:
                value_text = str(value)

            # Color based on pass/fail
            if pass_status is not None:
                color = self.colors['pass'] if pass_status else self.colors['fail']
            else:
                color = None

            value_widget = ctk.CTkLabel(metric_frame, text=value_text,
                                        font=ctk.CTkFont(size=10, weight="bold"))
            if color:
                value_widget.configure(text_color=color)
            value_widget.grid(row=0, column=1, sticky="e", padx=5, pady=2)

    def _get_status_color(self, status: str) -> str:
        """Get color for status."""
        status_lower = status.lower()
        if 'pass' in status_lower:
            return self.colors['pass']
        elif 'fail' in status_lower:
            return self.colors['fail']
        elif 'warning' in status_lower:
            return self.colors['warning']
        else:
            return '#7f8c8d'


class FileAnalysisWidget(ctk.CTkFrame):
    """
    Comprehensive file analysis display widget.

    Features:
    - File information display
    - Progress bar for ongoing analysis
    - Status indicator
    - Expandable track details
    - Action buttons
    """

    def __init__(self, parent, file_data: Optional[dict] = None,
                 on_view_plot: Optional[Callable] = None,
                 on_export: Optional[Callable] = None,
                 on_details: Optional[Callable] = None,
                 **kwargs):
        """
        Initialize FileAnalysisWidget.

        Args:
            parent: Parent widget
            file_data: Dictionary with file analysis data
            on_view_plot: Callback for view plot button
            on_export: Callback for export button
            on_details: Callback for details button
        """
        super().__init__(parent, **kwargs)

        self.file_data = file_data or {}
        self.on_view_plot = on_view_plot
        self.on_export = on_export
        self.on_details = on_details
        self.expanded = False
        self.track_widgets = []

        # Colors
        self.colors = {
            'pass': '#27ae60',
            'fail': '#e74c3c',
            'warning': '#f39c12',
            'processing': '#3498db',
            'bg_light': '#f8f9fa',
            'text_dark': '#2c3e50',
            'text_light': '#7f8c8d',
            'border': '#dee2e6'
        }

        self._setup_ui()
        if self.file_data:
            self.update_data(self.file_data)

    def _setup_ui(self):
        """Set up the widget UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        
        # Main container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.main_container.grid_columnconfigure(0, weight=1)

        # Header section
        self._create_header()

        # Progress section
        self._create_progress()

        # Action buttons
        self._create_action_section()

        # Expandable track details container
        self.details_container = ctk.CTkFrame(self)
        # Don't grid initially

    def _create_header(self):
        """Create header section with file info."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.header_frame.grid_columnconfigure(0, weight=1)

        # Left side - File info
        info_frame = ctk.CTkFrame(self.header_frame)
        info_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        info_frame.grid_columnconfigure(0, weight=1)

        # Filename
        self.filename_label = ctk.CTkLabel(info_frame, text="No file selected",
                                           font=ctk.CTkFont(size=14, weight="bold"))
        self.filename_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        # Model and Serial
        details_frame = ctk.CTkFrame(info_frame)
        details_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        details_frame.grid_columnconfigure([0, 1, 2], weight=1)

        self.model_label = ctk.CTkLabel(details_frame, text="Model: -",
                                        font=ctk.CTkFont(size=11))
        self.model_label.grid(row=0, column=0, sticky="w", padx=5)

        self.serial_label = ctk.CTkLabel(details_frame, text="Serial: -",
                                         font=ctk.CTkFont(size=11))
        self.serial_label.grid(row=0, column=1, sticky="w", padx=5)

        self.timestamp_label = ctk.CTkLabel(details_frame, text="",
                                            font=ctk.CTkFont(size=11))
        self.timestamp_label.grid(row=0, column=2, sticky="w", padx=5)

        # Right side - Status
        status_frame = ctk.CTkFrame(self.header_frame)
        status_frame.grid(row=0, column=1, sticky="e", padx=5, pady=5)

        self.status_label = ctk.CTkLabel(status_frame, text="PENDING",
                                         font=ctk.CTkFont(size=12, weight="bold"))
        self.status_label.grid(row=0, column=0, padx=10, pady=5)

        # Expand/collapse button for multi-track files
        self.expand_button = ctk.CTkButton(status_frame, text="▼",
                                           width=30, height=30, 
                                           command=self._toggle_expand)
        # Don't grid initially

    def _create_progress(self):
        """Create progress bar section."""
        self.progress_frame = ctk.CTkFrame(self.main_container)
        # Don't grid initially

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, width=300)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        self.progress_label = ctk.CTkLabel(self.progress_frame,
                                           text="Analyzing...",
                                           font=ctk.CTkFont(size=11))
        self.progress_label.grid(row=1, column=0, padx=10, pady=(0, 5))

    def _create_action_section(self):
        """Create action buttons section."""
        self.action_frame = ctk.CTkFrame(self.main_container)
        # Don't grid initially

        # View Plot button
        self.plot_button = ctk.CTkButton(self.action_frame, text="View Plot",
                                         command=self._on_view_plot, width=80)
        self.plot_button.grid(row=0, column=0, padx=5, pady=5)

        # Export button
        self.export_button = ctk.CTkButton(self.action_frame, text="Export",
                                           command=self._on_export, width=80)
        self.export_button.grid(row=0, column=1, padx=5, pady=5)

        # Details button
        self.details_button = ctk.CTkButton(self.action_frame, text="Details",
                                            command=self._on_details, width=80)
        self.details_button.grid(row=0, column=2, padx=5, pady=5)

    def update_data(self, file_data: dict):
        """Update widget with new file data."""
        self.file_data = file_data

        # Update labels
        filename = file_data.get('filename', 'Unknown')
        self.filename_label.configure(text=filename)

        model = file_data.get('model', 'Unknown')
        self.model_label.configure(text=f"Model: {model}")

        serial = file_data.get('serial', 'Unknown')
        self.serial_label.configure(text=f"Serial: {serial}")

        # Update timestamp if available
        if 'timestamp' in file_data:
            timestamp = file_data['timestamp']
            if isinstance(timestamp, datetime):
                time_str = timestamp.strftime('%Y-%m-%d %H:%M')
                self.timestamp_label.configure(text=time_str)

        # Update status
        status = file_data.get('status', 'Unknown')
        self._update_status(status)

        # Handle multi-track files
        if file_data.get('has_multi_tracks', False) and 'tracks' in file_data:
            self.expand_button.grid(row=0, column=1, padx=5, pady=5)

        # Show/hide progress or actions based on status
        if status.lower() == 'processing':
            self.progress_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
            self.action_frame.grid_forget()
            # Start indeterminate progress
            self.progress_bar.set(0.5)  # CTk doesn't have start() method
        else:
            self.progress_frame.grid_forget()
            self.action_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

            # Enable/disable buttons based on available data
            has_plot = file_data.get('plot_path') is not None
            plot_state = "normal" if has_plot else "disabled"
            self.plot_button.configure(state=plot_state)

    def _update_status(self, status: str):
        """Update status display."""
        status_upper = status.upper()
        self.status_label.configure(text=status_upper)

        # Color based on status
        if 'PASS' in status_upper:
            color = self.colors['pass']
        elif 'FAIL' in status_upper:
            color = self.colors['fail']
        elif 'WARNING' in status_upper:
            color = self.colors['warning']
        elif 'PROCESSING' in status_upper:
            color = self.colors['processing']
        else:
            color = self.colors['text_light']

        self.status_label.configure(text_color=color)

    def _toggle_expand(self):
        """Toggle expanded state for track details."""
        if self.expanded:
            self._collapse()
        else:
            self._expand()

    def _expand(self):
        """Expand to show track details."""
        self.expanded = True
        self.expand_button.configure(text="▲")

        # Clear existing track widgets
        for widget in self.track_widgets:
            widget.destroy()
        self.track_widgets.clear()

        # Grid details container
        self.details_container.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.details_container.grid_columnconfigure(0, weight=1)

        # Add track details
        tracks = self.file_data.get('tracks', {})
        for i, (track_id, track_data) in enumerate(tracks.items()):
            track_widget = TrackDetailWidget(self.details_container,
                                             track_id, track_data)
            track_widget.grid(row=i, column=0, sticky="ew", padx=5, pady=2)
            self.track_widgets.append(track_widget)

    def _collapse(self):
        """Collapse track details."""
        self.expanded = False
        self.expand_button.configure(text="▼")
        self.details_container.grid_forget()

    def _on_view_plot(self):
        """Handle view plot button click."""
        try:
            if self.on_view_plot:
                self.on_view_plot(self.file_data)
        except Exception as e:
            logger.error(f"Error in view plot callback: {e}")

    def _on_export(self):
        """Handle export button click."""
        try:
            if self.on_export:
                self.on_export(self.file_data)
        except Exception as e:
            logger.error(f"Error in export callback: {e}")

    def _on_details(self):
        """Handle details button click."""
        try:
            if self.on_details:
                self.on_details(self.file_data)
        except Exception as e:
            logger.error(f"Error in details callback: {e}")

    def set_progress(self, value: float, text: str = ""):
        """Set progress for determinate mode."""
        self.progress_bar.set(value)
        if text:
            self.progress_label.configure(text=text)

    def clear(self):
        """Clear all data from the widget."""
        self.file_data = {}
        self.filename_label.configure(text="No file selected")
        self.model_label.configure(text="Model: -")
        self.serial_label.configure(text="Serial: -")
        self.timestamp_label.configure(text="")
        self.status_label.configure(text="PENDING")
        
        # Hide optional elements
        self.expand_button.grid_forget()
        self.progress_frame.grid_forget()
        self.action_frame.grid_forget()
        self.details_container.grid_forget()
        
        # Clear track widgets
        for widget in self.track_widgets:
            widget.destroy()
        self.track_widgets.clear()
        self.expanded = False
