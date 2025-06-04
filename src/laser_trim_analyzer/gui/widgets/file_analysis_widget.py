"""
FileAnalysisWidget for QA Dashboard

A comprehensive widget for displaying file analysis status with expandable track details.
Perfect for showing real-time analysis progress and results.
"""

import tkinter as tk
from tkinter import ttk, font
from typing import Dict, List, Optional, Callable
from datetime import datetime


class TrackDetailWidget(ttk.Frame):
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
        # Configure the frame (remove unsupported parameters)
        # CustomTkinter doesn't support relief, borderwidth, padding

        # Track header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill='x', pady=(0, 5))

        track_label = ttk.Label(header_frame, text=f"Track {self.track_id}",
                                font=('Segoe UI', 10, 'bold'))
        track_label.pack(side='left')

        # Status badge
        status = self.track_data.get('status', 'Unknown')
        status_color = self._get_status_color(status)
        status_label = ttk.Label(header_frame, text=status.upper(),
                                 font=('Segoe UI', 9, 'bold'),
                                 foreground=status_color)
        status_label.pack(side='right')

        # Metrics grid
        metrics_frame = ttk.Frame(self)
        metrics_frame.pack(fill='x')

        # Key metrics
        metrics = [
            ('Sigma Gradient', self.track_data.get('sigma_gradient', 'N/A'),
             self.track_data.get('sigma_pass', False)),
            ('Linearity', self.track_data.get('linearity_pass', 'N/A'),
             self.track_data.get('linearity_pass', False)),
            ('Risk', self.track_data.get('risk_category', 'N/A'), None)
        ]

        for i, (label, value, pass_status) in enumerate(metrics):
            metric_frame = ttk.Frame(metrics_frame)
            metric_frame.grid(row=i // 2, column=i % 2, sticky='ew', padx=5, pady=2)

            ttk.Label(metric_frame, text=f"{label}:",
                      font=('Segoe UI', 9), foreground='#7f8c8d').pack(side='left')

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
                color = '#2c3e50'

            ttk.Label(metric_frame, text=value_text,
                      font=('Segoe UI', 9, 'bold'),
                      foreground=color).pack(side='left', padx=(5, 0))

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


class FileAnalysisWidget(ttk.Frame):
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
        # Configure the frame (remove unsupported parameters)
        # CustomTkinter doesn't support relief, borderwidth, padding

        # Main container
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill='both', expand=True)

        # Header section
        self._create_header()

        # Progress section
        self._create_progress()

        # Action buttons
        self._create_action_section()

        # Expandable track details container
        self.details_container = ttk.Frame(self)
        # Don't pack initially

    def _create_header(self):
        """Create header section with file info."""
        self.header_frame = ttk.Frame(self.main_container)
        self.header_frame.pack(fill='x', padx=10, pady=(10, 5))

        # Left side - File info
        info_frame = ttk.Frame(self.header_frame)
        info_frame.pack(side='left', fill='both', expand=True)

        # Filename
        self.filename_label = ttk.Label(info_frame, text="No file selected",
                                        font=('Segoe UI', 11, 'bold'),
                                        foreground=self.colors['text_dark'])
        self.filename_label.pack(anchor='w')

        # Model and Serial
        details_frame = ttk.Frame(info_frame)
        details_frame.pack(anchor='w', pady=(2, 0))

        self.model_label = ttk.Label(details_frame, text="Model: -",
                                     font=('Segoe UI', 9),
                                     foreground=self.colors['text_light'])
        self.model_label.pack(side='left', padx=(0, 15))

        self.serial_label = ttk.Label(details_frame, text="Serial: -",
                                      font=('Segoe UI', 9),
                                      foreground=self.colors['text_light'])
        self.serial_label.pack(side='left', padx=(0, 15))

        self.timestamp_label = ttk.Label(details_frame, text="",
                                         font=('Segoe UI', 9),
                                         foreground=self.colors['text_light'])
        self.timestamp_label.pack(side='left')

        # Right side - Status
        status_frame = ttk.Frame(self.header_frame)
        status_frame.pack(side='right', padx=(10, 0))

        self.status_label = ttk.Label(status_frame, text="PENDING",
                                      font=('Segoe UI', 10, 'bold'),
                                      foreground=self.colors['text_light'])
        self.status_label.pack()

        # Expand/collapse button for multi-track files
        self.expand_button = ttk.Button(status_frame, text="▼",
                                        width=3, command=self._toggle_expand)
        # Don't pack initially

    def _create_progress(self):
        """Create progress bar section."""
        self.progress_frame = ttk.Frame(self.main_container)
        # Don't pack initially

        self.progress_bar = ttk.Progressbar(self.progress_frame,
                                            mode='indeterminate',
                                            length=300)
        self.progress_bar.pack(fill='x')

        self.progress_label = ttk.Label(self.progress_frame,
                                        text="Analyzing...",
                                        font=('Segoe UI', 9),
                                        foreground=self.colors['text_light'])
        self.progress_label.pack(pady=(5, 0))

    def _create_action_section(self):
        """Create action buttons section."""
        self.action_frame = ttk.Frame(self.main_container)
        # Don't pack initially

        # View Plot button
        self.plot_button = ttk.Button(self.action_frame, text="View Plot",
                                      command=self._on_view_plot)
        self.plot_button.pack(side='left', padx=(0, 5))

        # Export button
        self.export_button = ttk.Button(self.action_frame, text="Export",
                                        command=self._on_export)
        self.export_button.pack(side='left', padx=(0, 5))

        # Details button
        self.details_button = ttk.Button(self.action_frame, text="Details",
                                         command=self._on_details)
        self.details_button.pack(side='left')

    def update_data(self, file_data: dict):
        """Update widget with new file data."""
        self.file_data = file_data

        # Update labels
        filename = file_data.get('filename', 'Unknown')
        self.filename_label.config(text=filename)

        model = file_data.get('model', 'Unknown')
        self.model_label.config(text=f"Model: {model}")

        serial = file_data.get('serial', 'Unknown')
        self.serial_label.config(text=f"Serial: {serial}")

        # Update timestamp if available
        if 'timestamp' in file_data:
            timestamp = file_data['timestamp']
            if isinstance(timestamp, datetime):
                time_str = timestamp.strftime('%Y-%m-%d %H:%M')
                self.timestamp_label.config(text=time_str)

        # Update status
        status = file_data.get('status', 'Unknown')
        self._update_status(status)

        # Handle multi-track files
        if file_data.get('has_multi_tracks', False) and 'tracks' in file_data:
            self.expand_button.pack(pady=(5, 0))

        # Show/hide progress or actions based on status
        if status.lower() == 'processing':
            self.progress_frame.pack(fill='x', pady=(10, 0))
            self.action_frame.pack_forget()
            self.progress_bar.start(10)
        else:
            self.progress_frame.pack_forget()
            self.progress_bar.stop()
            self.action_frame.pack(fill='x', pady=(10, 0))

            # Enable/disable buttons based on available data
            has_plot = file_data.get('plot_path') is not None
            self.plot_button.config(state='normal' if has_plot else 'disabled')

    def _update_status(self, status: str):
        """Update status display."""
        status_upper = status.upper()
        self.status_label.config(text=status_upper)

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

        self.status_label.config(foreground=color)

    def _toggle_expand(self):
        """Toggle expanded state for track details."""
        if self.expanded:
            self._collapse()
        else:
            self._expand()

    def _expand(self):
        """Expand to show track details."""
        self.expanded = True
        self.expand_button.config(text="▲")

        # Clear existing track widgets
        for widget in self.track_widgets:
            widget.destroy()
        self.track_widgets.clear()

        # Pack details container
        self.details_container.pack(fill='x', pady=(10, 0))

        # Add track details
        tracks = self.file_data.get('tracks', {})
        for track_id, track_data in tracks.items():
            track_widget = TrackDetailWidget(self.details_container,
                                             track_id, track_data)
            track_widget.pack(fill='x', pady=(0, 5))
            self.track_widgets.append(track_widget)

    def _collapse(self):
        """Collapse track details."""
        self.expanded = False
        self.expand_button.config(text="▼")
        self.details_container.pack_forget()

    def _on_view_plot(self):
        """Handle view plot button click."""
        if self.on_view_plot:
            self.on_view_plot(self.file_data)

    def _on_export(self):
        """Handle export button click."""
        if self.on_export:
            self.on_export(self.file_data)

    def _on_details(self):
        """Handle details button click."""
        if self.on_details:
            self.on_details(self.file_data)

    def set_progress(self, value: float, text: str = ""):
        """Set progress for determinate mode."""
        self.progress_bar.config(mode='determinate', value=value)
        if text:
            self.progress_label.config(text=text)


# Example usage and testing
if __name__ == "__main__":
    root = tk.Tk()
    root.title("FileAnalysisWidget Demo")
    root.geometry("600x700")
    root.configure(bg='#f0f0f0')

    # Create sample widgets
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill='both', expand=True)

    # Processing file
    processing_data = {
        'filename': '8340_A12345_2024.xlsx',
        'model': '8340',
        'serial': 'A12345',
        'status': 'Processing',
        'timestamp': datetime.now()
    }

    widget1 = FileAnalysisWidget(
        frame,
        processing_data,
        on_view_plot=lambda d: print(f"View plot: {d['filename']}"),
        on_export=lambda d: print(f"Export: {d['filename']}"),
        on_details=lambda d: print(f"Details: {d['filename']}")
    )
    widget1.pack(fill='x', pady=10)

    # Completed single-track file
    completed_data = {
        'filename': '8555_B67890_2024.xlsx',
        'model': '8555',
        'serial': 'B67890',
        'status': 'Pass',
        'timestamp': datetime.now(),
        'plot_path': '/path/to/plot.png',
        'sigma_gradient': 0.0234,
        'sigma_pass': True,
        'linearity_pass': True,
        'risk_category': 'Low'
    }

    widget2 = FileAnalysisWidget(frame, completed_data)
    widget2.pack(fill='x', pady=10)

    # Multi-track file with mixed results
    multi_track_data = {
        'filename': '6845_C11111_2024.xlsx',
        'model': '6845',
        'serial': 'C11111',
        'status': 'Warning',
        'timestamp': datetime.now(),
        'has_multi_tracks': True,
        'plot_path': '/path/to/plot.png',
        'tracks': {
            'TRK1': {
                'status': 'Pass',
                'sigma_gradient': 0.0156,
                'sigma_pass': True,
                'linearity_pass': True,
                'risk_category': 'Low'
            },
            'TRK2': {
                'status': 'Fail',
                'sigma_gradient': 0.0891,
                'sigma_pass': False,
                'linearity_pass': True,
                'risk_category': 'High'
            }
        }
    }

    widget3 = FileAnalysisWidget(frame, multi_track_data)
    widget3.pack(fill='x', pady=10)

    root.mainloop()