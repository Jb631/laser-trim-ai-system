"""
Individual Track Viewer Widget

Provides detailed view of individual tracks within multi-track units
with real-time data visualization and interaction.
"""

import tkinter as tk
import customtkinter as ctk
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

logger = logging.getLogger(__name__)


class IndividualTrackViewer(ctk.CTkFrame):
    """Widget for viewing individual track details in multi-track units."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.current_track_data = None
        self.tracks_data = {}
        self.selected_track = None
        
        self._create_viewer()
        
    def _create_viewer(self):
        """Create the track viewer interface."""
        # Header with track selector
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.pack(fill='x', padx=5, pady=5)
        
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Individual Track Viewer",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.title_label.pack(side='left', padx=10, pady=5)
        
        # Track selector
        self.track_selector_frame = ctk.CTkFrame(self.header_frame)
        self.track_selector_frame.pack(side='right', padx=10, pady=5)
        
        self.track_selector_label = ctk.CTkLabel(
            self.track_selector_frame,
            text="Select Track:",
            font=ctk.CTkFont(size=12)
        )
        self.track_selector_label.pack(side='left', padx=(0, 5))
        
        self.track_selector = ctk.CTkComboBox(
            self.track_selector_frame,
            values=["No tracks available"],
            command=self._on_track_selected,
            width=150,
            state="readonly"
        )
        self.track_selector.pack(side='left')
        self.track_selector.set("No tracks available")
        
        # Create tabview for different track views
        self.track_tabview = ctk.CTkTabview(self)
        self.track_tabview.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Add tabs
        self.track_tabview.add("Details")
        self.track_tabview.add("Error Profile")
        self.track_tabview.add("Statistics")
        self.track_tabview.add("Raw Data")
        
        # Create tab contents
        self._create_details_tab()
        self._create_error_profile_tab()
        self._create_statistics_tab()
        self._create_raw_data_tab()
        
    def _create_details_tab(self):
        """Create details tab content."""
        details_frame = self.track_tabview.tab("Details")
        
        # Scrollable details container
        self.details_container = ctk.CTkScrollableFrame(details_frame)
        self.details_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Track info section
        self.track_info_frame = ctk.CTkFrame(self.details_container)
        self.track_info_frame.pack(fill='x', padx=5, pady=5)
        
        info_title = ctk.CTkLabel(
            self.track_info_frame,
            text="Track Information",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        info_title.pack(anchor='w', padx=10, pady=(10, 5))
        
        self.info_labels = {}
        info_items = [
            ("track_id", "Track ID"),
            ("position", "Position"),
            ("serial", "Serial Number"),
            ("timestamp", "Test Date/Time"),
            ("status", "Status"),
            ("validation", "Validation")
        ]
        
        for key, label in info_items:
            frame = ctk.CTkFrame(self.track_info_frame)
            frame.pack(fill='x', padx=10, pady=2)
            
            label_widget = ctk.CTkLabel(
                frame,
                text=f"{label}:",
                font=ctk.CTkFont(size=11),
                width=120,
                anchor='w'
            )
            label_widget.pack(side='left', padx=(0, 10))
            
            value_widget = ctk.CTkLabel(
                frame,
                text="--",
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor='w'
            )
            value_widget.pack(side='left', fill='x', expand=True)
            
            self.info_labels[key] = value_widget
        
        # Key metrics section
        self.metrics_frame = ctk.CTkFrame(self.details_container)
        self.metrics_frame.pack(fill='x', padx=5, pady=(10, 5))
        
        metrics_title = ctk.CTkLabel(
            self.metrics_frame,
            text="Key Metrics",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        metrics_title.pack(anchor='w', padx=10, pady=(10, 5))
        
        self.metric_labels = {}
        metric_items = [
            ("sigma_gradient", "Sigma Gradient", "{:.6f}"),
            ("sigma_spec", "Sigma Spec", "{:.6f}"),
            ("sigma_margin", "Sigma Margin", "{:.2%}"),
            ("linearity_error", "Linearity Error", "{:.3f}%"),
            ("linearity_spec", "Linearity Spec", "{:.3f}%"),
            ("resistance_change", "Resistance Change", "{:.2f}%"),
            ("trim_stability", "Trim Stability", "{:.2f}"),
            ("industry_grade", "Industry Grade", "{}")
        ]
        
        for key, label, format_str in metric_items:
            frame = ctk.CTkFrame(self.metrics_frame)
            frame.pack(fill='x', padx=10, pady=2)
            
            label_widget = ctk.CTkLabel(
                frame,
                text=f"{label}:",
                font=ctk.CTkFont(size=11),
                width=120,
                anchor='w'
            )
            label_widget.pack(side='left', padx=(0, 10))
            
            value_widget = ctk.CTkLabel(
                frame,
                text="--",
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor='w'
            )
            value_widget.pack(side='left', fill='x', expand=True)
            
            self.metric_labels[key] = (value_widget, format_str)
            
    def _create_error_profile_tab(self):
        """Create error profile visualization tab."""
        profile_frame = self.track_tabview.tab("Error Profile")
        
        # Control frame for scaling
        control_frame = ctk.CTkFrame(profile_frame)
        control_frame.pack(fill='x', padx=5, pady=(5, 0))
        
        # Y-axis scaling controls
        scale_label = ctk.CTkLabel(control_frame, text="Y-Axis Scale:")
        scale_label.pack(side='left', padx=(10, 5))
        
        self.profile_scale_var = tk.DoubleVar(value=1.5)  # Default 1.5x spec limits
        self.profile_scale_slider = ctk.CTkSlider(
            control_frame,
            from_=0.5,
            to=3.0,
            variable=self.profile_scale_var,
            command=self._on_profile_scale_changed,
            width=150
        )
        self.profile_scale_slider.pack(side='left', padx=5)
        
        self.profile_scale_label = ctk.CTkLabel(control_frame, text="1.5x")
        self.profile_scale_label.pack(side='left', padx=(0, 10))
        
        # Create matplotlib figure
        self.profile_fig = Figure(figsize=(8, 5), dpi=100)
        self.profile_ax = self.profile_fig.add_subplot(111)
        
        self.profile_canvas = FigureCanvasTkAgg(self.profile_fig, profile_frame)
        self.profile_canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize empty plot
        self._update_error_profile_plot(None)
        
    def _create_statistics_tab(self):
        """Create statistics tab."""
        stats_frame = self.track_tabview.tab("Statistics")
        
        self.stats_text = ctk.CTkTextbox(
            stats_frame,
            height=400,
            font=ctk.CTkFont(family="Courier", size=11)
        )
        self.stats_text.pack(fill='both', expand=True, padx=5, pady=5)
        
    def _create_raw_data_tab(self):
        """Create raw data display tab."""
        raw_frame = self.track_tabview.tab("Raw Data")
        
        # Raw data text display
        self.raw_data_text = ctk.CTkTextbox(
            raw_frame,
            height=400,
            font=ctk.CTkFont(family="Courier", size=10)
        )
        self.raw_data_text.pack(fill='both', expand=True, padx=5, pady=5)
        
    def load_tracks(self, tracks_data: Dict[str, Any]):
        """Load multiple tracks data."""
        self.tracks_data = tracks_data
        
        # Update track selector
        if tracks_data:
            track_names = list(tracks_data.keys())
            self.track_selector.configure(values=track_names)
            self.track_selector.set(track_names[0])
            self._on_track_selected(track_names[0])
        else:
            self.track_selector.configure(values=["No tracks available"])
            self.track_selector.set("No tracks available")
            self._clear_display()
            
    def _on_track_selected(self, track_name: str):
        """Handle track selection."""
        if track_name in self.tracks_data:
            # Store selection for state persistence
            self.selected_track = track_name
            self.current_track_data = self.tracks_data[track_name]
            self._update_display()
            
            # Log selection for debugging
            logger.debug(f"Track '{track_name}' selected")
        elif track_name == "No tracks available":
            # Clear selection when no tracks
            self.selected_track = None
            self.current_track_data = None
            self._clear_display()
            
    def _update_display(self):
        """Update all displays with current track data."""
        if not self.current_track_data:
            self._clear_display()
            return
            
        # Update details tab
        self._update_details()
        
        # Update error profile
        self._update_error_profile_plot(self.current_track_data)
        
        # Update statistics
        self._update_statistics()
        
        # Update raw data
        self._update_raw_data()
        
    def _update_details(self):
        """Update details tab with track information."""
        data = self.current_track_data
        
        # Update info labels
        info_mapping = {
            'track_id': data.get('track_id', '--'),
            'position': data.get('position', '--'),
            'serial': data.get('serial', '--'),
            'timestamp': self._format_timestamp(data.get('timestamp')),
            'status': data.get('overall_status', '--'),
            'validation': data.get('validation_status', '--')
        }
        
        for key, value in info_mapping.items():
            if key in self.info_labels:
                self.info_labels[key].configure(text=str(value))
                
                # Color code status
                if key == 'status':
                    color = "green" if value == "Pass" else "red" if value == "Fail" else "gray"
                    self.info_labels[key].configure(text_color=color)
                elif key == 'validation':
                    color = "green" if value == "Valid" else "orange" if value == "Warning" else "red"
                    self.info_labels[key].configure(text_color=color)
        
        # Update metric labels
        for key, (label, format_str) in self.metric_labels.items():
            value = data.get(key)
            if value is not None:
                try:
                    formatted_value = format_str.format(value)
                    label.configure(text=formatted_value)
                    
                    # Color code based on status
                    if key == 'industry_grade':
                        grade_colors = {'A': 'green', 'B': 'blue', 'C': 'orange', 'D': 'red', 'F': 'darkred'}
                        color = grade_colors.get(str(value), 'gray')
                        label.configure(text_color=color)
                except:
                    label.configure(text="--")
            else:
                label.configure(text="--")
                
    def _update_error_profile_plot(self, track_data):
        """Update error profile plot."""
        self.profile_ax.clear()
        
        # Set white background for better visibility
        self.profile_ax.set_facecolor('white')
        self.profile_ax.tick_params(colors='black', labelcolor='black')
        for spine in self.profile_ax.spines.values():
            spine.set_color('#cccccc')
        
        if track_data and 'error_profile' in track_data:
            profile = track_data['error_profile']
            positions = profile.get('positions', [])
            errors = profile.get('errors', [])
            
            if positions and errors:
                self.profile_ax.plot(positions, errors, 'b-', linewidth=2, label='Error')
                
                # Add spec lines if available
                spec_limit = profile.get('spec_limit', 5.0)  # Default spec
                if 'spec_limit' in profile:
                    spec_limit = profile['spec_limit']
                elif 'linearity_spec' in track_data:
                    spec_limit = track_data.get('linearity_spec', 5.0)
                    
                self.profile_ax.axhline(y=spec_limit, color='r', linestyle='--', label=f'Spec: ±{spec_limit} V')
                self.profile_ax.axhline(y=-spec_limit, color='r', linestyle='--')
                
                # Add zero line
                self.profile_ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
                
                self.profile_ax.set_xlabel('Position (mm)', color='black')
                self.profile_ax.set_ylabel('Error (V)', color='black')
                self.profile_ax.set_title(f'Error Profile - {self.selected_track}', color='black')
                self.profile_ax.grid(True, alpha=0.3, color='#cccccc')
                
                # Calculate actual x-axis range from position data
                if positions:
                    x_min = min(positions)
                    x_max = max(positions)
                    x_range = x_max - x_min
                    x_padding = x_range * 0.05  # 5% padding
                    self.profile_ax.set_xlim(x_min - x_padding, x_max + x_padding)
                else:
                    # Fallback to default range
                    self.profile_ax.set_xlim(-200, 200)
                
                # Set y-axis limits based on scale factor
                scale_factor = self.profile_scale_var.get() if hasattr(self, 'profile_scale_var') else 1.5
                y_limit = spec_limit * scale_factor
                self.profile_ax.set_ylim(-y_limit, y_limit)
                
                self.profile_ax.legend(loc='best', framealpha=0.9)
            else:
                # Check if this is database data
                if profile and profile.get('note') == 'Raw data not available from database':
                    self._show_no_data_message(self.profile_ax, "Error profile data not available from database\n\nDatabase only stores calculated metrics.\nLoad files directly to see error profiles.")
                else:
                    self._show_no_data_message(self.profile_ax, "No error profile data available")
        else:
            self._show_no_data_message(self.profile_ax, "Select a track to view error profile")
        
        # Use tight_layout to prevent label cutoff
        try:
            self.profile_fig.tight_layout(pad=1.5)
        except Exception:
            # Fallback if tight_layout fails
            self.profile_fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
            
        self.profile_canvas.draw()
        
    def _update_statistics(self):
        """Update statistics display."""
        self.stats_text.delete("1.0", "end")
        
        if not self.current_track_data:
            self.stats_text.insert("1.0", "No track selected")
            return
            
        stats_text = f"Track Statistics: {self.selected_track}\n"
        stats_text += "=" * 50 + "\n\n"
        
        # Basic statistics
        stats_text += "Basic Metrics:\n"
        stats_text += "-" * 30 + "\n"
        
        metrics = [
            ("Sigma Gradient", self.current_track_data.get('sigma_gradient'), ".6f"),
            ("Linearity Error", self.current_track_data.get('linearity_error'), ".3f"),
            ("Resistance Change", self.current_track_data.get('resistance_change'), ".2f"),
            ("Trim Stability", self.current_track_data.get('trim_stability'), ".2f")
        ]
        
        for name, value, fmt in metrics:
            if value is not None:
                try:
                    formatted_value = format(value, fmt)
                    stats_text += f"{name:.<25} {formatted_value}\n"
                except (ValueError, TypeError) as e:
                    # Fallback for format errors
                    stats_text += f"{name:.<25} {str(value)}\n"
            else:
                stats_text += f"{name:.<25} N/A\n"
                
        # Statistical measures if available
        if 'statistics' in self.current_track_data:
            stats = self.current_track_data['statistics']
            stats_text += "\n\nStatistical Measures:\n"
            stats_text += "-" * 30 + "\n"
            
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    stats_text += f"{key.replace('_', ' ').title():.<25} {value:.4f}\n"
                else:
                    stats_text += f"{key.replace('_', ' ').title():.<25} {value}\n"
                    
        self.stats_text.insert("1.0", stats_text)
        
    def _update_raw_data(self):
        """Update raw data display."""
        self.raw_data_text.delete("1.0", "end")
        
        if not self.current_track_data:
            self.raw_data_text.insert("1.0", "No track selected")
            return
            
        # Format raw data as JSON-like structure
        import json
        try:
            # Create a serializable copy
            display_data = {}
            for key, value in self.current_track_data.items():
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    display_data[key] = value
                else:
                    display_data[key] = str(value)
                    
            raw_text = json.dumps(display_data, indent=2, sort_keys=True)
            self.raw_data_text.insert("1.0", raw_text)
        except Exception as e:
            self.raw_data_text.insert("1.0", f"Error formatting raw data: {str(e)}")
            
    def _clear_display(self):
        """Clear all displays."""
        # Clear info labels
        for label in self.info_labels.values():
            label.configure(text="--", text_color="gray")
            
        # Clear metric labels
        for label, _ in self.metric_labels.values():
            label.configure(text="--", text_color="gray")
            
        # Clear plots
        self._update_error_profile_plot(None)
        
        # Clear text displays
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", "No track selected")
        
        self.raw_data_text.delete("1.0", "end")
        self.raw_data_text.insert("1.0", "No track selected")
        
    def _format_timestamp(self, timestamp):
        """Format timestamp for display."""
        if not timestamp:
            return "--"
            
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return str(timestamp)
            
    def _on_profile_scale_changed(self, value):
        """Handle profile scale slider change."""
        scale_value = self.profile_scale_var.get()
        self.profile_scale_label.configure(text=f"{scale_value:.1f}x")
        # Update the plot with new scaling
        if self.current_track_data:
            self._update_error_profile_plot(self.current_track_data)
    
    def _show_no_data_message(self, ax, message):
        """Show no data message on plot."""
        # Set white background even for empty state
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            
        ax.text(0.5, 0.5, message, 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=12,
                color='gray',
                multialignment='center')
        ax.set_xticks([])
        ax.set_yticks([])