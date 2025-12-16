"""
Simple Chart Widget for Laser Trim Analyzer v3.

Consolidated from v2's 5-file, 3000+ line chart system into one clean widget.
Uses matplotlib with customtkinter integration.

Charts:
- Error vs Position (main analysis chart)
- SPC Control Chart (I-MR for trends)
- Distribution histogram
- Track comparison
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk

logger = logging.getLogger(__name__)

# Quality-focused color scheme
COLORS = {
    'pass': '#27ae60',       # Green
    'fail': '#e74c3c',       # Red
    'warning': '#f39c12',    # Orange
    'info': '#3498db',       # Blue
    'untrimmed': '#3498db',  # Blue
    'trimmed': '#27ae60',    # Green
    'spec_limit': '#e74c3c', # Red
    'threshold': '#f39c12',  # Orange
    'background': '#2b2b2b', # Dark background
    'text': '#ffffff',       # White text
    'grid': '#404040',       # Grid lines
}


@dataclass
class ChartStyle:
    """Chart styling configuration."""
    figure_size: Tuple[float, float] = (8, 5)
    dpi: int = 100
    font_size: int = 10
    title_size: int = 12
    line_width: float = 1.5
    marker_size: float = 4
    dark_mode: bool = True


class ChartWidget(ctk.CTkFrame):
    """
    Unified chart widget for v3.

    Features:
    - Error vs Position plot (main analysis)
    - SPC control charts
    - Histogram distributions
    - Track comparison
    - Dark mode support
    - Responsive resizing
    """

    def __init__(
        self,
        parent,
        style: Optional[ChartStyle] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)

        self.style = style or ChartStyle()
        self.figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self._resize_job = None  # For debouncing resize events

        self._setup_figure()

        # Bind to resize events for dynamic resizing
        self.bind("<Configure>", self._on_resize)

    def _setup_figure(self) -> None:
        """Initialize matplotlib figure with proper styling."""
        # Set matplotlib style for dark mode
        if self.style.dark_mode:
            plt.style.use('dark_background')

        self.figure = Figure(
            figsize=self.style.figure_size,
            dpi=self.style.dpi,
            facecolor=COLORS['background'] if self.style.dark_mode else 'white'
        )

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _on_resize(self, event) -> None:
        """Handle widget resize - update figure size to match."""
        # Debounce resize events to avoid excessive redraws
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(100, self._do_resize)

    def _do_resize(self) -> None:
        """Actually perform the resize after debounce."""
        if not self.figure or not self.canvas:
            return

        # Get the widget's current size
        width = self.winfo_width()
        height = self.winfo_height()

        # Avoid tiny sizes during initialization
        if width < 50 or height < 50:
            return

        # Convert pixels to inches for matplotlib
        dpi = self.style.dpi
        fig_width = width / dpi
        fig_height = height / dpi

        # Update figure size and redraw
        self.figure.set_size_inches(fig_width, fig_height, forward=True)
        try:
            self.figure.tight_layout()
        except Exception:
            pass  # tight_layout can fail with certain axes configurations
        self.canvas.draw_idle()

    def clear(self) -> None:
        """Clear the chart."""
        self.figure.clear()
        self.canvas.draw()

    def plot_error_vs_position(
        self,
        positions: List[float],
        trimmed_errors: List[float],
        upper_limits: Optional[List[float]] = None,
        lower_limits: Optional[List[float]] = None,
        untrimmed_positions: Optional[List[float]] = None,
        untrimmed_errors: Optional[List[float]] = None,
        offset: float = 0.0,
        title: str = "Error vs Position Analysis",
        fail_points: Optional[List[int]] = None,
    ) -> None:
        """
        Plot error vs position - the main analysis chart.

        Args:
            positions: Position values
            trimmed_errors: Trimmed error values
            upper_limits: Upper specification limits
            lower_limits: Lower specification limits
            untrimmed_positions: Untrimmed position values (optional)
            untrimmed_errors: Untrimmed error values (optional)
            offset: Optimal offset applied
            title: Chart title
            fail_points: Indices of fail points (optional)
        """
        self.clear()
        ax = self.figure.add_subplot(111)

        # Apply offset to trimmed errors
        shifted_errors = [e + offset for e in trimmed_errors]

        # Plot untrimmed data if available
        if untrimmed_positions and untrimmed_errors:
            ax.plot(
                untrimmed_positions, untrimmed_errors,
                color=COLORS['untrimmed'],
                linestyle='--',
                linewidth=self.style.line_width,
                alpha=0.7,
                label='Untrimmed'
            )

        # Plot trimmed data
        ax.plot(
            positions, shifted_errors,
            color=COLORS['trimmed'],
            linewidth=self.style.line_width,
            label=f'Trimmed (offset: {offset:.6f})'
        )

        # Plot specification limits
        if upper_limits and lower_limits:
            ax.plot(
                positions[:len(upper_limits)], upper_limits,
                color=COLORS['spec_limit'],
                linestyle='--',
                linewidth=1,
                alpha=0.8,
                label='Spec Limits'
            )
            ax.plot(
                positions[:len(lower_limits)], lower_limits,
                color=COLORS['spec_limit'],
                linestyle='--',
                linewidth=1,
                alpha=0.8
            )

            # Fill between limits
            ax.fill_between(
                positions[:min(len(upper_limits), len(lower_limits))],
                lower_limits[:min(len(upper_limits), len(lower_limits))],
                upper_limits[:min(len(upper_limits), len(lower_limits))],
                alpha=0.1,
                color=COLORS['spec_limit']
            )

        # Mark fail points
        if fail_points:
            fail_x = [positions[i] for i in fail_points if i < len(positions)]
            fail_y = [shifted_errors[i] for i in fail_points if i < len(shifted_errors)]
            ax.scatter(
                fail_x, fail_y,
                color=COLORS['fail'],
                marker='x',
                s=50,
                linewidths=2,
                label='Fail Points',
                zorder=5
            )

        # Styling
        ax.set_xlabel('Position', fontsize=self.style.font_size)
        ax.set_ylabel('Error', fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        ax.legend(loc='best', fontsize=self.style.font_size - 2)
        ax.grid(True, alpha=0.3, color=COLORS['grid'])

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_spc_control(
        self,
        values: List[float],
        dates: Optional[List[Any]] = None,
        ucl: Optional[float] = None,
        lcl: Optional[float] = None,
        center: Optional[float] = None,
        title: str = "SPC Control Chart",
        ylabel: str = "Sigma Gradient",
    ) -> None:
        """
        Plot SPC control chart (I-MR chart).

        Args:
            values: Data values
            dates: X-axis labels (dates or indices)
            ucl: Upper control limit
            lcl: Lower control limit
            center: Center line (mean)
            title: Chart title
            ylabel: Y-axis label
        """
        self.clear()
        ax = self.figure.add_subplot(111)

        x = dates if dates else list(range(len(values)))

        # Calculate control limits if not provided
        if center is None:
            center = np.mean(values)
        if ucl is None:
            ucl = center + 3 * np.std(values, ddof=1)
        if lcl is None:
            lcl = center - 3 * np.std(values, ddof=1)

        # Plot data
        ax.plot(x, values, 'o-', color=COLORS['info'],
                linewidth=self.style.line_width,
                markersize=self.style.marker_size)

        # Plot control limits
        ax.axhline(y=ucl, color=COLORS['fail'], linestyle='--',
                   label=f'UCL: {ucl:.4f}')
        ax.axhline(y=center, color=COLORS['warning'], linestyle='-',
                   label=f'Center: {center:.4f}')
        ax.axhline(y=lcl, color=COLORS['fail'], linestyle='--',
                   label=f'LCL: {lcl:.4f}')

        # Highlight out-of-control points
        for i, v in enumerate(values):
            if v > ucl or v < lcl:
                ax.scatter(x[i], v, color=COLORS['fail'], s=100, zorder=5)

        # Styling
        ax.set_xlabel('Sample', fontsize=self.style.font_size)
        ax.set_ylabel(ylabel, fontsize=self.style.font_size)
        ax.set_title(title, fontsize=self.style.title_size)
        ax.legend(loc='best', fontsize=self.style.font_size - 2)
        ax.grid(True, alpha=0.3, color=COLORS['grid'])

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_histogram(
        self,
        values: List[float],
        bins: int = 30,
        title: str = "Distribution",
        xlabel: str = "Value",
        spec_limit: Optional[float] = None,
    ) -> None:
        """
        Plot distribution histogram.

        Args:
            values: Data values
            bins: Number of bins
            title: Chart title
            xlabel: X-axis label
            spec_limit: Specification limit to show
        """
        self.clear()
        ax = self.figure.add_subplot(111)

        ax.hist(
            values, bins=bins,
            color=COLORS['info'],
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5
        )

        if spec_limit is not None:
            ax.axvline(x=spec_limit, color=COLORS['spec_limit'],
                       linestyle='--', linewidth=2,
                       label=f'Spec: {spec_limit:.4f}')
            ax.legend()

        # Add statistics
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        ax.axvline(x=mean, color=COLORS['warning'], linestyle='-',
                   alpha=0.8, label=f'Mean: {mean:.4f}')

        # Styling
        ax.set_xlabel(xlabel, fontsize=self.style.font_size)
        ax.set_ylabel('Count', fontsize=self.style.font_size)
        ax.set_title(f"{title}\n(Mean: {mean:.4f}, Std: {std:.4f})",
                     fontsize=self.style.title_size)
        ax.grid(True, alpha=0.3, color=COLORS['grid'], axis='y')

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_track_comparison(
        self,
        track1_positions: List[float],
        track1_errors: List[float],
        track2_positions: List[float],
        track2_errors: List[float],
        track1_label: str = "Track A",
        track2_label: str = "Track B",
        title: str = "Track Comparison",
    ) -> None:
        """
        Plot side-by-side track comparison.

        Args:
            track1_positions: Track 1 positions
            track1_errors: Track 1 errors
            track2_positions: Track 2 positions
            track2_errors: Track 2 errors
            track1_label: Label for track 1
            track2_label: Label for track 2
            title: Chart title
        """
        self.clear()

        # Create two subplots
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)

        # Plot track 1
        ax1.plot(
            track1_positions, track1_errors,
            color=COLORS['info'],
            linewidth=self.style.line_width
        )
        ax1.set_title(track1_label, fontsize=self.style.title_size)
        ax1.set_xlabel('Position', fontsize=self.style.font_size)
        ax1.set_ylabel('Error', fontsize=self.style.font_size)
        ax1.grid(True, alpha=0.3, color=COLORS['grid'])

        # Plot track 2
        ax2.plot(
            track2_positions, track2_errors,
            color=COLORS['pass'],
            linewidth=self.style.line_width
        )
        ax2.set_title(track2_label, fontsize=self.style.title_size)
        ax2.set_xlabel('Position', fontsize=self.style.font_size)
        ax2.set_ylabel('Error', fontsize=self.style.font_size)
        ax2.grid(True, alpha=0.3, color=COLORS['grid'])

        self.figure.suptitle(title, fontsize=self.style.title_size + 2)
        self.figure.tight_layout()
        self.canvas.draw()

    def show_placeholder(self, message: str = "No data to display") -> None:
        """Show placeholder message when no data."""
        self.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5, 0.5, message,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=14,
            color='gray'
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.canvas.draw()

    def show_status(
        self,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Show status display with metrics.

        Args:
            status: PASS, FAIL, or WARNING
            metrics: Dictionary of metrics to display
        """
        self.clear()
        ax = self.figure.add_subplot(111)

        # Status colors
        status_colors = {
            'PASS': COLORS['pass'],
            'FAIL': COLORS['fail'],
            'WARNING': COLORS['warning'],
            'ERROR': COLORS['fail'],
        }
        color = status_colors.get(status.upper(), COLORS['info'])

        # Large status text
        ax.text(
            0.5, 0.7, status.upper(),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=48,
            fontweight='bold',
            color=color
        )

        # Metrics below
        if metrics:
            metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
            ax.text(
                0.5, 0.3, metrics_text,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=12,
                color='white' if self.style.dark_mode else 'black'
            )

        ax.axis('off')
        self.figure.tight_layout()
        self.canvas.draw()

    def save_figure(self, filepath: str, dpi: int = 300) -> None:
        """Save the current figure to a file."""
        if self.figure:
            self.figure.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')

    def destroy(self):
        """Clean up matplotlib resources."""
        # Cancel any pending resize jobs
        if self._resize_job:
            self.after_cancel(self._resize_job)
            self._resize_job = None
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.figure:
            plt.close(self.figure)
        super().destroy()
