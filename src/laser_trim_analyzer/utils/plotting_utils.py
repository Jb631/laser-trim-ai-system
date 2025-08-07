"""
Plotting utilities for laser trim analysis visualization.

Creates professional QA plots with consistent styling and formatting.
"""

# Set matplotlib to use non-interactive backend to avoid thread warnings
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from datetime import datetime
import logging

from laser_trim_analyzer.core.models import TrackData, AnalysisResult

logger = logging.getLogger(__name__)

# Set default style
try:
    plt.style.use('seaborn')
except OSError:
    # Fallback if seaborn style not available
    plt.style.use('default')
    
sns.set_palette("husl")

# QA color scheme - theme-aware colors
# These match the colors used in ChartWidget for consistency
QA_COLORS = {
    'pass': '#27ae60',      # Success green (matches ChartWidget)
    'fail': '#e74c3c',      # Error red (matches ChartWidget)
    'warning': '#f39c12',   # Warning orange (matches ChartWidget)
    'info': '#3498db',      # Primary blue (matches ChartWidget)
    'untrimmed': '#3498db', # Primary blue
    'trimmed': '#27ae60',   # Success green
    'filtered': '#9b59b6',  # Secondary purple (matches ChartWidget)
    'spec_limit': '#e74c3c',# Error red
    'threshold': '#f39c12', # Warning orange
    'grid': '#95a5a6',      # Neutral gray (matches ChartWidget)
    'text': '#2c3e50',      # Dark text
    'background': '#ffffff', # White background
    'secondary_bg': '#f8f9fa' # Light gray background
}

# Dark theme colors
QA_COLORS_DARK = {
    'pass': '#2ecc71',      # Brighter green for dark theme
    'fail': '#c0392b',      # Darker red for dark theme
    'warning': '#d68910',   # Darker orange for dark theme
    'info': '#2980b9',      # Darker blue for dark theme
    'untrimmed': '#2980b9', # Darker blue
    'trimmed': '#2ecc71',   # Brighter green
    'filtered': '#8e44ad',  # Darker purple
    'spec_limit': '#c0392b',# Darker red
    'threshold': '#d68910', # Darker orange
    'grid': '#7f8c8d',      # Darker gray for visibility
    'text': '#ecf0f1',      # Light text for dark background
    'background': '#2b2b2b', # Dark background
    'secondary_bg': '#1e1e1e' # Darker background
}


def get_theme_colors(is_dark: bool = False) -> Dict[str, str]:
    """
    Get color scheme based on theme.
    
    Args:
        is_dark: Whether to use dark theme colors
        
    Returns:
        Dictionary of color mappings
    """
    return QA_COLORS_DARK if is_dark else QA_COLORS


def apply_theme_to_axes(ax, is_dark: bool = False):
    """
    Apply theme colors to matplotlib axes.
    
    Args:
        ax: Matplotlib axes
        is_dark: Whether to use dark theme
    """
    colors = get_theme_colors(is_dark)
    
    # Set background color
    ax.set_facecolor(colors['secondary_bg'])
    
    # Set spine colors
    for spine in ax.spines.values():
        spine.set_color(colors['grid'])
    
    # Set tick colors
    ax.tick_params(colors=colors['text'], labelcolor=colors['text'])
    
    # Control tick density to prevent MAXTICKS warnings
    # Apply reasonable limits to prevent overwhelming tick generation
    ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', max_nbins=12, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', max_nbins=10, prune='both'))
    
    # Set label colors
    ax.xaxis.label.set_color(colors['text'])
    ax.yaxis.label.set_color(colors['text'])
    ax.title.set_color(colors['text'])
    
    # Set grid
    ax.grid(True, alpha=0.3, color=colors['grid'])


def create_analysis_plot(
        track_data: TrackData,
        output_dir: Path,
        filename_prefix: str,
        dpi: int = 150,
        figsize: Tuple[int, int] = (15, 10)
) -> Path:
    """
    Create comprehensive analysis plot for track data.

    This creates a multi-panel plot showing:
    - Error vs position with spec limits
    - Sigma gradient analysis
    - Histogram of errors
    - Key metrics summary

    Args:
        track_data: Track analysis data
        output_dir: Output directory for plot
        filename_prefix: Prefix for filename
        dpi: Plot resolution
        figsize: Figure size in inches

    Returns:
        Path to saved plot file
    """
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
    fig.suptitle(f'Laser Trim Analysis - {filename_prefix}', fontsize=16, fontweight='bold')

    # Create grid spec for custom layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main error plot (top 2/3)
    ax_main = fig.add_subplot(gs[0:2, :])
    _plot_error_vs_position(ax_main, track_data)

    # Histogram (bottom left)
    ax_hist = fig.add_subplot(gs[2, 0])
    _plot_error_histogram(ax_hist, track_data)

    # Metrics summary (bottom middle)
    ax_metrics = fig.add_subplot(gs[2, 1])
    _plot_metrics_summary(ax_metrics, track_data)

    # Pass/Fail indicator (bottom right)
    ax_status = fig.add_subplot(gs[2, 2])
    _plot_status_indicator(ax_status, track_data)

    # Save plot
    output_path = output_dir / f"{filename_prefix}_analysis.png"
    save_plot(fig, output_path, dpi=dpi)

    plt.close(fig)
    return output_path


def _plot_error_vs_position(ax: plt.Axes, track_data: TrackData):
    """Plot error vs position with spec limits."""
    if not track_data.position_data or not track_data.error_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        return

    positions = np.array(track_data.position_data)
    errors = np.array(track_data.error_data)
    
    # Apply optimal offset to trimmed data for display
    offset = track_data.linearity_analysis.optimal_offset
    errors_shifted = errors + offset

    # Plot untrimmed data if available (no offset applied)
    if track_data.untrimmed_positions and track_data.untrimmed_errors:
        untrimmed_pos = np.array(track_data.untrimmed_positions)
        untrimmed_err = np.array(track_data.untrimmed_errors)
        ax.plot(untrimmed_pos, untrimmed_err, 'b--', linewidth=1.5, label='Untrimmed Data',
                color=QA_COLORS['untrimmed'], alpha=0.6)
    
    # Plot trimmed/final data with offset applied
    ax.plot(positions, errors_shifted, 'g-', linewidth=2, label='Trimmed Data (Offset Applied)',
            color=QA_COLORS['trimmed'])

    # Add spec limits if available - use actual varying limits from data
    if hasattr(track_data, 'upper_limits') and hasattr(track_data, 'lower_limits'):
        upper_limits = track_data.upper_limits
        lower_limits = track_data.lower_limits
        
        if upper_limits and lower_limits and len(upper_limits) == len(positions):
            # Debug log to verify alignment
            logger.debug(f"Plotting limits - positions: {len(positions)}, upper: {len(upper_limits)}, lower: {len(lower_limits)}")
            logger.debug(f"First few position/limit pairs: {[(positions[i], upper_limits[i], lower_limits[i]) for i in range(min(3, len(positions)))]}")
            
            # Plot varying spec limits
            ax.plot(positions, upper_limits, 'r--', linewidth=2, 
                   label='Upper Spec Limit', color=QA_COLORS['spec_limit'])
            ax.plot(positions, lower_limits, 'r--', linewidth=2,
                   label='Lower Spec Limit', color=QA_COLORS['spec_limit'])
            
            # Fill spec limit area
            ax.fill_between(positions, lower_limits, upper_limits, alpha=0.1,
                        color=QA_COLORS['spec_limit'])

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Auto-scale y-axis to show all data with maximum zoom
    if len(errors) > 0:
        # Get all data points that need to be visible (use shifted errors for trimmed data)
        all_data = list(errors_shifted)
        if hasattr(track_data, 'untrimmed_errors') and track_data.untrimmed_errors:
            all_data.extend(track_data.untrimmed_errors)
        
        # Get the actual min and max to ensure all points are visible
        data_min = np.nanmin(all_data)
        data_max = np.nanmax(all_data)
        
        # Also consider spec limits if available
        if hasattr(track_data, 'upper_limits') and hasattr(track_data, 'lower_limits'):
            if track_data.upper_limits and track_data.lower_limits:
                # Get valid spec limits
                valid_upper = [u for u in track_data.upper_limits if u is not None and not np.isnan(u)]
                valid_lower = [l for l in track_data.lower_limits if l is not None and not np.isnan(l)]
                
                if valid_upper and valid_lower:
                    spec_max = max(valid_upper)
                    spec_min = min(valid_lower)
                    
                    # Include spec limits in the range
                    data_min = min(data_min, spec_min)
                    data_max = max(data_max, spec_max)
        
        # Calculate the data range
        data_range = data_max - data_min
        
        # Add minimal padding (5%) to ensure points aren't cut off at edges
        # Use a minimum padding to handle cases where range is very small
        if data_range > 0:
            padding = data_range * 0.05  # 5% padding
        else:
            # If all data points are the same value
            padding = abs(data_max) * 0.1 if data_max != 0 else 0.1
        
        y_min = data_min - padding
        y_max = data_max + padding
        
        ax.set_ylim(y_min, y_max)
        
        # Log scaling info for debugging
        logger.info(f"Plot scaling: data range [{data_min:.6f}, {data_max:.6f}], "
                    f"plot limits [{y_min:.6f}, {y_max:.6f}]")

    # Always check and highlight fail points (even if count is 0, we want to verify)
    if hasattr(track_data, 'upper_limits') and hasattr(track_data, 'lower_limits'):
        # Find points outside their specific limits
        upper_limits = track_data.upper_limits
        lower_limits = track_data.lower_limits
        
        if upper_limits and lower_limits and len(upper_limits) == len(positions):
            # Find points outside their specific limits (using already shifted errors)
            fail_mask = np.zeros(len(errors_shifted), dtype=bool)
            fail_count = 0
            for i in range(len(errors_shifted)):
                if (i < len(upper_limits) and i < len(lower_limits) and 
                    upper_limits[i] is not None and lower_limits[i] is not None and
                    not np.isnan(upper_limits[i]) and not np.isnan(lower_limits[i]) and
                    not np.isnan(errors_shifted[i])):
                    # Only check points where limits are defined
                    if errors_shifted[i] > upper_limits[i] or errors_shifted[i] < lower_limits[i]:
                        fail_mask[i] = True
                        fail_count += 1
            
            if np.any(fail_mask):
                ax.scatter(positions[fail_mask], errors_shifted[fail_mask],
                           color=QA_COLORS['fail'], s=100, marker='x',
                           linewidth=3, label=f'Fail Points ({fail_count})', zorder=5)
            
            # Add text annotation if there are failures
            if fail_count > 0:
                ax.text(0.02, 0.98, f'FAIL: {fail_count} points out of spec (0 allowed)', 
                       transform=ax.transAxes, fontsize=12, color=QA_COLORS['fail'],
                       fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=QA_COLORS['fail']))

    # Labels and formatting
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Error (Volts)', fontsize=12)
    ax.set_title('Error vs Position Analysis', fontsize=14, fontweight='bold')
    
    # Add more tick marks for better readability
    # X-axis: Use ~10-15 ticks
    if len(positions) > 0:
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        # Calculate nice tick spacing
        if x_range > 0:
            # Aim for about 10-15 ticks
            raw_spacing = x_range / 12
            # Round to a nice number (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, etc.)
            magnitude = 10 ** np.floor(np.log10(raw_spacing))
            normalized = raw_spacing / magnitude
            if normalized <= 1:
                nice_spacing = 1 * magnitude
            elif normalized <= 2:
                nice_spacing = 2 * magnitude
            elif normalized <= 5:
                nice_spacing = 5 * magnitude
            else:
                nice_spacing = 10 * magnitude
            
            x_ticks = np.arange(np.floor(x_min / nice_spacing) * nice_spacing,
                               np.ceil(x_max / nice_spacing) * nice_spacing + nice_spacing,
                               nice_spacing)
            ax.set_xticks(x_ticks)
    
    # Y-axis: Use ~10-15 ticks with appropriate precision
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    if y_range > 0:
        # Similar calculation for y-axis
        raw_spacing = y_range / 12
        magnitude = 10 ** np.floor(np.log10(raw_spacing))
        normalized = raw_spacing / magnitude
        if normalized <= 1:
            nice_spacing = 1 * magnitude
        elif normalized <= 2:
            nice_spacing = 2 * magnitude
        elif normalized <= 5:
            nice_spacing = 5 * magnitude
        else:
            nice_spacing = 10 * magnitude
        
        y_ticks = np.arange(np.floor(y_min / nice_spacing) * nice_spacing,
                           np.ceil(y_max / nice_spacing) * nice_spacing + nice_spacing,
                           nice_spacing)
        ax.set_yticks(y_ticks)
        
        # Format y-axis labels based on scale
        if abs(y_max - y_min) < 0.01:  # Very small range, use more decimals
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.6f}'))
        elif abs(y_max - y_min) < 0.1:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.4f}'))
        elif abs(y_max - y_min) < 1:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    # Add minor ticks for even more detail
    ax.minorticks_on()
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.1, linestyle=':')
    
    ax.legend(loc='best', fontsize=9)

    # Add optimal offset info to title if offset is non-zero
    if track_data.linearity_analysis.optimal_offset != 0:
        offset = track_data.linearity_analysis.optimal_offset
        current_title = ax.get_title()
        ax.set_title(f'{current_title} (Offset: {offset:.6f} V)', fontsize=14, fontweight='bold')


def _plot_error_histogram(ax: plt.Axes, track_data: TrackData):
    """Plot histogram of errors."""
    if not track_data.error_data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        return

    errors = np.array(track_data.error_data)

    # Create histogram
    n, bins, patches = ax.hist(errors, bins=20, edgecolor='black',
                               alpha=0.7, color=QA_COLORS['info'])

    # Add normal distribution overlay
    mu, sigma = np.mean(errors), np.std(errors)
    x = np.linspace(errors.min(), errors.max(), 100)
    ax.plot(x, ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                np.exp(-0.5 * ((x - mu) / sigma) ** 2)) * len(errors) * (bins[1] - bins[0]),
            'r--', linewidth=2, label=f'μ={mu:.3f}, σ={sigma:.3f}')

    ax.set_xlabel('Error Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Error Distribution', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_metrics_summary(ax: plt.Axes, track_data: TrackData):
    """Plot key metrics summary."""
    ax.axis('off')

    # Prepare metrics text
    metrics = [
        f"Sigma Gradient: {track_data.sigma_analysis.sigma_gradient:.4f}",
        f"Sigma Threshold: {track_data.sigma_analysis.sigma_threshold:.4f}",
        f"Sigma Pass: {'✓' if track_data.sigma_analysis.sigma_pass else '✗'}",
        f"",
        f"Linearity Pass: {'✓' if track_data.linearity_analysis.linearity_pass else '✗'}",
        f"Fail Points: {track_data.linearity_analysis.linearity_fail_points} (0 allowed)",
        f"",
        f"Risk Category: {track_data.failure_prediction.risk_category.value if track_data.failure_prediction else 'N/A'}",
        f"Failure Prob: {track_data.failure_prediction.failure_probability:.1%}" if track_data.failure_prediction else ""
    ]

    # Display metrics
    y_pos = 0.9
    for metric in metrics:
        if metric:  # Skip empty lines
            color = QA_COLORS['text']
            if '✗' in metric:
                color = QA_COLORS['fail']
            elif '✓' in metric:
                color = QA_COLORS['pass']

            ax.text(0.05, y_pos, metric, transform=ax.transAxes,
                    fontsize=11, color=color, fontweight='bold' if '✗' in metric else 'normal')
        y_pos -= 0.12

    ax.set_title('Analysis Metrics', fontsize=12)


def _plot_status_indicator(ax: plt.Axes, track_data: TrackData):
    """Plot pass/fail status indicator."""
    ax.axis('off')

    # Determine status and color
    status = track_data.status.value
    if status == "Pass":
        color = QA_COLORS['pass']
        symbol = "✓"
        description = "PASS"
    elif status == "Fail":
        color = QA_COLORS['fail']
        symbol = "✗"
        description = "FAIL"
    else:
        color = QA_COLORS['warning']
        symbol = "⚠"
        description = "WARNING"

    # Draw status circle
    circle = plt.Circle((0.5, 0.7), 0.25, color=color, alpha=0.8)
    ax.add_patch(circle)

    # Add status text
    ax.text(0.5, 0.7, symbol, ha='center', va='center',
            fontsize=40, color='white', fontweight='bold')
    ax.text(0.5, 0.45, description, ha='center', va='center',
            fontsize=16, color=color, fontweight='bold')
    
    # Add status reason if available
    if hasattr(track_data, 'status_reason') and track_data.status_reason and status != "Pass":
        # Wrap long text
        import textwrap
        wrapped_text = textwrap.fill(track_data.status_reason, width=30)
        ax.text(0.5, 0.25, wrapped_text, ha='center', va='center',
                fontsize=9, color=color, style='italic', wrap=True)
    
    # Add small explanation text
    ax.text(0.5, 0.05, 'Overall Status', ha='center', va='center',
            fontsize=9, color='gray', style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def create_histogram(
        data: Union[List[float], np.ndarray],
        title: str = "Data Distribution",
        xlabel: str = "Value",
        bins: int = 30,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150
) -> Optional[Path]:
    """
    Create histogram plot with normal distribution overlay.

    Args:
        data: Data to plot
        title: Plot title
        xlabel: X-axis label
        bins: Number of histogram bins
        output_path: Path to save plot (optional)
        figsize: Figure size
        dpi: Plot resolution

    Returns:
        Path to saved plot or None if not saved
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    data = np.array(data)
    data_clean = data[~np.isnan(data)]

    if len(data_clean) == 0:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
    else:
        # Create histogram
        n, bins_edges, patches = ax.hist(data_clean, bins=bins, density=True,
                                         alpha=0.7, color=QA_COLORS['info'],
                                         edgecolor='black')

        # Add KDE
        from scipy import stats
        kde = stats.gaussian_kde(data_clean)
        x_range = np.linspace(data_clean.min(), data_clean.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

        # Add statistics
        mean = np.mean(data_clean)
        std = np.std(data_clean)
        ax.axvline(mean, color=QA_COLORS['pass'], linestyle='--',
                   linewidth=2, label=f'Mean: {mean:.3f}')
        ax.axvline(mean - std, color=QA_COLORS['warning'], linestyle=':',
                   linewidth=1.5, alpha=0.7)
        ax.axvline(mean + std, color=QA_COLORS['warning'], linestyle=':',
                   linewidth=1.5, alpha=0.7, label=f'±1σ: {std:.3f}')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        save_plot(fig, output_path, dpi=dpi)
        plt.close(fig)
        return output_path
    else:
        plt.show()
        # Always close the figure after showing to prevent memory leak
        plt.close(fig)
        return None


def create_trend_chart(
        timestamps: List[datetime],
        values: Dict[str, List[float]],
        title: str = "Trend Analysis",
        ylabel: str = "Value",
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 150
) -> Optional[Path]:
    """
    Create trend chart showing multiple metrics over time.

    Args:
        timestamps: List of timestamps
        values: Dictionary of metric_name -> values
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save plot
        figsize: Figure size
        dpi: Plot resolution

    Returns:
        Path to saved plot or None
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot each metric
    for i, (metric_name, metric_values) in enumerate(values.items()):
        color = list(QA_COLORS.values())[i % len(QA_COLORS)]
        ax.plot(timestamps, metric_values, marker='o', markersize=4,
                linewidth=2, label=metric_name, color=color)

    # Format x-axis
    fig.autofmt_xdate()

    # Add moving average for first metric
    if values:
        first_metric = list(values.values())[0]
        if len(first_metric) > 7:
            window = min(7, len(first_metric) // 3)
            ma = pd.Series(first_metric).rolling(window=window, center=True).mean()
            ax.plot(timestamps, ma, '--', color='black', alpha=0.5,
                    linewidth=2, label=f'{window}-point MA')

    ax.set_xlabel('Date/Time', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if output_path:
        save_plot(fig, output_path, dpi=dpi)
        plt.close(fig)
        return output_path
    else:
        plt.show()
        # Always close the figure after showing to prevent memory leak
        plt.close(fig)
        return None


def create_comparison_plot(
        models: List[str],
        metrics: Dict[str, List[float]],
        title: str = "Model Comparison",
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 150
) -> Optional[Path]:
    """
    Create comparison plot for multiple models.

    Args:
        models: List of model names
        metrics: Dictionary of metric_name -> values for each model
        title: Plot title
        output_path: Path to save plot
        figsize: Figure size
        dpi: Plot resolution

    Returns:
        Path to saved plot or None
    """
    n_metrics = len(metrics)
    n_models = len(models)

    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, dpi=dpi,
                             sharex=True)

    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_models)
    width = 0.6

    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]

        # Create bars
        bars = ax.bar(x, values, width, edgecolor='black')

        # Color bars based on performance
        for j, (bar, value) in enumerate(zip(bars, values)):
            if metric_name.lower().endswith('pass') or 'rate' in metric_name.lower():
                # Higher is better
                if value >= 95:
                    bar.set_facecolor(QA_COLORS['pass'])
                elif value >= 90:
                    bar.set_facecolor(QA_COLORS['warning'])
                else:
                    bar.set_facecolor(QA_COLORS['fail'])
            else:
                # Lower is better (like sigma gradient)
                if value <= 0.01:
                    bar.set_facecolor(QA_COLORS['pass'])
                elif value <= 0.02:
                    bar.set_facecolor(QA_COLORS['warning'])
                else:
                    bar.set_facecolor(QA_COLORS['fail'])

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.3f}' if value < 1 else f'{value:.1f}',
                    ha='center', va='bottom', fontsize=10)

        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} by Model', fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)

    # Set x-axis labels
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(models, rotation=45, ha='right')
    axes[-1].set_xlabel('Model', fontsize=12)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        save_plot(fig, output_path, dpi=dpi)
        plt.close(fig)
        return output_path
    else:
        plt.show()
        # Always close the figure after showing to prevent memory leak
        plt.close(fig)
        return None


def save_plot(
        fig: Figure,
        output_path: Path,
        dpi: int = 150,
        bbox_inches: str = 'tight',
        transparent: bool = False
) -> None:
    """
    Save matplotlib figure to file.

    Args:
        fig: Matplotlib figure
        output_path: Output file path
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box setting
        transparent: Save with transparent background
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
            facecolor='white' if not transparent else 'none'
        )
        logger.info(f"Plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
        raise


def create_batch_summary_plot(
        results: List[AnalysisResult],
        output_dir: Path,
        dpi: int = 150
) -> Path:
    """
    Create summary plot for batch processing results.

    Args:
        results: List of analysis results
        output_dir: Output directory
        dpi: Plot resolution

    Returns:
        Path to saved plot
    """
    fig = plt.figure(figsize=(16, 10), dpi=dpi)

    # Extract data
    models = [r.metadata.model for r in results]
    pass_rates = []
    avg_sigmas = []
    risk_distribution = {'Low': 0, 'Medium': 0, 'High': 0}

    for result in results:
        # Calculate pass rate
        track_passes = [
            t.sigma_analysis.sigma_pass and t.linearity_analysis.linearity_pass
            for t in result.tracks.values()
        ]
        pass_rates.append(sum(track_passes) / len(track_passes) * 100 if track_passes else 0)

        # Average sigma
        sigmas = [t.sigma_analysis.sigma_gradient for t in result.tracks.values()]
        avg_sigmas.append(np.mean(sigmas) if sigmas else 0)

        # Risk distribution
        for track in result.tracks.values():
            if track.failure_prediction:
                risk_distribution[track.failure_prediction.risk_category.value] += 1

    # Create subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Pass rate distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(pass_rates, bins=20, color=QA_COLORS['pass'],
             edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Pass Rate (%)')
    ax1.set_ylabel('Count')
    ax1.set_title('Pass Rate Distribution')
    ax1.grid(True, alpha=0.3)

    # Sigma distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(avg_sigmas, bins=20, color=QA_COLORS['info'],
             edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Average Sigma Gradient')
    ax2.set_ylabel('Count')
    ax2.set_title('Sigma Gradient Distribution')
    ax2.grid(True, alpha=0.3)

    # Risk distribution pie chart
    ax3 = fig.add_subplot(gs[0, 2])
    risk_colors = [QA_COLORS['pass'], QA_COLORS['warning'], QA_COLORS['fail']]
    wedges, texts, autotexts = ax3.pie(
        risk_distribution.values(),
        labels=risk_distribution.keys(),
        colors=risk_colors,
        autopct='%1.1f%%',
        startangle=90
    )
    ax3.set_title('Risk Category Distribution')

    # Model performance comparison
    ax4 = fig.add_subplot(gs[1, :])
    model_stats = {}
    for model, pass_rate, sigma in zip(models, pass_rates, avg_sigmas):
        if model not in model_stats:
            model_stats[model] = {'pass_rates': [], 'sigmas': []}
        model_stats[model]['pass_rates'].append(pass_rate)
        model_stats[model]['sigmas'].append(sigma)

    # Calculate averages per model
    model_names = list(model_stats.keys())
    model_pass_rates = [np.mean(stats['pass_rates']) for stats in model_stats.values()]
    model_sigmas = [np.mean(stats['sigmas']) for stats in model_stats.values()]

    x = np.arange(len(model_names))
    width = 0.35

    ax4_twin = ax4.twinx()

    bars1 = ax4.bar(x - width / 2, model_pass_rates, width,
                    label='Pass Rate (%)', color=QA_COLORS['pass'])
    bars2 = ax4_twin.bar(x + width / 2, model_sigmas, width,
                         label='Avg Sigma', color=QA_COLORS['info'])

    ax4.set_xlabel('Model')
    ax4.set_ylabel('Pass Rate (%)', color=QA_COLORS['pass'])
    ax4_twin.set_ylabel('Average Sigma Gradient', color=QA_COLORS['info'])
    ax4.set_title('Performance by Model')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.tick_params(axis='y', labelcolor=QA_COLORS['pass'])
    ax4_twin.tick_params(axis='y', labelcolor=QA_COLORS['info'])
    ax4.grid(True, alpha=0.3)

    # Add values on bars
    for bar, value in zip(bars1, model_pass_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{value:.1f}', ha='center', va='bottom')

    for bar, value in zip(bars2, model_sigmas):
        height = bar.get_height()
        ax4_twin.text(bar.get_x() + bar.get_width() / 2., height,
                      f'{value:.4f}', ha='center', va='bottom')

    # Overall title
    fig.suptitle(f'Batch Analysis Summary - {len(results)} Files Processed',
                 fontsize=16, fontweight='bold')

    # Save
    output_path = output_dir / "batch_summary_plot.png"
    save_plot(fig, output_path, dpi=dpi)
    plt.close(fig)

    return output_path


# Plot style manager for consistent theming
class PlotStyleManager:
    """Manages consistent plot styling across the application."""

    @staticmethod
    def apply_qa_style():
        """Apply QA-specific plot styling."""
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'sans-serif',
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            'axes.edgecolor': QA_COLORS['grid'],
            'axes.linewidth': 1.0,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.size': 5,
            'ytick.major.size': 5,
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'figure.autolayout': True,
            'figure.dpi': 100,
            'savefig.dpi': 150,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })

    @staticmethod
    def get_color_palette(n_colors: int = 6) -> List[str]:
        """Get QA color palette."""
        base_colors = [
            QA_COLORS['info'],
            QA_COLORS['pass'],
            QA_COLORS['warning'],
            QA_COLORS['fail'],
            QA_COLORS['untrimmed'],
            QA_COLORS['filtered']
        ]

        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        else:
            # Generate additional colors
            return sns.color_palette("husl", n_colors)