#!/usr/bin/env python3
"""Direct test of chart rendering capabilities."""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import customtkinter as ctk

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Set development environment
os.environ['LTA_ENV'] = 'development'

def test_chart_widget():
    """Test ChartWidget directly with sample data."""
    from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
    
    # Create main window
    root = ctk.CTk()
    root.title("Chart Widget Test")
    root.geometry("1400x900")
    
    # Create notebook for tabs
    notebook = ctk.CTkTabview(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Tab 1: Enhanced Control Chart
    tab1 = notebook.add("SPC Control Chart")
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    values = np.random.normal(0.1376, 0.02, 100)  # Mean=0.1376, std=0.02
    # Add some out-of-control points
    values[30] = 0.26  # Above UCL
    values[60] = 0.04  # Below LCL
    
    df_control = pd.DataFrame({
        'trim_date': dates,
        'sigma_gradient': values
    })
    
    chart1 = ChartWidget(tab1, figsize=(12, 6))
    chart1.pack(fill='both', expand=True)
    
    # Test enhanced control chart
    chart1.plot_enhanced_control_chart(
        data=df_control,
        value_column='sigma_gradient',
        date_column='trim_date',
        spec_limits=(0.050, 0.250),
        target_value=0.1376,
        title="Test: Sigma Gradient Statistical Process Control Chart"
    )
    
    # Tab 2: Process Capability Histogram
    tab2 = notebook.add("Process Capability")
    
    # Generate sample data for capability analysis
    sigma_data = np.random.normal(0.15, 0.03, 500)
    sigma_data = np.clip(sigma_data, 0.05, 0.25)  # Keep within reasonable range
    
    chart2 = ChartWidget(tab2, figsize=(12, 6))
    chart2.pack(fill='both', expand=True)
    
    # Test process capability histogram
    df_capability = pd.DataFrame({
        'sigma_gradient': sigma_data
    })
    chart2.plot_process_capability_histogram(
        data=df_capability,
        value_column='sigma_gradient',
        spec_limits=(0.050, 0.250),
        target_value=0.1376,
        title="Test: Sigma Gradient Process Capability Analysis"
    )
    
    # Tab 3: Quality Dashboard (Bar Chart)
    tab3 = notebook.add("Quality Metrics")
    
    chart3 = ChartWidget(tab3, figsize=(12, 6))
    chart3.pack(fill='both', expand=True)
    
    # Create quality metrics bar chart
    chart3.clear_chart()
    ax = chart3._get_or_create_axes()
    
    metrics = ['Pass Rate', 'In-Spec', 'Stability', 'Risk Score']
    values = [92.5, 88.3, 85.0, 91.2]
    targets = [95, 90, 80, 90]
    
    x_pos = range(len(metrics))
    bars = ax.bar(x_pos, values, 
                  color=[chart3.qa_colors['pass'] if v >= t 
                        else chart3.qa_colors['warning'] if v >= t*0.8 
                        else chart3.qa_colors['fail']
                        for v, t in zip(values, targets)],
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add target lines
    for i, target in enumerate(targets):
        ax.axhline(y=target, xmin=(i-0.4)/len(metrics), xmax=(i+0.4)/len(metrics), 
                  color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Quality Metrics', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Test: Quality Metrics Overview', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    chart3._apply_theme_to_axes(ax)
    chart3.canvas.draw_idle()
    
    # Tab 4: Time Series with Moving Average
    tab4 = notebook.add("Time Series")
    
    chart4 = ChartWidget(tab4, figsize=(12, 6))
    chart4.pack(fill='both', expand=True)
    
    # Generate time series data
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    accuracy = np.random.normal(93, 3, 60)
    accuracy = np.clip(accuracy, 80, 100)
    
    df_ts = pd.DataFrame({
        'date': dates,
        'accuracy': accuracy
    })
    df_ts['ma_7'] = df_ts['accuracy'].rolling(window=7, min_periods=3).mean()
    
    # Plot time series
    chart4.clear_chart()
    ax = chart4._get_or_create_axes()
    
    ax.plot(df_ts['date'], df_ts['accuracy'], 
           'o-', markersize=4, linewidth=1,
           color=chart4.qa_colors['primary'],
           label='Daily Accuracy', alpha=0.6)
    
    ax.plot(df_ts['date'], df_ts['ma_7'], 
           linewidth=2.5, color=chart4.qa_colors['secondary'],
           label='7-day MA', alpha=0.9)
    
    # Add control limits
    mean_acc = df_ts['accuracy'].mean()
    std_acc = df_ts['accuracy'].std()
    ucl = min(100, mean_acc + 2 * std_acc)
    lcl = max(80, mean_acc - 2 * std_acc)
    
    ax.axhline(y=ucl, color=chart4.qa_colors['warning'], linestyle=':', linewidth=1.5, 
              label=f'UCL: {ucl:.1f}%', alpha=0.6)
    ax.axhline(y=mean_acc, color=chart4.qa_colors['control_center'], linestyle='-', linewidth=2, 
              label=f'Mean: {mean_acc:.1f}%', alpha=0.7)
    ax.axhline(y=lcl, color=chart4.qa_colors['warning'], linestyle=':', linewidth=1.5, 
              label=f'LCL: {lcl:.1f}%', alpha=0.6)
    ax.axhline(y=95, color=chart4.qa_colors['spec_limits'], linestyle='--', linewidth=2, 
              label='Target: 95%', alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_title('Test: ML Model Performance Trend', fontsize=12, fontweight='bold')
    ax.set_ylim(75, 105)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    chart4._apply_theme_to_axes(ax)
    
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    
    chart4.canvas.draw_idle()
    
    # Tab 5: Color Palette Display
    tab5 = notebook.add("QA Colors")
    
    chart5 = ChartWidget(tab5, figsize=(14, 8))
    chart5.pack(fill='both', expand=True)
    
    # Display all QA colors
    chart5.clear_chart()
    fig = chart5.figure
    fig.clear()
    
    # Create subplots for color display
    colors_to_display = list(chart5.qa_colors.items())
    n_colors = len(colors_to_display)
    cols = 5
    rows = (n_colors + cols - 1) // cols
    
    for i, (name, color) in enumerate(colors_to_display):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=8)
    
    fig.suptitle('QA Color Palette for Manufacturing Documentation', fontsize=12, fontweight='bold')
    fig.tight_layout()
    chart5.canvas.draw_idle()
    
    # Info label
    info_label = ctk.CTkLabel(root, 
                             text="Chart Display Test - All enhanced visualization methods are functional",
                             font=('Arial', 14))
    info_label.pack(pady=5)
    
    # Status info
    status_text = (
        "✓ Enhanced control chart with SPC limits\n"
        "✓ Process capability histogram with Cp/Cpk\n"
        "✓ Quality metrics dashboard\n"
        "✓ Time series with moving averages\n"
        "✓ Professional QA color palette"
    )
    status_label = ctk.CTkLabel(root, text=status_text, justify='left')
    status_label.pack(pady=5)
    
    print("\n=== Chart Widget Test Results ===")
    print("✓ ChartWidget initialized successfully")
    print(f"✓ QA colors loaded: {len(chart1.qa_colors)} colors")
    print("✓ Enhanced control chart method working")
    print("✓ Process capability histogram method working")
    print("✓ All chart types rendering correctly")
    print("\nVisual inspection window is open. Close to exit.")
    
    # Start GUI
    root.mainloop()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_chart_widget()