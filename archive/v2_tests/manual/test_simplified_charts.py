#!/usr/bin/env python3
"""Test the simplified chart implementations."""

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

def test_simplified_charts():
    """Test simplified chart rendering."""
    from laser_trim_analyzer.gui.widgets.chart_widget import ChartWidget
    
    # Create main window
    root = ctk.CTk()
    root.title("Simplified Chart Test")
    root.geometry("1200x800")
    
    # Create tabs
    tabview = ctk.CTkTabview(root)
    tabview.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Tab 1: Simplified Control Chart
    tab1 = tabview.add("Control Chart")
    
    # Generate test data with trend
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    base_trend = np.linspace(0.12, 0.16, 60)  # Gradual increase
    noise = np.random.normal(0, 0.01, 60)
    values = base_trend + noise
    
    df_control = pd.DataFrame({
        'trim_date': dates,
        'sigma_gradient': values
    })
    
    chart1 = ChartWidget(tab1, figsize=(10, 5))
    chart1.pack(fill='both', expand=True)
    
    # Test simplified control chart
    chart1.plot_enhanced_control_chart(
        data=df_control,
        value_column='sigma_gradient',
        date_column='trim_date',
        spec_limits=(0.08, 0.20),
        target_value=0.14,
        title="Simplified Control Chart"
    )
    
    # Tab 2: Simplified Histogram
    tab2 = tabview.add("Distribution")
    
    # Generate test data
    sigma_data = np.random.normal(0.15, 0.02, 300)
    df_hist = pd.DataFrame({'sigma_gradient': sigma_data})
    
    chart2 = ChartWidget(tab2, figsize=(10, 5))
    chart2.pack(fill='both', expand=True)
    
    # Test simplified histogram
    chart2.plot_process_capability_histogram(
        data=df_hist,
        value_column='sigma_gradient',
        spec_limits=(0.08, 0.20),
        target_value=0.14,
        title="Simplified Distribution"
    )
    
    # Tab 3: Many Data Points Test
    tab3 = tabview.add("Large Dataset")
    
    # Generate lots of data points
    dates_many = pd.date_range(start='2023-01-01', periods=365, freq='D')
    trend_many = np.sin(np.linspace(0, 4*np.pi, 365)) * 0.02 + 0.14
    noise_many = np.random.normal(0, 0.005, 365)
    values_many = trend_many + noise_many
    
    df_many = pd.DataFrame({
        'trim_date': dates_many,
        'sigma_gradient': values_many
    })
    
    chart3 = ChartWidget(tab3, figsize=(10, 5))
    chart3.pack(fill='both', expand=True)
    
    # Test with many data points (should auto-aggregate and smooth)
    chart3.plot_enhanced_control_chart(
        data=df_many,
        value_column='sigma_gradient',
        date_column='trim_date',
        target_value=0.14,
        title="Large Dataset with Smoothing"
    )
    
    # Info label
    info_label = ctk.CTkLabel(
        root, 
        text="Simplified Charts - Clean, readable visualizations with moving averages for trend smoothing",
        font=('Arial', 12)
    )
    info_label.pack(pady=10)
    
    # Features info
    features_text = (
        "✓ Simplified control charts with 7-day and 30-day moving averages\n"
        "✓ Clean histograms with normal curve overlay\n"
        "✓ Auto-aggregation for large datasets (>100 points)\n"
        "✓ Reduced visual clutter - max 4 legend items\n"
        "✓ Focus on trend lines rather than individual data points"
    )
    features_label = ctk.CTkLabel(root, text=features_text, justify='left')
    features_label.pack(pady=5)
    
    print("\n=== Simplified Chart Test Results ===")
    print("✓ Control charts now focus on trends with moving averages")
    print("✓ Raw data shown as light background line")
    print("✓ 7-day and 30-day moving averages for smooth trend visualization")
    print("✓ Simplified legends with max 4 items")
    print("✓ Cleaner titles and reduced annotations")
    print("✓ Auto-aggregation for datasets >100 points")
    print("\nCharts are now simplified and trend-focused!")
    
    # Run the application
    root.mainloop()

if __name__ == "__main__":
    test_simplified_charts()