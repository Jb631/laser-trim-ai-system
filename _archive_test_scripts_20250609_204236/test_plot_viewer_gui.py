#!/usr/bin/env python3
"""Simple test script to verify plot viewer GUI functionality."""

import customtkinter as ctk
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from laser_trim_analyzer.gui.widgets.plot_viewer import PlotViewerWidget


def test_plot_viewer():
    """Test the plot viewer widget."""
    # Create test window
    root = ctk.CTk()
    root.title("Plot Viewer Test")
    root.geometry("900x700")
    
    # Create plot viewer
    plot_viewer = PlotViewerWidget(root)
    plot_viewer.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Look for a test plot
    test_plots = list(Path(".").rglob("*.png"))
    if test_plots:
        # Try to load the first plot found
        test_plot = test_plots[0]
        print(f"Loading test plot: {test_plot}")
        
        if plot_viewer.load_plot(test_plot):
            print("✓ Plot loaded successfully")
        else:
            print("✗ Failed to load plot")
    else:
        print("No test plots found. Widget created but no plot loaded.")
        # Show empty widget
        plot_viewer._show_error("No test plots found in directory")
    
    # Add test button
    test_btn = ctk.CTkButton(
        root,
        text="Test Zoom",
        command=lambda: [plot_viewer._zoom_in(), print("Zoomed in")]
    )
    test_btn.pack(pady=5)
    
    # Run the GUI
    print("Starting GUI test...")
    root.mainloop()


if __name__ == "__main__":
    try:
        test_plot_viewer()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()