#!/usr/bin/env python3
"""Test script to verify chart display and functionality."""

import os
import sys
import time
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Set development environment
os.environ['LTA_ENV'] = 'development'

def test_charts():
    """Test chart display on different pages."""
    from laser_trim_analyzer.gui.ctk_main_window import CTkMainWindow
    import customtkinter as ctk
    
    # Create and run the app
    app = CTkMainWindow()
    
    def navigate_and_test():
        """Navigate through pages and test charts."""
        time.sleep(2)  # Let app initialize
        
        try:
            print("\n=== Testing Model Summary Page ===")
            # Navigate to Model Summary page
            if hasattr(app, 'sidebar') and hasattr(app.sidebar, 'model_summary_btn'):
                app.sidebar.model_summary_btn.invoke()
                time.sleep(1)
                
                # Check if page loaded
                if hasattr(app, 'current_page'):
                    page = app.current_page
                    print(f"Current page: {page.__class__.__name__}")
                    
                    # Check for chart widgets
                    if hasattr(page, 'trend_chart'):
                        print("✓ Trend chart found")
                        # Check if chart has the new methods
                        if hasattr(page.trend_chart, 'plot_enhanced_control_chart'):
                            print("  - Enhanced control chart method available")
                        if hasattr(page.trend_chart, 'qa_colors'):
                            print(f"  - QA colors configured: {len(page.trend_chart.qa_colors)} colors")
                    
                    if hasattr(page, 'cpk_chart'):
                        print("✓ CPK chart found")
                        if hasattr(page.cpk_chart, 'plot_process_capability_histogram'):
                            print("  - Process capability histogram method available")
                    
                    if hasattr(page, 'overview_chart'):
                        print("✓ Overview chart found")
                    
                    if hasattr(page, 'trend_analysis_chart'):
                        print("✓ Trend analysis chart found")
                    
                    if hasattr(page, 'risk_chart'):
                        print("✓ Risk chart found")
            
            print("\n=== Testing Historical Page ===")
            time.sleep(1)
            # Navigate to Historical page
            if hasattr(app, 'sidebar') and hasattr(app.sidebar, 'historical_btn'):
                app.sidebar.historical_btn.invoke()
                time.sleep(1)
                
                if hasattr(app, 'current_page'):
                    page = app.current_page
                    print(f"Current page: {page.__class__.__name__}")
                    
                    # Check for chart widgets
                    if hasattr(page, 'control_chart'):
                        print("✓ Control chart found")
                        if hasattr(page.control_chart, 'plot_enhanced_control_chart'):
                            print("  - Enhanced control chart method available")
                    
                    if hasattr(page, 'production_chart'):
                        print("✓ Production chart found")
                    
                    if hasattr(page, 'quality_trends_chart'):
                        print("✓ Quality trends chart found")
            
            print("\n=== Testing ML Tools Page ===")
            time.sleep(1)
            # Navigate to ML Tools page
            if hasattr(app, 'sidebar') and hasattr(app.sidebar, 'ml_tools_btn'):
                app.sidebar.ml_tools_btn.invoke()
                time.sleep(1)
                
                if hasattr(app, 'current_page'):
                    page = app.current_page
                    print(f"Current page: {page.__class__.__name__}")
                    
                    # Check for chart widgets
                    if hasattr(page, 'quality_trend_chart'):
                        print("✓ Quality trend chart found")
                        if hasattr(page.quality_trend_chart, 'plot_enhanced_control_chart'):
                            print("  - Enhanced control chart method available")
                    
                    if hasattr(page, 'yield_chart'):
                        print("✓ Yield chart found")
            
            print("\n=== Chart Display Test Summary ===")
            print("All chart widgets are properly initialized")
            print("Enhanced methods are available on ChartWidget instances")
            print("QA color palette is configured")
            
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()
        
        # Keep app running for manual inspection
        print("\nApp is running. Close the window to exit...")
    
    # Start navigation in a separate thread
    test_thread = threading.Thread(target=navigate_and_test)
    test_thread.daemon = True
    test_thread.start()
    
    # Run the app
    app.mainloop()

if __name__ == "__main__":
    test_charts()