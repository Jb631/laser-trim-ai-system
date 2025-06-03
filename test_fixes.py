#!/usr/bin/env python3
"""
Quick test script to verify the fixes for AlertBanner glitching 
and processing options visibility.
"""

import tkinter as tk
from tkinter import ttk
import time

# Test the SimpleAlertStack
def test_simple_alerts():
    """Test the simplified alert system."""
    root = tk.Tk()
    root.title("Alert System Test")
    root.geometry("600x400")
    
    # Import the SimpleAlertStack from the updated analysis_page
    import sys
    sys.path.append('src')
    from laser_trim_analyzer.gui.pages.analysis_page import SimpleAlertStack
    
    # Create alert stack
    alert_stack = SimpleAlertStack(root)
    alert_stack.pack(fill='x', padx=20, pady=20)
    
    # Test buttons
    button_frame = ttk.Frame(root)
    button_frame.pack(pady=20)
    
    def add_info_alert():
        alert_stack.add_alert(
            alert_type='info',
            title='Info Test',
            message='This is a test info alert without glitching animations.',
            auto_dismiss=3
        )
    
    def add_warning_alert():
        alert_stack.add_alert(
            alert_type='warning',
            title='Warning Test',
            message='This is a test warning alert.',
            dismissible=True
        )
    
    def add_error_alert():
        alert_stack.add_alert(
            alert_type='error',
            title='Error Test',
            message='This is a test error alert with action buttons.',
            actions=[
                {'text': 'Retry', 'command': lambda: print('Retry clicked')},
                {'text': 'Details', 'command': lambda: print('Details clicked')}
            ]
        )
    
    def add_success_alert():
        alert_stack.add_alert(
            alert_type='success',
            title='Success Test',
            message='This is a test success alert.',
            auto_dismiss=5
        )
    
    def clear_all():
        alert_stack.clear_all()
    
    # Test buttons
    ttk.Button(button_frame, text="Add Info Alert", command=add_info_alert).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Add Warning Alert", command=add_warning_alert).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Add Error Alert", command=add_error_alert).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Add Success Alert", command=add_success_alert).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Clear All", command=clear_all).pack(side='left', padx=5)
    
    # Processing options test
    options_frame = ttk.LabelFrame(root, text="Processing Options Test", padding=15)
    options_frame.pack(fill='x', padx=20, pady=20)
    
    processing_mode = tk.StringVar(value='detail')
    enable_plots = tk.BooleanVar(value=True)
    enable_ml = tk.BooleanVar(value=True)
    enable_database = tk.BooleanVar(value=True)
    
    # Processing mode
    mode_frame = ttk.Frame(options_frame)
    mode_frame.pack(fill='x', pady=(0, 10))
    
    ttk.Label(mode_frame, text="Processing Mode:").pack(side='left')
    ttk.Radiobutton(mode_frame, text="Detail (with plots)", variable=processing_mode, value='detail').pack(side='left', padx=(10, 20))
    ttk.Radiobutton(mode_frame, text="Speed (no plots)", variable=processing_mode, value='speed').pack(side='left')
    
    # Feature toggles
    features_frame = ttk.Frame(options_frame)
    features_frame.pack(fill='x')
    
    ttk.Checkbutton(features_frame, text="Generate plots", variable=enable_plots).pack(side='left', padx=(0, 20))
    ttk.Checkbutton(features_frame, text="ML predictions", variable=enable_ml).pack(side='left', padx=(0, 20))
    ttk.Checkbutton(features_frame, text="Save to database", variable=enable_database).pack(side='left')
    
    # Status label
    status_label = ttk.Label(root, text="Test running - check that alerts don't cause glitching and options remain visible", foreground='blue')
    status_label.pack(pady=10)
    
    # Auto-test alerts
    def auto_test():
        add_info_alert()
        root.after(2000, add_warning_alert)
        root.after(4000, add_success_alert)
        root.after(6000, lambda: status_label.config(text="Auto-test complete - Options should still be visible above"))
    
    root.after(1000, auto_test)
    
    print("Alert system test started. Check that:")
    print("1. Alerts appear without screen glitching")
    print("2. Processing options remain visible at all times")
    print("3. Alerts can be dismissed properly")
    print("4. Multiple alerts don't cause performance issues")
    
    root.mainloop()

if __name__ == "__main__":
    test_simple_alerts() 