#!/usr/bin/env python3
"""
Test script to verify the single file page UI fix for duplicate windows issue.
"""

import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import sys
import os

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from laser_trim_analyzer.gui.pages.single_file_page import SingleFilePage


def test_single_file_page():
    """Test the single file page to ensure no duplicate windows."""
    
    # Create a test window
    root = ctk.CTk()
    root.title("Single File Page UI Test")
    root.geometry("1200x800")
    
    # Create a mock main window object with minimal required attributes
    class MockMainWindow:
        def __init__(self):
            self.db_manager = None  # SingleFilePage expects this
    
    mock_main = MockMainWindow()
    
    # Create the single file page
    page = SingleFilePage(root, main_window=mock_main)
    page.pack(fill='both', expand=True)
    
    # Add test buttons to simulate workflow
    test_frame = ctk.CTkFrame(root)
    test_frame.pack(side='bottom', fill='x', padx=10, pady=10)
    
    def simulate_results():
        """Simulate showing analysis results."""
        # This would normally happen when analysis completes
        # For testing, we'll manually trigger the display changes
        try:
            # Hide empty state and show analysis display
            page.empty_state_frame.pack_forget()
            page.analysis_display.pack(fill='both', expand=True, padx=15, pady=(0, 15))
            messagebox.showinfo("Test", "Simulated results display - check for duplicate windows")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to simulate results: {e}")
    
    def clear_results():
        """Clear the results display."""
        try:
            page._clear_results()
            messagebox.showinfo("Test", "Results cleared - empty state should be visible")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear results: {e}")
    
    ctk.CTkButton(
        test_frame,
        text="Simulate Results Display",
        command=simulate_results
    ).pack(side='left', padx=5)
    
    ctk.CTkButton(
        test_frame,
        text="Clear Results",
        command=clear_results
    ).pack(side='left', padx=5)
    
    ctk.CTkButton(
        test_frame,
        text="Exit",
        command=root.quit
    ).pack(side='right', padx=5)
    
    # Instructions
    instructions = ctk.CTkLabel(
        test_frame,
        text="Click 'Simulate Results Display' to test showing results, then 'Clear Results' to reset",
        font=ctk.CTkFont(size=12)
    )
    instructions.pack(side='left', padx=20)
    
    root.mainloop()


if __name__ == "__main__":
    print("Testing Single File Page UI Fix...")
    print("This test verifies that the duplicate windows issue has been resolved.")
    print("You should see only one window at a time - either the empty state or the results display.")
    print()
    
    test_single_file_page()