#!/usr/bin/env python3
"""
Layout test to verify processing options remain visible when files are added.
"""

import tkinter as tk
from tkinter import ttk
import sys
sys.path.append('src')

def test_layout_fix():
    """Test the fixed layout to ensure processing options stay visible."""
    
    # Create test window
    root = tk.Tk()
    root.title("Layout Fix Test")
    root.geometry("800x600")
    
    # Create a test container similar to analysis page
    container = ttk.Frame(root)
    container.pack(fill='both', expand=True, padx=20, pady=20)
    
    # Title
    title_label = ttk.Label(container, text="Layout Fix Test", font=('Segoe UI', 16, 'bold'))
    title_label.pack(pady=(0, 20))
    
    # Main content frame
    content_frame = ttk.Frame(container)
    content_frame.pack(fill='both', expand=True)
    
    # Left column with grid layout (like the fixed analysis page)
    left_column = ttk.Frame(content_frame)
    left_column.pack(side='left', fill='both', expand=True, padx=(0, 10))
    
    # Configure grid weights
    left_column.grid_rowconfigure(0, weight=3)  # File section
    left_column.grid_rowconfigure(1, weight=0)  # Options (fixed)
    left_column.grid_rowconfigure(2, weight=0)  # Actions (fixed)
    left_column.grid_columnconfigure(0, weight=1)
    
    # Right column for reference
    right_column = ttk.Frame(content_frame)
    right_column.pack(side='right', fill='both', expand=True, padx=(10, 0))
    
    # Create sections similar to analysis page
    
    # 1. File section with fixed height
    file_container = ttk.Frame(left_column)
    file_container.grid(row=0, column=0, sticky='nsew', pady=(0, 10))
    file_container.grid_rowconfigure(1, weight=1)
    file_container.grid_columnconfigure(0, weight=1)
    
    # Drop zone area
    drop_frame = ttk.LabelFrame(file_container, text="Select Files", padding=15)
    drop_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
    ttk.Label(drop_frame, text="Drop files here or browse").pack()
    ttk.Button(drop_frame, text="Browse Files", command=lambda: add_test_files()).pack(pady=5)
    
    # File list with FIXED HEIGHT
    files_frame = ttk.LabelFrame(file_container, text="Selected Files", padding=10)
    files_frame.grid(row=1, column=0, sticky='nsew')
    files_frame.grid_rowconfigure(0, weight=1)
    files_frame.grid_columnconfigure(0, weight=1)
    
    # Scrollable canvas with fixed height
    canvas = tk.Canvas(files_frame, height=250, bg='white')
    scrollbar = ttk.Scrollbar(files_frame, orient='vertical', command=canvas.yview)
    file_list_frame = ttk.Frame(canvas)
    
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.create_window((0, 0), window=file_list_frame, anchor='nw')
    
    canvas.grid(row=0, column=0, sticky='nsew')
    scrollbar.grid(row=0, column=1, sticky='ns')
    
    def configure_scroll(e):
        canvas.configure(scrollregion=canvas.bbox('all'))
    file_list_frame.bind('<Configure>', configure_scroll)
    
    # 2. Processing options section (FIXED POSITION)
    options_frame = ttk.LabelFrame(left_column, text="Processing Options", padding=15)
    options_frame.grid(row=1, column=0, sticky='ew', pady=(0, 10))
    options_frame.grid_columnconfigure(0, weight=1)
    
    # Processing mode
    mode_frame = ttk.Frame(options_frame)
    mode_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
    
    processing_mode = tk.StringVar(value='detail')
    ttk.Label(mode_frame, text="Processing Mode:").pack(side='left')
    ttk.Radiobutton(mode_frame, text="Detail (with plots)", variable=processing_mode, value='detail').pack(side='left', padx=(10, 20))
    ttk.Radiobutton(mode_frame, text="Speed (no plots)", variable=processing_mode, value='speed').pack(side='left')
    
    # Feature toggles
    features_frame = ttk.Frame(options_frame)
    features_frame.grid(row=1, column=0, sticky='ew')
    
    enable_plots = tk.BooleanVar(value=True)
    enable_ml = tk.BooleanVar(value=True)
    enable_database = tk.BooleanVar(value=True)
    
    ttk.Checkbutton(features_frame, text="Generate plots", variable=enable_plots).pack(side='left', padx=(0, 20))
    ttk.Checkbutton(features_frame, text="ML predictions", variable=enable_ml).pack(side='left', padx=(0, 20))
    ttk.Checkbutton(features_frame, text="Save to database", variable=enable_database).pack(side='left')
    
    # 3. Action buttons (FIXED POSITION)
    action_frame = ttk.Frame(left_column)
    action_frame.grid(row=2, column=0, sticky='ew', pady=(0, 10))
    
    ttk.Button(action_frame, text="Start Analysis").pack(side='left', padx=(0, 10))
    ttk.Button(action_frame, text="Cancel").pack(side='left', padx=(0, 10))
    ttk.Button(action_frame, text="Clear All", command=lambda: clear_files()).pack(side='left')
    
    # 4. Results section
    results_frame = ttk.LabelFrame(right_column, text="Analysis Results", padding=15)
    results_frame.pack(fill='both', expand=True)
    ttk.Label(results_frame, text="Results will appear here...").pack(expand=True)
    
    # Test functions
    file_count = 0
    
    def add_test_files():
        nonlocal file_count
        # Add multiple test file entries to simulate loading many files
        for i in range(10):
            file_count += 1
            file_widget = ttk.Frame(file_list_frame, relief='solid', borderwidth=1)
            file_widget.pack(fill='x', pady=2, padx=5)
            
            ttk.Label(file_widget, text=f"Test_File_{file_count:03d}.xlsx", font=('Segoe UI', 9)).pack(side='left', padx=5, pady=2)
            ttk.Label(file_widget, text="Ready", foreground='blue', font=('Segoe UI', 8)).pack(side='right', padx=5, pady=2)
        
        # Force canvas update
        file_list_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox('all'))
        
        # Check if options are still visible
        check_visibility()
    
    def clear_files():
        for widget in file_list_frame.winfo_children():
            widget.destroy()
        canvas.configure(scrollregion=canvas.bbox('all'))
        check_visibility()
    
    def check_visibility():
        try:
            # Check if options frame is visible
            visible = options_frame.winfo_viewable()
            y_pos = options_frame.winfo_y()
            height = root.winfo_height()
            
            status = "VISIBLE ✓" if visible and y_pos < height - 100 else "HIDDEN ✗"
            print(f"Options visibility: {status} (y={y_pos}, window_height={height})")
            
            # Update status in GUI
            status_label.config(text=f"Processing Options Status: {status}")
            
        except Exception as e:
            print(f"Visibility check error: {e}")
    
    # Status indicator
    status_label = ttk.Label(container, text="Processing Options Status: VISIBLE ✓", 
                            font=('Segoe UI', 10, 'bold'), foreground='green')
    status_label.pack(pady=10)
    
    # Test buttons
    test_frame = ttk.Frame(container)
    test_frame.pack(pady=10)
    
    ttk.Button(test_frame, text="Add 10 Files", command=add_test_files).pack(side='left', padx=5)
    ttk.Button(test_frame, text="Add 50 Files", command=lambda: [add_test_files() for _ in range(5)]).pack(side='left', padx=5)
    ttk.Button(test_frame, text="Clear All Files", command=clear_files).pack(side='left', padx=5)
    ttk.Button(test_frame, text="Check Visibility", command=check_visibility).pack(side='left', padx=5)
    
    # Instructions
    instructions = ttk.Label(container, 
                            text="Test: Add files and verify that 'Processing Options' section remains visible below the file list",
                            font=('Segoe UI', 9), foreground='blue')
    instructions.pack(pady=5)
    
    # Initial visibility check
    root.after(500, check_visibility)
    
    print("Layout test started!")
    print("1. Click 'Add 10 Files' or 'Add 50 Files' to simulate file loading")
    print("2. Check that 'Processing Options' section remains visible")
    print("3. File list should scroll instead of pushing options off screen")
    
    root.mainloop()

if __name__ == "__main__":
    test_layout_fix() 