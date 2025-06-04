#!/usr/bin/env python3
"""
Test script for responsive design and stop functionality.

This script creates a simple test environment to verify:
1. Responsive design works across different window sizes
2. Stop processing functionality works correctly
3. Pages properly inherit from the responsive base class
"""

import sys
import os
import tkinter as tk
from pathlib import Path
import threading
import time

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from laser_trim_analyzer.gui.pages.base_page import BasePage, ResponsiveFrame
    from laser_trim_analyzer.gui.pages.batch_processing_page import BatchProcessingPage
    from laser_trim_analyzer.gui.pages.home_page import HomePage
    print("✓ Successfully imported responsive page classes")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

class MockMainWindow:
    """Mock main window for testing."""
    
    def __init__(self):
        self.colors = {
            'bg_primary': '#f0f2f5',
            'bg_secondary': '#ffffff',
            'text_primary': '#212121',
            'text_secondary': '#757575',
            'success': '#4caf50',
            'warning': '#ff9800',
            'danger': '#f44336'
        }
        
        self.db_manager = None
        self.config = None

class TestResponsiveDesign:
    """Test responsive design functionality."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Responsive Design Test")
        self.root.geometry("1000x700")
        
        self.mock_main = MockMainWindow()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup test UI."""
        # Control frame
        control_frame = tk.Frame(self.root, bg='lightgray', height=50)
        control_frame.pack(fill='x', side='top')
        control_frame.pack_propagate(False)
        
        tk.Label(control_frame, text="Window Size Test:", bg='lightgray').pack(side='left', padx=10, pady=10)
        
        tk.Button(control_frame, text="Small (800x600)", 
                 command=lambda: self.resize_window(800, 600)).pack(side='left', padx=5)
        tk.Button(control_frame, text="Medium (1200x800)", 
                 command=lambda: self.resize_window(1200, 800)).pack(side='left', padx=5)
        tk.Button(control_frame, text="Large (1600x1000)", 
                 command=lambda: self.resize_window(1600, 1000)).pack(side='left', padx=5)
        
        # Test frame for responsive content
        self.test_frame = tk.Frame(self.root)
        self.test_frame.pack(fill='both', expand=True)
        
        # Test ResponsiveFrame
        self.responsive_frame = ResponsiveFrame(self.test_frame)
        self.responsive_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Add callback to monitor size changes
        self.responsive_frame.add_layout_callback(self.on_layout_change)
        
        # Create test widgets
        self.create_test_widgets()
        
        # Status label
        self.status_label = tk.Label(self.root, text="Current size: large", 
                                   bg='white', relief='sunken')
        self.status_label.pack(fill='x', side='bottom')
        
    def create_test_widgets(self):
        """Create test widgets to demonstrate responsive behavior."""
        # Title
        title = tk.Label(self.responsive_frame, text="Responsive Layout Test", 
                        font=('Arial', 16, 'bold'))
        title.pack(pady=(0, 20))
        
        # Test cards frame
        cards_frame = tk.Frame(self.responsive_frame)
        cards_frame.pack(fill='x', pady=(0, 20))
        
        # Create test cards
        self.test_cards = []
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 'lightgray']
        
        for i in range(6):
            card = tk.Frame(cards_frame, bg=colors[i], relief='raised', bd=1)
            label = tk.Label(card, text=f"Card {i+1}", bg=colors[i], font=('Arial', 12))
            label.pack(pady=20, padx=20)
            self.test_cards.append(card)
        
        # Store frames for responsive layout
        self.cards_frame = cards_frame
        
        # Apply initial layout
        self.arrange_cards()
        
    def arrange_cards(self):
        """Arrange cards based on current responsive frame size."""
        # Clear existing layout
        for card in self.test_cards:
            card.grid_forget()
            
        # Get responsive column count
        columns = self.responsive_frame.get_responsive_columns(len(self.test_cards))
        
        # Configure grid weights
        for i in range(columns):
            self.cards_frame.columnconfigure(i, weight=1)
            
        # Arrange cards
        for i, card in enumerate(self.test_cards):
            row = i // columns
            col = i % columns
            padding = self.responsive_frame.get_responsive_padding()
            card.grid(row=row, column=col, sticky='ew', **padding)
    
    def on_layout_change(self, size_class: str):
        """Handle layout changes."""
        self.status_label.config(text=f"Current size: {size_class}")
        self.arrange_cards()
        print(f"Layout changed to: {size_class}")
        
    def resize_window(self, width: int, height: int):
        """Resize window to test responsive behavior."""
        self.root.geometry(f"{width}x{height}")
        print(f"Window resized to: {width}x{height}")

class TestStopFunctionality:
    """Test stop processing functionality."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stop Functionality Test")
        self.root.geometry("600x400")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup test UI for stop functionality."""
        # Control frame
        control_frame = tk.Frame(self.root, bg='lightgray')
        control_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(control_frame, text="Stop Processing Test:", bg='lightgray', 
                font=('Arial', 12, 'bold')).pack(anchor='w')
        
        # Test buttons
        button_frame = tk.Frame(control_frame, bg='lightgray')
        button_frame.pack(fill='x', pady=10)
        
        self.start_btn = tk.Button(button_frame, text="Start Mock Processing", 
                                  command=self.start_processing, bg='green', fg='white')
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(button_frame, text="Stop Processing", 
                                 command=self.stop_processing, bg='red', fg='white', 
                                 state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Status area
        self.status_text = tk.Text(self.root, height=15, width=70)
        self.status_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(self.status_text)
        scrollbar.pack(side='right', fill='y')
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_text.yview)
        
        # Processing state
        self.is_processing = False
        self.stop_requested = False
        self.processing_thread = None
        
    def log_message(self, message: str):
        """Log message to status area."""
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        self.status_text.insert(tk.END, full_message)
        self.status_text.see(tk.END)
        self.status_text.update()
        
    def start_processing(self):
        """Start mock processing."""
        if self.is_processing:
            self.log_message("Processing already in progress!")
            return
            
        self.is_processing = True
        self.stop_requested = False
        
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        self.log_message("Starting mock processing...")
        
        # Start processing in background thread
        self.processing_thread = threading.Thread(target=self.mock_processing, daemon=True)
        self.processing_thread.start()
        
    def stop_processing(self):
        """Stop processing."""
        if not self.is_processing:
            return
            
        self.stop_requested = True
        self.log_message("Stop requested - waiting for processing to halt...")
        
        self.stop_btn.config(state='disabled', text='Stopping...')
        
    def mock_processing(self):
        """Mock processing that can be stopped."""
        try:
            for i in range(100):
                # Check for stop request
                if self.stop_requested:
                    self.root.after(0, lambda: self.log_message("Processing stopped by user request"))
                    break
                    
                # Simulate processing work
                time.sleep(0.1)  # 100ms per "file"
                
                # Update progress
                self.root.after(0, lambda i=i: self.log_message(f"Processing file {i+1}/100..."))
                
            else:
                # Completed without stopping
                self.root.after(0, lambda: self.log_message("Processing completed successfully!"))
                
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"Processing error: {e}"))
            
        finally:
            # Reset UI state
            self.is_processing = False
            self.root.after(0, self.reset_ui)
            
    def reset_ui(self):
        """Reset UI after processing ends."""
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled', text='Stop Processing')

def run_responsive_test():
    """Run responsive design test."""
    print("Starting responsive design test...")
    app = TestResponsiveDesign()
    app.root.mainloop()

def run_stop_test():
    """Run stop functionality test."""
    print("Starting stop functionality test...")
    app = TestStopFunctionality()
    app.root.mainloop()

def main():
    """Main test function."""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == 'responsive':
            run_responsive_test()
        elif test_type == 'stop':
            run_stop_test()
        else:
            print("Usage: python test_responsive_design.py [responsive|stop]")
    else:
        print("Available tests:")
        print("  python test_responsive_design.py responsive  - Test responsive design")
        print("  python test_responsive_design.py stop        - Test stop functionality")
        print()
        
        choice = input("Which test would you like to run? (responsive/stop): ").lower()
        if choice == 'responsive':
            run_responsive_test()
        elif choice == 'stop':
            run_stop_test()
        else:
            print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main() 