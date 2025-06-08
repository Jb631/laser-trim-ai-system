#!/usr/bin/env python3
"""Test script to verify UI fixes for home page and metric cards."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.laser_trim_analyzer.gui.main_window import LaserTrimAnalyzerApp
import customtkinter as ctk
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_home_page():
    """Test the home page fixes."""
    print("Testing Home Page UI fixes...")
    
    # Create app
    app = LaserTrimAnalyzerApp()
    
    # Navigate to home page
    app._show_page('home')
    
    # Check if home page loaded without errors
    home_page = app.pages.get('home')
    if home_page:
        print("✓ Home page loaded successfully")
        
        # Check metric cards
        if hasattr(home_page, 'stat_cards'):
            print(f"✓ Found {len(home_page.stat_cards)} metric cards")
            for name, card in home_page.stat_cards.items():
                print(f"  - {name}: {type(card).__name__}")
                # Verify it's using CustomTkinter
                if 'ctk' in str(type(card).__module__):
                    print(f"    ✓ Using CustomTkinter")
                else:
                    print(f"    ✗ Not using CustomTkinter!")
        
        # Check activity list
        if hasattr(home_page, 'activity_list'):
            print("✓ Activity list widget found")
            if isinstance(home_page.activity_list, ctk.CTkTextbox):
                print("  ✓ Using CTkTextbox")
            else:
                print("  ✗ Not using CTkTextbox!")
    else:
        print("✗ Failed to load home page")
    
    # Run for a moment to allow UI to render
    app.after(2000, app.quit)
    app.mainloop()

def test_other_pages():
    """Test other pages for metric card fixes."""
    print("\nTesting other pages...")
    
    pages_to_test = [
        'single_file',
        'batch_processing', 
        'ml_tools',
        'historical'
    ]
    
    app = LaserTrimAnalyzerApp()
    
    for page_name in pages_to_test:
        print(f"\nChecking {page_name} page...")
        app._show_page(page_name)
        
        page = app.pages.get(page_name)
        if page:
            print(f"✓ {page_name} page loaded")
            
            # Look for metric cards
            for attr_name in dir(page):
                attr = getattr(page, attr_name)
                if 'MetricCard' in str(type(attr).__name__):
                    print(f"  Found MetricCard: {attr_name}")
                    if 'ctk' in str(type(attr).__module__):
                        print(f"    ✓ Using CustomTkinter")
                    else:
                        print(f"    ✗ Not using CustomTkinter!")
        else:
            print(f"✗ Failed to load {page_name}")
    
    app.after(1000, app.quit)
    app.mainloop()

if __name__ == "__main__":
    print("UI Fix Test Script")
    print("=" * 50)
    
    try:
        test_home_page()
        test_other_pages()
        print("\n✓ All tests completed")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()