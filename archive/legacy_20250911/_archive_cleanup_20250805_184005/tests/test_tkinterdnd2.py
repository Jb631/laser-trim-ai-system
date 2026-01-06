#!/usr/bin/env python3
"""Test script to verify tkinterdnd2 installation and functionality"""

import sys
import tkinter as tk

# Test basic import
try:
    import tkinterdnd2
    print("✓ tkinterdnd2 module imported successfully")
    print(f"  Version info: {tkinterdnd2.__file__}")
except ImportError as e:
    print(f"✗ Failed to import tkinterdnd2: {e}")
    print("\nTo install tkinterdnd2, run:")
    print("  pip install tkinterdnd2")
    sys.exit(1)

# Test TkinterDnD window creation
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    print("✓ TkinterDnD and DND_FILES imported successfully")
    
    # Try to create a window
    try:
        root = TkinterDnD.Tk()
        print("✓ TkinterDnD window created successfully")
        
        # Test drop target registration
        try:
            test_frame = tk.Frame(root)
            test_frame.pack()
            test_frame.drop_target_register(DND_FILES)
            print("✓ Drop target registration successful")
            
            # Clean up
            root.destroy()
            print("\n✅ All tests passed! tkinterdnd2 is properly installed and functional.")
            
        except Exception as e:
            print(f"✗ Drop target registration failed: {e}")
            root.destroy()
            
    except Exception as e:
        print(f"✗ Failed to create TkinterDnD window: {e}")
        print("\nThis might be due to missing system dependencies.")
        print("On Windows, tkinterdnd2 should work out of the box.")
        print("On Linux, you may need to install tkdnd2:")
        print("  sudo apt-get install tkdnd")
        
except Exception as e:
    print(f"✗ Failed to import TkinterDnD components: {e}")