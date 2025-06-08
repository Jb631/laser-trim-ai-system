#!/usr/bin/env python3
"""Test the import sequence to find where KeyboardInterrupt occurs"""

import sys
import os
from pathlib import Path

# Add src to path like run_dev.py does
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Set development environment
os.environ["LASER_TRIM_DEV_MODE"] = "1"

print("Starting import sequence test...\n")

def test_import(module_name):
    """Test importing a module and catch any errors"""
    try:
        print(f"Importing {module_name}...", end=" ")
        exec(f"import {module_name}")
        print("✓ SUCCESS")
        return True
    except KeyboardInterrupt:
        print("✗ KEYBOARD INTERRUPT!")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test imports in sequence
modules_to_test = [
    "laser_trim_analyzer",
    "laser_trim_analyzer.core",
    "laser_trim_analyzer.core.config",
    "laser_trim_analyzer.core.utils",
    "laser_trim_analyzer.core.constants",
    "laser_trim_analyzer.database",
    "laser_trim_analyzer.database.models",
    "laser_trim_analyzer.database.manager",
    "laser_trim_analyzer.gui",
    "laser_trim_analyzer.gui.main_window",
]

print("Testing module imports in sequence...")
for module in modules_to_test:
    if not test_import(module):
        print(f"\nFailed at module: {module}")
        break

print("\nDone.")