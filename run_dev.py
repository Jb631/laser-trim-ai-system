#!/usr/bin/env python3
"""
Development runner for Laser Trim Analyzer
Run this instead of building .exe during development
"""

import sys
import os
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Set development environment
os.environ["LASER_TRIM_DEV_MODE"] = "1"
os.environ["LTA_ENV"] = "development"

# Run the application
if __name__ == "__main__":
    from src.__main__ import main
    main()
    