"""
Entry point for running the laser_trim_analyzer package as a module.

This allows running: python -m laser_trim_analyzer
"""

import sys
from pathlib import Path

# Add the src directory to the path so imports work
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

if __name__ == "__main__":
    # Import and run the main function from src/__main__.py
    from src.__main__ import main
    main() 