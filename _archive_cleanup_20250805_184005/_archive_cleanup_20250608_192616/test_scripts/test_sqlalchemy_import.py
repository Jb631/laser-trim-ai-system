#!/usr/bin/env python
"""Test SQLAlchemy import in isolation"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    print("Attempting to import sqlalchemy...")
    import sqlalchemy
    print(f"SQLAlchemy imported successfully! Version: {sqlalchemy.__version__}")
    
    print("\nTesting basic SQLAlchemy imports...")
    from sqlalchemy import create_engine, Column, Integer, String
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    print("All basic imports successful!")
    
except KeyboardInterrupt:
    print("\nKeyboardInterrupt caught during import!")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"\nError during import: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()