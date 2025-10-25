#!/usr/bin/env python3
"""
Fix Risk Category Enum Values

This script updates any lowercase risk category values in the database
to their proper uppercase enum values.

Usage:
    python scripts/fix_risk_category_enum.py
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from sqlalchemy import create_engine, text
    from laser_trim_analyzer.core.config import get_config
    from laser_trim_analyzer.database.models import Base
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you've run: pip install -e .")
    sys.exit(1)


def fix_risk_categories():
    """Update risk category values to proper case."""
    config = get_config()
    
    print("Fixing Risk Category Enum Values")
    print("=" * 80)
    
    # Create engine
    db_path = os.path.expandvars(str(config.database.path))
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Map lowercase to uppercase values
    updates = [
        ("UPDATE track_results SET risk_category = 'HIGH' WHERE risk_category = 'high'", "high -> HIGH"),
        ("UPDATE track_results SET risk_category = 'MEDIUM' WHERE risk_category = 'medium'", "medium -> MEDIUM"),
        ("UPDATE track_results SET risk_category = 'LOW' WHERE risk_category = 'low'", "low -> LOW"),
        ("UPDATE track_results SET risk_category = 'UNKNOWN' WHERE risk_category = 'unknown'", "unknown -> UNKNOWN"),
        # Also check prediction_risk table if exists
        ("UPDATE prediction_risk SET predicted_risk_category = 'HIGH' WHERE predicted_risk_category = 'high'", "high -> HIGH (predictions)"),
        ("UPDATE prediction_risk SET predicted_risk_category = 'MEDIUM' WHERE predicted_risk_category = 'medium'", "medium -> MEDIUM (predictions)"),
        ("UPDATE prediction_risk SET predicted_risk_category = 'LOW' WHERE predicted_risk_category = 'low'", "low -> LOW (predictions)"),
        ("UPDATE prediction_risk SET predicted_risk_category = 'UNKNOWN' WHERE predicted_risk_category = 'unknown'", "unknown -> UNKNOWN (predictions)"),
    ]
    
    with engine.connect() as conn:
        for query, description in updates:
            try:
                result = conn.execute(text(query))
                if result.rowcount > 0:
                    print(f"✓ Updated {result.rowcount} rows: {description}")
                conn.commit()
            except Exception as e:
                # Table might not exist, that's okay
                if "no such table" not in str(e).lower():
                    print(f"⚠ Error updating {description}: {e}")
    
    print("\n✓ Risk category enum values fixed")


if __name__ == "__main__":
    fix_risk_categories()