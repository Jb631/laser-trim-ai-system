"""
Database Package for Laser Trim AI System

This package provides database functionality for storing and analyzing
historical laser trim data, enabling continuous improvement through
machine learning and trend analysis.

Components:
- DatabaseManager: Core database operations
- HistoricalAnalyzer: Advanced historical data analysis
- TrendReporter: Comprehensive excel_reporter with visualizations
- DataMigrator: Import/export and data migration tools

Author: Laser Trim AI System
Date: 2024
"""

from database.database_manager import DatabaseManager
from database.historical_analyzer import HistoricalAnalyzer
from database.trend_reporter import TrendReporter
from database.data_migrator import DataMigrator

__all__ = [
    'DatabaseManager',
    'HistoricalAnalyzer',
    'TrendReporter',
    'DataMigrator'
]

# Version info
__version__ = '1.0.0'