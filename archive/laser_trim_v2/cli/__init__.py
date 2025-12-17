"""
Command Line Interface for Laser Trim Analyzer.

Provides a comprehensive CLI for QA specialists to run analyses,
train models, generate reports, and query data without the GUI.
"""

from laser_trim_analyzer.cli.commands import cli

__all__ = ['cli']

__version__ = '2.0.0'