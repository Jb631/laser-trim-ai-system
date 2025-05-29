"""
Excel report generation for laser trim analysis.
"""

from excel_reporter.excel_reporter import ExcelReporter
from excel_reporter.excel_config import ExcelReportConfig

# Import adapter for GUI compatibility
try:
    from excel_reporter.excel_report_adapter import ExcelReportGenerator
except ImportError:
    # Fallback
    ExcelReportGenerator = ExcelReporter

__all__ = ['ExcelReporter', 'ExcelReportConfig', 'ExcelReportGenerator']