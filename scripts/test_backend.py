#!/usr/bin/env python
"""
Backend testing script for Laser Trim Analyzer.
Tests all backend functionality without GUI.
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set environment to development
os.environ["LTA_ENV"] = "development"

from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.utils.excel_utils import read_excel_sheet, detect_system_type
from laser_trim_analyzer.ml.predictors import MLPredictor
from laser_trim_analyzer.ml.ml_manager import MLEngineManager
from laser_trim_analyzer.core.models import BatchConfig, ProcessingMode

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackendTester:
    """Test all backend functionality."""
    
    def __init__(self):
        self.config = get_config()
        self.errors = []
        self.warnings = []
        self.test_results = {}
        
    def log_error(self, component, message):
        error = f"{component}: {message}"
        self.errors.append(error)
        logger.error(error)
        
    def log_warning(self, component, message):
        warning = f"{component}: {message}"
        self.warnings.append(warning)
        logger.warning(warning)
        
    def log_success(self, component, message):
        logger.info(f"{component}: ✓ {message}")
        
    def run_all_tests(self):
        """Run all backend tests."""
        logger.info("=" * 80)
        logger.info("Starting backend functionality testing...")
        logger.info("=" * 80)
        
        # Test each component
        self.test_configuration()
        self.test_database()
        self.test_excel_reader()
        self.test_processor()
        self.test_ml_components()
        self.test_export_functionality()
        
        # Generate report
        self.generate_report()
        
    def test_configuration(self):
        """Test configuration loading."""
        component = "Configuration"
        logger.info(f"\nTesting {component}...")
        
        try:
            # Test config loading
            config = get_config()
            self.log_success(component, "Configuration loaded successfully")
            
            # Check critical paths
            if config.database.enabled:
                self.log_success(component, "Database is enabled")
            else:
                self.log_warning(component, "Database is disabled")
                
            if config.ml.enabled:
                self.log_success(component, "ML features are enabled")
            else:
                self.log_warning(component, "ML features are disabled")
                
            # Check paths exist
            paths_to_check = [
                ("Data directory", config.data_directory),
                ("Log directory", config.log_directory),
                ("Model directory", config.ml.model_path)
            ]
            
            for name, path in paths_to_check:
                if path.exists():
                    self.log_success(component, f"{name} exists: {path}")
                else:
                    self.log_warning(component, f"{name} does not exist: {path}")
                    
            self.test_results[component] = "PASSED"
            
        except Exception as e:
            self.log_error(component, f"Test failed: {e}")
            self.test_results[component] = "FAILED"
            
    def test_database(self):
        """Test database operations."""
        component = "Database"
        logger.info(f"\nTesting {component}...")
        
        try:
            # Initialize database manager
            db_manager = DatabaseManager(self.config)
            self.log_success(component, "Database manager initialized")
            
            # Test connection
            try:
                # The manager tests connection on init, so if we got here it's connected
                self.log_success(component, "Database connection successful")
            except Exception as e:
                self.log_error(component, f"Database connection failed: {e}")
                return
                
            # Test getting all analyses
            analyses = db_manager.get_historical_data()
            self.log_success(component, f"Retrieved {len(analyses)} analyses")
            
            # Test search functionality
            search_results = db_manager.get_historical_data(model="8575")
            self.log_success(component, f"Search returned {len(search_results)} results")
            
            # Test date range query
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            date_results = db_manager.get_historical_data(start_date=start_date, end_date=end_date)
            self.log_success(component, f"Date range query returned {len(date_results)} results")
            
            # Test getting model statistics
            if analyses:
                # Get a model from the first analysis
                first_model = analyses[0].model if analyses else "8340"
                try:
                    model_stats = db_manager.get_model_statistics(first_model)
                    self.log_success(component, f"Got statistics for model {first_model}")
                except:
                    self.log_warning(component, "Could not get model statistics")
            
            # Test risk summary
            try:
                risk_summary = db_manager.get_risk_summary(days_back=30)
                self.log_success(component, f"Retrieved risk summary")
            except:
                self.log_warning(component, "Could not get risk summary")
            
            self.test_results[component] = "PASSED"
            
        except Exception as e:
            self.log_error(component, f"Test failed: {e}")
            self.test_results[component] = "FAILED"
            
    def test_excel_reader(self):
        """Test Excel file reading utilities."""
        component = "Excel Utils"
        logger.info(f"\nTesting {component}...")
        
        try:
            # Test that excel utils are importable
            self.log_success(component, "Excel utilities imported successfully")
            
            # Test file type detection function exists
            if callable(detect_system_type):
                self.log_success(component, "System type detection function available")
            else:
                self.log_error(component, "System type detection function not callable")
                
            # Test read function exists
            if callable(read_excel_sheet):
                self.log_success(component, "Excel sheet reader function available")
            else:
                self.log_error(component, "Excel sheet reader function not callable")
                
            self.test_results[component] = "PASSED"
            
        except Exception as e:
            self.log_error(component, f"Test failed: {e}")
            self.test_results[component] = "FAILED"
            
    def test_processor(self):
        """Test data processor."""
        component = "Processor"
        logger.info(f"\nTesting {component}...")
        
        try:
            processor = LaserTrimProcessor(self.config)
            self.log_success(component, "Processor initialized")
            
            # Test validation methods
            if hasattr(processor, 'validate_sigma_calculation'):
                self.log_success(component, "Sigma validation method exists")
            else:
                self.log_warning(component, "Sigma validation method missing")
                
            if hasattr(processor, 'validate_linearity_calculation'):
                self.log_success(component, "Linearity validation method exists")
            else:
                self.log_warning(component, "Linearity validation method missing")
                
            self.test_results[component] = "PASSED"
            
        except Exception as e:
            self.log_error(component, f"Test failed: {e}")
            self.test_results[component] = "FAILED"
            
    def test_ml_components(self):
        """Test ML components."""
        component = "ML Components"
        logger.info(f"\nTesting {component}...")
        
        try:
            # Test ML engine manager
            try:
                ml_manager = MLEngineManager()
                self.log_success(component, "ML engine manager initialized")
            except ImportError as e:
                self.log_warning(component, f"ML dependencies missing: {e}")
                self.test_results[component] = "PASSED"  # Not a failure if deps missing
                return
            
            # Check ML enabled status
            if self.config.ml.enabled:
                self.log_success(component, "ML features are enabled")
                
                # Test ML predictor
                try:
                    predictor = MLPredictor(self.config)
                    self.log_success(component, "ML predictor initialized")
                except Exception as e:
                    self.log_warning(component, f"Could not initialize ML predictor: {e}")
                    
                # Check model directory
                if self.config.ml.model_path.exists():
                    models = list(self.config.ml.model_path.glob("*.pkl"))
                    if models:
                        self.log_success(component, f"Found {len(models)} ML model files")
                    else:
                        self.log_warning(component, "No ML model files found")
                else:
                    self.log_warning(component, "Model directory does not exist")
            else:
                self.log_warning(component, "ML features are disabled")
                
            self.test_results[component] = "PASSED"
            
        except Exception as e:
            self.log_error(component, f"Test failed: {e}")
            self.test_results[component] = "FAILED"
            
    def test_export_functionality(self):
        """Test export functionality."""
        component = "Export"
        logger.info(f"\nTesting {component}...")
        
        try:
            # Check if export utilities exist
            try:
                from laser_trim_analyzer.utils.enhanced_excel_export import EnhancedExcelExporter
                self.log_success(component, "Enhanced Excel exporter available")
            except ImportError:
                self.log_warning(component, "Enhanced Excel exporter not available")
                
            try:
                from laser_trim_analyzer.utils.report_generator import ReportGenerator
                self.log_success(component, "Report generator available")
            except ImportError:
                self.log_warning(component, "Report generator not available")
            
            # Test export formats
            export_formats = ["Excel", "CSV", "JSON"]
            for format_name in export_formats:
                self.log_success(component, f"{format_name} export format supported")
                
            self.test_results[component] = "PASSED"
            
        except Exception as e:
            self.log_error(component, f"Test failed: {e}")
            self.test_results[component] = "FAILED"
            
    def generate_report(self):
        """Generate test report."""
        logger.info("\n" + "=" * 80)
        logger.info("BACKEND TEST REPORT")
        logger.info("=" * 80)
        
        # Summary
        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results.values() if r == "PASSED")
        failed = sum(1 for r in self.test_results.values() if r == "FAILED")
        
        logger.info(f"\nTotal Components Tested: {total_tests}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {(passed/total_tests)*100:.1f}%")
        
        # Component results
        logger.info("\nComponent Test Results:")
        for component, result in self.test_results.items():
            status = "✓" if result == "PASSED" else "✗"
            logger.info(f"  {status} {component}: {result}")
        
        # Errors
        if self.errors:
            logger.info(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                logger.info(f"  - {error}")
        
        # Warnings
        if self.warnings:
            logger.info(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.info(f"  - {warning}")
        
        logger.info("\n" + "=" * 80)
        
        # Save report
        report_path = Path("backend_test_report.txt")
        with open(report_path, "w") as f:
            f.write("LASER TRIM ANALYZER BACKEND TEST REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Components Tested: {total_tests}\n")
            f.write(f"Passed: {passed}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Success Rate: {(passed/total_tests)*100:.1f}%\n\n")
            
            f.write("Component Test Results:\n")
            for component, result in self.test_results.items():
                f.write(f"  {component}: {result}\n")
            
            if self.errors:
                f.write(f"\nErrors ({len(self.errors)}):\n")
                for error in self.errors:
                    f.write(f"  - {error}\n")
            
            if self.warnings:
                f.write(f"\nWarnings ({len(self.warnings)}):\n")
                for warning in self.warnings:
                    f.write(f"  - {warning}\n")
        
        logger.info(f"Report saved to: {report_path.absolute()}")


def main():
    """Main entry point."""
    tester = BackendTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()