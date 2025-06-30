#!/usr/bin/env python
"""
Comprehensive GUI testing script for Laser Trim Analyzer.
Tests all pages, buttons, and functionality.
"""

import sys
import os
import time
import logging
from pathlib import Path
import threading
import tkinter as tk
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set environment to development
os.environ["LTA_ENV"] = "development"

from laser_trim_analyzer.gui.main_window import MainWindow
from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.database.manager import DatabaseManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GUITester:
    """Automated GUI testing for Laser Trim Analyzer."""
    
    def __init__(self):
        self.config = get_config()
        self.app = None
        self.errors = []
        self.warnings = []
        self.test_results = {}
        
    def log_error(self, page, message):
        """Log an error for a specific page."""
        error = f"{page}: {message}"
        self.errors.append(error)
        logger.error(error)
        
    def log_warning(self, page, message):
        """Log a warning for a specific page."""
        warning = f"{page}: {message}"
        self.warnings.append(warning)
        logger.warning(warning)
        
    def log_success(self, page, message):
        """Log a success for a specific page."""
        logger.info(f"{page}: ✓ {message}")
        
    def start_app(self):
        """Start the application."""
        logger.info("Starting Laser Trim Analyzer...")
        self.app = MainWindow(self.config)
        
        # Create a thread to run tests after app starts
        def run_tests_after_delay():
            time.sleep(2)  # Wait for app to initialize
            if self.app.ctk_window:
                self.app.ctk_window.after(100, self.run_all_tests)
            else:
                logger.error("CTK window not initialized")
        
        test_thread = threading.Thread(target=run_tests_after_delay)
        test_thread.daemon = True
        test_thread.start()
        
        # Run the app
        self.app.run()
        
    def run_all_tests(self):
        """Run all page tests."""
        logger.info("=" * 80)
        logger.info("Starting comprehensive page testing...")
        logger.info("=" * 80)
        
        try:
            # Test each page
            self.test_home_page()
            self.test_single_file_page()
            self.test_batch_processing_page()
            self.test_multi_track_page()
            self.test_final_test_comparison_page()
            self.test_model_summary_page()
            self.test_historical_page()
            self.test_ml_tools_page()
            self.test_ai_insights_page()
            self.test_settings_page()
            
            # Test cross-page functionality
            self.test_navigation()
            self.test_database_operations()
            
            # Generate report
            self.generate_report()
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}", exc_info=True)
        finally:
            # Close app after tests
            self.app.ctk_window.after(2000, self.app.ctk_window.quit)
    
    def test_home_page(self):
        """Test Home page functionality."""
        page_name = "Home Page"
        logger.info(f"\nTesting {page_name}...")
        
        try:
            # Navigate to home page
            self.app.ctk_window._show_page("home")
            time.sleep(0.5)
            
            # Get the page
            home_page = self.app.ctk_window.pages.get("home")
            if not home_page:
                self.log_error(page_name, "Page not found in pages dictionary")
                return
                
            self.log_success(page_name, "Page loaded successfully")
            
            # Test page components
            if hasattr(home_page, 'overview_frame'):
                self.log_success(page_name, "Overview frame exists")
            else:
                self.log_error(page_name, "Overview frame missing")
                
            if hasattr(home_page, 'quick_actions_frame'):
                self.log_success(page_name, "Quick actions frame exists")
            else:
                self.log_error(page_name, "Quick actions frame missing")
                
            # Test recent analyses display
            if hasattr(home_page, 'refresh_recent_analyses'):
                home_page.refresh_recent_analyses()
                self.log_success(page_name, "Recent analyses refresh successful")
            else:
                self.log_warning(page_name, "No refresh_recent_analyses method")
                
            self.test_results[page_name] = "PASSED"
            
        except Exception as e:
            self.log_error(page_name, f"Test failed: {e}")
            self.test_results[page_name] = "FAILED"
    
    def test_single_file_page(self):
        """Test Single File Analysis page."""
        page_name = "Single File Page"
        logger.info(f"\nTesting {page_name}...")
        
        try:
            # Navigate to page
            self.app.ctk_window._show_page("single_file")
            time.sleep(0.5)
            
            page = self.app.ctk_window.pages.get("single_file")
            if not page:
                # Try to create it
                if "single_file" in self.app.ctk_window.page_classes:
                    page_class = self.app.ctk_window.page_classes["single_file"]
                    if page_class:
                        self.log_warning(page_name, "Page not loaded, but class exists")
                    else:
                        self.log_error(page_name, "Page class is None")
                else:
                    self.log_error(page_name, "Page not in page_classes")
                return
                
            self.log_success(page_name, "Page loaded successfully")
            
            # Test components
            if hasattr(page, 'file_input_frame'):
                self.log_success(page_name, "File input frame exists")
            else:
                self.log_error(page_name, "File input frame missing")
                
            if hasattr(page, 'analysis_frame'):
                self.log_success(page_name, "Analysis frame exists")
            else:
                self.log_error(page_name, "Analysis frame missing")
                
            self.test_results[page_name] = "PASSED"
            
        except Exception as e:
            self.log_error(page_name, f"Test failed: {e}")
            self.test_results[page_name] = "FAILED"
    
    def test_batch_processing_page(self):
        """Test Batch Processing page."""
        page_name = "Batch Processing Page"
        logger.info(f"\nTesting {page_name}...")
        
        try:
            self.app.ctk_window._show_page("batch")
            time.sleep(0.5)
            
            page = self.app.ctk_window.pages.get("batch")
            if not page:
                if "batch" in self.app.ctk_window.page_classes:
                    self.log_warning(page_name, "Page not loaded")
                else:
                    self.log_error(page_name, "Page not available")
                return
                
            self.log_success(page_name, "Page loaded successfully")
            self.test_results[page_name] = "PASSED"
            
        except Exception as e:
            self.log_error(page_name, f"Test failed: {e}")
            self.test_results[page_name] = "FAILED"
    
    def test_multi_track_page(self):
        """Test Multi-Track Analysis page."""
        page_name = "Multi-Track Page"
        logger.info(f"\nTesting {page_name}...")
        
        try:
            self.app.ctk_window._show_page("multi_track")
            time.sleep(0.5)
            
            page = self.app.ctk_window.pages.get("multi_track")
            if not page:
                self.log_warning(page_name, "Page not loaded")
                return
                
            self.log_success(page_name, "Page loaded successfully")
            self.test_results[page_name] = "PASSED"
            
        except Exception as e:
            self.log_error(page_name, f"Test failed: {e}")
            self.test_results[page_name] = "FAILED"
    
    def test_final_test_comparison_page(self):
        """Test Final Test Comparison page."""
        page_name = "Final Test Comparison Page"
        logger.info(f"\nTesting {page_name}...")
        
        try:
            self.app.ctk_window._show_page("final_test")
            time.sleep(0.5)
            
            page = self.app.ctk_window.pages.get("final_test")
            if not page:
                self.log_warning(page_name, "Page not loaded")
                return
                
            self.log_success(page_name, "Page loaded successfully")
            
            # Test file selection components
            if hasattr(page, 'file1_path') and hasattr(page, 'file2_path'):
                self.log_success(page_name, "File selection components exist")
            else:
                self.log_error(page_name, "File selection components missing")
                
            self.test_results[page_name] = "PASSED"
            
        except Exception as e:
            self.log_error(page_name, f"Test failed: {e}")
            self.test_results[page_name] = "FAILED"
    
    def test_model_summary_page(self):
        """Test Model Summary page."""
        page_name = "Model Summary Page"
        logger.info(f"\nTesting {page_name}...")
        
        try:
            self.app.ctk_window._show_page("model_summary")
            time.sleep(0.5)
            
            page = self.app.ctk_window.pages.get("model_summary")
            if not page:
                self.log_warning(page_name, "Page not loaded")
                return
                
            self.log_success(page_name, "Page loaded successfully")
            self.test_results[page_name] = "PASSED"
            
        except Exception as e:
            self.log_error(page_name, f"Test failed: {e}")
            self.test_results[page_name] = "FAILED"
    
    def test_historical_page(self):
        """Test Historical Data page."""
        page_name = "Historical Page"
        logger.info(f"\nTesting {page_name}...")
        
        try:
            self.app.ctk_window._show_page("historical")
            time.sleep(0.5)
            
            page = self.app.ctk_window.pages.get("historical")
            if not page:
                self.log_warning(page_name, "Page not loaded")
                return
                
            self.log_success(page_name, "Page loaded successfully")
            
            # Test search functionality
            if hasattr(page, 'search_analyses'):
                # Try a search
                page.search_analyses()
                self.log_success(page_name, "Search functionality works")
            else:
                self.log_error(page_name, "Search method missing")
                
            # Test if treeview exists
            if hasattr(page, 'tree'):
                self.log_success(page_name, "Data treeview exists")
                # Check if data is displayed
                items = page.tree.get_children()
                if items:
                    self.log_success(page_name, f"Found {len(items)} records")
                else:
                    self.log_warning(page_name, "No data displayed")
            else:
                self.log_error(page_name, "Treeview missing")
                
            self.test_results[page_name] = "PASSED"
            
        except Exception as e:
            self.log_error(page_name, f"Test failed: {e}")
            self.test_results[page_name] = "FAILED"
    
    def test_ml_tools_page(self):
        """Test ML Tools page."""
        page_name = "ML Tools Page"
        logger.info(f"\nTesting {page_name}...")
        
        try:
            self.app.ctk_window._show_page("ml_tools")
            time.sleep(0.5)
            
            page = self.app.ctk_window.pages.get("ml_tools")
            if not page:
                self.log_warning(page_name, "Page not loaded")
                return
                
            self.log_success(page_name, "Page loaded successfully")
            
            # Test ML components
            if hasattr(page, 'model_training_frame'):
                self.log_success(page_name, "Model training frame exists")
            else:
                self.log_warning(page_name, "Model training frame missing")
                
            self.test_results[page_name] = "PASSED"
            
        except Exception as e:
            self.log_error(page_name, f"Test failed: {e}")
            self.test_results[page_name] = "FAILED"
    
    def test_ai_insights_page(self):
        """Test AI Insights page."""
        page_name = "AI Insights Page"
        logger.info(f"\nTesting {page_name}...")
        
        try:
            self.app.ctk_window._show_page("ai_insights")
            time.sleep(0.5)
            
            page = self.app.ctk_window.pages.get("ai_insights")
            if not page:
                self.log_warning(page_name, "Page not loaded")
                return
                
            self.log_success(page_name, "Page loaded successfully")
            self.test_results[page_name] = "PASSED"
            
        except Exception as e:
            self.log_error(page_name, f"Test failed: {e}")
            self.test_results[page_name] = "FAILED"
    
    def test_settings_page(self):
        """Test Settings page."""
        page_name = "Settings Page"
        logger.info(f"\nTesting {page_name}...")
        
        try:
            self.app.ctk_window._show_page("settings")
            time.sleep(0.5)
            
            page = self.app.ctk_window.pages.get("settings")
            if not page:
                self.log_warning(page_name, "Page not loaded")
                return
                
            self.log_success(page_name, "Page loaded successfully")
            
            # Test tabview
            if hasattr(page, 'tabview'):
                self.log_success(page_name, "Tabview exists")
                # Try to access tabs
                tabs = ["General", "Analysis", "Database", "ML Settings"]
                for tab in tabs:
                    try:
                        page.tabview.set(tab)
                        self.log_success(page_name, f"Tab '{tab}' accessible")
                    except:
                        self.log_warning(page_name, f"Tab '{tab}' not accessible")
            else:
                self.log_error(page_name, "Tabview missing")
                
            self.test_results[page_name] = "PASSED"
            
        except Exception as e:
            self.log_error(page_name, f"Test failed: {e}")
            self.test_results[page_name] = "FAILED"
    
    def test_navigation(self):
        """Test navigation between pages."""
        logger.info("\nTesting navigation...")
        
        try:
            # Test all navigation buttons
            for page_name in ["home", "single_file", "batch", "multi_track", 
                             "final_test", "model_summary", "historical", 
                             "ml_tools", "ai_insights", "settings"]:
                self.app.ctk_window._show_page(page_name)
                time.sleep(0.2)
                
                # Check if page switched
                if self.app.ctk_window.current_page == page_name:
                    self.log_success("Navigation", f"Navigated to {page_name}")
                else:
                    self.log_error("Navigation", f"Failed to navigate to {page_name}")
                    
        except Exception as e:
            self.log_error("Navigation", f"Test failed: {e}")
    
    def test_database_operations(self):
        """Test database operations."""
        logger.info("\nTesting database operations...")
        
        try:
            db_manager = self.app.ctk_window.db_manager
            if not db_manager:
                self.log_error("Database", "Database manager not initialized")
                return
                
            # Test getting analyses
            analyses = db_manager.get_all_analyses()
            self.log_success("Database", f"Retrieved {len(analyses)} analyses")
            
            # Test search
            results = db_manager.search_analyses(model="8575")
            self.log_success("Database", f"Search returned {len(results)} results")
            
            # Test date range query
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            date_results = db_manager.get_analyses_by_date_range(start_date, end_date)
            self.log_success("Database", f"Date range query returned {len(date_results)} results")
            
        except Exception as e:
            self.log_error("Database", f"Test failed: {e}")
    
    def generate_report(self):
        """Generate test report."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST REPORT")
        logger.info("=" * 80)
        
        # Summary
        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results.values() if r == "PASSED")
        failed = sum(1 for r in self.test_results.values() if r == "FAILED")
        
        logger.info(f"\nTotal Pages Tested: {total_tests}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {(passed/total_tests)*100:.1f}%")
        
        # Page results
        logger.info("\nPage Test Results:")
        for page, result in self.test_results.items():
            status = "✓" if result == "PASSED" else "✗"
            logger.info(f"  {status} {page}: {result}")
        
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
        
        # Save report to file
        report_path = Path("test_report.txt")
        with open(report_path, "w") as f:
            f.write("LASER TRIM ANALYZER TEST REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Pages Tested: {total_tests}\n")
            f.write(f"Passed: {passed}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Success Rate: {(passed/total_tests)*100:.1f}%\n\n")
            
            f.write("Page Test Results:\n")
            for page, result in self.test_results.items():
                f.write(f"  {page}: {result}\n")
            
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
    tester = GUITester()
    tester.start_app()


if __name__ == "__main__":
    main()