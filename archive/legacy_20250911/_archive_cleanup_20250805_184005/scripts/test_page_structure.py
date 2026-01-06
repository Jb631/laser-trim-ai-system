#!/usr/bin/env python
"""
Page structure testing script for Laser Trim Analyzer.
Tests page implementations without running the GUI.
"""

import sys
import os
from pathlib import Path
import logging
import importlib
import inspect
import ast
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set environment to development
os.environ["LTA_ENV"] = "development"

from laser_trim_analyzer.core.config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PageStructureTester:
    """Test page structure and implementation."""
    
    def __init__(self):
        self.config = get_config()
        self.errors = []
        self.warnings = []
        self.test_results = {}
        self.pages_path = Path(__file__).parent.parent / "src" / "laser_trim_analyzer" / "gui" / "pages"
        
    def log_error(self, page, message):
        error = f"{page}: {message}"
        self.errors.append(error)
        logger.error(error)
        
    def log_warning(self, page, message):
        warning = f"{page}: {message}"
        self.warnings.append(warning)
        logger.warning(warning)
        
    def log_success(self, page, message):
        logger.info(f"{page}: ✓ {message}")
        
    def run_all_tests(self):
        """Run all page structure tests."""
        logger.info("=" * 80)
        logger.info("Starting page structure testing...")
        logger.info("=" * 80)
        
        # Test each page
        self.test_page_imports()
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
        
        # Generate report
        self.generate_report()
        
    def test_page_imports(self):
        """Test that all page modules can be imported."""
        page_name = "Page Imports"
        logger.info(f"\nTesting {page_name}...")
        
        pages = [
            "home_page", "single_file_page", "batch_processing_page",
            "multi_track_page", "final_test_comparison_page", "model_summary_page",
            "historical_page", "ml_tools_page", "ai_insights_page", "settings_page"
        ]
        
        for page in pages:
            try:
                module = importlib.import_module(f"laser_trim_analyzer.gui.pages.{page}")
                self.log_success(page_name, f"Successfully imported {page}")
            except ImportError as e:
                self.log_error(page_name, f"Cannot import {page}: {e}")
                
        self.test_results[page_name] = "PASSED" if not self.errors else "FAILED"
        
    def analyze_page_file(self, page_file: str, expected_class: str, required_methods: list, required_widgets: list):
        """Analyze a page file for required components."""
        page_path = self.pages_path / page_file
        
        if not page_path.exists():
            return None, f"File not found: {page_path}"
            
        try:
            with open(page_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the AST
            tree = ast.parse(content)
            
            # Find the main class
            class_found = False
            methods_found = set()
            widgets_found = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == expected_class:
                    class_found = True
                    
                    # Check methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods_found.add(item.name)
                            
                # Check for widget creation (simplified check)
                if isinstance(node, ast.Attribute):
                    attr_name = node.attr
                    if any(widget in attr_name for widget in ["Frame", "Button", "Label", "Entry", "Text"]):
                        widgets_found.add(attr_name)
                        
            # Verify requirements
            missing_methods = set(required_methods) - methods_found
            
            analysis = {
                "class_found": class_found,
                "methods_found": list(methods_found),
                "missing_methods": list(missing_methods),
                "widgets_found": len(widgets_found) > 0
            }
            
            return analysis, None
            
        except Exception as e:
            return None, f"Error analyzing file: {e}"
            
    def test_home_page(self):
        """Test Home page structure."""
        page_name = "Home Page"
        logger.info(f"\nTesting {page_name} structure...")
        
        analysis, error = self.analyze_page_file(
            "home_page.py",
            "HomePage",
            ["__init__", "_create_overview_section", "_create_quick_actions", "refresh_recent_analyses"],
            ["Frame", "Button", "Label"]
        )
        
        if error:
            self.log_error(page_name, error)
            self.test_results[page_name] = "FAILED"
            return
            
        if analysis["class_found"]:
            self.log_success(page_name, "HomePage class found")
        else:
            self.log_error(page_name, "HomePage class not found")
            
        if not analysis["missing_methods"]:
            self.log_success(page_name, "All required methods found")
        else:
            self.log_warning(page_name, f"Missing methods: {analysis['missing_methods']}")
            
        if analysis["widgets_found"]:
            self.log_success(page_name, "GUI widgets found")
        else:
            self.log_warning(page_name, "No GUI widgets found")
            
        self.test_results[page_name] = "PASSED" if analysis["class_found"] else "FAILED"
        
    def test_single_file_page(self):
        """Test Single File page structure."""
        page_name = "Single File Page"
        logger.info(f"\nTesting {page_name} structure...")
        
        analysis, error = self.analyze_page_file(
            "single_file_page.py",
            "SingleFilePage",
            ["__init__", "_create_file_input", "process_file", "_handle_file_drop"],
            ["Frame", "Button", "Label", "Entry"]
        )
        
        if error:
            self.log_error(page_name, error)
            self.test_results[page_name] = "FAILED"
            return
            
        if analysis["class_found"]:
            self.log_success(page_name, "SingleFilePage class found")
            
            # Check for drag-and-drop handling
            if "_handle_file_drop" in analysis["methods_found"]:
                self.log_success(page_name, "Drag-and-drop support implemented")
            else:
                self.log_warning(page_name, "Drag-and-drop support missing")
                
        self.test_results[page_name] = "PASSED" if analysis["class_found"] else "FAILED"
        
    def test_batch_processing_page(self):
        """Test Batch Processing page structure."""
        page_name = "Batch Processing Page"
        logger.info(f"\nTesting {page_name} structure...")
        
        analysis, error = self.analyze_page_file(
            "batch_processing_page.py",
            "BatchProcessingPage",
            ["__init__", "_create_batch_controls", "start_batch_processing"],
            ["Frame", "Button", "ProgressBar"]
        )
        
        if error:
            self.log_error(page_name, error)
            self.test_results[page_name] = "FAILED"
            return
            
        if analysis["class_found"]:
            self.log_success(page_name, "BatchProcessingPage class found")
            
        self.test_results[page_name] = "PASSED" if analysis["class_found"] else "FAILED"
        
    def test_multi_track_page(self):
        """Test Multi-Track page structure."""
        page_name = "Multi-Track Page"
        logger.info(f"\nTesting {page_name} structure...")
        
        analysis, error = self.analyze_page_file(
            "multi_track_page.py",
            "MultiTrackPage",
            ["__init__", "_create_track_selector", "display_track_data"],
            ["Frame", "OptionMenu", "Canvas"]
        )
        
        if error:
            self.log_error(page_name, error)
            self.test_results[page_name] = "FAILED"
            return
            
        if analysis["class_found"]:
            self.log_success(page_name, "MultiTrackPage class found")
            
        self.test_results[page_name] = "PASSED" if analysis["class_found"] else "FAILED"
        
    def test_final_test_comparison_page(self):
        """Test Final Test Comparison page structure."""
        page_name = "Final Test Comparison Page"
        logger.info(f"\nTesting {page_name} structure...")
        
        analysis, error = self.analyze_page_file(
            "final_test_comparison_page.py",
            "FinalTestComparisonPage",
            ["__init__", "_create_file_selection", "compare_files"],
            ["Frame", "Button", "Label"]
        )
        
        if error:
            self.log_error(page_name, error)
            self.test_results[page_name] = "FAILED"
            return
            
        if analysis["class_found"]:
            self.log_success(page_name, "FinalTestComparisonPage class found")
            
        self.test_results[page_name] = "PASSED" if analysis["class_found"] else "FAILED"
        
    def test_model_summary_page(self):
        """Test Model Summary page structure."""
        page_name = "Model Summary Page"
        logger.info(f"\nTesting {page_name} structure...")
        
        analysis, error = self.analyze_page_file(
            "model_summary_page.py",
            "ModelSummaryPage",
            ["__init__", "_create_model_selector", "display_model_stats"],
            ["Frame", "OptionMenu", "Label"]
        )
        
        if error:
            self.log_error(page_name, error)
            self.test_results[page_name] = "FAILED"
            return
            
        if analysis["class_found"]:
            self.log_success(page_name, "ModelSummaryPage class found")
            
        self.test_results[page_name] = "PASSED" if analysis["class_found"] else "FAILED"
        
    def test_historical_page(self):
        """Test Historical page structure."""
        page_name = "Historical Page"
        logger.info(f"\nTesting {page_name} structure...")
        
        analysis, error = self.analyze_page_file(
            "historical_page.py",
            "HistoricalPage",
            ["__init__", "_create_search_frame", "search_analyses", "_populate_tree"],
            ["Frame", "Entry", "Button", "Treeview"]
        )
        
        if error:
            self.log_error(page_name, error)
            self.test_results[page_name] = "FAILED"
            return
            
        if analysis["class_found"]:
            self.log_success(page_name, "HistoricalPage class found")
            
            # Check for database interaction methods
            if "search_analyses" in analysis["methods_found"]:
                self.log_success(page_name, "Database search method found")
                
        self.test_results[page_name] = "PASSED" if analysis["class_found"] else "FAILED"
        
    def test_ml_tools_page(self):
        """Test ML Tools page structure."""
        page_name = "ML Tools Page"
        logger.info(f"\nTesting {page_name} structure...")
        
        analysis, error = self.analyze_page_file(
            "ml_tools_page.py",
            "MLToolsPage",
            ["__init__", "_create_model_training", "_create_prediction_frame"],
            ["Frame", "Button", "Label"]
        )
        
        if error:
            self.log_error(page_name, error)
            self.test_results[page_name] = "FAILED"
            return
            
        if analysis["class_found"]:
            self.log_success(page_name, "MLToolsPage class found")
            
        self.test_results[page_name] = "PASSED" if analysis["class_found"] else "FAILED"
        
    def test_ai_insights_page(self):
        """Test AI Insights page structure."""
        page_name = "AI Insights Page"
        logger.info(f"\nTesting {page_name} structure...")
        
        analysis, error = self.analyze_page_file(
            "ai_insights_page.py",
            "AIInsightsPage",
            ["__init__", "_create_insights_display"],
            ["Frame", "Text", "Label"]
        )
        
        if error:
            self.log_error(page_name, error)
            self.test_results[page_name] = "FAILED"
            return
            
        if analysis["class_found"]:
            self.log_success(page_name, "AIInsightsPage class found")
            
        self.test_results[page_name] = "PASSED" if analysis["class_found"] else "FAILED"
        
    def test_settings_page(self):
        """Test Settings page structure."""
        page_name = "Settings Page"
        logger.info(f"\nTesting {page_name} structure...")
        
        analysis, error = self.analyze_page_file(
            "settings_page.py",
            "SettingsPage",
            ["__init__", "_create_general_tab", "_create_analysis_tab", "save_settings"],
            ["Frame", "Notebook", "Entry", "Button"]
        )
        
        if error:
            self.log_error(page_name, error)
            self.test_results[page_name] = "FAILED"
            return
            
        if analysis["class_found"]:
            self.log_success(page_name, "SettingsPage class found")
            
            # Check for save functionality
            if "save_settings" in analysis["methods_found"]:
                self.log_success(page_name, "Settings save functionality found")
                
        self.test_results[page_name] = "PASSED" if analysis["class_found"] else "FAILED"
        
    def generate_report(self):
        """Generate test report."""
        logger.info("\n" + "=" * 80)
        logger.info("PAGE STRUCTURE TEST REPORT")
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
        
        # Save report
        report_path = Path("page_structure_report.txt")
        with open(report_path, "w") as f:
            f.write("LASER TRIM ANALYZER PAGE STRUCTURE TEST REPORT\n")
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
    tester = PageStructureTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()