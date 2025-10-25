#!/usr/bin/env python3
"""
Systematic test script to identify broken features in Laser Trim Analyzer V2
Tests each page and documents all issues found
"""

import sys
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import customtkinter as ctk
from laser_trim_analyzer.gui.ctk_main_window import CTkMainWindow
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'broken_features_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class FeatureTester:
    def __init__(self):
        self.broken_features = {
            "crashes": [],
            "non_functional": [],
            "error_states": [],
            "incorrect_behavior": []
        }
        self.app = None
        self.logger = logging.getLogger(__name__)
        
    def start_app(self):
        """Start the application"""
        try:
            self.app = CTkMainWindow()
            self.logger.info("Application started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start application: {e}")
            self.broken_features["crashes"].append({
                "location": "Application Startup",
                "feature": "Main Window",
                "expected": "Application starts",
                "actual": f"Crash: {str(e)}",
                "error": traceback.format_exc(),
                "priority": "BLOCKING"
            })
            return False
    
    def test_page_navigation(self):
        """Test navigation to each page"""
        pages = [
            "home", "single_file", "batch", "multi_track", "final_test",
            "model_summary", "historical", "ml_tools", "ai_insights", "settings"
        ]
        
        for page in pages:
            try:
                self.logger.info(f"Testing navigation to {page} page...")
                self.app._show_page(page)
                self.app.update()
                time.sleep(0.5)  # Allow page to render
                
                # Check if page loaded
                if page not in self.app.pages:
                    self.broken_features["non_functional"].append({
                        "location": "Navigation",
                        "feature": f"{page.title()} Page",
                        "expected": "Page loads",
                        "actual": "Page not found in pages dict",
                        "error": "Page failed to initialize",
                        "priority": "HIGH"
                    })
                elif not self.app.pages[page].winfo_viewable():
                    self.broken_features["incorrect_behavior"].append({
                        "location": "Navigation",
                        "feature": f"{page.title()} Page Display",
                        "expected": "Page is visible",
                        "actual": "Page loaded but not visible",
                        "error": "Display issue",
                        "priority": "MEDIUM"
                    })
                    
            except Exception as e:
                self.broken_features["crashes"].append({
                    "location": "Page Navigation",
                    "feature": f"Navigate to {page.title()} Page",
                    "expected": "Navigate successfully",
                    "actual": f"Navigation failed: {str(e)}",
                    "error": traceback.format_exc(),
                    "priority": "HIGH"
                })
    
    def test_home_page(self):
        """Test Home Page features"""
        self.logger.info("Testing Home Page features...")
        try:
            self.app._show_page("home")
            self.app.update()
            page = self.app.pages.get("home")
            
            if page:
                # Test refresh button
                try:
                    if hasattr(page, 'refresh_button'):
                        page.refresh_button.invoke()
                        self.app.update()
                except Exception as e:
                    self.broken_features["error_states"].append({
                        "location": "Home Page",
                        "feature": "Refresh Button",
                        "expected": "Refreshes data",
                        "actual": f"Error: {str(e)}",
                        "error": traceback.format_exc(),
                        "priority": "MEDIUM"
                    })
                
                # Test quick action buttons
                quick_actions = ['single_file_btn', 'batch_btn', 'ml_tools_btn']
                for btn_name in quick_actions:
                    try:
                        if hasattr(page, btn_name):
                            btn = getattr(page, btn_name)
                            # Don't actually click to avoid navigation
                            if not btn.winfo_viewable():
                                self.broken_features["non_functional"].append({
                                    "location": "Home Page",
                                    "feature": f"Quick Action: {btn_name}",
                                    "expected": "Button visible",
                                    "actual": "Button not visible",
                                    "error": "UI issue",
                                    "priority": "LOW"
                                })
                    except Exception as e:
                        self.broken_features["error_states"].append({
                            "location": "Home Page",
                            "feature": f"Quick Action: {btn_name}",
                            "expected": "Button exists",
                            "actual": f"Error: {str(e)}",
                            "error": str(e),
                            "priority": "LOW"
                        })
                        
        except Exception as e:
            self.broken_features["crashes"].append({
                "location": "Home Page",
                "feature": "Page Testing",
                "expected": "Test completes",
                "actual": f"Test crashed: {str(e)}",
                "error": traceback.format_exc(),
                "priority": "HIGH"
            })
    
    def test_single_file_page(self):
        """Test Single File Analysis Page features"""
        self.logger.info("Testing Single File Analysis Page features...")
        try:
            self.app._show_page("single_file")
            self.app.update()
            page = self.app.pages.get("single_file")
            
            if page:
                # Test browse button
                try:
                    if hasattr(page, 'browse_button'):
                        # Check if button is enabled
                        state = page.browse_button.cget("state")
                        if state == "disabled":
                            self.broken_features["incorrect_behavior"].append({
                                "location": "Single File Page",
                                "feature": "Browse Button",
                                "expected": "Button enabled",
                                "actual": "Button disabled",
                                "error": "Initial state issue",
                                "priority": "HIGH"
                            })
                except Exception as e:
                    self.broken_features["error_states"].append({
                        "location": "Single File Page",
                        "feature": "Browse Button State Check",
                        "expected": "Can check state",
                        "actual": f"Error: {str(e)}",
                        "error": traceback.format_exc(),
                        "priority": "MEDIUM"
                    })
                
                # Test analyze button (should be disabled without file)
                try:
                    if hasattr(page, 'analyze_button'):
                        state = page.analyze_button.cget("state")
                        if state == "normal" and not hasattr(page, 'current_file'):
                            self.broken_features["incorrect_behavior"].append({
                                "location": "Single File Page",
                                "feature": "Analyze Button State",
                                "expected": "Disabled without file",
                                "actual": "Enabled without file",
                                "error": "State logic issue",
                                "priority": "MEDIUM"
                            })
                except Exception as e:
                    self.broken_features["error_states"].append({
                        "location": "Single File Page",
                        "feature": "Analyze Button Check",
                        "expected": "Can check state",
                        "actual": f"Error: {str(e)}",
                        "error": traceback.format_exc(),
                        "priority": "MEDIUM"
                    })
                    
        except Exception as e:
            self.broken_features["crashes"].append({
                "location": "Single File Page",
                "feature": "Page Testing",
                "expected": "Test completes",
                "actual": f"Test crashed: {str(e)}",
                "error": traceback.format_exc(),
                "priority": "HIGH"
            })
    
    def test_batch_processing_page(self):
        """Test Batch Processing Page features"""
        self.logger.info("Testing Batch Processing Page features...")
        try:
            self.app._show_page("batch")
            self.app.update()
            page = self.app.pages.get("batch")
            
            if page:
                # Test add files button
                try:
                    if hasattr(page, 'add_files_button'):
                        state = page.add_files_button.cget("state")
                        if state == "disabled":
                            self.broken_features["incorrect_behavior"].append({
                                "location": "Batch Processing Page",
                                "feature": "Add Files Button",
                                "expected": "Button enabled",
                                "actual": "Button disabled",
                                "error": "Initial state issue",
                                "priority": "HIGH"
                            })
                except Exception as e:
                    self.broken_features["error_states"].append({
                        "location": "Batch Processing Page",
                        "feature": "Add Files Button Check",
                        "expected": "Can check state",
                        "actual": f"Error: {str(e)}",
                        "error": traceback.format_exc(),
                        "priority": "MEDIUM"
                    })
                
                # Test clear button
                try:
                    if hasattr(page, 'clear_button'):
                        page.clear_button.invoke()
                        self.app.update()
                except Exception as e:
                    self.broken_features["error_states"].append({
                        "location": "Batch Processing Page",
                        "feature": "Clear Button",
                        "expected": "Clears file list",
                        "actual": f"Error: {str(e)}",
                        "error": traceback.format_exc(),
                        "priority": "LOW"
                    })
                    
        except Exception as e:
            self.broken_features["crashes"].append({
                "location": "Batch Processing Page",
                "feature": "Page Testing",
                "expected": "Test completes",
                "actual": f"Test crashed: {str(e)}",
                "error": traceback.format_exc(),
                "priority": "HIGH"
            })
    
    def test_settings_page(self):
        """Test Settings Page features"""
        self.logger.info("Testing Settings Page features...")
        try:
            self.app._show_page("settings")
            self.app.update()
            page = self.app.pages.get("settings")
            
            if page:
                # Test save button
                try:
                    if hasattr(page, 'save_button'):
                        # Try to save without changes
                        page.save_button.invoke()
                        self.app.update()
                except Exception as e:
                    self.broken_features["error_states"].append({
                        "location": "Settings Page",
                        "feature": "Save Button",
                        "expected": "Saves settings",
                        "actual": f"Error: {str(e)}",
                        "error": traceback.format_exc(),
                        "priority": "MEDIUM"
                    })
                
                # Test reset button
                try:
                    if hasattr(page, 'reset_button'):
                        page.reset_button.invoke()
                        self.app.update()
                except Exception as e:
                    self.broken_features["error_states"].append({
                        "location": "Settings Page",
                        "feature": "Reset Button",
                        "expected": "Resets to defaults",
                        "actual": f"Error: {str(e)}",
                        "error": traceback.format_exc(),
                        "priority": "LOW"
                    })
                    
        except Exception as e:
            self.broken_features["crashes"].append({
                "location": "Settings Page",
                "feature": "Page Testing",
                "expected": "Test completes",
                "actual": f"Test crashed: {str(e)}",
                "error": traceback.format_exc(),
                "priority": "MEDIUM"
            })
    
    def generate_report(self):
        """Generate comprehensive broken features report"""
        report = []
        report.append("# BROKEN FEATURES INVENTORY - LASER TRIM ANALYZER V2")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        total_issues = sum(len(issues) for issues in self.broken_features.values())
        report.append("## SUMMARY")
        report.append(f"Total Issues Found: {total_issues}")
        report.append(f"- Crashes: {len(self.broken_features['crashes'])}")
        report.append(f"- Non-functional: {len(self.broken_features['non_functional'])}")
        report.append(f"- Error States: {len(self.broken_features['error_states'])}")
        report.append(f"- Incorrect Behavior: {len(self.broken_features['incorrect_behavior'])}")
        report.append("")
        
        # Detailed issues by category
        categories = [
            ("CRASHES (Application stops working)", "crashes"),
            ("NON-FUNCTIONAL (Feature does nothing)", "non_functional"),
            ("ERROR STATES (Features show errors)", "error_states"),
            ("INCORRECT BEHAVIOR (Features work wrong)", "incorrect_behavior")
        ]
        
        for title, key in categories:
            issues = self.broken_features[key]
            if issues:
                report.append(f"## {title}")
                report.append("-" * 80)
                for i, issue in enumerate(issues, 1):
                    report.append(f"\n### Issue #{i}")
                    report.append(f"**Priority:** {issue['priority']}")
                    report.append(f"**Location:** {issue['location']}")
                    report.append(f"**Feature:** {issue['feature']}")
                    report.append(f"**Expected:** {issue['expected']}")
                    report.append(f"**Actual:** {issue['actual']}")
                    if 'error' in issue and len(issue['error']) > 100:
                        report.append(f"**Error:** See detailed log")
                    else:
                        report.append(f"**Error:** {issue.get('error', 'N/A')}")
                    report.append("")
                report.append("")
        
        # Priority summary
        report.append("## PRIORITY BREAKDOWN")
        report.append("-" * 80)
        priority_counts = {"BLOCKING": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for category in self.broken_features.values():
            for issue in category:
                priority_counts[issue["priority"]] += 1
        
        for priority, count in priority_counts.items():
            if count > 0:
                report.append(f"{priority}: {count} issues")
        
        return "\n".join(report)
    
    def run_tests(self):
        """Run all tests"""
        print("Starting Laser Trim Analyzer V2 Feature Testing...")
        print("=" * 80)
        
        # Start app
        if not self.start_app():
            print("Failed to start application. Aborting tests.")
            return
        
        # Test navigation
        self.test_page_navigation()
        
        # Test individual pages
        self.test_home_page()
        self.test_single_file_page()
        self.test_batch_processing_page()
        self.test_settings_page()
        
        # Generate report
        report = self.generate_report()
        
        # Save report
        report_file = f"BROKEN_FEATURES_INVENTORY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_file}")
        print("\n" + report)
        
        # Close app
        try:
            self.app.destroy()
        except:
            pass

if __name__ == "__main__":
    tester = FeatureTester()
    tester.run_tests()