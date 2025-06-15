#!/usr/bin/env python3
"""
Comprehensive GUI Testing Script for Laser Trim Analyzer
Systematically tests all pages, buttons, UI elements, and features
"""

import sys
import os
import traceback
import logging
from pathlib import Path
import time
from datetime import datetime

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gui_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveGUITester:
    """Systematic GUI testing class"""
    
    def __init__(self):
        self.test_results = {}
        self.errors_found = []
        self.pages_tested = []
        self.buttons_tested = []
        self.features_tested = []
        
        # Initialize the app
        self.app = None
        self.test_data_dir = project_root / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)
        
    def run_comprehensive_test(self):
        """Run comprehensive testing of all GUI components"""
        logger.info("🔍 Starting Comprehensive GUI Testing")
        logger.info("=" * 60)
        
        try:
            # Initialize application
            self._initialize_app()
            
            # Test each page systematically
            self._test_all_pages()
            
            # Test cross-page functionality
            self._test_cross_page_features()
            
            # Test database connectivity
            self._test_database_features()
            
            # Generate final report
            self._generate_test_report()
            
        except Exception as e:
            logger.error(f"Critical error during testing: {e}")
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def _initialize_app(self):
        """Initialize the application for testing"""
        logger.info("🚀 Initializing Application...")
        
        try:
            from laser_trim_analyzer.core.config import get_config
            from laser_trim_analyzer.gui.ctk_main_window import CTkMainWindow
            
            config = get_config()
            self.app = CTkMainWindow(config)
            self.app.withdraw()  # Hide window during testing
            
            # Give the app time to initialize
            self.app.update()
            time.sleep(1)
            
            logger.info("✅ Application initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize application: {e}")
            raise
    
    def _test_all_pages(self):
        """Test all pages systematically"""
        logger.info("\n📄 Testing All Pages...")
        logger.info("-" * 40)
        
        # List of all pages to test
        pages_to_test = [
            'home', 'single_file', 'batch', 'multi_track', 
            'final_test', 'model_summary', 'historical', 
            'ml_tools', 'ai_insights', 'settings'
        ]
        
        for page_name in pages_to_test:
            self._test_single_page(page_name)
    
    def _test_single_page(self, page_name):
        """Test a single page comprehensively"""
        logger.info(f"\n🔍 Testing {page_name.upper()} Page...")
        
        page_results = {
            'loaded': False,
            'widgets_created': False,
            'buttons_functional': False,
            'errors': []
        }
        
        try:
            # Navigate to page
            if hasattr(self.app, '_show_page'):
                self.app._show_page(page_name)
                self.app.update()
                time.sleep(0.5)
                page_results['loaded'] = True
                logger.info(f"  ✅ {page_name} page loaded")
            else:
                raise Exception("App missing _show_page method")
            
            # Get page object
            page = self.app.pages.get(page_name)
            if not page:
                raise Exception(f"Page {page_name} not found in app.pages")
            
            # Test page widgets
            self._test_page_widgets(page, page_name, page_results)
            
            # Test page buttons
            self._test_page_buttons(page, page_name, page_results)
            
        except Exception as e:
            error_msg = f"Error testing {page_name}: {str(e)}"
            logger.error(f"  ❌ {error_msg}")
            page_results['errors'].append(error_msg)
            self.errors_found.append(error_msg)
        
        self.test_results[page_name] = page_results
        self.pages_tested.append(page_name)
    
    def _test_page_widgets(self, page, page_name, results):
        """Test that page widgets are properly created"""
        try:
            widget_count = 0
            
            # Count all attributes that look like widgets
            for attr_name in dir(page):
                if not attr_name.startswith('_'):
                    attr = getattr(page, attr_name)
                    if hasattr(attr, 'winfo_exists'):
                        try:
                            if attr.winfo_exists():
                                widget_count += 1
                        except:
                            pass
            
            if widget_count > 0:
                results['widgets_created'] = True
                logger.info(f"  ✅ {widget_count} widgets found in {page_name}")
            else:
                logger.warning(f"  ⚠️  No widgets found in {page_name}")
                
        except Exception as e:
            error_msg = f"Widget testing error in {page_name}: {str(e)}"
            logger.error(f"  ❌ {error_msg}")
            results['errors'].append(error_msg)
    
    def _test_page_buttons(self, page, page_name, results):
        """Test that page buttons exist and are functional"""
        try:
            button_count = 0
            functional_buttons = 0
            
            # Find all button attributes
            for attr_name in dir(page):
                if 'button' in attr_name.lower() and not attr_name.startswith('_'):
                    button = getattr(page, attr_name)
                    if hasattr(button, 'winfo_exists'):
                        try:
                            if button.winfo_exists():
                                button_count += 1
                                
                                # Check if button has a command
                                if hasattr(button, 'cget'):
                                    try:
                                        command = button.cget('command')
                                        if command:
                                            functional_buttons += 1
                                            self.buttons_tested.append(f"{page_name}.{attr_name}")
                                    except:
                                        pass
                        except:
                            pass
            
            if button_count > 0:
                results['buttons_functional'] = functional_buttons > 0
                logger.info(f"  ✅ {button_count} buttons found ({functional_buttons} functional) in {page_name}")
            else:
                logger.info(f"  ℹ️  No buttons found in {page_name}")
                
        except Exception as e:
            error_msg = f"Button testing error in {page_name}: {str(e)}"
            logger.error(f"  ❌ {error_msg}")
            results['errors'].append(error_msg)
    
    def _test_cross_page_features(self):
        """Test features that work across pages"""
        logger.info("\n🔗 Testing Cross-Page Features...")
        logger.info("-" * 40)
        
        try:
            # Test navigation between pages
            for page_name in ['home', 'single_file', 'batch']:
                if page_name in self.app.pages:
                    self.app._show_page(page_name)
                    self.app.update()
                    time.sleep(0.2)
                    logger.info(f"  ✅ Navigation to {page_name} works")
                    
        except Exception as e:
            error_msg = f"Cross-page navigation error: {str(e)}"
            logger.error(f"  ❌ {error_msg}")
            self.errors_found.append(error_msg)
    
    def _test_database_features(self):
        """Test database connectivity and features"""
        logger.info("\n🗄️  Testing Database Features...")
        logger.info("-" * 40)
        
        try:
            if hasattr(self.app, 'db_manager') and self.app.db_manager:
                # Test database connection
                with self.app.db_manager.get_session() as session:
                    logger.info("  ✅ Database connection successful")
                    
                # Test basic queries
                try:
                    results = self.app.db_manager.get_historical_data(limit=1)
                    logger.info(f"  ✅ Historical data query successful ({len(results)} records)")
                except Exception as e:
                    logger.warning(f"  ⚠️  Historical data query failed: {e}")
                    
            else:
                logger.warning("  ⚠️  No database manager found")
                
        except Exception as e:
            error_msg = f"Database testing error: {str(e)}"
            logger.error(f"  ❌ {error_msg}")
            self.errors_found.append(error_msg)
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("\n📊 Generating Test Report...")
        logger.info("=" * 60)
        
        # Summary statistics
        total_pages = len(self.test_results)
        successful_pages = sum(1 for r in self.test_results.values() if r['loaded'])
        total_errors = len(self.errors_found)
        
        logger.info(f"📈 TEST SUMMARY:")
        logger.info(f"  Total Pages Tested: {total_pages}")
        logger.info(f"  Successfully Loaded: {successful_pages}")
        logger.info(f"  Total Errors Found: {total_errors}")
        
        # Detailed page results
        logger.info(f"\n📋 DETAILED RESULTS:")
        for page_name, results in self.test_results.items():
            status = "✅" if results['loaded'] and not results['errors'] else "❌"
            logger.info(f"  {status} {page_name.upper()}:")
            logger.info(f"    Loaded: {'✅' if results['loaded'] else '❌'}")
            logger.info(f"    Widgets: {'✅' if results['widgets_created'] else '❌'}")
            logger.info(f"    Buttons: {'✅' if results['buttons_functional'] else '❌'}")
            if results['errors']:
                for error in results['errors']:
                    logger.info(f"    Error: {error}")
        
        # Error summary
        if self.errors_found:
            logger.info(f"\n🚨 ERRORS FOUND:")
            for i, error in enumerate(self.errors_found, 1):
                logger.info(f"  {i}. {error}")
        else:
            logger.info(f"\n🎉 NO CRITICAL ERRORS FOUND!")
    
    def _cleanup(self):
        """Clean up test resources"""
        logger.info("\n🧹 Cleaning up...")
        
        try:
            if self.app:
                self.app.quit()
                self.app.destroy()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    """Main testing function"""
    print("🔍 LASER TRIM ANALYZER - COMPREHENSIVE GUI TESTING")
    print("=" * 60)
    
    tester = ComprehensiveGUITester()
    tester.run_comprehensive_test()
    
    print("\n✅ Testing completed! Check the logs for detailed results.")

if __name__ == "__main__":
    main() 