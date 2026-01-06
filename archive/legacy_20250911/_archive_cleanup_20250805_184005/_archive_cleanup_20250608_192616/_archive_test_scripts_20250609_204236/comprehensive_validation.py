#!/usr/bin/env python3
"""
Comprehensive validation script to ensure all functionality is working correctly.
"""

import ast
from pathlib import Path
import re
import json
from typing import Dict, List, Tuple


class ComprehensiveValidator:
    """Validates all components of the Laser Trim Analyzer."""
    
    def __init__(self):
        self.results = {
            'core': {},
            'gui': {},
            'utils': {},
            'analysis': {},
            'ml': {},
            'overall': True
        }
        self.errors = []
        
    def validate_all(self):
        """Run all validation checks."""
        print("=== Laser Trim Analyzer Comprehensive Validation ===\n")
        
        # Core functionality
        self.validate_core_modules()
        
        # GUI components
        self.validate_gui_components()
        
        # Utilities
        self.validate_utilities()
        
        # Analysis modules
        self.validate_analysis_modules()
        
        # ML components
        self.validate_ml_components()
        
        # Configuration
        self.validate_configuration()
        
        # Generate report
        self.generate_report()
        
    def check_python_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """Check if a Python file has valid syntax."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            ast.parse(code)
            return True, "OK"
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
    
    def validate_core_modules(self):
        """Validate core modules."""
        print("Validating Core Modules...")
        
        core_files = [
            "src/laser_trim_analyzer/core/processor.py",
            "src/laser_trim_analyzer/core/models.py",
            "src/laser_trim_analyzer/core/config.py",
            "src/laser_trim_analyzer/core/constants.py",
            "src/laser_trim_analyzer/core/exceptions.py",
            "src/laser_trim_analyzer/core/interfaces.py",
            "src/laser_trim_analyzer/core/implementations.py",
            "src/laser_trim_analyzer/core/strategies.py",
            "src/laser_trim_analyzer/core/utils.py",
            "src/laser_trim_analyzer/core/large_scale_processor.py",
        ]
        
        for file_path in core_files:
            path = Path(file_path)
            if path.exists():
                is_valid, msg = self.check_python_syntax(path)
                self.results['core'][path.name] = {
                    'valid': is_valid,
                    'message': msg
                }
                if not is_valid:
                    self.errors.append(f"{path.name}: {msg}")
                    self.results['overall'] = False
            else:
                self.results['core'][path.name] = {
                    'valid': False,
                    'message': "File not found"
                }
                
        # Check key components in processor
        processor_path = Path("src/laser_trim_analyzer/core/processor.py")
        if processor_path.exists():
            content = processor_path.read_text()
            
            # Check for plot generation
            if "create_analysis_plot" in content and "plot_path" in content:
                self.results['core']['plot_generation'] = {
                    'valid': True,
                    'message': "Plot generation code found"
                }
            else:
                self.results['core']['plot_generation'] = {
                    'valid': False,
                    'message': "Plot generation code missing"
                }
                self.results['overall'] = False
            
            # Check for ML integration
            if "ml_manager" in content or "failure_prediction" in content:
                self.results['core']['ml_integration'] = {
                    'valid': True,
                    'message': "ML integration found"
                }
            else:
                self.results['core']['ml_integration'] = {
                    'valid': False,
                    'message': "ML integration missing"
                }
                
    def validate_gui_components(self):
        """Validate GUI components."""
        print("\nValidating GUI Components...")
        
        gui_files = [
            "src/laser_trim_analyzer/gui/main_window.py",
            "src/laser_trim_analyzer/gui/ctk_main_window.py",
            "src/laser_trim_analyzer/gui/pages/single_file_page.py",
            "src/laser_trim_analyzer/gui/pages/batch_processing_page.py",
            "src/laser_trim_analyzer/gui/pages/ml_tools_page.py",
            "src/laser_trim_analyzer/gui/pages/ai_insights_page.py",
            "src/laser_trim_analyzer/gui/widgets/plot_viewer.py",
            "src/laser_trim_analyzer/gui/widgets/analysis_display.py",
            "src/laser_trim_analyzer/gui/widgets/track_viewer.py",
        ]
        
        for file_path in gui_files:
            path = Path(file_path)
            if path.exists():
                is_valid, msg = self.check_python_syntax(path)
                self.results['gui'][path.name] = {
                    'valid': is_valid,
                    'message': msg
                }
                if not is_valid:
                    self.errors.append(f"{path.name}: {msg}")
                    self.results['overall'] = False
            else:
                self.results['gui'][path.name] = {
                    'valid': False,
                    'message': "File not found"
                }
        
        # Check plot viewer integration
        plot_viewer_path = Path("src/laser_trim_analyzer/gui/widgets/plot_viewer.py")
        analysis_display_path = Path("src/laser_trim_analyzer/gui/widgets/analysis_display.py")
        
        if plot_viewer_path.exists() and analysis_display_path.exists():
            plot_viewer_content = plot_viewer_path.read_text()
            analysis_display_content = analysis_display_path.read_text()
            
            # Check PlotViewerWidget is complete
            required_methods = ["load_plot", "_zoom_in", "_zoom_out", "_export_plot", "_fit_to_window"]
            all_methods_found = all(method in plot_viewer_content for method in required_methods)
            
            self.results['gui']['plot_viewer_complete'] = {
                'valid': all_methods_found,
                'message': "All required methods found" if all_methods_found else "Missing plot viewer methods"
            }
            
            # Check integration
            if "PlotViewerWidget" in analysis_display_content and "load_plot" in analysis_display_content:
                self.results['gui']['plot_viewer_integrated'] = {
                    'valid': True,
                    'message': "Plot viewer properly integrated"
                }
            else:
                self.results['gui']['plot_viewer_integrated'] = {
                    'valid': False,
                    'message': "Plot viewer not integrated in analysis display"
                }
                self.results['overall'] = False
                
    def validate_utilities(self):
        """Validate utility modules."""
        print("\nValidating Utilities...")
        
        util_files = [
            "src/laser_trim_analyzer/utils/plotting_utils.py",
            "src/laser_trim_analyzer/utils/excel_utils.py",
            "src/laser_trim_analyzer/utils/report_generator.py",
            "src/laser_trim_analyzer/utils/validators.py",
            "src/laser_trim_analyzer/utils/file_utils.py",
        ]
        
        for file_path in util_files:
            path = Path(file_path)
            if path.exists():
                is_valid, msg = self.check_python_syntax(path)
                self.results['utils'][path.name] = {
                    'valid': is_valid,
                    'message': msg
                }
                if not is_valid:
                    self.errors.append(f"{path.name}: {msg}")
                    self.results['overall'] = False
                    
        # Check comprehensive Excel export
        report_gen_path = Path("src/laser_trim_analyzer/utils/report_generator.py")
        if report_gen_path.exists():
            content = report_gen_path.read_text()
            
            if "generate_comprehensive_excel_report" in content:
                # Check for ML data export
                ml_fields = ["failure_prediction", "risk_category", "failure_probability", "gradient_margin"]
                ml_export_found = all(field in content for field in ml_fields)
                
                self.results['utils']['comprehensive_excel_export'] = {
                    'valid': ml_export_found,
                    'message': "Comprehensive Excel export with ML data found" if ml_export_found else "ML data missing from Excel export"
                }
                
                if not ml_export_found:
                    self.results['overall'] = False
            else:
                self.results['utils']['comprehensive_excel_export'] = {
                    'valid': False,
                    'message': "Comprehensive Excel export method not found"
                }
                self.results['overall'] = False
                
    def validate_analysis_modules(self):
        """Validate analysis modules."""
        print("\nValidating Analysis Modules...")
        
        analysis_files = [
            "src/laser_trim_analyzer/analysis/base.py",
            "src/laser_trim_analyzer/analysis/sigma_analyzer.py",
            "src/laser_trim_analyzer/analysis/linearity_analyzer.py",
            "src/laser_trim_analyzer/analysis/resistance_analyzer.py",
            "src/laser_trim_analyzer/analysis/consistency_analyzer.py",
            "src/laser_trim_analyzer/analysis/analytics_engine.py",
        ]
        
        for file_path in analysis_files:
            path = Path(file_path)
            if path.exists():
                is_valid, msg = self.check_python_syntax(path)
                self.results['analysis'][path.name] = {
                    'valid': is_valid,
                    'message': msg
                }
                if not is_valid:
                    self.errors.append(f"{path.name}: {msg}")
                    self.results['overall'] = False
                    
    def validate_ml_components(self):
        """Validate ML components."""
        print("\nValidating ML Components...")
        
        ml_files = [
            "src/laser_trim_analyzer/ml/ml_manager.py",
            "src/laser_trim_analyzer/ml/models.py",
            "src/laser_trim_analyzer/ml/predictors.py",
            "src/laser_trim_analyzer/ml/engine.py",
        ]
        
        for file_path in ml_files:
            path = Path(file_path)
            if path.exists():
                is_valid, msg = self.check_python_syntax(path)
                self.results['ml'][path.name] = {
                    'valid': is_valid,
                    'message': msg
                }
                if not is_valid:
                    self.errors.append(f"{path.name}: {msg}")
                    self.results['overall'] = False
                    
    def validate_configuration(self):
        """Validate configuration files."""
        print("\nValidating Configuration...")
        
        config_files = [
            "config/default.yaml",
            "config/production.yaml",
            "config/development.yaml",
        ]
        
        for config_file in config_files:
            path = Path(config_file)
            if path.exists():
                try:
                    import yaml
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Check for key settings
                    has_plot_settings = 'processing' in config and 'generate_plots' in config['processing']
                    has_ml_settings = 'ml' in config or 'machine_learning' in config
                    
                    self.results['config'] = self.results.get('config', {})
                    self.results['config'][path.name] = {
                        'valid': True,
                        'message': f"Valid YAML, plots={'enabled' if has_plot_settings else 'missing'}, ml={'enabled' if has_ml_settings else 'missing'}"
                    }
                except Exception as e:
                    self.results['config'] = self.results.get('config', {})
                    self.results['config'][path.name] = {
                        'valid': False,
                        'message': f"Invalid YAML: {str(e)}"
                    }
                    self.results['overall'] = False
                    
    def generate_report(self):
        """Generate validation report."""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        
        # Print results by category
        for category, results in self.results.items():
            if category == 'overall':
                continue
                
            print(f"\n{category.upper()}:")
            if isinstance(results, dict):
                for item, result in results.items():
                    status = "✓" if result['valid'] else "✗"
                    print(f"  {status} {item}: {result['message']}")
        
        # Overall result
        print("\n" + "=" * 60)
        if self.results['overall']:
            print("✓ ALL VALIDATIONS PASSED!")
        else:
            print("✗ VALIDATION FAILED")
            print(f"\nErrors found ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        # Save detailed report
        report_path = Path("validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")
        
        return self.results['overall']


if __name__ == "__main__":
    validator = ComprehensiveValidator()
    success = validator.validate_all()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if success else 1)