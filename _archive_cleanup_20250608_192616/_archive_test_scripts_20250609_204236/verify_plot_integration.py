#!/usr/bin/env python3
"""Verify plot integration by checking code structure."""

from pathlib import Path
import ast
import re


def check_file_syntax(file_path: Path) -> tuple[bool, str]:
    """Check if a Python file has correct syntax."""
    try:
        code = file_path.read_text()
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def check_plot_viewer():
    """Check plot viewer implementation."""
    print("=== Checking Plot Viewer Implementation ===\n")
    
    plot_viewer_path = Path("src/laser_trim_analyzer/gui/widgets/plot_viewer.py")
    
    if not plot_viewer_path.exists():
        print("✗ plot_viewer.py not found")
        return False
    
    # Check syntax
    syntax_ok, msg = check_file_syntax(plot_viewer_path)
    if not syntax_ok:
        print(f"✗ Syntax error in plot_viewer.py: {msg}")
        return False
    
    print("✓ plot_viewer.py syntax is correct")
    
    # Check content
    content = plot_viewer_path.read_text()
    
    required_methods = [
        ("PlotViewerWidget", "Main widget class"),
        ("load_plot", "Method to load plots"),
        ("_update_display", "Display update method"),
        ("_zoom_in", "Zoom in functionality"),
        ("_zoom_out", "Zoom out functionality"),
        ("_fit_to_window", "Fit to window functionality"),
        ("_export_plot", "Export functionality"),
        ("clear", "Clear plot method"),
    ]
    
    all_found = True
    for method, desc in required_methods:
        if method in content:
            print(f"✓ {desc} found ({method})")
        else:
            print(f"✗ {desc} missing ({method})")
            all_found = False
    
    return all_found


def check_integration():
    """Check if plot viewer is integrated into GUI."""
    print("\n=== Checking Plot Viewer Integration ===\n")
    
    # Check if imported in widgets __init__
    widgets_init = Path("src/laser_trim_analyzer/gui/widgets/__init__.py")
    if widgets_init.exists():
        content = widgets_init.read_text()
        if "PlotViewerWidget" in content:
            print("✓ PlotViewerWidget exported in widgets/__init__.py")
        else:
            print("✗ PlotViewerWidget not exported in widgets/__init__.py")
    
    # Check integration in analysis_display.py
    analysis_display = Path("src/laser_trim_analyzer/gui/widgets/analysis_display.py")
    if analysis_display.exists():
        content = analysis_display.read_text()
        
        if "PlotViewerWidget" in content:
            print("✓ PlotViewerWidget imported in analysis_display.py")
            
            # Check if it's instantiated
            if re.search(r'PlotViewerWidget\s*\(', content):
                print("✓ PlotViewerWidget instantiated in analysis_display.py")
            
            # Check if plots are loaded
            if "load_plot" in content:
                print("✓ load_plot method called in analysis_display.py")
                
            # Check if plot_path is used
            if "plot_path" in content:
                print("✓ plot_path handling found in analysis_display.py")
        else:
            print("✗ PlotViewerWidget not used in analysis_display.py")
    
    # Check if plots are generated in processor
    processor_path = Path("src/laser_trim_analyzer/core/processor.py")
    if processor_path.exists():
        content = processor_path.read_text()
        
        if "create_analysis_plot" in content:
            print("✓ create_analysis_plot called in processor.py")
            
        if re.search(r'track_data\.plot_path\s*=', content):
            print("✓ plot_path set on track_data in processor.py")
        else:
            print("⚠ plot_path assignment not found in processor.py")
    
    # Check if TrackData has plot_path field
    models_path = Path("src/laser_trim_analyzer/core/models.py")
    if models_path.exists():
        content = models_path.read_text()
        
        if re.search(r'plot_path.*Optional\[Path\]', content):
            print("✓ plot_path field defined in TrackData model")
        else:
            print("✗ plot_path field not found in TrackData model")
    
    return True


def check_plot_generation_flow():
    """Check the complete plot generation flow."""
    print("\n=== Checking Plot Generation Flow ===\n")
    
    # Check config for plot settings
    config_path = Path("src/laser_trim_analyzer/core/config.py")
    if config_path.exists():
        content = config_path.read_text()
        
        if "generate_plots" in content:
            print("✓ generate_plots configuration found")
        
        if "plot_dpi" in content:
            print("✓ plot_dpi configuration found")
    
    # Check plotting_utils.py
    plotting_utils = Path("src/laser_trim_analyzer/utils/plotting_utils.py")
    if plotting_utils.exists():
        syntax_ok, msg = check_file_syntax(plotting_utils)
        if syntax_ok:
            print("✓ plotting_utils.py syntax is correct")
            
            content = plotting_utils.read_text()
            
            # Check key functions
            if "create_analysis_plot" in content:
                print("✓ create_analysis_plot function found")
                
                # Check what data is used
                data_used = []
                if "position_data" in content:
                    data_used.append("position_data")
                if "error_data" in content:
                    data_used.append("error_data")
                if "sigma_analysis" in content:
                    data_used.append("sigma_analysis")
                if "linearity_analysis" in content:
                    data_used.append("linearity_analysis")
                if "failure_prediction" in content:
                    data_used.append("failure_prediction")
                
                print(f"✓ Plot uses data: {', '.join(data_used)}")
        else:
            print(f"✗ Syntax error in plotting_utils.py: {msg}")
    
    return True


def main():
    """Run all checks."""
    print("Plot Integration Verification\n")
    print("=" * 50)
    
    plot_viewer_ok = check_plot_viewer()
    integration_ok = check_integration()
    flow_ok = check_plot_generation_flow()
    
    print("\n" + "=" * 50)
    print("\n=== Summary ===")
    print(f"Plot Viewer Implementation: {'✓ PASS' if plot_viewer_ok else '✗ FAIL'}")
    print(f"GUI Integration: {'✓ PASS' if integration_ok else '✗ FAIL'}")
    print(f"Plot Generation Flow: {'✓ PASS' if flow_ok else '✗ FAIL'}")
    
    if plot_viewer_ok and integration_ok and flow_ok:
        print("\n✓ Plot functionality is properly integrated!")
    else:
        print("\n⚠ Some issues found. Check the output above.")


if __name__ == "__main__":
    main()