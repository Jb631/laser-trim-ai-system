#!/usr/bin/env python3
"""
Test script to find plot generation and display issues in the laser trim analyzer.

This script will:
1. Check if plots are being generated
2. Find where plots should be saved
3. Check if plots are displayed in the GUI
4. Identify any missing plot display functionality
"""

import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_plot_generation():
    """Check if plot generation is configured and working."""
    logger.info("=== Checking Plot Generation Configuration ===")
    
    try:
        from laser_trim_analyzer.core.config import get_config
        config = get_config()
        
        logger.info(f"Plot generation enabled: {config.processing.generate_plots}")
        logger.info(f"Plot DPI: {config.processing.plot_dpi}")
        
        # Check if plotting utilities are available
        from laser_trim_analyzer.utils.plotting_utils import create_analysis_plot
        logger.info("✓ Plotting utilities available")
        
        # Check matplotlib
        import matplotlib
        logger.info(f"✓ Matplotlib version: {matplotlib.__version__}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Plot generation check failed: {e}")
        return False

def find_plot_paths():
    """Find where plots are saved."""
    logger.info("\n=== Finding Plot Storage Locations ===")
    
    try:
        from laser_trim_analyzer.core.processor import LaserTrimProcessor
        from laser_trim_analyzer.core.config import get_config
        
        config = get_config()
        processor = LaserTrimProcessor(config)
        
        # Check default output directories
        possible_dirs = [
            Path.cwd() / "output",
            Path.cwd() / "plots",
            Path.cwd() / "results",
            Path.home() / ".laser_trim_analyzer" / "plots",
            Path.home() / ".laser_trim_analyzer" / "output"
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists():
                logger.info(f"✓ Found directory: {dir_path}")
                # List any plot files
                plot_files = list(dir_path.glob("*.png")) + list(dir_path.glob("*.pdf"))
                if plot_files:
                    logger.info(f"  Found {len(plot_files)} plot files")
                    for pf in plot_files[:5]:  # Show first 5
                        logger.info(f"    - {pf.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Plot path check failed: {e}")
        return False

def check_plot_display_in_gui():
    """Check if GUI has plot display functionality."""
    logger.info("\n=== Checking Plot Display in GUI ===")
    
    issues = []
    
    try:
        # Check if AnalysisDisplayWidget shows plots
        from laser_trim_analyzer.gui.widgets.analysis_display import AnalysisDisplayWidget
        
        # Check source code for plot display
        import inspect
        source = inspect.getsource(AnalysisDisplayWidget)
        
        if "plot" not in source.lower():
            issues.append("AnalysisDisplayWidget doesn't seem to display plots")
            logger.warning("✗ AnalysisDisplayWidget missing plot display functionality")
        
        # Check FileAnalysisWidget
        from laser_trim_analyzer.gui.widgets.file_analysis_widget import FileAnalysisWidget
        source = inspect.getsource(FileAnalysisWidget)
        
        if "plot_path" in source:
            logger.info("✓ FileAnalysisWidget has plot_path support")
        else:
            issues.append("FileAnalysisWidget missing plot_path support")
        
        # Check for plot viewer
        plot_viewer_found = False
        gui_path = Path(__file__).parent / "src" / "laser_trim_analyzer" / "gui"
        
        for py_file in gui_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                if any(term in content for term in ["PlotViewer", "ImageViewer", "show_plot", "display_plot"]):
                    logger.info(f"✓ Found plot display code in: {py_file.name}")
                    plot_viewer_found = True
            except:
                pass
        
        if not plot_viewer_found:
            issues.append("No dedicated plot viewer found in GUI")
            logger.warning("✗ No dedicated plot viewer widget found")
        
        return issues
        
    except Exception as e:
        logger.error(f"✗ GUI check failed: {e}")
        return [str(e)]

def test_plot_generation_flow():
    """Test the complete plot generation flow."""
    logger.info("\n=== Testing Plot Generation Flow ===")
    
    try:
        from laser_trim_analyzer.core.processor import LaserTrimProcessor
        from laser_trim_analyzer.core.config import get_config
        from laser_trim_analyzer.core.models import TrackData, SigmaAnalysis, LinearityAnalysis
        from laser_trim_analyzer.utils.plotting_utils import create_analysis_plot
        import numpy as np
        
        # Create test data
        track_data = TrackData(
            track_id="test_track",
            position_data=list(range(100)),
            error_data=list(np.random.normal(0, 0.01, 100)),
            untrimmed_positions=list(range(100)),
            untrimmed_errors=list(np.random.normal(0, 0.02, 100)),
            sigma_analysis=SigmaAnalysis(
                sigma_gradient=0.015,
                sigma_threshold=0.02,
                sigma_pass=True
            ),
            linearity_analysis=LinearityAnalysis(
                linearity_spec=0.05,
                linearity_pass=True,
                linearity_fail_points=0,
                optimal_offset=0.001,
                final_linearity_error_shifted=0.03
            )
        )
        
        # Create output directory
        output_dir = Path.cwd() / "test_plots"
        output_dir.mkdir(exist_ok=True)
        
        # Generate plot
        plot_path = create_analysis_plot(
            track_data,
            output_dir,
            "test_plot",
            dpi=150
        )
        
        if plot_path.exists():
            logger.info(f"✓ Test plot generated successfully: {plot_path}")
            logger.info(f"  File size: {plot_path.stat().st_size / 1024:.1f} KB")
            return True
        else:
            logger.error("✗ Plot file not created")
            return False
            
    except Exception as e:
        logger.error(f"✗ Plot generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_fixes():
    """Suggest fixes for plot display issues."""
    logger.info("\n=== Suggested Fixes ===")
    
    logger.info("""
1. Add plot display to AnalysisDisplayWidget:
   - Add a plot_image_label to show the generated plot
   - Load plot using PIL/Pillow and display with CTkImage
   
2. Implement plot viewer dialog:
   - Create a PlotViewerDialog class
   - Support zoom, pan, and export functionality
   - Open plots in external viewer as fallback
   
3. Ensure plot paths are stored:
   - Verify TrackData.plot_path is set after generation
   - Pass plot_path through to GUI components
   
4. Add "View Plot" functionality:
   - Implement callback in single_file_page.py
   - Add plot viewing to batch results
   
5. Check output directory:
   - Ensure output directory is created
   - Use consistent output path for all plots
""")

def main():
    """Run all plot-related checks."""
    logger.info("Starting plot generation and display diagnostics...\n")
    
    # Run checks
    plot_gen_ok = check_plot_generation()
    plot_paths_ok = find_plot_paths()
    gui_issues = check_plot_display_in_gui()
    test_ok = test_plot_generation_flow()
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Plot generation configured: {'✓' if plot_gen_ok else '✗'}")
    logger.info(f"Plot paths found: {'✓' if plot_paths_ok else '✗'}")
    logger.info(f"GUI plot display issues: {len(gui_issues)}")
    
    if gui_issues:
        logger.warning("GUI Issues found:")
        for issue in gui_issues:
            logger.warning(f"  - {issue}")
    
    logger.info(f"Test plot generation: {'✓' if test_ok else '✗'}")
    
    # Provide fixes if issues found
    if gui_issues or not test_ok:
        suggest_fixes()
    
    return len(gui_issues) == 0 and test_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)