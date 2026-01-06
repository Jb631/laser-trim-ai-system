#!/usr/bin/env python3
"""Basic test to verify plot generation is working."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

def test_basic_plot():
    """Test basic plot generation."""
    print("Testing basic plot generation...")
    
    # Create test data
    positions = np.linspace(0, 100, 100)
    errors = np.random.normal(0, 0.01, 100)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(positions, errors, 'b-', linewidth=2, label='Test Data')
    ax.axhline(y=0.05, color='r', linestyle='--', label='Spec Limit')
    ax.axhline(y=-0.05, color='r', linestyle='--')
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Error')
    ax.set_title('Test Plot Generation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save plot
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"test_plot_{timestamp}.png"
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if output_path.exists():
        file_size = output_path.stat().st_size / 1024
        print(f"✓ Plot generated successfully: {output_path}")
        print(f"  File size: {file_size:.1f} KB")
        return True
    else:
        print("✗ Plot generation failed")
        return False


def test_plot_viewer_code():
    """Test if plot viewer code is syntactically correct."""
    print("\nTesting plot viewer code...")
    
    try:
        import ast
        plot_viewer_path = Path("src/laser_trim_analyzer/gui/widgets/plot_viewer.py")
        
        if plot_viewer_path.exists():
            code = plot_viewer_path.read_text()
            ast.parse(code)
            print("✓ Plot viewer code is syntactically correct")
            
            # Check for key components
            if "PlotViewerWidget" in code:
                print("✓ PlotViewerWidget class found")
            if "load_plot" in code:
                print("✓ load_plot method found")
            if "_zoom_in" in code and "_zoom_out" in code:
                print("✓ Zoom methods found")
            if "_export_plot" in code:
                print("✓ Export method found")
                
            return True
        else:
            print("✗ Plot viewer file not found")
            return False
            
    except SyntaxError as e:
        print(f"✗ Syntax error in plot viewer: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking plot viewer: {e}")
        return False


if __name__ == "__main__":
    print("=== Plot Functionality Test ===\n")
    
    plot_ok = test_basic_plot()
    viewer_ok = test_plot_viewer_code()
    
    print("\n=== Summary ===")
    print(f"Basic plot generation: {'✓ PASS' if plot_ok else '✗ FAIL'}")
    print(f"Plot viewer code: {'✓ PASS' if viewer_ok else '✗ FAIL'}")
    
    if plot_ok and viewer_ok:
        print("\nAll plot functionality tests passed!")
    else:
        print("\nSome tests failed. Check the output above.")