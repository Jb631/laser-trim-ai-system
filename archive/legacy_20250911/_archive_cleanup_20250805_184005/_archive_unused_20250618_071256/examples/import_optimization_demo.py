"""Demo script showing import optimization capabilities."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from laser_trim_analyzer.utils.import_optimizer import ImportAnalyzer, ImportOptimizer
from laser_trim_analyzer.utils.lazy_imports import lazy_imports, lazy_import, temporary_import


def demo_import_analysis():
    """Demonstrate import analysis capabilities."""
    print("=== Import Analysis Demo ===\n")
    
    analyzer = ImportAnalyzer()
    
    # Analyze a specific file
    target_file = Path(__file__).parent.parent / 'src' / 'laser_trim_analyzer' / 'core' / 'processor.py'
    
    if target_file.exists():
        print(f"Analyzing: {target_file}")
        analysis = analyzer.analyze_file(target_file)
        
        print(f"\nImport Summary:")
        print(f"  Total imports: {len(analysis.imports)}")
        print(f"  Unused imports: {len(analysis.unused_imports)}")
        print(f"  Circular imports: {len(analysis.circular_imports)}")
        
        if analysis.unused_imports:
            print(f"\nUnused imports found:")
            for imp in analysis.unused_imports[:5]:
                print(f"  - Line {imp.line}: {imp.module}")
                
        if analysis.suggestions:
            print(f"\nSuggestions:")
            for suggestion in analysis.suggestions:
                print(f"  - {suggestion}")
                
        # Show performance metrics
        if analysis.import_time:
            print(f"\nImport Performance:")
            sorted_times = sorted(analysis.import_time.items(), key=lambda x: x[1], reverse=True)
            for module, time in sorted_times[:5]:
                print(f"  - {module}: {time:.3f}s")


def demo_lazy_loading():
    """Demonstrate lazy loading capabilities."""
    print("\n\n=== Lazy Loading Demo ===\n")
    
    # Check if numpy is loaded
    print(f"Numpy loaded initially: {lazy_imports.is_loaded('numpy')}")
    
    # Use numpy lazily
    print("Accessing numpy...")
    np = lazy_imports.np
    array = np.array([1, 2, 3])
    print(f"Created array: {array}")
    print(f"Numpy loaded now: {lazy_imports.is_loaded('numpy')}")
    
    # Show loaded modules
    print(f"\nCurrently loaded lazy modules: {lazy_imports.get_loaded_modules()}")


def demo_lazy_decorator():
    """Demonstrate lazy import decorator."""
    print("\n\n=== Lazy Decorator Demo ===\n")
    
    @lazy_import('json')
    def process_json_data():
        # json module is imported only when function is called
        data = {'key': 'value', 'number': 42}
        return json.dumps(data, indent=2)
    
    print("Function defined (json not imported yet)")
    result = process_json_data()
    print(f"Function result:\n{result}")


def demo_temporary_import():
    """Demonstrate temporary import context manager."""
    print("\n\n=== Temporary Import Demo ===\n")
    
    import_name = 'collections'
    
    print(f"'{import_name}' in sys.modules before: {import_name in sys.modules}")
    
    with temporary_import(import_name) as collections:
        print(f"'{import_name}' in sys.modules during: {import_name in sys.modules}")
        counter = collections.Counter(['a', 'b', 'a', 'c', 'b', 'a'])
        print(f"Counter result: {counter}")
    
    print(f"'{import_name}' in sys.modules after: {import_name in sys.modules}")


def demo_import_optimization():
    """Demonstrate import optimization."""
    print("\n\n=== Import Optimization Demo ===\n")
    
    # Create a sample file with unused imports
    sample_file = Path(__file__).parent / 'sample_with_unused.py'
    
    sample_content = '''"""Sample module with unused imports."""
import os
import sys
import json  # unused
import time
from pathlib import Path
from typing import List, Dict  # Dict is unused

def process_files(files: List[Path]):
    """Process a list of files."""
    for file in files:
        if file.exists():
            print(f"Processing {file}")
            time.sleep(0.1)
            
    return os.getcwd()

if __name__ == "__main__":
    process_files([Path("test.txt")])
'''
    
    # Write sample file
    sample_file.write_text(sample_content)
    
    try:
        # Analyze the file
        analyzer = ImportAnalyzer()
        optimizer = ImportOptimizer(analyzer)
        
        analysis = analyzer.analyze_file(sample_file)
        print(f"Analysis of {sample_file.name}:")
        print(f"  Unused imports: {[imp.module for imp in analysis.unused_imports]}")
        
        # Optimize (dry run)
        result = optimizer.optimize_file(sample_file, dry_run=True)
        print(f"\nOptimization result (dry run):")
        print(f"  Status: {result['status']}")
        print(f"  Unused imports to remove: {result.get('unused_imports', 0)}")
        
    finally:
        # Clean up
        if sample_file.exists():
            sample_file.unlink()


def main():
    """Run all demos."""
    demos = [
        demo_import_analysis,
        demo_lazy_loading,
        demo_lazy_decorator,
        demo_temporary_import,
        demo_import_optimization
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\nError in {demo.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n\n=== Demo Complete ===")


if __name__ == "__main__":
    main()