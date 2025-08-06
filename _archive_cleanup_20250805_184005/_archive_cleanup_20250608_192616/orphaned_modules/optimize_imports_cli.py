"""Command-line interface for the import optimization tool."""

import argparse
import sys
from pathlib import Path
from typing import Optional
import json
from collections import defaultdict
from tabulate import tabulate

from .import_optimizer import (
    ImportAnalyzer, ImportOptimizer, ImportCleaner,
    analyze_imports, optimize_imports
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze and optimize Python imports",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'target',
        nargs='?',
        type=Path,
        default=None,
        help='File or directory to analyze (default: current directory)'
    )
    
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze imports without optimization'
    )
    
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Optimize imports (remove unused, sort, etc.)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Show what would be changed without modifying files (default: True)'
    )
    
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually apply the optimizations (disables dry-run)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Show detailed output'
    )
    
    parser.add_argument(
        '--show-circular',
        action='store_true',
        help='Show circular import dependencies'
    )
    
    parser.add_argument(
        '--show-performance',
        action='store_true',
        help='Show import performance metrics'
    )
    
    parser.add_argument(
        '--show-heavy',
        action='store_true',
        help='Show heavy module imports that could be lazy-loaded'
    )
    
    args = parser.parse_args()
    
    # Determine target
    target = args.target or Path.cwd()
    
    # If applying changes, disable dry-run
    if args.apply:
        args.dry_run = False
    
    try:
        if args.optimize:
            run_optimization(target, args)
        else:
            run_analysis(target, args)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_analysis(target: Path, args):
    """Run import analysis."""
    print(f"Analyzing imports in: {target}")
    print("-" * 60)
    
    analyzer = ImportAnalyzer()
    
    if target.is_file():
        results = {str(target): analyzer.analyze_file(target)}
    else:
        results = analyzer.analyze_project()
        
    if args.json:
        # Convert to JSON-serializable format
        json_results = {}
        for file_path, analysis in results.items():
            json_results[file_path] = {
                'imports': [
                    {
                        'module': imp.module,
                        'names': imp.names,
                        'alias': imp.alias,
                        'line': imp.line,
                        'is_from_import': imp.is_from_import
                    }
                    for imp in analysis.imports
                ],
                'unused_imports': [
                    {
                        'module': imp.module,
                        'line': imp.line
                    }
                    for imp in analysis.unused_imports
                ],
                'circular_imports': analysis.circular_imports,
                'suggestions': analysis.suggestions,
                'import_time': analysis.import_time,
                'memory_usage': analysis.memory_usage
            }
        print(json.dumps(json_results, indent=2))
    else:
        # Display human-readable results
        display_analysis_results(results, args)


def display_analysis_results(results, args):
    """Display analysis results in a human-readable format."""
    total_files = len(results)
    total_imports = sum(len(a.imports) for a in results.values())
    total_unused = sum(len(a.unused_imports) for a in results.values())
    
    print(f"\nSummary:")
    print(f"  Files analyzed: {total_files}")
    print(f"  Total imports: {total_imports}")
    print(f"  Unused imports: {total_unused}")
    
    if args.verbose or total_unused > 0:
        print(f"\nUnused Imports by File:")
        for file_path, analysis in results.items():
            if analysis.unused_imports:
                print(f"\n  {file_path}:")
                for imp in analysis.unused_imports:
                    print(f"    Line {imp.line}: {imp.module}")
                    
    if args.show_circular:
        print(f"\nCircular Import Dependencies:")
        circular_found = False
        for file_path, analysis in results.items():
            if analysis.circular_imports:
                circular_found = True
                print(f"\n  {file_path}:")
                for circular in analysis.circular_imports:
                    print(f"    {circular}")
                    
        if not circular_found:
            print("  No circular imports detected")
            
    if args.show_performance:
        print(f"\nImport Performance Metrics:")
        all_times = {}
        for analysis in results.values():
            all_times.update(analysis.import_time)
            
        if all_times:
            sorted_times = sorted(all_times.items(), key=lambda x: x[1], reverse=True)
            table_data = [
                [module, f"{time:.3f}s", f"{results[list(results.keys())[0]].memory_usage.get(module, 0) / 1024 / 1024:.1f}MB"]
                for module, time in sorted_times[:10]
            ]
            print(tabulate(
                table_data,
                headers=["Module", "Import Time", "Memory"],
                tablefmt="simple"
            ))
            
    if args.show_heavy:
        print(f"\nHeavy Module Imports:")
        heavy_modules = {
            'numpy', 'pandas', 'matplotlib', 'scipy', 'tensorflow',
            'torch', 'sklearn', 'cv2', 'PIL', 'plotly', 'seaborn'
        }
        
        heavy_found = defaultdict(list)
        for file_path, analysis in results.items():
            for imp in analysis.imports:
                if any(heavy in imp.module for heavy in heavy_modules):
                    heavy_found[imp.module].append(file_path)
                    
        if heavy_found:
            for module, files in heavy_found.items():
                print(f"\n  {module}:")
                for file_path in files[:5]:  # Show max 5 files
                    print(f"    - {file_path}")
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more files")
                    
    # Show suggestions
    print(f"\nOptimization Suggestions:")
    suggestion_count = defaultdict(int)
    for analysis in results.values():
        for suggestion in analysis.suggestions:
            suggestion_count[suggestion] += 1
            
    if suggestion_count:
        for suggestion, count in sorted(suggestion_count.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                print(f"  - {suggestion} (in {count} files)")
            else:
                print(f"  - {suggestion}")
    else:
        print("  No optimization suggestions")


def run_optimization(target: Path, args):
    """Run import optimization."""
    print(f"Optimizing imports in: {target}")
    print(f"Mode: {'Dry run' if args.dry_run else 'Apply changes'}")
    print("-" * 60)
    
    results = optimize_imports(target, dry_run=args.dry_run)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Display results
        files_optimized = 0
        total_removed = 0
        
        for result in results:
            if result['status'] == 'optimized':
                files_optimized += 1
                print(f"\n✓ {result['file']}")
                for change in result['changes']:
                    print(f"  - {change}")
                    if 'Removed unused import' in change:
                        total_removed += 1
                        
            elif result['status'] == 'dry_run' and result.get('unused_imports', 0) > 0:
                print(f"\n→ {result['file']}")
                print(f"  - Would remove {result['unused_imports']} unused imports")
                for suggestion in result.get('suggestions', []):
                    print(f"  - {suggestion}")
                    
        print(f"\nSummary:")
        if args.dry_run:
            print(f"  Would optimize {len([r for r in results if r.get('unused_imports', 0) > 0])} files")
        else:
            print(f"  Optimized {files_optimized} files")
            print(f"  Removed {total_removed} unused imports")


if __name__ == '__main__':
    main()