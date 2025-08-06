"""Import optimization tools for analyzing and optimizing imports throughout the codebase."""

import ast
import os
import sys
import time
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import tracemalloc
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str
    names: List[str] = field(default_factory=list)
    alias: Optional[str] = None
    line: int = 0
    is_from_import: bool = False
    is_relative: bool = False
    level: int = 0  # Relative import level


@dataclass
class ImportAnalysis:
    """Results of import analysis for a file."""
    file_path: str
    imports: List[ImportInfo] = field(default_factory=list)
    used_names: Set[str] = field(default_factory=set)
    unused_imports: List[ImportInfo] = field(default_factory=list)
    circular_imports: List[str] = field(default_factory=list)
    import_time: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, int] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


class ImportAnalyzer:
    """Analyzes Python imports for optimization opportunities."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.module_cache: Dict[str, Any] = {}
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.heavy_modules = {
            'numpy', 'pandas', 'matplotlib', 'scipy', 'tensorflow',
            'torch', 'sklearn', 'cv2', 'PIL', 'plotly', 'seaborn'
        }
        
    def analyze_file(self, file_path: Path) -> ImportAnalysis:
        """Analyze imports in a single Python file."""
        analysis = ImportAnalysis(file_path=str(file_path))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Extract imports
            imports = self._extract_imports(tree)
            analysis.imports = imports
            
            # Extract used names
            used_names = self._extract_used_names(tree)
            analysis.used_names = used_names
            
            # Find unused imports
            analysis.unused_imports = self._find_unused_imports(imports, used_names)
            
            # Measure import performance
            self._measure_import_performance(imports, analysis)
            
            # Check for circular imports
            module_name = self._get_module_name(file_path)
            analysis.circular_imports = self._check_circular_imports(module_name, imports)
            
            # Generate suggestions
            analysis.suggestions = self._generate_suggestions(analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            
        return analysis
    
    def analyze_project(self, extensions: List[str] = None) -> Dict[str, ImportAnalysis]:
        """Analyze all Python files in the project."""
        if extensions is None:
            extensions = ['.py']
            
        results = {}
        
        for file_path in self._find_python_files(self.project_root, extensions):
            relative_path = file_path.relative_to(self.project_root)
            results[str(relative_path)] = self.analyze_file(file_path)
            
        return results
    
    def _extract_imports(self, tree: ast.AST) -> List[ImportInfo]:
        """Extract all import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        alias=alias.asname,
                        line=node.lineno,
                        is_from_import=False
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                level = node.level
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=module,
                        names=[alias.name],
                        alias=alias.asname,
                        line=node.lineno,
                        is_from_import=True,
                        is_relative=level > 0,
                        level=level
                    ))
                    
        return imports
    
    def _extract_used_names(self, tree: ast.AST) -> Set[str]:
        """Extract all names used in the code."""
        used_names = set()
        
        class NameVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                used_names.add(node.id)
                self.generic_visit(node)
                
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
                self.generic_visit(node)
                
        visitor = NameVisitor()
        visitor.visit(tree)
        
        return used_names
    
    def _find_unused_imports(self, imports: List[ImportInfo], used_names: Set[str]) -> List[ImportInfo]:
        """Find imports that are not used in the code."""
        unused = []
        
        for imp in imports:
            # Check if the import or its alias is used
            name_to_check = imp.alias or imp.module.split('.')[0]
            
            if imp.is_from_import and imp.names:
                # For from imports, check each imported name
                for name in imp.names:
                    actual_name = imp.alias or name
                    if actual_name not in used_names:
                        unused.append(imp)
            elif name_to_check not in used_names:
                unused.append(imp)
                
        return unused
    
    def _measure_import_performance(self, imports: List[ImportInfo], analysis: ImportAnalysis):
        """Measure the performance impact of imports."""
        for imp in imports:
            module_name = imp.module
            
            if module_name in self.module_cache:
                continue
                
            try:
                # Measure import time
                start_time = time.time()
                tracemalloc.start()
                
                module = importlib.import_module(module_name)
                
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                import_time = time.time() - start_time
                
                analysis.import_time[module_name] = import_time
                analysis.memory_usage[module_name] = current
                
                self.module_cache[module_name] = module
                
            except ImportError:
                logger.debug(f"Could not import {module_name} for performance analysis")
    
    def _check_circular_imports(self, module_name: str, imports: List[ImportInfo]) -> List[str]:
        """Check for circular import dependencies."""
        circular = []
        
        # Build import graph
        for imp in imports:
            self.import_graph[module_name].add(imp.module)
            
        # Check for cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.import_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    circular.append(f"{node} -> {neighbor}")
                    return True
                    
            rec_stack.remove(node)
            return False
            
        if module_name not in visited:
            has_cycle(module_name)
            
        return circular
    
    def _generate_suggestions(self, analysis: ImportAnalysis) -> List[str]:
        """Generate optimization suggestions based on analysis."""
        suggestions = []
        
        # Unused imports
        if analysis.unused_imports:
            suggestions.append(f"Remove {len(analysis.unused_imports)} unused imports")
            
        # Heavy imports
        heavy_imports = [
            imp.module for imp in analysis.imports 
            if any(heavy in imp.module for heavy in self.heavy_modules)
        ]
        if heavy_imports:
            suggestions.append(f"Consider lazy loading for heavy modules: {', '.join(heavy_imports)}")
            
        # Slow imports
        slow_imports = [
            (module, time) for module, time in analysis.import_time.items() 
            if time > 0.1
        ]
        if slow_imports:
            suggestions.append(f"Slow imports detected: {', '.join(f'{m} ({t:.2f}s)' for m, t in slow_imports)}")
            
        # Circular imports
        if analysis.circular_imports:
            suggestions.append(f"Circular imports detected: {', '.join(analysis.circular_imports)}")
            
        # Import organization
        if len(analysis.imports) > 20:
            suggestions.append("Consider breaking up this module - too many imports")
            
        return suggestions
    
    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        try:
            relative_path = file_path.relative_to(self.project_root)
            parts = relative_path.with_suffix('').parts
            return '.'.join(parts)
        except ValueError:
            return file_path.stem
            
    def _find_python_files(self, root: Path, extensions: List[str]) -> List[Path]:
        """Find all Python files in the project."""
        files = []
        for ext in extensions:
            files.extend(root.rglob(f'*{ext}'))
        return [f for f in files if f.is_file() and 'venv' not in f.parts and '__pycache__' not in f.parts]


class ImportOptimizer:
    """Optimizes imports based on analysis results."""
    
    def __init__(self, analyzer: ImportAnalyzer):
        self.analyzer = analyzer
        
    def optimize_file(self, file_path: Path, dry_run: bool = True) -> Dict[str, Any]:
        """Optimize imports in a single file."""
        analysis = self.analyzer.analyze_file(file_path)
        
        if not analysis.unused_imports and not analysis.suggestions:
            return {'status': 'no_changes_needed', 'file': str(file_path)}
            
        if dry_run:
            return {
                'status': 'dry_run',
                'file': str(file_path),
                'unused_imports': len(analysis.unused_imports),
                'suggestions': analysis.suggestions
            }
            
        # Apply optimizations
        changes = self._apply_optimizations(file_path, analysis)
        
        return {
            'status': 'optimized',
            'file': str(file_path),
            'changes': changes
        }
        
    def _apply_optimizations(self, file_path: Path, analysis: ImportAnalysis) -> List[str]:
        """Apply import optimizations to a file."""
        changes = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Remove unused imports
        lines_to_remove = set()
        for unused in analysis.unused_imports:
            lines_to_remove.add(unused.line - 1)  # Convert to 0-based index
            changes.append(f"Removed unused import: {unused.module}")
            
        # Filter out removed lines
        new_lines = [
            line for i, line in enumerate(lines) 
            if i not in lines_to_remove
        ]
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
        return changes
        
    def generate_lazy_imports(self, analysis_results: Dict[str, ImportAnalysis]) -> Dict[str, List[str]]:
        """Generate lazy import recommendations for heavy modules."""
        lazy_recommendations = defaultdict(list)
        
        for file_path, analysis in analysis_results.items():
            for imp in analysis.imports:
                if any(heavy in imp.module for heavy in self.analyzer.heavy_modules):
                    lazy_recommendations[imp.module].append(file_path)
                    
        return dict(lazy_recommendations)
        
    def generate_import_map(self, analysis_results: Dict[str, ImportAnalysis]) -> Dict[str, Set[str]]:
        """Generate a map of which modules import which other modules."""
        import_map = defaultdict(set)
        
        for file_path, analysis in analysis_results.items():
            module_name = self.analyzer._get_module_name(Path(file_path))
            for imp in analysis.imports:
                import_map[module_name].add(imp.module)
                
        return dict(import_map)


class ImportCleaner:
    """Automated import cleanup utilities."""
    
    @staticmethod
    def sort_imports(file_path: Path) -> None:
        """Sort imports according to PEP 8 conventions."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content)
        
        # Group imports
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_line = f"import {alias.name}"
                    if alias.asname:
                        import_line += f" as {alias.asname}"
                    
                    if ImportCleaner._is_stdlib(alias.name):
                        stdlib_imports.append(import_line)
                    elif ImportCleaner._is_local(alias.name):
                        local_imports.append(import_line)
                    else:
                        third_party_imports.append(import_line)
                        
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:  # Relative import
                    local_imports.append(ast.unparse(node))
                elif ImportCleaner._is_stdlib(node.module):
                    stdlib_imports.append(ast.unparse(node))
                elif ImportCleaner._is_local(node.module):
                    local_imports.append(ast.unparse(node))
                else:
                    third_party_imports.append(ast.unparse(node))
                    
        # Sort each group
        stdlib_imports.sort()
        third_party_imports.sort()
        local_imports.sort()
        
        # TODO: Rewrite file with sorted imports
        
    @staticmethod
    def _is_stdlib(module_name: str) -> bool:
        """Check if a module is part of the standard library."""
        if not module_name:
            return False
            
        stdlib_modules = {
            'os', 'sys', 'time', 'datetime', 'json', 'math', 'random',
            'collections', 'itertools', 'functools', 'pathlib', 'typing',
            'logging', 're', 'ast', 'importlib', 'traceback', 'io',
            'unittest', 'subprocess', 'threading', 'multiprocessing'
        }
        
        return module_name.split('.')[0] in stdlib_modules
        
    @staticmethod
    def _is_local(module_name: str) -> bool:
        """Check if a module is a local/project module."""
        if not module_name:
            return True
            
        # Simple heuristic: if it starts with a known project prefix
        local_prefixes = ['laser_trim_analyzer', 'src', 'tests']
        return any(module_name.startswith(prefix) for prefix in local_prefixes)


def analyze_imports(target: Optional[Path] = None) -> Dict[str, ImportAnalysis]:
    """Convenience function to analyze imports in a project or file."""
    analyzer = ImportAnalyzer()
    
    if target is None:
        return analyzer.analyze_project()
    elif target.is_file():
        return {str(target): analyzer.analyze_file(target)}
    elif target.is_dir():
        analyzer.project_root = target
        return analyzer.analyze_project()
    else:
        raise ValueError(f"Invalid target: {target}")


def optimize_imports(target: Optional[Path] = None, dry_run: bool = True) -> List[Dict[str, Any]]:
    """Convenience function to optimize imports."""
    analyzer = ImportAnalyzer()
    optimizer = ImportOptimizer(analyzer)
    
    results = []
    
    if target is None:
        for file_path, _ in analyzer.analyze_project().items():
            result = optimizer.optimize_file(Path(file_path), dry_run=dry_run)
            results.append(result)
    elif target.is_file():
        result = optimizer.optimize_file(target, dry_run=dry_run)
        results.append(result)
    elif target.is_dir():
        analyzer.project_root = target
        for file_path, _ in analyzer.analyze_project().items():
            result = optimizer.optimize_file(Path(file_path), dry_run=dry_run)
            results.append(result)
            
    return results