"""
Script to remove analysis methods from multi_track_page.py that have been moved to AnalysisMixin.

Methods to remove:
- _select_track_file
- _analyze_folder
- _analyze_track_file
- _run_file_analysis
- _analyze_folder_tracks
- _run_folder_analysis
- _show_unit_selection_dialog
"""

import re
import sys
from pathlib import Path


def remove_method_by_lines(content: str, method_name: str) -> str:
    """Remove a method definition from the content by finding start line and tracking indentation."""
    lines = content.split('\n')
    result_lines = []
    in_method = False
    method_indent = None
    removed_lines = 0
    method_start = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped) if stripped else 9999  # Empty lines have "infinite" indent

        if in_method:
            # Check if we've reached the end of the method
            # A method ends when we find a line with <= method_indent that's not empty
            if stripped and current_indent <= method_indent:
                # We've hit the next method or class-level item
                in_method = False
                method_indent = None
                # Don't skip this line - it's the start of the next item
            else:
                # Still in the method body, skip this line
                removed_lines += 1
                i += 1
                continue

        # Check if this is the start of the method we want to remove
        # Handle multi-line signatures by looking for 'def method_name('
        if f'def {method_name}(' in stripped:
            method_start = i + 1
            method_indent = len(line) - len(stripped)
            in_method = True
            removed_lines += 1

            # Handle multi-line signatures - keep skipping until we see ):
            # (The first line might not have the closing paren)
            if '):' not in line and ')' not in line:
                # Multi-line signature - consume until we find the end
                i += 1
                while i < len(lines):
                    sig_line = lines[i]
                    removed_lines += 1
                    if '):' in sig_line or (')' in sig_line and ':' in sig_line):
                        # Found end of signature
                        break
                    elif ') ->' in sig_line:
                        # Also handle return type annotation
                        i += 1
                        while i < len(lines) and ':' not in lines[i]:
                            removed_lines += 1
                            i += 1
                        if i < len(lines):
                            removed_lines += 1
                        break
                    i += 1

            i += 1
            continue

        result_lines.append(line)
        i += 1

    if removed_lines > 0:
        print(f"  Found {method_name} at line {method_start}, removed {removed_lines} lines")
    else:
        print(f"  WARNING: {method_name} not found!")

    return '\n'.join(result_lines)


def update_class_declaration(content: str) -> str:
    """Update the class declaration to inherit from AnalysisMixin."""
    old = 'class MultiTrackPage(ExportMixin, ctk.CTkFrame):'
    new = 'class MultiTrackPage(AnalysisMixin, ExportMixin, ctk.CTkFrame):'

    if old in content:
        print("  Updating class declaration to inherit from AnalysisMixin")
        content = content.replace(old, new)

    return content


def update_imports(content: str) -> str:
    """Update the import statement to include AnalysisMixin."""
    old_import = 'from laser_trim_analyzer.gui.pages.multi_track.export_mixin import ExportMixin'
    new_import = '''from laser_trim_analyzer.gui.pages.multi_track.export_mixin import ExportMixin
from laser_trim_analyzer.gui.pages.multi_track.analysis_mixin import AnalysisMixin'''

    if old_import in content and 'AnalysisMixin' not in content:
        print("  Updating imports to include AnalysisMixin")
        content = content.replace(old_import, new_import)

    return content


def update_docstring(content: str) -> str:
    """Update the class docstring to mention AnalysisMixin."""
    old_docstring = '''"""Multi-track comparison page for analyzing multiple tracks in a unit.

    Uses mixins for modular functionality:
    - ExportMixin: Export/report functionality (Excel, PDF)
    """'''

    new_docstring = '''"""Multi-track comparison page for analyzing multiple tracks in a unit.

    Uses mixins for modular functionality:
    - AnalysisMixin: File/folder analysis logic
    - ExportMixin: Export/report functionality (Excel, PDF)
    """'''

    if old_docstring in content:
        print("  Updating class docstring")
        content = content.replace(old_docstring, new_docstring)

    return content


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    file_path = project_root / 'src' / 'laser_trim_analyzer' / 'gui' / 'pages' / 'multi_track_page.py'

    print(f"Processing: {file_path}")

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    # Read the file
    content = file_path.read_text(encoding='utf-8')
    original_lines = len(content.split('\n'))
    print(f"Original file: {original_lines} lines")
    print()

    # Update imports first
    print("Step 1: Updating imports and class declaration...")
    content = update_imports(content)
    content = update_class_declaration(content)
    content = update_docstring(content)
    print()

    # Methods to remove (in reverse order of appearance to maintain line numbers)
    methods_to_remove = [
        '_show_unit_selection_dialog',
        '_run_folder_analysis',
        '_analyze_folder_tracks',
        '_run_file_analysis',
        '_analyze_track_file',
        '_analyze_folder',
        '_select_track_file',
    ]

    print("Step 2: Removing methods...")
    for method in methods_to_remove:
        content = remove_method_by_lines(content, method)
    print()

    # Clean up excessive blank lines
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    # Write the result
    file_path.write_text(content, encoding='utf-8')

    new_lines = len(content.split('\n'))
    print(f"Result: {new_lines} lines (removed {original_lines - new_lines} lines)")
    print("Done!")


if __name__ == '__main__':
    main()
