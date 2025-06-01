#!/usr/bin/env python3
"""
Fix all syntax errors related to imports in the project
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


class SyntaxErrorFixer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixes_made = []

    def fix_all_files(self):
        """Fix syntax errors in all Python files"""
        print(f"Scanning for syntax errors in: {self.project_root}")
        print("-" * 80)

        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            # Skip virtual environments and cache
            if any(part in file_path.parts for part in ['.venv', 'venv', '__pycache__', '.git']):
                continue

            if self._fix_file(file_path):
                print(f"✓ Fixed: {file_path.relative_to(self.project_root)}")

        self._report_results()

    def _fix_file(self, file_path: Path) -> bool:
        """Fix syntax errors in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Fix pattern 1: from ... import
            content = re.sub(r'\)from\s+', 'from ', content)

            # Fix pattern 2: from ... import
            content = re.sub(r'\)\s*from\s+', ')\nfrom ', content)

            # Fix pattern 3: Multiple imports on one line after closing paren
            content = re.sub(r'\)([a-z]+\s+import)', ')\n\\1', content)

            # Fix unclosed parentheses in imports
            content = self._fix_unclosed_parens(content)

            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.fixes_made.append(str(file_path.relative_to(self.project_root)))
                return True

        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")

        return False

    def _fix_unclosed_parens(self, content: str) -> str:
        """Fix unclosed parentheses in import statements"""
        lines = content.splitlines()
        fixed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for multi-line import with unclosed parenthesis
            if 'from typing import (' in line or re.match(r'from .+ import \($', line.strip()):
                # Start of multi-line import
                import_lines = [line]
                open_parens = line.count('(') - line.count(')')
                i += 1

                # Collect lines until parentheses are balanced
                while i < len(lines) and open_parens > 0:
                    import_lines.append(lines[i])
                    open_parens += lines[i].count('(') - lines[i].count(')')
                    i += 1

                # If parentheses are not balanced, add closing paren
                if open_parens > 0:
                    # Find the last non-empty line
                    for j in range(len(import_lines) - 1, -1, -1):
                        if import_lines[j].strip():
                            import_lines[j] = import_lines[j].rstrip() + '\n)'
                            break

                fixed_lines.extend(import_lines)
            else:
                fixed_lines.append(line)
                i += 1

        return '\n'.join(fixed_lines)

    def _report_results(self):
        """Report the results"""
        print("\n" + "=" * 80)
        print("SYNTAX ERROR FIX SUMMARY")
        print("=" * 80)

        if not self.fixes_made:
            print("No syntax errors found!")
        else:
            print(f"Fixed {len(self.fixes_made)} files:")
            for file in self.fixes_made:
                print(f"  - {file}")

        print("=" * 80)


def quick_fix_specific_files(project_root: Path):
    """Quick fix for the specific files we know have issues"""
    files_to_fix = [
        "src/laser_trim_analyzer/database/manager.py",
        "src/laser_trim_analyzer/analysis/base.py",
    ]

    for file_path in files_to_fix:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"\nQuick fixing: {file_path}")

            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Fix the specific syntax errors
            fixed_lines = []
            for i, line in enumerate(lines):
                # Fix from pattern
                if ')from' in line:
                    line = line.replace(')from', ')\nfrom')
                    print(f"  Fixed line {i + 1}: removed ')from' pattern")

                # Fix unclosed parenthesis in typing import
                if 'from typing import (' in line and i + 1 < len(lines):
                    # Check if the import is properly closed
                    j = i + 1
                    found_close = False
                    while j < len(lines) and j < i + 10:  # Check next 10 lines max
                        if ')' in lines[j]:
                            found_close = True
                            break
                        j += 1

                    if not found_close:
                        # Insert proper closing
                        # Find the last import line
                        last_import = i
                        while last_import + 1 < len(lines) and lines[last_import + 1].strip().startswith(
                                ('    ', '\t')):
                            last_import += 1

                        lines[last_import] = lines[last_import].rstrip() + '\n)\n'
                        print(f"  Fixed unclosed parenthesis in typing import")

                fixed_lines.append(line)

            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)


def main():
    # Get the project root directory
    project_root = input("Enter the path to your project root (or press Enter for current directory): ").strip()
    if not project_root:
        project_root = os.getcwd()

    project_root = Path(project_root)

    if not project_root.exists():
        print(f"Error: Path '{project_root}' does not exist")
        return

    print("\nThis script will fix syntax errors in import statements.")
    print("Common issues it fixes:")
    print("  - ')from' patterns")
    print("  - Unclosed parentheses in multi-line imports")
    print()

    response = input("Do you want to proceed? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return

    # First do quick fixes for known problematic files
    quick_fix_specific_files(project_root)

    # Then scan all files
    fixer = SyntaxErrorFixer(project_root)
    fixer.fix_all_files()

    print("\n✅ All syntax error fixes completed!")
    print("\nTry running your application again.")


if __name__ == "__main__":
    main()