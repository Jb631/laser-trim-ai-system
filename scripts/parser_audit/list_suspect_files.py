"""Run all FT samples through the parser and list files flagged by SANITY warnings.

Use this whenever you want a fresh list of files whose linearity_error value
is suspect (parser detected the error column may not actually contain errors).
The values are NOT changed by the parser — these files just need manual review.

Usage:
    python scripts/parser_audit/list_suspect_files.py
"""
import sys
import logging
import warnings
from pathlib import Path
from collections import defaultdict
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)


class SanityCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.current_file = None
        self.flagged = []

    def emit(self, record):
        msg = record.getMessage()
        if "column D may not be errors" in msg:
            self.flagged.append((self.current_file, msg))


def main():
    handler = SanityCapture()
    logging.getLogger("laser_trim_analyzer.core.final_test_parser").addHandler(handler)
    logging.getLogger("laser_trim_analyzer.core.final_test_parser").setLevel(logging.WARNING)

    from laser_trim_analyzer.core.parser import detect_file_type
    from laser_trim_analyzer.core.final_test_parser import FinalTestParser

    work_root = ROOT / "Work Files"
    files = sorted([*work_root.rglob("*.xls"), *work_root.rglob("*.xlsx")])
    ft_files = [f for f in files if detect_file_type(f) == "final_test"]
    print(f"Scanning {len(ft_files)} FT files...\n")

    p = FinalTestParser()
    for f in ft_files:
        handler.current_file = str(f.relative_to(work_root))
        try:
            p.parse_file(f)
        except Exception:
            pass

    print(f"Files flagged with suspect column-layout: {len(handler.flagged)}\n")

    by_model = defaultdict(list)
    for rel, msg in handler.flagged:
        parts = rel.split("/")
        model = parts[2] if len(parts) >= 3 else "?"
        m = re.search(r"median \|file_error\|=([\d.]+).*spec≈([\d.None]+)", msg)
        med = m.group(1) if m else "?"
        spec = m.group(2) if m else "?"
        by_model[model].append((rel, med, spec))

    print(f"{'model':<20} {'count':>6}  example file (median / spec)")
    print("-" * 100)
    for model in sorted(by_model):
        items = by_model[model]
        rel, med, spec = items[0]
        print(f"{model:<20} {len(items):>6}  {Path(rel).name[:50]:<50} (med={med} spec={spec})")
        for rel, med, spec in items[1:]:
            print(f"{'':<20} {'':>6}  {Path(rel).name[:50]:<50} (med={med} spec={spec})")
    print()
    print(f"Total flagged: {len(handler.flagged)} files across {len(by_model)} models")


if __name__ == "__main__":
    main()
