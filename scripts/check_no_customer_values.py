"""Refuse to let a customer's numbers reach a commit.

    python scripts/check_no_customer_values.py [<git range>]        # default origin/main..HEAD

The backlog export under `Work Files/` carries customer names, PO numbers and unit prices. None of
them may appear in the repository -- not in code, not in a test, not in a fixture, not in a commit
message. This scans EVERY VERSION of every text file in a git range, plus every commit message, and
prints only WHERE a problem is, never the value itself. Exit code = number of problems.

Why it exists (2026-09-20): Claude wrote an example test row using a real (model, price) pair from
the backlog, an implementer transcribed it faithfully, and it was committed. It was caught before
the push by this check, and the commit was rewritten. Example data is INVENTED -- 11.0, 22.5,
1234.0 -- never a number read out of the customer file, not even as an illustration.

With no backlog export present (a fresh checkout, the work machine) it says so and passes: it can
only check against a file it can read.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Every text format this repository commits. .html/.css/.js/.csv/.xml/.svg were missing until
# 2026-09-23, when a design mockup (.html) went through a push unscanned -- it was clean, checked
# by hand, but only by luck of what it happened to contain. A guard that skips a file type is a
# guard with a door in it.
TEXT = (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ps1", ".bat", ".command",
        ".cfg", ".ini", ".html", ".htm", ".css", ".js", ".csv", ".tsv", ".xml", ".svg", ".sh")


# A number is only a candidate price when it is not part of a path, a filename or a
# date/time. Serial numbers and timestamps in this project's data filenames
# ("8035-2_107_TEST DATA_9-5-2025_9-00 AM.xls") otherwise produce matches that are
# arithmetic coincidences, and a guard that cries wolf is one nobody reads.
_NUM = re.compile(r"(?<![\w.])(\d{2,6}(?:\.\d+)?)(?![\w.])")
_DATEISH = re.compile(r"\d+[-/]\d+[-/]\d+")


def _price_like(line: str):
    """Every number on `line` that could be a unit price, as strings."""
    out = []
    for m in _NUM.finditer(line):
        lo = max((line.rfind(c, 0, m.start()) for c in ' \t"\',[('), default=-1) + 1
        hi = min((h for h in (line.find(c, m.end()) for c in ' \t"\',])') if h >= 0), default=len(line))
        token = line[lo:hi]
        if "/" in token or "\\" in token or _DATEISH.search(token):
            continue                        # a path or a date, not a price
        if re.search(r"\.[A-Za-z]{2,4}$", token.rstrip('":,')):
            continue                        # a filename
        out.append(m.group(1))
    return out


def _backlog():
    """(prices by model, customer names, PO numbers) from the newest backlog export, or None."""
    exports = sorted(Path(REPO / "Work Files").glob("Backlog*.xls*")) if (REPO / "Work Files").is_dir() else []
    if not exports:
        return None
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    df = pd.read_excel(exports[-1], header=None)
    head = next((i for i in range(min(30, len(df)))
                 if any(str(v).strip() == "Item ID" for v in df.iloc[i].tolist())), None)
    if head is None:
        return None
    df.columns = [str(c).strip() for c in df.iloc[head].tolist()]
    df = df.iloc[head + 1:]
    prices: dict = {}
    for _, row in df.iterrows():
        p = pd.to_numeric(row.get("Unit Price"), errors="coerce")
        # A zero or negative unit price (a no-charge line, a credit) is not customer
        # information -- it reveals nothing and it matches everywhere. Left in, 0.00
        # matched the "00" of a "9-00 AM" timestamp inside a data filename and reported
        # a leak in a commit from April that had been on origin/main for months.
        if pd.notna(p) and float(p) > 0:
            prices.setdefault(str(row["Item ID"]).strip(), set()).add(round(float(p), 2))
    names = {str(v).strip() for v in df.get("Customer Name", pd.Series(dtype=object)).dropna().unique()
             if len(str(v).strip()) > 3}
    pos = {str(v).strip() for v in df.get("PO Number", pd.Series(dtype=object)).dropna().unique()
           if len(str(v).strip()) > 4}
    return prices, names, pos, exports[-1].name


def findings_in(text: str, prices, names, pos):
    """Every problem in `text`, as (kind, line number, note). Never the value itself.

    Module level so the guard itself can be tested -- it is the only thing standing
    between the backlog export and a public commit, and it had two defects of its own
    (2026-09-20): a zero unit price matched everywhere, and numbers inside data
    filenames were read as prices.
    """
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        nums = {round(float(x), 2) for x in _price_like(line)}
        for model, real in sorted(prices.items()):
            if nums & real and re.search(r"(?<![\w-])" + re.escape(model) + r"(?![\w-])", line):
                out.append(("PRICE", n, f"  (model {model} beside one of its real unit prices)"))
        if any(c in line for c in names):
            out.append(("CUSTOMER", n, ""))
        if any(p in line for p in pos):
            out.append(("PO NUMBER", n, ""))
    return out


def main() -> int:
    data = _backlog()
    if data is None:
        print("no backlog export under 'Work Files/' -- nothing to check against (that is not a failure)")
        return 0
    prices, names, pos, source = data
    rng = sys.argv[1] if len(sys.argv) > 1 else "origin/main..HEAD"
    commits = subprocess.run(["git", "rev-list", rng], cwd=REPO, capture_output=True, text=True).stdout.split()
    problems = 0
    seen: set = set()

    def scan(label: str, text: str) -> None:
        nonlocal problems
        for where in findings_in(text, prices, names, pos):
            problems += 1
            print(f"  {where[0]:<9} {label}:{where[1]}{where[2]}")

    for c in commits:
        scan(f"{c[:7]} <commit message>",
             subprocess.run(["git", "log", "-1", "--format=%B", c], cwd=REPO,
                            capture_output=True, text=True).stdout)
        files = subprocess.run(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", c],
                               cwd=REPO, capture_output=True, text=True).stdout.split("\n")
        for f in files:
            if not f or not f.endswith(TEXT) or (c, f) in seen:
                continue
            blob = subprocess.run(["git", "show", f"{c}:{f}"], cwd=REPO, capture_output=True,
                                  text=True, errors="replace")
            if blob.returncode == 0:
                seen.add((c, f))
                scan(f"{c[:7]}:{f}", blob.stdout)
    print(f"{len(commits)} commits, {len(seen)} file versions scanned in {rng} against {source}; "
          f"problems: {problems}")
    return min(problems, 250)


if __name__ == "__main__":
    raise SystemExit(main())
