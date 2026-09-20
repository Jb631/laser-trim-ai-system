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
TEXT = (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ps1", ".bat", ".command", ".cfg", ".ini")


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
        if pd.notna(p):
            prices.setdefault(str(row["Item ID"]).strip(), set()).add(round(float(p), 2))
    names = {str(v).strip() for v in df.get("Customer Name", pd.Series(dtype=object)).dropna().unique()
             if len(str(v).strip()) > 3}
    pos = {str(v).strip() for v in df.get("PO Number", pd.Series(dtype=object)).dropna().unique()
           if len(str(v).strip()) > 4}
    return prices, names, pos, exports[-1].name


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
        for n, line in enumerate(text.splitlines(), 1):
            nums = {round(float(x), 2) for x in re.findall(r"(?<![\w.])(\d{2,6}(?:\.\d+)?)(?![\w.])", line)}
            for model, real in prices.items():
                if nums & real and re.search(r"(?<![\w-])" + re.escape(model) + r"(?![\w-])", line):
                    problems += 1
                    print(f"  PRICE     {label}:{n}  (model {model} beside one of its real unit prices)")
            if any(c in line for c in names):
                problems += 1
                print(f"  CUSTOMER  {label}:{n}")
            if any(p in line for p in pos):
                problems += 1
                print(f"  PO NUMBER {label}:{n}")

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
