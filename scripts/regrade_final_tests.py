"""Re-grade stored final tests against the window the sheet actually grades.

Why this exists (2026-09-13)
----------------------------
A final-test sheet grades a WINDOW of its sweep. Column I carries a per-point
flag written only on the rows the station judged, and the lead-in/run-out named
by "# of elements to ignore at start / at end" are left ungraded. Until today
the app had never read that column and graded every row — including model
8232-1's six-point lead-in, where the pot reads 0 V against a theory of
-0.045 V and the phantom 0.047 "error" is five times a +/-0.010 band. That
failed 98.5% of the model's files at a station that passed them.

The parser and the grader are fixed, so every file ingested from now on is
graded correctly. Rows already in the database are not: this script re-parses
each one and re-grades it through the SAME code path the processor uses
(`core.ft_regrade.grade_ft_track`), then writes the new verdict back.

The verdict stays the APP's. The sheet's own PASSED/FAILED is recorded beside
it as a reference and never replaces it — the point of the fix is to grade the
right cells, not to copy the spreadsheet.

Usage
-----
    python scripts/regrade_final_tests.py path/to/analysis.db            # dry run
    python scripts/regrade_final_tests.py path/to/analysis.db --apply    # writes
    python scripts/regrade_final_tests.py path/to/analysis.db --all      # every row
    python scripts/regrade_final_tests.py path/to/analysis.db --limit 500

The database path is REQUIRED and nothing is written without `--apply`: the
dry run parses and grades everything and prints exactly what would change, per
model, so the scale of the shift is visible before it happens.

Unlike the QA harnesses this script does NOT refuse `data/analysis.db`. It is a
repair tool and the production database is its target — that is the whole
point. TAKE A BACKUP FIRST; the verdicts it rewrites cannot be recovered from
the database itself.

Everything downstream of a final-test verdict moves when this runs: fail rates,
escapes and overkills, the FOCUS list, and any ML labels trained on final test
as ground truth.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _fmt_verdict(value) -> str:
    return "PASS" if value else ("FAIL" if value is False else "not graded")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    apply_changes = "--apply" in flags
    only_legacy = "--all" not in flags
    limit = None
    for flag in flags:
        if flag.startswith("--limit="):
            limit = int(flag.split("=", 1)[1])
    if "--limit" in flags:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])
            args = [a for a in args if a != str(limit)]

    if not args:
        print(__doc__)
        return 2

    db_path = Path(args[0])
    if not db_path.exists():
        # Checked BEFORE DatabaseManager: constructing it CREATES an empty
        # database, and "re-grading" a fresh empty file reports success for
        # nothing at all.
        print(f"FATAL | no database at {db_path}")
        return 1

    from laser_trim_analyzer.core.ft_regrade import regrade_final_tests
    from laser_trim_analyzer.database.manager import DatabaseManager

    db = DatabaseManager(db_path)
    legacy = db.count_legacy_ft_verdicts()

    print(f"database          : {db_path}")
    print(f"selection         : {'rows graded before the window fix' if only_legacy else 'EVERY final-test row'}")
    print(f"legacy rows       : {legacy:,}")
    print(f"mode              : {'APPLY — the database will be written' if apply_changes else 'DRY RUN — nothing is written'}")
    if apply_changes:
        print()
        print("  *** BACK UP THE DATABASE FIRST. ***")
        print("  This rewrites final-test verdicts in place. Every final-test")
        print("  number in the app changes afterwards: fail rates, escapes and")
        print("  overkills, the FOCUS list, and ML labels trained on FT.")
    print()

    from laser_trim_analyzer.core.ft_regrade import format_regrade_line

    def progress(done: int, total: int, name: str, rate, eta: str) -> None:
        # The driver already rate-limits to one call every ~2 s; the same line
        # the Settings section shows, so a screenshot and a terminal cannot
        # disagree about how far along the same run is.
        line = format_regrade_line(done, total, rate, eta)
        print(f"\r  {line}  {name[:40]:<40}", end="", flush=True)

    report = regrade_final_tests(
        db, progress=progress, only_legacy=only_legacy,
        apply=apply_changes, limit=limit)
    print()
    print()

    if not report.examined:
        print("Nothing to do — no final-test rows matched the selection.")
        return 0

    models = report.by_model()
    changed_models = {m: c for m, c in models.items() if c["changed"] or c["missing"]}
    if changed_models:
        print(f"{'model':<16}{'rows':>8}{'changed':>9}{'PASS>FAIL':>11}"
              f"{'FAIL>PASS':>11}{'>not graded':>13}{'missing':>9}")
        print("-" * 77)
        for model in sorted(changed_models, key=lambda m: -changed_models[m]["changed"]):
            c = changed_models[model]
            print(f"{model[:15]:<16}{c['examined']:>8,}{c['changed']:>9,}"
                  f"{c['pass_to_fail']:>11,}{c['fail_to_pass']:>11,}"
                  f"{c['to_null']:>13,}{c['missing']:>9,}")
        print("-" * 77)
    else:
        print("No verdict changes in any model.")
    print()
    print(report.summary())

    if report.missing:
        print()
        print(f"{report.missing:,} source file(s) could not be opened — those rows "
              f"were left exactly as they were.")
        print("Run this on the work network, where the plant share is reachable.")
    if report.errors:
        print()
        print(f"{report.errors:,} row(s) failed to re-parse and were left alone:")
        for outcome in report.outcomes:
            if outcome.result == "parse_error":
                print(f"  {outcome.filename}: {outcome.detail}")
                break
    if not apply_changes and report.changed:
        print()
        print(f"Nothing was written. Re-run with --apply to make these "
              f"{report.changed:,} change(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
