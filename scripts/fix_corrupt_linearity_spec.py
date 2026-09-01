#!/usr/bin/env python3
"""Remediate stored tracks whose linearity_spec came from a corrupt limit column.

Background (measured 2026-08-30 against data/analysis.db)
---------------------------------------------------------
Model 8888 stored linearity_spec = 63.03 on 13 tracks. This is NOT a parser
bug. The source workbooks genuinely contain, in the Upper Lin Lim column,
0.03 repeated 23 times followed by 1.03, 2.03, 3.03 ... 148.03 — Excel's
fill-handle "Fill Series" applied to a decimal when the template was built.
63.03 is the honest median of that column. A 63 V limit passes every unit,
so all 13 tracks were stored linearity_pass = True regardless of the data.

Two remediation classes:

  RECOVERABLE (fill-series artifact) — the true spec survives as the modal
  half-width of the column, and the leading run of correct rows confirms it.
  These are corrected in place: the limit arrays are rewritten to the modal
  band and linearity is RE-GRADED with the real analyzer against the stored
  position/error arrays.

  UNRECOVERABLE (sign error, position column in the limits, ragged limits) —
  there is no trustworthy spec to recover, so nothing is invented. These are
  marked with linearity_spec_warning and linearity_pass is set to NULL:
  indeterminate, not a manufactured PASS.

Dry-run by default. Pass --apply to write.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laser_trim_analyzer.core.analyzer import Analyzer          # noqa: E402
from laser_trim_analyzer.core.parser import ExcelParser         # noqa: E402

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "analysis.db"


def recover_fill_series_band(upper, lower):
    """Recover the intended +/- band from a fill-series-corrupted column.

    Deliberately narrow: this recovers the ONE defect whose true value is
    actually knowable. Excel's fill handle, dragged down a decimal cell,
    increments the INTEGER part and leaves the fraction alone — 0.03 becomes
    1.03, 2.03, 3.03. So the signature is exact: every half-width in the
    column is the true spec plus a non-negative whole number. 8888 is
    0.03 + n and 8094-2 is 0.025 + n (its ramp even restarts once, going
    ...0.025, 1.025, 0.025, 1.025, 2.025..., so monotonicity is the wrong
    test; the integer-offset rule handles it).

    Requiring that exact arithmetic is what stops a merely-ragged column from
    being "recovered" to an invented band: model 7965's limits jump between
    -0.05, 0.0, 0.02 and 0.05, giving half-widths 0.075/0.05/0.04/0.025 whose
    gaps are not whole numbers. Its real spec is 0.05, so the naive
    "smallest value wins" rule would have re-graded it against 0.025 — a
    tighter-than-real spec, manufacturing failures. It stays unrecoverable.

    Returns (spec, n_supporting_rows) or (None, 0).
    """
    half = [
        round((u - l) / 2, 9)
        for u, l in zip(upper, lower)
        if u is not None and l is not None and (u - l) > 0
    ]
    if len(half) < 10:
        return None, 0
    base = min(half)
    if base <= 0:
        return None, 0
    support = sum(1 for h in half if h == base)
    if support < 5:
        return None, 0
    if support == len(half):
        return None, 0  # clean column; nothing to recover
    for h in half:
        offset = h - base
        if abs(offset - round(offset)) > 1e-6 or round(offset) < 0:
            return None, 0
    return base, support


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--apply", action="store_true",
                    help="write the corrections (default: dry run)")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"ERROR: no such database: {args.db}")
        return 2

    parser = ExcelParser()
    analyzer = Analyzer()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # The app adds this column on startup (DatabaseManager migration), but this
    # script has to run standalone against a DB the app has not opened yet.
    # Idempotent: SQLite has no ADD COLUMN IF NOT EXISTS, so probe first.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(track_results)")}
    if "linearity_spec_warning" not in cols:
        if not args.apply:
            print("NOTE: track_results has no linearity_spec_warning column yet; "
                  "--apply would add it.\n")
        else:
            conn.execute("ALTER TABLE track_results ADD COLUMN linearity_spec_warning TEXT")
            conn.commit()
            print("added missing column: track_results.linearity_spec_warning\n")

    rows = conn.execute(
        "SELECT t.id, a.model, a.filename, t.linearity_spec, t.linearity_pass, "
        "       t.upper_limits, t.lower_limits, t.position_data, t.error_data "
        "FROM track_results t JOIN analysis_results a ON a.id = t.analysis_id "
        "WHERE t.upper_limits IS NOT NULL AND t.lower_limits IS NOT NULL"
    ).fetchall()

    scanned = 0
    recovered, unrecoverable = [], []
    for r in rows:
        try:
            U = json.loads(r["upper_limits"]) or []
            L = json.loads(r["lower_limits"]) or []
        except (TypeError, ValueError):
            continue
        if not U or not L:
            continue
        scanned += 1
        spec = parser._calculate_linearity_spec(U, L)
        reason = parser._validate_limit_columns(U, L, spec)
        if not reason:
            continue

        true_spec, support = recover_fill_series_band(U, L)
        if true_spec is None:
            unrecoverable.append((r, reason))
            continue

        # Re-grade with the real analyzer against a corrected band.
        try:
            P = json.loads(r["position_data"]) or []
            E = json.loads(r["error_data"]) or []
        except (TypeError, ValueError):
            P, E = [], []
        if not P or not E:
            unrecoverable.append((r, reason + " (no stored arrays to re-grade)"))
            continue

        n = min(len(P), len(E))
        up = [true_spec] * n
        lo = [-true_spec] * n
        (_off, _k, lin_err, lin_pass, fail_pts, raw_err,
         raw_fail) = analyzer._calculate_linearity(P[:n], E[:n], up, lo, true_spec)
        recovered.append((r, true_spec, support, bool(lin_pass), int(fail_pts),
                          float(lin_err), reason))

    print(f"scanned {scanned} tracks with limit data\n")

    print(f"=== RECOVERABLE — correct in place + re-grade ({len(recovered)}) ===")
    if recovered:
        print(f"{'model':9s} {'file':34s} {'stored':>8s} {'true':>7s} "
              f"{'was':>5s} {'now':>5s} {'fails':>5s}")
    for r, spec, support, lin_pass, fail_pts, lin_err, _reason in recovered:
        was = "PASS" if r["linearity_pass"] else "FAIL"
        now = "PASS" if lin_pass else "FAIL"
        flag = "   <-- FLIPS" if was != now else ""
        print(f"{r['model']:9s} {r['filename'][:34]:34s} {r['linearity_spec']:8.4g} "
              f"{spec:7.4g} {was:>5s} {now:>5s} {fail_pts:5d}{flag}")
    n_flip = sum(1 for r, _s, _c, lp, *_ in recovered if bool(r["linearity_pass"]) != lp)
    print(f"\n  {n_flip} verdict(s) change once graded against the real spec")

    print(f"\n=== UNRECOVERABLE — mark indeterminate ({len(unrecoverable)}) ===")
    for r, reason in unrecoverable:
        print(f"{r['model']:9s} {r['filename'][:40]:40s} spec={r['linearity_spec']:<8.4g} {reason}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to commit these changes.")
        return 0

    cur = conn.cursor()
    # linearity_spec_warning means exactly one thing — "this row's spec is not
    # trustworthy, so linearity was NOT graded". A corrected row does not
    # qualify: its limit arrays have been rewritten to the clean band and it
    # has been genuinely graded, so the column stays NULL and the audit trail
    # goes to a log file beside the database instead of overloading the flag.
    audit = [
        "# linearity_spec remediation — fill-series artifact",
        "# corrected in place: limits rewritten to the recovered band, linearity re-graded",
        "id\tmodel\tfile\tstored_spec\ttrue_spec\tsupport_rows\twas\tnow\tfail_points",
    ]
    for r, spec, support, lin_pass, fail_pts, lin_err, reason in recovered:
        cur.execute(
            "UPDATE track_results SET linearity_spec=?, upper_limits=?, lower_limits=?, "
            "linearity_pass=?, linearity_fail_points=?, final_linearity_error_shifted=?, "
            "linearity_spec_warning=NULL WHERE id=?",
            (spec,
             json.dumps([spec] * len(json.loads(r["upper_limits"]))),
             json.dumps([-spec] * len(json.loads(r["lower_limits"]))),
             1 if lin_pass else 0, fail_pts, lin_err,
             r["id"]))
        audit.append(
            f"{r['id']}\t{r['model']}\t{r['filename']}\t{r['linearity_spec']:.6g}\t"
            f"{spec:.6g}\t{support}\t{'PASS' if r['linearity_pass'] else 'FAIL'}\t"
            f"{'PASS' if lin_pass else 'FAIL'}\t{fail_pts}")
    audit.append("# marked indeterminate: no trustworthy spec to recover")
    for r, reason in unrecoverable:
        cur.execute(
            "UPDATE track_results SET linearity_pass=NULL, linearity_spec_warning=? "
            "WHERE id=?", (reason, r["id"]))
        audit.append(f"{r['id']}\t{r['model']}\t{r['filename']}\t"
                     f"{r['linearity_spec']:.6g}\t-\t-\t"
                     f"{'PASS' if r['linearity_pass'] else 'FAIL'}\tINDETERMINATE\t-")
    conn.commit()
    log = args.db.parent / "linearity_spec_remediation.log"
    log.write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(f"\nAPPLIED: {len(recovered)} corrected+re-graded, "
          f"{len(unrecoverable)} marked indeterminate.")
    print(f"Audit trail written to {log}")
    print("\nNOTE: the SOURCE workbooks are still corrupt. Re-processing those "
          "files will re-flag them as indeterminate (the guard doing its job), "
          "undoing this repair. Fix the workbooks to make it permanent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
