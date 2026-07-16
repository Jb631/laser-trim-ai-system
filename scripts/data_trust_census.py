"""Data-trust census: quantify every known file-quirk class, per model.

James, 2026-07-16: "the real root is that the files themselves are
untrustworthy... every model has these little quirks." This is the
measurement pass that precedes the trust layer: grade EVERY record against
the quirk classes we know about, so the fix targets the classes that
actually drive distrust — instead of more chart annotations.

Usage:  python scripts/data_trust_census.py [path/to/analysis.db]
Read-only. Prints an overall census + the worst models per class.
"""
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

DB = sys.argv[1] if len(sys.argv) > 1 else "data/analysis.db"


def main() -> int:
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    q = lambda s, *a: c.execute(s, a).fetchall()  # noqa: E731

    print(f"=== DATA TRUST CENSUS — {DB} — {datetime.now():%Y-%m-%d %H:%M} ===\n")
    horizon = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    total_units = q("SELECT COUNT(*) FROM analysis_results")[0][0]
    total_tracks = q("SELECT COUNT(*) FROM track_results")[0][0]
    total_ft = q("SELECT COUNT(*) FROM final_test_results")[0][0]
    print(f"population: {total_units:,} trim units / {total_tracks:,} tracks / {total_ft:,} FT records\n")

    classes = []  # (name, count, denominator, per-model Counter)

    # 1. No sweep stored (unit can never be point-graded or charted)
    rows = q("""SELECT a.model, COUNT(*) FROM analysis_results a
                WHERE NOT EXISTS (SELECT 1 FROM track_results t
                                  WHERE t.analysis_id=a.id AND t.error_data IS NOT NULL)
                GROUP BY a.model""")
    classes.append(("trim: no sweep arrays stored", Counter(dict(rows)), total_units))

    # 2. Sweep present but LIMITS absent -> zero-tolerance ungradeable per point
    rows = q("""SELECT a.model, COUNT(*) FROM track_results t
                JOIN analysis_results a ON a.id=t.analysis_id
                WHERE t.error_data IS NOT NULL
                  AND (t.upper_limits IS NULL OR t.lower_limits IS NULL)
                GROUP BY a.model""")
    classes.append(("trim: sweep without spec limits", Counter(dict(rows)), total_tracks))

    # 3. Spec missing entirely (linearity_spec NULL on gradeable rows)
    rows = q("""SELECT a.model, COUNT(*) FROM track_results t
                JOIN analysis_results a ON a.id=t.analysis_id
                WHERE t.linearity_spec IS NULL
                  AND a.overall_status IN ('PASS','WARNING','FAIL')
                GROUP BY a.model""")
    classes.append(("trim: no linearity spec on gradeable row", Counter(dict(rows)), total_tracks))

    # 4. Date anomalies
    rows = q(f"""SELECT model, COUNT(*) FROM analysis_results
                 WHERE file_date IS NULL OR file_date < '2000-01-01'
                    OR file_date > '{horizon}' GROUP BY model""")
    classes.append(("trim: missing/epoch/future dates", Counter(dict(rows)), total_units))
    rows = q(f"""SELECT model, COUNT(*) FROM final_test_results
                 WHERE COALESCE(test_date, file_date) IS NULL
                    OR COALESCE(test_date, file_date) < '2000-01-01'
                    OR COALESCE(test_date, file_date) > '{horizon}' GROUP BY model""")
    classes.append(("FT: missing/epoch/future dates", Counter(dict(rows)), total_ft))

    # 5. Scale-anomalous resistance (the 2.1e8 class)
    rows = q("""SELECT a.model, COUNT(*) FROM track_results t
                JOIN analysis_results a ON a.id=t.analysis_id
                WHERE t.untrimmed_resistance > 1e7 GROUP BY a.model""")
    classes.append(("trim: absurd resistance (>10M ohm)", Counter(dict(rows)), total_tracks))

    # 6. FT records without sweep tracks (verdict-only files)
    rows = q("""SELECT f.model, COUNT(*) FROM final_test_results f
                WHERE NOT EXISTS (SELECT 1 FROM final_test_tracks t
                                  WHERE t.final_test_id=f.id AND t.position_data IS NOT NULL)
                GROUP BY f.model""")
    classes.append(("FT: no sweep arrays stored", Counter(dict(rows)), total_ft))

    # 7. Stored verdict vs arrays RECOMPUTE disagreement (trim, sampled).
    #    The core trust question: does the recorded PASS/FAIL match what the
    #    stored sweep + limits + offset actually say?
    sample = q("""SELECT a.model, t.linearity_pass, t.optimal_offset,
                         t.error_data, t.upper_limits, t.lower_limits
                  FROM track_results t JOIN analysis_results a ON a.id=t.analysis_id
                  WHERE t.error_data IS NOT NULL AND t.upper_limits IS NOT NULL
                    AND t.lower_limits IS NOT NULL AND t.linearity_pass IS NOT NULL
                  ORDER BY a.id DESC LIMIT 4000""")
    dis = Counter()
    checked = 0
    for m, lp, off, e, u, l in sample:
        try:
            err, up, lo = json.loads(e), json.loads(u), json.loads(l)
            o = off or 0.0
            n = min(len(err), len(up), len(lo))
            fails = any(
                err[i] is not None and up[i] is not None and lo[i] is not None
                and not (lo[i] <= err[i] + o <= up[i]) for i in range(n))
            checked += 1
            if bool(lp) == fails:          # stored pass but arrays fail, or vice versa
                dis[m] += 1
        except Exception:
            continue
    classes.append((f"trim: stored verdict vs sweep recompute disagree (sample {checked})",
                    dis, checked))

    # 8. Same check on FT tracks (the '6.4% FT verdict gap' thread)
    sample = q("""SELECT f.model, t.linearity_pass, t.optimal_offset,
                         t.error_data, t.upper_limits, t.lower_limits
                  FROM final_test_tracks t JOIN final_test_results f ON f.id=t.final_test_id
                  WHERE t.error_data IS NOT NULL AND t.upper_limits IS NOT NULL
                    AND t.lower_limits IS NOT NULL AND t.linearity_pass IS NOT NULL
                  ORDER BY t.id DESC LIMIT 4000""")
    dis_ft = Counter()
    checked_ft = 0
    for m, lp, off, e, u, l in sample:
        try:
            err, up, lo = json.loads(e), json.loads(u), json.loads(l)
            o = off or 0.0
            n = min(len(err), len(up), len(lo))
            fails = any(
                err[i] is not None and up[i] is not None and lo[i] is not None
                and not (lo[i] <= err[i] + o <= up[i]) for i in range(n))
            checked_ft += 1
            if bool(lp) == fails:
                dis_ft[m] += 1
        except Exception:
            continue
    classes.append((f"FT: stored verdict vs sweep recompute disagree (sample {checked_ft})",
                    dis_ft, checked_ft))

    # 9. Trim vs FT spec disagreement — POSITION-MATCHED envelope comparison
    #    on linked pairs. James (2026-07-16): bowtie specs are graded point by
    #    point and the two stations sample different position tables, so
    #    comparing scalar spec summaries flags phantom disagreements (the
    #    first cut said 81 models; the honest method says 18). Only compare
    #    limits at NEAR-COINCIDENT positions (tolerance = half the finer
    #    station's spacing) — never interpolate across a bowtie knee.
    pair_rows = q("""
        SELECT f.model, t.position_data, t.upper_limits, t.lower_limits,
               ft.position_data, ft.upper_limits, ft.lower_limits
        FROM final_test_results f
        JOIN analysis_results a ON a.id = f.linked_trim_id
        JOIN track_results t ON t.analysis_id = a.id
        JOIN final_test_tracks ft ON ft.final_test_id = f.id
        WHERE t.position_data IS NOT NULL AND ft.position_data IS NOT NULL
          AND t.upper_limits IS NOT NULL AND ft.upper_limits IS NOT NULL
          AND t.lower_limits IS NOT NULL AND ft.lower_limits IS NOT NULL
        ORDER BY f.id DESC""")
    per_model_pairs: dict = defaultdict(list)
    for r in pair_rows:
        if len(per_model_pairs[r[0]]) < 3:
            per_model_pairs[r[0]].append(r[1:])

    def _pts(p, u, l):
        out = [(x, a, b) for x, a, b in zip(json.loads(p), json.loads(u), json.loads(l))
               if x is not None and a is not None and b is not None]
        out.sort()
        return out

    def _spacing(P):
        gaps = sorted(P[i + 1][0] - P[i][0] for i in range(len(P) - 1)
                      if P[i + 1][0] > P[i][0])
        return gaps[len(gaps) // 2] if gaps else 1.0

    spec_dis = Counter()
    spec_examples = []
    for model, pairs in per_model_pairs.items():
        matched = differing = 0
        for tp, tu, tl, fp, fu, fl in pairs:
            try:
                T, F = _pts(tp, tu, tl), _pts(fp, fu, fl)
            except Exception:
                continue
            if len(T) < 3 or len(F) < 3:
                continue
            tol = min(_spacing(T), _spacing(F)) / 2.0
            j = 0
            for x, u, l in T:
                while j < len(F) - 1 and F[j][0] < x - tol:
                    j += 1
                if abs(F[j][0] - x) <= tol:
                    width = max(u - l, 1e-9)
                    matched += 1
                    if max(abs(u - F[j][1]), abs(l - F[j][2])) > 0.2 * width:
                        differing += 1
        if matched >= 10 and 100.0 * differing / matched > 10:
            spec_dis[model] = differing
            spec_examples.append((model, differing, matched))
    spec_examples.sort(key=lambda r: -r[1] / r[2])
    classes.append((f"model spec: trim vs FT bands TRULY differ at matched positions "
                    f"({len(spec_dis)} of {len(per_model_pairs)} models with linked pairs)",
                    spec_dis, None))

    # 10. Serial hygiene
    rows = q("""SELECT model, COUNT(*) FROM analysis_results
                WHERE serial IS NULL OR TRIM(serial)='' OR LOWER(serial) LIKE '%golden%'
                   OR LOWER(serial) LIKE '%test%' GROUP BY model""")
    classes.append(("trim: blank/golden/test serials", Counter(dict(rows)), total_units))

    # ---- report ----
    for name, per_model, denom in classes:
        n = sum(per_model.values())
        pct = f" ({100.0 * n / denom:.1f}%)" if denom else ""
        print(f"{name}: {n:,}{pct}")
        for m, k in per_model.most_common(5):
            print(f"    {m}: {k:,}")
    if spec_examples:
        print("\ntrue spec disagreements (model: differing/matched positions):")
        for m, d, n in spec_examples[:15]:
            print(f"    {m}: {d}/{n} ({100.0 * d / n:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
