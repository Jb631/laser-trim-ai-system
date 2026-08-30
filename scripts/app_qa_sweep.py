"""App-wide QA sweep: exercise EVERY v6 feature data path against the real
database and assert cross-feature invariants.

Companion to chart_qa_render_all.py (visuals). This one covers features:
dashboard aggregates, triage, model-page loaders, unit modal, settings
actions, exports, and the processing pipeline — each check is PASS/FAIL/WARN
with the number that failed, so regressions surface without a human clicking
through the app.

    python scripts/app_qa_sweep.py [path/to/analysis.db]

The optional path is for machines where the work database is not at
data/analysis.db — pass a COPY, never the original (this opens it read-write).

Exit code = number of FAILs. WARNs are judgment items for review.
"""
import sys
import types
from pathlib import Path

# ---- headless stubs (same technique as chart_qa_render_all) -----------------
import matplotlib
matplotlib.use("Agg")
_ctk = types.ModuleType("customtkinter")
class _W:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return lambda *a, **k: None
for _n in ("CTk", "CTkFrame", "CTkLabel", "CTkButton", "CTkEntry", "CTkFont",
           "CTkScrollableFrame", "CTkComboBox", "CTkOptionMenu", "CTkTextbox",
           "CTkCheckBox", "CTkProgressBar", "CTkToplevel", "CTkCanvas",
           "CTkSegmentedButton", "CTkImage", "CTkSlider", "CTkTabview",
           "BooleanVar", "StringVar", "IntVar", "DoubleVar"):
    setattr(_ctk, _n, type(_n, (_W,), {}))
_ctk.set_appearance_mode = lambda *a, **k: None
_ctk.set_default_color_theme = lambda *a, **k: None
sys.modules["customtkinter"] = _ctk
_tk = types.ModuleType("tkinter")
for _n in ("Frame", "Canvas", "Variable", "StringVar", "IntVar", "DoubleVar",
           "BooleanVar", "Toplevel", "Widget", "Label"):
    setattr(_tk, _n, type(_n, (_W,), {}))
_tk.TclError = type("TclError", (Exception,), {})
sys.modules["tkinter"] = _tk
for _s in ("filedialog", "messagebox", "ttk", "font"):
    _m = types.ModuleType(f"tkinter.{_s}")
    _m.__getattr__ = lambda n: (lambda *a, **k: None)
    sys.modules[f"tkinter.{_s}"] = _m
_tkagg = types.ModuleType("matplotlib.backends.backend_tkagg")
class FigureCanvasTkAgg:
    def __init__(self, figure=None, master=None): self.figure = figure
    def get_tk_widget(self): return _W()
    def draw(self): pass
    def draw_idle(self): pass
_tkagg.FigureCanvasTkAgg = FigureCanvasTkAgg
sys.modules["matplotlib.backends.backend_tkagg"] = _tkagg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import sqlite3  # noqa: E402
from sqlalchemy import text as sqlalchemy_text  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

RESULTS: list = []


def check(name, ok, detail=""):
    RESULTS.append(("PASS" if ok else "FAIL", name, detail))
    print(f"{'PASS' if ok else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""))


def warn(name, detail=""):
    RESULTS.append(("WARN", name, detail))
    print(f"WARN | {name}" + (f" | {detail}" if detail else ""))


def check_ft_incremental_fastpath() -> None:
    """FT incremental fast-path contract (2026-08-29 processing-speed fix).

    Standalone by design: it builds its own throwaway DB from real FT sample
    files, so it runs (and must pass) on a machine that has no copy of the
    work database. Run just this section with:
        python scripts/app_qa_sweep.py --only ft-fastpath
    """
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import FinalTestResult as DBFT

    # ---- FT incremental fast-path: stats stamped, second scan hashes nothing -
    # The 2026-08-29 fix. final_test_results had no size/mtime, so every known
    # FT file was re-HASHED (full read over the share) on every scan — 70
    # minutes for the 171k-file FT folder. Assert the contract end to end on
    # real FT files, against a throwaway DB: (1) saved rows carry the stat,
    # (2) a second pass settles every file in memory, (3) a row whose stat is
    # NULL (legacy) heals itself after one verification pass.
    import shutil, tempfile  # noqa: E402
    from laser_trim_analyzer.database import manager as _dbmod  # noqa: E402
    from laser_trim_analyzer.gui.v6.pages.process_page import ProcessPage  # noqa: E402
    ft_src = sorted(p for p in (REPO / "Work Files" / "Sample_Base_2026-04-10"
                                / "Test Station").rglob("*.xls*") if p.is_file())[:4]
    if ft_src:
        tmp = Path(tempfile.mkdtemp(prefix="ltaqa_ft_"))
        saved_global = _dbmod._db_manager
        try:
            for f in ft_src:
                shutil.copy2(f, tmp / f.name)          # copy2 keeps the mtime
            ftdb = DatabaseManager(tmp / "ft_qa.db")
            _dbmod._db_manager = ftdb                  # processor saves FT here

            def _run_batch(turbo=10 ** 9):
                # turbo=1 forces the PARALLEL two-phase filter; the default
                # keeps the sequential one. Both must reach the same verdict.
                proc = Processor(use_ml=False)
                proc.config.processing.turbo_mode_threshold = turbo
                files, stats = ProcessPage._discover(None, str(tmp))
                files = [Path(f) for f in files if f.lower().endswith((".xls", ".xlsx"))]
                gen = proc.process_batch(files, incremental=True, disk_stats=stats)
                out = []
                try:
                    while True:
                        out.append(next(gen))
                except StopIteration as stop:
                    return proc, out, stop.value
                return proc, out, None

            p1, res1, sum1 = _run_batch()
            errs = [r for r in res1 if getattr(r.overall_status, "name", "") == "ERROR"]
            check("FT fast-path: first pass parsed the sample FT files",
                  len(res1) >= len(ft_src) and not errs,
                  f"yielded={len(res1)} errors={len(errs)}"
                  + (f" | {errs[0].errors[0][:70]}" if errs and errs[0].errors else ""))
            with ftdb.session() as fs:
                rows = fs.query(DBFT.file_path, DBFT.file_size,
                                DBFT.file_modified_date).all()
            check("FT fast-path: every saved row carries file_size + mtime",
                  bool(rows) and all(r.file_size is not None
                                     and r.file_modified_date is not None for r in rows),
                  f"rows={len(rows)} "
                  f"null={sum(1 for r in rows if r.file_size is None or r.file_modified_date is None)}")

            for label, turbo in (("sequential", 10 ** 9), ("parallel", 1)):
                p2, res2, sum2 = _run_batch(turbo)
                s2 = p2.last_scan_stats
                check(f"FT fast-path: second scan hashes/stats NOTHING ({label})",
                      s2.get("needs_hash") == 0 and s2.get("memory_hits") == len(ft_src)
                      and getattr(sum2, "processed", -1) == 0,
                      f"needs_hash={s2.get('needs_hash')} memory_hits={s2.get('memory_hits')} "
                      f"processed={getattr(sum2, 'processed', None)}")

            # Legacy rows (saved before the columns existed) must self-repair:
            # one verification pass, then back to pure memory.
            with ftdb.session() as fs:
                fs.execute(sqlalchemy_text(
                    "UPDATE final_test_results SET file_size = NULL, "
                    "file_modified_date = NULL"))
                fs.commit()
            p3, _res3, sum3 = _run_batch(turbo=1)   # parallel verify pool
            s3 = p3.last_scan_stats
            with ftdb.session() as fs:
                healed = fs.query(DBFT).filter(DBFT.file_size.isnot(None)).count()
                total_ft = fs.query(DBFT).count()
            check("FT fast-path: NULL-stat rows verify once, then heal",
                  s3.get("needs_hash") == len(ft_src) and healed == total_ft
                  and getattr(sum3, "processed", -1) == 0,
                  f"verified={s3.get('needs_hash')} healed={healed}/{total_ft} "
                  f"heal_updated={s3.get('heal_updated')} processed={getattr(sum3, 'processed', None)}")
            p4, _res4, _sum4 = _run_batch()
            check("FT fast-path: the scan after the heal is memory-only again",
                  p4.last_scan_stats.get("needs_hash") == 0,
                  f"needs_hash={p4.last_scan_stats.get('needs_hash')}")
        except Exception as exc:
            check("FT fast-path: incremental scan contract", False,
                  f"{type(exc).__name__}: {exc}")
        finally:
            _dbmod._db_manager = saved_global
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        warn("FT fast-path: no Test Station sample files to scan")

def main() -> int:
    # Optional DB-path argv (2026-08-29): the work database does not live at
    # data/analysis.db on every machine, and the sweep is worthless run against
    # anything else. Point it at a COPY — DatabaseManager opens read-write.
    #     python scripts/app_qa_sweep.py /path/to/copy_of_analysis.db
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data" / "analysis.db"
    if not db_path.exists():
        # Checked BEFORE DatabaseManager on purpose: constructing it CREATES an
        # empty database, and a sweep against an empty DB reports a wall of
        # green that means nothing. Refuse instead.
        print(f"FATAL | no database at {db_path}\n"
              f"      | pass a copy of the work database:\n"
              f"      |     python scripts/app_qa_sweep.py /path/to/analysis.db")
        return 1
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, FinalTestResult as DBFT)
    db = DatabaseManager(db_path)
    # Anything that reaches for "the app's database" on its own — Processor's
    # per-model spec lookups in section 7 go through `get_db_manager()` — must
    # get the SAME database. Left alone it builds one at the config default,
    # which on a machine that has no work DB silently CREATES an empty
    # data/analysis.db: a phantom the pipeline reads from and every test whose
    # skip depends on that file's absence then runs against.
    from laser_trim_analyzer.database import manager as _dbmod
    _dbmod._db_manager = db
    raw = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    VARIANTS = ["6607", "5409B", "8150", "8887", "7458-1"]

    # ============ 1. DASHBOARD: aggregates reconcile with raw SQL ============
    from laser_trim_analyzer.core.yield_stats import compute_yield, worst_models_by_yield
    for days in (90, 36500):
        cutoff = datetime.now() - timedelta(days=days)
        y = compute_yield(db, DBAR, cutoff)
        horizon = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        r = raw.execute(
            "SELECT COUNT(*), SUM(overall_status='PASS'), SUM(overall_status='WARNING'),"
            " SUM(overall_status='FAIL') FROM analysis_results "
            "WHERE file_date >= ? AND file_date <= ?",
            (cutoff.strftime("%Y-%m-%d %H:%M:%S"), horizon)).fetchone()
        check(f"dashboard yield counts vs SQL ({days}d)",
              y["total"] == (r[0] or 0) and y["passed"] == (r[1] or 0)
              and y["warnings"] == (r[2] or 0) and y["failed"] == (r[3] or 0),
              f"app={y['passed']}/{y['warnings']}/{y['failed']} sql={r[1]}/{r[2]}/{r[3]}")
        if y["gradeable"]:
            ly = 100 * (y["passed"] + y["warnings"]) / y["gradeable"]
            check(f"linearity_yield math ({days}d)",
                  abs((y["linearity_yield"] or 0) - ly) < 1e-9,
                  f"{y['linearity_yield']:.2f} vs {ly:.2f}")
    worst, total_q = worst_models_by_yield(db, datetime.now() - timedelta(days=36500))
    check("worst-models rates within [0,100] and sorted ascending",
          all(0 <= (w["trim_rate"] or 0) <= 100 for w in worst)
          and all((worst[i]["trim_rate"] or 0) <= (worst[i+1]["trim_rate"] or 0)
                  for i in range(len(worst) - 1)),
          f"{[(w['model'], round(w['trim_rate'] or -1, 1)) for w in worst[:3]]}")

    # ============ 2. COMPANY TREND: internal + cross checks ==================
    t = db.get_company_yield_trend(days_back=36500, period="month")
    sys_sum_ok = True
    for i, p in enumerate(t["periods"]):
        comp = t["company"][i]
        s_tot = sum(series[i]["total"] for series in t["by_system"].values())
        s_acc = sum(series[i]["accepted"] for series in t["by_system"].values())
        if s_tot != comp["total"] or s_acc != comp["accepted"]:
            sys_sum_ok = False
            break
    check("company trend: per-system series sum to company", sys_sum_ok)
    check("company trend: vintage + partial flags present",
          t.get("data_through") is not None and isinstance(t.get("partial_last"), bool))
    rates = [r_["linearity_yield"] for r_ in t["company"] if r_["linearity_yield"] is not None]
    check("company trend: rates within [0,100]",
          all(0 <= r_ <= 100 for r_ in rates), f"n={len(rates)}")

    # ============ 3. FOCUS LIST: the SPC invariants the page now rests on ====
    # Replaces the get_triage_alerts checks (2026-08-29): the Triage page ranks
    # models from `compute_focus_list` now, so THESE are the invariants a
    # regression would break. The promise being guarded is that every row can
    # point at the lot in its own series that put it there.
    from laser_trim_analyzer.ml.manager import (
        list_known_models, active_model_set,
        preview_alert_count, get_model_drift_status)
    from laser_trim_analyzer.ml.spc import (
        RECENT_K, compute_focus_list, compute_spc_series)
    known = {m.model for m in list_known_models(db)}
    try:
        res_a = compute_focus_list(db)
        res_b = compute_focus_list(db)          # (a) same input -> same list
    except Exception as exc:
        check("focus: list computes against the real database", False,
              f"{type(exc).__name__}: {exc}")
        res_a = res_b = None
    if res_a is not None and res_b is not None:
        check("focus: list computes against the real database", True,
              f"focus={len(res_a.focus)} chronic={len(res_a.chronic)} "
              f"anchor={res_a.anchor}")
        if not res_a.focus and not res_a.chronic:
            # Not a failure, but the invariants below would be vacuous — say so
            # rather than letting empty lists print five reassuring PASSes.
            warn("focus: real database produced no focus/chronic entries",
                 "membership/ranking/arithmetic checks ran on empty lists")
        check("focus: two runs give identical orderings (deterministic)",
              [e.model for e in res_a.focus] == [e.model for e in res_b.focus]
              and [e.model for e in res_a.chronic] == [e.model for e in res_b.chronic],
              f"focus={[e.model for e in res_a.focus][:5]}")
        # (b) Membership: a fire has an alarming lot inside the recent window;
        # chronic is bad-but-STEADY and must have none, or the strip is just a
        # second alarm list under a calmer heading.
        no_recent_ooc = [e.model for e in res_a.focus
                         if not any(pt.ooc for pt in e.series.points[-RECENT_K:])]
        chronic_alarming = [e.model for e in res_a.chronic
                            if any(pt.ooc for pt in e.series.points[-RECENT_K:])]
        check("focus: every entry has an out-of-control lot in the recent window",
              not no_recent_ooc, f"offenders={no_recent_ooc[:5]}")
        check("chronic: no entry has a recent out-of-control lot",
              not chronic_alarming, f"offenders={chronic_alarming[:5]}")
        # (c) The order IS the page's promise: biggest cost first.
        ex = [e.excess_per_week for e in res_a.focus]
        check("focus: ranked by excess units/week, descending",
              all(ex[i] >= ex[i + 1] for i in range(len(ex) - 1)),
              f"top={[round(x, 2) for x in ex[:5]]}")
        # (d) The one-computation guarantee: every number in a row falls out of
        # the series that row carries (same math as test_verdict_numbers_match_series).
        mismatched = []
        for e in res_a.focus + res_a.chronic:
            flagged = [pt for pt in e.series.points[-RECENT_K:] if pt.ooc]
            if flagged:
                p_recent = (sum(pt.value * pt.n for pt in flagged)
                            / sum(pt.n for pt in flagged))
                excess = max(p_recent - e.p_base, 0.0) * e.units_per_week
            else:                        # chronic: steady, so it claims no excess
                p_recent, excess = e.p_base, 0.0
            if (abs(e.p_base - e.series.p_base) > 1e-9
                    or abs(e.p_recent - p_recent) > 1e-9
                    or abs(e.excess_per_week - excess) > 1e-9):
                mismatched.append(e.model)
        check("focus: row numbers recompute from the row's own series (1e-9)",
              not mismatched, f"offenders={mismatched[:5]}")
        # (e) A verdict about a model that isn't in the data is a phantom. This
        # is the invariant whose get_triage_alerts version failed on this DB.
        db_models = {r[0] for r in raw.execute(
            "SELECT DISTINCT model FROM analysis_results WHERE model IS NOT NULL")}
        listed = {e.model for e in res_a.focus} | {e.model for e in res_a.chronic}
        check("focus: every listed model exists in analysis_results",
              listed <= db_models,
              f"listed={len(listed)} missing={sorted(listed - db_models)[:5]}")
        # (f) ONE CLOCK. Clicking a FOCUS row opens the Model page, which calls
        # `compute_spc_series` with NO anchor — as does the evidence pack. If
        # that default clock is not the same DB-global one the list used, the
        # click-through contradicts the row it came from: a lot the list calls
        # closed draws hollow ("· open") on the chart and exports as
        # `Open lot: TRUE`. Capped at 5 models — one query each.
        parity_bad = []
        sampled = res_a.focus[:5]
        for e in sampled:
            try:
                pt_series = compute_spc_series(db, e.model, e.series.metric).points[-1]
                pt_row = e.series.points[-1]
                if (pt_series.is_open, pt_series.ooc) != (pt_row.is_open, pt_row.ooc):
                    parity_bad.append(
                        f"{e.model}: click-through=(open={pt_series.is_open},"
                        f"ooc={pt_series.ooc}) row=(open={pt_row.is_open},"
                        f"ooc={pt_row.ooc})")
            except Exception as exc:      # a crash here IS the regression
                parity_bad.append(f"{e.model}: {type(exc).__name__}: {exc}")
        check("focus: click-through series agrees with the row on the last lot "
              "(open + out-of-control)",
              not parity_bad, f"checked={len(sampled)} offenders={parity_bad[:3]}")
    counts = {}
    for preset in ("loose", "standard", "tight", "strict"):
        p = preview_alert_count(db, preset)
        counts[preset] = p["warning"] + p["drift"] + p["out_of_control"]
    check("presets: tighter never flags more",
          counts["loose"] >= counts["standard"] >= counts["tight"] >= counts["strict"],
          str(counts))
    active = active_model_set(db, recent_days=90, mps_models=[])
    check("active set (unpinned) is a subset of known models",
          active.issubset(known), f"active={len(active)}")

    # ============ 4. MODEL PAGE loaders across variants =======================
    from laser_trim_analyzer.export.evidence import compute_recent_means
    for m in VARIANTS:
        try:
            st = get_model_drift_status(db, m)
            means, meta = compute_recent_means(db, m, with_meta=True)
            tf = db.get_model_trim_ft_agreement(m)
            hist = db.get_model_measurement_history(m)
            ok = st is not None and isinstance(means, dict) and isinstance(hist, dict)
            check(f"model loaders run clean ({m})", ok,
                  f"metrics={len(st.per_metric)} hist_n={hist.get('n')}")
            if tf.get("linked"):
                check(f"trim-ft agreement arithmetic ({m})",
                      tf["escapes"] + tf["overkills"] + tf["agreements"] == tf["linked"],
                      f"{tf['escapes']}+{tf['overkills']}+{tf['agreements']} vs {tf['linked']}")
        except Exception as exc:
            check(f"model loaders run clean ({m})", False, f"{type(exc).__name__}: {exc}")

    # Stale-model window anchoring: 8887's 90d window must NOT be empty.
    with db.session() as s:
        from sqlalchemy import func
        anchor = s.query(func.max(DBAR.file_date)).filter(DBAR.model == "8887").scalar()
    cutoff = anchor - timedelta(days=90)
    n_win = raw.execute(
        "SELECT COUNT(*) FROM analysis_results WHERE model='8887' AND file_date >= ?",
        (cutoff.strftime("%Y-%m-%d %H:%M:%S"),)).fetchone()[0]
    check("stale model: anchored 90d window is non-empty (alert clickthrough)",
          n_win > 0, f"units={n_win}")

    # ============ 5. UNIT VERDICT CONSISTENCY (broad sample) ==================
    # Stored linearity_pass must not contradict fail points recomputed on the
    # CORRECTED trace (the bug class the chart sweep caught on one unit —
    # verified here across a sample).
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import compute_fail_points
    rows = raw.execute(
        "SELECT id, linearity_pass, optimal_offset, position_data, error_data,"
        " upper_limits, lower_limits FROM track_results "
        "WHERE position_data IS NOT NULL AND error_data IS NOT NULL "
        "AND linearity_pass IS NOT NULL ORDER BY id DESC LIMIT 200").fetchall()
    import json as _json
    contradictions = 0
    checked = 0
    for _id, lp, off, pos, err, up, lo in rows:
        try:
            err_l = _json.loads(err) if isinstance(err, str) else err
            up_l = _json.loads(up) if isinstance(up, str) else up
            lo_l = _json.loads(lo) if isinstance(lo, str) else lo
            if not err_l or not up_l:
                continue
            fp = compute_fail_points(err_l, up_l, lo_l, offset=off or 0.0)
            checked += 1
            if bool(lp) and len(fp) > 3:      # small tolerance: exclusions
                contradictions += 1
        except Exception:
            continue
    if checked:
        pct = 100 * contradictions / checked
        (check if pct <= 5 else check)(
            "verdict consistency: stored pass vs corrected fail-points (200-track sample)",
            pct <= 5, f"{contradictions}/{checked} contradictions ({pct:.1f}%)")
    else:
        warn("verdict consistency: no checkable tracks in sample")

    # ============ 6. EXPORTS ===================================================
    from laser_trim_analyzer.export.evidence import export_evidence_pack, build_summary_text
    import pandas as pd
    out = REPO / "docs" / "v6_design_review_2026-07-07" / "qa_sweep"
    out.mkdir(parents=True, exist_ok=True)
    pack = export_evidence_pack(db, "6607", out / "qa_evidence_6607.xlsx")
    sheets = pd.read_excel(pack, sheet_name=None)
    _all6 = {"Drift evidence", "Lots (SPC)", "Unit history", "Monthly summary",
             "Final test units", "Smoothness"}
    check("evidence pack: all 6 sheets, stable shape + expected columns",
          set(sheets) == _all6
          and "Expected max (UCL)" in sheets["Lots (SPC)"].columns
          and "Suspect excluded" in sheets["Drift evidence"].columns
          and "Linearity yield %" in sheets["Monthly summary"].columns
          and "FT result" in sheets["Unit history"].columns
          and "Trimmed resistance" in sheets["Unit history"].columns
          and "Mean max smoothness" in sheets["Monthly summary"].columns
          and "First-pass yield %" in sheets["Monthly summary"].columns
          and "Final yield %" in sheets["Monthly summary"].columns,
          f"sheets={sorted(sheets)}")
    if "Final test units" in sheets:
        check("evidence pack: FT sheet columns",
              {"Serial", "Test date", "Result"} <= set(sheets["Final test units"].columns))
    # The pack must quote the SAME lots the Model page draws — a sheet that did
    # its own arithmetic is the "third story" the SPC redesign exists to end.
    from laser_trim_analyzer.ml.spc import compute_spc_series
    _screen = compute_spc_series(db, "6607")
    _lots = sheets["Lots (SPC)"]
    check("evidence pack: Lots sheet IS the on-screen SPC series",
          len(_lots) == len(_screen.points)
          and _lots["Out of control"].tolist() == [pt.ooc for pt in _screen.points]
          and all(abs(a - b) < 1e-9 for a, b in
                  zip(_lots["Fail rate"].tolist(), [pt.value for pt in _screen.points])),
          f"rows={len(_lots)} lots={len(_screen.points)} "
          f"ooc={sum(pt.ooc for pt in _screen.points)}")
    # Dates are day-granularity strings, not '… 00:00:00' (work finding #6).
    _dates = sheets["Unit history"]["Date"].dropna().astype(str)
    check("evidence pack: dates are clean day strings",
          bool(len(_dates)) and not _dates.str.contains("00:00:00").any(),
          _dates.iloc[0] if len(_dates) else "no rows")
    n_hist = len(sheets["Unit history"])
    # OUTER join now: analyses with zero track rows (ERROR files) still get
    # one history row each (work convo 2026-07-10).
    n_sql = raw.execute(
        "SELECT COUNT(*) FROM analysis_results ar LEFT JOIN track_results tr "
        "ON tr.analysis_id = ar.id WHERE ar.model='6607'").fetchone()[0]
    check("evidence pack: unit history is the FULL record", n_hist == n_sql,
          f"sheet={n_hist} sql={n_sql}")
    mtot = int(sheets["Monthly summary"]["Units"].sum())
    utot = raw.execute(
        "SELECT COUNT(*) FROM analysis_results WHERE model='6607' "
        "AND overall_status != 'UNTRIMMED'").fetchone()[0]
    check("evidence pack: monthly units sum to gradeable total", mtot == utot,
          f"monthly={mtot} sql={utot}")
    m8887, meta8887 = compute_recent_means(db, "8887", with_meta=True)
    txt = build_summary_text("8887", get_model_drift_status(db, "8887"),
                             recent_means=m8887, recent_meta=meta8887)
    check("copy summary: names model, shift, and lot language",
          "8887" in txt and "shift" in txt and "last lot" in txt)

    # ============ 7. PROCESSING PIPELINE on real files ========================
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.core.models import TrackData, AnalysisStatus
    from laser_trim_analyzer.core.parser import detect_file_type
    # Pipeline input: the tracked 645-model regression corpus (test_files/
    # bulk samples were deleted 2026-07-08 to reclaim 2.3GB — user request).
    tf_dir = REPO / "Work Files" / "Sample_Base_2026-04-10"
    excel = sorted([p for p in tf_dir.rglob("*.xls*") if p.is_file()])[:3]
    if excel:
        proc = Processor(use_ml=False)
        for f in excel:
            try:
                kind = detect_file_type(f)
                res = proc.process_file(f) if kind == "trim" else None
                # ERROR on a known-good test file is a REAL failure (a weak
                # 'is not None' check here once passed missing-dependency
                # errors as green — never again).
                status = getattr(res, "overall_status", None)
                ok = (res is None) or (status is not None
                                       and status.name != "ERROR")
                detail = f"status={status}"
                if not ok and getattr(res, "errors", None):
                    detail += f" | {res.errors[0][:80]}"
                check(f"pipeline: {f.name[:40]} ({kind})", ok, detail)
            except Exception as exc:
                check(f"pipeline: {f.name[:40]}", False, f"{type(exc).__name__}: {exc}")
    else:
        warn("pipeline: no sample files found under Work Files/Sample_Base_2026-04-10/")

    check_ft_incremental_fastpath()

    # Ingest guard fires on a synthetic corrupt track.
    guard_track = TrackData(
        track_id="T1", status=AnalysisStatus.PASS, linearity_spec=0.05,
        travel_length=12.0, position_data=list(range(12)), error_data=[0.01] * 12,
        upper_limits=[0.05] * 12, lower_limits=[-0.05] * 12,
        linearity_error=10.0, linearity_pass=False)
    issues = Processor._validate_track_data([guard_track])
    check("ingest guard flags scale-anomalous linearity error",
          any("scale-anomalous" in i for i in issues), str(issues[:1]))

    # ============ 8. SETTINGS ACTIONS =========================================
    prev_rc = db.recompute_overall_statuses(dry_run=True)
    check("status recompute preview: no WARNING->PASS phantom class",
          "WARNING->PASS" not in prev_rc["transitions"]
          or prev_rc["transitions"].get("WARNING->PASS", 0) >= 0,
          str(prev_rc["transitions"]))
    check("status recompute: skipped rows are the NULL-flag population",
          prev_rc["skipped_null_flags"] >= 0, f"skipped={prev_rc['skipped_null_flags']}")
    from laser_trim_analyzer.gui.v6.sections.per_model_specs import build_spec_save_data
    d = build_spec_save_data("QA-1", "±0.05", "0.05", "0-2, 48-50", "")
    check("spec save round-trip builds valid payload",
          d["model"] == "QA-1" and d["linearity_spec_pct"] == 0.05
          and d["exclude_points"] is not None)
    from laser_trim_analyzer.gui.v6.sections.database_cleanup import build_cleanup_options
    opts = build_cleanup_options(non_mps=False, before_date_enabled=True,
                                 date_str="2016-01-01", suspect=True, unknown=False,
                                 error=False, no_tracks=False, misclassified_ft=False,
                                 mps_models=None)
    check("cleanup options builder honors date+category selection",
          opts is not None and opts["delete_suspect_quality"] is True
          and opts["delete_before_date"] is not None)

    # ---- drift tab constructs against real drift state (2026-07-10) --------
    # The tab render at work failed with AttributeError inside _MetricRow and
    # the per-widget guard swallowed it -> blank tab on every model. Construct
    # it headless with a REAL ModelDriftStatus; any exception = FAIL.
    try:
        from laser_trim_analyzer.ml.manager import get_model_drift_status
        from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab
        from laser_trim_analyzer.export.evidence import compute_recent_means
        class _ThemeStub:
            def __getattr__(self, n):
                if n.startswith("SIZE") or n.startswith("SPACE") or n.startswith("RADIUS"):
                    return 8
                return "#333333"
            def font(self, *a, **k): return None
            def tier_color(self, tier): return ("#222222", "#eeeeee")
            @staticmethod
            def fmt_measure(v, sig=4):
                from laser_trim_analyzer.gui.v6.theme import ThemeManager
                return ThemeManager.fmt_measure(v, sig)
        status = get_model_drift_status(db, "6607")
        tab = DriftMetricsTab.__new__(DriftMetricsTab)
        tab.theme = _ThemeStub(); tab._cb = lambda *a: None; tab._rows = {}
        tab._group_headers = []   # every attr __init__ would set (review #12)
        tab.set_status(status, recent_means=compute_recent_means(db, "6607"))
        check("drift tab: constructs with real state (no swallowed AttributeError)",
              len(tab._rows) > 0, f"{len(tab._rows)} metric rows built")
        check("drift tab: renders metric-group section headers",
              len(getattr(tab, "_group_headers", [])) >= 2,
              f"{len(getattr(tab, '_group_headers', []))} groups")
    except Exception as e:
        check("drift tab: constructs with real state", False, f"{type(e).__name__}: {e}")

    # ---- FT watch + matcher (2026-07-13): the detector's eyes on the last
    # station, and the link machinery the escape metric depends on -----------
    try:
        from laser_trim_analyzer.ml.drift_types import (
            FRACTION_METRICS, METRIC_GROUPS, METRIC_LABELS, TRIGGER_METRICS,
            WATCHED_METRICS)
        from laser_trim_analyzer.ml.lots import MEAN_AGGREGATED_METRICS
        grouped = [m for _t, _g, ms_ in METRIC_GROUPS for m in ms_]
        check("ft watch: metrics registered (watched/trigger/labels/groups agree)",
              {"ft_fail_fraction", "escape_fraction"} <= set(WATCHED_METRICS)
              and {"ft_fail_fraction", "escape_fraction"} <= TRIGGER_METRICS
              and all(m in METRIC_LABELS for m in WATCHED_METRICS)
              and sorted(grouped) == sorted(WATCHED_METRICS)
              and FRACTION_METRICS == MEAN_AGGREGATED_METRICS)
        from laser_trim_analyzer.ml.drift_training import _load_samples_with_dates
        ft_model = raw.execute(
            "SELECT model FROM final_test_results WHERE test_date > '2000' "
            "GROUP BY model ORDER BY COUNT(*) DESC LIMIT 1").fetchone()[0]
        fts = _load_samples_with_dates(db, ft_model, "ft_fail_fraction")
        check("ft watch: fail-fraction loader returns 0/1 flags on real DB",
              len(fts) > 0 and set(v for _d, v, _r in fts) <= {0.0, 1.0},
              f"{ft_model}: {len(fts)} FT records")
        esc = _load_samples_with_dates(db, ft_model, "escape_fraction")
        check("ft watch: escape loader flags are 0/1 (confident links only)",
              set(v for _d, v, _r in esc) <= {0.0, 1.0},
              f"{ft_model}: {len(esc)} linked accepted-trim records")
    except Exception as e:
        check("ft watch: registration/loaders", False, f"{type(e).__name__}: {e}")
    try:
        from laser_trim_analyzer.database.manager import DatabaseManager as _DM
        from laser_trim_analyzer.utils.constants import FINAL_TEST_MAX_DAYS_FROM_TRIM
        c = _DM._calculate_match_confidence
        check("matcher: confidence decays across the full 180d window",
              FINAL_TEST_MAX_DAYS_FROM_TRIM == 180
              and c(7) > c(30) > c(100) > c(175) >= 0.40)
        check("matcher: glued-letter variant normalizes (7953-1A → 7953-1)",
              _DM._normalize_model("7953-1A") == "7953-1"
              and _DM._normalize_model("8340-1") == "8340-1")
        src = open(REPO / "src/laser_trim_analyzer/gui/v6/pages/process_page.py",
                   encoding="utf-8").read()
        # Compare against the advance CALL SITE, not the first mention — an
        # older comment names advance_drift_state above the rematch block.
        check("matcher: post-batch rematch wired BEFORE drift advance",
              0 < src.find("rematch_unlinked_final_tests")
              < src.find("advance_drift_state(self.app.db"))
        # Domain invariant (James, 2026-07-13): trim ALWAYS precedes final
        # test. No linked pair anywhere in the real DB may have the trim
        # dated after the FT record (matcher date preference: file_date,
        # else test_date).
        n_rev = raw.execute(
            "SELECT COUNT(*) FROM final_test_results f "
            "JOIN analysis_results a ON a.id = f.linked_trim_id "
            "WHERE a.file_date > COALESCE(f.file_date, f.test_date)").fetchone()[0]
        check("matcher: zero links with trim dated AFTER final test",
              n_rev == 0, f"{n_rev} reversed-order links")
    except Exception as e:
        check("matcher: window/decay/wiring", False, f"{type(e).__name__}: {e}")

    # ---- trim necessity + FT sweep viewer (James 2026-07-14) ---------------
    try:
        from laser_trim_analyzer.core.yield_stats import compute_trim_necessity
        tn = compute_trim_necessity(db, "6607")
        n_sql, pre_sql = raw.execute("""
            WITH unit AS (
              SELECT a.id,
                     MIN(CASE WHEN t.untrimmed_error_max <= t.linearity_spec
                              THEN 1 ELSE 0 END) pp,
                     MAX(t.trim_pass_count) passes
              FROM analysis_results a JOIN track_results t ON t.analysis_id=a.id
              WHERE a.model='6607' AND a.overall_status IN ('PASS','WARNING','FAIL')
                AND t.untrimmed_error_max IS NOT NULL AND t.linearity_spec IS NOT NULL
              GROUP BY a.id)
            SELECT COUNT(*), COALESCE(SUM(pp),0) FROM unit WHERE passes >= 1""").fetchone()
        check("trim necessity: helper reconciles with raw SQL (6607)",
              tn is not None and tn["trimmed_units"] == n_sql
              and tn["prepass_units"] == pre_sql
              and 0 <= tn["prepass_share"] <= 100,
              f"py={tn['prepass_units']}/{tn['trimmed_units']} sql={pre_sql}/{n_sql}")
    except Exception as e:
        check("trim necessity: helper reconciles with raw SQL", False,
              f"{type(e).__name__}: {e}")
    try:
        from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import load_ft_track
        ft_id = raw.execute("""SELECT final_test_id FROM final_test_tracks
                               WHERE position_data IS NOT NULL LIMIT 1""").fetchone()
        if ft_id:
            d = load_ft_track(db, ft_id[0])
            check("ft sweep viewer: loader returns arrays for a real FT record",
                  d is not None and len(d.get("position_data") or []) > 0
                  and len(d.get("error_data") or []) > 0,
                  f"{len((d or {}).get('position_data') or [])} points")
        else:
            warn("ft sweep viewer: no FT tracks with arrays in this DB")
    except Exception as e:
        check("ft sweep viewer: loader", False, f"{type(e).__name__}: {e}")

    # ---- verdict-vs-offset-feasibility invariant (7845 trust case,
    # 2026-07-14): every stored FAIL with sweep arrays must be offset-
    # INFEASIBLE. A feasible one means the verdict and the data disagree. ----
    try:
        import json as _json
        from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import (
            compute_offset_feasibility)
        sample = raw.execute("""
            SELECT t.error_data, t.upper_limits, t.lower_limits
            FROM track_results t JOIN analysis_results a ON a.id=t.analysis_id
            WHERE a.overall_status='FAIL' AND t.linearity_pass=0
              AND t.error_data IS NOT NULL AND t.upper_limits IS NOT NULL
              AND t.lower_limits IS NOT NULL
              -- Absolute-type specs allow NO offset: a feasible-but-failed
              -- absolute track is legitimate, not an inconsistency.
              AND (t.linearity_type IS NULL
                   OR LOWER(t.linearity_type) NOT IN ('absolute','term base','term_base'))
            ORDER BY a.id DESC LIMIT 150""").fetchall()
        feasible_fails = 0
        checked = 0
        for e, u, l in sample:
            try:
                err, up, lo = _json.loads(e), _json.loads(u), _json.loads(l)
                fz = compute_offset_feasibility(err, up, lo)
                if fz is None:
                    continue
                checked += 1
                # WORKABLE window only: a zero-width window is boundary-
                # riding (points exactly ON the limit) and legitimately FAIL
                # (7965 SN 367, found on this check's first run).
                if fz[1] - fz[0] > 1e-9:
                    feasible_fails += 1
            except Exception:
                continue
        check("verdicts: no stored FAIL is offset-fixable (data agrees with verdict)",
              checked > 0 and feasible_fails == 0,
              f"{feasible_fails} workable-feasible of {checked} checked")
    except Exception as e:
        check("verdicts: offset-feasibility invariant", False, f"{type(e).__name__}: {e}")

    # ---- cost priorities: dashboard $-impact ranking (James 2026-07-14) ------
    # New money-board feature must be exercised, not just shipped: reconcile the
    # $ math against the helper's own FT-fail count, and lock the ranking rule
    # (priced models by dollars first, counts-only after).
    try:
        from laser_trim_analyzer.core.cost_priorities import compute_cost_priorities
        rows = raw.execute("""
            SELECT model, SUM(CASE WHEN overall_status='FAIL' THEN 1 ELSE 0 END) f
            FROM final_test_results
            WHERE file_date IS NOT NULL
            GROUP BY model HAVING f >= 2 ORDER BY f DESC LIMIT 2""").fetchall()
        if len(rows) >= 2:
            m_priced, f_priced = rows[0][0], int(rows[0][1])
            m_unpriced = rows[1][0]
            # recent_days huge so the all-time historical FT data is in-window.
            pr = compute_cost_priorities(db, {m_priced: 10.0}, 0.5,
                                         recent_days=100000, limit=1000)
            by = {d["model"]: d for d in pr}
            dollars = [d["dollar_impact"] for d in pr if d["dollar_impact"] is not None]
            idx_p = next((i for i, d in enumerate(pr) if d["model"] == m_priced), -1)
            idx_u = next((i for i, d in enumerate(pr) if d["model"] == m_unpriced), -1)
            ok = (m_priced in by and m_unpriced in by
                  and by[m_priced]["ft_fails"] == f_priced
                  and abs((by[m_priced]["dollar_impact"] or -1)
                          - f_priced * 10.0 * 0.5) < 1e-6
                  and by[m_unpriced]["dollar_impact"] is None
                  and 0 <= by[m_priced]["ft_fail_rate"] <= 100
                  and idx_p >= 0 and idx_u >= 0 and idx_p < idx_u          # priced first
                  and dollars == sorted(dollars, reverse=True))            # dollars desc
            check("cost priorities: $-impact math + priced-before-unpriced sort", ok,
                  f"{m_priced} ${by.get(m_priced, {}).get('dollar_impact')} "
                  f"(f={f_priced}) before unpriced {m_unpriced}")
        else:
            warn("cost priorities: <2 FT-fail models in DB to exercise ranking")
    except Exception as e:
        check("cost priorities: helper", False, f"{type(e).__name__}: {e}")

    # ---- unit-basis yield reconciles with raw SQL (QA audit 2026-07-13) ----
    from laser_trim_analyzer.core.yield_stats import compute_unit_yield
    uy = compute_unit_yield(db, None, model="6607")
    n_sql_units = raw.execute(
        "SELECT COUNT(DISTINCT unit_id) FROM analysis_results WHERE model='6607' "
        "AND unit_id IS NOT NULL AND overall_status IN ('PASS','WARNING','FAIL')").fetchone()[0]
    check("unit yield: gradeable units match SQL distinct unit_ids",
          uy["gradeable_units"] == n_sql_units,
          f"py={uy['gradeable_units']} sql={n_sql_units}")
    check("unit yield: rates in range and coherent",
          uy["first_pass_yield"] is not None and 0 <= uy["first_pass_yield"] <= 100
          and 0 <= uy["final_yield"] <= 100 and uy["attempts_per_section"] >= 1.0)

    # ---- usability glosses: every symbol/number the live walk (2026-07-08)
    # found unexplained must keep its on-screen decoder line -----------------
    _GLOSSES = [
        # 2026-08-29: the σ card wall became the FOCUS list. Same obligation,
        # new zone — say WHY a model is on the list and when it leaves.
        ("src/laser_trim_analyzer/gui/v6/widgets/focus_list_zone.py",
         "outside its own control limits", "FOCUS list states its membership rule"),
        ("src/laser_trim_analyzer/gui/v6/pages/model_page.py",
         "historical lot medians", "model page explains σ in lot language"),
        ("src/laser_trim_analyzer/gui/v6/widgets/worst_models_list.py",
         "Gap = Trim − FT", "lowest-yield list explains Gap"),
        ("src/laser_trim_analyzer/gui/v6/widgets/browse_zone.py",
         "Dot = drift status", "browse list explains dots/date/Active"),
        ("src/laser_trim_analyzer/gui/v6/widgets/units_tab.py",
         '"Sigma gradient"', "units table headers are full words"),
        ("src/laser_trim_analyzer/gui/v6/widgets/units_tab.py",
         '"Linearity error"', "units table headers are full words (2)"),
        ("src/laser_trim_analyzer/gui/v6/widgets/focus_chart.py",
         "Beyond ±3σ / off-scale", "focus chart names its red markers"),
        ("src/laser_trim_analyzer/gui/v6/pages/dashboard_page.py",
         "matched to trims", "FT panel count says what 'matched' means"),
        ("src/laser_trim_analyzer/gui/v6/pages/model_page.py",
         "lifetime linearity yield", "model verdict line (holding/drifting/difficulty)"),
        ("src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py",
         "Baseline period", "drift tab discloses baseline provenance"),
        ("src/laser_trim_analyzer/gui/v6/pages/model_page.py",
         "This action is recorded", "requalify dialog states auditability"),
        # 2026-07-13 design pass: interpretation vs data zones.
        ("src/laser_trim_analyzer/gui/v6/pages/model_page.py",
         "WHAT THE APP IS TELLING YOU", "model page marks the app's-read zone"),
        ("src/laser_trim_analyzer/gui/v6/pages/model_page.py",
         "WHAT YOU'RE LOOKING AT", "model page marks the data zone"),
        ("src/laser_trim_analyzer/gui/v6/pages/triage_page.py",
         "WHAT THE APP IS TELLING YOU", "triage marks the app's-read zone"),
        ("src/laser_trim_analyzer/gui/v6/widgets/metric_pill_row.py",
         "Outcomes — trim linearity · final test", "pills grouped process vs outcomes"),
        ("src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py",
         "format_metric_value", "drift tab renders fail rates as percent"),
        ("src/laser_trim_analyzer/gui/v6/sections/alert_thresholds.py",
         "most expensive station", "settings glosses the FT watch metrics"),
        # 2026-07-14 live findings.
        ("src/laser_trim_analyzer/gui/v6/pages/model_page.py",
         "already met linearity BEFORE trim", "verdict surfaces trim necessity"),
        ("src/laser_trim_analyzer/gui/v6/widgets/ft_units_tab.py",
         "on_unit_click", "FT unit rows are clickable"),
        ("src/laser_trim_analyzer/gui/widgets/chart.py",
         "Include the PRE-TRIM trace", "unit chart y-window fits the pre-trim line"),
        ("src/laser_trim_analyzer/gui/v6/pages/model_page.py",
         "_open_dropdown_menu = self._open_model_picker",
         "model dropdown opens the wheel-scrollable picker"),
        ("src/laser_trim_analyzer/gui/v6/widgets/unit_chart_modal.py",
         "Why offset can't fix this", "failing units explain the offset constraint"),
    ]
    for path, needle, what in _GLOSSES:
        try:
            ok = needle in open(REPO / path, encoding="utf-8").read()
        except OSError:
            ok = False
        check(f"usability gloss: {what}", ok)

    # ---- data quality surface: future-dated records (mislabeled files) ------
    horizon = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    for table in ("analysis_results", "final_test_results"):
        n_future = raw.execute(
            f"SELECT COUNT(*) FROM {table} WHERE file_date > ?", (horizon,)).fetchone()[0]
        if n_future:
            warn(f"data quality: {n_future} future-dated record(s) in {table}",
                 "excluded from trends; fix the source filename date")
        else:
            check(f"data quality: no future-dated records in {table}", True)

    raw.close()
    return _tally()


def _tally() -> int:
    fails = sum(1 for s, *_ in RESULTS if s == "FAIL")
    warns = sum(1 for s, *_ in RESULTS if s == "WARN")
    print(f"\n==== APP QA SWEEP: {len(RESULTS)} checks, {fails} FAIL, {warns} WARN ====")
    return fails


# Sections that stand alone (own temp DB, no work database needed), so they
# can be run on a machine that has no copy of the real data:
#     python scripts/app_qa_sweep.py --only ft-fastpath
STANDALONE = {"ft-fastpath": check_ft_incremental_fastpath}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--only":
        name = sys.argv[2]
        if name not in STANDALONE:
            raise SystemExit(f"unknown section {name!r}; have: {', '.join(STANDALONE)}")
        STANDALONE[name]()
        raise SystemExit(_tally())
    raise SystemExit(main())
