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
    from laser_trim_analyzer.core.ingest_run import discover_excel_files  # noqa: E402
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
                files, stats = discover_excel_files(str(tmp))
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

def check_model_stats_vs_sql(db, raw) -> None:
    """INVESTIGATE stats table == raw SQL, filter and all.

    Written as SQL that redoes the work independently: drop the records whose
    processing failed, take the median over the model's positive readings, then
    the [median/100, median*100] band, then COUNT/AVG/MIN/MAX inside it. If the
    module's filter drifted, or was never applied, these numbers separate
    immediately — 6607's untrimmed resistance reads 4,282 filtered against
    32,079 raw, and 8856's sigma gradient 0.0012 against 433.5.

    Any exception here is a FAIL, never a skip: a check that can pass on an
    ERROR result is the exact weak assertion CLAUDE.md forbids.
    """
    from laser_trim_analyzer.core.model_stats import (
        POSITIVE_RATIO, compute_model_stats, metric_policy)

    JOIN = ("FROM track_results t JOIN analysis_results a ON a.id = t.analysis_id "
            "WHERE a.model = ?")
    # A record whose processing FAILED holds sentinels, not measurements (all
    # 94 in the work DB carry sigma_gradient = 999.999). Out of both columns.
    USABLE = " AND a.overall_status NOT IN ('ERROR','PROCESSING_FAILED')"
    BROKEN = " AND a.overall_status IN ('ERROR','PROCESSING_FAILED')"
    LIN = " AND a.overall_status IN ('PASS','WARNING')"
    METRICS = ["untrimmed_resistance", "trimmed_resistance",
               "measured_electrical_angle", "final_linearity_error_shifted",
               "margin_to_spec", "sigma_gradient"]

    def sql_median(model, col):
        """Median of the model's POSITIVE usable readings — SQLite's
        ORDER/OFFSET idiom, averaging the two middle values on an even count
        exactly like Python's statistics.median."""
        n = raw.execute(f"SELECT COUNT(*) {JOIN}{USABLE} AND t.{col} > 0",
                        (model,)).fetchone()[0]
        if not n:
            return None
        return raw.execute(
            f"SELECT AVG(x) FROM (SELECT t.{col} AS x {JOIN}{USABLE} "
            f"AND t.{col} > 0 ORDER BY x LIMIT {2 - (n % 2)} "
            f"OFFSET {(n - 1) // 2})", (model,)).fetchone()[0]

    for model in ("6607", "8340-1", "8856"):
        try:
            stats = compute_model_stats(db, model)
            broken = raw.execute(f"SELECT COUNT(*) {JOIN}{BROKEN}",
                                 (model,)).fetchone()[0]
            check(f"stats table vs SQL: {model} failed-processing records dropped",
                  stats.errored == broken, f"py={stats.errored} sql={broken}")
            for col in METRICS:
                med = sql_median(model, col) if metric_policy(col) == POSITIVE_RATIO else None
                if med is not None:
                    keep = (f" AND t.{col} > 0 AND t.{col} BETWEEN {med / 100.0!r} "
                            f"AND {med * 100.0!r}")
                else:
                    keep = f" AND t.{col} IS NOT NULL"
                row = next(r for r in stats.rows if r.key == col)
                for label, cell, extra, errs in (("ALL", row.all_, USABLE, broken),
                                                 ("LIN", row.lin_passing, LIN, 0)):
                    n, avg, lo, hi = raw.execute(
                        f"SELECT COUNT(*), AVG(t.{col}), MIN(t.{col}), MAX(t.{col}) "
                        f"{JOIN}{extra}{keep}", (model,)).fetchone()
                    nulls = raw.execute(
                        f"SELECT COUNT(*) {JOIN}{extra} AND t.{col} IS NULL",
                        (model,)).fetchone()[0]
                    total = raw.execute(f"SELECT COUNT(*) {JOIN}{extra}",
                                        (model,)).fetchone()[0]
                    ok = (cell.n == n and cell.missing == nulls
                          and cell.errored == errs
                          and cell.excluded == total - n - nulls
                          and (n == 0 or (abs(cell.avg - avg) <= 1e-9 * max(1.0, abs(avg))
                                          and cell.low == lo and cell.high == hi)))
                    check(f"stats table vs SQL: {model} {col} [{label}]", ok,
                          f"py n={cell.n} avg={cell.avg} min={cell.low} max={cell.high} "
                          f"excl={cell.excluded} err={cell.errored} null={cell.missing} | "
                          f"sql n={n} avg={avg} min={lo} max={hi} "
                          f"excl={total - n - nulls} err={errs} null={nulls}")
            # Rate rows: the same discipline, and the NULL rows must sit in
            # `missing`, never in the denominator.
            for key, keep_sql, null_sql in (
                    ("trim_passed_linearity",
                     " AND t.linearity_pass = 1",
                     " AND t.linearity_pass IS NULL"),
                    ("already_met_spec",
                     " AND t.untrimmed_error_max <= t.linearity_spec",
                     " AND (t.untrimmed_error_max IS NULL OR t.linearity_spec IS NULL)")):
                row = next(r for r in stats.rows if r.key == key)
                for label, cell, extra in (("ALL", row.all_, USABLE),
                                           ("LIN", row.lin_passing, LIN)):
                    hits = raw.execute(f"SELECT COUNT(*) {JOIN}{extra}{keep_sql}",
                                       (model,)).fetchone()[0]
                    nulls = raw.execute(f"SELECT COUNT(*) {JOIN}{extra}{null_sql}",
                                        (model,)).fetchone()[0]
                    total = raw.execute(f"SELECT COUNT(*) {JOIN}{extra}",
                                        (model,)).fetchone()[0]
                    ok = (cell.count == hits and cell.missing == nulls
                          and cell.n == total - nulls
                          and (cell.n == 0
                               or abs(cell.pct - 100.0 * hits / cell.n) < 1e-9))
                    check(f"stats table vs SQL: {model} {key} [{label}]", ok,
                          f"py {cell.count}/{cell.n} null={cell.missing} | "
                          f"sql {hits}/{total - nulls} null={nulls}")
            # The corrupt-reading disclosure is the whole point of the filter:
            # 6607 must SAY it dropped readings, not silently launder them.
            if model == "6607":
                ur = next(r for r in stats.rows if r.key == "untrimmed_resistance")
                check("stats table: 6607 discloses the open-circuit readings it dropped",
                      ur.all_.excluded >= 7 and ur.all_.avg < 5000.0
                      and ur.all_.high < 1e5,
                      f"excluded={ur.all_.excluded} avg={ur.all_.avg:.1f} "
                      f"max={ur.all_.high:.0f}")
            # 8856 is the ERROR-sentinel case: 75 of its 173 sigma readings are
            # 999.999 on records that failed processing. Averaged in they read
            # 433.5 against a true 0.0012 — a 430,000x error that no VALUE
            # policy catches (a band around a 0.001 median would delete
            # 8340-1's real 1.x values on failed units). Pinned by number.
            if model == "8856":
                sg = next(r for r in stats.rows if r.key == "sigma_gradient")
                check("stats table: 8856's sigma gradient is 0.0012, not 433",
                      sg.all_.avg is not None and sg.all_.avg < 0.01
                      and sg.all_.high < 0.01 and sg.all_.errored > 0,
                      f"avg={sg.all_.avg} max={sg.all_.high} "
                      f"errored={sg.all_.errored}")
        except Exception as exc:
            check(f"stats table vs SQL ({model})", False,
                  f"{type(exc).__name__}: {exc}")

    # Window + lot narrowing must actually narrow (and never widen).
    try:
        from sqlalchemy import func as _f
        from laser_trim_analyzer.database.models import AnalysisResult as _AR
        with db.session() as s:
            anchor = (s.query(_f.max(_AR.file_date))
                      .filter(_AR.model == "6607").scalar())
        full = compute_model_stats(db, "6607")
        win = compute_model_stats(db, "6607", cutoff=anchor - timedelta(days=90))
        check("stats table: a 90d window is a strict subset of all history",
              0 < win.tracks <= full.tracks,
              f"90d={win.tracks} all={full.tracks}")
    except Exception as exc:
        check("stats table: window narrowing", False, f"{type(exc).__name__}: {exc}")


def check_trim_ft_disposition_vs_sql(db, raw) -> None:
    """Escapes/overkills == raw SQL, and the re-trim confound stays fixed
    (2026-08-30).

    A unit that fails linearity is re-trimmed until it passes; every attempt
    writes its own file with the same calendar date, and the clock time that
    orders them lives only in the filename. While the parser discarded that
    time, all attempts tied on file_date and `_find_matching_trim` linked an
    arbitrary one — so a unit re-trimmed INTO spec, which final test then
    passed, was reported as the trim station's "overkill". On the work DB that
    was 124 phantom overkills and 32 hidden escapes over 12 months.

    The correction is bounded to the trim DAY on purpose: shop numbers get
    reused, so repeated serials span ~6 years and "the best run for this
    serial" would credit a 2012 unit's pass to a 2026 unit. The
    not-overcorrected check below is what pins that, and it is the one that
    fails if someone "simplifies" the fix into pooling by serial.

    Any exception here is a FAIL, never a skip: a check that can pass on an
    ERROR result is the exact weak assertion CLAUDE.md forbids.
    """
    from laser_trim_analyzer.database.manager import DatabaseManager

    CONF, DAYS = 0.70, 365
    cutoff = (datetime.now() - timedelta(days=DAYS)).strftime("%Y-%m-%d %H:%M:%S")

    # Re-derived independently of the ORM. The disposition is the UNIT-DAY's,
    # not the linked file's: per track take the LAST attempt of the day, then
    # require every track to pass. One unit-day spans several rows two ways —
    # one file per track on a multi-track unit (4,659 unit-days), and one file
    # per re-trim attempt on a track (13,104) — and they need opposite
    # treatment. `unit_id` ("<model>/<shop>/<date>") is the unit-day key.
    DISP = """
      att AS (
        SELECT a.unit_id uid, t.linearity_pass lp,
               ROW_NUMBER() OVER (PARTITION BY a.unit_id, t.track_id
                                  ORDER BY a.file_date DESC, a.id DESC) rn
        FROM analysis_results a JOIN track_results t ON t.analysis_id = a.id
        WHERE t.status <> 'UNTRIMMED' AND a.unit_id IS NOT NULL AND a.unit_id <> ''),
      disp AS (SELECT uid, MIN(CASE WHEN lp=1 THEN 1 ELSE 0 END) tp
               FROM att WHERE rn = 1 GROUP BY uid),
    """
    LINKED = """
      SELECT f.id fid, f.model, f.serial, a.id tid, date(a.file_date) tday,
             a.file_date tstamp, f.linearity_pass ft_pass,
             COALESCE(d.tp, MIN(CASE WHEN t.linearity_pass=1 THEN 1 ELSE 0 END))
               trim_pass
      FROM final_test_results f
      JOIN analysis_results a ON a.id = f.linked_trim_id
      JOIN track_results t ON t.analysis_id = a.id
      LEFT JOIN disp d ON d.uid = a.unit_id
      WHERE f.linked_trim_id IS NOT NULL AND f.linearity_pass IS NOT NULL
        AND f.match_confidence >= ? AND t.status <> 'UNTRIMMED' """
    WINDOW = " AND f.file_date >= ? "
    GROUP = " GROUP BY f.id, d.tp "

    # ---- 0. the repair actually ran on this database -----------------------
    # Rows the OLD parser wrote sit at midnight with the time still in the
    # filename. If any survive, linked_trim_id is still resolving same-day
    # attempts arbitrarily and every number below is measuring the old bug.
    # SQLite has no REGEXP, so the filename test runs in Python — through the
    # very parser the app uses, so the check tracks the parser rather than a
    # second copy of its pattern.
    from datetime import time as _time
    from laser_trim_analyzer.core.parser import ExcelParser
    _p = ExcelParser()
    midnight = raw.execute(
        "SELECT filename FROM analysis_results "
        "WHERE file_date IS NOT NULL "
        "AND strftime('%H:%M:%S', file_date) = '00:00:00'").fetchall()
    stranded = sum(1 for (fn,) in midnight
                   if (_ts := _p._extract_date_from_filename(fn or ""))
                   and _ts.time() != _time(0, 0))
    check("trim times: no row still stranded at midnight with a time in its "
          "filename", stranded == 0,
          f"stranded={stranded} of {len(midnight)} midnight rows | remedy: "
          f"python scripts/repair_trim_ft_links.py <db>")

    # ---- 1. linked trim IS the day's final attempt --------------------------
    late = raw.execute(f"""
      WITH {DISP} L AS ({LINKED}{GROUP})
      SELECT COUNT(*) FROM L
      WHERE EXISTS (SELECT 1 FROM analysis_results a2
                    WHERE a2.model = (SELECT model FROM analysis_results WHERE id = L.tid)
                      AND a2.serial = (SELECT serial FROM analysis_results WHERE id = L.tid)
                      AND date(a2.file_date) = L.tday
                      AND (a2.file_date > L.tstamp
                           OR (a2.file_date = L.tstamp AND a2.id > L.tid)))
    """, (CONF,)).fetchone()[0]
    check("trim/FT link points at the day's FINAL trim attempt", late == 0,
          f"links with a later same-day attempt = {late}" + ("" if late == 0 else
          " | same remedy: python scripts/repair_trim_ft_links.py <db> — the "
          "backfill gives same-day attempts an order and the rematch re-points "
          "the links; escapes/overkills are close but not exact until then"))

    # ---- 2. the confound itself is gone ------------------------------------
    # A unit whose every track's LAST attempt of the day passed must never be
    # counted as an overkill — that is the process working as designed. Note
    # this is per TRACK: a passing later attempt on Track A does NOT excuse a
    # Track B that never passed, which is why check 3's numbers moved up on
    # multi-track models rather than down.
    phantom = raw.execute(f"""
      WITH {DISP} L AS ({LINKED}{WINDOW}{GROUP})
      SELECT COUNT(*) FROM L
      WHERE L.trim_pass = 0 AND L.ft_pass = 1
        AND NOT EXISTS (
          SELECT 1 FROM analysis_results a2
          JOIN track_results t2 ON t2.analysis_id = a2.id
          WHERE a2.unit_id = (SELECT unit_id FROM analysis_results WHERE id = L.tid)
            AND t2.status <> 'UNTRIMMED'
            AND a2.id = (SELECT a3.id FROM analysis_results a3
                         JOIN track_results t3 ON t3.analysis_id = a3.id
                         WHERE a3.unit_id = a2.unit_id AND t3.track_id = t2.track_id
                           AND t3.status <> 'UNTRIMMED'
                         ORDER BY a3.file_date DESC, a3.id DESC LIMIT 1)
            AND (t2.linearity_pass IS NOT 1))
    """, (CONF, cutoff)).fetchone()[0]
    check("overkills: none where every track's last attempt of the day passed",
          phantom == 0, f"re-trim-confounded overkills = {phantom}")

    # ---- 3. API == raw SQL --------------------------------------------------
    n, esc, ovk = raw.execute(f"""
      WITH {DISP} L AS ({LINKED}{WINDOW}{GROUP})
      SELECT COUNT(*),
             SUM(CASE WHEN trim_pass=1 AND ft_pass=0 THEN 1 ELSE 0 END),
             SUM(CASE WHEN trim_pass=0 AND ft_pass=1 THEN 1 ELSE 0 END) FROM L
    """, (CONF, cutoff)).fetchone()
    api = db.get_escape_overkill_analysis(days_back=DAYS, min_confidence=CONF)
    check("Gap: company escapes/overkills match raw SQL",
          (api["total_linked"], api["escapes"], api["overkills"]) == (n, esc, ovk),
          f"api=({api['total_linked']},{api['escapes']},{api['overkills']}) "
          f"sql=({n},{esc},{ovk})")
    check("Gap: agreements complete the partition",
          api["escapes"] + api["overkills"] + api["agreements"] == api["total_linked"],
          f"{api['escapes']}+{api['overkills']}+{api['agreements']} "
          f"vs {api['total_linked']}")

    # ---- 4. per-model surface agrees with the company surface --------------
    # Same rows, same classifier — the trim-vs-FT tab and the Gap cannot drift.
    for m in ("6607", "8340-1", "8232-1"):
        mn, mesc, movk = raw.execute(f"""
          WITH {DISP} L AS ({LINKED} AND f.model = ? {WINDOW}{GROUP})
          SELECT COUNT(*),
                 SUM(CASE WHEN trim_pass=1 AND ft_pass=0 THEN 1 ELSE 0 END),
                 SUM(CASE WHEN trim_pass=0 AND ft_pass=1 THEN 1 ELSE 0 END) FROM L
        """, (CONF, m, cutoff)).fetchone()
        tf = db.get_model_trim_ft_agreement(
            m, cutoff_date=datetime.now() - timedelta(days=DAYS), min_confidence=CONF)
        check(f"trim-vs-FT tab matches raw SQL ({m})",
              (tf["linked"], tf["escapes"], tf["overkills"]) == (mn, mesc or 0, movk or 0),
              f"api=({tf['linked']},{tf['escapes']},{tf['overkills']}) "
              f"sql=({mn},{mesc},{movk})")
        check(f"trim-vs-FT tab lists one serial per counted unit ({m})",
              len(tf["escape_units"]) == tf["escapes"]
              and len(tf["overkill_units"]) == tf["overkills"],
              f"escape_units={len(tf['escape_units'])}/{tf['escapes']} "
              f"overkill_units={len(tf['overkill_units'])}/{tf['overkills']}")

    # ---- 5. NOT overcorrected ----------------------------------------------
    # The tempting "wrong" fix is to take the best/last run for the serial over
    # all history. Shop numbers get reused, so that credits a different physical
    # unit's pass and zeroes the metric out. These two pin that it did not.
    check("overkills survive the correction (metric not zeroed)",
          (ovk or 0) > 0, f"overkills={ovk}")
    cross_day = raw.execute(f"""
      WITH {DISP} L AS ({LINKED}{WINDOW}{GROUP}),
      RUN AS (SELECT a.id tid, a.model, a.serial, a.file_date,
                     MIN(CASE WHEN t.linearity_pass=1 THEN 1 ELSE 0 END) ap
              FROM analysis_results a JOIN track_results t ON t.analysis_id=a.id
              WHERE t.status <> 'UNTRIMMED' GROUP BY a.id)
      SELECT COUNT(*) FROM L
      WHERE L.trim_pass = 0 AND L.ft_pass = 1
        AND EXISTS (SELECT 1 FROM RUN r
                    JOIN analysis_results la ON la.id = L.tid
                    WHERE r.model = la.model AND r.serial = la.serial
                      AND date(r.file_date) <> L.tday AND r.ap = 1)
    """, (CONF, cutoff)).fetchone()[0]
    check("recycled shop numbers do NOT cancel overkills (correction is "
          "bounded to the trim day)", cross_day > 0,
          f"overkills whose serial passed on some OTHER day = {cross_day} "
          f"(pooling by serial would wrongly erase these)")

    # ---- 5b. dual-naming unit-days stay visible and stay handled -----------
    # Two unit-days in the work DB record the SAME physical tracks under BOTH
    # naming conventions on one day — 8555/25/2016-01-15 has TA/TB files
    # ('Track A', 'Track B') plus a legacy 8555_25.xls carrying TRK1+TRK2. The
    # disposition rule cannot know 'Track A' means the same element as 'TRK1',
    # so it evaluates 4 tracks for a 2-track unit. That errs conservatively —
    # an extra track can only make a unit FAIL, never hide a failure — which is
    # the safe direction for a zero-tolerance metric, so it is deliberately not
    # "fixed" by guessing that the two conventions are equivalent.
    #
    # The check exists because this population is invisible in the usual
    # one-file-per-track / re-trim split (those two shapes are `rows = tracks`
    # and `rows > tracks`; this is `rows < tracks` and falls in neither). If it
    # ever grows, the conservative bias grows with it and someone should decide
    # whether the conventions must be reconciled at ingest instead.
    dual = raw.execute("""
      WITH u AS (SELECT a.unit_id uid, COUNT(DISTINCT a.id) rows_,
                        COUNT(DISTINCT t.track_id) tracks
                 FROM analysis_results a JOIN track_results t ON t.analysis_id = a.id
                 WHERE a.unit_id IS NOT NULL AND a.unit_id <> ''
                 GROUP BY a.unit_id HAVING rows_ > 1)
      SELECT COUNT(*) FROM u WHERE rows_ < tracks""").fetchone()[0]
    check("unit-days mixing BOTH track-naming conventions stay a handful",
          dual <= 10,
          f"{dual} unit-day(s) with more distinct track_ids than files; the rule "
          f"treats 'Track A' and 'TRK1' as separate tracks (conservative)")
    # And they must not silently drop OUT of the metric: every such unit-day
    # still has to yield a disposition.
    undecided = raw.execute("""
      WITH att AS (
        SELECT a.unit_id uid, t.linearity_pass lp,
               ROW_NUMBER() OVER (PARTITION BY a.unit_id, t.track_id
                                  ORDER BY a.file_date DESC, a.id DESC) rn
        FROM analysis_results a JOIN track_results t ON t.analysis_id = a.id
        WHERE t.status <> 'UNTRIMMED' AND a.unit_id IS NOT NULL AND a.unit_id <> ''),
      u AS (SELECT a.unit_id uid, COUNT(DISTINCT a.id) rows_,
                   COUNT(DISTINCT t.track_id) tracks
            FROM analysis_results a JOIN track_results t ON t.analysis_id = a.id
            WHERE a.unit_id IS NOT NULL AND a.unit_id <> ''
            GROUP BY a.unit_id HAVING rows_ > 1)
      SELECT COUNT(*) FROM u
      WHERE u.rows_ < u.tracks
        AND (SELECT MIN(CASE WHEN lp=1 THEN 1 ELSE 0 END)
             FROM att WHERE att.uid = u.uid AND att.rn = 1) IS NULL""").fetchone()[0]
    check("dual-naming unit-days still produce a trim disposition",
          undecided == 0, f"{undecided} left undecided")

    # ---- 6. the classifier is the only definition --------------------------
    cls = DatabaseManager.classify_trim_ft
    check("classify_trim_ft covers the truth table exactly",
          (cls(True, False), cls(False, True), cls(True, True), cls(False, False))
          == (DatabaseManager.ESCAPE, DatabaseManager.OVERKILL,
              DatabaseManager.AGREEMENT, DatabaseManager.AGREEMENT),
          "escape / overkill / agreement / agreement")


def check_multi_folder_ingest() -> None:
    """Home's one-click batch IS the Process page's batch (2026-08-31).

    Standalone: real trim files, a throwaway DB, no work database needed.
        python scripts/app_qa_sweep.py --only ingest

    The contract this pins is the one a second implementation would break —
    every folder attempted in order, a dead folder reported instead of
    silently skipped, the counts adding up, and the second pass finding
    nothing new because the first pass really saved.
    """
    import shutil, tempfile  # noqa: E402
    from laser_trim_analyzer.config import Config  # noqa: E402
    from laser_trim_analyzer.core.ingest_run import (  # noqa: E402
        format_ingest_summary, run_folders)
    from laser_trim_analyzer.core.parser import detect_file_type  # noqa: E402
    from laser_trim_analyzer.database import manager as _dbmod  # noqa: E402
    from laser_trim_analyzer.database.manager import DatabaseManager  # noqa: E402
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR  # noqa: E402

    src = []
    for p in sorted((REPO / "Work Files" / "Sample_Base_2026-04-10").rglob("*.xls*")):
        if not p.is_file():
            continue
        try:
            if detect_file_type(p) != "trim":
                continue
        except Exception:
            continue
        src.append(p)
        if len(src) == 5:
            break
    if len(src) < 5:
        warn("multi-folder ingest: fewer than 5 trim samples available")
        return

    tmp = Path(tempfile.mkdtemp(prefix="ltaqa_ingest_"))
    saved_global = _dbmod._db_manager
    try:
        a, b = tmp / "laser_a", tmp / "laser_b"
        a.mkdir(); b.mkdir()
        for f in src[:3]:
            shutil.copy2(f, a / f.name)
        for f in src[3:]:
            shutil.copy2(f, b / f.name)
        offline = str(tmp / "offline_share")          # never created

        cfg = Config()
        cfg.database.path = tmp / "ingest_qa.db"
        db = DatabaseManager(cfg.database.path)
        # The processor reaches for the global manager for its incremental
        # index; without this it would consult (and open) the real database.
        _dbmod._db_manager = db

        seen = []
        rep = run_folders([str(a), offline, str(b)], db=db, config=cfg,
                          incremental=True,
                          on_folder_start=lambda i, n, f: seen.append(f))
        check("multi-folder ingest: every folder attempted, in order",
              seen == [str(a), offline, str(b)], f"{seen}")
        check("multi-folder ingest: the dead folder is reported, not skipped",
              [r.folder for r in rep.failed] == [offline]
              and bool(rep.failed[0].error), f"{[r.error for r in rep.failed]}")
        check("multi-folder ingest: a dead folder does not abort the rest",
              rep.new_files == 5 and rep.results[0].new_files == 3
              and rep.results[2].new_files == 2,
              f"new_files={rep.new_files} per-folder="
              f"{[r.new_files for r in rep.results]}")
        with db.session() as s:
            saved = s.query(DBAR.id).count()
        check("multi-folder ingest: every processed file really landed in the DB",
              saved == 5, f"rows={saved}")
        line = format_ingest_summary(rep)
        check("multi-folder ingest: the summary line counts and names the failure",
              "3 folders" in line and "5 new files" in line and offline in line,
              line)

        rep2 = run_folders([str(a), str(b)], db=db, config=cfg, incremental=True)
        with db.session() as s:
            saved2 = s.query(DBAR.id).count()
        check("multi-folder ingest: the second pass processes nothing new",
              rep2.ok and rep2.new_files == 0 and saved2 == 5,
              f"new_files={rep2.new_files} rows={saved2}")
        check("multi-folder ingest: 'no new files' is what an idle run says",
              "no new files" in format_ingest_summary(rep2),
              format_ingest_summary(rep2))
    except Exception as exc:
        check("multi-folder ingest: shared pipeline contract", False,
              f"{type(exc).__name__}: {exc}")
    finally:
        _dbmod._db_manager = saved_global
        shutil.rmtree(tmp, ignore_errors=True)


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
        # (c) The order IS the page's promise: biggest cost first — discounted
        # by the lots a model has run CLEAN since its alarm (2026-08-30, the
        # 6126 case: a hairline blip that has behaved since is not today's
        # fire). rank_score is what the list sorts on; excess_per_week stays
        # the measured number the verdict quotes, so it is NOT monotonic here.
        rs = [e.rank_score for e in res_a.focus]
        bad_score = [f"{e.model}: rank={e.rank_score:.3f} vs "
                     f"excess={e.excess_per_week:.3f}/(1+{e.clean_since})"
                     for e in res_a.focus
                     if abs(e.rank_score
                            - e.excess_per_week / (1.0 + e.clean_since)) > 1e-9]
        check("focus: ranked by rank_score (excess discounted by clean lots), "
              "descending",
              all(rs[i] >= rs[i + 1] for i in range(len(rs) - 1)) and not bad_score,
              f"top={[(e.model, round(e.rank_score, 2), e.clean_since) for e in res_a.focus[:5]]}"
              + (f" bad_score={bad_score[:3]}" if bad_score else ""))
        # The recovery marker and the ranking must tell the same story: a row
        # that says "has run at baseline since" is exactly a row discounted.
        marker_bad = [e.model for e in res_a.focus
                      if (" · has run at baseline since" in e.sub_line)
                      != (e.clean_since >= 1)]
        check("focus: the 'has run at baseline since' marker matches the discount",
              not marker_bad, f"offenders={marker_bad[:5]}")
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
        # (g) The likely-driver hint (2026-08-30) must be honest: either None
        # (rendered "driver unclear") or the plain-language label of a real,
        # NON-outcome watched metric. A raw key, an outcome metric, or free
        # text here means the enrichment drifted from drift_types' vocabulary.
        from laser_trim_analyzer.ml.drift_types import (
            FRACTION_METRICS, WATCHED_METRICS, metric_label)
        valid_labels = {metric_label(m) for m in WATCHED_METRICS
                        if m not in FRACTION_METRICS}
        bad_drivers = []
        for e in res_a.focus:
            if e.driver is None:
                continue
            if not any(e.driver.startswith(lbl) for lbl in valid_labels):
                bad_drivers.append(f"{e.model}: {e.driver!r}")
        for e in res_a.chronic:
            if e.driver is not None:     # chronic rows never carry a driver
                bad_drivers.append(f"{e.model} (chronic): {e.driver!r}")
        check("focus: driver hints name real process metrics (or are None)",
              not bad_drivers, f"offenders={bad_drivers[:3]}")
        # (h) Trim-vs-FT spec alignment (2026-08-30). 6126 is census-verified
        # ground truth: its linked trim/FT pairs disagree at essentially every
        # matched position, so the comparison MUST say "differs" here. This
        # check is the guard on the pairing as much as on the arithmetic — an
        # earlier cut sampled each station's newest tracks independently, which
        # on this model matched a fifth as many positions and read "aligned".
        try:
            from laser_trim_analyzer.core.spec_alignment import (
                compare_station_specs)
            c6126 = compare_station_specs(db, "6126")
            check("spec alignment: 6126's trim and FT specs differ (census "
                  "ground truth)", c6126.status == "differs",
                  f"status={c6126.status} matched={c6126.matched_positions} "
                  f"pct={c6126.pct_positions_differing:.2f} | {c6126.note}")
        except Exception as exc:
            check("spec alignment: 6126's trim and FT specs differ (census "
                  "ground truth)", False, f"{type(exc).__name__}: {exc}")
        # Every focus row must carry a real bool: the enrichment degrades to
        # False on failure, so a None/exception here means it did not run at all.
        try:
            flags = [(e.model, e.spec_mismatch) for e in res_a.focus]
            bad_flags = [m for m, v in flags if not isinstance(v, bool)]
            check("focus: every row carries a boolean spec_mismatch flag",
                  not bad_flags,
                  f"flagged={[m for m, v in flags if v]} offenders={bad_flags[:3]}")
        except Exception as exc:
            check("focus: every row carries a boolean spec_mismatch flag",
                  False, f"{type(exc).__name__}: {exc}")
    # ---- HOME and TRIAGE cannot disagree about what is drifting ------------
    # Two landing screens showing two different FOCUS lists would be worse
    # than either of them being wrong, so both go through focus_data.load_focus
    # and it must be a faithful pass-through of the computation.
    try:
        from laser_trim_analyzer.gui.v6 import focus_data
        from laser_trim_analyzer.gui.v6.pages import home_page, triage_page
        check("home/triage: both landing screens call ONE focus loader",
              home_page.load_focus is focus_data.load_focus
              is triage_page.load_focus)
        loaded, last_seen = focus_data.load_focus(db)
        raw_focus = compute_focus_list(db)
        check("home/triage: the loader passes the computation through untouched",
              [e.model for e in loaded.focus] == [e.model for e in raw_focus.focus]
              and [e.model for e in loaded.chronic] == [e.model for e in raw_focus.chronic]
              and loaded.anchor == raw_focus.anchor,
              f"{len(loaded.focus)} focus / {len(loaded.chronic)} chronic, "
              f"anchor={loaded.anchor}")
        expect_last = max((m.last_processed for m in list_known_models(db)
                           if m.last_processed), default=None)
        check("home/triage: the empty-state stamp is the newest data on record",
              last_seen == expect_last, f"{last_seen} vs {expect_last}")
    except Exception as exc:
        check("home/triage: one FOCUS loader behind both screens", False,
              f"{type(exc).__name__}: {exc}")

    # ---- every sidebar row points at a page that exists --------------------
    # A nav row whose key was never registered is a dead click with no error;
    # the keys are also what FOCUS deep-links navigate by, so they are a
    # contract, not decoration.
    try:
        import re as _re
        from laser_trim_analyzer.gui.v6.sidebar import Sidebar
        app_src = open(REPO / "src/laser_trim_analyzer/gui/v6/app.py",
                       encoding="utf-8").read()
        registered = set(_re.findall(r'add_page\(\s*"([a-z_]+)"', app_src))
        keys = [k for k, _ in Sidebar.ITEMS]
        check("shell: every sidebar row has a registered page",
              set(keys) == registered, f"sidebar={keys} registered={sorted(registered)}")
        check("shell: Home leads, Investigate keeps the 'model' key",
              keys[:3] == ["home", "model", "settings"]
              and dict(Sidebar.ITEMS)["model"] == "Investigate"
              and Sidebar.MUTED == {"dashboard", "triage", "process"},
              f"{Sidebar.ITEMS}")
        check("shell: nothing reachable was lost",
              {"dashboard", "triage", "process"} <= registered,
              f"registered={sorted(registered)}")
    except Exception as exc:
        check("shell: sidebar/page registration contract", False,
              f"{type(exc).__name__}: {exc}")

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

    # ---- INVESTIGATE stats table vs RAW SQL (2026-08-30) -------------------
    # The table replaces an Excel round trip, so it has to agree with the
    # database to the digit. The SQL below REPRODUCES the plausibility filter
    # (median, then the 100x band) rather than assuming it: on 6607 the raw
    # average of untrimmed_resistance is 32,079 ohms against a true 4,282, so a
    # check that compared against a bare AVG() would pass on the wrong number.
    check_model_stats_vs_sql(db, raw)

    # ---- trim-vs-FT disposition vs RAW SQL (2026-08-30) --------------------
    # Escapes/overkills read the LAST trim attempt of the day, because a unit
    # is re-trimmed until it passes and only the final attempt is the
    # disposition it carried to final test. Bounded to the day on purpose:
    # shop numbers get reused across lots.
    check_trim_ft_disposition_vs_sql(db, raw)

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
    _all7 = {"Drift evidence", "Lots (SPC)", "Unit history", "Monthly summary",
             "Final test units", "Smoothness", "Stats table"}
    check("evidence pack: all 7 sheets, stable shape + expected columns",
          set(sheets) == _all7
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
    # ---- the Excel stats sheet IS the on-screen table (spec line 95) -------
    # Not "agrees with": the same characters. The sheet is what James hands an
    # engineer, so a value rounded differently there than on the page he read
    # it off is the contradiction this whole redesign exists to end. Rebuilt
    # here from the SCREEN's own helpers and compared cell by cell.
    try:
        from laser_trim_analyzer.core.model_stats import (
            cell_texts, compute_model_stats, disclosure_text)
        _stats = compute_model_stats(db, "6607")
        _sheet = pd.read_excel(pack, sheet_name="Stats table", header=2)
        _by_metric = {r["Metric"]: r for _, r in _sheet.iterrows()}
        _bad = []
        for _row in _stats.rows:
            _cells = _by_metric.get(_row.label)
            if _cells is None:
                _bad.append(f"{_row.label}: missing from the sheet")
                continue
            for _side, _cell, _prefix in (("ALL", _row.all_, "ALL"),
                                          ("LIN", _row.lin_passing, "LIN-PASSING")):
                _shown = cell_texts(_row, _cell)
                # n stays a NUMBER in the sheet so Excel can sort and sum it;
                # the screen prints the same number with a thousands separator.
                # Everything else is compared as characters.
                if int(_cells[f"{_prefix} n"]) != _cell.n:
                    _bad.append(f"{_row.label}[{_side}] n "
                                f"{_cells[f'{_prefix} n']} != {_cell.n}")
                _sheet_cells = [_cells[f"{_prefix} avg / count"],
                                _cells[f"{_prefix} min / %"]]
                if _row.kind == "distribution":
                    _sheet_cells.append(_cells[f"{_prefix} max"])
                if _sheet_cells != _shown[1:]:
                    _bad.append(f"{_row.label}[{_side}] {_sheet_cells} != {_shown[1:]}")
            _left = _cells["Left out"]
            _left = _left if isinstance(_left, str) else ""
            if _left != disclosure_text(_row.all_):
                _bad.append(f"{_row.label} disclosure {_left!r} "
                            f"!= {disclosure_text(_row.all_)!r}")
        check("evidence pack: Stats sheet is character-for-character the screen",
              not _bad and len(_sheet) == len(_stats.rows),
              "; ".join(_bad[:3]) or f"{len(_sheet)} rows match the table")
        # The window and the lot the numbers describe, above the table: a
        # column of numbers with neither on it is not evidence.
        _head = pd.read_excel(pack, sheet_name="Stats table", header=None, nrows=2)
        check("evidence pack: Stats sheet says which window and lot it describes",
              "track measurements over" in str(_head.iloc[0, 0])
              and str(_head.iloc[1, 0]).strip() not in ("", "nan"),
              f"{str(_head.iloc[0, 0])[:60]} | {str(_head.iloc[1, 0])[:40]}")
    except Exception as _exc:
        check("evidence pack: Stats sheet is character-for-character the screen",
              False, f"{type(_exc).__name__}: {_exc}")
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
    check_multi_folder_ingest()

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

    # ---- ingest folder list: survives a real YAML round trip ---------------
    # Home's one-click batch walks this list IN ORDER, so a round trip that
    # reorders it, de-dupes it or mangles a UNC path silently changes what
    # gets processed. Written to a temp file, not the user's config.
    import tempfile as _tempfile
    from laser_trim_analyzer.config import Config as _Config, missing_ingest_folders
    with _tempfile.TemporaryDirectory() as _td:
        _cfgp = Path(_td) / "config.yaml"
        _c = _Config()
        _c.database.path = Path(_td) / "unused.db"
        # The offline entry is a path under this temp dir that is never
        # created — unreachable on every platform, unlike a real UNC share
        # which may genuinely exist on the work machine.
        _offline = str(Path(_td) / "offline_share")
        _wanted = ["\\\\192.168.66.9\\Public\\LaserTrim", str(REPO / "Work Files"),
                   _offline]
        for _f in _wanted:
            _c.ingest.add(_f)
        _c.ingest.add(_wanted[0] + "\\")          # duplicate: must not land
        _c.save(_cfgp)
        _back = _Config.load(_cfgp).ingest.folders
        check("ingest folders: config round-trip preserves the exact order",
              _back == _wanted, f"{_back}")
        _bad = dict(missing_ingest_folders(_back))
        check("ingest folders: an unreachable folder is reported with a reason",
              str(REPO / "Work Files") not in _bad and bool(_bad.get(_offline)),
              f"{len(_bad)} unreachable of {len(_back)}: {_bad.get(_offline)}")

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
        # The post-batch order lives in core/ingest_run.py since 2026-08-31
        # (one pipeline behind both the Process page and Home).
        src = open(REPO / "src/laser_trim_analyzer/core/ingest_run.py",
                   encoding="utf-8").read()
        # Compare against the advance CALL SITE, not the first mention — an
        # older comment names advance_drift_state above the rematch block.
        check("matcher: post-batch rematch wired BEFORE drift advance",
              0 < src.find("db.rematch_unlinked_final_tests")
              < src.find("advance_drift_state(db, model="))
        # Domain invariant (James, 2026-07-13): trim ALWAYS precedes final
        # test. No linked pair anywhere in the real DB may have the trim
        # dated after the FT record (matcher date preference: file_date,
        # else test_date).
        #
        # Compared by calendar DATE, not raw timestamp (2026-08-30): trim rows
        # now carry the clock time that orders same-day re-trim attempts, while
        # final-test rows are still stored at midnight. A trim at 14:30 and its
        # same-day FT are in the correct order — 6,713 real pairs are same-day —
        # and a raw timestamp compare would read every one of them as reversed.
        n_rev = raw.execute(
            "SELECT COUNT(*) FROM final_test_results f "
            "JOIN analysis_results a ON a.id = f.linked_trim_id "
            "WHERE date(a.file_date) > date(COALESCE(f.file_date, f.test_date))"
        ).fetchone()[0]
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

    # ---- data quality: corrupt linearity_spec limit columns (2026-08-30) ----
    # Model 8888 stored a 63.03 V "spec" on 13 tracks. Not a parser bug: the
    # source workbooks really hold 0.03 x23 then 1.03, 2.03 ... 148.03 — an
    # Excel fill-handle "Fill Series" artifact — and 63.03 is the honest
    # median of that. A 63 V limit passes every unit, so those tracks' verdicts
    # were meaningless. ExcelParser._validate_limit_columns now rejects such a
    # band at ingest and linearity is recorded as indeterminate instead.
    #
    # This check re-runs that guard over every stored limit column. It reports
    # counts rather than a bare boolean on purpose: a scan that silently drops
    # to zero rows examined would otherwise read as green.
    try:
        import json as _json
        from laser_trim_analyzer.core.parser import ExcelParser
        _p = ExcelParser()
        scanned = 0
        offenders: dict = {}
        for model, up, lo in raw.execute(
                "SELECT a.model, t.upper_limits, t.lower_limits "
                "FROM track_results t JOIN analysis_results a ON a.id = t.analysis_id "
                "WHERE t.upper_limits IS NOT NULL AND t.lower_limits IS NOT NULL"):
            try:
                U = _json.loads(up) or []
                L = _json.loads(lo) or []
            except (TypeError, ValueError):
                continue
            if not U or not L:
                continue
            scanned += 1
            reason = _p._validate_limit_columns(U, L, _p._calculate_linearity_spec(U, L))
            if reason:
                offenders.setdefault(model, [0, reason])
                offenders[model][0] += 1
        # Guard against a silently-empty scan (weak-assertion trap).
        check("data quality: limit-column scan actually examined rows",
              scanned > 100_000, f"{scanned} tracks with limit data scanned")
        n_bad = sum(v[0] for v in offenders.values())
        if n_bad:
            detail = "; ".join(f"{m}={v[0]}" for m, v in sorted(offenders.items()))
            warn(f"data quality: {n_bad} track(s) with an unusable linearity_spec "
                 f"limit column across {len(offenders)} model(s)",
                 f"{detail} | linearity NOT graded on these; fix the source workbook")
        else:
            check("data quality: every stored limit column is a usable spec band",
                  True, f"0 of {scanned} tracks flagged")
    except Exception as e:
        check("data quality: linearity_spec limit-column guard", False,
              f"{type(e).__name__}: {e}")

    # ---- data quality: linearity error MAGNITUDE coverage (2026-08-31) -----
    # Model 8232-1 recorded linearity_pass, fail_points, spec and offset on
    # 3,108 tracks since 2023 and the error magnitude on NONE of them; 8770
    # lost 206 more. Cause: analyzer took the magnitude with
    # `max(abs(e) for e in errors)`, and Python's max() returns NaN when the
    # FIRST element is NaN — these files open with a six-point unmeasured
    # lead-in. The NaN was then coerced to None and stored as NULL. Nothing
    # looked wrong because the disposition columns were all intact.
    #
    # That asymmetry IS the signature, so this check looks for exactly it:
    # a model that grades its tracks but cannot say by how much. Anything
    # that can be graded has a magnitude; if a model's magnitude coverage
    # collapses while its pass/fail coverage stays complete, the number is
    # being dropped somewhere and the zero-tolerance metric is blind.
    try:
        rows = raw.execute(
            "SELECT a.model, COUNT(*) n,"
            "       SUM(t.linearity_pass IS NOT NULL) graded,"
            "       SUM(t.final_linearity_error_shifted IS NOT NULL) mag "
            "FROM track_results t JOIN analysis_results a ON a.id = t.analysis_id "
            "WHERE t.status != 'UNTRIMMED' AND a.file_date >= '2023-01-01' "
            "GROUP BY a.model HAVING n >= 50").fetchall()
        # Weak-assertion trap: an empty or tiny result set must not read green.
        check("linearity magnitude: coverage scan examined real models",
              len(rows) >= 20 and sum(r[1] for r in rows) > 10_000,
              f"{len(rows)} models, {sum(r[1] for r in rows)} graded tracks")
        blind = []
        for model, n, graded, mag in rows:
            # "Grades but cannot measure": >=90% dispositioned, <10% measured.
            if graded >= 0.90 * n and mag < 0.10 * n:
                blind.append(f"{model}: {graded}/{n} graded but only {mag} magnitudes")
        check("linearity magnitude: no model grades tracks it cannot measure",
              not blind,
              "; ".join(blind) if blind
              else f"{len(rows)} models all carry magnitudes where they carry verdicts")
        # A softer companion: a real regression usually shows as a partial
        # slide before it becomes a collapse, so surface those too.
        thin = [f"{m}: {mag}/{n}" for m, n, graded, mag in rows
                if graded >= 0.90 * n and 0.10 * n <= mag < 0.70 * n]
        if thin:
            warn(f"linearity magnitude: {len(thin)} model(s) measure under 70% "
                 f"of the tracks they grade", "; ".join(thin))
        else:
            check("linearity magnitude: no model sits in the thin-coverage band",
                  True, f"0 of {len(rows)} models between 10% and 70%")
    except Exception as e:
        check("linearity magnitude: coverage guard", False,
              f"{type(e).__name__}: {e}")

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
STANDALONE = {"ft-fastpath": check_ft_incremental_fastpath,
              "ingest": check_multi_folder_ingest}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--only":
        name = sys.argv[2]
        if name not in STANDALONE:
            raise SystemExit(f"unknown section {name!r}; have: {', '.join(STANDALONE)}")
        STANDALONE[name]()
        raise SystemExit(_tally())
    raise SystemExit(main())
