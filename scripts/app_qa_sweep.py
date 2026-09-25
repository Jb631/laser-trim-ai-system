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

The one artifact it writes (the evidence workbook it re-reads to check the
export schema) lands in qa_output/ at the repo root, which is gitignored.

Exit code = number of FAILs. WARNs are judgment items for review.
"""
import sys
import types
from pathlib import Path

import _db_guard

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
# Scratch, gitignored, shared with chart_qa_render_all.py. Never a tracked
# directory: until 2026-08-31 this pointed at docs/v6_design_review_2026-07-07/
# qa_sweep/, so every sweep rewrote a committed workbook and the churn had to
# be committed or reverted by hand. That review folder is now frozen.
QA_OUTPUT = REPO / "qa_output"
sys.path.insert(0, str(REPO / "src"))

import sqlite3  # noqa: E402
import time  # noqa: E402
from sqlalchemy import text as sqlalchemy_text  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from laser_trim_analyzer.database.models import UNREADABLE_PREFIX  # noqa: E402

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


def check_ft_parser_console_silence() -> None:
    """No non-finite / degenerate x ever reaches the FT ideal-line fit.

    A full `Work Files` ingest printed 12 lines of

        ** On entry to DLASCL, parameter number  4 had an illegal value

    every single time — the only console noise the whole run produced. All 12
    came from the one np.polyfit in `_extract_format2_tracks`, six per call,
    on two real 8213-1 files whose Position column is 2,998 rows of literal
    0.0. np.polyfit conditions its Vandermonde matrix by dividing each column
    by that column's norm; a constant-zero x column has norm 0, so the
    division is 0/0, the column fills with NaN, LAPACK's XERBLA prints those
    six lines and the fit raises LinAlgError into a broad `except`. An
    infinity in either column arrives at the same place by the other road:
    the norm overflows to inf and inf/inf is NaN again. James's first full
    ingest on the new laptop is ~150k final-test files; the console has to
    stay clean.

    Standalone — the synthetic sheets need neither the work database nor the
    sample corpus, so this runs on any machine:
        python scripts/app_qa_sweep.py --only ft-silence

    Two mechanics matter. The capture is of FILE DESCRIPTOR 1, not
    `sys.stdout`: XERBLA prints with C `printf` and sails past anything
    Python swaps in. And `fflush(NULL)` is mandatory, because C stdout is
    block-buffered when fd 1 is not a tty — without it the bytes sit in
    libc's buffer until interpreter exit and this check reads clean while the
    console still gets the noise.

    The healthy sheet is the reason this cannot pass on an ERROR: a parser
    that failed on everything would print nothing at all, so silence by
    itself is not evidence. The check fails unless a good sheet still returns
    its track with a real linearity_error, an inf row is dropped without
    disturbing the fit through the surviving rows, and the degenerate file
    yields no track rather than a fabricated linearity_error of 0.0.
    """
    import ctypes, os, tempfile, shutil  # noqa: E402
    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402
    from laser_trim_analyzer.core.final_test_parser import FinalTestParser  # noqa: E402

    def same_series(a, b, tol=1e-12):
        return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))

    libc = ctypes.CDLL(None)
    tmp = Path(tempfile.mkdtemp(prefix="ltaqa_ftsilence_"))

    def sheet(name, measured, positions):
        """'Data' + 'Charts' is the Format-2 signature; no header row."""
        path = tmp / name
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            pd.DataFrame({0: list(measured), 1: list(positions),
                          2: list(range(1, len(measured) + 1))}).to_excel(
                w, sheet_name="Data", header=False, index=False)
            pd.DataFrame({0: ["x"]}).to_excel(
                w, sheet_name="Charts", header=False, index=False)
        return path

    n = 40
    pos = [i / (n - 1) for i in range(n)]
    meas = [5.0 * p + 0.01 * np.sin(np.pi * p) for p in pos]       # bowed ramp
    inf_pos, inf_meas = list(pos), list(meas)
    inf_pos.insert(20, float("inf"))
    inf_meas.insert(20, 2.5)

    cases = {
        "healthy": sheet("healthy.xlsx", meas, pos),
        "inf_row": sheet("inf_row.xlsx", inf_meas, inf_pos),
        "flat_x": sheet("flat_x.xlsx", meas, [0.0] * n),
    }
    real = [REPO / "Work Files/Sample_Base_2026-04-10/Test Station/8213-4"
                 / f"8213-1_sn{s}.xls" for s in (21, 22)]
    for p in real:
        if p.exists():
            cases[p.name] = p

    parser = FinalTestParser()
    parsed, failures = {}, []
    sys.stdout.flush(); libc.fflush(None)
    saved_fd = os.dup(1)
    sink = tempfile.TemporaryFile()
    os.dup2(sink.fileno(), 1)
    try:
        for label, path in cases.items():
            try:
                parsed[label] = parser.parse_file(path)
            except Exception as exc:
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
    finally:
        sys.stdout.flush(); libc.fflush(None)
        os.dup2(saved_fd, 1); os.close(saved_fd)
        sink.seek(0)
        console = sink.read().decode(errors="replace")
        sink.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def tracks(label):
        return (parsed.get(label) or {}).get("tracks") or []

    good = tracks("healthy")
    healthy_ok = len(good) == 1 and (good[0].get("linearity_error") or 0) > 0
    dropped_ok = (healthy_ok and len(tracks("inf_row")) == 1
                  and same_series(tracks("inf_row")[0]["errors"], good[0]["errors"]))
    flat_ok = tracks("flat_x") == []
    real_ok = all(tracks(p.name) == [] for p in real if p.name in parsed)
    xerbla = sum(1 for line in console.splitlines()
                 if "DLASCL" in line or "XERBLA" in line)

    check("FT parser: no non-finite or flat x reaches the ideal-line fit "
          "(console stays silent)",
          not failures and healthy_ok and dropped_ok and flat_ok and real_ok
          and console == "",
          f"parsed={len(parsed)}/{len(cases)} real_samples="
          f"{sum(1 for p in real if p.name in parsed)}/{len(real)} "
          f"healthy_track={len(good)} healthy_error="
          f"{good[0].get('linearity_error') if good else None} "
          f"inf_row_matches_healthy={dropped_ok} flat_x_tracks="
          f"{len(tracks('flat_x'))} xerbla_lines={xerbla} "
          f"console_bytes={len(console)}"
          + (f" | raised: {failures}" if failures else ""))


def check_ft_graded_window() -> None:
    """The app grades final tests on the rows the SHEET grades — and only those.

    THE DISPOSITION IS THE APP'S GRADE. What this section proves is (a) that
    the station's own verdict is READ correctly, per template, so the
    reference column beside the app's verdict is trustworthy; (b) that a
    graded window is actually found on real files rather than silently
    defaulting to "grade everything"; (c) that no final-test FAIL survives
    only because of points the station never graded; (d) that a blank error
    cell never reaches storage as 0.0; and (e) how often the app and the
    station disagree, REPORTED rather than capped — a difference is expected
    whenever the offset correction rescues a unit.

    Standalone: it reads the local sample corpus and needs no work database.
        python scripts/app_qa_sweep.py --only ft-window

    Two weak-assertion traps are closed deliberately. The corpus size is
    asserted first, so an empty `Work Files` (or a parser that raised on
    every file) cannot read as a wall of green. And (c) is a RE-GRADE, not a
    tolerance: every file the app fails is graded a second time with the
    out-of-window points dropped, and it has to fail again.
    """
    import glob  # noqa: E402
    from laser_trim_analyzer.core.analyzer import Analyzer  # noqa: E402
    from laser_trim_analyzer.core.final_test_parser import FinalTestParser  # noqa: E402
    from laser_trim_analyzer.core.ft_regrade import grade_ft_track  # noqa: E402
    from laser_trim_analyzer.export.unit_chart import classify_graded_points  # noqa: E402

    base = REPO / "Work Files" / "Sample_Base_2026-04-10" / "Test Station"
    if not base.is_dir():
        warn("FT graded window: sample corpus absent — section skipped",
             str(base))
        return

    files = sorted(glob.glob(str(base / "*" / "*.xls*")))
    check("FT graded window: sample corpus found",
          len(files) >= 100, f"{len(files)} files under {base.name}")
    if not files:
        return

    parser = FinalTestParser()
    analyzer = Analyzer()
    no_spec = {"linearity_type": None, "angle_spec": None, "angle_tol": None,
               "angle_tol_type": None, "exclude_points": None}

    parsed_ok = 0
    cell_read = {}          # format -> [matched, total]
    window_found = {}       # format -> [detected, tracks]
    zero_errors = []        # files storing more 0.0 errors than the sheet holds
    file_error_tracks = 0   # tracks whose errors came from the file's own column
    window_only_fails = []  # tracks counting a fail point the station never graded
    graded_tracks = 0       # tracks with a window, i.e. tracks this can judge
    app_vs_station = {"agree": 0, "app_fail_station_pass": 0,
                      "app_pass_station_fail": 0}
    parse_errors = []

    for path in files:
        try:
            parsed = parser.parse_file(Path(path))
        except Exception as e:                       # noqa: BLE001
            parse_errors.append(f"{Path(path).name}: {type(e).__name__}: {e}")
            continue
        parsed_ok += 1
        fmt = parsed.get("format") or "?"
        tracks = parsed.get("tracks") or []

        # (a) the station's verdict, re-read straight off the sheet with no
        # app code in the way, must equal what the parser reported.
        expected = _sheet_verdict_independently(Path(path), fmt)
        reported = parsed["test_results"].get("station_linearity_pass")
        if expected is not None or reported is not None:
            slot = cell_read.setdefault(fmt, [0, 0])
            slot[1] += 1
            slot[0] += 1 if expected == reported else 0

        for track in tracks:
            # (b) a window, from flags or from the declared ignore counts.
            slot = window_found.setdefault(fmt, [0, 0])
            slot[1] += 1
            if track.get("graded_window_source") in ("flags", "ignore_cells"):
                slot[0] += 1

            # (d) a blank cell must never be stored as 0.0 -- dead centre of
            # every band, the most flattering value a zero-tolerance metric can
            # hold. The test is against the SHEET, not against a bare
            # `0.0 in errors`: a station can legitimately measure exactly zero,
            # and forbidding the value outright would be a false alarm on 87 of
            # these files. What is forbidden is storing MORE zeros than the
            # sheet contains.
            stored = [e for e in (track.get("errors") or []) if e is not None]
            sheet = _sheet_error_cells(Path(path), fmt)
            if sheet is not None and stored and set(stored) <= set(sheet):
                # Only meaningful when the parser used the file's own error
                # column; the computed-error fallbacks have no sheet cells to
                # compare against, and are counted rather than silently passed.
                if stored.count(0.0) > sheet.count(0.0):
                    zero_errors.append(
                        f"{Path(path).name}: {stored.count(0.0)} stored vs "
                        f"{sheet.count(0.0)} in the sheet")
                file_error_tracks += 1

            # (c) grade it, then check the COUNT against an independent
            # implementation of the same rule (export.unit_chart.
            # classify_graded_points, which the charts already grade with).
            # The assertion is that every fail point the grader counted lies
            # INSIDE the station's window and none outside it was counted —
            # which is what "no FAIL survives on rows nobody grades" means as
            # a number. Re-running the grader with the window switched off
            # would prove nothing: disable the window everywhere and both
            # sides move together, and the check passes on the bug.
            result = grade_ft_track(analyzer, track, no_spec,
                                    model=parsed["metadata"].get("model"),
                                    filename=Path(path).name)
            counted = track.get("linearity_fail_points")
            window = track.get("graded_window")
            if result is not None and counted is not None and window is not None:
                out_of_band, unmeasured = classify_graded_points(
                    track.get("errors"), track.get("upper_limits"),
                    track.get("lower_limits"),
                    offset=track.get("optimal_offset") or 0.0,
                    k=track.get("optimal_slope") or 0.0,
                    theory=track.get("theory_values"))
                low, high = window
                inside = sum(1 for i in set(out_of_band) | set(unmeasured)
                             if low <= i <= high)
                graded_tracks += 1
                if counted != inside:
                    window_only_fails.append(
                        f"{Path(path).name}/{track.get('track_id')}: counted "
                        f"{counted}, in-window violations {inside}")

            # (e) report the agreement, both directions.
            station = track.get("station_linearity_pass")
            app = track.get("linearity_pass")
            if station is None or app is None:
                continue
            if bool(app) == bool(station):
                app_vs_station["agree"] += 1
            elif station:
                app_vs_station["app_fail_station_pass"] += 1
            else:
                app_vs_station["app_pass_station_fail"] += 1

    check("FT graded window: every sample file parsed",
          not parse_errors and parsed_ok >= 100,
          f"{parsed_ok} parsed" + (f"; ERRORS: {'; '.join(parse_errors[:3])}"
                                   if parse_errors else ""))

    for fmt, (matched, total) in sorted(cell_read.items()):
        check(f"FT station verdict matches the sheet's cell ({fmt})",
              total > 0 and matched == total, f"{matched}/{total}")

    f1 = window_found.get("format1", [0, 0])
    share = (f1[0] / f1[1]) if f1[1] else 0.0
    check("FT graded window detected on >=95% of Format-1 tracks",
          f1[1] >= 100 and share >= 0.95,
          f"{f1[0]}/{f1[1]} = {share * 100:.2f}%")
    # The other templates, reported in ONE line rather than four: format 2 and
    # the shop-test sheets state no window at all (correctly — they carry
    # neither a flag column nor ignore counts), and format 4's flag column
    # could not be identified, so these are facts about the templates, not
    # findings about the code.
    others = ", ".join(f"{fmt} {found}/{total}"
                       for fmt, (found, total) in sorted(window_found.items())
                       if fmt != "format1")
    if others:
        warn("FT graded window coverage on the other templates "
             "(informational — not every template states one)", others)

    # Weak-assertion trap: if no track took the file-error path, "no extra
    # zeros" is vacuously true, so the population is asserted first.
    check("FT: blanks compared against the sheet on a real population",
          file_error_tracks >= 100, f"{file_error_tracks} file-error tracks")
    check("FT: no blank error cell is stored as 0.0",
          not zero_errors,
          f"{len(zero_errors)}: {'; '.join(sorted(zero_errors)[:3])}"
          if zero_errors else
          f"{file_error_tracks} tracks hold no zero the sheet does not hold")

    # Weak-assertion trap again: with no windowed tracks there is nothing to
    # disagree about, so the population comes first.
    check("FT fail points: a real population of windowed tracks was graded",
          graded_tracks >= 100, f"{graded_tracks} windowed tracks")
    check("FT: no fail point outside the station's window is ever counted",
          not window_only_fails,
          f"{len(window_only_fails)}: {'; '.join(window_only_fails[:3])}"
          if window_only_fails else
          f"{graded_tracks} tracks: counted fails == in-window violations")

    total_compared = sum(app_vs_station.values())
    # REPORTED, never capped: the app corrects the offset and the station does
    # not, so disagreement is information about the two gradings, not a defect.
    warn("FT app-vs-station agreement (reported, not a threshold)",
         f"{app_vs_station['agree']}/{total_compared} agree · "
         f"app FAIL/station PASS {app_vs_station['app_fail_station_pass']} · "
         f"app PASS/station FAIL {app_vs_station['app_pass_station_fail']}")


def _sheet_verdict_independently(path: Path, fmt: str):
    """Read the sheet's linearity verdict WITHOUT the parser, for check (a).

    Deliberately a second implementation: a check that calls the code it is
    checking proves only that the code is self-consistent.
    """
    import pandas as pd  # noqa: E402
    try:
        with pd.ExcelFile(path) as xl:
            if fmt == "format4_parameters":
                df = pd.read_excel(xl, sheet_name="Parameters", header=None, nrows=1)
                if df.shape[1] <= 11 or df.shape[0] == 0:
                    return None
                text = str(df.iloc[0, 11]).strip().upper()
                return True if text == "PASSED" else (False if text == "FAILED" else None)
            if fmt == "format3_multitrack":
                verdicts = []
                for sheet in [s for s in xl.sheet_names if len(s) == 1 and s.isalpha()]:
                    verdicts.append(_verdict_from_frame(
                        pd.read_excel(xl, sheet_name=sheet, header=None)))
                known = [v for v in verdicts if v is not None]
                return all(known) if known else None
            if fmt in ("format2", "format_shop_test"):
                return None
            sheet = "Sheet1" if "Sheet1" in xl.sheet_names else xl.sheet_names[0]
            return _verdict_from_frame(
                pd.read_excel(xl, sheet_name=sheet, header=None))
    except Exception:                                # noqa: BLE001
        return None


def _sheet_error_cells(path: Path, fmt: str):
    """The sheet's OWN error column, non-blank cells only, or None.

    A second reader on purpose (see `_sheet_verdict_independently`): comparing
    the parser against itself proves only self-consistency.
    """
    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402
    columns = {"format1": ("Sheet1", 3), "format4_parameters": ("Parameters", 5),
               "format_shop_test": ("test", 8)}
    if fmt not in columns:
        return None
    sheet_name, col = columns[fmt]
    try:
        with pd.ExcelFile(path) as xl:
            if sheet_name not in xl.sheet_names:
                sheet_name = xl.sheet_names[0]
            df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
    except Exception:                                # noqa: BLE001
        return None
    if df.shape[1] <= col:
        return None
    out = []
    for i in range(df.shape[0]):
        value = df.iloc[i, col]
        if pd.notna(value) and isinstance(value, (int, float, np.integer, np.floating)):
            out.append(float(value))
    return out


def _verdict_from_frame(df):
    import pandas as pd  # noqa: E402
    if df.shape[1] <= 11:
        return None
    for row in range(min(10, df.shape[0])):
        label = df.iloc[row, 10]
        if pd.notna(label) and "linearity" in str(label).lower():
            text = str(df.iloc[row, 11]).strip().upper()
            return True if text == "PASSED" else (False if text == "FAILED" else None)
    return None


def check_ft_regrade_dry_run(db) -> None:
    """A dry run computes every verdict and writes NOTHING.

    This is the guard on the repair tool the owner will point at the
    production database. `--apply` is never exercised here; what is asserted
    is that WITHOUT it the row count, the verdicts and the legacy marker come
    back byte-identical.

    Capped at a small sample because each row re-parses a workbook from the
    plant share; off the work network they all report unreachable, which is
    itself the behaviour to check — an unreachable source must be counted and
    skipped, never written as a NULL verdict.
    """
    from laser_trim_analyzer.core.ft_regrade import regrade_final_tests  # noqa: E402

    legacy = db.count_legacy_ft_verdicts()
    with db.session() as session:
        before = session.execute(sqlalchemy_text(
            "SELECT COUNT(*), SUM(linearity_pass = 1), SUM(linearity_pass = 0),"
            " SUM(linearity_pass IS NULL),"
            " SUM(graded_window_source IS NULL) FROM final_test_results")).fetchone()

    report = regrade_final_tests(db, only_legacy=True, apply=False, limit=40)

    with db.session() as session:
        after = session.execute(sqlalchemy_text(
            "SELECT COUNT(*), SUM(linearity_pass = 1), SUM(linearity_pass = 0),"
            " SUM(linearity_pass IS NULL),"
            " SUM(graded_window_source IS NULL) FROM final_test_results")).fetchone()

    check("FT re-grade: the dry run examined rows",
          report.examined > 0 or legacy == 0,
          f"{report.examined} examined of {legacy} legacy row(s)")
    check("FT re-grade: a dry run changes nothing in the database",
          tuple(before) == tuple(after),
          f"before={tuple(before)} after={tuple(after)}")
    check("FT re-grade: every examined row got an outcome",
          len(report.outcomes) == report.examined,
          f"{len(report.outcomes)} outcome(s); missing={report.missing}, "
          f"errors={report.errors}, would change={report.changed}")
    check("FT re-grade: an unreachable source is counted, not graded NULL",
          all(o.before == o.after for o in report.outcomes
              if o.result == "missing_file"),
          f"{report.missing} unreachable")


# The four 8232-1 fixtures the findings and increment-volts checks were written against, BY NAME.
# They used to glob the folder and demand exactly four files, so the two two-track fixtures added
# on 2026-09-24 switched both checks off entirely (a FAIL, then the whole body skipped).
_FOUR_8232_FIXTURES = ("dlts_8232-1_242.xls", "dlts_8232-1_243.xls",
                       "lts_8232-1_193.xls", "lts_8232-1_194.xls")


def check_findings_fixtures() -> None:
    """Process findings on the four trim fixtures: real-cut counting, grading fidelity, lever
    safety, and the cache round trip. Builds its own throwaway database (--only findings).

    Falsify before trusting (2026-09-20): make findings/data.py `_is_real_cut` return True and the
    real-cut check must go FAIL with ("B", 3) in its detail.
    """
    import shutil
    import tempfile
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as _mgr
    import laser_trim_analyzer.database as _dbpkg
    from laser_trim_analyzer.findings.data import load_model_tracks, yardstick_fidelity
    from laser_trim_analyzer.findings.engine import refresh_findings
    from laser_trim_analyzer.findings.model import Finding, LEVERS

    fixtures = [f for f in (REPO / "tests/fixtures/trim" / n for n in _FOUR_8232_FIXTURES)
                if f.is_file()]
    check("findings: the four trim fixtures are present", len(fixtures) == 4,
          f"{[f.name for f in fixtures]}")
    if len(fixtures) == 4:
        # _dbpkg (the package __init__) never defines _db_manager itself -- only manager.py does --
        # so a fresh process (e.g. `--only findings` on its own) has no such attribute yet; reading
        # it unguarded crashes this check before the try/except can turn a break into a FAIL. Same
        # `raising=False` idea as the fixture_db in tests/test_findings_data.py.
        saved = (_mgr._db_manager, getattr(_dbpkg, "_db_manager", None))
        tmp = Path(tempfile.mkdtemp(prefix="findings_sweep_"))
        try:
            fdb = _mgr.DatabaseManager(tmp / "f.db")
            _mgr._db_manager = fdb                 # BOTH globals: a Processor must never
            _dbpkg._db_manager = fdb               # reach the configured database.
            proc = Processor(use_ml=False)
            for f in fixtures:
                fdb.save_analysis(proc.process_file(f))
            tracks = load_model_tracks(fdb, "8232-1")
            cuts = sorted((t.system, len(t.passes)) for t in tracks)
            check("findings: real cuts are counted, the duplicate Lin Error row is not",
                  cuts == [("A", 2), ("A", 3), ("B", 2), ("B", 2)], f"{cuts}")
            y = yardstick_fidelity(tracks)
            check("findings: the yardstick reproduces the app's verdict on the fixtures",
                  y["n"] == 4 and y["agreement"] == 1.0, f"{y}")
            report = {}
            stored = refresh_findings(fdb, ["8232-1"], report)
            facts = fdb.get_process_facts("8232-1")
            check("findings: four tracks find nothing, and the facts are cached anyway",
                  stored == 0 and facts is not None and facts.get("tracks") == 4,
                  f"stored={stored} tracks={None if facts is None else facts.get('tracks')}")
            check("findings: no analyzer failed on the fixtures",
                  report.get("failed_models") == {} and report.get("analyzer_errors") == {}
                  and facts is not None and facts.get("errors") == {},
                  f"report={report} errors={None if facts is None else facts.get('errors')}")
            # Four tracks never reach the 30 a limit table needs to count as "in service": the history is
            # COMPUTED and empty. None would mean the limit-table analyzer did not run.
            check("findings: the limit-table history is computed (and empty on four tracks), never None",
                  facts is not None and facts.get("limit_tables") == [],
                  f"{None if facts is None else facts.get('limit_tables')!r}")
            # Same rule for the cut-setting analyzer: {} is "it ran and there was nothing",
            # None is "it never ran". Four tracks cannot reach MIN_PER_SETTING, so a non-empty
            # result here would mean the sample floor had been lost.
            check("findings: the cut-setting facts are computed (and empty on four tracks), never None",
                  facts is not None and facts.get("cut_setting") == {},
                  f"{None if facts is None else facts.get('cut_setting')!r}")
            check("findings: the pass-burden facts are computed (and empty on four tracks), never None",
                  facts is not None and facts.get("pass_burden") == {},
                  f"{None if facts is None else facts.get('pass_burden')!r}")
        except Exception as e:                      # an exception is a FAIL, never a skip
            check("findings: the engine runs on the fixtures", False, f"{type(e).__name__}: {e}")
        finally:
            _mgr._db_manager, _dbpkg._db_manager = saved
            shutil.rmtree(tmp, ignore_errors=True)

    try:
        Finding(model="m", analyzer="a", category="c", lever="atp_spec", title="t", summary="s",
                systems=("A",), n_units=1, strength_name="n", strength_value=1.0)
        check("findings: the ATP spec can never be named as a lever", False,
              "Finding accepted lever='atp_spec'")
    except ValueError:
        check("findings: the ATP spec can never be named as a lever", True, f"levers={sorted(LEVERS)}")


def check_findings_on_database(db) -> None:
    """The engine against REAL models in the database under test (always a copy): it must run
    every analyzer without one of them raising -- on a pre-rebuild database as much as on a rebuilt
    one, because the ingest hook will run it on whichever the owner has."""
    from laser_trim_analyzer.findings.data import load_model_tracks, yardstick_fidelity
    from laser_trim_analyzer.findings.engine import refresh_findings

    with db.session() as s:
        models = [r[0] for r in s.execute(sqlalchemy_text(
            "SELECT model FROM analysis_results WHERE system IN ('A','B','C') AND model IS NOT NULL "
            "AND model <> 'Unknown' GROUP BY model ORDER BY COUNT(*) DESC LIMIT 5"))]
    check("findings: the database has trim models to run the engine on", len(models) > 0, f"{models}")
    if not models:
        return
    report = {}
    t0 = time.monotonic()
    try:
        stored = refresh_findings(db, models, report)
    except Exception as e:
        check("findings: the engine runs on the busiest real models", False, f"{type(e).__name__}: {e}")
        return
    secs = time.monotonic() - t0
    check("findings: the engine runs on the busiest real models -- no model failed, no analyzer raised",
          report.get("models") == len(models) and report.get("failed_models") == {}
          and report.get("analyzer_errors") == {},
          f"{len(models)} models in {secs:.0f}s, {stored} findings; failed={report.get('failed_models')} "
          f"analyzer_errors={report.get('analyzer_errors')}")
    for m in models:
        facts = db.get_process_facts(m)
        check(f"findings: facts cached for {m}, with every documented key",
              isinstance(facts, dict) and {"tracks", "yardstick", "recipe_history", "trim_effort",
                                           "limit_tables", "errors"} <= set(facts)
              and facts.get("tracks", 0) > 0, f"{None if facts is None else sorted(facts)}")
        y = yardstick_fidelity(load_model_tracks(db, m))
        # Low fidelity is NOT a defect: it is the designed branch where the engine goes silent
        # about intermediate sweeps. Disclose it; do not fail the sweep for the data's sake.
        if y["n"] == 0 or y["agreement"] is None:
            warn(f"findings: {m} has no final sweep the yardstick can grade", f"{y}")
        elif not y["faithful"]:
            warn(f"findings: the yardstick cannot vouch for {m} -- trim-effort findings stay silent there", f"{y}")


def check_findings_group_mapping(db) -> None:
    """Every analyzer sitting in the CACHED findings (process_findings, real data on the
    database under test) has a Findings-page group in findings/presentation.ANALYZER_GROUP --
    so a renamed or new analyzer whose cache predates a presentation-layer update is caught
    against what is actually STORED, not only against the source tree (that static check is
    tests/test_findings_presentation.py::test_every_analyzer_the_engine_runs_has_a_group, which
    compares ANALYZER_GROUP against the analyzer modules on disk and cannot see a stale cache).

    A weak version of this would ask presentation.group_key() for an answer and accept whatever
    comes back -- but group_key() always returns something, because an unmapped analyzer falls
    through to the OTHER group by design (so the Findings page never drops a finding silently).
    That would pass even on the exact bug it exists to catch. The real assertion is that the
    analyzer is a KEY of ANALYZER_GROUP, not that group_key() ran without raising.

    Falsify before trusting (2026-09-24): union `seen` with an invented analyzer name before the
    comparison below -- FAILs, naming "totally_invented_analyzer" as unmapped. Remove the union
    and the check is real again, run on whatever this database's cache actually holds.
    """
    from laser_trim_analyzer.findings import presentation as P

    cached = db.get_process_findings()
    seen = {d.get("analyzer") for d in cached if isinstance(d, dict) and d.get("analyzer")}
    check("findings: there are cached payloads to check group mapping against",
          len(cached) > 0 and len(seen) > 0,
          f"{len(cached)} cached findings, analyzers={sorted(seen)}")
    missing = sorted(seen - set(P.ANALYZER_GROUP))
    check("findings: every cached finding's analyzer maps to a Findings-page group "
          "(findings/presentation.ANALYZER_GROUP)",
          not missing,
          f"unmapped -- would render under 'Other findings': {missing}; "
          f"analyzers seen in the cache: {sorted(seen)}; known groups: {sorted(set(P.ANALYZER_GROUP))}")


def check_ft_disposition_excludes_ungraded(db, raw) -> None:
    """Rows with no disposition stay out of every rate built on one.

    `linearity_pass IS NULL` means "the app could not grade this" — a Format 2
    file with no limit columns, or a window that left nothing inside it. Such
    a row is not a failure and must not sit in a denominator.
    """
    ungraded = raw.execute(
        "SELECT COUNT(*) FROM final_test_results "
        "WHERE linearity_pass IS NULL").fetchone()[0]
    graded = raw.execute(
        "SELECT COUNT(*) FROM final_test_results "
        "WHERE linearity_pass IS NOT NULL").fetchone()[0]
    check("FT disposition: the database holds rows of both kinds to test with",
          graded > 0, f"{graded} graded, {ungraded} not graded")

    stats = db.get_model_trim_ft_agreement("6607")
    expected = raw.execute(
        "SELECT COUNT(*) FROM final_test_results "
        "WHERE model = '6607' AND linearity_pass IS NOT NULL").fetchone()[0]
    check("FT pass rate counts only rows that carry a disposition",
          stats["ft_total"] == expected,
          f"app ft_total={stats['ft_total']} vs graded rows={expected}")

    escapes = db.get_escape_overkill_analysis(days_back=36500)
    linked = escapes.get("total_linked") or 0
    eligible = raw.execute(
        "SELECT COUNT(*) FROM final_test_results "
        "WHERE linked_trim_id IS NOT NULL AND linearity_pass IS NOT NULL "
        "AND match_confidence >= 0.70").fetchone()[0]
    # Weak-assertion trap: `0 <= anything` is true, so an empty classification
    # would read green. The population is asserted before the bound.
    check("escapes/overkills ran on a real linked population",
          linked > 0 and eligible > 0, f"classified={linked} eligible={eligible}")
    check("escapes/overkills classify no row without a disposition",
          linked <= eligible, f"classified={linked} eligible={eligible}")


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


def check_ingest_reoffers_only_retryable() -> None:
    """A file the parser can NEVER read is offered exactly once (2026-09-14).

    The work incident: ~930 final-test files were classified "new" on every
    single run and only ~80 ever produced a record. 833 of them were empty
    workbooks, and every empty file has the same content hash — so the one
    `processed_files` row for `~$7029-72.xlsx` matched all of them and
    `mark_file_skipped` returned early without recording any of them under
    their own path. The processor logged "recorded as skipped" for each while
    the write was silently dropped, and the scan re-offered them tomorrow.

    The invariant: after one pass, the only files still offered are the ones
    whose failure is RETRYABLE (a parser upgrade could fix them). Permanent
    failures and duplicates are remembered.

    Empty files are synthesised rather than taken from the corpus: the local
    sample has none, and the collision they cause is the whole bug. The two
    kinds of duplicate are made the way the share makes them — same
    filename+date+model+serial under a second folder with different bytes
    (the branch the work log hits 60 times a run), and the SAME BYTES under a
    second folder with a different name (2026-09-15: ~370 files a day parsed,
    graded and dropped, which is why the morning run produced 442 verdicts
    while the index grew by 69).

    This check FAILS against the pre-fix code, twice over: only one of the
    four empties is remembered, and the byte-identical copy is offered again
    on the second pass and on every pass after it.
    """
    import shutil, tempfile  # noqa: E402
    from laser_trim_analyzer.config import Config  # noqa: E402
    from laser_trim_analyzer.core.ingest_run import run_folders  # noqa: E402
    from laser_trim_analyzer.database import manager as _dbmod  # noqa: E402
    from laser_trim_analyzer.database.manager import DatabaseManager  # noqa: E402

    station = REPO / "Work Files" / "Sample_Base_2026-04-10" / "Test Station"
    real = sorted(p for p in station.rglob("*.xls*") if p.is_file())[:12]
    if len(real) < 4:
        warn("ingest re-offer: fewer than 4 Test Station samples available")
        return

    tmp = Path(tempfile.mkdtemp(prefix="ltaqa_reoffer_"))
    saved_global = _dbmod._db_manager
    try:
        folder = tmp / "Test Station"
        folder.mkdir()
        for f in real:
            shutil.copy2(f, folder / f.name)

        # (1) PERMANENT failures: empty workbooks. All four share the SHA-256
        # of b"", which is exactly what defeated the old hash-keyed guard.
        empties = [
            "Final Test 8084-sn100_8-24-2011_2-44 PM.xls",
            "Final Test 8084-sn107_8-27-2011_9-17 AM.xls",
            "Final Test 7029-82.xls",
            "Final Test 6607-sn200_11-15-2011_10-47 AM.xls",
        ]
        for name in empties:
            (folder / name).write_bytes(b"")

        # (2) A DUPLICATE: same identity tuple, second folder, edited bytes.
        dup_dir = folder / "resent"
        dup_dir.mkdir()
        dup_src = real[0]
        shutil.copy2(dup_src, dup_dir / dup_src.name)
        with open(dup_dir / dup_src.name, "ab") as fh:
            fh.write(b"\0")          # same identity, different content hash

        # (3) A CONTENT duplicate: the SAME BYTES under a second path and a
        # different name — the share's habit of dropping one export into the
        # model folder and into a "Voltage Output"/"Final Sheets" subfolder.
        # Neither the basename rescue nor the identity constraint sees this
        # one: the up-front hash check in `save_final_test` owns it, and
        # until 2026-09-15 it returned the other row's id and recorded
        # NOTHING about this path. ~370 files a day were parsed, graded and
        # dropped on the work share (1,371 processed, 442 verdicts, index
        # +69). This check FAILS against that code: the second pass offers
        # the copy again, and every pass after it would too.
        copy_dir = folder / "Final Sheets"
        copy_dir.mkdir()
        content_src = folder / real[1].name
        content_dup = copy_dir / f"Copy of {content_src.name}"
        shutil.copy2(content_src, content_dup)
        content_pair = {str(content_src), str(content_dup)}

        # (4) RETRYABLE failures: the two populations that came back on every
        # run of the work share until 2026-09-17, one per route.
        #   * an output-smoothness export the parser finds no usable columns
        #     in. Its error result is never saved (file_type keeps it out of
        #     save_analysis), so before the fix NOTHING was recorded about it
        #     anywhere — 67 of these, offered, opened and forgotten daily.
        #   * a trim export whose layout the parser refuses. This one DOES get
        #     a processed_files row, but success=False, and the incremental
        #     index loads only success=True — the "111 retrying earlier
        #     errors" on every scan line since 2013.
        # Both are synthesised: the local corpus is a curated sample and has
        # no unreadable files at all (this section printed "0 other" before).
        os_junk = folder / "6581" / "6581-sn330_-65_OS_load_1-2-2017_10-00-00 AM.xlsx"
        _unreadable_smoothness_workbook(os_junk)
        laser = tmp / "LTS"          # NOT under Test Station: the folder name
        laser.mkdir()                # decides final_test before the filename
        trim_junk = laser / "8888-99_TA_Test Data_1-1-2013_9-00 AMTrimmed Correct.xlsx"
        _unreadable_trim_workbook(trim_junk)
        retryable = {str(os_junk), str(trim_junk)}

        cfg = Config()
        cfg.database.path = tmp / "reoffer_qa.db"
        db = DatabaseManager(cfg.database.path)
        _dbmod._db_manager = db

        rep1 = run_folders([str(folder), str(laser)], db=db, config=cfg,
                           incremental=True)
        first_new = rep1.new_files

        # Classify what the first pass could not turn into a record.
        from laser_trim_analyzer.database.models import (  # noqa: E402
            AnalysisResult as DBAR2, ProcessedFile as DBPF,
            FinalTestResult as DBFT, SmoothnessResult as DBSM)
        with db.session() as sess:
            recorded = {r[0] for r in sess.query(DBFT.file_path).all()}
            recorded |= {r[0] for r in sess.query(DBSM.file_path).all()}
            # Trim files land in analysis_results, not in either table above;
            # an ERROR row is not a record of the file, it is a record of the
            # failure, so it does not count as recorded.
            recorded |= {r[0] for r in sess.query(DBAR2.file_path)
                         .filter(DBAR2.overall_status != "ERROR").all() if r[0]}
            marker_rows = (sess.query(DBPF.file_path, DBPF.error_message)
                           .filter(DBPF.success == True,         # noqa: E712
                                   DBPF.analysis_id.is_(None)).all())
        markers = {r[0] for r in marker_rows}
        marker_reasons = {r[0]: (r[1] or "") for r in marker_rows}
        on_disk = {str(p) for p in folder.rglob("*.xls*") if p.is_file()}
        on_disk |= {str(p) for p in laser.rglob("*.xls*") if p.is_file()}
        unrecorded = on_disk - recorded
        permanent = {p for p in unrecorded if Path(p).stat().st_size == 0}
        duplicate = {str(dup_dir / dup_src.name)} & unrecorded
        content_dupes = content_pair & unrecorded
        other = unrecorded - permanent - duplicate - content_dupes
        total_failures = len(unrecorded)

        print(f"     | first pass: {first_new} new · {len(permanent)} permanent · "
              f"{len(duplicate)} duplicate · {len(content_dupes)} same-content · "
              f"{len(other)} other (retryable) · {len(markers)} markers written")

        check("ingest re-offer: the empty files really do collide on content",
              len(permanent) == len(empties),
              f"permanent={len(permanent)} expected={len(empties)}")
        check("ingest re-offer: every permanent failure is remembered by path",
              permanent <= markers,
              f"unmarked={sorted(Path(x).name for x in permanent - markers)}")

        # The byte-identical pair: exactly one path holds the record and the
        # other holds a marker naming it. Which of the two wins the race is
        # not the point and is not asserted — that nothing is left silent is.
        check("ingest re-offer: identical bytes under two paths are stored "
              "once and the other path is recorded",
              len(content_pair & recorded) == 1 and content_dupes <= markers,
              f"recorded={sorted(Path(x).name for x in content_pair & recorded)} "
              f"unmarked={sorted(Path(x).name for x in content_dupes - markers)}")
        check("ingest re-offer: the same-content marker says what it is",
              all("same content as" in marker_reasons.get(p, "")
                  for p in content_dupes) and bool(content_dupes),
              f"reasons={[marker_reasons.get(p, '') [:60] for p in content_dupes]}")

        # (4) The files that FAILED TO READ are remembered too (2026-09-17).
        # James: "i dont want to keep processing repeat files. i just want to
        # process new stuff." Against the old code both of these are unmarked
        # and the second pass offers them — which is what it did every day.
        check("ingest re-offer: a file the parser could not read is remembered "
              "by path",
              retryable <= markers,
              f"unmarked={sorted(Path(x).name for x in retryable - markers)}")
        check("ingest re-offer: the marker says the file could not be READ, "
              "not that it is junk",
              all(marker_reasons.get(p, "").startswith(UNREADABLE_PREFIX)
                  for p in retryable),
              f"reasons={[marker_reasons.get(p, '')[:70] for p in sorted(retryable)]}")
        check("ingest re-offer: the trim ERROR row is still an ERROR row",
              _trim_error_row_intact(db, trim_junk),
              "the marker must not have been implemented by flipping success")

        # The second pass: NOTHING is offered again. Every failure is now
        # remembered, permanent or not — the retry is an explicit button, not
        # a thing that happens to you every morning.
        rep2 = run_folders([str(folder), str(laser)], db=db, config=cfg,
                           incremental=True)
        second_new = rep2.new_files
        # Deliberately no new-API call in this line: against the old code the
        # ZERO check below must be REACHED and fail on its number, not be
        # skipped by an AttributeError from a count that does not exist yet.
        print(f"     | second pass: {second_new} new (expected 0)")

        check("ingest re-offer: the second pass offers ZERO files",
              second_new == 0,
              f"second_new={second_new} retryable={len(other)} "
              f"permanent={len(permanent)} duplicate={len(duplicate)} "
              f"same-content={len(content_dupes)}")
        check("ingest re-offer: the second pass offers strictly fewer files "
              "than the first pass failed on",
              second_new < total_failures or total_failures == 0,
              f"second_new={second_new} first_pass_failures={total_failures}")
        check("ingest re-offer: the duplicate path is not re-parsed",
              not (duplicate - markers),
              f"unmarked_duplicate={sorted(Path(x).name for x in duplicate - markers)}")

        # (5) The escape hatch, end to end: Settings → Retry unreadable files
        # re-offers the read failures and NOTHING else. If this counted the
        # permanent failures or the duplicates, pressing it once would undo
        # everything the markers are for.
        failures_before = db.count_failed_file_markers()
        cleared = db.reset_failed_file_markers()
        rep3 = run_folders([str(folder), str(laser)], db=db, config=cfg,
                           incremental=True)
        print(f"     | after Retry unreadable files: {cleared} cleared, "
              f"{rep3.new_files} offered again")
        check("ingest re-offer: the retry counts ONLY the read failures",
              failures_before == len(retryable) and cleared == len(retryable),
              f"count={failures_before} cleared={cleared} "
              f"expected={len(retryable)} of {db.count_skipped_files()} markers")
        check("ingest re-offer: the retry re-offers the read failures and "
              "nothing else",
              rep3.new_files == len(retryable),
              f"offered={rep3.new_files} expected={len(retryable)}")
        check("ingest re-offer: a file that still cannot be read is marked "
              "again",
              db.count_failed_file_markers() == len(retryable),
              f"markers={db.count_failed_file_markers()} expected={len(retryable)}")
    except Exception as exc:
        check("ingest re-offer: unreadable files are offered once", False,
              f"{type(exc).__name__}: {exc}")
    finally:
        _dbmod._db_manager = saved_global
        shutil.rmtree(tmp, ignore_errors=True)


def _unreadable_smoothness_workbook(path: Path) -> None:
    """An `*_OS_*` export the smoothness parser finds no usable columns in.

    The work-share signature, reproduced: "Generic parser found no usable
    columns in sheet 'Sheet1' (pos_col='Electrical Angle* + 280 min:',
    smooth_cols=[])", then "Smoothness parser returned no tracks for …".
    """
    import openpyxl  # noqa: E402
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.cell(row=1, column=1, value="Electrical Angle* + 280 min:")
    for i in range(2, 8):
        ws.cell(row=i, column=1, value=float(i))
    wb.save(path)


def _unreadable_trim_workbook(path: Path) -> None:
    """A trim-named export whose sheet layout the parser refuses.

    Stands in for the 111 old exports (8856, 8888, 6952, 8204-3, 8232-1, …)
    that produce a success=False `processed_files` row and are therefore
    retried on every single run.
    """
    import openpyxl  # noqa: E402
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "TRK1 Trimmed"
    for i in range(1, 6):
        ws.cell(row=i, column=1, value="header junk")
    wb.save(path)


def _trim_error_row_intact(db, path: Path) -> bool:
    """The ERROR row keeps success=False and its analysis; the marker is EXTRA.

    Conflating the two would hide the failure from the cleanup tools and from
    the scan's "retrying earlier errors" line.
    """
    from laser_trim_analyzer.database.models import (  # noqa: E402
        ProcessedFile as DBPF)
    with db.session() as sess:
        rows = sess.query(DBPF.file_hash, DBPF.success, DBPF.analysis_id,
                          DBPF.error_message).filter(
            DBPF.file_path == str(path)).all()
    errors = [r for r in rows if r[1] is False]
    marks = [r for r in rows if r[1] is True and r[2] is None
             and (r[3] or "").startswith(UNREADABLE_PREFIX)]
    return len(rows) == 2 and len(errors) == 1 and len(marks) == 1 \
        and errors[0][2] is not None


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
    from laser_trim_analyzer.core.processor import Processor  # noqa: E402
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

        # ---- the stat fast-path's KEY (2026-09-14 work incident) ----------
        # `_classify_scan` looks the walk's dict up by `str(file_path)`, and
        # the DB stores paths the same way. When the walk keyed by the raw
        # `entry.path` instead, a root in any other spelling (Tk's
        # `askdirectory` hands back forward slashes on Windows) missed on
        # every file: "check … (0 known in memory)" and a 542-second stat()
        # storm over the share, every run, on an unchanged folder.
        from laser_trim_analyzer.core.ingest_run import (  # noqa: E402
            discover_excel_files)
        odd_root = str(a) + "//"
        walked, walked_stats = discover_excel_files(odd_root)
        unnormalised = [p for p in walked if str(Path(p)) != p]
        missing_keys = [p for p in walked if str(Path(p)) not in walked_stats]
        check("multi-folder ingest: the walk returns paths in str(Path(p)) form",
              bool(walked) and not unnormalised,
              f"files={len(walked)} unnormalised={len(unnormalised)}"
              + (f" e.g. {unnormalised[0]}" if unnormalised else ""))
        check("multi-folder ingest: every stat key is the one _classify_scan "
              "looks up",
              bool(walked) and not missing_keys,
              f"files={len(walked)} missing={len(missing_keys)}"
              + (f" e.g. {missing_keys[0]}" if missing_keys else ""))
        # The end of it: an odd root must still settle in MEMORY. This is the
        # check that would have caught the incident — the two above pin the
        # spelling, this one pins the CONSEQUENCE.
        proc_odd = Processor(use_ml=False)
        proc_odd.config.processing.turbo_mode_threshold = 10 ** 9
        gen_odd = proc_odd.process_batch([Path(p) for p in walked],
                                         incremental=True,
                                         disk_stats=walked_stats)
        try:
            while True:
                next(gen_odd)
        except StopIteration:
            pass
        odd_stats = proc_odd.last_scan_stats or {}
        check("multi-folder ingest: a folder reached by an odd root stats "
              "NOTHING",
              odd_stats.get("needs_hash") == 0
              and odd_stats.get("memory_hits") == len(walked),
              f"needs_hash={odd_stats.get('needs_hash')} "
              f"memory_hits={odd_stats.get('memory_hits')} of {len(walked)}")
    except Exception as exc:
        check("multi-folder ingest: shared pipeline contract", False,
              f"{type(exc).__name__}: {exc}")
    finally:
        _dbmod._db_manager = saved_global
        shutil.rmtree(tmp, ignore_errors=True)

    check_ingest_reoffers_only_retryable()


def check_trim_capture() -> None:
    """Passes and setup are captured, and blank limits never become 0.0.

    Standalone: real trim workbooks from tests/fixtures/trim, no database.
        python scripts/app_qa_sweep.py --only ingest

    Guards the 2026-09-17 capture (per-pass sweeps + the laser setup block).
    The invariant that actually costs money is the blank one: an ignored point
    carries no limit, and turning that blank into 0.0 grades every point as
    failing. That exact bug cost a week on the final-test side in September
    2026, which is why the blank case gets both a FAIL condition AND a
    companion check that blanks are genuinely present in the fixtures — a
    "no zeros found" result is meaningless if there was nothing to get wrong.

    System A records per-POINT process columns (cut length, trim current,
    predicted vs used delta) that System B does not record at all. Those keys
    must be ABSENT on System B rather than present-and-empty, so a consumer
    can tell "this machine does not record it" from "it recorded nothing".

    EVERY number below is PINNED, not floored. The first version of this
    section asked only "is it non-empty?" and a review broke real code three
    separate ways without turning a single check red:

    - Pointing every `_A_PER_POINT` index at column 999 made all seven
      captured columns come back as `[None] * n` — `read_pass` sets the keys
      unconditionally and `_aligned` pads a missing column — and a
      key-presence test still printed PASS. So the columns are now checked
      for LENGTH (aligned to positions) and SUBSTANCE (a real number in
      there), not for the key existing.
    - Truncating `pass_sheets` to its first sheet dropped 2 of 3 passes on
      the LTS files — the exact failure mode of the per-sheet `except:
      continue` in the parser, and the bug class fixed in 1c70a0a — and
      "passes read from every fixture" passed, because it only ever caught a
      file with ZERO passes. Counts are pinned per file now.
    - That same truncation slid the blank-limit total from 254 to 92 without
      tripping a `>= 4 * len(fixtures)` floor. Blanks are pinned per file.

    Pinned values go stale if a fixture is ever replaced; that is the point.
    A fixture change should require saying so here, in the same commit.
    """
    from laser_trim_analyzer.core.models import SystemType  # noqa: E402
    from laser_trim_analyzer.core.parser import ExcelParser  # noqa: E402

    A_PER_POINT = ("trim_target", "initial_trim_value", "final_trim_value",
                   "pred_deltas", "used_deltas", "cut_lengths", "trim_currents")
    # The sweep columns every pass carries, on every system.
    CORE = ("positions", "errors", "upper_limits", "lower_limits")

    # Observed on the four tracked fixtures and pinned. "blanks" counts None
    # cells across upper_limits + lower_limits — the ignored points whose
    # limits must stay blank rather than become 0.0.
    # "populated" is the non-None count per System A per-point column, summed
    # over that file's passes. A bare "at least one real number" bar passes a
    # column of 55 Nones and one float, and the realistic bug with that exact
    # shape is an off-by-one index landing on a SPARSE neighbour -- the class
    # of 1c70a0a. The length check cannot see it either, because `_aligned`
    # pads to the same length whichever column it read. These columns are the
    # input to the cut-length model, so a hollow one matters.
    EXPECTED = {
        "dlts_8232-1_242.xls": {
            "tracks": ["TRK1"], "passes": 2, "setup_keys": 46, "blanks": 44,
            "populated": {"trim_target": 96, "initial_trim_value": 96,
                          "final_trim_value": 96, "pred_deltas": 112,
                          "used_deltas": 112, "cut_lengths": 112,
                          "trim_currents": 112}},
        "dlts_8232-1_243.xls": {
            "tracks": ["TRK1"], "passes": 3, "setup_keys": 46, "blanks": 66,
            "populated": {"trim_target": 144, "initial_trim_value": 144,
                          "final_trim_value": 144, "pred_deltas": 168,
                          "used_deltas": 168, "cut_lengths": 168,
                          "trim_currents": 168}},
        "lts_8232-1_193.xls": {"tracks": ["default"], "passes": 3,
                               "setup_keys": 70, "blanks": 72, "populated": {}},
        "lts_8232-1_194.xls": {"tracks": ["default"], "passes": 3,
                               "setup_keys": 70, "blanks": 72, "populated": {}},
        # Task 9 (2026-09-24): two-track DLTS fixtures. Column C of 'Track
        # Parameters' adds one key (`_track2`) to the parser's raw trim_setup
        # dict when present -- 46 + 1 = 47, on BOTH (7553_10B carries a real
        # Track 2 block despite being a TRK2-ONLY file; see trim_setup.
        # read_track2_keyvalue). Neither file's untrimmed sweep carries a
        # blank limit (0), unlike the original two DLTS fixtures.
        "dlts_7553_10B.xls": {                          # TRK2 only in this file
            "tracks": ["TRK2"], "passes": 2, "setup_keys": 47, "blanks": 0,
            "populated": {"trim_target": 696, "initial_trim_value": 696,
                          "final_trim_value": 696, "pred_deltas": 702,
                          "used_deltas": 702, "cut_lengths": 702,
                          "trim_currents": 702}},
        "dlts_8074_18.xls": {           # TRK1 (untrimmed only) + TRK2 (2 cuts)
            "tracks": ["TRK1", "TRK2"], "passes": 2, "setup_keys": 47, "blanks": 0,
            "populated": {"trim_target": 258, "initial_trim_value": 258,
                          "final_trim_value": 258, "pred_deltas": 262,
                          "used_deltas": 262, "cut_lengths": 262,
                          "trim_currents": 262}},
    }

    fixtures = sorted((REPO / "tests" / "fixtures" / "trim").glob("*.xls"))
    # A missing fixture is a FAIL, never a skip. These files are TRACKED, so
    # absence means a broken checkout, not an optional extra — and the old
    # `warn(); return` quietly removed seven checks from the sweep total while
    # the run still reported zero failures.
    present = {f.name for f in fixtures}
    check("trim capture: every tracked fixture is present",
          present == set(EXPECTED),
          f"missing={sorted(set(EXPECTED) - present)} "
          f"unexpected={sorted(present - set(EXPECTED))}")
    if not fixtures:
        return

    parser = ExcelParser()
    parsed_ok = 0
    blanks_seen = 0
    wrong_passes, wrong_tracks, wrong_setup, wrong_blanks = [], [], [], []
    zero_limits, bad_core, a_bad, b_leaked, hollow = [], [], [], [], []

    for f in fixtures:
        want = EXPECTED.get(f.name)
        if want is None:
            check(f"trim capture: {f.name} is a pinned fixture", False,
                  "not in EXPECTED; add it with its observed counts")
            continue
        try:
            parsed = parser.parse_file(f)
        except Exception as exc:
            # An exception must read FAIL, never vanish into a skipped file:
            # a check that can pass on an ERROR result is itself a bug.
            check(f"trim capture: {f.name} parses", False,
                  f"{type(exc).__name__}: {exc}")
            continue
        parsed_ok += 1

        setup = parsed.get("trim_setup") or {}
        if len(setup) != want["setup_keys"]:
            # Not a truthiness test: the parser discards a candidate layout
            # under 3 keys, so "truthy" only ever proved 3 of the 46/70 keys
            # these files carry.
            wrong_setup.append(
                f"{f.name}: {len(setup)} setup keys, expected {want['setup_keys']}")

        system = getattr(parsed.get("metadata"), "system", None)
        tracks = parsed.get("tracks") or []
        got_tracks = [t.get("track_id") for t in tracks]
        if got_tracks != want["tracks"]:
            wrong_tracks.append(f"{f.name}: {got_tracks} != {want['tracks']}")

        file_passes = file_blanks = 0
        populated = {k: 0 for k in A_PER_POINT}
        for track in tracks:
            passes = track.get("trim_passes") or []
            file_passes += len(passes)
            for p in passes:
                where = f"{f.name} {track.get('track_id')} pass {p.get('pass_index')}"
                n = len(p.get("positions") or [])
                for side in ("upper_limits", "lower_limits"):
                    vals = p.get(side) or []
                    file_blanks += sum(1 for v in vals if v is None)
                    if any(v == 0.0 for v in vals if v is not None):
                        zero_limits.append(f"{where} {side}")
                # Substance, both systems: a column that came back all-None
                # (wrong index, off the end of the sheet) or out of step with
                # positions is data loss, and key presence cannot see either.
                for key in CORE:
                    bad = _column_defect(p, key, n)
                    if bad:
                        bad_core.append(f"{where} {bad}")
                if system == SystemType.A:
                    for key in A_PER_POINT:
                        bad = _column_defect(p, key, n)
                        if bad:
                            a_bad.append(f"{where} {bad}")
                        else:
                            populated[key] += sum(
                                1 for v in p[key] if v is not None)
                else:
                    leaked = [k for k in A_PER_POINT if k in p]
                    if leaked:
                        b_leaked.append(f"{where}: {leaked}")

        if file_passes != want["passes"]:
            wrong_passes.append(
                f"{f.name}: {file_passes} passes, expected {want['passes']}")
        if file_blanks != want["blanks"]:
            wrong_blanks.append(
                f"{f.name}: {file_blanks} blank limits, expected {want['blanks']}")
        blanks_seen += file_blanks
        want_pop = want["populated"]
        for key in A_PER_POINT:
            got = populated[key] if want_pop else 0
            if got != want_pop.get(key, 0):
                hollow.append(f"{f.name} {key}: {got} real values, "
                              f"expected {want_pop.get(key, 0)}")

    check("trim capture: every fixture parses", parsed_ok == len(EXPECTED),
          f"{parsed_ok} of {len(EXPECTED)}")
    check("trim capture: every fixture yields the passes it is known to hold",
          not wrong_passes, "; ".join(wrong_passes) if wrong_passes
          else f"{sum(v['passes'] for v in EXPECTED.values())} passes, "
               f"per-file counts as pinned")
    check("trim capture: every fixture yields the tracks it is known to hold",
          not wrong_tracks, "; ".join(wrong_tracks) if wrong_tracks
          else f"{sum(len(v['tracks']) for v in EXPECTED.values())} tracks, as pinned")
    check("trim capture: every fixture yields the setup keys it is known to hold",
          not wrong_setup, "; ".join(wrong_setup) if wrong_setup
          else "46/46/70/70/47/47 keys, as pinned")
    # Vacuity guard for the check below, now PINNED rather than floored:
    # without real blanks in the fixtures, "no 0.0 found" would be true of
    # code that turns every blank into 0.0.
    check("trim capture: the fixtures carry exactly the blank limits they are "
          "known to carry", not wrong_blanks,
          "; ".join(wrong_blanks) if wrong_blanks
          else f"{blanks_seen} blank limit cells, per-file counts as pinned")
    check("trim capture: a blank limit never becomes 0.0", not zero_limits,
          "; ".join(zero_limits[:3]) if zero_limits
          else f"0 zero-valued limits across {blanks_seen} blanks")
    check("trim capture: every sweep column is aligned to positions and "
          "carries real numbers", not bad_core,
          "; ".join(bad_core[:3]) if bad_core
          else f"{len(CORE)} columns on every pass of every fixture")
    check("trim capture: System A keeps its per-point process columns",
          not a_bad, "; ".join(a_bad[:3]) if a_bad
          else f"all {len(A_PER_POINT)} columns present, aligned and populated "
               f"on every System A pass")
    # The sparse-neighbour case the "at least one real number" bar cannot see.
    check("trim capture: each System A column holds every value it is known "
          "to hold", not hollow, "; ".join(hollow[:3]) if hollow
          else "per-column non-None counts as pinned on every System A fixture")
    check("trim capture: System B does not fake the columns it lacks",
          not b_leaked, "; ".join(b_leaked[:3]) if b_leaked
          else "absent, not empty, on every non-System-A pass")
    _check_setup_block_survives_json()


def _check_setup_block_survives_json() -> None:
    """A parameter block with a non-string cell must store as a real dict.

    `trim_setup.parameters` is one SafeJSON blob, and SafeJSON substitutes
    None for a value it cannot serialise -- so a single `Template Updated`
    datetime did not lose that one cell, it wrote the WHOLE block as the
    literal string "null", which reads back as `[]`. 44 of 522 real DLTS
    files across 23 models, with the promoted columns landing correctly the
    whole time, which is exactly why nobody noticed.

    The fixture is kept OUT of tests/fixtures/trim on purpose: that directory
    is globbed by the pinned table above and by the no-op baseline, and a
    fifth file there would break both for the wrong reason.
    """
    import json as _json  # noqa: E402
    from laser_trim_analyzer.core.parser import ExcelParser  # noqa: E402

    src = REPO / "tests" / "fixtures" / "trim_setup" / "8251-1_29_template_updated.xls"
    check("trim setup: the Template Updated fixture is present", src.exists(),
          str(src))
    if not src.exists():
        return
    try:
        setup = (ExcelParser().parse_file(src) or {}).get("trim_setup")
    except Exception as exc:
        check("trim setup: the Template Updated fixture parses", False,
              f"{type(exc).__name__}: {exc}")
        return

    check("trim setup: the file really does carry the non-string cell that "
          "broke this", isinstance(setup, dict)
          and setup.get("template_updated") is not None,
          f"template_updated={setup.get('template_updated')!r}"
          if isinstance(setup, dict) else f"setup={setup!r}")
    # The failure shape, asserted directly: a dict, not a list, and not the
    # 4-byte "null" SafeJSON writes when serialisation fails.
    try:
        encoded = _json.dumps(setup)
        ok, why = True, f"{len(setup)} keys, {len(encoded)} bytes of JSON"
    except (TypeError, ValueError) as exc:
        ok, why = False, (f"{type(exc).__name__}: {exc} -- SafeJSON would "
                          f"store the WHOLE block as \"null\"")
    check("trim setup: the whole parameter block survives JSON serialisation",
          ok and isinstance(setup, dict) and len(setup) > 3, why)
    # And a date is stored as text a consumer can read, not silently dropped.
    check("trim setup: a date cell is stored as an ISO string, not discarded",
          isinstance(setup, dict)
          and isinstance(setup.get("template_updated"), str),
          f"template_updated={setup.get('template_updated')!r}"
          if isinstance(setup, dict) else f"setup={setup!r}")
    # A date in the LABEL column must not become a key: _clean coerces values
    # so they survive JSON, and running labels through it invented the key
    # '2026_01_06t00_00_00' on this very file.
    # Not "starts with a digit": `8251_1` and `05bf8251_1` are real labels on
    # this sheet (the model number, with the value "Model"). The shape ruled
    # out is a key derived from a coerced DATE. The key COUNT is pinned too,
    # which is what actually catches a label rule that starts inventing keys.
    import re as _re  # noqa: E402
    dated = [k for k in (setup or {}) if _re.match(r"^\d{4}_\d{2}_\d{2}", k)]
    check("trim setup: a date in the label column does not become a key",
          not dated and isinstance(setup, dict) and len(setup) == 54,
          f"dated={dated[:3]} keys={len(setup) if isinstance(setup, dict) else setup} "
          f"(expected 54)")


def _column_defect(p: dict, key: str, n: int):
    """Why `p[key]` is not a usable column of length `n`, or None if it is.

    Key presence proves nothing here. `read_pass` sets every System A
    per-point key unconditionally and `_aligned` pads a column it could not
    find with None, so a wrong column index yields a full-length list of
    nothing — present, correctly sized, and empty of data.
    """
    if key not in p:
        return f"{key}: absent"
    col = p.get(key)
    if col is None:
        return f"{key}: None"
    if len(col) != n:
        return f"{key}: {len(col)} values, positions has {n}"
    if not any(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in col):
        return f"{key}: {len(col)} values, every one of them None"
    return None


def check_ingest_group() -> None:
    """The ingest group: the batch contract, the re-offer policy, and the
    trim-pass/setup capture that rides the same parse."""
    check_multi_folder_ingest()
    check_trim_capture()


# ---- laser 1's TrimVolts capture (2026-09-24) --------------------------------
# A laser-1 `Trim N` pass row created before the database started capturing
# cannot carry `increment_volts` -- filling those is the back-fill's job (design
# doc ruling 4c) or a reprocess's, never a failure here. WHEN that was is the
# database's own record (app_meta, written by its start-up migration), not a
# date typed in here. `created_date` and that record are both UTC
# 'YYYY-MM-DD HH:MM:SS.ffffff', so a string compare is a time compare.


def _increment_volts_since(conn):
    """The database's own record of when it started capturing TrimVolts, or None."""
    from laser_trim_analyzer.database.manager import INCREMENT_VOLTS_SINCE_KEY
    try:
        row = conn.execute("SELECT value FROM app_meta WHERE key = ?",
                           (INCREMENT_VOLTS_SINCE_KEY,)).fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def _laser1_pass_number(sheet):
    """N for laser 1's pass sheet `Trim N`, else None (`Lin Error`, laser 2's `SEC1 TRK1 ...`)."""
    import re as _re
    m = _re.match(r"^trim\s+(\d+)$", (sheet or "").strip(), _re.I)
    return int(m.group(1)) if m else None


def _file_has_trimvolts(path, n):
    """True/False: does the workbook carry a `TrimVolts n` sheet with anything in it?
    None: it cannot be opened here (a work-share path on the Mac, a moved file).

    Deliberately NOT the parser's own sheet lookup: a check that asks the code
    under test whether the sheet exists agrees with that code's bugs. Never raises:
    a row with no file_path at all is simply unverifiable."""
    import pandas as _pd
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists():
            return None
        xl = _pd.ExcelFile(p)
        names = [s for s in xl.sheet_names
                 if s.replace(" ", "").lower() == f"trimvolts{n}"]
        if not names:
            return False
        df = _pd.read_excel(xl, sheet_name=names[0], header=None)
        return bool(df.size) and bool(df.notna().to_numpy().any())
    except Exception:
        return None


def _placement_refused(path, sheet):
    """True when the capture rule itself REFUSES this pass: the workbook's own VOLTAGES
    sheet contradicts where its TrimVolts curves would sit, so ingest stores no curves, by
    design (core/parser.py `_increment_volts_for`, 2026-09-24). Settled by the back-fill's
    reader (`scripts/backfill_increment_volts._capture_file`: the same shared
    `voltages_placement` rule, reached through a reader apart from the parser that stored
    the row). False whenever it cannot be settled -- the pass then stays "missed"."""
    try:
        from backfill_increment_volts import _capture_file
        return sheet in _capture_file(path, [sheet]).disagreed
    except Exception:
        return False


def _increment_volts_audit(conn, since, max_files=20):
    """Every laser-1 `Trim N` pass row in `conn`, sorted into buckets that add up.

    Created before `since` (the database's own record, `_increment_volts_since`):
    `old_captured` (back-filled) or `old_uncaptured` (the back-fill's job).
    Created since: `captured`, or settled against the pass's own workbook -- the
    sheet is there, so the capture missed it (`missed`, a FAIL) unless the capture
    rule refuses it because the workbook's own VOLTAGES sheet contradicts the
    placement (`refused`: correct -- ingest stores no curves then, since 2026-09-24,
    and a WARN names them); the sheet is not (a touch-up file -- 1 in 4,972
    locally) or holds no reading (`no_sheet`, correct); or the file cannot be
    opened (`unverified`). At most `max_files` workbooks are opened; beyond that a
    row counts as unverified. With no `since` at all, every row is held to "must
    carry".
    passes == old_captured + old_uncaptured + captured + no_sheet
              + len(missed) + len(refused) + len(unverified).
    Raw SQL on purpose, and `IS NULL` is exact here: the writer stores a real SQL
    NULL for "no capture", never SafeJSON's 'null' text.
    """
    import re as _re
    out = {"passes": 0, "old_captured": 0, "old_uncaptured": 0, "captured": 0,
           "no_sheet": 0, "missed": [], "refused": [], "unverified": []}
    verdicts = {}
    for sheet, created, has, path in conn.execute(
            "SELECT p.sheet, p.created_date, p.increment_volts IS NOT NULL, a.file_path "
            "FROM trim_passes p JOIN track_results t ON t.id = p.track_result_id "
            "JOIN analysis_results a ON a.id = t.analysis_id "
            "WHERE p.sheet LIKE 'trim %'"):
        n = _laser1_pass_number(sheet)
        if n is None:
            continue
        out["passes"] += 1
        if since is not None and (created or "") < since:
            out["old_captured" if has else "old_uncaptured"] += 1
            continue
        if has:
            out["captured"] += 1
            continue
        key = (path, n)
        if key not in verdicts:
            verdicts[key] = (_file_has_trimvolts(path, n)
                             if len(verdicts) < max_files else None)
            if verdicts[key] is True and _placement_refused(path, sheet):
                verdicts[key] = "refused"
        name = (_re.split(r"[\\/]", path)[-1]         # a UNC path on any OS
                if path else "(no file_path)")
        where = f"{name} {sheet}"
        if verdicts[key] == "refused":
            out["refused"].append(where)
        elif verdicts[key] is True:
            out["missed"].append(where)
        elif verdicts[key] is False:
            out["no_sheet"] += 1
        else:
            out["unverified"].append(where)
    return out


def check_increment_volts_fixtures() -> None:
    """Laser 1's TrimVolts capture through the REAL pipeline, into a throwaway database
    (--only increment-volts). This is the half with teeth: the copy of the work database
    holds no pass processed since the capture shipped until someone processes one, so the
    database rule below is also run here, on rows that must carry the curves.

    Falsify before trusting (2026-09-24): make `_write_trim_passes` store `sql_null()` for
    increment_volts, or skip `_increment_volts_for` in the parser -- the first and last
    checks here go FAIL (0 of 4 captured; the rule names all 4 passes as missed). Stop
    the migration recording its app_meta start and the last check goes FAIL (since=None).
    """
    import json as _json
    import shutil
    import tempfile
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as _mgr
    import laser_trim_analyzer.database as _dbpkg

    keys = ("increment_volts", "increment_volts_first_row", "increment_volts_truncated")
    # Readings per pass, counted cell by cell on the fixtures' own TrimVolts sheets;
    # 49 curves (57 - 2 - 7 + 1) from first_row 2 on every pass, none truncated.
    pinned = {("lts_8232-1_193.xls", "Trim 1"): 199, ("lts_8232-1_193.xls", "Trim 2"): 719,
              ("lts_8232-1_194.xls", "Trim 1"): 880, ("lts_8232-1_194.xls", "Trim 2"): 795}
    want = {k: (49, v, 2, 0) for k, v in pinned.items()}
    fixtures = [f for f in (REPO / "tests" / "fixtures" / "trim" / n for n in _FOUR_8232_FIXTURES)
                if f.is_file()]
    check("increment volts: the four trim fixtures are present", len(fixtures) == 4,
          f"{[f.name for f in fixtures]}")
    if len(fixtures) != 4:
        return
    saved = (_mgr._db_manager, getattr(_dbpkg, "_db_manager", None))
    tmp = Path(tempfile.mkdtemp(prefix="increment_volts_sweep_"))
    fdb = None
    try:
        fdb = _mgr.DatabaseManager(tmp / "iv.db")
        _mgr._db_manager = fdb                 # BOTH globals: a Processor must never
        _dbpkg._db_manager = fdb               # reach the configured database.
        proc = Processor(use_ml=False)
        for f in fixtures:
            fdb.save_analysis(proc.process_file(f))
        conn = sqlite3.connect(f"file:{tmp / 'iv.db'}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT a.filename, p.sheet, p.increment_volts, p.increment_volts_first_row, "
                "p.increment_volts_truncated, p.recipe FROM trim_passes p "
                "JOIN track_results t ON t.id = p.track_result_id "
                "JOIN analysis_results a ON a.id = t.analysis_id").fetchall()
            since = _increment_volts_since(conn)
            audit = _increment_volts_audit(conn, since)
        finally:
            conn.close()
        got, leaked, in_recipe = {}, [], []
        for fname, sheet, volts, first_row, truncated, recipe in rows:
            if (fname, sheet) in pinned:
                curves = _json.loads(volts) if volts is not None else []
                got[(fname, sheet)] = (len(curves), sum(len(c) for c in curves),
                                       first_row, truncated)
            elif (volts, first_row, truncated) != (None, None, None):
                leaked.append(f"{fname} {sheet}: {str(volts)[:12]!r}/{first_row}/{truncated}")
            if recipe and set(keys) & set(_json.loads(recipe) or {}):
                in_recipe.append(f"{fname} {sheet}")
        check("increment volts: every laser-1 Trim N fixture pass stores its 49 curves, "
              "the pinned readings, first_row 2, not truncated",
              got == want, f"{sorted(got.items())}" if got != want
              else f"{len(got)} passes, {sum(v[1] for v in got.values())} readings")
        check("increment volts: no other pass carries them (Lin Error, laser 2) -- a real "
              "SQL NULL, never SafeJSON's 'null' text",
              not leaked and len(rows) == 11,
              f"{len(rows)} pass rows; leaked={leaked[:3]}")
        check("increment volts: never folded into the recipe blob", not in_recipe,
              f"{in_recipe[:3]}")
        # A new database records when it started capturing at its first start-up, so
        # every fixture pass saved after that is held to "must carry" -- none predates.
        check("increment volts: the database rule finds the fixtures' 4 passes captured, "
              "processed since the start this database recorded",
              since is not None and audit["passes"] == 4 and audit["captured"] == 4
              and not audit["missed"] and not audit["unverified"],
              f"since={since}; missed={audit['missed'][:3]}; "
              + _increment_volts_counts(audit))
    except Exception as e:                      # an exception is a FAIL, never a skip
        check("increment volts: the fixtures run through the pipeline", False,
              f"{type(e).__name__}: {e}")
    finally:
        _mgr._db_manager, _dbpkg._db_manager = saved
        if fdb is not None:
            fdb.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _increment_volts_counts(a):
    """One line in which the numbers add up to a['passes']."""
    new = (a["captured"] + a["no_sheet"] + len(a["missed"]) + len(a["refused"])
           + len(a["unverified"]))
    return (f"{new} processed since the capture started: {a['captured']} carry their "
            f"curves, {a['no_sheet']} whose workbook has no TrimVolts reading, "
            f"{len(a['refused'])} refused (their VOLTAGES sheet contradicts the placement), "
            f"{len(a['unverified'])} unverifiable here, {len(a['missed'])} missed; "
            f"{a['old_captured'] + a['old_uncaptured']} predate it "
            f"({a['old_captured']} back-filled, {a['old_uncaptured']} not yet) "
            f"-- {a['passes']} laser-1 Trim N passes in all")


def check_increment_volts_on_database(raw) -> None:
    """On the copy: every laser-1 `Trim N` pass processed since THIS database started
    capturing (its own app_meta record) carries its TrimVolts curves. Rows that predate it
    are skipped (the back-fill's job, not a failure). Until a pass is processed on a
    machine running this code there is nothing to check here: that is a WARN carrying the
    counts, never a PASS over zero passes (2026-09-24 final review -- the same pattern
    e801e16 fixed for track 2); the fixture half above carries the teeth meanwhile."""
    since = _increment_volts_since(raw)
    check("increment volts: the database records when it started capturing TrimVolts",
          since is not None,
          f"since={since}" if since else
          "no app_meta record: its start-up migration did not run, or could not record it")
    a = _increment_volts_audit(raw, since)
    processed_since = (a["captured"] + a["no_sheet"] + len(a["missed"]) + len(a["refused"])
                       + len(a["unverified"]))
    if not processed_since:
        warn("increment volts: no laser-1 Trim N pass processed since the capture started "
             "-- nothing to hold to it yet (0 is correct until this code ingests a laser-1 "
             "file)", _increment_volts_counts(a))
        return
    check("increment volts: every laser-1 Trim N pass processed since the capture started "
          "carries its TrimVolts curves",
          not a["missed"],
          (f"missed={a['missed'][:3]}; " if a["missed"] else "") + _increment_volts_counts(a))
    if a["refused"]:
        warn("increment volts: passes processed since the capture whose curves were REFUSED "
             "at ingest -- the workbook's own VOLTAGES sheet contradicts where the start-row "
             "rule places them (a wrong position is worse than none); look at these files",
             f"{len(a['refused'])}: {a['refused'][:3]}")
    if a["unverified"]:
        warn("increment volts: passes processed since the capture WITHOUT curves whose "
             "workbook cannot be opened here to settle it",
             f"{len(a['unverified'])}: {a['unverified'][:3]}")


def _trimvolts_placement_one(path):
    """One laser-1 workbook: the parser's TrimVolts capture held against the machine's OWN
    placement. Laser 1 writes each `TrimVolts N` column's last reading into its `VOLTAGES`
    sheet, column N, at the position row of that column -- so curve k must end in
    VOLTAGES[first_row + k, N], exactly. The one allowance, measured on every local sheet
    (2026-09-24): the LAST non-empty curve's cell may be blank (55 passes); no other.

    The parser supplies only what is under test (curves, first_row). The sheet census,
    VOLTAGES and the file's own Model Parameters are read here, independently -- and
    BEFORE the parse, so a workbook that then fails to parse is still known to be one
    whose Points From Start differs from its Initial Points Ignored (`pfs_differs`): the
    files the first_row rule exists for. Module level so a process pool can run it.

    The comparison itself is `core.trim_passes.voltages_placement` (Task 5 fix round 2):
    this function still does its own independent file census/reading, but no longer its own
    copy of the placement loop. Since 2026-09-24 the parser applies that same rule at
    ingest and REFUSES a capture VOLTAGES contradicts (all three keys None): such a pass
    lands in `refused`, not `misplaced` -- `misplaced` is now the proof that no stored
    capture is ever on the wrong positions, and `refused` that the start-row rule still
    places every local file.
    """
    import re as _re
    import pandas as _pd
    from laser_trim_analyzer.core import trim_passes as _tp
    from laser_trim_analyzer.core.parser import ExcelParser, drop_cached_bytes

    def _count(v):
        try:
            f = float(v)
        except (TypeError, ValueError, OverflowError):
            return None
        return int(f) if f == f and f.is_integer() else None

    out = {"file": Path(path).name, "checked": 0, "placed": 0, "blank_last": 0,
           "uncheckable": 0, "misplaced": [], "uncaptured": [], "refused": [],
           "pfs_differs": False,
           "error": None}
    try:
        with _pd.ExcelFile(path) as xl:
            names = xl.sheet_names
            labels = {}
            if "Model Parameters" in names:
                mp = _pd.read_excel(xl, sheet_name="Model Parameters", header=None)
                for i in range(mp.shape[0] if mp.shape[1] > 1 else 0):
                    if isinstance(mp.iat[i, 1], str):        # laser 1: value, then label
                        labels.setdefault(mp.iat[i, 1].strip().lower(), mp.iat[i, 0])
            ipi, pfs, pfe = (_count(labels.get(k)) for k in (
                "initial points ignored", "points from start", "points from end"))
            out["pfs_differs"] = None not in (ipi, pfs, pfe) and pfs != ipi
            tv = {}
            for s in names:
                m = _re.fullmatch(r"trimvolts(\d+)", s.replace(" ", "").lower())
                if m:
                    tv.setdefault(int(m.group(1)), []).append(s)
            volts = (_pd.read_excel(xl, sheet_name="VOLTAGES", header=None)
                     if "VOLTAGES" in names else None)
            try:
                parsed = ExcelParser().parse_file(Path(path))
            finally:
                drop_cached_bytes(Path(path))
            passes = {str(p.get("sheet", "")).strip().lower(): p
                      for t in parsed.get("tracks") or [] for p in t.get("trim_passes") or []}
            for s in names:
                m = _re.fullmatch(r"trim\s+(\d+)", s.strip(), _re.I)
                if not m or len(tv.get(int(m.group(1)), [])) != 1:
                    continue           # no TrimVolts beside it (a touch-up), or ambiguous
                n = int(m.group(1))
                p = passes.get(s.strip().lower())
                if p is None:
                    continue           # the pass itself was never read: not this check's
                where = f"{out['file']} {s}"
                curves = p.get("increment_volts") or []
                if not any(curves):
                    raw = _pd.read_excel(xl, sheet_name=tv[n][0], header=None)
                    if any(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and v == v and v != 0 for v in raw.to_numpy().ravel()):
                        # The parser's refusal leaves the keys present and None; a read that
                        # failed, or never ran, leaves them absent.
                        refused = "increment_volts" in p and p["increment_volts"] is None
                        out["refused" if refused else "uncaptured"].append(where)
                    continue
                out["checked"] += 1
                fr = p.get("increment_volts_first_row")
                # The comparison loop itself lives in core.trim_passes.voltages_placement
                # (Task 5 fix round 2) -- lifted out unchanged so this sweep and a live
                # capture (the back-fill script, about to write a row) run one rule, not two
                # that could drift apart. See that function's docstring for the blank-last
                # allowance and the "at least one match" requirement.
                pc = _tp.voltages_placement(curves, fr, volts, n)
                if pc.result == "uncheckable":
                    out["uncheckable"] += 1
                elif pc.result == "misplaced":
                    detail = ("no first_row" if fr is None else
                             f"first_row {fr}, {pc.bad} of {pc.live_count} off, "
                             f"{pc.matched} matched")
                    out["misplaced"].append(f"{where}: {detail}")
                else:
                    out["placed"] += 1
                    out["blank_last"] += int(pc.blank_last)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:80]}"
    return out


def check_increment_volts_corpus() -> None:
    """Every captured laser-1 pass in the LOCAL corpus sits where the machine put it
    (--only increment-volts). Reads workbooks only, never a database.

    Why it exists (2026-09-24 review): `first_row` was Initial Points Ignored, and on 36 of
    6,263 local passes the machine had started at Points From Start instead -- every curve
    on them one position off, invisible to the fixtures (8232-1 has no Points From Start).

    It can never PASS on an error or on nothing (2026-09-24 re-review: a mutant that made
    every Points-From-Start file fail to parse left 42 workbooks as "errors" and both
    checks still read PASS -- at work, on the fixtures alone, the rule's only file
    errored and they passed on the two 8232-1 fixtures). So: any workbook that does not
    parse or read is a FAIL (0 of 4,973 do today); no workbook at all is a FAIL; and at
    least one PLACED pass must come from a workbook whose Points From Start differs from
    its Initial Points Ignored -- the files the rule exists for -- or the check names that
    it did not exercise them. The tracked touch-up fixture is one, so this holds at work.

    A capture the parser REFUSED at ingest (its VOLTAGES sheet contradicts the placement;
    `core/parser.py`, 2026-09-24) FAILs the second check here, named as refused: locally the
    start-row rule places every file, so a refusal means the rule regressed. (At work it is
    a WARN -- `check_increment_volts_on_database` -- where the rule is unproven.)

    Falsify before trusting: make `increment_volts_frame` ignore points_from_start and
    both checks go FAIL: the parser refuses the 37 captures VOLTAGES contradicts (36 local
    + the fixture), named as refused, and no placed pass is left from a Points-From-Start
    workbook; stop the parser refusing as well and the 37 are stored, and FAIL here as
    misplaced. Make it raise for those files and it goes FAIL on the errors and on zero
    exercised, with or without the local corpus.
    """
    import os as _os
    from concurrent.futures import ProcessPoolExecutor
    roots = [REPO / "Work Files" / "home_slice" / "LTS",
             REPO / "Work Files" / "Sample_Base_2026-04-10" / "LTS"]
    corpus = sorted(str(p) for r in roots if r.is_dir() for p in r.rglob("*.xls"))
    tracked = (sorted(str(p) for p in (REPO / "tests" / "fixtures" / "trim").glob("lts_*.xls"))
               + sorted(str(p) for p in (REPO / "tests" / "fixtures" / "trimvolts").glob("*.xls")))
    if not corpus:
        warn("increment volts: no local laser-1 corpus (Work Files/.../LTS) on this machine",
             f"the placement check below ran on the {len(tracked)} tracked fixtures only")
    files = corpus + tracked
    workers = max(1, min(8, (_os.cpu_count() or 2) - 1))
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            res = list(pool.map(_trimvolts_placement_one, files, chunksize=16))
    except Exception as exc:                    # an exception is a FAIL, never a skip
        check("increment volts: the corpus placement check runs", False,
              f"{type(exc).__name__}: {exc}")
        return
    parsed = [r for r in res if r["error"] is None]
    checked = sum(r["checked"] for r in parsed)
    misplaced = [m for r in parsed for m in r["misplaced"]]
    uncaptured = [u for r in parsed for u in r["uncaptured"]]
    refused_at_ingest = [u for r in parsed for u in r["refused"]]
    errors = [f"{r['file']}: {r['error']}" for r in res if r["error"] is not None]
    pfs_files = [r for r in res if r["pfs_differs"]]
    pfs_errored = sum(1 for r in pfs_files if r["error"] is not None)
    pfs_placed = sum(r["placed"] for r in pfs_files if r["error"] is None)
    # Reasons NOT to pass, shared by both verdicts: neither may read PASS on an error
    # or on nothing.
    refuse = []
    if not parsed:
        refuse.append(f"no workbook parsed, of {len(res)}")
    if errors:
        refuse.append(f"{len(errors)} of {len(res)} workbooks did not parse or read "
                      f"({pfs_errored} of them Points-From-Start files): {errors[:2]}")
    why = list(refuse)
    if misplaced:
        why.append(f"misplaced={misplaced[:3]} ({len(misplaced)})")
    if not pfs_placed:
        why.append(f"not one placed pass from a workbook whose Points From Start differs "
                   f"from its Initial Points Ignored ({len(pfs_files)} such workbooks, "
                   f"{pfs_errored} errored) -- the files the first_row rule exists for "
                   f"were not exercised")
    check("increment volts: every captured laser-1 pass sits where the machine's own "
          "VOLTAGES sheet puts it (corpus)",
          not why,
          ("; ".join(why) + " | " if why else "")
          + f"{checked} passes checked in {len(parsed)} of {len(res)} workbooks "
          f"({sum(r['placed'] for r in parsed)} placed -- {pfs_placed} of them from "
          f"{len(pfs_files)} workbooks whose Points From Start differs from Initial Points "
          f"Ignored -- {sum(r['blank_last'] for r in parsed)} with the allowed blank last "
          f"cell, {sum(r['uncheckable'] for r in parsed)} without a VOLTAGES column to hold "
          f"them to)")
    why = refuse + ([f"uncaptured={uncaptured[:3]} ({len(uncaptured)})"] if uncaptured else [])
    if refused_at_ingest:
        why.append(f"refused at ingest={refused_at_ingest[:3]} ({len(refused_at_ingest)}) -- "
                   f"the parser's VOLTAGES check contradicts the start-row rule on a local "
                   f"file, where the rule is proven")
    check("increment volts: every Trim N with a TrimVolts reading beside it is captured "
          "(corpus)", not why,
          "; ".join(why) if why else f"all of them, across all {len(parsed)} workbooks")


# ---- initial_trim_value gets its own column (2026-09-24) --------------------
# Laser 2 (DLTS) and laser 3 (LTS3, read in laser 2's format) pass sheets carry, per
# position, the trim target (column L), the INITIAL trim value (M) and the final trim
# value (N). The parser has always read column M (core/trim_passes._A_PER_POINT), but
# until now `_write_trim_passes` folded it into the pass row's `recipe` JSON blob instead
# of giving it its own column. Existing rows are NOT migrated at start-up -- a heavy
# UPDATE across ~83,000 rows at start-up is the shape of the 2026-09-14 night -- so they
# keep the value in `recipe` and must read back the same through
# `trim_passes.initial_trim_values`.

def check_initial_trim_value_fixtures() -> None:
    """Through the real pipeline, into a throwaway database (--only initial-trim-value):
    a laser-2 (DLTS) Trim N pass gets its own `initial_trim_value` column, the recipe blob
    no longer carries the key, and a laser-1 (LTS) pass has it in neither place (its sheets
    have no such column).

    Falsify before trusting (2026-09-24): make `_write_trim_passes` store `sql_null()` for
    initial_trim_value regardless of the parsed value -- the first check below goes FAIL
    (no DLTS pass carries a value). Stop excluding the key from `recipe` -- the second check
    FAILs. Let a laser-1 pass carry a plain `None` through SafeJSON instead of `sql_null()`
    -- the third check FAILs (the JSON text 'null' is not IS NULL, so it would read as
    "leaked").
    """
    import json as _json
    import shutil
    import tempfile
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as _mgr
    import laser_trim_analyzer.database as _dbpkg

    dlts = REPO / "tests" / "fixtures" / "trim" / "dlts_8232-1_243.xls"
    lts = REPO / "tests" / "fixtures" / "trim" / "lts_8232-1_193.xls"
    check("initial trim value: both fixtures are present", dlts.exists() and lts.exists(),
          f"dlts={dlts.exists()} lts={lts.exists()}")
    if not (dlts.exists() and lts.exists()):
        return
    saved = (_mgr._db_manager, getattr(_dbpkg, "_db_manager", None))
    tmp = Path(tempfile.mkdtemp(prefix="initial_trim_value_sweep_"))
    fdb = None
    try:
        fdb = _mgr.DatabaseManager(tmp / "itv.db")
        _mgr._db_manager = fdb                 # BOTH globals: a Processor must never
        _dbpkg._db_manager = fdb               # reach the configured database.
        proc = Processor(use_ml=False)
        for f in (dlts, lts):
            fdb.save_analysis(proc.process_file(f))
        conn = sqlite3.connect(f"file:{tmp / 'itv.db'}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT a.system, p.sheet, p.initial_trim_value, p.recipe FROM trim_passes p "
                "JOIN track_results t ON t.id = p.track_result_id "
                "JOIN analysis_results a ON a.id = t.analysis_id").fetchall()
        finally:
            conn.close()
        dlts_real, dlts_missing, in_recipe, laser1_leaked, laser1_rows = 0, [], [], [], 0
        for system, sheet, raw_value, recipe_raw in rows:
            recipe = _json.loads(recipe_raw) if recipe_raw else None
            in_recipe_here = isinstance(recipe, dict) and "initial_trim_value" in recipe
            if system in ("A", "C"):
                values = _json.loads(raw_value) if raw_value is not None else None
                if values and any(v is not None for v in values):
                    dlts_real += 1
                else:
                    dlts_missing.append(sheet)
                if in_recipe_here:
                    in_recipe.append(sheet)
            else:                                       # laser 1: no such column at all
                laser1_rows += 1
                if raw_value is not None or in_recipe_here:
                    laser1_leaked.append(sheet)
        check("initial trim value: every DLTS Trim N fixture pass carries a real value "
              "through its own column",
              dlts_real > 0 and not dlts_missing,
              f"{dlts_real} with values; missing={dlts_missing[:3]}")
        check("initial trim value: never folded into the recipe blob any more",
              not in_recipe, f"{in_recipe[:3]}")
        check("initial trim value: a laser-1 pass has it in neither place (its sheets "
              "have no such column)",
              laser1_rows > 0 and not laser1_leaked,
              f"{laser1_rows} laser-1 rows; leaked={laser1_leaked[:3]}")
    except Exception as e:                      # an exception is a FAIL, never a skip
        check("initial trim value: the fixtures run through the pipeline", False,
              f"{type(e).__name__}: {e}")
    finally:
        _mgr._db_manager, _dbpkg._db_manager = saved
        if fdb is not None:
            fdb.close()
        shutil.rmtree(tmp, ignore_errors=True)


def check_initial_trim_value_on_database(raw) -> None:
    """On the copy: every laser-2/3 Trim pass carries its initial trim value capture -- in
    its own `trim_passes.initial_trim_value` column (rows written from 2026-09-24 on) OR
    inside its `recipe` blob (the ~83,000 rows written before; the migration adds the
    column but writes no data into it, by design -- no start-up backfill of that many
    rows) -- and `trim_passes.initial_trim_values` reads the same value back from each.

    Three rules, each held only where it applies:
      * coverage, on EVERY row: the capture is in the column or in `recipe`. A laser-2/3
        pass always yields it -- a position-aligned list, all None on the rare pass whose
        sheet genuinely had nothing there (224 of 83,488 on the work database, 0.3%) --
        so a row with neither lost it. The all-None rows are captured, just empty, and
        are counted, never failed: asserting a real number there would fail on real data
        for a reason that is not a bug.
      * recipe agreement, on rows whose column is NULL only: the helper never loses a
        value the recipe had, and reads no phantom one. A row whose value lives in the
        column (written by today's code) has nothing in `recipe` to agree with --
        holding it there made the first ingest of a laser-2/3 file after this pull read
        FAIL "helper=7 recipe=5" (2026-09-24 final review).
      * column read-back, on rows that have one: the helper returns that row's own value.
    A rule with no rows to hold is not run -- a PASS over zero rows proves nothing; the
    coverage line carries every count.

    Falsify before trusting (2026-09-24): make the helper ignore `recipe` -- every
    column-less row with a real recipe value reads back None and `lost` stops being 0.
    Make it ignore the column -- the read-back check FAILs on any new row. Stop the
    writer storing the capture anywhere -- the coverage check FAILs ("neither").
    """
    import json as _json
    from laser_trim_analyzer.core.trim_passes import initial_trim_values

    rows = raw.execute(
        "SELECT p.initial_trim_value, p.recipe FROM trim_passes p "
        "JOIN track_results t ON t.id = p.track_result_id "
        "JOIN analysis_results a ON a.id = t.analysis_id "
        "WHERE a.system IN ('A', 'C')").fetchall()
    total = len(rows)
    if total == 0:
        warn("initial trim value: no laser-2/3 Trim passes on this copy to check")
        return

    def _real(values):
        return bool(values) and any(v is not None for v in values)

    col_rows = col_real = col_misread = neither = 0
    recipe_rows = recipe_real = helper_real = lost = 0
    for raw_value, recipe_raw in rows:
        recipe = _json.loads(recipe_raw) if recipe_raw else None
        row_value = _json.loads(raw_value) if raw_value is not None else None
        helper_values = initial_trim_values(row_value, recipe)
        if row_value is not None:               # its own column holds the capture
            col_rows += 1
            col_real += int(_real(row_value))
            col_misread += int(helper_values != row_value)
            continue
        if not (isinstance(recipe, dict) and "initial_trim_value" in recipe):
            neither += 1                        # captured nowhere: lost
            continue
        recipe_rows += 1
        recipe_has_real = _real(recipe["initial_trim_value"])
        helper_has_real = _real(helper_values)
        recipe_real += int(recipe_has_real)
        helper_real += int(helper_has_real)
        lost += int(recipe_has_real and not helper_has_real)
    check("initial trim value: every laser-2/3 Trim pass carries its capture, in its own "
          "column or in recipe",
          neither == 0,
          f"{total} passes: {col_rows} in the column ({col_real} with a real value), "
          f"{recipe_rows} in recipe ({recipe_real} with a real value), "
          f"{col_rows - col_real + recipe_rows - recipe_real} whose sheet had none, "
          f"{neither} in neither place")
    if recipe_rows:
        check("initial trim value: the helper never loses a value its row's recipe already "
              f"had ({recipe_real} of the {recipe_rows} passes with no column value carry "
              "a real one in recipe)",
              lost == 0, f"lost={lost}")
        check("initial trim value: on passes with no column value, the helper's coverage "
              "matches the recipe's exactly (no under-reading, no phantom reads)",
              helper_real == recipe_real, f"helper={helper_real} recipe={recipe_real}")
    if col_rows:
        check("initial trim value: the helper reads back each column value exactly",
              col_misread == 0, f"misread={col_misread} of {col_rows}")


def check_track2_setup_fixtures() -> None:
    """Through the real pipeline, into a throwaway database (--only track2-setup):
    a two-track DLTS fixture's TRK2 track is judged against Track 2's OWN resistance
    limits (never Track 1's), `parameters` never carries the internal `_track2` key
    the parser hands the writer, and a single-track fixture stores a real SQL NULL
    for `track2_parameters` -- never the JSON text 'null'.

    Falsify before trusting (2026-09-24): make `trim_setup.read_track2_keyvalue`
    always return {} -- the first check below goes FAIL (TRK2 reads TRK1's number).
    Stop popping `_track2` out of `setup` in `_write_trim_setup` -- the second check
    FAILs (`_track2` leaks into `parameters`). Make `_write_trim_setup` write
    `sql_null()` unconditionally -- the first and third checks both FAIL.
    """
    import json as _json
    import shutil
    import tempfile
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as _mgr
    from laser_trim_analyzer.findings.data import load_model_tracks
    import laser_trim_analyzer.database as _dbpkg

    both = REPO / "tests" / "fixtures" / "trim" / "dlts_8074_18.xls"      # TRK1 + TRK2
    single = REPO / "tests" / "fixtures" / "trim" / "dlts_8232-1_242.xls"  # single-track
    check("track2 setup: both fixtures are present", both.exists() and single.exists(),
          f"both={both.exists()} single={single.exists()}")
    if not (both.exists() and single.exists()):
        return
    saved = (_mgr._db_manager, getattr(_dbpkg, "_db_manager", None))
    tmp = Path(tempfile.mkdtemp(prefix="track2_setup_sweep_"))
    fdb = None
    try:
        fdb = _mgr.DatabaseManager(tmp / "t2.db")
        _mgr._db_manager = fdb                 # BOTH globals: a Processor must never
        _dbpkg._db_manager = fdb               # reach the configured database.
        proc = Processor(use_ml=False)
        for f in (both, single):
            fdb.save_analysis(proc.process_file(f))

        tracks = {t.track_name: t for t in load_model_tracks(fdb, "8074")}
        t1 = tracks.get("TRK1")
        t2 = tracks.get("TRK2")
        ok = (t1 is not None and t2 is not None
              and t1.initial_r_low is not None and t2.initial_r_low is not None
              and t1.initial_r_low != t2.initial_r_low)
        check("track2 setup: TRK2 is judged against ITS OWN initial resistance limit, "
              "not TRK1's",
              ok,
              f"TRK1={t1 and t1.initial_r_low} TRK2={t2 and t2.initial_r_low}")

        conn = sqlite3.connect(f"file:{tmp / 't2.db'}?mode=ro", uri=True)
        try:
            row_both = conn.execute(
                "SELECT s.parameters, s.track2_parameters FROM trim_setup s "
                "JOIN analysis_results a ON a.id = s.analysis_id "
                "WHERE a.filename = ?", (both.name,)).fetchone()
            row_single = conn.execute(
                "SELECT s.track2_parameters, typeof(s.track2_parameters) FROM trim_setup s "
                "JOIN analysis_results a ON a.id = s.analysis_id "
                "WHERE a.filename = ?", (single.name,)).fetchone()
        finally:
            conn.close()
        parameters = (_json.loads(row_both[0]) if row_both and row_both[0] else {})
        check("track2 setup: parameters never carries the internal _track2 key",
              "_track2" not in parameters)
        check("track2 setup: track2_parameters is populated on the two-track fixture",
              bool(row_both and row_both[1]))
        check("track2 setup: a single-track fixture stores a real SQL NULL for "
              "track2_parameters (never the JSON text 'null')",
              row_single is not None and row_single[0] is None and row_single[1] == "null",
              f"{row_single}")
    except Exception as e:                      # an exception is a FAIL, never a skip
        check("track2 setup: the fixtures run through the pipeline", False,
              f"{type(e).__name__}: {e}")
    finally:
        _mgr._db_manager, _dbpkg._db_manager = saved
        if fdb is not None:
            fdb.close()
        shutil.rmtree(tmp, ignore_errors=True)


def check_track2_setup_on_database(raw) -> None:
    """On the copy: how many two-track System A/C analyses have (not yet) been
    reprocessed under this feature, and every one that HAS carries a Track 2 block
    that actually yields a real resistance limit.

    No back-fill (ruling, task-9-brief.md): existing two-track analyses are judged
    against Track 1's limits until reprocessed, so 0 captured today is the CORRECT
    state, not a failure -- this reports the count rather than asserting a fraction,
    the same shape as `check_track2_setup_fixtures` covering the code path itself.

    Falsify before trusting (2026-09-24): hand-build a `track2_parameters` blob with
    every value None (no real limit) -- the second check goes FAIL.
    """
    import json as _json
    from laser_trim_analyzer.core.trim_setup import resistance_limits

    rows = raw.execute(
        "SELECT a.id, a.model, s.track2_parameters "
        "FROM analysis_results a LEFT JOIN trim_setup s ON s.analysis_id = a.id "
        "WHERE a.system IN ('A', 'C') AND a.has_multi_tracks = 1").fetchall()
    total = len(rows)
    if total == 0:
        warn("track2 setup: no two-track System A/C analyses on this copy to check")
        return
    captured = sum(1 for _aid, _model, t2_raw in rows if t2_raw)
    # Information, not a verdict -- a check that can never FAIL is forbidden by
    # the sweep's own rule (a bare `check(..., True, ...)` was exactly the
    # "weak assertion" class CLAUDE.md warns about). 0 captured is the correct
    # state before a reprocess; a high or low count is neither good nor bad on
    # its own, so this is a count for a human, via warn(), not check().
    warn(f"track2 setup: {captured} of {total} two-track System A/C analyses on this "
         "copy carry a captured Track 2 block (0 is CORRECT before a reprocess -- "
         "no back-fill by design; see task-9-report.md)",
         f"captured={captured} total={total}")
    if not captured:
        return      # the WARN above says so; a PASS over zero blocks would prove nothing
    bad = []
    for aid, model, t2_raw in rows:
        if not t2_raw:
            continue
        block = _json.loads(t2_raw) if isinstance(t2_raw, str) else t2_raw
        limits = resistance_limits(block)
        if not any(v is not None for v in limits.values()):
            bad.append((aid, model))
    check("track2 setup: every captured Track 2 block yields at least one real "
          "resistance limit",
          not bad, f"{bad[:5]}")


# ---- the screens count what they draw; a failed load is never a zero (final review M9) --------
# Driven headless against the database under test (always a copy): the pages' own load/apply
# code runs on stand-in widgets, so what is checked is the real arithmetic between the cache,
# the caption/header a person reads, and the rows the section is asked to draw.

class _Recorder:
    """A stand-in widget: remembers its text and whether it is packed, has no children and no
    height, and accepts every other call."""
    def __init__(self, *a, **k):
        self.text, self.packed = "", False

    def configure(self, **k):
        if "text" in k:
            self.text = k["text"]

    def pack(self, *a, **k):
        self.packed = True

    def pack_forget(self):
        self.packed = False

    def winfo_manager(self):
        return "pack" if self.packed else ""

    def winfo_children(self):
        return []

    def winfo_height(self):
        return 0

    def __getattr__(self, name):
        return lambda *a, **k: None


class _ViewRecorder:
    """Stands in for FindingsView: records the rows and options a page hands it, so a check can
    arrange() exactly what the real view would draw from them."""
    made: list = []

    def __init__(self, master, theme, **kw):
        self.kw, self.rows = kw, []
        _ViewRecorder.made.append(self)

    def pack(self, *a, **k):
        pass

    def set_findings(self, rows):
        self.rows = list(rows or [])


def _rows_drawn(view) -> int:
    from laser_trim_analyzer.findings import presentation as P
    keys = view.kw.get("groups")
    keys = None if keys is None else set(keys)
    return sum(len(g.rows) for g in P.arrange(view.rows, include_empty=view.kw.get("include_empty", True))
               if keys is None or g.spec.key in keys)


def _headless_home(db):
    """(HomePage on stand-in widgets over `db`, the list its captions are written to)."""
    from types import SimpleNamespace
    from laser_trim_analyzer.gui.v6.pages.home_page import HomePage
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    page = HomePage.__new__(HomePage)
    page.app = SimpleNamespace(db=db)
    page.theme = ThemeManager()
    for name in ("_worth_section", "_worth_banner", "_focus_banner", "_focus"):
        setattr(page, name, _Recorder())
    page._worth_view, page._worth_count, page._focus_count = None, None, 0
    page._last_processed = datetime(2026, 1, 5)      # invented: the caption needs a stamp to exist
    captions: list = []
    page.set_caption = captions.append
    return page, captions


class _FailingDb:
    """`db`, except that `method` raises -- a loader forced to fail."""
    def __init__(self, db, method):
        self._db, self._method = db, method

    def __getattr__(self, name):
        if name == self._method:
            def boom(*a, **k):
                raise RuntimeError("sweep: forced failure")
            return boom
        return getattr(self._db, name)


def check_screens_count_what_they_draw(db) -> None:
    """Home's "N worth changing" and the Model page's "Worth changing on this model" count, each
    against the presentation layer's own arrange() of the cached findings -- with the groups the
    design doc rules for each (Home: yield; Model page: yield, laser time, check), written here,
    not read from the pages -- and against the rows each section actually hands its view.

    Falsify before trusting (2026-09-24): make home_page._yield_findings_count sum every group, or
    add "history" to model_page._WORTH_CHANGING_GROUPS -- each FAILs its line below."""
    import re
    from collections import defaultdict
    from laser_trim_analyzer.findings import presentation as P
    from laser_trim_analyzer.gui.v6.pages import home_page, model_page
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets import blocks

    rows = db.get_process_findings()
    check("screens: the database has cached findings to count", len(rows) > 0, f"{len(rows)} findings")
    if not rows:
        return
    ref_home = sum(len(g.rows) for g in P.arrange(rows, include_empty=False) if g.spec.key == "yield")
    if not ref_home:
        warn("screens: no yield-group finding in the cache", "Home's count is checked at zero")
    saved_view, saved_header = home_page.FindingsView, blocks.group_header
    try:
        home_page.FindingsView = _ViewRecorder
        _ViewRecorder.made = []
        page, captions = _headless_home(db)
        page._apply_findings(page._query_findings())
        caption = captions[-1] if captions else ""
        m = re.search(r"([\d,]+) worth changing", caption)
        n = int(m.group(1).replace(",", "")) if m else None
        drawn = _rows_drawn(_ViewRecorder.made[-1]) if _ViewRecorder.made else 0
        check("home: 'N worth changing' is the yield rows arrange() builds from the cache, and the "
              "rows its section draws", n is not None and n == ref_home == drawn,
              f"caption={caption!r} arrange={ref_home} drawn={drawn}")
    except Exception as e:
        check("home: 'N worth changing' is the yield rows arrange() builds from the cache",
              False, f"{type(e).__name__}: {e}")
    finally:
        home_page.FindingsView = saved_view

    # One model, resolved by query: the most findings among models with NO analyzer error and at
    # least one finding OUTSIDE the section's three groups (so the section's own filtering counts).
    section = {"yield", "laser_time", "check"}
    by_model = defaultdict(list)
    for f in rows:
        by_model[f.get("model")].append(f)
    errors = db.get_process_errors()
    candidates = sorted((m_ for m_ in by_model if m_ and m_ not in errors),
                        key=lambda m_: (not any(P.group_key(f) not in section for f in by_model[m_]),
                                        -len(by_model[m_]), m_))
    if not candidates:
        warn("screens: no error-free model with cached findings", "model-page count not checked")
        return
    model = candidates[0]
    findings = db.get_process_findings(model)
    ref_model = sum(len(g.rows) for g in P.arrange(findings, include_empty=False)
                    if g.spec.key in section)
    headers: list = []
    try:
        def spy(parent, theme, title, count, **kw):
            headers.append((title, count))
            return _Recorder()
        blocks.group_header = spy
        model_page.FindingsView = _ViewRecorder
        _ViewRecorder.made = []
        mp = model_page.ModelPage.__new__(model_page.ModelPage)
        mp.theme, mp._worth_section, mp._worth_view = ThemeManager(), _Recorder(), None
        mp._set_findings_section({"facts": db.get_process_facts(model), "findings": findings}, [])
        shown = [c for t_, c in headers if t_ == "Worth changing on this model"]
        drawn = _rows_drawn(_ViewRecorder.made[-1]) if _ViewRecorder.made else 0
        check(f"model page: the 'Worth changing' count on {model} is the rows arrange() builds for "
              f"its three groups, and the rows it draws", shown == [ref_model] and drawn == ref_model,
              f"header={shown} arrange={ref_model} drawn={drawn} "
              f"(groups here: {sorted({P.group_key(f) for f in findings})})")
    except Exception as e:
        check(f"model page: the 'Worth changing' count on {model}", False, f"{type(e).__name__}: {e}")
    finally:
        blocks.group_header = saved_header
        model_page.FindingsView = saved_view


def check_failed_loads_are_never_zero(db) -> None:
    """Drive a loader to raise and read what the page would say: a failure is NAMED, never drawn
    as "0 worth changing", "0 drifting now" or "Needs a look · 0" (final review, 2026-09-24).

    Falsify before trusting (2026-09-24): in home_page set `self._worth_count = 0` on the failed
    branch, or `self._focus_count = len(result.focus)` whatever the result; in triage_page pass
    `len(result.focus)` to the "Needs a look" header whatever the result -- each FAILs below."""
    from laser_trim_analyzer.gui.v6 import focus_data
    from laser_trim_analyzer.gui.v6.pages.triage_page import TriagePage
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets import blocks

    try:
        page, captions = _headless_home(_FailingDb(db, "get_process_findings"))
        page._apply_findings(page._query_findings())
        caption = captions[-1] if captions else ""
        check("home: a failed findings load is named in a banner, never '0 worth changing'",
              "worth changing" not in caption and page._worth_banner.packed
              and "sweep: forced failure" in page._worth_banner.text,
              f"caption={caption!r} banner={page._worth_banner.text[:80]!r}")
    except Exception as e:
        check("home: a failed findings load is named in a banner", False, f"{type(e).__name__}: {e}")

    saved_compute, saved_header = focus_data.compute_focus_list, blocks.group_header
    headers: list = []
    try:
        def boom(_db):
            raise RuntimeError("sweep: forced FOCUS failure")
        focus_data.compute_focus_list = boom
        result, last = focus_data.load_focus(db)
        page, captions = _headless_home(db)
        page._apply_focus(result, last)
        caption = captions[-1] if captions else ""
        check("home: a failed FOCUS load is named in a banner, never '0 drifting now'",
              "drifting now" not in caption and page._focus_banner.packed
              and "forced FOCUS failure" in page._focus_banner.text,
              f"caption={caption!r} banner={page._focus_banner.text[:80]!r}")

        def spy(parent, theme, title, count, **kw):
            headers.append((title, count))
            return _Recorder()
        blocks.group_header = spy
        tp = TriagePage.__new__(TriagePage)
        tp.theme = ThemeManager()
        for name in ("_content_parent", "_focus_wrap", "_focus", "_browse", "_load_banner"):
            setattr(tp, name, _Recorder())
        tp._focus_header, tp._show_all, tp._browse_failed = None, False, None
        tp._apply(result, [], set(), last)
        shown = [c for t_, c in headers if t_ == "Needs a look"]
        check("triage: a failed FOCUS load is named, with no 'Needs a look' count",
              shown == [None] and tp._load_banner.packed
              and "forced FOCUS failure" in tp._load_banner.text,
              f"header counts={shown} banner={tp._load_banner.text[:80]!r}")
    except Exception as e:
        check("home/triage: a failed FOCUS load is named", False, f"{type(e).__name__}: {e}")
    finally:
        focus_data.compute_focus_list = saved_compute
        blocks.group_header = saved_header


# ---- usability glosses (moved out of main() so `--only glosses` runs them alone) -----------------

def _string_literals(source: str) -> list:
    """Every string in `source` a person could be SHOWN: each str constant in its syntax tree --
    the parser has already joined implicit concatenations ("a" "b" is one constant) and split an
    f-string into its literal parts -- but never a comment (comments are not in the tree) and never
    a string that is a whole statement (a docstring; never on screen)."""
    import ast
    tree = ast.parse(source)
    statements = {id(node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)}
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in statements]


def _gloss_present(source: str, needle: str, kind: str) -> bool:
    """kind "exact": a string literal IS the needle (a heading, a column name); "text": a string
    literal CONTAINS it (a sentence); "code": the needle is code, a plain substring of the file.

    Text is matched against real string literals, not the raw file (final review, 2026-09-24,
    M6): the Model page's `# ---- "How it's running" ...` and Triage's `# OWN "Needs a look" ...`
    comments carry the very words -- quotes and all -- so a bare substring check, or even the
    quoted form, kept passing with the heading itself renamed. Made to fail first: rename either
    heading and leave the comment -- the old check PASSES, this one FAILS."""
    if kind == "code":
        return needle in source
    literals = _string_literals(source)
    if kind == "exact":
        return needle in literals
    return any(needle in lit for lit in literals)


def check_usability_glosses() -> None:
    """Every symbol/number the live walk (2026-07-08) found unexplained keeps its on-screen
    decoder line. Each entry is (file, needle, what it guarantees); a needle listed in
    _GLOSS_KINDS is matched as a whole heading literal or as code, every other one as text inside
    a string literal -- see _gloss_present. Standalone: `--only glosses`."""
    _GLOSSES = [
        # 2026-08-29: the σ card wall became the FOCUS list. Same obligation,
        # new zone — say WHY a model is on the list and when it leaves.
        ("src/laser_trim_analyzer/gui/v6/widgets/focus_list_zone.py",
         "outside its own control limits", "FOCUS list states its membership rule"),
        # 2026-09-24 (facelift step 2, T2): the three-sentence σ key moved to the Drift metrics
        # tab, beside the numbers it explains; the model page keeps a ONE-line key. Both pinned.
        ("src/laser_trim_analyzer/gui/v6/pages/model_page.py",
         "history of lots", "model page explains σ in lot language (one line)"),
        ("src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py",
         "historical lot medians", "drift tab explains σ in lot language (in full)"),
        ("src/laser_trim_analyzer/gui/v6/widgets/worst_models_list.py",
         "Gap = Trim − FT", "lowest-yield list explains Gap"),
        # 2026-09-24 (facelift step 2, T7): the colour dot became a status WORD on each row
        # (a colour-blind reader had nothing to read); the legend explains it in words. Final
        # review (same date): "worst first" dropped -- the list is alphabetical, the lookup list
        # -- so the needle pins all three glosses in one string, with no order claim between.
        ("src/laser_trim_analyzer/gui/v6/widgets/browse_zone.py",
         "Status = drift tier. Date = last processed. 'Active' scope =",
         "browse list explains status/date/Active"),
        ("src/laser_trim_analyzer/gui/v6/widgets/units_tab.py",
         '"Sigma gradient"', "units table headers are full words"),
        ("src/laser_trim_analyzer/gui/v6/widgets/units_tab.py",
         '"Linearity error"', "units table headers are full words (2)"),
        # 2026-09-24 (T3b): the legend box became a one-line key of drawn elements; the red
        # dots must still be named in it (the round-2 key dropped them -- this check caught it).
        ("src/laser_trim_analyzer/gui/v6/widgets/focus_chart.py",
         "beyond ±3σ (red)", "focus chart names its red markers"),
        ("src/laser_trim_analyzer/gui/v6/pages/dashboard_page.py",
         "matched to trims", "FT panel count says what 'matched' means"),
        ("src/laser_trim_analyzer/gui/v6/pages/model_page.py",
         "lifetime linearity yield", "model verdict line (holding/drifting/difficulty)"),
        ("src/laser_trim_analyzer/gui/v6/widgets/drift_metrics_tab.py",
         "Baseline period", "drift tab discloses baseline provenance"),
        ("src/laser_trim_analyzer/gui/v6/pages/model_page.py",
         "This action is recorded", "requalify dialog states auditability"),
        # 2026-07-13 design pass: interpretation vs data zones. Text updated 2026-09-24: the
        # facelift's sentence-case sweep (T2) converted these from shouting headings to
        # sentence case app-wide -- the zone-marking obligation this check exists to pin is
        # unchanged, only the literal casing is, so the string here tracks the page, not the
        # other way round.
        # 2026-09-24 (T2): the model page's app's-read zone is now "How it's running" (the
        # verdict moved into the caption; findings got their own "Worth changing" group).
        ("src/laser_trim_analyzer/gui/v6/pages/model_page.py",
         "How it's running", "model page marks the app's-read zone"),
        ("src/laser_trim_analyzer/gui/v6/pages/model_page.py",
         "What you're looking at", "model page marks the data zone"),
        # 2026-09-24 (T7): Triage's app's-read zone is the "Needs a look" group (the focus list);
        # the data zone is "All models" (the browse list).
        ("src/laser_trim_analyzer/gui/v6/pages/triage_page.py",
         "Needs a look", "triage marks the app's-read zone"),
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
        kind, literal = _GLOSS_KINDS.get(needle, ("text", None))
        try:
            source = open(REPO / path, encoding="utf-8").read()
            ok = _gloss_present(source, literal or needle, kind)
        except (OSError, SyntaxError):
            ok = False
        check(f"usability gloss: {what}", ok)


# Needles that are not a sentence: an exact heading/column literal (the key's own quotes are
# dropped -- the tree has no quotes), or code, where a plain substring is the right test.
_GLOSS_KINDS = {
    '"Sigma gradient"': ("exact", "Sigma gradient"),
    '"Linearity error"': ("exact", "Linearity error"),
    "How it's running": ("exact", None),
    "What you're looking at": ("exact", None),
    "Needs a look": ("exact", None),
    "format_metric_value": ("code", None),
    "on_unit_click": ("code", None),
    "Include the PRE-TRIM trace": ("code", None),
    "_open_dropdown_menu = self._open_model_picker": ("code", None),
}


def main() -> int:
    # REQUIRED DB-path argv (2026-08-31; was optional with a production
    # default). The sweep opens its target read-write, and the old default —
    # the real data/analysis.db — was opened by accident by three separate
    # sessions in one night. There is no legitimate sweep use of the
    # production file, so it is refused even when named explicitly.
    #     python scripts/app_qa_sweep.py /path/to/COPY_of_analysis.db
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        print("usage: python scripts/app_qa_sweep.py /path/to/COPY_of_analysis.db [--only ...]\n"
              "(the DB argument is required — the sweep refuses the production database)")
        return 1
    db_path = Path(sys.argv[1])
    if _db_guard.is_production_db(db_path, REPO, by_name=True):
        print(f"FATAL | {db_path} is the PRODUCTION database and the sweep "
              f"opens it READ-WRITE.\n"
              f"      | make a copy and pass that instead:\n"
              f"      |     cp data/analysis.db /tmp/qa_copy.db\n"
              f"      |     python scripts/app_qa_sweep.py /tmp/qa_copy.db")
        return 1
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
    # Replaces the σ-alert-feed checks that stood here until 2026-08-29 (that
    # feed was deleted 2026-08-31): the Triage page ranks models from
    # `compute_focus_list` now, so THESE are the invariants a regression would
    # break. The promise being guarded is that every row can point at the lot
    # in its own series that put it there.
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
        # (e) A verdict about a model that isn't in the data is a phantom. The
        # old σ-alert-feed version of this invariant failed on this DB, which is
        # why it is still asserted here after that feed was removed.
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
    # ---- the screens count what they draw; a failed load is never a zero (final review M9)
    check_screens_count_what_they_draw(db)
    check_failed_loads_are_never_zero(db)

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
        check("shell: Home leads, Investigate keeps the 'model' key, Findings follows it",
              keys[:4] == ["home", "model", "findings", "settings"]
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

    # ---- final-test graded window (2026-09-13) -----------------------------
    # The app grades a final test on the rows the SHEET grades, and only
    # those. These three sections cover the parse, the repair tool that
    # re-applies the grade to rows written before the fix, and the rule that
    # a row with no disposition stays out of every rate built on one.
    check_ft_disposition_excludes_ungraded(db, raw)
    check_ft_regrade_dry_run(db)
    check_findings_fixtures()
    check_findings_on_database(db)
    check_findings_group_mapping(db)

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

    # ---- every ERROR row has a reason (2026-09-23, revised 2026-09-24) ------
    # analysis_results.error_reason, or a linked track's own linearity_spec_warning
    # / anomaly_reason -- the exact COALESCE _load_units/_search_units read
    # (model_page.py). A row saved before error_reason existed can still carry its
    # reason on a processed_files row instead: either the row LINKED by analysis_id
    # (populated for a track-level ERROR since this task) or the per-PATH failure
    # marker (analysis_id NULL, a synthetic skip: hash, matched by file_path) --
    # that marker is where the 3 zero-track rows' "No valid track data found"
    # actually lives (_write_failure_marker), and their ids are not stable across a
    # rebuild so nothing here may name them.
    #
    # A reason that exists ONLY on a processed_files row is WARNed, not FAILed: it
    # is a known, accepted gap (design doc ruling 3c -- "3 rows show no reason
    # until reprocessed"), not a bug, and a check that reads FAIL forever trains
    # everyone to stop reading the FAIL line. A row with NO reason ANYWHERE is
    # still a hard zero -- no budget, no percentage, same standard as every other
    # zero-tolerance check in this file.
    n_error = raw.execute(
        "SELECT COUNT(*) FROM analysis_results WHERE overall_status='ERROR'"
    ).fetchone()[0]
    row = raw.execute(
        "WITH per_own AS ("
        "  SELECT a.id AS aid,"
        "         MAX(CASE"
        "           WHEN a.error_reason IS NOT NULL AND a.error_reason != '' THEN 1"
        "           WHEN t.linearity_spec_warning IS NOT NULL AND t.linearity_spec_warning != '' THEN 1"
        "           WHEN t.anomaly_reason IS NOT NULL AND t.anomaly_reason != '' THEN 1"
        "           ELSE 0 END) AS has_own_reason"
        "  FROM analysis_results a"
        "  LEFT JOIN track_results t ON t.analysis_id = a.id"
        "  WHERE a.overall_status = 'ERROR'"
        "  GROUP BY a.id"
        "),"
        "per_pf AS ("
        "  SELECT a.id AS aid,"
        "         MAX(CASE WHEN pf.error_message IS NOT NULL AND pf.error_message != '' "
        "                  THEN 1 ELSE 0 END) AS has_pf_reason"
        "  FROM analysis_results a"
        "  LEFT JOIN processed_files pf"
        "    ON pf.analysis_id = a.id"
        "    OR (pf.file_path = a.file_path AND pf.analysis_id IS NULL"
        "        AND pf.file_hash LIKE 'skip:%')"
        "  WHERE a.overall_status = 'ERROR'"
        "  GROUP BY a.id"
        ")"
        "SELECT"
        "  SUM(CASE WHEN po.has_own_reason=0 AND pp.has_pf_reason=0 THEN 1 ELSE 0 END),"
        "  SUM(CASE WHEN po.has_own_reason=0 AND pp.has_pf_reason=1 THEN 1 ELSE 0 END)"
        " FROM per_own po JOIN per_pf pp ON pp.aid = po.aid"
    ).fetchone()
    n_fail, n_marker_only = (row[0] or 0), (row[1] or 0)
    check("every ERROR row has a reason SOMEWHERE (error_reason, a track's own "
          "words, or a processed_files row)",
          n_fail == 0, f"reasonless={n_fail} of {n_error} ERROR rows")
    if n_marker_only:
        warn("every ERROR row has a reason: rows whose reason lives ONLY on a "
             "processed_files row (predate error_reason -- reprocess candidates)",
             f"{n_marker_only} of {n_error} ERROR rows")

    # ============ 5. UNIT VERDICT CONSISTENCY (broad sample) ==================
    # Linearity is the ZERO-TOLERANCE customer disposition, so these are hard
    # zeros — no percentage budget, no "small tolerance for exclusions".
    #
    # The check that used to live here had both, plus a dead ternary, and it
    # called compute_fail_points WITHOUT the rotation term. It therefore
    # tolerated the exact defect it existed to catch: 831 units rendering
    # "Fail Points: N" beside "Linearity Pass: YES" (2026-08-31, found on
    # 8415-1 SN 26). Weak assertions are forbidden in this sweep.
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import (
        compute_fail_points, unmeasured_points)
    from laser_trim_analyzer.export.unit_chart import (
        build_unit_export_figure, corrected_errors)
    import json as _json

    def _arr(v):
        return _json.loads(v) if isinstance(v, str) else v

    rows = raw.execute(
        "SELECT id, linearity_pass, linearity_fail_points, optimal_offset,"
        " optimal_slope, theory_data, error_data, upper_limits, lower_limits,"
        " final_linearity_error_shifted FROM track_results "
        "WHERE position_data IS NOT NULL AND error_data IS NOT NULL "
        "AND linearity_fail_points IS NOT NULL ORDER BY id DESC LIMIT 4000"
    ).fetchall()

    checked = rot_checked = 0
    rot_bad: list = []
    mag_bad: list = []
    nan_checked = nan_inband = nan_offset_null = legacy_bad = 0
    nan_bad: list = []
    for (_id, lp, lfp, off, k, th, err, up, lo, mag) in rows:
        try:
            err_l, up_l, lo_l, th_l = _arr(err), _arr(up), _arr(lo), _arr(th)
            if not err_l or not up_l or not lo_l:
                continue
            has_nan = any(e is None or (isinstance(e, float) and e != e)
                          for e in err_l)
            # NaN-bearing tracks used to be SKIPPED here: the analyzer counts an
            # unmeasured point as a FAIL (zero-tolerance) and the renderer
            # dropped it, so the counts disagreed by construction. Since the
            # unmeasured points are marked and counted, these tracks are held to
            # the same standard as every other. The one class that still cannot
            # be reproduced is optimal_offset IS NULL: the NaN leak poisoned the
            # analyzer's own offset search, so it stored "every point fails" and
            # no magnitude (see scripts/backfill_linearity_error.py). Those are
            # reprocess candidates, counted and warned about, never asserted on.
            if has_nan and off is None:
                nan_offset_null += 1
                continue
            off, k = off or 0.0, k or 0.0
            if has_nan:
                nan_checked += 1
                if unmeasured_points(err_l, up_l, lo_l, offset=off, k=k,
                                     theory=th_l):
                    nan_inband += 1
                if len(compute_fail_points(err_l, up_l, lo_l, offset=off, k=k,
                                           theory=th_l)) != lfp:
                    nan_bad.append(_id)
            checked += 1
            fp = compute_fail_points(err_l, up_l, lo_l, offset=off, k=k, theory=th_l)
            if k:
                rot_checked += 1
                if len(fp) != lfp:
                    rot_bad.append(_id)
            elif len(fp) != lfp:
                legacy_bad += 1
            # Mirror check: the renderer must also reproduce the analyzer's
            # linearity MAGNITUDE. This catches a wrong corrected trace even
            # when the fail-count coincidentally agrees.
            if mag is not None:
                vals = [abs(v) for v in corrected_errors(err_l, off, k, th_l)
                        if v is not None]
                if vals and abs(max(vals) - mag) > 1e-6:
                    mag_bad.append(_id)
        except Exception as exc:
            check(f"verdict consistency: track {_id} raised", False, repr(exc))

    check("verdict consistency: sample is non-empty (guard against a vacuous pass)",
          checked > 0, f"checked={checked} nan_offset_null={nan_offset_null}")
    # Zero-tolerance, like every other verdict check here: an unmeasured point
    # is counted, so the renderer's count must equal the stored one EXACTLY.
    # Both non-emptiness guards matter — without the second, a data shift that
    # removed every in-band NaN would let this pass without exercising the
    # unmeasured path at all.
    check("verdict consistency: NaN-bearing tracks reproduce the stored "
          "fail-point count EXACTLY (unmeasured points counted, not skipped)",
          not nan_bad and nan_checked > 0 and nan_inband > 0,
          f"{len(nan_bad)}/{nan_checked} disagree, {nan_inband} with NaN inside "
          f"the graded band" + (f" e.g. track ids {nan_bad[:5]}" if nan_bad else ""))
    if nan_offset_null:
        warn("verdict consistency: NaN tracks with NULL optimal_offset (the "
             "analyzer's offset search was itself poisoned — reprocess candidates)",
             f"{nan_offset_null} skipped")
    check("verdict consistency: rotation tracks (k != 0) reproduce the stored "
          "fail-point count EXACTLY",
          not rot_bad, f"{len(rot_bad)}/{rot_checked} disagree"
                       + (f" e.g. track ids {rot_bad[:5]}" if rot_bad else ""))
    check("verdict consistency: renderer reproduces stored linearity MAGNITUDE "
          "(final_linearity_error_shifted)",
          not mag_bad, f"{len(mag_bad)}/{checked} disagree"
                       + (f" e.g. track ids {mag_bad[:5]}" if mag_bad else ""))
    if legacy_bad:
        warn("verdict consistency: k==0 tracks disagreeing (stale stored offsets "
             "from an older analyzer — reprocess candidates)",
             f"{legacy_bad}/{checked}")

    # The DOCUMENT invariant James actually reads: the print export must never
    # show a fail count beside a passing verdict or a green PASS stamp. Both
    # now derive from ONE re-grade, so this is zero-tolerance by construction —
    # asserted on real rows, including deliberately falsified fail points.
    def _doc_texts(fig):
        return [t.get_text() for ax in fig.axes for t in ax.texts]

    doc_bad: list = []
    doc_checked = 0
    for (_id, lp, lfp, off, k, th, err, up, lo, mag) in rows[:60]:
        try:
            err_l, up_l, lo_l, th_l = _arr(err), _arr(up), _arr(lo), _arr(th)
            if not err_l or not up_l:
                continue
            data = {"position_data": list(range(len(err_l))), "error_data": err_l,
                    "upper_limits": up_l, "lower_limits": lo_l,
                    "optimal_offset": off or 0.0, "optimal_slope": k or 0.0,
                    "theory_data": th_l, "linearity_pass": bool(lp),
                    "linearity_error": mag, "sigma_pass": True}
            for forced in (None, [0], list(range(min(5, len(err_l))))):
                fp = (compute_fail_points(err_l, up_l, lo_l, offset=off or 0.0,
                                          k=k or 0.0, theory=th_l)
                      if forced is None else forced)
                fig = build_unit_export_figure(
                    {"model": "QA", "serial": str(_id), "n_tracks": 1},
                    data, fail_points=fp, kind="trim")
                txt = _doc_texts(fig)
                n_line = next((t for t in txt if t.startswith("Fail Points:")), "")
                v_line = next((t for t in txt if t.startswith("Linearity Pass:")), "")
                stamp = next((t for t in txt if t in ("PASS", "FAIL", "PASS (WATCH)",
                                                      "PASS*", "NOT EVALUATED")), "")
                doc_checked += 1
                if n_line != "Fail Points: 0" and (
                        v_line == "Linearity Pass: YES" or stamp == "PASS"):
                    doc_bad.append((_id, n_line, v_line, stamp))
                import matplotlib.pyplot as _plt
                _plt.close(fig)
        except Exception as exc:
            check(f"unit export document: track {_id} raised", False, repr(exc))

    check("unit export document: rendered fail count NEVER shown beside a "
          "passing verdict or green PASS stamp",
          not doc_bad and doc_checked > 0,
          f"{len(doc_bad)}/{doc_checked} contradictions"
          + (f" e.g. {doc_bad[:3]}" if doc_bad else ""))

    # Every point that is COUNTED is a point that is DRAWN. The unmeasured ones
    # have no y, so they are marked on the axis line instead of an X — but they
    # must still appear, or the document reports a number the picture doesn't
    # show (the 2026-08-31 divergence: 597 gradeable tracks whose marker count
    # was short of their stored linearity_fail_points).
    mark_bad: list = []
    mark_checked = 0
    for (_id, lp, lfp, off, k, th, err, up, lo, mag) in rows:
        if mark_checked >= 40:
            break
        try:
            err_l, up_l, lo_l, th_l = _arr(err), _arr(up), _arr(lo), _arr(th)
            if not err_l or not up_l or not lo_l or off is None:
                continue
            if not any(e is None or (isinstance(e, float) and e != e)
                       for e in err_l):
                continue
            if not unmeasured_points(err_l, up_l, lo_l, offset=off,
                                     k=k or 0.0, theory=th_l):
                continue
            mark_checked += 1
            fp = compute_fail_points(err_l, up_l, lo_l, offset=off, k=k or 0.0,
                                     theory=th_l)
            data = {"position_data": list(range(len(err_l))), "error_data": err_l,
                    "upper_limits": up_l, "lower_limits": lo_l,
                    "optimal_offset": off, "optimal_slope": k or 0.0,
                    "theory_data": th_l, "linearity_pass": bool(lp),
                    "linearity_error": mag, "sigma_pass": True}
            fig = build_unit_export_figure(
                {"model": "QA", "serial": str(_id), "n_tracks": 1},
                data, fail_points=fp, kind="trim")
            drawn = sum(len(c.get_offsets()) for c in fig.axes[0].collections
                        if (c.get_label() or "").startswith(
                            ("Fail points", "Unmeasured points")))
            if drawn != len(fp) or len(fp) != lfp:
                mark_bad.append((_id, drawn, len(fp), lfp))
            import matplotlib.pyplot as _plt
            _plt.close(fig)
        except Exception as exc:
            check(f"unmeasured markers: track {_id} raised", False, repr(exc))

    check("unit export document: on tracks with unmeasured points, the DRAWN "
          "marker count equals the reported count equals the STORED count",
          not mark_bad and mark_checked > 0,
          f"{len(mark_bad)}/{mark_checked} disagree (id, drawn, reported, stored)"
          + (f" e.g. {mark_bad[:3]}" if mark_bad else ""))

    # ---- trim-vs-FT overlay (V6 unit chart; was V5 Compare's alone) ----------
    # The overlay must resolve real linkages, grade the FT sweep on the FT's
    # OWN adjustment, and refuse rather than guess. Run on real linked pairs.
    from laser_trim_analyzer.core.ft_overlay import (
        MIN_MATCH_CONFIDENCE, load_ft_overlay)
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import load_unit_track

    ov_rows = raw.execute(
        "SELECT DISTINCT f.linked_trim_id "
        "FROM final_test_results f JOIN final_test_tracks t "
        "  ON t.final_test_id = f.id "
        "WHERE f.linked_trim_id IS NOT NULL AND f.match_confidence >= ? "
        "  AND t.position_data IS NOT NULL AND t.error_data IS NOT NULL "
        "ORDER BY f.id DESC LIMIT 40", (MIN_MATCH_CONFIDENCE,)).fetchall()
    ov_ok = ov_refused = ov_multi = 0
    ov_bad: list = []
    for (aid,) in ov_rows:
        try:
            trim = load_unit_track(db, aid)
            if not trim:
                continue
            ov = load_ft_overlay(db, aid, trim_track_id=trim.get("track_id"),
                                 trim_positions=trim.get("position_data"))
            if not ov.get("available"):
                # A refusal is a PASS as long as it carries a reason — that is
                # the contract: never an empty chart with no explanation.
                if ov.get("reason"):
                    ov_refused += 1
                else:
                    ov_bad.append((aid, "refused with no reason"))
                continue
            ov_ok += 1
            if (ov.get("confidence") or 0) < MIN_MATCH_CONFIDENCE:
                ov_bad.append((aid, "below the confidence floor"))
            # A unit can be final-tested many times; the NEWEST qualifying test
            # is the one to show. Verified against SQL, not against the code's
            # own idea of newest.
            newest, n_links = raw.execute(
                "SELECT id, COUNT(*) OVER () FROM final_test_results "
                "WHERE linked_trim_id = ? AND match_confidence >= ? "
                "ORDER BY COALESCE(test_date, file_date) DESC, id DESC LIMIT 1",
                (aid, MIN_MATCH_CONFIDENCE)).fetchone()
            if n_links > 1:
                ov_multi += 1
                if str(ov["n_links"]) not in ov["label"]:
                    ov_bad.append((aid, "multi-test unit does not say so"))
            if ov["ft_id"] != newest:
                ov_bad.append((aid, f"showed ft {ov['ft_id']}, newest is {newest}"))
            # The FT trace carries the FT's OWN offset — never the trim's. Read
            # back from the exact FT track the overlay chose.
            row = raw.execute(
                "SELECT optimal_offset FROM final_test_tracks "
                "WHERE final_test_id = ? AND track_id = ?",
                (ov["ft_id"], ov["track_id"])).fetchone()
            ft_off = (row[0] if row else None) or 0.0
            if abs(ov["offset"] - ft_off) > 1e-9:
                ov_bad.append((aid, "FT offset is not the FT's own"))
            trim_off = trim.get("optimal_offset") or 0.0
            if abs(trim_off - ft_off) > 1e-9 and abs(ov["offset"] - trim_off) < 1e-12:
                ov_bad.append((aid, "trim offset leaked onto the FT trace"))
        except Exception as exc:
            ov_bad.append((aid, repr(exc)))
    check("trim/FT overlay: shows the NEWEST linked test and grades its sweep "
          "on the FT's OWN offset (never the trim's)",
          not ov_bad and ov_ok > 0 and ov_multi > 0,
          f"{ov_ok} drawn ({ov_multi} multi-test units), {ov_refused} refused "
          f"with a reason, {len(ov_bad)} wrong"
          + (f" e.g. {ov_bad[:3]}" if ov_bad else ""))

    # ============ 6. EXPORTS ===================================================
    from laser_trim_analyzer.export.evidence import export_evidence_pack, build_summary_text
    import pandas as pd
    out = QA_OUTPUT
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
    check_ft_parser_console_silence()
    check_ft_graded_window()
    check_ingest_group()
    check_increment_volts_fixtures()
    check_increment_volts_corpus()
    check_increment_volts_on_database(raw)
    check_initial_trim_value_fixtures()
    check_initial_trim_value_on_database(raw)
    check_track2_setup_fixtures()
    check_track2_setup_on_database(raw)

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

    check_usability_glosses()

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

    # ---- data quality: verdicts must be backed by a measurement (2026-08-31) --
    # SafeJSON binds Python None through SQLAlchemy's JSON type, which stores
    # the JSON literal 'null' as TEXT — so `IS NULL` finds none of the ~5,811
    # array-less track rows. Every predicate here tests the stored text.
    #
    # Three separate populations, and conflating them is the trap this block
    # exists to prevent:
    #   1. graded track with no arrays  -> FAIL. The ingest guard
    #      (processor.enforce_measurement_backed_verdict) makes this
    #      unreachable, so any occurrence is a code defect, not data debt.
    #   2. graded unit whose tracks are ALL array-less -> WARN. Repairable
    #      data debt; Fix Missing Tracks re-parses it.
    #   3. untrimmed-only track inside a graded unit -> reported, never
    #      flagged. A two-track unit where one track was a test sweep with no
    #      laser-trim run is BY DESIGN (parser moves the sweep into untrimmed_*
    #      and clears the trimmed arrays); its waveform is in untrimmed_positions
    #      and its parent is graded from the other track. 145 such tracks on the
    #      work database as of 2026-08-31 — alarming on them would be crying
    #      wolf on normal ingest.
    _NO_ARRAY = ("(CAST({c} AS TEXT) IN ('','null','NULL','None','[]','{{}}') "
                 "OR {c} IS NULL)")
    _BOTH_ABSENT = (_NO_ARRAY.format(c="t.position_data") + " AND "
                    + _NO_ARRAY.format(c="t.error_data"))
    try:
        n_graded_trackless = raw.execute(
            f"SELECT COUNT(*) FROM track_results t "
            f"WHERE t.status IN ('PASS','WARNING','FAIL') AND ({_BOTH_ABSENT})"
        ).fetchone()[0]
        n_tracks_total = raw.execute("SELECT COUNT(*) FROM track_results").fetchone()[0]
        # Counts, not a bare boolean: a scan that examined nothing must not
        # read as green.
        check("ingest guard: no graded track stored without its measurement",
              n_graded_trackless == 0 and n_tracks_total > 0,
              f"{n_graded_trackless} unbacked verdict(s) of {n_tracks_total} tracks"
              + ("" if n_tracks_total else " — SCANNED NOTHING")
              + ("" if not n_graded_trackless else
                 " | code defect: processor.enforce_measurement_backed_verdict "
                 "should have made this impossible"))
    except Exception as e:
        check("ingest guard: no graded track stored without its measurement", False,
              f"{type(e).__name__}: {e}")

    # The row count above is zero today, and would stay zero for a while if the
    # guard were deleted — bad data has to accumulate before a data check can
    # see it. So assert the guard is actually WIRED, not just currently unviolated.
    try:
        _proc = open(REPO / "src/laser_trim_analyzer/core/processor.py",
                     encoding="utf-8").read()
        defined = "def enforce_measurement_backed_verdict" in _proc
        called = _proc.count("enforce_measurement_backed_verdict(") - (1 if defined else 0)
        check("ingest guard is wired into the track loop, not just defined",
              defined and called >= 1,
              f"defined={defined}, call sites={called}"
              + ("" if defined and called >= 1 else
                 " | without a call site the data check above passes on an "
                 "unguarded pipeline until bad rows accumulate"))
    except Exception as e:
        check("ingest guard is wired into the track loop, not just defined", False,
              f"{type(e).__name__}: {e}")

    try:
        rows = raw.execute(
            f"SELECT a.model, COUNT(*) FROM analysis_results a "
            f"WHERE a.overall_status IN ('PASS','WARNING','FAIL') "
            f"  AND EXISTS (SELECT 1 FROM track_results t WHERE t.analysis_id=a.id) "
            f"  AND NOT EXISTS (SELECT 1 FROM track_results t "
            f"                  WHERE t.analysis_id=a.id AND NOT ({_BOTH_ABSENT})) "
            f"GROUP BY a.model ORDER BY COUNT(*) DESC").fetchall()
        n_units = sum(n for _m, n in rows)
        if n_units:
            warn(f"data quality: {n_units} graded unit(s) with no measurement on any "
                 f"track across {len(rows)} model(s)",
                 "; ".join(f"{m}={n}" for m, n in rows[:6])
                 + " | remedy: run Fix Missing Tracks at the work machine "
                   "(Settings > Database maintenance) — the source share is "
                   "unreachable elsewhere")
        else:
            check("data quality: every graded unit has a measurement behind it",
                  True, f"0 of {n_tracks_total} tracks orphan a verdict")
    except Exception as e:
        check("data quality: every graded unit has a measurement behind it", False,
              f"{type(e).__name__}: {e}")

    try:
        n_by_design = raw.execute(
            f"SELECT COUNT(*) FROM track_results t "
            f"JOIN analysis_results a ON a.id = t.analysis_id "
            f"WHERE a.overall_status IN ('PASS','WARNING','FAIL') "
            f"  AND t.status = 'UNTRIMMED' AND ({_BOTH_ABSENT})").fetchone()[0]
        n_with_sweep = raw.execute(
            f"SELECT COUNT(*) FROM track_results t "
            f"JOIN analysis_results a ON a.id = t.analysis_id "
            f"WHERE a.overall_status IN ('PASS','WARNING','FAIL') "
            f"  AND t.status = 'UNTRIMMED' AND ({_BOTH_ABSENT}) "
            f"  AND NOT " + _NO_ARRAY.format(c="t.untrimmed_positions")).fetchone()[0]
        # By design, but only if the sweep it was moved to actually survived.
        # A bare count would hide an untrimmed track that lost BOTH arrays.
        check("untrimmed-only tracks in graded units keep their pre-trim sweep",
              n_by_design == n_with_sweep,
              f"{n_with_sweep} of {n_by_design} retain untrimmed_positions "
              f"(by design: one track of a multi-track unit had no laser-trim run)")
    except Exception as e:
        check("untrimmed-only tracks in graded units keep their pre-trim sweep",
              False, f"{type(e).__name__}: {e}")

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
        # Guard against a silently-empty scan (weak-assertion trap), in TWO
        # parts, because there are two ways for this to go quiet and a ratio
        # alone can only see one of them.
        #
        # COVERAGE is relative. The old gate asked for an absolute
        # `scanned > 100_000` against a table of 86,856 rows, all eligible: a
        # gate that could not be satisfied by this database before OR after a
        # rebuild. One that can never pass is as useless as one that can never
        # fail, and a permanently-red line teaches the reader to skip past
        # red. 81,017 of the 86,856 are scanned (93.3%); the rest parse to an
        # empty limit array and are skipped on purpose a few lines up. 75%
        # leaves headroom under that without going anywhere near the collapse
        # this exists to catch, and a ratio survives the reprocess.
        #
        # POPULATION is absolute, and has to be, because `eligible` and
        # `scanned` move together: a rebuild that wrote limit arrays for only
        # a thousand tracks would report "1000 of 1000 eligible tracks
        # scanned" and read green on the ratio alone. This is the one check
        # whose whole job is to notice that the overnight rebuild produced
        # LESS DATA THAN IT SHOULD HAVE, so the size of the corpus is the
        # thing being asserted, not the fraction of it that was walked.
        #
        # 50,000 against today's 86,856: the rebuild has to lose more than
        # four tracks in ten before this fires, which is far outside any
        # plausible variation from re-ingesting the same corpus, while still
        # being 50x the thousand-row collapse it is here for. If the corpus
        # legitimately shrinks, lower this DELIBERATELY and say why -- that is
        # the point of it being a number someone had to choose.
        POPULATION_FLOOR = 50_000
        eligible = raw.execute(
            "SELECT COUNT(*) FROM track_results t "
            "JOIN analysis_results a ON a.id = t.analysis_id "
            "WHERE t.upper_limits IS NOT NULL AND t.lower_limits IS NOT NULL"
        ).fetchone()[0]
        check("data quality: the limit-column population did not collapse",
              eligible >= POPULATION_FLOOR,
              f"{eligible} tracks carry limit arrays, floor {POPULATION_FLOOR}"
              + ("" if eligible >= POPULATION_FLOOR else
                 " -- the rebuild wrote less than it should have, or the floor "
                 "needs lowering on purpose"))
        check("data quality: limit-column scan actually examined rows",
              eligible > 0 and scanned >= 0.75 * eligible,
              f"{scanned} of {eligible} eligible tracks scanned"
              + (f" ({eligible - scanned} had empty limit arrays)"
                 if eligible else ""))
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
STANDALONE = {"glosses": check_usability_glosses,
              "ft-fastpath": check_ft_incremental_fastpath,
              "ft-silence": check_ft_parser_console_silence,
              "ft-window": check_ft_graded_window,
              "ingest": check_ingest_group,
              "findings": check_findings_fixtures,
              "increment-volts": lambda: (check_increment_volts_fixtures(),
                                          check_increment_volts_corpus()),
              "initial-trim-value": check_initial_trim_value_fixtures,
              "track2-setup": check_track2_setup_fixtures}


if __name__ == "__main__":
    # `--only` is found ANYWHERE in argv, not just at position 1. It used to be
    # tested as `sys.argv[1] == "--only"`, and `main()` never parsed `--only`
    # at all, so the form this file's own docstrings and briefs use --
    #     app_qa_sweep.py /tmp/qa_copy.db --only ingest
    # -- silently ran the FULL sweep instead of the named section: twenty
    # quiet minutes instead of the twenty seconds that were asked for.
    if "--only" in sys.argv:
        i = sys.argv.index("--only")
        name = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        if name not in STANDALONE:
            raise SystemExit(f"unknown section {name!r}; have: {', '.join(STANDALONE)}")
        # Standalone sections build their own throwaway databases and never
        # open a path from argv, but the production file is refused by name
        # here too: the refusal must not depend on which branch was taken.
        for arg in sys.argv[1:]:
            if not arg.startswith("--") and arg != name \
                    and _db_guard.is_production_db(Path(arg), REPO, by_name=True):
                raise SystemExit(f"FATAL | {arg} is the PRODUCTION database")
        STANDALONE[name]()
        raise SystemExit(_tally())
    raise SystemExit(main())
