"""Chart QA sweep: render EVERY v6 chart across realistic data variants.

Purpose (2026-07-07): design defects (batch zigzag lines, unexplained symbols,
blank error states) shipped because charts were only eyeballed with one lucky
data shape. This harness renders the full chart surface against the variant
matrix below so every change gets the same sweep — run it headless anywhere:

    python scripts/chart_qa_render_all.py [output_dir] [path/to/analysis.db]

The optional DB path is for machines where the work database is not at
data/analysis.db — pass a COPY, never the original (it is opened read-write).

Output defaults to qa_output/ at the repo root, which is gitignored: a QA run
is scratch, and defaulting it into a tracked docs folder (as this did until
2026-08-31) left ~25 rewritten PNGs in `git status` after every sweep, which
trains you to ignore a dirty tree. docs/v6_design_review_2026-07-07/qa_sweep/
stays as committed — a frozen snapshot of that review, not a live target.

Variants: dense-batch model, sparse model, single-unit model, stale flagged
model, smoothness-rich model, fail-heavy unit, multi-track unit, no-sweep
unit, empty/None inputs. Output: PNGs + MANIFEST.txt listing what to inspect.
"""
import sys
import types
from pathlib import Path

# ---- make the GUI importable headless (stub Tk/CTk; chart logic untouched) --
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
# Scratch, gitignored, shared with app_qa_sweep.py. Never a tracked directory.
QA_OUTPUT = REPO / "qa_output"
sys.path.insert(0, str(REPO / "src"))

from datetime import datetime  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


class _Theme:
    BG="#1a1f2e"; SURFACE="#1e2435"; CARD="#263244"; ELEVATED="#2f3b50"
    ACCENT="#3b82f6"; ACCENT_HOVER="#60a5fa"; ACCENT_PRESSED="#2563eb"
    TEXT_PRIMARY="#e8eef5"; TEXT_SECONDARY="#9ca8bd"; TEXT_DISABLED="#5a6478"
    TEXT_INVERSE="#1a1f2e"; DIVIDER="#2a3142"; BORDER="#3a4456"
    TIER_STABLE="#1e2435"; TIER_WARNING="#f59e0b"; TIER_DRIFT="#f97316"
    TIER_OOC="#ef4444"; TIER_WARNING_BG="#3d2f1a"; TIER_DRIFT_BG="#3d2418"
    TIER_OOC_BG="#3d1818"
    SPACE_XS=4; SPACE_SM=8; SPACE_MD=14; SPACE_LG=22
    RADIUS_SM=6; RADIUS_MD=10; SIZE_CAPTION=11; SIZE_BODY=13
    SIZE_HEADING=15; SIZE_TITLE=18
    def font(self, *a, **k): return None
    @staticmethod
    def fmt_measure(v, sig: int = 4) -> str:
        if v is None: return "—"
        av = abs(float(v))
        if av >= 1e7 or (av != 0 and av < 1e-4): return f"{v:.{sig}g}"
        if av >= 1000: return f"{v:,.0f}"
        return f"{v:.{sig}g}"


def _focus():
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
    o = FocusChart.__new__(FocusChart); o.theme = _Theme()
    o._fig = Figure(figsize=(8, 3), dpi=110, facecolor=o.theme.CARD)
    o._ax = o._fig.add_subplot(111); o.canvas = FigureCanvasTkAgg(o._fig)
    o._style(); return o


def _spc():
    """The SPC lot view is the SAME widget in its other mode (`set_spc_series`),
    so it gets the same headless figure — a separate factory only so the
    p-chart block below reads for itself."""
    return _focus()


def _company():
    from laser_trim_analyzer.gui.v6.widgets.company_trend_chart import CompanyTrendChart
    o = CompanyTrendChart.__new__(CompanyTrendChart); o.theme = _Theme()
    o._fig = Figure(figsize=(8, 2.6), dpi=110, facecolor=o.theme.CARD)
    o._ax = o._fig.add_subplot(111); o._vol_ax = o._ax.twinx()
    o.canvas = FigureCanvasTkAgg(o._fig); o._style(); return o


def _mini():
    from laser_trim_analyzer.gui.v6.widgets.mini_trend_chart import MiniTrendChart
    o = MiniTrendChart.__new__(MiniTrendChart); o.theme = _Theme()
    o._fig = Figure(figsize=(3.2, 1.0), dpi=110, facecolor=o.theme.CARD)
    o._ax = o._fig.add_subplot(111); o.canvas = FigureCanvasTkAgg(o._fig)
    if hasattr(type(o), "_style"): o._style()
    return o


def _history():
    from laser_trim_analyzer.gui.v6.widgets.history_tab import HistoryTab
    o = HistoryTab.__new__(HistoryTab); o.theme = _Theme()
    o._fig = Figure(figsize=(8, 5), dpi=110, facecolor=o.theme.CARD)
    o._ax_val = o._fig.add_subplot(211); o._ax_pr = o._fig.add_subplot(212)
    o.canvas = FigureCanvasTkAgg(o._fig)
    o._stats = types.SimpleNamespace(configure=lambda **k: None)
    o._menu = types.SimpleNamespace(set=lambda *a: None, configure=lambda **k: None)
    return o


def _v5chart(figsize=(9, 5)):
    from laser_trim_analyzer.gui.widgets.chart import ChartWidget, ChartStyle
    o = ChartWidget.__new__(ChartWidget)
    o.style = ChartStyle(figure_size=figsize)
    o.figure = Figure(figsize=figsize, dpi=110)
    o.ax = o.figure.add_subplot(111); o.canvas = FigureCanvasTkAgg(o.figure)
    for a in ("_placeholder_visible", "_heatmap_original_size", "_colorbar",
              "_secondary_ax", "_overlay_ax", "_legend", "_placeholder_label"):
        setattr(o, a, None)
    return o


def _save(obj, path, manifest, note):
    fig = getattr(obj, "_fig", None)
    if not isinstance(fig, Figure):
        fig = getattr(obj, "figure", None)
    if not isinstance(fig, Figure):
        fig = obj
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    manifest.append(f"{Path(path).name} | {note}")
    print("saved", Path(path).name)


# Variant matrix (real models spanning the data shapes charts must survive).
DENSE, SPARSE, SINGLE, STALE = "6607", "5409B", "8150", "8887"
SMOOTH_RICH = "7458-1"
# Same five models app_qa_sweep.py exercises, so a shape that breaks a loader
# and a shape that breaks a chart are found on the same data.
VARIANTS = [DENSE, SPARSE, SINGLE, STALE, SMOOTH_RICH]
# Unit fixtures are RESOLVED BY QUERY, never hard-coded.
#
# They used to be literal analysis ids with a model/serial in the comment, and
# both had silently drifted: FAIL_HEAVY_AID=90636 ("8340-1 / SN 153") was in
# fact SN 336 with ZERO fail points, and MULTITRACK_AID=88725 ("8074-1,
# 2 tracks") was a SINGLE-track 8415-1. So the visual sweep rendered neither
# the fail-marker path nor the track selector it exists to exercise, while
# still producing two confident-looking PNGs. `analysis_results.id` is a
# surrogate key — it moves whenever the database is rebuilt or reprocessed, so
# any literal here is a fixture with an expiry date nobody can see.
#
# Resolving by the PROPERTY each fixture is named for makes drift impossible:
# the fail-heavy unit is whichever unit has the most fail points, the
# multi-track unit is one that genuinely has more than one track. Both raise
# if the database holds no such unit, because rendering a substitute that
# quietly lacks the property is the failure being fixed.

def resolve_unit_fixtures(db):
    """(fail_heavy_aid, multitrack_aid) for THIS database. Never guesses."""
    from sqlalchemy import func

    # Imported here, not at module scope: the GUI stubs at the top of this file
    # must be installed before anything pulls in the package.
    from laser_trim_analyzer.database.models import TrackResult as DBTR

    with db.session() as s:
        heavy = (s.query(DBTR.analysis_id,
                         func.sum(func.coalesce(DBTR.linearity_fail_points, 0)).label("fp"))
                 .group_by(DBTR.analysis_id)
                 .having(func.sum(func.coalesce(DBTR.linearity_fail_points, 0)) > 0)
                 .order_by(func.sum(func.coalesce(DBTR.linearity_fail_points, 0)).desc())
                 .first())
        multi = (s.query(DBTR.analysis_id)
                 .group_by(DBTR.analysis_id)
                 .having(func.count(DBTR.id) > 1)
                 .order_by(func.count(DBTR.id).desc(), DBTR.analysis_id.desc())
                 .first())
    if heavy is None:
        raise SystemExit("FATAL | no unit in this database has any fail points — "
                         "the fail-marker rendering path cannot be exercised")
    if multi is None:
        raise SystemExit("FATAL | no multi-track unit in this database — "
                         "the track selector cannot be exercised")
    return int(heavy[0]), int(multi[0])


def resolve_ft_overlay_unit(db):
    """A trim analysis whose linked final test actually has a drawable sweep.

    Same rule as the fixtures above: resolved by the PROPERTY, never a literal
    id. Prefers a unit final-tested more than once, so the "newest of N"
    disclosure is exercised too. None when this database has no such pair —
    the overlay render is then skipped with a printed reason rather than
    rendering a trim chart and calling it an overlay.
    """
    from laser_trim_analyzer.core.ft_overlay import MIN_MATCH_CONFIDENCE
    from laser_trim_analyzer.database.models import (
        FinalTestResult as DBFT, FinalTestTrack as DBFTT)
    from sqlalchemy import func

    with db.session() as s:
        q = (s.query(DBFT.linked_trim_id, func.count(DBFT.id).label("n"))
             .join(DBFTT, DBFTT.final_test_id == DBFT.id)
             .filter(DBFT.linked_trim_id.isnot(None),
                     DBFT.match_confidence >= MIN_MATCH_CONFIDENCE,
                     DBFTT.position_data.isnot(None),
                     DBFTT.error_data.isnot(None))
             .group_by(DBFT.linked_trim_id)
             .order_by(func.count(DBFT.id).desc(), DBFT.linked_trim_id.desc())
             .first())
    return int(q[0]) if q else None


def main(out_dir: str, db_path: Path) -> int:
    if not db_path.exists():
        # BEFORE DatabaseManager, which CREATES the file it is handed: charts
        # rendered from an empty database look fine and prove nothing.
        print(f"FATAL | no database at {db_path}\n"
              f"      | pass a copy of the work database:\n"
              f"      |     python scripts/chart_qa_render_all.py OUTDIR /path/to/analysis.db")
        return 1
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    manifest: list = []

    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SmoothnessResult as DBSR,
        ModelMetricState)
    from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS
    db = DatabaseManager(db_path)

    def series(model, metric):
        with db.session() as s:
            if metric == "max_smoothness_value":
                rows = (s.query(DBSR.file_date, DBSR.max_smoothness_value)
                        .filter(DBSR.model == model,
                                DBSR.max_smoothness_value.isnot(None))
                        .order_by(DBSR.file_date).all())
            else:
                col = TRACK_METRIC_COLUMNS[metric]
                rows = (s.query(DBAR.file_date, col)
                        .join(DBTR, DBTR.analysis_id == DBAR.id)
                        .filter(DBAR.model == model, col.isnot(None))
                        .order_by(DBAR.file_date).all())
            ms = s.query(ModelMetricState).filter_by(model=model, metric=metric).first()
            bl = (ms.baseline_mean, ms.baseline_std) if ms and ms.is_trained else (None, None)
        pairs = [(d, v) for d, v in rows if d is not None and v is not None]
        return [p[0] for p in pairs], [p[1] for p in pairs], bl

    # ---- 1. FocusChart across variants ----
    for model, metric, note in [
        (DENSE, "untrimmed_resistance", "dense batches: dots + daily median, no zigzag"),
        (DENSE, "linearity_error", "dense linearity: median hugs band, off-scale honest"),
        (SPARSE, "untrimmed_resistance", "sparse model: readable with 12 units"),
        (SINGLE, "untrimmed_resistance", "single unit: no crash, sensible axes"),
        (STALE, "linearity_error", "stale flagged model, full record"),
        (SMOOTH_RICH, "max_smoothness_value", "smoothness trend (scatter mode)"),
    ]:
        d, v, (bm, bs) = series(model, metric)
        fc = _focus()
        fc.set_series(metric, d, v, baseline_mean=bm, baseline_std=bs)
        _save(fc, out / f"focus_{model}_{metric}.png", manifest,
              f"FocusChart {model}/{metric} n={len(v)} — {note}")

    # ---- 1b. Window-switch regression (2026-07-08). One REUSED FocusChart,
    # All -> narrow window: the x-axis must track the new data, not hold the
    # widest range ever rendered (matplotlib's lazy autoscale never shrank it,
    # so after 'All' every window showed a decade with data bunched right).
    import matplotlib.dates as _mdt
    from datetime import timedelta as _td
    d_all, v_all, (bm_, bs_) = series("8340-1", "untrimmed_error_max")
    fc = _focus()
    fc.set_series("untrimmed_error_max", d_all, v_all, baseline_mean=bm_, baseline_std=bs_)
    wide = fc._ax.get_xlim()
    cut = max(d_all) - _td(days=90)
    d_n, v_n = zip(*[(d, v) for d, v in zip(d_all, v_all) if d >= cut])
    fc.set_series("untrimmed_error_max", list(d_n), list(v_n),
                  baseline_mean=bm_, baseline_std=bs_)
    narrow = fc._ax.get_xlim()
    wide_span, narrow_span = wide[1] - wide[0], narrow[1] - narrow[0]
    data_span = _mdt.date2num(max(d_n)) - _mdt.date2num(min(d_n))
    if narrow_span > max(data_span * 1.5, 30.0):
        raise AssertionError(
            f"WINDOW-SWITCH REGRESSION: 90d xlim spans {narrow_span:.0f} days "
            f"(data {data_span:.0f}d; previous All render {wide_span:.0f}d) — "
            "x-axis is holding the old window")
    # Red out-of-limit markers must be NAMED in the legend (unexplained red
    # dots finding, 2026-07-08). 8340-1/untrimmed_error_max has off-scale
    # points, so the entry must be present.
    leg = fc._ax.get_legend()
    leg_texts = [t_.get_text() for t_ in (leg.get_texts() if leg else [])]
    if not any("Beyond ±3σ" in t_ for t_ in leg_texts):
        raise AssertionError(f"legend misses the red-marker entry: {leg_texts}")
    _save(fc, out / "focus_8340-1_window_switch.png", manifest,
          f"FocusChart window-switch: All(xlim {wide_span:.0f}d) -> 90d(xlim {narrow_span:.0f}d) — axis tracks the window; red markers in legend")

    # ---- 1c. SPC lot p-chart (2026-08-29 FOCUS redesign) ----
    # The view the Model page opens on and the FOCUS list links to. Rendered
    # per variant because the shapes that break it are data shapes: a model
    # with too few lots (no band at all), a one-lot model, a stale model whose
    # newest lot is months old, and a dense model with real excursions.
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import spc_draw_params
    from laser_trim_analyzer.ml.spc import compute_spc_series
    for model in VARIANTS:
        spc = compute_spc_series(db, model)
        sc = _spc()
        sc.set_spc_series(spc)
        # What to LOOK for differs by variant — telling a reader to check for
        # annotations on a chart that has no limits is how a sweep gets skimmed.
        dp = spc_draw_params(spc)
        recent, older = len(dp["flag_idx"]), dp["old_ooc_count"]
        expect = ("no band at all, with the reason said out loud" if not spc.judged
                  else f"band + {recent} red annotated excursion(s) whose sentences "
                       f"do not overlap, {older} amber counted" if recent or older
                       else "band, no excursions flagged")
        _save(sc, out / f"spc_pchart_{model}.png", manifest,
              f"SPC p-chart {model} lots={len(spc.points)} judged={spc.judged} "
              f"— expect {expect}; n= under every lot, dates in order")

    # ---- 2. Company trend ----
    for days, period in [(90, "week"), (365, "month"), (36500, "month"), (30, "week")]:
        cc = _company()
        # All-time + weekly is auto-coarsened by the Dashboard; render the
        # coarsened form WITH its disclosure note (2026-07-08).
        note = ("shown monthly — weekly is too dense for this window"
                if days > 730 else None)
        cc.set_data(db.get_company_yield_trend(days_back=days, period=period), period, note=note)
        _save(cc, out / f"company_{days}d_{period}.png", manifest,
              f"CompanyTrend {days}d/{period} — linearity basis, partial marker, vintage"
              + (", coarsen note" if note else ""))

    # ---- 3. Mini trends ----
    from laser_trim_analyzer.core.yield_stats import compute_yield
    from laser_trim_analyzer.database.models import FinalTestResult as DBFT
    from datetime import timedelta
    for cls, name in [(DBAR, "trim"), (DBFT, "ft")]:
        y = compute_yield(db, cls, datetime.now() - timedelta(days=36500))
        mc = _mini()
        mc.set_points([(p["date"], p.get("rate", p.get("pass_rate"))) for p in y["trend"]])
        _save(mc, out / f"mini_{name}.png", manifest,
              f"MiniTrend {name} pts={len(y['trend'])} — sparkline follows linearity basis")

    # ---- 4. History tab ----
    for model, metric in [(DENSE, "measured_electrical_angle"),
                          (DENSE, "untrimmed_resistance"),
                          (SINGLE, "untrimmed_resistance")]:
        h = _history()
        h._data = db.get_model_measurement_history(model)
        h._metric = metric
        h._render()
        _save(h, out / f"history_{model}_{metric}.png", manifest,
              f"HistoryTab {model}/{metric} — panel titles clear of ticks")

    # ---- 5. Unit modal chart + print export ----
    from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import (
        load_unit_track, compute_fail_points)
    from laser_trim_analyzer.export.unit_chart import build_unit_export_figure

    def unit_meta(aid):
        with db.session() as s:
            r = s.query(DBAR.serial, DBAR.file_date).filter(DBAR.id == aid).first()
        return str(r[0]), str(r[1]).split(" ")[0]

    fail_heavy_aid, multitrack_aid = resolve_unit_fixtures(db)
    for aid, note in [(fail_heavy_aid, "fail-heavy unit (most fail points in this DB)"),
                      (multitrack_aid, "multi-track unit (track selector target)")]:
        data = load_unit_track(db, aid)
        serial, fdate = unit_meta(aid)
        _k = data.get("optimal_slope") or 0.0
        _theory = data.get("theory_data")
        fp = compute_fail_points(data["error_data"], data["upper_limits"], data["lower_limits"],
                                 offset=data.get("optimal_offset") or 0.0,
                                 k=_k, theory=_theory)
        cw = _v5chart()
        cw.plot_error_vs_position(
            positions=data["position_data"], trimmed_errors=data["error_data"],
            upper_limits=data["upper_limits"] or None,
            lower_limits=data["lower_limits"] or None,
            untrimmed_positions=data["untrimmed_positions"] or None,
            untrimmed_errors=data["untrimmed_errors"] or None,
            offset=data.get("optimal_offset") or 0.0,
            k=_k, theory_data=_theory,
            trim_improvement_percent=data.get("trim_improvement_percent"),
            trim_date=fdate, fail_points=fp,
            title=f"Unit {serial}", serial_number=serial)
        _save(cw, out / f"unit_screen_{aid}.png", manifest,
              f"Unit modal screen chart aid={aid} fails={len(fp)} — {note}")

        meta = {"model": data.get("model"), "serial": serial,
                "system": data.get("system"), "trim_date": fdate,
                "track_id": data.get("track_id"), "n_tracks": data.get("n_tracks", 1)}
        fig = build_unit_export_figure(meta, data, fp)
        fig.savefig(out / f"unit_export_{aid}.png", facecolor="white", bbox_inches="tight")
        manifest.append(f"unit_export_{aid}.png | print export aid={aid} — {note}")
        print("saved", f"unit_export_{aid}.png")

    # ---- 5b. Trim-vs-FT overlay (V6's replacement for V5 Compare) ----
    from laser_trim_analyzer.core.ft_overlay import load_ft_overlay

    ov_aid = resolve_ft_overlay_unit(db)
    if ov_aid is None:
        print("skipped overlay render | no trim in this database has a linked "
              "final test with a stored sweep")
    else:
        data = load_unit_track(db, ov_aid)
        serial, fdate = unit_meta(ov_aid)
        ov = load_ft_overlay(db, ov_aid, trim_track_id=data.get("track_id"),
                             trim_positions=data.get("position_data"))
        if not ov.get("available"):
            print(f"skipped overlay render | aid={ov_aid}: {ov.get('reason')}")
        else:
            _k = data.get("optimal_slope") or 0.0
            fp = compute_fail_points(data["error_data"], data["upper_limits"],
                                     data["lower_limits"],
                                     offset=data.get("optimal_offset") or 0.0,
                                     k=_k, theory=data.get("theory_data"))
            cw = _v5chart()
            cw.plot_error_vs_position(
                positions=data["position_data"], trimmed_errors=data["error_data"],
                upper_limits=data["upper_limits"] or None,
                lower_limits=data["lower_limits"] or None,
                offset=data.get("optimal_offset") or 0.0,
                k=_k, theory_data=data.get("theory_data"),
                trim_date=fdate, fail_points=fp, ft_overlay=ov,
                title=f"Unit {serial} + final test", serial_number=serial)
            _save(cw, out / f"unit_ft_overlay_screen_{ov_aid}.png", manifest,
                  f"Unit + FT overlay aid={ov_aid} — {ov['label']}; FT trace "
                  "must carry its OWN band and correction")
            meta = {"model": data.get("model"), "serial": serial,
                    "system": data.get("system"), "trim_date": fdate,
                    "track_id": data.get("track_id"),
                    "n_tracks": data.get("n_tracks", 1)}
            fig = build_unit_export_figure(meta, data, fp, ft_overlay=ov)
            fig.savefig(out / f"unit_ft_overlay_export_{ov_aid}.png",
                        facecolor="white", bbox_inches="tight")
            manifest.append(f"unit_ft_overlay_export_{ov_aid}.png | print export "
                            f"with FT overlay — {ov['label']}")
            print("saved", f"unit_ft_overlay_export_{ov_aid}.png")

    # ---- 6. Evidence pack + copy summary (data products) ----
    from laser_trim_analyzer.export.evidence import (
        export_evidence_pack, build_summary_text, compute_recent_means)
    from laser_trim_analyzer.ml.manager import get_model_drift_status
    pack = export_evidence_pack(db, DENSE, out / f"evidence_{DENSE}.xlsx")
    manifest.append(f"evidence_{DENSE}.xlsx | 3 sheets, full history")
    means, meta_ = compute_recent_means(db, STALE, with_meta=True)
    txt = build_summary_text(STALE, get_model_drift_status(db, STALE),
                             recent_means=means, recent_meta=meta_)
    (out / f"copy_summary_{STALE}.txt").write_text(txt)
    manifest.append(f"copy_summary_{STALE}.txt | suspect-exclusion disclosure expected")
    print("saved evidence + summary")

    (out / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    print(f"\n{len(manifest)} artifacts -> {out}")
    return 0


def refuse_production_db(db_path: Path) -> str | None:
    """The one database this harness must never open.

    Opening it is a WRITE: DatabaseManager's engine setup commits index
    creation into the file. Three separate sessions opened the real DB by
    accident on 2026-08-31 — every one of them through the no-argument
    default that used to live below. There is no legitimate harness use of
    the production file, so this refuses it even when passed explicitly.
    Returns the error text, or None when the path is fine.
    """
    if db_path.resolve() == (REPO / "data" / "analysis.db").resolve():
        return (f"FATAL | {db_path} is the PRODUCTION database and opening it "
                f"WRITES to it.\n"
                f"      | make a copy and pass that instead:\n"
                f"      |     cp data/analysis.db /tmp/qa_copy.db\n"
                f"      |     python scripts/chart_qa_render_all.py "
                f"qa_output /tmp/qa_copy.db")
    return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python scripts/chart_qa_render_all.py OUTPUT_DIR "
              "/path/to/COPY_of_analysis.db\n"
              "(the DB argument is required — the harness refuses the "
              "production database)")
        raise SystemExit(1)
    db_arg = Path(sys.argv[2])
    err = refuse_production_db(db_arg)
    if err:
        print(err)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1], db_arg))
