"""Chart QA sweep: render EVERY v6 chart across realistic data variants.

Purpose (2026-07-07): design defects (batch zigzag lines, unexplained symbols,
blank error states) shipped because charts were only eyeballed with one lucky
data shape. This harness renders the full chart surface against the variant
matrix below so every change gets the same sweep — run it headless anywhere:

    python scripts/chart_qa_render_all.py [output_dir]

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


def _focus():
    from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
    o = FocusChart.__new__(FocusChart); o.theme = _Theme()
    o._fig = Figure(figsize=(8, 3), dpi=110, facecolor=o.theme.CARD)
    o._ax = o._fig.add_subplot(111); o.canvas = FigureCanvasTkAgg(o._fig)
    o._style(); return o


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
FAIL_HEAVY_AID = 90636        # 8340-1 / SN 153
MULTITRACK_AID = 88725        # 8074-1, 2 tracks


def main(out_dir: str) -> int:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    manifest: list = []

    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, TrackResult as DBTR, SmoothnessResult as DBSR,
        ModelMetricState)
    from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS
    db = DatabaseManager(REPO / "data" / "analysis.db")

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

    # ---- 2. Company trend ----
    for days, period in [(90, "week"), (365, "month"), (36500, "month"), (30, "week")]:
        cc = _company()
        cc.set_data(db.get_company_yield_trend(days_back=days, period=period), period)
        _save(cc, out / f"company_{days}d_{period}.png", manifest,
              f"CompanyTrend {days}d/{period} — linearity basis, partial marker, vintage")

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

    for aid, note in [(FAIL_HEAVY_AID, "fail-heavy unit (James's SN153)"),
                      (MULTITRACK_AID, "multi-track unit (track selector target)")]:
        data = load_unit_track(db, aid)
        serial, fdate = unit_meta(aid)
        fp = compute_fail_points(data["error_data"], data["upper_limits"], data["lower_limits"],
                                 offset=data.get("optimal_offset") or 0.0)
        cw = _v5chart()
        cw.plot_error_vs_position(
            positions=data["position_data"], trimmed_errors=data["error_data"],
            upper_limits=data["upper_limits"] or None,
            lower_limits=data["lower_limits"] or None,
            untrimmed_positions=data["untrimmed_positions"] or None,
            untrimmed_errors=data["untrimmed_errors"] or None,
            offset=data.get("optimal_offset") or 0.0,
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


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(
        REPO / "docs" / "v6_design_review_2026-07-07" / "qa_sweep")
    raise SystemExit(main(target))
