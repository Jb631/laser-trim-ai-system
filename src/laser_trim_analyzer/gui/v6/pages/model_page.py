"""The Model page: one model, in James's layout C (Graphite redesign, 2026-10-02).

"there is so much going on its hard to see what is what" (James) -- the page stacked twelve metric
pills, a stats table, a chart and seven tabs. He picked layout C ("i like c") from three drawn on
8504-2's real runs: the header's controls, then a HEADER LINE -- the model, a status word (Drifting /
Steady / Inactive), its 90-day linearity pass % with its units, the lasers it ran on -- over FOUR tabs:

  * Summary    -- the verdict in one sentence; the run chart of the moving signal (Lots / Units);
                  "Also moving", every OTHER signal above stable; "Worth changing", the model's
                  findings (with the rest of what the analyzers measured folded under it); and
                  "All 12 signals", the drift table, folded.
  * Units      -- the stats table (the run menu drives it), the unit list, smoothness.
  * Final test -- the final-test units, trim vs final test, the predictor.
  * History    -- the measurement history.

The pills and their one-line σ key are gone -- their numbers are in "All 12 signals" and the Units
tab -- and nothing else was lost: it moved one level down. Each tab scrolls on its own; the header
line and the tabs stay put. Every loader runs on a worker (CLAUDE.md rule 5); Tk only in apply()."""
import logging
import threading
import tkinter
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import customtkinter as ctk

logger = logging.getLogger(__name__)

from laser_trim_analyzer.core.activity import inactive_tag, load_activity, trusted_until
from laser_trim_analyzer.core.model_stats import (
    compute_lot_verdicts, compute_model_stats, default_lot_index, model_lots)
from laser_trim_analyzer.core.models import LASER_ORDER, laser_label
from laser_trim_analyzer.core.spec_alignment import compare_station_specs
from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAR, ModelMetricState, SmoothnessResult as DBSR, TrackResult as DBTR, StatusType)
from laser_trim_analyzer.findings import presentation as P
from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets import blocks
from laser_trim_analyzer.gui.v6.widgets.drift_metrics_tab import DriftMetricsTab
from laser_trim_analyzer.gui.v6.widgets.findings_tab import (
    NOT_COMPUTED_TEXT, FindingsTab, analyzer_name)
from laser_trim_analyzer.gui.v6.widgets.findings_view import FindingsView
from laser_trim_analyzer.gui.v6.widgets.focus_chart import FocusChart
from laser_trim_analyzer.gui.v6.widgets.history_tab import HistoryTab
from laser_trim_analyzer.gui.v6.widgets.predictor_panel import PredictorPanel
from laser_trim_analyzer.gui.v6.widgets.smoothness_tab import SmoothnessTab
from laser_trim_analyzer.gui.v6.widgets.stats_table import StatsTableZone
from laser_trim_analyzer.gui.v6.widgets.tab_view import ThemedTabView
from laser_trim_analyzer.gui.v6.widgets.trim_ft_tab import TrimFtTab
from laser_trim_analyzer.gui.v6.widgets.ft_units_tab import FtUnitsTab
from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import UnitChartModal
from laser_trim_analyzer.gui.v6.widgets.units_tab import UnitsTab
from laser_trim_analyzer.ml.drift_training import TRACK_METRIC_COLUMNS, drift_exclusions
from laser_trim_analyzer.ml.drift_types import (
    WATCHED_METRICS, DriftTier, format_metric_value, metric_label)
from laser_trim_analyzer.ml.manager import get_model_drift_status, list_known_models
from laser_trim_analyzer.ml.spc import compute_focus_list, compute_spc_series

_WINDOW_DAYS = {"30d": 30, "90d": 90, "365d": 365, "All": None}
# The window control's own default -- what the page opens on, before anyone picks.
_DEFAULT_WINDOW_CHOICE = "90d"

# Headline-chart views (2026-08-29 FOCUS/SPC redesign). Production runs in
# LOTS, so a lot — not a unit — is what goes in or out of control, and the lot
# chart is what the FOCUS list sends people here to look at. Lots is therefore
# the default for EVERY metric; the per-unit scatter that used to be the only
# view stays one click away for "what did each unit measure".
_VIEW_LOTS, _VIEW_UNITS = "Lots · SPC", "Units"

# Lot selector's "no lot" entry. Named, not blank: "all history" is the answer
# to James's main question (what has this model ever done), so it is a real
# choice, not an absence of one.
_ALL_HISTORY = "All history (no lot)"

# Default focus metric when no alert-triggered focus is supplied. The headline
# element-production drift signal (post-trim sigma_gradient is no longer
# watched — see drift_types.WATCHED_METRICS / the D-SIGMA rationale).
_DEFAULT_METRIC = "untrimmed_sigma_gradient"

# The four tabs, in James's order (layout C).
TAB_SUMMARY, TAB_UNITS, TAB_FINAL_TEST, TAB_HISTORY = "Summary", "Units", "Final test", "History"
TAB_NAMES = (TAB_SUMMARY, TAB_UNITS, TAB_FINAL_TEST, TAB_HISTORY)

# Where `set_model_route(model, tab=...)` lands: {route word: (tab, the part to open or scroll to)}.
# Route words are plain and compared case-insensitively. Seven are the names of the tabs this page
# had before layout C, and each lands where its content went -- the brief's map: findings -> Summary
# at "Worth changing", its fold open; drift -> Summary with "All 12 signals" open. The callers today
# (findings_page.py, home_page.py) both pass "findings"; tests/test_spec3c_model.py greps the source
# for every tab= route and fails on one this map does not know.
_ROUTES: Dict[str, Tuple[str, Optional[str]]] = {
    "summary": (TAB_SUMMARY, None),
    "findings": (TAB_SUMMARY, "findings"),
    "worth changing": (TAB_SUMMARY, "findings"),
    "drift": (TAB_SUMMARY, "signals"),
    "drift metrics": (TAB_SUMMARY, "signals"),
    "signals": (TAB_SUMMARY, "signals"),
    "units": (TAB_UNITS, None),
    "smoothness": (TAB_UNITS, "smoothness"),
    "final test": (TAB_FINAL_TEST, None),
    "final test units": (TAB_FINAL_TEST, None),
    "trim vs final test": (TAB_FINAL_TEST, "trim vs final test"),
    "history": (TAB_HISTORY, None),
}


def route_destination(name) -> Optional[Tuple[str, Optional[str]]]:
    """(tab, part) a route word lands on, or None for a word the page does not know -- which is
    ignored, never raised: a stale route must never crash the page."""
    return _ROUTES.get(str(name or "").strip().lower())


# The two folds on Summary, by their words.
_SIGNALS_FOLD = f"All {len(WATCHED_METRICS)} signals"
_FINDINGS_FOLD = "What the analyzers measured"

# The verdict's clauses are joined by this (_compute_verdict); the first is Summary's headline.
_CLAUSE = "  ·  "

# ---- the header line's 90-day pass rate ---------------------------------------------------------
# The Overview's definition, so a model reads the same number on its card and on its page: the
# app's headline yield (core/yield_stats.compute_yield's linearity_yield) -- PASS or WARNING out of
# PASS + WARNING + FAIL; ERROR, PROCESSING_FAILED and UNTRIMMED are not graded -- over the 90 days
# ending at the newest graded trim file in the database (one clock for every model, so an inactive
# model reads "no units", never its last 90 days of 2019). Future-dated files (core/activity's
# trusted_until) and files marked suspect are left out, the rule the drift watch and the FOCUS list
# follow since 2026-10-02. The window is the Overview's own: the 90 CALENDAR days ending on the
# newest file's day (overview_data.WINDOW_DAYS) -- counted to the minute, a file late on the day
# before them counted here and not on the card (caught at the 2026-10-02 merge).
from laser_trim_analyzer.gui.v6.overview_data import WINDOW_DAYS as HEADER_DAYS  # noqa: E402
_GRADED = (StatusType.PASS, StatusType.WARNING, StatusType.FAIL)


def _graded_counts(rows) -> Tuple[int, int]:
    """(graded, passed) from (status, count) rows -- summed, as a status can come once per laser. A
    WARNING ships: sigma is a watch, not a reject."""
    by: Dict[str, int] = {}
    for st, n in rows:
        name = getattr(st, "name", str(st))
        by[name] = by.get(name, 0) + (n or 0)
    graded = sum(by.get(k, 0) for k in ("PASS", "WARNING", "FAIL"))
    return graded, by.get("PASS", 0) + by.get("WARNING", 0)


def header_facts(db, model: str, *, now: Optional[datetime] = None) -> dict:
    """What the header line says about `model`'s last 90 days, from the database (a worker call).

    {"end": the window's end (None: no graded trim on file at all), "basis": "trim" | "final test"
    | None, "units": graded units, "passed": of them, "lasers": laser_label()s in laser order,
    "lasers_in_window": whether those are the window's lasers (else all its graded trims')}. A
    model with no trims in the window but final tests in it (8506: its trims are stored as
    8506A/B) is graded on those -- the Overview's rule."""
    from sqlalchemy import func, or_
    from laser_trim_analyzer.database.models import FinalTestResult as DBFT
    horizon = trusted_until(now)
    clean = or_(DBAR.data_quality.is_(None), DBAR.data_quality != "suspect")
    out = {"end": None, "basis": None, "units": 0, "passed": 0, "lasers": [],
           "lasers_in_window": False}
    with db.session() as s:
        end = (s.query(func.max(DBAR.file_date))
               .filter(DBAR.overall_status.in_(_GRADED), DBAR.model.isnot(None),
                       DBAR.file_date <= horizon, clean)
               .scalar())
        if isinstance(end, str):                       # raw SQLite text, as _window_cutoff sees it
            end = datetime.fromisoformat(end[:19])
        if end is None:
            return out
        start = datetime.combine(end.date() - timedelta(days=HEADER_DAYS - 1), time.min)
        mine = (DBAR.model == model, DBAR.overall_status.in_(_GRADED), DBAR.file_date <= horizon,
                clean)
        window = (s.query(DBAR.overall_status, DBAR.system, func.count(DBAR.id))
                  .filter(*mine, DBAR.file_date >= start)
                  .group_by(DBAR.overall_status, DBAR.system).all())
        units, passed = _graded_counts([(st, n) for st, _sys, n in window])
        systems = {getattr(sys_, "value", sys_) for _st, sys_, _n in window}
        out.update(end=end, lasers_in_window=bool(units))
        if units:
            out.update(basis="trim", units=units, passed=passed)
        else:
            # ...and stops at the end of the newest TRIM file's day, as the Overview does: a final
            # test dated after it is past the window (no trim can be -- `end` is their newest).
            stop = datetime.combine(end.date() + timedelta(days=1), time.min)
            ft = (s.query(DBFT.overall_status, func.count(DBFT.id))
                  .filter(DBFT.model == model, DBFT.overall_status.in_(_GRADED),
                          DBFT.file_date >= start, DBFT.file_date < stop,
                          DBFT.file_date <= horizon)
                  .group_by(DBFT.overall_status).all())
            ft_units, ft_passed = _graded_counts(ft)
            if ft_units:
                out.update(basis="final test", units=ft_units, passed=ft_passed)
            systems = {getattr(r[0], "value", r[0])
                       for r in s.query(DBAR.system).filter(*mine).distinct().all()}
    order = {letter: i for i, letter in enumerate(LASER_ORDER)}
    out["lasers"] = [laser_label(x) for x in sorted((x for x in systems if x),
                                                    key=lambda x: (order.get(x, 99), str(x)))]
    return out


def _pct_text(passed: int, units: int) -> str:
    """The pass rate as the header shows it: whole percent -- but never "100%" while a unit
    failed (linearity is zero-tolerance; 299 of 300 reads "99.7%")."""
    pct = 100.0 * passed / units
    text = f"{pct:.0f}%"
    return f"{pct:.1f}%" if (text == "100%" and passed < units) else text


def header_texts(facts: Optional[dict]) -> Tuple[str, str]:
    """(the pass % in large type, the words after it) for the header line. ("", "") when the load
    failed: the page's banner names it, and no number stands in for one."""
    if not facts:
        return "", ""
    lasers = ", ".join(facts.get("lasers") or [])
    if facts.get("units"):
        units = facts["units"]
        parts = ["linearity pass" if facts.get("basis") == "trim" else "final-test pass",
                 f"{units:,} unit{'s' if units != 1 else ''} in the last {HEADER_DAYS} days"]
        if lasers:
            parts.append(lasers if facts.get("lasers_in_window") else f"last ran on {lasers}")
        return _pct_text(facts["passed"], units), " · ".join(parts)
    parts = [f"No units in the last {HEADER_DAYS} days"]
    if lasers:
        parts.append(f"last ran on {lasers}")
    return "", " · ".join(parts)


def status_word(*, detector_flagged: Optional[bool], on_focus_list: Optional[bool],
                inactive: Optional[bool], last_trimmed: Optional[datetime]) -> Optional[str]:
    """The header's status word. "Drifting" when the drift detector flags the model (a tier above
    stable) or it is on the "Drifting now" list (ml/spc.compute_focus_list) -- the two lists the
    Overview's cards are made of; else "Inactive · last trimmed Mon YYYY" (core/activity); else
    "Steady". Each input is None when its load failed, and then no word is claimed (None) unless
    "Drifting" is known anyway: "Steady" over a crashed load is a failure looking like a result."""
    if detector_flagged or on_focus_list:
        return "Drifting"
    if detector_flagged is None or on_focus_list is None or inactive is None:
        return None
    if inactive:
        return inactive_tag(last_trimmed)
    return "Steady"


def also_moving(status, charted: Optional[str], recent_means: Optional[dict] = None,
                fmt=None) -> List[Tuple[str, str]]:
    """Summary's "Also moving": [(metric, "baseline → last lot  ↑")] for every signal above stable
    other than the charted one, worst tier first, then the furthest moved. "Last lot" is the page's
    recent mean where it has one, else the detector's -- the drift table's own rule (_MetricRow)."""
    recent_means = recent_means or {}
    rows = []
    for metric, ms in (getattr(status, "per_metric", None) or {}).items():
        if metric == charted or ms.tier <= DriftTier.STABLE:
            continue
        recent = recent_means.get(metric)
        recent = recent if recent is not None else ms.recent_mean
        base = ms.baseline_mean
        moved = (abs(recent - base) / ms.baseline_std
                 if (recent is not None and base is not None and ms.baseline_std) else 0.0)
        arrow = ("" if recent is None or base is None or recent == base
                 else "↑" if recent > base else "↓")
        text = (f"{format_metric_value(metric, base, fmt)} → "
                f"{format_metric_value(metric, recent, fmt)}" + (f"  {arrow}" if arrow else ""))
        rows.append((-int(ms.tier), -moved, metric, text))
    return [(metric, text) for _t, _m, metric, text in sorted(rows)]


def _scroll_into_view(frame, widget) -> None:
    """Scroll the CTkScrollableFrame `frame` so `widget`, somewhere inside it, sits at the top of
    its view (or as near as the content's height allows). `_parent_canvas` is CustomTkinter's own
    canvas -- private API, safe under the pinned 5.2.2 (requirements-pinned.txt)."""
    try:
        frame.update_idletasks()
        height = frame.winfo_height()
        if height <= 1:
            return
        offset = widget.winfo_rooty() - frame.winfo_rooty()
        frame._parent_canvas.yview_moveto(max(0.0, min(1.0, offset / height)))
    except (tkinter.TclError, AttributeError):
        pass                                 # destroyed, or not laid out yet

# "recent" window for the baseline-vs-recent comparison shown in the Drift table.
_RECENT_DAYS = 30


def _unit_row(r) -> dict:
    """One Units-tab row from the shared SELECT of `_load_units` / `_search_units`:
    (analysis id, serial, file_date, ANALYSIS status, sigma, linearity error, reason,
    TRACK status). One row per TRACK, so the row carries the track's own status as well
    as the unit's -- the tab decides "not graded" and the sigma dash on the TRACK's
    (`track_status`, the enum NAME `core.model_stats.failed_processing` reads), never on
    the analysis's: a graded TRK1 inside an ERROR analysis owns its linearity error."""
    return {"analysis_id": r[0], "serial": r[1], "file_date": r[2],
            "overall_status": getattr(r[3], "value", str(r[3])),
            "sigma_gradient": r[4], "linearity_error": r[5],
            "error_reason": r[6],
            "track_status": getattr(r[7], "name", r[7])}


# The three groups the page's "Worth changing on this model" section shows (design doc
# 2026-09-24-facelift-step2-pages-design.md §1, ruling 1 item 3). "history" (recipe changes,
# already happened -- nothing to decide) and "other" (an analyzer this page does not know) stay
# off it; both are in the fold under it ("What the analyzers measured"), one click away.
_WORTH_CHANGING_GROUPS = ("yield", "laser_time", "check")


def _worth_changing_count(findings) -> int:
    """Rows the 'Worth changing' section will show, across its three groups -- the SAME
    arrange() FindingsView itself calls when it draws them, so the header's count and the
    rows under it can never disagree."""
    groups = P.arrange(findings or [], include_empty=False)
    keys = set(_WORTH_CHANGING_GROUPS)
    return sum(len(g.rows) for g in groups if g.spec.key in keys)


class ModelPage(PageBase):
    page_title = "Model"

    def __init__(self, master, *, theme, app, page_title="Model"):
        self._current_model: Optional[str] = None
        self._current_metric: str = _DEFAULT_METRIC
        self._window_choice: str = _DEFAULT_WINDOW_CHOICE
        self._reload_gen = 0
        # The "Worth changing" FindingsView, when this model has findings to show --
        # None otherwise (no findings, or the load failed). Rebuilt by
        # _set_findings_section() on every apply().
        self._worth_view = None
        self._user_picked_metric = False
        self._chart_view = "lots"            # "lots" | "units" — see _VIEW_LOTS
        # Both chart views are loaded by the SAME _reload pass and cached here,
        # so flipping the toggle is a re-render, never a second DB round trip.
        self._spc_series = None
        self._unit_series = (_DEFAULT_METRIC, [], [], (None, None))
        # Lot selection. Held by LABEL, not index: `_reload` re-reads the lots
        # every pass (new files may have arrived, or the model may have
        # changed), and an index would silently point at a different run.
        # None = all history.
        self._lot_label: Optional[str] = None
        self._lots: List = []
        self._lot_default_applied_for: Optional[str] = None
        # Summary's "Also moving" lines, by metric -- rebuilt with the section on every apply().
        self._also_lines: Dict[str, ctk.CTkFrame] = {}
        # The two folds' state. They survive reloads and model switches: unfolding one is the
        # reader's choice, and a tab= route ("findings", "drift") opens one on purpose.
        self._findings_open = False
        self._signals_open = False
        self._findings_fold_shown = False   # is there anything under "Worth changing" to unfold?
        # A route's scroll target, for the reload it came with: (part, reload generation).
        self._scroll_target: Optional[Tuple[str, int]] = None
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    @staticmethod
    def _resolve_focus_metric(status, user_picked, current):
        """Pick the metric to focus: the user's explicit pick wins; otherwise the
        model's worst flagged metric; otherwise the current fallback."""
        if user_picked:
            return current
        if status is not None and status.worst_metric and status.worst_metric in WATCHED_METRICS:
            return status.worst_metric
        return current

    # ---- header (built INTO the actions parent — no reparenting) ----
    def header_actions(self, parent):
        t = self.theme
        self._model_selector = ctk.CTkComboBox(parent, values=[], width=200,
                                                command=self._on_model_selected, fg_color=t.CARD,
                                                border_color=t.BORDER, button_color=t.SEGMENT_SELECTED,
                                                button_hover_color=t.SEGMENT_SELECTED_HOVER,
                                                text_color=t.TEXT_PRIMARY)
        self._model_selector.set("Select model…")
        # Typing a model number + Enter must load it — the combobox command
        # only fires on dropdown picks, and scrolling 279 entries to reach a
        # typed-out name is unusable (found live, 2026-07-08).
        self._model_selector.bind(
            "<Return>", lambda e: self._on_model_selected(self._model_selector.get().strip()))
        # Thumbwheel (work finding #2): the wheel steps prev/next model while
        # hovering the CLOSED selector.
        self._model_selector.bind("<MouseWheel>", self._on_selector_wheel)
        self._model_selector.bind("<Button-4>", lambda e: self._on_selector_wheel(e, step=-1))
        self._model_selector.bind("<Button-5>", lambda e: self._on_selector_wheel(e, step=1))
        # The OPEN dropdown is a tkinter.Menu underneath — it cannot wheel-
        # scroll on Windows and is a wall at 451 models (James, 2026-07-14:
        # "the mouse thumbwheel still doesnt work ... when i hit the model
        # dropdown"). Replace what the arrow OPENS: a searchable, wheel-
        # scrollable picker. Private-API override is safe under the pinned
        # customtkinter 5.2.2 (see requirements-pinned.txt).
        self._model_selector._open_dropdown_menu = self._open_model_picker
        self._model_selector.pack(side="left", padx=(0, t.SPACE_SM))
        self._window_menu = ctk.CTkOptionMenu(parent, values=list(_WINDOW_DAYS), width=80,
                                              command=self._on_window_change, fg_color=t.CARD,
                                              button_color=t.SEGMENT_SELECTED,
                                              button_hover_color=t.SEGMENT_SELECTED_HOVER,
                                              text_color=t.TEXT_PRIMARY)
        self._window_menu.set(self._window_choice)
        self._window_menu.pack(side="left", padx=(0, t.SPACE_SM))
        # Lot selector (app-shape spec §2): production runs newest first, from
        # the SAME clustering the FOCUS list ranks and the lot chart draws.
        # Sits with the model and window pickers because all three answer "what
        # am I looking at"; it is what turns the stats table from a history
        # into "is THIS run different".
        self._lot_menu = ctk.CTkOptionMenu(parent, values=[_ALL_HISTORY], width=210,
                                           command=self._on_lot_change, fg_color=t.CARD,
                                           button_color=t.SEGMENT_SELECTED,
                                           button_hover_color=t.SEGMENT_SELECTED_HOVER,
                                           text_color=t.TEXT_PRIMARY)
        self._lot_menu.set(_ALL_HISTORY)
        self._lot_menu.pack(side="left", padx=(0, t.SPACE_SM))
        ctk.CTkButton(parent, text="Copy summary", command=self._on_copy_summary, fg_color=t.CARD,
                      hover_color=t.ELEVATED, text_color=t.TEXT_PRIMARY, corner_radius=t.RADIUS_SM)\
            .pack(side="left", padx=(0, t.SPACE_SM))
        ctk.CTkButton(parent, text="Export model to Excel", command=self._on_export, fg_color=t.ACCENT,
                      hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE, corner_radius=t.RADIUS_SM)\
            .pack(side="left")

    def build_content(self, parent):
        t = self.theme
        self._empty_label = ctk.CTkLabel(
            parent, text="Pick a model above, or click one on the Overview.",
            font=t.font(t.SIZE_HEADING), text_color=t.TEXT_SECONDARY)
        # A plain frame: the header line, then the tab view filling the rest of the page. Each tab
        # scrolls on its own (layout C). The old page scrolled as a whole, with the tab view at its
        # foot -- which is how a tab got squeezed to nothing once the parts above it outgrew the
        # window (render_pages.py --audit, 6607's Smoothness tab at 1280x720).
        self._body = ctk.CTkFrame(parent, fg_color="transparent")
        self._build_header_line(self._body)
        self._tabs = ThemedTabView(self._body, theme=t)
        self._tabs.pack(side="top", fill="both", expand=True)
        # add() order is the tab order: Summary, Units, Final test, History.
        self._summary = self._scrolling_tab(TAB_SUMMARY)
        self._units_scroll = self._scrolling_tab(TAB_UNITS)
        self._ft_scroll = self._scrolling_tab(TAB_FINAL_TEST)
        self._history_tab = HistoryTab(self._tabs.add(TAB_HISTORY), theme=t)
        self._history_tab.pack(fill="both", expand=True)
        self._build_summary(self._summary)
        self._build_units(self._units_scroll)
        self._build_final_test(self._ft_scroll)
        self._show_empty()

    def _scrolling_tab(self, name: str) -> ctk.CTkScrollableFrame:
        frame = ctk.CTkScrollableFrame(self._tabs.add(name), fg_color="transparent")
        frame.pack(fill="both", expand=True)
        return frame

    def _build_header_line(self, parent) -> None:
        """The model, its status word, its 90-day pass % and the words after it (units, lasers).
        Set by _set_header; the words wrap in whatever room the three before them leave."""
        t = self.theme
        self._header_line = ctk.CTkFrame(parent, fg_color="transparent")
        self._header_line.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        self._model_title = ctk.CTkLabel(self._header_line, text="", anchor="w",
                                         font=t.font(t.SIZE_TITLE, "bold"),
                                         text_color=t.TEXT_PRIMARY)
        self._model_title.pack(side="left", padx=(0, t.SPACE_MD))
        self._status_word = ctk.CTkLabel(self._header_line, text="", anchor="w",
                                         font=t.font(t.SIZE_BODY, "bold"),
                                         text_color=t.TEXT_SECONDARY)
        self._status_word.pack(side="left", padx=(0, t.SPACE_MD))
        self._pass_pct = ctk.CTkLabel(self._header_line, text="", anchor="w",
                                      font=t.mono(t.SIZE_HEADING, "bold"),
                                      text_color=t.TEXT_PRIMARY)
        self._pass_pct.pack(side="left", padx=(0, t.SPACE_SM))
        self._header_detail = ctk.CTkLabel(self._header_line, text="", anchor="w",
                                           justify="left", font=t.font(t.SIZE_BODY),
                                           text_color=t.TEXT_SECONDARY)
        self._header_detail.pack(side="left", fill="x", expand=True)
        # Built once with the page, so this binds once (blocks.wrap_to_width's rule).
        self._header_line.bind("<Configure>", lambda _e: self._rewrap_header(), add="+")

    def _rewrap_header(self) -> None:
        """Wrap the header's words to the room beside the model, status word and pass % -- the
        units blocks.wrap_to_width explains: widths are real pixels, wraplength CustomTkinter's."""
        try:
            width = self._header_line.winfo_width()
            if width <= 1:
                return                                  # not laid out yet
            t, label = self.theme, self._header_detail
            unscale = label._reverse_widget_scaling
            beside = sum(unscale(w.winfo_reqwidth())
                         for w in (self._model_title, self._status_word, self._pass_pct))
            room = unscale(width) - beside - (2 * t.SPACE_MD + t.SPACE_SM)
            label.configure(wraplength=int(max(120, room)))
        except (tkinter.TclError, AttributeError):
            pass

    def _build_summary(self, s) -> None:
        t = self.theme
        # Banners at the top of Summary (design doc item 2): a failed loader, and a trim-vs-final-
        # test spec mismatch -- check tone, packed only when they have something to say, directly
        # above the headline (_set_load_banner / _set_spec_banner pack them before=_headline_box).
        self._load_banner = blocks.banner(s, t, "", wrap_to=s)
        self._spec_banner = blocks.banner(s, t, "", wrap_to=s)
        # (1) The verdict in ONE sentence -- the first clause of _compute_verdict's line -- and the
        # evidence clauses after it, quieter, beneath. `s` lives as long as the page, so each
        # wrap binds once.
        self._headline_box = ctk.CTkFrame(s, fg_color="transparent")
        self._headline_box.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        self._headline = ctk.CTkLabel(self._headline_box, text="—", anchor="w", justify="left",
                                      font=t.font(t.SIZE_HEADING, "bold"),
                                      text_color=t.TEXT_PRIMARY)
        self._headline.pack(side="top", fill="x")
        self._headline_detail = ctk.CTkLabel(self._headline_box, text="", anchor="w",
                                             justify="left", font=t.font(t.SIZE_BODY),
                                             text_color=t.TEXT_SECONDARY)
        self._headline_detail.pack(side="top", fill="x")
        blocks.wrap_to_width(self._headline, s)
        blocks.wrap_to_width(self._headline_detail, s)
        # (2) The run chart of the moving signal. The view toggle sits with the chart it controls,
        # styled like every other "this switches what you're looking at" control.
        chart_head = ctk.CTkFrame(s, fg_color="transparent")
        chart_head.pack(side="top", fill="x", pady=(0, t.SPACE_XS))
        self._chart_toggle = ctk.CTkSegmentedButton(
            chart_head, values=[_VIEW_LOTS, _VIEW_UNITS], width=200,
            command=self._on_chart_view_change,
            fg_color=t.CARD, selected_color=t.SEGMENT_SELECTED,
            selected_hover_color=t.SEGMENT_SELECTED_HOVER, unselected_color=t.CARD,
            unselected_hover_color=t.ELEVATED, text_color=t.TEXT_PRIMARY)
        self._chart_toggle.set(_VIEW_LOTS if self._chart_view == "lots" else _VIEW_UNITS)
        self._chart_toggle.pack(side="right")
        ctk.CTkLabel(chart_head,
                     text=("Lots = one point per production run, judged against this "
                           "model's own history. Units = every measurement."),
                     font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY,
                     anchor="w").pack(side="left")
        self._focus_chart = FocusChart(s, theme=t)
        self._focus_chart.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        # (3) "Also moving" and (4) "Worth changing": rebuilt on every apply. Each always keeps a
        # child, even when it has nothing to say -- a Tk frame whose last child goes keeps its old
        # height, so an emptied section would leave a blank gap (measured).
        self._also_section = ctk.CTkFrame(s, fg_color="transparent")
        self._also_section.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        ctk.CTkFrame(self._also_section, height=1, fg_color="transparent").pack(fill="x")
        self._worth_section = ctk.CTkFrame(s, fg_color="transparent")
        self._worth_section.pack(side="top", fill="x", pady=(0, t.SPACE_XS))
        # ...and folded under it, the rest of what the Findings tab showed: what was measured, what
        # changed, the station and laser comparisons -- never the rows already drawn above.
        self._findings_fold = ctk.CTkFrame(s, fg_color="transparent")
        self._findings_fold.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
        ctk.CTkFrame(self._findings_fold, height=1, fg_color="transparent").pack(fill="x")
        self._findings_toggle = blocks.link_button(self._findings_fold, t, f"{_FINDINGS_FOLD} ▸",
                                                   lambda: self._set_findings_open(
                                                       not self._findings_open))
        self._findings_tab = FindingsTab(self._findings_fold, theme=t,
                                         shown_elsewhere=_WORTH_CHANGING_GROUPS)
        # (5) "All 12 signals", folded: the drift table, with the σ key, what the drift check left
        # out, and "Requalify baseline…".
        self._signals_fold = ctk.CTkFrame(s, fg_color="transparent")
        self._signals_fold.pack(side="top", fill="x")
        ctk.CTkFrame(self._signals_fold, height=1, fg_color="transparent").pack(fill="x")
        self._signals_toggle = ctk.CTkButton(
            self._signals_fold, text=f"{_SIGNALS_FOLD} ▸", anchor="w",
            command=lambda: self._set_signals_open(not self._signals_open),
            font=t.font(t.SIZE_HEADING, "bold"), fg_color="transparent", hover_color=t.CARD,
            text_color=t.TEXT_PRIMARY, corner_radius=t.RADIUS_SM, height=32)
        self._signals_toggle.pack(side="top", fill="x")
        self._drift_tab = DriftMetricsTab(self._signals_fold, theme=t,
                                          on_requalify=self._on_requalify,
                                          on_metric_select=self._on_metric_select)

    def _build_units(self, s) -> None:
        t = self.theme
        self._stats_table = StatsTableZone(s, theme=t)
        self._stats_table.pack(side="top", fill="x", pady=(0, t.SPACE_LG))
        blocks.group_header(s, t, "Unit list", None).pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        self._units_tab = UnitsTab(s, theme=t,
                                   on_unit_click=self._on_unit_click, on_export=self._on_export,
                                   on_export_charts=self._on_export_charts,
                                   on_search=self._on_unit_search)
        self._units_tab.pack(side="top", fill="x", pady=(0, t.SPACE_LG))
        self._smoothness_heading = blocks.group_header(s, t, "Smoothness", None)
        self._smoothness_heading.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        self._smoothness_tab = SmoothnessTab(s, theme=t)
        self._smoothness_tab.pack(side="top", fill="x")

    def _build_final_test(self, s) -> None:
        t = self.theme
        blocks.group_header(s, t, "Final-test units", None).pack(side="top", fill="x",
                                                                 pady=(0, t.SPACE_SM))
        self._ft_units_tab = FtUnitsTab(s, theme=t, on_unit_click=self._on_ft_unit_click,
                                        on_export_charts=self._on_export_ft_charts)
        self._ft_units_tab.pack(side="top", fill="x", pady=(0, t.SPACE_LG))
        self._trimft_heading = blocks.group_header(s, t, "Trim vs final test", None)
        self._trimft_heading.pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        self._trimft_tab = TrimFtTab(s, theme=t)
        self._trimft_tab.pack(side="top", fill="x")
        # It predicts final test, so it lives with it.
        self._predictor = PredictorPanel(s, theme=t, db=self.app.db)
        self._predictor.pack(side="top", fill="x", pady=(t.SPACE_LG, 0))

    # ---- visibility ----
    def _show_empty(self):
        self._body.pack_forget()
        self._empty_label.pack(expand=True)

    def _show_body(self):
        self._empty_label.pack_forget()
        self._body.pack(fill="both", expand=True)

    # ---- lifecycle ----
    def on_show(self):
        model, focus = self.app.consume_model_route_full()
        if model:
            self._current_model = model
            self._user_picked_metric = False
        if focus and focus in WATCHED_METRICS:
            self._current_metric = focus
            self._user_picked_metric = True
        tab = self.app.consume_model_tab()
        # Refresh the selector's model list each show.
        threading.Thread(target=self._refresh_selector_values, daemon=True).start()
        if self._current_model:
            self._show_body()
            self._predictor.set_model(self._current_model)
            self._reload()
            if tab:
                self._select_tab(tab)
        else:
            self._show_empty()

    def _select_tab(self, name: str) -> None:
        """Land where a tab= route asks (route_destination): select its tab, open its fold, and
        scroll its part into view -- now, and again once the reload it came with has drawn, since
        that is when the part's position is known. A word the page does not know is ignored, never
        raised (CTkTabview.set() raises on a name it does not have)."""
        dest = route_destination(name)
        if dest is None:
            return
        tab, part = dest
        self._tabs.set(tab)
        if part == "findings":
            self._set_findings_open(True)
        elif part == "signals":
            self._set_signals_open(True)
        if part is not None:
            self._scroll_target = (part, self._reload_gen)
            self.after_idle(lambda: self._scroll_to(part))

    def _scroll_to(self, part: str) -> None:
        frame, widget = {"findings": (self._summary, self._worth_section),
                         "signals": (self._summary, self._signals_fold),
                         "smoothness": (self._units_scroll, self._smoothness_heading),
                         "trim vs final test": (self._ft_scroll, self._trimft_heading)}[part]
        _scroll_into_view(frame, widget)

    def _apply_pending_scroll(self, gen: int) -> None:
        """apply()'s last step: the route that came with THIS reload scrolls once its parts are
        drawn. A target from an older reload is dropped -- never scroll a page the reader has been
        using since."""
        target, self._scroll_target = self._scroll_target, None
        if target is not None and target[1] == gen:
            self.after_idle(lambda: self._scroll_to(target[0]))

    # ---- the two folds ----
    def _set_signals_open(self, open_: bool) -> None:
        """'All 12 signals': the drift table, unfolded in place under its toggle."""
        self._signals_open = bool(open_)
        self._signals_toggle.configure(text=f"{_SIGNALS_FOLD} {'▾' if self._signals_open else '▸'}")
        if self._signals_open:
            if self._drift_tab.winfo_manager() == "":
                self._drift_tab.pack(side="top", fill="x", pady=(self.theme.SPACE_XS, 0))
        else:
            self._drift_tab.pack_forget()

    def _set_findings_open(self, open_: bool) -> None:
        self._findings_open = bool(open_)
        self._render_findings_fold()

    def _set_findings_fold(self, findings_data, findings_error) -> None:
        """Show the fold under 'Worth changing' when it has something the section does not: the
        facts of a computed model, or the error of a failed load. Not for NOT COMPUTED -- the
        section already says so, and the fold would only say it again."""
        self._findings_fold_shown = bool(findings_error) or bool((findings_data or {}).get("facts"))
        self._render_findings_fold()

    def _render_findings_fold(self) -> None:
        self._findings_toggle.configure(
            text=f"{_FINDINGS_FOLD} {'▾' if self._findings_open else '▸'}")
        self._findings_toggle.pack_forget()
        self._findings_tab.pack_forget()
        if self._findings_fold_shown:
            self._findings_toggle.pack(side="top", anchor="w")
            if self._findings_open:
                self._findings_tab.pack(side="top", fill="x")

    def _refresh_selector_values(self):
        try:
            models = [m.model for m in list_known_models(self.app.db)]
        except Exception:
            models = []
        def apply():
            self._model_selector.configure(values=models)
            if self._current_model:
                self._model_selector.set(self._current_model)
        self.safe_after(apply)

    # ---- reload with generation token (I3) ----
    def reload_now(self):
        """Synchronous reload + apply (the test path; also the main-thread apply)."""
        self._reload(sync=True)

    def _reload(self, *, sync=False):
        if not self._current_model:
            return
        self._reload_gen += 1
        gen = self._reload_gen
        model, metric = self._current_model, self._current_metric
        def work():
            # Each loader is independently guarded: one failure must not blank
            # the whole page (the old single try/except silently zeroed every
            # tab when any loader threw). Failures are logged, not swallowed.
            # `failed` names every loader that raised, in human words a
            # process engineer would recognise — built here on the worker and
            # only READ in apply() below, never a Tk call itself, so the
            # load banner can say "this failed" instead of the empty default
            # reading as a silent, factual "there is nothing here".
            failed: list = []
            status, chosen = None, metric
            dates, values, baseline = [], [], (None, None)
            spc = None
            units, smoothness, recent = [], [], {}
            trim_ft, history = {}, {}
            findings_data = None
            findings_error = None          # "ExcType: message" when the findings load crashed
            # One anchored cutoff for every tab (None = All): anchored to the
            # model's latest data so stale-but-flagged models still show their
            # record instead of empty windows.
            cutoff = self._window_cutoff(model)
            requal = None
            try:
                status = get_model_drift_status(self.app.db, model)
                chosen = self._resolve_focus_metric(status, self._user_picked_metric, metric)
                requal = self.app.db.get_baseline_requalification(model)
            except Exception:
                logger.exception("Model %s: drift status failed", model)
                failed.append("drift status")
            left_out = None                     # None = the count failed (said on the tab)
            try:
                left_out = drift_exclusions(self.app.db, model)
            except Exception:
                logger.exception("Model %s: drift exclusion count failed", model)
                failed.append("drift exclusions")
            try:
                dates, values, baseline = self._load_focus_series(model, chosen)
            except Exception:
                logger.exception("Model %s: focus series failed", model)
                failed.append("focus chart")
            try:
                # The headline LOT chart. Built here, off the Tk thread, and
                # cached by the apply below so the Lots/Units toggle never
                # re-queries. It carries its OWN window (the last SERIES_WINDOW
                # lots) on purpose — a control chart needs enough history to
                # have limits, so the header's 30d/90d choice filters the unit
                # view and the tabs, not this. `compute_spc_series` picks the
                # fraction vs continuous builder from the metric itself.
                spc = compute_spc_series(self.app.db, model, chosen)
            except Exception:
                logger.exception("Model %s: SPC lot series failed", model)
                failed.append("lot chart")
            try:
                units = self._load_units(model)
            except Exception:
                logger.exception("Model %s: units failed", model)
                failed.append("unit list")
            try:
                smoothness = self._load_smoothness(model)
            except Exception:
                logger.exception("Model %s: smoothness failed", model)
                failed.append("smoothness")
            try:
                recent = self._recent_means(model)
            except Exception:
                logger.exception("Model %s: recent means failed", model)
                failed.append("recent means")
            try:
                trim_ft = self.app.db.get_model_trim_ft_agreement(model, cutoff_date=cutoff)
            except Exception:
                logger.exception("Model %s: trim-vs-FT failed", model)
                failed.append("trim vs final test")
            try:
                history = self.app.db.get_model_measurement_history(model, cutoff_date=cutoff)
            except Exception:
                logger.exception("Model %s: history failed", model)
                failed.append("measurement history")
            ft_units = []
            try:
                ft_units = self._load_ft_units(model, cutoff)
            except Exception:
                logger.exception("Model %s: final-test units failed", model)
                failed.append("final-test units")
            on_focus, focus_entry = None, None   # None: the "Drifting now" list failed to load
            try:
                # The list itself, not a re-derivation of its rule: the header's "Drifting" must
                # agree with the Overview's cards, which are built from it.
                focus_entry = next((e for e in compute_focus_list(self.app.db).focus
                                    if e.model == model), None)
                on_focus = focus_entry is not None
            except Exception:
                logger.exception("Model %s: drifting-now list failed", model)
                failed.append("drifting-now list")
            verdict = None
            try:
                verdict = self._compute_verdict(model, cutoff, status, recent, focus=focus_entry)
            except Exception:
                logger.exception("Model %s: verdict failed", model)
                failed.append("verdict")
            spec = None
            try:
                # Two small sampling queries, memoized per model — cheap, but
                # it is still I/O, so it belongs here on the worker with the
                # rest of the loaders, never in the apply below.
                spec = compare_station_specs(self.app.db, model)
            except Exception:
                logger.exception("Model %s: spec alignment failed", model)
                failed.append("station spec comparison")
            smooth_models = []
            try:
                from sqlalchemy import func as _f
                with self.app.db.session() as s_:
                    smooth_models = (s_.query(DBSR.model, _f.count(DBSR.id))
                                     .group_by(DBSR.model)
                                     .order_by(_f.count(DBSR.id).desc()).limit(12).all())
            except Exception:
                logger.exception("smoothness model list failed")
            # ---- the stats table + lot-vs-history (app-shape spec §2) -------
            # All three reads happen HERE, on the worker, with the rest of the
            # loaders. Each is one bulk query; on 6607's ~10,000 tracks the
            # table is ~50 ms and the verdicts ~200 ms.
            stats = lot_stats = None
            lots, verdicts, lot_label = [], {}, ""
            try:
                stats = compute_model_stats(self.app.db, model, cutoff=cutoff)
            except Exception:
                logger.exception("Model %s: stats table failed", model)
                failed.append("stats table")
            try:
                lots = model_lots(self.app.db, model)
                chosen_lot = self._resolve_lot(model, lots)
                if chosen_lot is not None:
                    lot_label = chosen_lot.label
                    lot_stats = compute_model_stats(self.app.db, model,
                                                    lot=chosen_lot.window)
                    verdicts = compute_lot_verdicts(self.app.db, model,
                                                    chosen_lot.window)
            except Exception:
                logger.exception("Model %s: lot stats failed", model)
                failed.append("lot stats")
            try:
                process_facts = self.app.db.get_process_facts(model)
                if process_facts:
                    findings_data = {"facts": process_facts,
                                     "findings": self.app.db.get_process_findings(model)}
            except Exception as exc:
                logger.exception("Model %s: process findings failed", model)
                failed.append("process findings")
                findings_error = f"{type(exc).__name__}: {exc}"
            inactive = {}                  # {model: newest trim file, or None} when it is inactive
            is_inactive = None             # None: could not be worked out (the header says nothing)
            try:
                # Worked out on every load, never read from cached findings (F5, James 2026-09-25):
                # a model turns inactive when OTHER models' newer files move the fleet forward.
                activity = load_activity(self.app.db)
                is_inactive = activity.is_inactive(model)
                if is_inactive:
                    inactive = {model: activity.last_trimmed(model)}   # None: no trims on record
            except Exception:
                logger.exception("Model %s: last-trimmed date failed", model)
                failed.append("last-trimmed date")
            # ---- the header line (layout C): the status word and the 90-day pass rate ----------
            # (on_focus, loaded with the verdict above.)
            facts = None
            try:
                facts = header_facts(self.app.db, model)
            except Exception:
                logger.exception("Model %s: 90-day pass rate failed", model)
                failed.append("90-day pass rate")
            flagged = (None if (status is None or "drift status" in failed)
                       else status.overall_tier > DriftTier.STABLE)
            word = status_word(detector_flagged=flagged, on_focus_list=on_focus,
                               inactive=is_inactive, last_trimmed=inactive.get(model))
            pct_text, detail_text = header_texts(facts)
            if findings_data is not None:
                # The final-test predictor's own AUC, set beside loss_origin's in the fold under
                # "Worth changing" (spec ruling 2). Its own guard: a failed read is named there,
                # and it must not cost the fold the findings themselves.
                try:
                    findings_data["predictor_auc"] = self.app.db.get_predictor_auc(model)
                except Exception as exc:
                    logger.exception("Model %s: predictor AUC failed", model)
                    findings_data["predictor_auc_error"] = f"{type(exc).__name__}: {exc}"

            def apply():
                if gen != self._reload_gen:
                    return  # a newer reload superseded this one
                self._current_metric = chosen

                def _try(what, fn):
                    # Per-widget guard: a render bug in one tab must not stop
                    # the tabs after it from updating.
                    try:
                        fn()
                    except Exception:
                        logger.exception("Model %s: %s render failed", model, what)

                if status:
                    _try("drift tab", lambda: self._drift_tab.set_status(status, recent_means=recent))
                else:
                    # The drift-status loader failed (or genuinely has nothing
                    # yet) — reset to the just-constructed look rather than
                    # skipping the update, or the PREVIOUS model's drift rows
                    # would keep showing under this model's name.
                    _try("drift tab", lambda: self._drift_tab.clear())
                _try("baseline info", lambda: self._drift_tab.set_baseline_info(requal))
                _try("left out", lambda: self._drift_tab.set_exclusions(left_out))
                _try("header", lambda: self._set_header(model, word, pct_text, detail_text))
                # Always set — never left showing a PREVIOUS model's verdict when this
                # model's verdict failed to compute (M1). The verdict is Summary's
                # headline now (layout C; it was the page caption, and before that the
                # _verdict label) -- and never a verdict built on a drift status that
                # FAILED to load: _compute_verdict does not raise on status=None, it
                # answers "NOT TRAINED -- run drift training in Settings" -- confident,
                # specific, and false when the real reason is a crashed query. The banner
                # below says what happened. (An inactive model's "Inactive · last trimmed"
                # is the header's status word, set above.)
                shown = verdict if (verdict and "drift status" not in failed) else None
                _try("headline", lambda: self._set_headline(shown))
                _try("also moving", lambda: self._set_also_moving(
                    status if "drift status" not in failed else None, chosen, recent))
                # Load banner first, spec banner second: both pack with
                # before=self._headline_box (a fixed anchor, never each other -- see
                # _set_load_banner / _set_spec_banner), and pack(before=X) always lands a
                # widget immediately next to X -- so calling load then spec puts spec
                # (packed second) closer to X, i.e. load (an error) leads and spec follows
                # when both have something to say on the same pass.
                _try("load banner", lambda: self._set_load_banner(failed))
                _try("spec banner", lambda: self._set_spec_banner(spec))
                _try("worth changing", lambda: self._set_findings_section(findings_data, failed,
                                                                          inactive=inactive))
                _try("findings fold", lambda: self._set_findings_fold(findings_data,
                                                                      findings_error))
                _try("lot selector", lambda: self._set_lot_choices(lots, lot_label))
                _try("stats table", lambda: self._stats_table.set_stats(
                    stats, lot_stats=lot_stats, verdicts=verdicts,
                    lot_label=lot_label))
                _try("focus chart", lambda: self._set_chart_data(
                    chosen, spc, dates, values, baseline))
                _try("units tab", lambda: self._units_tab.set_units(units))
                _try("smoothness tab", lambda: self._smoothness_tab.set_records(smoothness))
                _try("smoothness hint", lambda: self._smoothness_tab.set_models_hint(smooth_models))
                _try("trim-vs-FT tab", lambda: self._trimft_tab.set_data(trim_ft))
                _try("FT units tab", lambda: self._ft_units_tab.set_units(ft_units))
                _try("history tab", lambda: self._history_tab.set_data(history))
                # A failed load is the tab's own FAILED state, naming the error -- never its "not
                # computed yet" line under the banner that names the crash (facelift F4).
                _try("findings tab", lambda: self._findings_tab.set_data(
                    findings_data, failed=findings_error, inactive=inactive))
                # Last: a tab= route's part is in place only now that everything above is drawn.
                _try("route", lambda: self._apply_pending_scroll(gen))
            if sync:
                apply()                 # already on the Tk thread — post nothing
            else:
                self.safe_after(apply)
        if sync:
            work()
        else:
            threading.Thread(target=work, daemon=True).start()

    # ---- lot selection ----
    def _resolve_lot(self, model, lots):
        """Which lot the stats table should describe — worker-side, no Tk.

        The rule (app-shape spec §2): the CURRENT lot when one is open, and
        all history otherwise. The default is applied once per model, so it
        cannot fight the user — once he picks "all history" on a model, a
        refresh does not put him back on the open lot.
        """
        self._lots = lots
        if self._lot_default_applied_for != model:
            self._lot_default_applied_for = model
            index = default_lot_index(lots)
            self._lot_label = lots[index].label if index is not None else None
        if self._lot_label is None:
            return None
        # Re-resolve by label: new files can reshape the newest lot between
        # reloads, and a stale index would quietly describe a different run.
        for lot in lots:
            if lot.label == self._lot_label:
                return lot
        self._lot_label = None
        return None

    def _set_lot_choices(self, lots, lot_label) -> None:
        values = [_ALL_HISTORY] + [lot.label for lot in lots]
        self._lot_menu.configure(values=values)
        self._lot_menu.set(lot_label or _ALL_HISTORY)

    def _on_lot_change(self, choice):
        self._lot_label = None if choice == _ALL_HISTORY else choice
        # Applied by the user: never override it with the per-model default.
        self._lot_default_applied_for = self._current_model
        self._reload()

    def _set_spec_banner(self, comparison) -> None:
        """Show the check-tone banner only when the two stations really do differ.

        "aligned" and "insufficient" both say nothing: one is good news that
        needs no banner, the other is an unanswered question, and dressing an
        unanswered question as a warning is how a warning stops being believed.

        `before=self._headline_box`: a fixed, always-present sibling at the top
        of Summary, so pack() re-inserts this banner in the same place --
        directly above the verdict it qualifies -- every time it is shown
        again, instead of re-appending it at the foot of the tab. See the
        load-then-spec call order in apply() for how the two banners end up
        ordered load-first when both fire on the same pass.
        """
        if comparison is None or comparison.status != "differs":
            self._spec_banner.pack_forget()
            return
        # "at those positions", not a flat "compare different requirements":
        # the banner now fires from a tenth of the travel upward, and the note
        # it follows already says what share that is.
        self._spec_banner.configure(
            text=("⚠ " + comparison.note + " — cross-station numbers "
                  "(escapes, Gap) compare different requirements at those "
                  "positions."))
        self._spec_banner.pack(side="top", fill="x",
                               pady=(0, self.theme.SPACE_SM),
                               before=self._headline_box)

    def _set_load_banner(self, failed) -> None:
        """Name every loader that raised this pass, so a crash never reads as
        an empty-but-healthy page (the hazard this page exists to remove —
        see the module docstring / code review 2026-09-20). `failed` is a
        plain list built on the worker thread; this method only reads it.

        At the top of Summary (layout C), `before=self._headline_box` -- same
        anchor as `_set_spec_banner` and the same reason: pack() would
        otherwise re-append the label at the foot of the tab every time it is
        shown again.
        """
        if not failed:
            self._load_banner.pack_forget()
            return
        self._load_banner.configure(
            text=("⚠ Could not load: " + ", ".join(failed) + ". Those parts "
                  "of this page may be empty or out of date — this is an "
                  "error, not an absence of data. The log has the details."))
        self._load_banner.pack(side="top", fill="x",
                               pady=(0, self.theme.SPACE_SM),
                               before=self._headline_box)

    def _set_findings_section(self, findings_data, failed, *, inactive=None) -> None:
        """"Worth changing on this model" (design doc item 3; Summary's, since layout C): the
        model's findings, in the same FindingsView the Findings page draws, capped to 3
        rows across the three ACTIONABLE groups (_WORTH_CHANGING_GROUPS) -- "history"
        (recipe changes, already happened) and "other" stay off it, one click away in the
        fold under it, which draws ONLY them (FindingsTab's shown_elsewhere): one view of
        the findings, not two. `findings_data` is the SAME dict the fold gets (built once
        in _reload's work()); this is a second consumer of it, not a second load.

        Rebuilt whole on every apply(), same as FindingsView._render() and
        FindingsTab.set_data() do -- the header's count can only ever agree with the
        rows under it if both come from the same pass.

        A FAILED "process findings" load says NOTHING here, not even the empty-state
        line: the general load banner above already names it ("Could not load: process
        findings, ...", set by _set_load_banner just before this runs), and drawing
        "nothing worth changing" over a load that actually crashed is exactly the
        CLAUDE.md hazard this page exists to remove -- a failure must never look like
        a result.

        Otherwise three states, never two (final review, 2026-09-24 -- "nothing worth
        changing" used to be said for all three):
          * NOT COMPUTED -- no cached facts: the Findings tab's own "not computed yet"
            line, and no count pill (unknown is not zero);
          * FAILED -- an analyzer crashed (facts["errors"]): a check-tone banner naming
            each one, never the "nothing" line; rows the other analyzers found still show,
            with the banner saying the list may be short;
          * EMPTY -- computed, nothing in these three groups: "Nothing worth changing
            stands out", under a 0. Decided on the COUNT, not on `findings`: a model with
            only history findings (off this section) was a "0" over a blank body.

        Everything is built into `body`, a frame rebuilt with it on every apply, and the texts
        that wrap wrap to IT (facelift F4: the NOT COMPUTED line and the analyzer banner wrapped
        at a fixed 1000 units, cut at 960x640). Wrapping them to _worth_section itself would bind
        one more <Configure> handler to a frame that lives as long as the page on every refresh
        (blocks.wrap_to_width's rule).
        """
        t = self.theme
        for child in self._worth_section.winfo_children():
            child.destroy()
        self._worth_view = None
        if "process findings" in (failed or []):
            # Nothing to say -- but one child, so the section collapses instead of keeping the
            # height of what it last drew (a Tk frame whose last child goes keeps its size).
            ctk.CTkFrame(self._worth_section, height=1, fg_color="transparent").pack(fill="x")
            return
        body = ctk.CTkFrame(self._worth_section, fg_color="transparent")
        body.pack(fill="x")
        title = "Worth changing on this model"
        facts = (findings_data or {}).get("facts")
        if not facts:
            blocks.group_header(body, t, title, None).pack(fill="x", pady=(0, t.SPACE_XS))
            line = ctk.CTkLabel(body, text=NOT_COMPUTED_TEXT, font=t.font(t.SIZE_BODY),
                                text_color=t.TEXT_SECONDARY, anchor="w", justify="left")
            line.pack(fill="x", padx=t.SPACE_SM, pady=(0, t.SPACE_SM))
            blocks.wrap_to_width(line, body, padding=2 * t.SPACE_SM)
            return
        findings = (findings_data or {}).get("findings") or []
        count = _worth_changing_count(findings)
        errors = facts.get("errors") or {}
        blocks.group_header(body, t, title,
                            None if (errors and not count) else count,
                            tone="act").pack(fill="x", pady=(0, t.SPACE_XS))
        if errors:
            named = "; ".join(f"{analyzer_name(k)} ({errors[k]})" for k in sorted(errors))
            blocks.banner(body, t,
                          f"Could not be worked out on the last refresh: {named}. Anything "
                          f"{'it' if len(errors) == 1 else 'they'} would have found is missing "
                          f"here — this is an error, not a result. The log has the details.",
                          wrap_to=body).pack(fill="x", pady=(0, t.SPACE_SM))
        if not count:
            if not errors:
                ctk.CTkLabel(body,
                             text="Nothing worth changing stands out for this model.",
                             font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY,
                             anchor="w").pack(fill="x", padx=t.SPACE_SM, pady=(0, t.SPACE_SM))
            return
        view = FindingsView(body, t, on_open=None, include_empty=False,
                            rows_per_group=3, groups=_WORTH_CHANGING_GROUPS)
        view.pack(fill="x")
        view.set_findings(findings, inactive=inactive)        # {model: last trimmed} when inactive
        self._worth_view = view

    # ---- the header line, the headline, "Also moving" (layout C) ----
    def _set_header(self, model, word, pct_text, detail_text) -> None:
        """The header line, from the worker's results. A word the page could not work out is
        blank (status_word), never a guess; Drifting carries the worse/fail colour WITH its word."""
        t = self.theme
        tone = {"Drifting": t.CHECK, "Steady": t.TEXT_PRIMARY}.get(word, t.TEXT_SECONDARY)
        self._model_title.configure(text=model)
        self._status_word.configure(text=word or "", text_color=tone)
        self._pass_pct.configure(text=pct_text)
        self._header_detail.configure(text=detail_text)
        self.after_idle(self._rewrap_header)        # the words beside it may have changed width

    def _set_headline(self, shown) -> None:
        """Summary's first line: the verdict's first clause, in ONE sentence -- the page's own
        verdict wording (_compute_verdict) -- and its evidence clauses, quieter, beneath. "—"
        when there is no verdict to show (it failed, or rests on a load that did)."""
        t = self.theme
        text, color = shown if shown else ("—", t.TEXT_PRIMARY)
        head, _sep, rest = text.partition(_CLAUSE)
        self._headline.configure(text=head, text_color=color)
        self._headline_detail.configure(text=rest)

    def _set_also_moving(self, status, charted, recent) -> None:
        """'Also moving': one line per OTHER signal above stable (also_moving) -- label, baseline →
        last lot, ↑/↓ -- each a click that charts it above. Nothing at all when nothing else moves,
        or when the drift status did not load (the banner names that; "nothing moving" over a
        crashed load would be a failure looking like a result). Rebuilt on every apply()."""
        t = self.theme
        for child in self._also_section.winfo_children():
            child.destroy()
        self._also_lines = {}
        body = ctk.CTkFrame(self._also_section, fg_color="transparent", height=1)
        body.pack(fill="x")           # one child always: an emptied frame keeps its old height
        moving = also_moving(status, charted, recent, t.fmt_measure) if status is not None else []
        if not moving:
            return
        blocks.group_header(body, t, "Also moving", len(moving)).pack(fill="x",
                                                                      pady=(0, t.SPACE_XS))
        key = ctk.CTkLabel(body, text="Baseline → last lot. Click a signal to chart it above.",
                           font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY, anchor="w",
                           justify="left")
        key.pack(fill="x", padx=t.SPACE_SM, pady=(0, t.SPACE_XS))
        blocks.wrap_to_width(key, body, padding=2 * t.SPACE_SM)   # `body` is rebuilt with it
        for metric, text in moving:
            self._also_lines[metric] = self._also_line(body, metric, text)

    def _also_line(self, parent, metric: str, text: str) -> ctk.CTkFrame:
        t = self.theme
        line = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=t.RADIUS_SM)
        line.pack(fill="x")
        name = ctk.CTkLabel(line, text=metric_label(metric), font=t.font(t.SIZE_BODY),
                            text_color=t.TEXT_PRIMARY, anchor="w")
        name.pack(side="left", padx=(t.SPACE_SM, t.SPACE_MD), pady=t.SPACE_XS)
        moved = ctk.CTkLabel(line, text=text, font=t.mono(t.SIZE_BODY),
                             text_color=t.TEXT_SECONDARY, anchor="w")
        moved.pack(side="left", pady=t.SPACE_XS)

        def click(_event=None) -> None:
            self._on_metric_select(metric)

        def hover(on: bool) -> None:
            line.configure(fg_color=t.ELEVATED if on else "transparent")

        def leave(event) -> None:
            # <Leave> fires when the pointer moves onto a CHILD label too (blocks.row's rule).
            under = line.winfo_containing(event.x_root, event.y_root)
            own = str(line)
            if under is None or not (str(under) == own or str(under).startswith(own + ".")):
                hover(False)

        for w in (line, name, moved):
            w.bind("<Button-1>", click, add="+")
            w.bind("<Enter>", lambda _e: hover(True), add="+")
            w.bind("<Leave>", leave, add="+")
            try:
                w.configure(cursor="hand2")
            except Exception:             # some CTk widgets refuse a cursor; clicks still work
                pass
        line._on_click = click            # test hook: the real bound handler
        return line

    # ---- headline chart: two views over ONE load ----
    def _set_chart_data(self, metric, spc, dates, values, baseline):
        """Cache both views of the focus metric, then draw the selected one."""
        self._spc_series = spc
        self._unit_series = (metric, dates, values, baseline)
        self._render_focus_chart()

    def _render_focus_chart(self):
        """Draw whichever view is selected — from cached data, never the DB.

        Falls back to the unit view when the lot series is missing (its build
        failed): an empty card would say nothing about why.
        """
        if self._chart_view == "lots" and self._spc_series is not None:
            self._focus_chart.set_spc_series(self._spc_series)
            return
        metric, dates, values, baseline = self._unit_series
        # Exactly the window the control loaded (_WINDOW_DAYS), no framing on top: the view shows
        # what the user picked, "All" included, and past ~18 months the rolling median widens to
        # 90 days, as designed (FocusChart.set_series).
        self._focus_chart.set_series(metric=metric, dates=dates, values=values,
                                     baseline_mean=baseline[0], baseline_std=baseline[1])

    # ---- loaders (all materialize to plain values inside the session — I8) ----
    def _window_cutoff(self, model: Optional[str] = None,
                       metric: Optional[str] = None) -> Optional[datetime]:
        """Window cutoff anchored to the MODEL'S latest data, not wall-clock now.

        This is batch-loaded historical data: a model's newest unit can be
        months old (loaded lots lag production). Anchoring to now() made every
        window empty for such models — Triage would flag a model, the click-
        through landed on 'No measurements in the selected window'. Same
        reasoning as compute_recent_means (evidence.py).

        FT-axis metrics anchor to the FINAL-TEST table's newest date instead
        (code-review finding #6, 2026-07-13): FT ingest lags trim ingest, so a
        trim-anchored 90-day window could exclude every FT row — clicking a
        flagged FT pill landed on an empty chart. FT-only models (no trim
        rows) get a working anchor for the same reason.
        """
        days = _WINDOW_DAYS.get(self._window_choice)
        if days is None:
            return None
        anchor = None
        model = model or self._current_model
        if model:
            try:
                from sqlalchemy import func
                with self.app.db.session() as s:
                    if metric in ("ft_fail_fraction", "escape_fraction"):
                        from laser_trim_analyzer.database.models import (
                            FinalTestResult as DBFT)
                        anchor = (s.query(func.max(func.coalesce(
                                      DBFT.test_date, DBFT.file_date)))
                                  .filter(DBFT.model == model).scalar())
                    if anchor is None:
                        anchor = (s.query(func.max(DBAR.file_date))
                                  .filter(DBAR.model == model).scalar())
            except Exception:
                logger.exception("window anchor query failed for %s", model)
        if anchor is None:
            anchor = datetime.now()
        if isinstance(anchor, str):        # raw SQLite string from coalesce
            anchor = datetime.fromisoformat(anchor[:19])
        return anchor - timedelta(days=days)

    def _load_focus_series(self, model, metric):
        cutoff = self._window_cutoff(metric=metric)
        with self.app.db.session() as s:
            if metric == "max_smoothness_value":
                q = s.query(DBSR.file_date, DBSR.max_smoothness_value).filter(
                    DBSR.model == model, DBSR.max_smoothness_value.isnot(None))
                if cutoff:
                    q = q.filter(DBSR.file_date >= cutoff)
                rows = q.order_by(DBSR.file_date).all()
            elif metric == "linearity_fail_fraction":
                from sqlalchemy import func as _fn, case as _case
                q = (s.query(DBAR.file_date,
                             _fn.avg(_case((DBAR.overall_status == StatusType.FAIL, 1.0),
                                           else_=0.0)))
                     .filter(DBAR.model == model,
                             DBAR.overall_status.in_([StatusType.PASS, StatusType.WARNING,
                                                      StatusType.FAIL])))
                if cutoff:
                    q = q.filter(DBAR.file_date >= cutoff)
                rows = q.group_by(_fn.date(DBAR.file_date)).order_by(DBAR.file_date).all()
            elif metric == "ft_fail_fraction":
                # Daily FINAL-TEST fail rate — same shape as the detector's
                # lot observations, on the FT date axis. COALESCE: some FT
                # files never parse a test_date cell; file_date covers them
                # (code-review finding #4).
                from sqlalchemy import func as _fn, case as _case
                from laser_trim_analyzer.database.models import FinalTestResult as DBFT
                ft_date = _fn.coalesce(DBFT.test_date, DBFT.file_date)
                q = (s.query(ft_date,
                             _fn.avg(_case((DBFT.overall_status == StatusType.FAIL, 1.0),
                                           else_=0.0)))
                     .filter(DBFT.model == model,
                             ft_date.isnot(None),
                             ft_date > datetime(2000, 1, 1),
                             DBFT.overall_status.in_([StatusType.PASS, StatusType.WARNING,
                                                      StatusType.FAIL])))
                if cutoff:
                    q = q.filter(ft_date >= cutoff)
                rows = q.group_by(_fn.date(ft_date)).order_by(ft_date).all()
            elif metric == "escape_fraction":
                # Daily escape rate: of confidently-linked FT records whose
                # trim was ACCEPTED, the share that failed final test.
                from sqlalchemy import func as _fn, case as _case
                from laser_trim_analyzer.database.models import FinalTestResult as DBFT
                from laser_trim_analyzer.ml.drift_training import ESCAPE_MIN_CONFIDENCE
                ft_date = _fn.coalesce(DBFT.test_date, DBFT.file_date)
                q = (s.query(ft_date,
                             _fn.avg(_case((DBFT.overall_status == StatusType.FAIL, 1.0),
                                           else_=0.0)))
                     .join(DBAR, DBFT.linked_trim_id == DBAR.id)
                     .filter(DBFT.model == model,
                             ft_date.isnot(None),
                             ft_date > datetime(2000, 1, 1),
                             DBFT.match_confidence >= ESCAPE_MIN_CONFIDENCE,
                             DBFT.overall_status.in_([StatusType.PASS, StatusType.FAIL]),
                             DBAR.overall_status.in_([StatusType.PASS, StatusType.WARNING])))
                if cutoff:
                    q = q.filter(ft_date >= cutoff)
                rows = q.group_by(_fn.date(ft_date)).order_by(ft_date).all()
            elif metric in TRACK_METRIC_COLUMNS:
                col = TRACK_METRIC_COLUMNS[metric]      # Q4: SAME column the detector trained on
                q = (s.query(DBAR.file_date, col).join(DBTR, DBTR.analysis_id == DBAR.id)
                     .filter(DBAR.model == model, col.isnot(None)))
                if cutoff:
                    q = q.filter(DBAR.file_date >= cutoff)
                rows = q.order_by(DBAR.file_date).all()
            else:
                rows = []
            # Q5; _coerce_dt: SQLite hands COALESCE dates back as strings.
            from laser_trim_analyzer.ml.drift_training import _coerce_dt
            pairs = [(_coerce_dt(r[0]), r[1]) for r in rows
                     if r[0] is not None and r[1] is not None]
            ms = s.query(ModelMetricState).filter_by(model=model, metric=metric).first()
            baseline = (ms.baseline_mean, ms.baseline_std) if ms else (None, None)
        return [p[0] for p in pairs], [p[1] for p in pairs], baseline

    def _compute_verdict(self, model, cutoff, status, recent_means, focus=None) -> tuple:
        """One-line answer to the daily question: (text, color).

        Drift state from the trained detectors; direction = window linearity
        yield vs the model's lifetime; difficulty = lifetime yield in plain
        words. Everything shown is verifiable on the tabs below it.

        `focus`: the model's row on the "Drifting now" list (ml/spc.compute_focus_list),
        or None. A model on it reads "Drifting" in the header (layout C) whatever the
        detectors say, so when none of them is above stable -- or none is trained --
        the state says what the LIST saw, in its own words: never "Holding" under a
        "Drifting" header.
        """
        from sqlalchemy import case as _case, func as _f
        t = self.theme

        def _yield(cut):
            with self.app.db.session() as s_:
                q = (s_.query(
                        _f.sum(_case((DBAR.overall_status.in_(
                            [StatusType.PASS, StatusType.WARNING]), 1), else_=0)),
                        _f.sum(_case((DBAR.overall_status.in_(
                            [StatusType.PASS, StatusType.WARNING, StatusType.FAIL]), 1), else_=0)))
                     .filter(DBAR.model == model))
                if cut is not None:
                    q = q.filter(DBAR.file_date >= cut)
                acc, grad = q.first()
            return (acc / grad * 100.0) if grad else None, (grad or 0)

        win_y, win_n = _yield(cutoff)
        life_y, life_n = _yield(None)

        # Drift state: worst non-stable watched metric by honest shift.
        worst = None
        trained = 0
        for m, ms in (getattr(status, "per_metric", {}) or {}).items():
            trained += 1
            tier = getattr(ms.tier, "name", str(ms.tier))
            if tier in ("STABLE",):
                continue
            rv = recent_means.get(m) if recent_means.get(m) is not None else ms.recent_mean
            shift = ((rv - ms.baseline_mean) / ms.baseline_std
                     if (rv is not None and ms.baseline_std) else None)
            key = abs(shift) if shift is not None else 0.0
            if worst is None or key > worst[2]:
                worst = (m, shift, key, tier)

        if trained == 0:
            state_txt, color = "Not trained — run drift training in Settings", t.TEXT_DISABLED
        elif worst is None:
            state_txt, color = "Holding — all watched metrics stable", t.TEXT_PRIMARY
        else:
            from laser_trim_analyzer.ml.drift_types import metric_label as _ml
            shift_txt = f"{worst[1]:+.1f}σ" if worst[1] is not None else "flagged"
            state_txt = f"Drifting — {_ml(worst[0])}: last lot {shift_txt} vs baseline lots"
            color = t.TIER_OOC if worst[3] == "OUT_OF_CONTROL" else t.TIER_DRIFT
        if focus is not None and worst is None:
            state_txt = (f"Drifting — lot fail rate {focus.p_base * 100:.0f}% → "
                         f"{focus.p_recent * 100:.0f}%, {focus.verdict}")
            color = t.TIER_OOC

        parts = [state_txt]
        try:
            from laser_trim_analyzer.core.yield_stats import compute_unit_yield
            u = compute_unit_yield(self.app.db, cutoff, model=model)
            if u.get("gradeable_units"):
                parts.append(
                    f"first-pass {u['first_pass_yield']:.0f}% → final "
                    f"{u['final_yield']:.0f}% ({u['attempts_per_section']:.2f} trims/section)")
        except Exception:
            logger.exception("verdict unit yield failed")
        if win_y is not None and life_y is not None and cutoff is not None:
            d = win_y - life_y
            trend = "better than" if d > 2 else ("worse than" if d < -2 else "in line with")
            parts.append(f"window yield {win_y:.0f}% ({win_n} units) {trend} "
                         f"lifetime {life_y:.0f}%")
        if life_y is not None:
            difficulty = ("historically difficult" if life_y < 75
                          else "historically mixed" if life_y < 90
                          else "historically strong")
            parts.append(f"{difficulty} ({life_y:.0f}% lifetime linearity yield, "
                         f"{life_n:,} unit{'s' if life_n != 1 else ''})")
        # Trim necessity (James's question, 2026-07-14): were these units
        # already meeting linearity BEFORE the laser? A high share means the
        # trim only served the resistance target — a candidate for raising
        # the as-fired resistance so the trim (and its laser time) goes away.
        try:
            from laser_trim_analyzer.core.yield_stats import compute_trim_necessity
            tn = compute_trim_necessity(self.app.db, model, cutoff)
            if tn and tn["trimmed_units"] >= 20:
                share = tn["prepass_share"]
                if share >= 20:
                    msg = (f"⚠ {share:.0f}% of trimmed units ({tn['prepass_units']} of "
                           f"{tn['trimmed_units']}) already met linearity BEFORE trim — "
                           "avoidable laser time")
                    rec = tn.get("recommendation")
                    if rec and rec.get("recommended_target"):
                        msg += (f". Raise {model}'s as-fired resistance target to "
                                f"~{rec['recommended_target']:,.0f} Ω (spec centre; now "
                                f"~{rec['asfired_median']:,.0f} Ω) and {rec['res_driven_units']} "
                                "of these trims disappear")
                    else:
                        msg += "; candidate for raising the as-fired resistance target"
                    parts.append(msg)
                elif share >= 5:
                    parts.append(f"{share:.0f}% already met linearity before trim")
        except Exception:
            logger.exception("trim necessity failed for %s", model)
        return "  ·  ".join(parts), color

    def _load_ft_units(self, model, cutoff) -> list:
        """Final-test records for the model (work finding #3, 2026-07-10:
        'no way to view final test units'). Newest first, capped at 500."""
        from laser_trim_analyzer.database.models import FinalTestResult as DBFT
        with self.app.db.session() as s:
            q = s.query(DBFT.serial, DBFT.file_date, DBFT.overall_status,
                        DBFT.linked_trim_id, DBFT.match_confidence, DBFT.id)\
                 .filter(DBFT.model == model)
            if cutoff is not None:
                q = q.filter(DBFT.file_date >= cutoff)
            rows = q.order_by(DBFT.file_date.desc()).limit(500).all()
        # .name ("FAIL"), not .value ("Fail") — the tab's color map and
        # fail-count compare against upper-case names; the title-case value
        # silently missed both (found by the FT-modal test, 2026-07-14).
        return [{"serial": r[0], "file_date": r[1],
                 "result": getattr(r[2], "name", str(r[2])),
                 "linked": r[3] is not None,
                 "match": (round(r[4] * 100) if r[4] is not None else None),
                 "id": r[5]}
                for r in rows]

    def _recent_means(self, model) -> dict:
        """Mean of each watched metric over the model's most recent window of DATA.

        Delegates to the shared evidence helper so the UI table, copy-summary, and the
        Excel evidence pack all derive 'recent' the same way (anchored to the model's
        latest file_date, since this is batch-loaded data that can be weeks old).
        Metric -> float|None.
        """
        from laser_trim_analyzer.export.evidence import compute_recent_means
        return compute_recent_means(self.app.db, model, recent_days=_RECENT_DAYS)

    def _load_units(self, model) -> List[dict]:
        from sqlalchemy import func as _f
        # Why this ERROR is an ERROR: the column set 2026-09-23, or (for rows
        # written before it existed) the same track's own words -- so all 237
        # rows on the rebuild explain themselves without a back-fill.
        reason = _f.coalesce(DBAR.error_reason, DBTR.linearity_spec_warning, DBTR.anomaly_reason)
        cutoff = self._window_cutoff()
        with self.app.db.session() as s:
            q = (s.query(DBAR.id, DBAR.serial, DBAR.file_date, DBAR.overall_status,
                         DBTR.sigma_gradient, DBTR.final_linearity_error_shifted, reason,
                         DBTR.status)
                 .join(DBTR, DBTR.analysis_id == DBAR.id).filter(DBAR.model == model))
            if cutoff:
                q = q.filter(DBAR.file_date >= cutoff)
            rows = q.order_by(DBAR.file_date.desc()).limit(200).all()
            return [_unit_row(r) for r in rows]

    def _search_units(self, model, query: str) -> List[dict]:
        """Serial lookup for the model — ignores the window and the recent cap so an old
        unit can still be found. Case-insensitive substring match on serial."""
        from sqlalchemy import func as _f
        reason = _f.coalesce(DBAR.error_reason, DBTR.linearity_spec_warning, DBTR.anomaly_reason)
        like = f"%{query}%"
        with self.app.db.session() as s:
            rows = (s.query(DBAR.id, DBAR.serial, DBAR.file_date, DBAR.overall_status,
                            DBTR.sigma_gradient, DBTR.final_linearity_error_shifted, reason,
                            DBTR.status)
                    .join(DBTR, DBTR.analysis_id == DBAR.id)
                    .filter(DBAR.model == model, DBAR.serial.ilike(like))
                    .order_by(DBAR.file_date.desc()).limit(500).all())
            return [_unit_row(r) for r in rows]

    def _on_unit_search(self, query: str) -> None:
        model = self._current_model
        if not model:
            return
        query = (query or "").strip()
        if not query:
            # Cleared → restore the recent list for the current window.
            def restore():
                units = self._load_units(model)
                self.safe_after(lambda: self._units_tab.set_units(units))
            threading.Thread(target=restore, daemon=True).start()
            return

        def work():
            try:
                results = self._search_units(model, query)
            except Exception:
                results = []
            cap = (f"{len(results)} match(es) for '{query}'"
                   + (" (showing first 500)" if len(results) == 500 else "")
                   + " — all dates"
                   if results else f"No units matching '{query}' for {model}")
            self.safe_after(lambda: self._units_tab.set_units(results, caption=cap))
        threading.Thread(target=work, daemon=True).start()

    def _load_smoothness(self, model) -> List[dict]:
        cutoff = self._window_cutoff()
        with self.app.db.session() as s:
            q = s.query(DBSR).filter(DBSR.model == model)
            if cutoff:
                q = q.filter(DBSR.file_date >= cutoff)
            rows = q.order_by(DBSR.file_date.desc()).limit(200).all()
            return [{"serial": r.serial, "file_date": r.file_date,
                     "max_smoothness_value": r.max_smoothness_value,
                     "smoothness_spec": r.smoothness_spec,
                     "smoothness_pass": r.smoothness_pass,
                     "overall_status": getattr(r.overall_status, "value", str(r.overall_status))}
                    for r in rows]

    # ---- events ----
    def _on_model_selected(self, model):
        if model and model != "Select model…":
            self._current_model = model
            self._user_picked_metric = False   # new model → auto-focus its worst metric
            self._show_body()
            self._predictor.set_model(model)
            self._reload()

    def _on_selector_wheel(self, event, step=None):
        values = list(self._model_selector.cget("values") or [])
        if not values:
            return
        if step is None:  # Windows/mac <MouseWheel>: delta sign gives direction
            step = -1 if getattr(event, "delta", 0) > 0 else 1
        cur = self._model_selector.get()
        try:
            idx = values.index(cur)
        except ValueError:
            idx = -1 if step > 0 else 0
        new_val = values[max(0, min(len(values) - 1, idx + step))]
        if new_val != cur:
            self._model_selector.set(new_val)
            self._on_model_selected(new_val)

    def _open_model_picker(self):
        """Searchable, WHEEL-SCROLLABLE replacement for the combobox dropdown
        (James, 2026-07-14). The native dropdown is a tkinter.Menu — no wheel
        support on Windows and unusable at 451 models. This popup: type to
        filter, wheel or drag to scroll, click (or Enter) to load."""
        # One at a time — clicking the arrow again closes the open picker.
        existing = getattr(self, "_picker", None)
        if existing is not None and existing.winfo_exists():
            existing.destroy()
            self._picker = None
            return
        t = self.theme
        values = list(self._model_selector.cget("values") or [])
        pop = ctk.CTkToplevel(self)
        self._picker = pop
        pop.overrideredirect(True)          # borderless, menu-like
        pop.configure(fg_color=t.CARD)
        x = self._model_selector.winfo_rootx()
        y = self._model_selector.winfo_rooty() + self._model_selector.winfo_height() + 2
        pop.geometry(f"300x420+{x}+{y}")
        pop.attributes("-topmost", True)
        search = ctk.CTkEntry(pop, placeholder_text="Type to filter…",
                              font=t.font(t.SIZE_BODY), fg_color=t.SURFACE,
                              border_color=t.BORDER, text_color=t.TEXT_PRIMARY)
        search.pack(side="top", fill="x", padx=t.SPACE_XS, pady=t.SPACE_XS)
        lst = ctk.CTkScrollableFrame(pop, fg_color="transparent")
        lst.pack(side="top", fill="both", expand=True, padx=t.SPACE_XS,
                 pady=(0, t.SPACE_XS))
        rows: list = []
        state = {"matches": values}

        def pick(name):
            try:
                pop.destroy()
            except Exception:
                pass
            self._picker = None
            self._model_selector.set(name)
            self._on_model_selected(name)

        def render():
            for r in rows:
                try:
                    r.destroy()
                except Exception:
                    pass
            rows.clear()
            flt = search.get().strip().lower()
            matches = [m for m in values if not flt or flt in m.lower()]
            state["matches"] = matches
            shown = matches[:200]
            for name in shown:
                is_cur = (name == self._current_model)
                lbl = ctk.CTkLabel(lst, text=name, anchor="w",
                                   font=t.font(t.SIZE_BODY, "bold" if is_cur else None),
                                   text_color=t.ACCENT if is_cur else t.TEXT_PRIMARY)
                lbl.pack(side="top", fill="x", padx=t.SPACE_XS)
                lbl.bind("<Button-1>", lambda e, n=name: pick(n))
                rows.append(lbl)
            if len(matches) > len(shown):
                cap = ctk.CTkLabel(lst, text=f"…{len(matches) - len(shown)} more — keep typing",
                                   font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY,
                                   anchor="w")
                cap.pack(side="top", fill="x", padx=t.SPACE_XS)
                rows.append(cap)
            if not matches:
                empty = ctk.CTkLabel(lst, text="No models match.",
                                     font=t.font(t.SIZE_CAPTION),
                                     text_color=t.TEXT_SECONDARY, anchor="w")
                empty.pack(side="top", fill="x", padx=t.SPACE_XS)
                rows.append(empty)

        search.bind("<KeyRelease>", lambda e: render())
        # Enter = load the first match; Escape or the arrow button closes.
        # (No FocusOut auto-close: on a borderless Toplevel it can fire when
        # the search entry itself takes focus — instant self-close.)
        search.bind("<Return>", lambda e: state["matches"] and pick(state["matches"][0]))
        search.bind("<Escape>", lambda e: (pop.destroy(), setattr(self, "_picker", None)))
        pop.bind("<Escape>", lambda e: (pop.destroy(), setattr(self, "_picker", None)))
        render()
        pop.after(50, search.focus_set)

    def _on_metric_select(self, metric):
        """A signal clicked -- an "Also moving" line, or a row of "All 12 signals": chart it. Back
        to the top of Summary, where the chart is."""
        self._user_picked_metric = True        # explicit choice; don't auto-override it
        self._current_metric = metric
        try:
            self._summary._parent_canvas.yview_moveto(0.0)    # CTk 5.2.2's own canvas (pinned)
        except (tkinter.TclError, AttributeError):
            pass
        self._reload()

    def _on_window_change(self, choice):
        self._window_choice = choice
        self._reload()

    def _on_chart_view_change(self, value):
        # Pure VIEW switch: both series came from the same _reload pass, so
        # re-render what is already in memory. Re-querying here would make the
        # toggle stutter and — worse — could redraw a DIFFERENT dataset than the
        # one the rest of the page is describing.
        self._chart_view = "units" if value == _VIEW_UNITS else "lots"
        self._render_focus_chart()

    def _on_unit_click(self, unit):
        UnitChartModal(self, theme=self.theme, db=self.app.db, unit=unit)

    def _on_ft_unit_click(self, ft_unit):
        """James 2026-07-14: clicking a final-test unit now shows its sweep."""
        from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import FtUnitChartModal
        FtUnitChartModal(self, theme=self.theme, db=self.app.db, ft_unit=ft_unit)

    def _on_copy_summary(self):
        if not self._current_model:
            return
        model = self._current_model
        from laser_trim_analyzer.export.evidence import build_summary_text

        # The drift status + recent-means queries are heavy (~10 aggregate
        # queries). Running them here blocked the UI behind the DB lock —
        # do them on a worker; only the clipboard write returns to Tk.
        def work():
            try:
                from laser_trim_analyzer.export.evidence import compute_recent_means
                status = get_model_drift_status(self.app.db, model)
                means, meta = compute_recent_means(self.app.db, model,
                                                   recent_days=_RECENT_DAYS, with_meta=True)
                text = build_summary_text(model, status, recent_means=means,
                                          recent_meta=meta)
            except Exception:
                logger.exception("Copy summary failed for %s", model)
                return
            def to_clipboard():
                self.clipboard_clear()
                self.clipboard_append(text)
            self.safe_after(to_clipboard)
        threading.Thread(target=work, daemon=True).start()

    def _on_requalify(self):
        """Per-model baseline requalification (design change). Confirm with
        effective date + reason, record the audit row, retrain THIS model
        from the effective date forward, reload."""
        model = self._current_model
        if not model:
            return
        from datetime import date as _date
        import tkinter as tk

        dlg = ctk.CTkToplevel(self)
        dlg.title(f"Requalify baseline — {model}")
        dlg.geometry("520x260")
        dlg.transient(self.winfo_toplevel())
        t = self.theme
        ctk.CTkLabel(dlg, text=(f"Reset {model}'s drift baselines because the design/"
                                "process changed. Data BEFORE the effective date is "
                                "excluded from the new baselines. If too little data "
                                "exists after the date, metrics read \"Not trained\" until "
                                "enough new lots accumulate. This action is recorded."),
                     font=t.font(t.SIZE_BODY), wraplength=480, justify="left",
                     text_color=t.TEXT_PRIMARY).pack(padx=16, pady=(16, 8), anchor="w")
        row1 = ctk.CTkFrame(dlg, fg_color="transparent"); row1.pack(fill="x", padx=16)
        ctk.CTkLabel(row1, text="Effective date (YYYY-MM-DD):", font=t.font(t.SIZE_BODY),
                     text_color=t.TEXT_SECONDARY).pack(side="left")
        date_e = ctk.CTkEntry(row1, width=130); date_e.pack(side="left", padx=8)
        date_e.insert(0, _date.today().isoformat())
        row2 = ctk.CTkFrame(dlg, fg_color="transparent"); row2.pack(fill="x", padx=16, pady=8)
        ctk.CTkLabel(row2, text="Reason (audit note):", font=t.font(t.SIZE_BODY),
                     text_color=t.TEXT_SECONDARY).pack(side="left")
        note_e = ctk.CTkEntry(row2, width=280); note_e.pack(side="left", padx=8)
        status_lbl = ctk.CTkLabel(dlg, text="", font=t.font(t.SIZE_CAPTION),
                                  text_color=t.TIER_WARNING)
        status_lbl.pack(padx=16, anchor="w")

        def go():
            from datetime import datetime as _dt
            raw = date_e.get().strip()
            try:
                eff = _dt.fromisoformat(raw)
            except ValueError:
                status_lbl.configure(text="Date must be YYYY-MM-DD.")
                return
            note = note_e.get().strip()
            status_lbl.configure(text="Requalifying + retraining this model…")
            def work():
                try:
                    self.app.db.set_baseline_requalification(model, eff.date().isoformat(), note)
                    from laser_trim_analyzer.ml.drift_training import train_drift_detector
                    preset = getattr(self.app.config.ml, "drift_sensitivity", "standard")
                    train_drift_detector(self.app.db, sensitivity_preset=preset, model=model)
                except Exception:
                    logger.exception("requalification failed for %s", model)
                    self.safe_after(lambda: status_lbl.winfo_exists() and status_lbl.configure(
                        text="Failed — see the log."))
                    return
                def done():
                    try:
                        dlg.destroy()
                    except Exception:
                        pass
                    self._reload()
                self.safe_after(done)
            threading.Thread(target=work, daemon=True).start()

        btns = ctk.CTkFrame(dlg, fg_color="transparent"); btns.pack(fill="x", padx=16, pady=12)
        ctk.CTkButton(btns, text="Requalify + retrain", fg_color=t.ACCENT,
                      hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE,
                      command=go, corner_radius=t.RADIUS_SM).pack(side="right")
        ctk.CTkButton(btns, text="Cancel", fg_color=t.CARD, hover_color=t.ELEVATED,
                      text_color=t.TEXT_PRIMARY, border_width=1, border_color=t.BORDER,
                      command=dlg.destroy, corner_radius=t.RADIUS_SM).pack(side="right", padx=8)

    def _on_export_charts(self):
        """Trim tab: export the checked unit charts (or all shown if none are
        checked) as one multi-page print-ready PDF (work finding #4, + James
        2026-07-14: pick a subset, and PDF not image)."""
        self._export_charts_pdf(kind="trim")

    def _on_export_ft_charts(self):
        """Final Test tab: same subset → multi-page PDF export, FT layout."""
        self._export_charts_pdf(kind="ft")

    def _export_charts_pdf(self, kind: str):
        if not self._current_model:
            return
        from tkinter import filedialog
        is_ft = kind == "ft"
        tab = self._ft_units_tab if is_ft else self._units_tab
        units = tab.get_selected_units()
        if not units:
            tab.set_caption("No units to export.")
            return
        model = self._current_model
        label = "final_test" if is_ft else "unit"
        path = filedialog.asksaveasfilename(
            title="Save charts (PDF)", defaultextension=".pdf",
            initialfile=f"{model}_{label}_charts_{len(units)}.pdf",
            filetypes=[("PDF", "*.pdf")])
        if not path:
            return

        def work():
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            from laser_trim_analyzer.gui.v6.widgets.unit_chart_modal import (
                load_unit_track, load_ft_track, compute_fail_points)
            from laser_trim_analyzer.export.unit_chart import build_unit_export_figure
            done = err = 0
            try:
                with PdfPages(path) as pdf:
                    for i, u in enumerate(units):
                        try:
                            data = (load_ft_track(self.app.db, u.get("id")) if is_ft
                                    else load_unit_track(self.app.db, u.get("analysis_id")))
                            if not data:
                                err += 1
                                continue
                            fp = compute_fail_points(
                                data.get("error_data"), data.get("upper_limits"),
                                data.get("lower_limits"),
                                offset=data.get("optimal_offset") or 0.0)
                            date_s = data.get("date") if is_ft else None
                            if not date_s:
                                fd = u.get("file_date")
                                date_s = (fd.strftime("%Y-%m-%d")
                                          if hasattr(fd, "strftime") else "nodate")
                            meta = {"model": data.get("model") or model,
                                    "serial": data.get("serial") or u.get("serial"),
                                    "system": data.get("system", ""),
                                    "trim_date": date_s,
                                    "track_id": data.get("track_id"),
                                    "n_tracks": data.get("n_tracks", 1)}
                            fig = build_unit_export_figure(
                                meta, data, fp, kind="ft" if is_ft else "trim")
                            pdf.savefig(fig, facecolor="white", bbox_inches="tight")
                            plt.close(fig)
                            done += 1
                        except Exception:
                            logger.exception("chart export failed for %s", u.get("serial"))
                            err += 1
                        if i % 5 == 0:
                            self.safe_after(lambda d=done, n=len(units):
                                            tab.set_caption(f"Exporting… {d}/{n}"))
            except Exception:
                logger.exception("PDF chart export failed")
                self.safe_after(lambda: tab.set_caption("Export failed — see the log."))
                return
            self.safe_after(lambda: tab.set_caption(
                f"Exported {done} chart(s) → {path}"
                + (f" · {err} failed (see log)" if err else "")))
        threading.Thread(target=work, daemon=True).start()

    def _on_export(self):
        if not self._current_model:
            return
        from tkinter import filedialog
        from laser_trim_analyzer.export.evidence import export_evidence_pack
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            initialfile=f"evidence_{self._current_model}.xlsx",
                                            filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        # Always export the FULL record: the Excel pack is the model's history
        # of record for judging process direction (the on-screen window only
        # controls the view). James' workflow: analyze units on screen, export
        # the model to Excel for full history, unit charts for the team.
        #
        # The LOT is the exception, and goes with him: the stats sheet compares
        # the run he currently has selected, so the sheet he hands an engineer
        # answers the question he was just looking at rather than re-deriving
        # its own default.
        lot = next((l_.window for l_ in self._lots
                    if l_.label == self._lot_label), None)
        threading.Thread(
            target=lambda: export_evidence_pack(self.app.db, self._current_model,
                                                path, lot=lot),
            daemon=True).start()
